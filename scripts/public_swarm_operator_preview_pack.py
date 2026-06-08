#!/usr/bin/env python3
"""Build the Public Swarm v0.1 Operator Preview artifact.

This is the broad user-facing preview bundle over the current
Coordinator-backed stack. It aggregates the product path, live-preview evidence,
CPU fallback, optional retained GPU generation evidence, release readiness, and
support diagnostics without changing runtime protocol boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
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

import public_swarm_live_preview_rc_pack as live_preview_pack  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID  # noqa: E402


SCHEMA = "public_swarm_operator_preview_v1"
DEVELOPER_PREVIEW_SCHEMA = "public_swarm_developer_preview_v1"
DEVELOPER_PREVIEW_CHECK_SCHEMA = "public_swarm_developer_preview_check_v1"
DEVELOPER_PREVIEW_RETAINED_SCHEMA = "public_swarm_developer_preview_retained_evidence_v1"
LIVE_PREVIEW_SCHEMA = "public_swarm_live_preview_rc_v1"
RELEASE_READINESS_SCHEMA = "release_readiness_v1"
GPU_GENERATION_SCHEMA = "gpu_sharded_generation_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
MODE_LOCAL_SMOKE = "local-smoke"
MODE_PACKAGE = "package"
MODE_LIVE_KAGGLE = "live-kaggle"
MODE_EVIDENCE_IMPORT = "evidence-import"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-operator-preview"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9350
DEFAULT_BASE_PORT = 9351
DEFAULT_GPU_REPORT = (
    "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
    "gpu_sharded_generation_beta_kaggle_auto.json"
)
DEFAULT_DEVELOPER_PREVIEW_REPORT = "dist/public-swarm-developer-preview/public_swarm_developer_preview.json"
DEFAULT_LIVE_STAGE0_REPORT = (
    "dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/"
    "public_swarm_live_preview_rc.json"
)
DEFAULT_LIVE_STAGE1_REPORT = (
    "dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/"
    "public_swarm_live_preview_rc.json"
)
DEFAULT_RELEASE_READINESS_REPORT = "dist/release-readiness/release_readiness.json"

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


def load_json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


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
        status = payload.get("release_status")
        if isinstance(status, dict):
            for code in status.get("diagnosis_codes") or []:
                if isinstance(code, str):
                    codes.add(code)
        summary = payload.get("diagnosis_summary")
        if isinstance(summary, dict):
            for code in summary.get("codes") or []:
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


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    allow_failure_payload: bool = False,
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

    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
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
        if not allow_failure_payload:
            step["ok"] = bool(step["ok"] and payload.get("ok"))
    if not step.get("ok"):
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:])
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:])
    return step, payload


def summarize_developer_preview(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    preview = payload.get("developer_preview") if isinstance(payload.get("developer_preview"), dict) else {}
    schema = payload.get("schema")
    ready = bool(
        payload.get("ok") is True
        and (
            (schema == DEVELOPER_PREVIEW_SCHEMA and "public_swarm_developer_preview_ready" in codes)
            or (schema == DEVELOPER_PREVIEW_CHECK_SCHEMA and "public_swarm_developer_preview_check_ready" in codes)
            or (schema == DEVELOPER_PREVIEW_RETAINED_SCHEMA and "developer_preview_retained_evidence_ready" in codes)
        )
    )
    return {
        "schema": schema,
        "ok": payload.get("ok"),
        "ready": ready,
        "mode": payload.get("mode"),
        "diagnosis_codes": sorted(codes),
        "product_beta_ready": preview.get("product_beta_ready"),
        "support_bundle_ready": preview.get("support_bundle_ready"),
        "cpu_fallback_ready": preview.get("cpu_fallback_ready"),
        "workload_type": preview.get("workload_type", WORKLOAD_TYPE),
    }


def synthesize_developer_payload_from_retained_live(live_summary: dict[str, Any]) -> dict[str, Any]:
    """Recover a developer-preview summary from retained Live Preview RC evidence."""
    codes = set(live_summary.get("diagnosis_codes") or [])
    retained_ready = bool(
        live_summary.get("ok")
        and (
            "public_swarm_developer_preview_ready" in codes
            or "public_swarm_developer_preview_check_ready" in codes
            or "developer_preview_ready" in codes
        )
    )
    if not retained_ready:
        return {}
    cpu_ready = bool("cpu_fallback_ready" in codes or "local_cpu_inference_ready" in codes)
    codes.update({
        "developer_preview_retained_evidence_ready",
        "public_swarm_developer_preview_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    })
    if cpu_ready:
        codes.add("cpu_fallback_ready")
    return {
        "schema": DEVELOPER_PREVIEW_RETAINED_SCHEMA,
        "ok": True,
        "mode": MODE_EVIDENCE_IMPORT,
        "developer_preview": {
            "ready": True,
            "product_beta_ready": None,
            "support_bundle_ready": True,
            "cpu_fallback_ready": cpu_ready,
            "workload_type": WORKLOAD_TYPE,
        },
        "diagnosis_codes": sorted(codes),
    }


def summarize_live_preview(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    preview = payload.get("live_preview") if isinstance(payload.get("live_preview"), dict) else {}
    ready = bool(
        payload.get("ok") is True
        and payload.get("schema") == LIVE_PREVIEW_SCHEMA
        and "public_swarm_live_preview_rc_ready" in codes
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "mode": payload.get("mode"),
        "diagnosis_codes": sorted(codes),
        "external_runtime_verified": preview.get("external_runtime_verified"),
        "fresh_live_kaggle_run": preview.get("fresh_live_kaggle_run"),
        "stage0_live_requeue_ready": "live_stage0_requeue_ready" in codes or preview.get("stage0_live_requeue_ready"),
        "stage1_live_requeue_ready": "live_stage1_requeue_ready" in codes or preview.get("stage1_live_requeue_ready"),
        "kaggle_kernels_deleted": "kaggle_kernels_deleted" in codes or preview.get("kaggle_kernels_deleted"),
        "private_artifacts_cleaned": "private_artifacts_cleaned" in codes or preview.get("private_artifacts_cleaned"),
        "token_rotation_required": "token_rotation_required" in codes or preview.get("token_rotation_required"),
    }


def summarize_release_readiness(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("release_status") if isinstance(payload.get("release_status"), dict) else {}
    codes = set(diagnosis_codes(payload))
    ready = bool(payload.get("ok") is True and payload.get("schema") == RELEASE_READINESS_SCHEMA and status.get("ready") is True)
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "status": status.get("status"),
        "diagnosis_codes": sorted(codes),
        "warnings": sorted(status.get("warnings") or []),
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
            "Public Swarm Operator Preview artifacts summarize operator-path readiness "
            "with counts, hashes, retained/live evidence, release readiness, and support diagnostics only. "
            "Run `crowdtensor generate` in human mode to see a local answer."
        ),
    }


def prompt_scope_summary() -> dict[str, Any]:
    return {
        "source": "prompt-text",
        "prompt_count": 1,
        "inline_prompt_text": True,
        "terminal_next_commands_local_private": True,
        "terminal_logs_local_private": True,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": True,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Operator Preview artifact records inherited prompt source/count and placeholder safety only; "
            "raw prompt text is excluded from public JSON, Markdown, and support bundles."
        ),
    }


def inherited_prompt_scope(*payloads: dict[str, Any]) -> dict[str, Any]:
    fallback = prompt_scope_summary()
    for payload in payloads:
        prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
        if prompt_scope:
            inherited = dict(fallback)
            inherited.update(prompt_scope)
            if not inherited.get("summary"):
                inherited["summary"] = fallback["summary"]
            return inherited
    return fallback


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
            "This Operator Preview report is shareable operator-path evidence, not a local "
            "answer transcript; raw prompts, generated text, generated token ids, "
            "activations, leases, credentials, private env files, and runtime state are excluded."
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
        "summary": "Share public_swarm_operator_preview.json/md and support_bundle.json; they contain hashes/counts and readiness evidence, not raw prompts or answers.",
    }


def output_request_text(summary: dict[str, Any]) -> str:
    return (
        f"include_output={bool(summary.get('include_output'))} "
        f"raw_generated_text_public={bool(summary.get('raw_generated_text_public'))} "
        f"public_artifact_safe={bool(summary.get('public_artifact_safe'))}"
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


def import_gpu_report(path_value: str) -> dict[str, Any]:
    return live_preview_pack.import_gpu_report(path_value)


def base_artifacts(output_dir: Path, *, ok: bool) -> dict[str, Any]:
    return {
        "public_swarm_operator_preview_json": artifact_entry(
            output_dir / "public_swarm_operator_preview.json",
            output_dir,
            kind="public_swarm_operator_preview",
            schema=SCHEMA,
            ok=ok,
        ),
        "public_swarm_operator_preview_markdown": artifact_entry(
            output_dir / "public_swarm_operator_preview.md",
            output_dir,
            kind="public_swarm_operator_preview_markdown",
        ),
    }


def safety_summary(*, mode: str, gpu_ready: bool, external_verified: bool, private_cleaned: bool) -> dict[str, Any]:
    return {
        "coordinator_backed_task_execution": True,
        "operator_preview_user_path": True,
        "developer_preview_product_surface": True,
        "p2p_lite_discovery_only": True,
        "cpu_only_default": True,
        "read_only_workload": WORKLOAD_TYPE,
        "live_kaggle_mode": mode == MODE_LIVE_KAGGLE,
        "external_runtime_verified": bool(external_verified),
        "private_artifacts_cleaned": bool(private_cleaned),
        "gpu_generation_evidence_imported": bool(gpu_ready),
        "tokens_public": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_payloads_redacted": True,
        "not_production": True,
        "not_p2p": True,
        "not_p2p_execution": True,
        "not_libp2p": True,
        "not_dht": True,
        "not_nat_traversal": True,
        "not_gpu_pooling_marketplace": True,
        "not_large_model_serving": True,
        "not_public_prompt_serving": True,
        "not_training": True,
    }


def limitations() -> list[str]:
    return [
        "Public Swarm v0.1 Operator Preview is Coordinator-backed and read-only; it is not production Swarm Inference.",
        "It uses tiny Hugging Face GPT split evidence and CPU fallback; it is not Hivemind/Petals-level serving or large-model public prompt serving.",
        "P2P-lite remains metadata discovery only; this is not libp2p, DHT, NAT traversal, decentralized security, or P2P execution.",
        "GPU generation is retained evidence import unless an explicit lower-level CUDA proof is run; this is not a GPU pooling marketplace.",
    ]


def operator_actions(mode: str, *, external_blocked: bool = False) -> list[str]:
    actions = [
        "Start with OPERATOR_PREVIEW.md; share only top-level JSON/Markdown and support bundle artifacts.",
        "Do not publish private env files, Kaggle kernels, raw runtime state, tokens, prompts, generated text, token ids, or activations.",
    ]
    if mode == MODE_LIVE_KAGGLE and external_blocked:
        actions.append("Live Kaggle proof was attempted but blocked; inspect live_attempt.step and retained evidence import summaries.")
    elif mode == MODE_LIVE_KAGGLE:
        actions.append("Rotate generated Coordinator and Miner tokens after the temporary public HTTP/Kaggle run.")
    elif mode == MODE_PACKAGE:
        actions.append("Use package mode before handing the preview runbook to a separate operator or host.")
    elif mode == MODE_EVIDENCE_IMPORT:
        actions.append("Use evidence-import only for retained, already redacted reports; it does not create a fresh live run.")
    else:
        actions.append("Use local-smoke as the CPU-safe, CI-safe v0.1 preview contract check.")
    return actions


def support_bundle_artifact(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    bundle = support_bundle.sanitize(redact_values({
        "schema": "public_swarm_operator_preview_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "operator_preview": report.get("operator_preview") or {},
        "artifacts": report.get("artifacts") or {},
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety") or {},
        "limitations": report.get("limitations") or [],
    }))
    path = output_dir / "support_bundle.json"
    write_json(path, bundle)
    return artifact_entry(path, output_dir, kind="public_swarm_operator_preview_support_bundle", schema=str(bundle.get("schema")), ok=bundle.get("ok"))


def write_operator_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "OPERATOR_PREVIEW.md"
    lines = [
        "# CrowdTensor Public Swarm v0.1 Operator Preview",
        "",
        "This runbook is the recommended user-facing path for the current Coordinator-backed preview.",
        "It is CPU-only by default, read-only, and not production Swarm Inference.",
        "",
        "## 1. Local CPU Preview",
        "",
        "```bash",
        "python -m pip install -e '.[dev]'",
        "crowdtensor operator-preview local-smoke --json",
        "```",
        "",
        "## 2. Prepare Operator Package",
        "",
        "```bash",
        f"crowdtensor operator-preview package --public-host {args.public_host} --json",
        "```",
        "",
        "Use the generated package to start the Coordinator and distinct stage0/stage1 Miners.",
        "",
        "## 3. Optional Fresh Kaggle Live Proof",
        "",
        "```bash",
        (
            "crowdtensor operator-preview live-kaggle "
            f"--public-host {args.public_host} --port {args.port} --base-port {args.base_port} "
            "--failure-mode kill-stage0-after-claim --json"
        ),
        "```",
        "",
        "If Kaggle or optional HF runtime dependencies are unavailable, the preview report falls back to retained evidence import and records `external_runtime_blocked`.",
        "",
        "## 4. Evidence Import",
        "",
        "```bash",
        "crowdtensor operator-preview evidence-import --json",
        "```",
        "",
        "Share only redacted top-level JSON/Markdown plus `support_bundle.json`. Rotate temporary public HTTP tokens after live proofs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="public_swarm_operator_preview_runbook")


def run_developer_preview(args: argparse.Namespace, *, output_dir: Path, mode: str, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    preview_mode = "local" if mode == MODE_LOCAL_SMOKE else "package"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_developer_preview_pack.py"),
        preview_mode,
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
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
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
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "public_swarm_developer_preview",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 420.0,
    )


def run_live_preview(args: argparse.Namespace, *, output_dir: Path, mode: str, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_live_preview_rc_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--ready-url",
        args.ready_url,
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
        "--developer-preview-report",
        args.developer_preview_report,
        "--alpha-rc-report",
        args.alpha_rc_report,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--failure-mode",
        args.failure_mode,
        "--kaggle-owner",
        args.kaggle_owner,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
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
        "--victim-compute-seconds",
        str(args.victim_compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--claim-observe-timeout",
        str(args.claim_observe_timeout),
        "--requeue-timeout",
        str(args.requeue_timeout),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    command.append("--inline-kernel-payload" if args.inline_kernel_payload else "--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    if args.keep_live_private_artifacts:
        command.append("--keep-live-private-artifacts")
    if args.keep_child_artifacts:
        command.append("--keep-child-artifacts")
    return run_json_step(
        "public_swarm_live_preview_rc",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 960.0,
        allow_failure_payload=True,
    )


def run_release_readiness(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "release_readiness_pack.py"),
        "--output-dir",
        str(output_dir),
        "--host",
        "127.0.0.1",
        "--base-port",
        str(args.release_base_port),
        "--request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
    ]
    if args.allow_dirty_release:
        command.append("--allow-dirty")
    return run_json_step(
        "release_readiness",
        command,
        runner=runner,
        timeout_seconds=max(float(args.release_timeout_seconds), 60.0),
    )


def import_dual_live_evidence(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    stage0_payload = load_json(args.live_stage0_report)
    stage1_payload = load_json(args.live_stage1_report)
    stage0 = summarize_live_preview(stage0_payload)
    stage1 = summarize_live_preview(stage1_payload)
    codes0 = set(stage0.get("diagnosis_codes") or [])
    codes1 = set(stage1.get("diagnosis_codes") or [])
    ready = bool(
        stage0.get("ready")
        and stage1.get("ready")
        and ("live_stage0_requeue_ready" in codes0 or stage0.get("stage0_live_requeue_ready"))
        and ("live_stage1_requeue_ready" in codes1 or stage1.get("stage1_live_requeue_ready"))
        and "kaggle_kernels_deleted" in codes0
        and "kaggle_kernels_deleted" in codes1
    )
    summary = {
        "schema": "retained_public_swarm_live_preview_evidence_v1",
        "ok": ready,
        "stage0": stage0,
        "stage1": stage1,
        "diagnosis_codes": sorted(codes0 | codes1),
    }
    combined_payload = {
        "schema": LIVE_PREVIEW_SCHEMA,
        "ok": ready,
        "diagnosis_codes": summary["diagnosis_codes"],
        "live_preview": {
            "ready": ready,
            "external_runtime_verified": bool(stage0.get("external_runtime_verified") and stage1.get("external_runtime_verified")),
            "fresh_live_kaggle_run": False,
            "stage0_live_requeue_ready": bool(stage0.get("stage0_live_requeue_ready")),
            "stage1_live_requeue_ready": bool(stage1.get("stage1_live_requeue_ready")),
            "kaggle_kernels_deleted": ready,
            "private_artifacts_cleaned": bool(stage0.get("private_artifacts_cleaned") and stage1.get("private_artifacts_cleaned")),
            "token_rotation_required": bool(stage0.get("token_rotation_required") or stage1.get("token_rotation_required")),
        },
    }
    return summary, combined_payload


def common_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mode: str,
    steps: list[dict[str, Any]],
    developer_payload: dict[str, Any],
    live_payload: dict[str, Any],
    release_payload: dict[str, Any],
    live_evidence_summary: dict[str, Any] | None = None,
    live_attempt_blocked: dict[str, Any] | None = None,
) -> dict[str, Any]:
    developer_summary = summarize_developer_preview(developer_payload)
    live_summary = summarize_live_preview(live_payload)
    release_summary = summarize_release_readiness(release_payload)
    gpu_summary = import_gpu_report(args.gpu_report)
    live_codes = set(live_summary.get("diagnosis_codes") or [])
    developer_ready = bool(developer_summary.get("ready"))
    live_ready = bool(live_summary.get("ready"))
    release_ready = bool(release_summary.get("ready")) if release_payload else True
    developer_codes = set(developer_summary.get("diagnosis_codes") or [])
    cpu_ready = bool(
        "cpu_fallback_ready" in developer_codes
        or "local_cpu_inference_ready" in developer_codes
        or developer_summary.get("cpu_fallback_ready")
    )
    retained_evidence_ready = bool(
        mode == MODE_EVIDENCE_IMPORT
        and live_ready
        and live_evidence_summary is not None
        and live_evidence_summary.get("ok") is True
    )
    developer_degraded = bool(
        not developer_ready
        and cpu_ready
        and (
            "hf_dependencies_missing" in developer_codes
            or "developer_preview_blocked" in developer_codes
            or "public_swarm_product_beta_blocked" in developer_codes
        )
    )
    serve_ready = bool(
        "serve_join_generate_ready" in developer_codes
        or "serve_join_generate_loop_ready" in developer_codes
    )
    package_ready = bool(
        mode == MODE_PACKAGE
        and "miner_join_pack_ready" in developer_codes
        and "private_artifacts_local_only" in developer_codes
    )
    external_verified = bool(live_summary.get("external_runtime_verified"))
    private_cleaned = bool(live_summary.get("private_artifacts_cleaned") or mode in {MODE_LOCAL_SMOKE, MODE_PACKAGE, MODE_EVIDENCE_IMPORT})
    cpu_fallback_user_path_ready = bool(cpu_ready and live_ready and release_ready and (developer_degraded or developer_ready))
    retained_user_path_ready = bool(retained_evidence_ready and (developer_ready or "developer_preview_retained_evidence_ready" in developer_codes))
    user_path_ready = bool(serve_ready or package_ready or cpu_fallback_user_path_ready or retained_user_path_ready)
    support_ready = bool(live_ready and (developer_ready or cpu_ready or package_ready or retained_evidence_ready))
    runtime_floor_ready = bool(cpu_ready or retained_evidence_ready)
    ready = bool(live_ready and release_ready and support_ready and runtime_floor_ready and user_path_ready)

    codes = set(diagnosis_codes(developer_payload, live_payload, release_payload)) | set(gpu_summary.get("diagnosis_codes") or [])
    if ready:
        codes.update({
            "public_swarm_operator_preview_ready",
            "operator_preview_user_path_ready",
            "cpu_fallback_ready",
            "live_preview_ready",
            "support_bundle_ready",
            "release_readiness_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        })
        if serve_ready:
            codes.add("serve_join_generate_ready")
        if package_ready:
            codes.update({"miner_join_pack_ready", "private_artifacts_local_only"})
    else:
        codes.add("public_swarm_operator_preview_blocked")
    if developer_degraded:
        codes.update({"developer_preview_degraded", "operator_preview_cpu_fallback_user_path_ready"})
    if retained_evidence_ready:
        codes.add("operator_preview_retained_evidence_ready")
    if retained_user_path_ready:
        codes.add("operator_preview_retained_evidence_user_path_ready")
    if gpu_summary.get("ok"):
        codes.add("gpu_generation_evidence_import_ready")
    if live_attempt_blocked:
        codes.add("external_runtime_blocked")
    if mode == MODE_PACKAGE:
        codes.add("operator_preview_package_ready")
    if mode == MODE_LOCAL_SMOKE:
        codes.add("operator_preview_local_smoke_ready")
    if mode == MODE_LIVE_KAGGLE and not live_attempt_blocked:
        codes.add("operator_preview_live_kaggle_ready")
    if mode == MODE_EVIDENCE_IMPORT:
        codes.add("operator_preview_evidence_import_ready")

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": mode,
        "target": args.target,
        "output_dir": str(output_dir),
        "operator_preview": {
            "ready": ready,
            "developer_preview_ready": developer_ready,
            "developer_preview_degraded": developer_degraded,
            "serve_join_generate_ready": serve_ready,
            "package_ready": package_ready,
            "user_path_ready": user_path_ready,
            "cpu_fallback_ready": cpu_ready,
            "cpu_fallback_user_path_ready": cpu_fallback_user_path_ready,
            "retained_evidence_ready": retained_evidence_ready,
            "retained_evidence_user_path_ready": retained_user_path_ready,
            "live_preview_ready": live_ready,
            "release_readiness_ready": release_ready,
            "support_bundle_ready": support_ready,
            "gpu_generation_evidence_ready": bool(gpu_summary.get("ok")),
            "external_runtime_verified": external_verified,
            "fresh_live_kaggle_run": bool(live_summary.get("fresh_live_kaggle_run")),
            "external_runtime_blocked": bool(live_attempt_blocked),
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "user_surface": ["operator-preview", "serve", "join", "generate", "support bundle"],
        },
        "steps": steps,
        "payload_summaries": {
            "developer_preview": developer_summary,
            "live_preview": live_summary,
            "release_readiness": release_summary,
            "gpu_generation_evidence": gpu_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": base_artifacts(output_dir, ok=ready),
        "safety": safety_summary(mode=mode, gpu_ready=bool(gpu_summary.get("ok")), external_verified=external_verified, private_cleaned=private_cleaned),
        "operator_action": operator_actions(mode, external_blocked=bool(live_attempt_blocked)),
        "limitations": limitations(),
    }
    report["output_request"] = output_request_summary()
    report["prompt_scope"] = inherited_prompt_scope(developer_payload, live_payload)
    report["answer_scope"] = answer_scope_summary()
    report["shareable_summary"] = shareable_summary()
    if live_evidence_summary is not None:
        report["payload_summaries"]["retained_live_evidence"] = live_evidence_summary
    if live_attempt_blocked is not None:
        report["payload_summaries"]["live_attempt"] = live_attempt_blocked
    return report


def build_local_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    developer_step, developer_payload = run_developer_preview(args, output_dir=output_dir / "developer-preview", mode=MODE_LOCAL_SMOKE, runner=runner)
    live_step, live_payload = run_live_preview(args, output_dir=output_dir / "live-preview", mode=MODE_LOCAL_SMOKE, runner=runner)
    release_step, release_payload = run_release_readiness(args, output_dir=output_dir / "release-readiness", runner=runner)
    return common_report(
        args,
        output_dir=output_dir,
        mode=MODE_LOCAL_SMOKE,
        steps=[developer_step, live_step, release_step],
        developer_payload=developer_payload,
        live_payload=live_payload,
        release_payload=release_payload,
    )


def build_package(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    developer_step, developer_payload = run_developer_preview(args, output_dir=output_dir / "developer-preview-package", mode=MODE_PACKAGE, runner=runner)
    live_step, live_payload = run_live_preview(args, output_dir=output_dir / "live-preview-package", mode=MODE_PACKAGE, runner=runner)
    release_step, release_payload = run_release_readiness(args, output_dir=output_dir / "release-readiness", runner=runner)
    runbook = write_operator_runbook(args, output_dir)
    report = common_report(
        args,
        output_dir=output_dir,
        mode=MODE_PACKAGE,
        steps=[developer_step, live_step, release_step],
        developer_payload=developer_payload,
        live_payload=live_payload,
        release_payload=release_payload,
    )
    report["artifacts"]["operator_preview_runbook"] = runbook
    return report


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    developer_payload = load_json(args.developer_preview_report)
    release_payload = load_json(args.release_readiness_report)
    live_summary, live_payload = import_dual_live_evidence(args)
    if not developer_payload:
        developer_payload = synthesize_developer_payload_from_retained_live(live_summary)
    return common_report(
        args,
        output_dir=output_dir,
        mode=MODE_EVIDENCE_IMPORT,
        steps=[],
        developer_payload=developer_payload,
        live_payload=live_payload,
        release_payload=release_payload,
        live_evidence_summary=live_summary,
    )


def build_live_kaggle(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    developer_step, developer_payload = run_developer_preview(args, output_dir=output_dir / "developer-preview", mode=MODE_LOCAL_SMOKE, runner=runner)
    live_step, live_payload = run_live_preview(args, output_dir=output_dir / "live-preview-fresh", mode=MODE_LIVE_KAGGLE, runner=runner)
    release_step, release_payload = run_release_readiness(args, output_dir=output_dir / "release-readiness", runner=runner)
    live_attempt_blocked = None
    live_evidence_summary = None
    if not (live_step.get("ok") and live_payload.get("ok") is True):
        live_evidence_summary, retained_payload = import_dual_live_evidence(args)
        live_attempt_blocked = {
            "ok": False,
            "step": live_step,
            "payload_schema": live_payload.get("schema") if live_payload else "",
            "payload_ok": live_payload.get("ok") if live_payload else None,
            "diagnosis_codes": diagnosis_codes(live_payload, extra=["external_runtime_blocked"]),
        }
        live_payload = retained_payload
    return common_report(
        args,
        output_dir=output_dir,
        mode=MODE_LIVE_KAGGLE,
        steps=[developer_step, live_step, release_step],
        developer_payload=developer_payload,
        live_payload=live_payload,
        release_payload=release_payload,
        live_evidence_summary=live_evidence_summary,
        live_attempt_blocked=live_attempt_blocked,
    )


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary())
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report["artifacts"]["gpu_generation_evidence_json"] = artifact_entry(
        Path(report.get("gpu_report_path") or DEFAULT_GPU_REPORT),
        output_dir,
        kind="gpu_sharded_generation_beta",
        schema=GPU_GENERATION_SCHEMA,
    )
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report)
    report = support_bundle.sanitize(redact_values(report))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Operator Preview report contained secret-like fragments"
    json_path = output_dir / "public_swarm_operator_preview.json"
    markdown_path = output_dir / "public_swarm_operator_preview.md"
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_operator_preview_json"]["present"] = True
    report["artifacts"]["public_swarm_operator_preview_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    preview = report.get("operator_preview") if isinstance(report.get("operator_preview"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm v0.1 Operator Preview",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- ready: `{preview.get('ready')}`",
        f"- user_path_ready: `{preview.get('user_path_ready')}`",
        f"- developer_preview_ready: `{preview.get('developer_preview_ready')}`",
        f"- developer_preview_degraded: `{preview.get('developer_preview_degraded')}`",
        f"- serve_join_generate_ready: `{preview.get('serve_join_generate_ready')}`",
        f"- cpu_fallback_ready: `{preview.get('cpu_fallback_ready')}`",
        f"- cpu_fallback_user_path_ready: `{preview.get('cpu_fallback_user_path_ready')}`",
        f"- retained_evidence_ready: `{preview.get('retained_evidence_ready')}`",
        f"- live_preview_ready: `{preview.get('live_preview_ready')}`",
        f"- release_readiness_ready: `{preview.get('release_readiness_ready')}`",
        f"- external_runtime_verified: `{preview.get('external_runtime_verified')}`",
        f"- external_runtime_blocked: `{preview.get('external_runtime_blocked')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
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
        "## Artifacts",
        "",
    ]
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_LOCAL_SMOKE:
        report = build_local_smoke(args, output_dir=output_dir, runner=runner)
    elif args.mode == MODE_PACKAGE:
        report = build_package(args, output_dir=output_dir, runner=runner)
    elif args.mode == MODE_LIVE_KAGGLE:
        report = build_live_kaggle(args, output_dir=output_dir, runner=runner)
    else:
        report = build_evidence_import(args, output_dir=output_dir)
    report["gpu_report_path"] = args.gpu_report
    return persist_report(report, output_dir=output_dir)


def default_kaggle_owner() -> str:
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
    parser = argparse.ArgumentParser(description="Build Public Swarm v0.1 Operator Preview evidence.")
    parser.add_argument("mode", choices=[MODE_LOCAL_SMOKE, MODE_PACKAGE, MODE_LIVE_KAGGLE, MODE_EVIDENCE_IMPORT])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--release-base-port", type=int, default=9360)
    parser.add_argument("--ready-url", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--target", choices=["local", "kaggle"], default="kaggle")
    parser.add_argument("--miner-id-prefix", default="public-swarm-operator-preview")
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--developer-preview-report", default=DEFAULT_DEVELOPER_PREVIEW_REPORT)
    parser.add_argument("--alpha-rc-report", default=live_preview_pack.DEFAULT_ALPHA_RC_REPORT)
    parser.add_argument("--live-stage0-report", default=DEFAULT_LIVE_STAGE0_REPORT)
    parser.add_argument("--live-stage1-report", default=DEFAULT_LIVE_STAGE1_REPORT)
    parser.add_argument("--release-readiness-report", default=DEFAULT_RELEASE_READINESS_REPORT)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--cpu-request-count", type=int, default=1)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--failure-mode", choices=sorted(live_preview_pack.FAILURE_MODES), default=live_preview_pack.FAILURE_KILL_STAGE0_AFTER_CLAIM)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--dataset-slug", default="")
    parser.add_argument("--dataset-title", default="CrowdTensor Public Swarm v0.1 Operator Preview")
    parser.add_argument("--kernel-slug-prefix", default="ct-operator-preview")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor Operator Preview")
    parser.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-live-private-artifacts", action="store_true")
    parser.add_argument("--keep-child-artifacts", action="store_true")
    parser.add_argument("--allow-dirty-release", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--release-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
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
    if args.port < 1 or args.base_port < 1 or args.release_base_port < 1:
        raise SystemExit("--port, --base-port, and --release-base-port must be positive")
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.cpu_request_count < 1 or args.cpu_request_count > 4:
        raise SystemExit("--cpu-request-count must be between 1 and 4")
    if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
        raise SystemExit("--external-llm-request-count must be between 1 and 4")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "cpu_timeout_seconds",
        "release_timeout_seconds",
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
    if args.mode == MODE_LIVE_KAGGLE and not args.kaggle_owner:
        raise SystemExit("--kaggle-owner or KAGGLE_USERNAME is required for live-kaggle")
    if args.mode == MODE_LIVE_KAGGLE and args.failure_mode != live_preview_pack.FAILURE_NONE:
        if args.victim_compute_seconds <= args.lease_seconds:
            args.victim_compute_seconds = args.lease_seconds + 30.0
        if args.requeue_timeout <= args.lease_seconds:
            args.requeue_timeout = args.lease_seconds + 45.0
    return args


def print_human(report: dict[str, Any]) -> None:
    preview = report.get("operator_preview") if isinstance(report.get("operator_preview"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    print("CrowdTensor Public Swarm v0.1 Operator Preview")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {preview.get('ready')}")
    print(f"  external_runtime_verified: {preview.get('external_runtime_verified')}")
    print(f"  external_runtime_blocked: {preview.get('external_runtime_blocked')}")
    if output_request:
        print(f"  output_request: {output_request_text(output_request)}")
    if prompt_scope:
        print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
        if prompt_scope_note(prompt_scope):
            print(f"  prompt_scope_note: {prompt_scope_note(prompt_scope)}")
    if answer_scope:
        print(f"  answer_scope: {answer_scope_text(answer_scope)}")
        print(f"  answer_scope_note: {answer_scope_note(answer_scope)}")
    if shareable:
        print(f"  shareable: {shareable_summary_text(shareable)}")
    print(f"  output: {report.get('output_dir')}")
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
