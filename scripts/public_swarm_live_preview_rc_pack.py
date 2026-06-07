#!/usr/bin/env python3
"""Build the Public Swarm Live Preview release-candidate artifact.

This layer turns the existing Developer Preview and Public Swarm live proof
paths into one shareable RC artifact. It does not add a new runtime protocol;
it aggregates the Coordinator-backed, read-only tiny GPT split evidence and
keeps the production/P2P/large-model boundaries explicit.
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

import support_bundle  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID  # noqa: E402


SCHEMA = "public_swarm_live_preview_rc_v1"
DEVELOPER_PREVIEW_SCHEMA = "public_swarm_developer_preview_v1"
DEVELOPER_PREVIEW_CHECK_SCHEMA = "public_swarm_developer_preview_check_v1"
ALPHA_SCHEMA = "public_swarm_inference_alpha_v1"
ALPHA_CHECK_SCHEMA = "public_swarm_inference_alpha_check_v1"
ALPHA_RC_SCHEMA = "public_swarm_inference_alpha_rc_v1"
GPU_GENERATION_SCHEMA = "gpu_sharded_generation_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
MODE_LOCAL_SMOKE = "local-smoke"
MODE_PACKAGE = "package"
MODE_LIVE_KAGGLE = "live-kaggle"
MODE_EVIDENCE_IMPORT = "evidence-import"
FAILURE_NONE = "none"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"
FAILURE_MODES = {
    FAILURE_NONE,
    FAILURE_KILL_STAGE0_AFTER_CLAIM,
    FAILURE_KILL_STAGE1_AFTER_CLAIM,
}
DEFAULT_OUTPUT_DIR = "dist/public-swarm-live-preview-rc"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9340
DEFAULT_BASE_PORT = 9341
DEFAULT_DEVELOPER_PREVIEW_REPORT = "dist/public-swarm-developer-preview/public_swarm_developer_preview.json"
DEFAULT_ALPHA_RC_REPORT = "dist/public-swarm-inference-alpha-rc/public_swarm_inference_alpha_rc.json"
DEFAULT_GPU_REPORT = (
    "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
    "gpu_sharded_generation_beta_kaggle_auto.json"
)

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
    '"generated_text":',
    '"generated_token_ids":',
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
        step["ok"] = bool(step["ok"] and payload.get("ok"))
    if not step.get("ok"):
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:], secret_values)
    return step, payload


def import_gpu_report(path_value: str) -> dict[str, Any]:
    payload = load_json(Path(path_value))
    if not payload:
        return {
            "ok": False,
            "present": False,
            "schema": "",
            "path": path_value,
            "diagnosis_codes": ["gpu_generation_evidence_missing"],
        }
    codes = set(diagnosis_codes(payload))
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    raw_public = generation.get("raw_generated_text_public", payload.get("raw_generated_text_public"))
    generated_count = generation.get("generated_token_count", payload.get("generated_token_count"))
    generated_hash = generation.get("generated_text_hash", payload.get("generated_text_hash"))
    ready = bool(
        payload.get("schema") == GPU_GENERATION_SCHEMA
        and payload.get("ok") is True
        and "multi_token_generation_ready" in codes
        and raw_public is False
    )
    return {
        "ok": ready,
        "present": True,
        "schema": payload.get("schema"),
        "path": path_value,
        "diagnosis_codes": sorted(codes),
        "generated_token_count": generated_count,
        "generated_text_hash": generated_hash,
        "raw_generated_text_public": raw_public,
    }


def summarize_developer_preview(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    preview = payload.get("developer_preview") if isinstance(payload.get("developer_preview"), dict) else {}
    ready = bool(
        payload.get("ok") is True
        and (
            payload.get("schema") == DEVELOPER_PREVIEW_SCHEMA
            or payload.get("schema") == DEVELOPER_PREVIEW_CHECK_SCHEMA
        )
        and (
            "public_swarm_developer_preview_ready" in codes
            or "public_swarm_developer_preview_check_ready" in codes
        )
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "mode": payload.get("mode"),
        "target": payload.get("target"),
        "diagnosis_codes": sorted(codes),
        "product_beta_ready": preview.get("product_beta_ready"),
        "support_bundle_ready": preview.get("support_bundle_ready"),
        "cpu_fallback_ready": preview.get("cpu_fallback_ready"),
        "workload_type": preview.get("workload_type", WORKLOAD_TYPE),
    }


def summarize_alpha(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    cleanup = payload.get("artifact_cleanup") if isinstance(payload.get("artifact_cleanup"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": bool(payload.get("ok") is True and payload.get("schema") == ALPHA_SCHEMA),
        "mode": payload.get("mode"),
        "failure_mode": payload.get("failure_mode"),
        "diagnosis_codes": sorted(codes),
        "external_runtime_verified": session.get("live_external_runtime_verified"),
        "decoded_tokens_match": session.get("decoded_tokens_match"),
        "distinct_stage_miners": session.get("distinct_stage_miners"),
        "stage_assignment_valid": session.get("stage_assignment_valid"),
        "local_stage_requeue_verified": session.get("local_stage_requeue_verified"),
        "live_stage_requeue_verified": session.get("live_stage_requeue_verified"),
        "live_kaggle_kernels_deleted": session.get("live_kaggle_kernels_deleted"),
        "child_artifacts_pruned": cleanup.get("child_artifacts_pruned"),
    }


def summarize_alpha_rc(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    rc = payload.get("release_candidate") if isinstance(payload.get("release_candidate"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": bool(payload.get("ok") is True and payload.get("schema") == ALPHA_RC_SCHEMA),
        "mode": payload.get("mode"),
        "diagnosis_codes": sorted(codes),
        "stage0_live_requeue_ready": rc.get("stage0_live_requeue_ready"),
        "stage1_live_requeue_ready": rc.get("stage1_live_requeue_ready"),
        "evidence_imported": rc.get("evidence_imported"),
    }


def common_safety(*, mode: str, gpu_ready: bool, external_verified: bool, private_cleaned: bool) -> dict[str, Any]:
    return {
        "coordinator_backed_task_execution": True,
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
        "token_values_redacted": True,
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
        "Public Swarm Live Preview RC is Coordinator-backed and read-only; it is not production Swarm Inference.",
        "It aggregates tiny Hugging Face GPT split evidence; it is not Hivemind/Petals-level serving or large-model public prompt serving.",
        "P2P-lite remains metadata discovery only; this is not libp2p, DHT, NAT traversal, decentralized security, or P2P execution.",
        "GPU generation is retained evidence import unless an explicit lower-level CUDA proof is run; this is not a GPU pooling marketplace.",
    ]


def operator_actions(mode: str) -> list[str]:
    actions = [
        "Share only the top-level JSON/Markdown report and support bundle.",
        "Do not publish private Kaggle kernels, private env files, raw runtime state, tokens, activations, prompt text, generated text, or token ids.",
    ]
    if mode == MODE_LIVE_KAGGLE:
        actions.append("Rotate generated Coordinator and Miner tokens after every temporary public HTTP/Kaggle run.")
    elif mode == MODE_PACKAGE:
        actions.append("Use package mode to prepare operator materials before starting a controlled public Coordinator.")
    elif mode == MODE_EVIDENCE_IMPORT:
        actions.append("Use evidence-import only for retained, already redacted reports; it does not create a fresh live run.")
    else:
        actions.append("Use local-smoke as a CI-safe contract check; it does not create Kaggle resources.")
    return actions


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Live Preview RC artifacts summarize live preview readiness, "
            "stage requeue evidence, Kaggle cleanup state, GPU retained-evidence "
            "import, hashes/counts, and support diagnostics only. Run `crowdtensor "
            "generate` in local human mode to display answer text."
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
            "This Live Preview RC report is shareable release evidence, not a local "
            "answer transcript; raw prompts, generated text, generated token ids, "
            "activations, leases, credentials, private env files, and raw runtime "
            "state are excluded."
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
            "Share public_swarm_live_preview_rc.json/md and support_bundle.json; "
            "they contain readiness evidence, cleanup state, hashes, counts, and "
            "diagnostics, not raw prompts or answers."
        ),
    }


def support_bundle_artifact(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    bundle = support_bundle.sanitize(redact_values({
        "schema": "public_swarm_live_preview_rc_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "preview": report.get("live_preview") or {},
        "artifacts": report.get("artifacts") or {},
        "safety": report.get("safety") or {},
        "output_request": report.get("output_request") or output_request_summary(),
        "answer_scope": report.get("answer_scope") or answer_scope_summary(),
        "shareable_summary": report.get("shareable_summary") or shareable_summary(),
        "limitations": report.get("limitations") or [],
    }))
    path = output_dir / "support_bundle.json"
    write_json(path, bundle)
    return artifact_entry(path, output_dir, kind="public_swarm_live_preview_rc_support_bundle", schema=str(bundle.get("schema")), ok=bundle.get("ok"))


def base_artifacts(output_dir: Path, *, ok: bool) -> dict[str, Any]:
    return {
        "public_swarm_live_preview_rc_json": artifact_entry(
            output_dir / "public_swarm_live_preview_rc.json",
            output_dir,
            kind="public_swarm_live_preview_rc",
            schema=SCHEMA,
            ok=ok,
        ),
        "public_swarm_live_preview_rc_markdown": artifact_entry(
            output_dir / "public_swarm_live_preview_rc.md",
            output_dir,
            kind="public_swarm_live_preview_rc_markdown",
        ),
    }


def run_developer_preview_check(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_developer_preview_check.py"),
        "--mode",
        "local",
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    return run_json_step(
        "public_swarm_developer_preview_contract",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), 60.0) + 60.0,
    )


def run_alpha_contract(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_alpha_check.py"),
        "--mode",
        "live-kaggle",
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--json",
    ]
    return run_json_step(
        "public_swarm_inference_alpha_contract",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), 60.0) + 60.0,
    )


def run_developer_preview_package(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_developer_preview_pack.py"),
        "package",
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
        "public_swarm_developer_preview_package",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 420.0,
    )


def run_alpha_live(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_alpha_pack.py"),
        "--mode",
        "live-kaggle",
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
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--request-count",
        str(args.request_count),
        "--hf-model-id",
        args.hf_model_id,
        "--failure-mode",
        args.failure_mode,
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
    if args.keep_child_artifacts:
        command.append("--keep-child-artifacts")
    return run_json_step(
        "public_swarm_inference_alpha_live_kaggle",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 300.0) + 900.0,
    )


def write_package_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "PUBLIC_SWARM_LIVE_PREVIEW_RC.md"
    lines = [
        "# CrowdTensor Public Swarm Live Preview RC",
        "",
        "This package prepares a controlled public live preview. It is Coordinator-backed, read-only, and not production Swarm Inference.",
        "",
        "## Fresh Live Proof",
        "",
        "```bash",
        (
            "crowdtensor live-preview live-kaggle "
            f"--public-host {args.public_host} --port {args.port} --base-port {args.base_port} "
            "--failure-mode kill-stage0-after-claim --json"
        ),
        "```",
        "",
        "Run a second proof with `--failure-mode kill-stage1-after-claim` before publishing RC evidence.",
        "",
        "## Evidence Import",
        "",
        "```bash",
        "crowdtensor live-preview evidence-import --json",
        "```",
        "",
        "Rotate tokens after temporary public HTTP/Kaggle proofs and share only redacted top-level JSON/Markdown artifacts.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="public_swarm_live_preview_rc_runbook")


def build_local_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    developer_step, developer_payload = run_developer_preview_check(args, output_dir=output_dir / "developer-preview-check", runner=runner)
    alpha_step, alpha_payload = run_alpha_contract(args, output_dir=output_dir / "public-swarm-alpha-check", runner=runner)
    developer_summary = summarize_developer_preview(developer_payload)
    alpha_codes = set(diagnosis_codes(alpha_payload))
    alpha_ready = bool(
        alpha_step.get("ok")
        and alpha_payload.get("schema") == ALPHA_CHECK_SCHEMA
        and "public_swarm_inference_alpha_check_ready" in alpha_codes
    )
    developer_ready = bool(developer_step.get("ok") and developer_summary.get("ready"))
    gpu_summary = import_gpu_report(args.gpu_report)
    ready = bool(developer_ready and alpha_ready)
    codes = set(diagnosis_codes(developer_payload, alpha_payload)) | set(gpu_summary.get("diagnosis_codes") or [])
    if ready:
        codes.update({
            "public_swarm_live_preview_rc_ready",
            "public_swarm_live_preview_local_smoke_ready",
            "public_swarm_live_preview_contract_ready",
            "developer_preview_ready",
            "public_swarm_developer_preview_ready",
            "support_bundle_ready",
            "cpu_fallback_ready",
            "local_cpu_inference_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        })
    else:
        codes.add("public_swarm_live_preview_local_smoke_blocked")
    if gpu_summary.get("ok"):
        codes.add("gpu_generation_evidence_import_ready")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_LOCAL_SMOKE,
        "target": args.target,
        "output_dir": str(output_dir),
        "live_preview": {
            "ready": ready,
            "developer_preview_ready": developer_ready,
            "alpha_contract_ready": alpha_ready,
            "external_runtime_verified": False,
            "fresh_live_kaggle_run": False,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
        },
        "steps": [developer_step, alpha_step],
        "payload_summaries": {
            "developer_preview": developer_summary,
            "public_swarm_alpha_contract": {
                "schema": alpha_payload.get("schema"),
                "ok": alpha_payload.get("ok"),
                "ready": alpha_ready,
                "diagnosis_codes": sorted(alpha_codes),
            },
            "gpu_generation_evidence": gpu_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": base_artifacts(output_dir, ok=ready),
        "safety": common_safety(mode=MODE_LOCAL_SMOKE, gpu_ready=bool(gpu_summary.get("ok")), external_verified=False, private_cleaned=True),
        "operator_action": operator_actions(MODE_LOCAL_SMOKE),
        "limitations": limitations(),
    }
    report["artifacts"]["gpu_generation_evidence_json"] = artifact_entry(
        Path(args.gpu_report),
        output_dir,
        kind="gpu_sharded_generation_beta",
        schema=GPU_GENERATION_SCHEMA,
        ok=bool(gpu_summary.get("ok")),
    )
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report)
    return report


def build_package(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    developer_step, developer_payload = run_developer_preview_package(args, output_dir=output_dir / "developer-preview-package", runner=runner)
    developer_summary = summarize_developer_preview(developer_payload)
    developer_codes = set(diagnosis_codes(developer_payload))
    developer_ready = bool(
        developer_step.get("ok")
        and developer_summary.get("ready")
        and "developer_preview_package_ready" in developer_codes
        and "private_artifacts_local_only" in developer_codes
    )
    gpu_summary = import_gpu_report(args.gpu_report)
    runbook = write_package_runbook(args, output_dir)
    ready = bool(developer_ready and runbook.get("present"))
    codes = set(developer_codes) | set(gpu_summary.get("diagnosis_codes") or [])
    if ready:
        codes.update({
            "public_swarm_live_preview_rc_ready",
            "public_swarm_live_preview_package_ready",
            "developer_preview_ready",
            "public_swarm_developer_preview_ready",
            "miner_join_pack_ready",
            "private_artifacts_local_only",
            "support_bundle_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        })
    else:
        codes.add("public_swarm_live_preview_package_blocked")
    if gpu_summary.get("ok"):
        codes.add("gpu_generation_evidence_import_ready")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_PACKAGE,
        "target": args.target,
        "output_dir": str(output_dir),
        "live_preview": {
            "ready": ready,
            "developer_preview_ready": developer_ready,
            "package_ready": ready,
            "external_runtime_verified": False,
            "fresh_live_kaggle_run": False,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
        },
        "steps": [developer_step],
        "payload_summaries": {
            "developer_preview": developer_summary,
            "gpu_generation_evidence": gpu_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": base_artifacts(output_dir, ok=ready),
        "safety": common_safety(mode=MODE_PACKAGE, gpu_ready=bool(gpu_summary.get("ok")), external_verified=False, private_cleaned=True),
        "operator_action": operator_actions(MODE_PACKAGE),
        "limitations": limitations(),
    }
    report["artifacts"]["developer_preview_json"] = artifact_entry(
        output_dir / "developer-preview-package" / "public_swarm_developer_preview.json",
        output_dir,
        kind="public_swarm_developer_preview",
        schema=DEVELOPER_PREVIEW_SCHEMA,
        ok=developer_payload.get("ok") if developer_payload else None,
    )
    report["artifacts"]["gpu_generation_evidence_json"] = artifact_entry(
        Path(args.gpu_report),
        output_dir,
        kind="gpu_sharded_generation_beta",
        schema=GPU_GENERATION_SCHEMA,
        ok=bool(gpu_summary.get("ok")),
    )
    report["artifacts"]["public_swarm_live_preview_runbook"] = runbook
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report)
    return report


def build_live_kaggle(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    developer_step, developer_payload = run_developer_preview_check(args, output_dir=output_dir / "developer-preview-check", runner=runner)
    alpha_step, alpha_payload = run_alpha_live(args, output_dir=output_dir / "public-swarm-alpha-live", runner=runner)
    developer_summary = summarize_developer_preview(developer_payload)
    alpha_summary = summarize_alpha(alpha_payload)
    gpu_summary = import_gpu_report(args.gpu_report)
    alpha_codes = set(diagnosis_codes(alpha_payload))
    expected_requeue_code = ""
    if args.failure_mode == FAILURE_KILL_STAGE0_AFTER_CLAIM:
        expected_requeue_code = "live_stage0_requeue_ready"
    elif args.failure_mode == FAILURE_KILL_STAGE1_AFTER_CLAIM:
        expected_requeue_code = "live_stage1_requeue_ready"
    required_alpha_codes = {
        "public_swarm_inference_alpha_ready",
        "public_swarm_session_ready",
        "public_swarm_live_kaggle_ready",
        "external_runtime_verified",
        "kaggle_kernels_deleted",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "token_rotation_required",
    }
    if args.failure_mode != FAILURE_NONE:
        required_alpha_codes.add("external_stage_requeue_ready")
        required_alpha_codes.add(expected_requeue_code)
    alpha_ready = bool(
        alpha_step.get("ok")
        and alpha_payload.get("schema") == ALPHA_SCHEMA
        and alpha_payload.get("ok") is True
        and required_alpha_codes <= alpha_codes
    )
    developer_ready = bool(developer_step.get("ok") and developer_summary.get("ready"))
    private_cleaned = bool(
        alpha_summary.get("child_artifacts_pruned")
        and "kaggle_kernels_deleted" in alpha_codes
        and not args.keep_child_artifacts
        and not args.keep_live_private_artifacts
    )
    ready = bool(developer_ready and alpha_ready and private_cleaned)
    codes = set(alpha_codes) | set(diagnosis_codes(developer_payload)) | set(gpu_summary.get("diagnosis_codes") or [])
    if ready:
        codes.update({
            "public_swarm_live_preview_rc_ready",
            "public_swarm_live_preview_live_kaggle_ready",
            "developer_preview_ready",
            "public_swarm_developer_preview_ready",
            "support_bundle_ready",
            "private_artifacts_cleaned",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        })
    else:
        codes.add("public_swarm_live_preview_live_kaggle_blocked")
    if args.failure_mode != FAILURE_NONE and expected_requeue_code in alpha_codes:
        codes.add(expected_requeue_code)
    if gpu_summary.get("ok"):
        codes.add("gpu_generation_evidence_import_ready")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_LIVE_KAGGLE,
        "target": args.target,
        "output_dir": str(output_dir),
        "public_host": args.public_host,
        "port": args.port,
        "base_port": args.base_port,
        "failure_mode": args.failure_mode,
        "live_preview": {
            "ready": ready,
            "developer_preview_ready": developer_ready,
            "public_swarm_alpha_ready": alpha_ready,
            "external_runtime_verified": "external_runtime_verified" in alpha_codes,
            "fresh_live_kaggle_run": True,
            "stage_requeue_ready": args.failure_mode == FAILURE_NONE or "external_stage_requeue_ready" in alpha_codes,
            "kaggle_kernels_deleted": "kaggle_kernels_deleted" in alpha_codes,
            "private_artifacts_cleaned": private_cleaned,
            "token_rotation_required": "token_rotation_required" in alpha_codes,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
        },
        "steps": [developer_step, alpha_step],
        "payload_summaries": {
            "developer_preview": developer_summary,
            "public_swarm_alpha": alpha_summary,
            "gpu_generation_evidence": gpu_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": base_artifacts(output_dir, ok=ready),
        "safety": common_safety(mode=MODE_LIVE_KAGGLE, gpu_ready=bool(gpu_summary.get("ok")), external_verified="external_runtime_verified" in alpha_codes, private_cleaned=private_cleaned),
        "operator_action": operator_actions(MODE_LIVE_KAGGLE),
        "limitations": limitations(),
    }
    report["artifacts"]["public_swarm_alpha_json"] = artifact_entry(
        output_dir / "public-swarm-alpha-live" / "public_swarm_inference_alpha.json",
        output_dir,
        kind="public_swarm_inference_alpha",
        schema=ALPHA_SCHEMA,
        ok=alpha_payload.get("ok") if alpha_payload else None,
    )
    report["artifacts"]["gpu_generation_evidence_json"] = artifact_entry(
        Path(args.gpu_report),
        output_dir,
        kind="gpu_sharded_generation_beta",
        schema=GPU_GENERATION_SCHEMA,
        ok=bool(gpu_summary.get("ok")),
    )
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report)
    return report


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    developer_path = Path(args.developer_preview_report).resolve()
    alpha_rc_path = Path(args.alpha_rc_report).resolve()
    developer_payload = load_json(developer_path)
    alpha_rc_payload = load_json(alpha_rc_path)
    developer_summary = summarize_developer_preview(developer_payload)
    alpha_rc_summary = summarize_alpha_rc(alpha_rc_payload)
    gpu_summary = import_gpu_report(args.gpu_report)
    developer_ready = bool(developer_summary.get("ready"))
    alpha_rc_codes = set(diagnosis_codes(alpha_rc_payload))
    alpha_rc_ready = bool(
        alpha_rc_summary.get("ready")
        and "public_swarm_inference_alpha_rc_ready" in alpha_rc_codes
        and "stage0_live_requeue_evidence_ready" in alpha_rc_codes
        and "stage1_live_requeue_evidence_ready" in alpha_rc_codes
        and "public_swarm_live_requeue_evidence_ready" in alpha_rc_codes
    )
    ready = bool(developer_ready and alpha_rc_ready)
    codes = set(diagnosis_codes(developer_payload, alpha_rc_payload)) | set(gpu_summary.get("diagnosis_codes") or [])
    if ready:
        codes.update({
            "public_swarm_live_preview_rc_ready",
            "public_swarm_live_preview_evidence_import_ready",
            "developer_preview_ready",
            "public_swarm_developer_preview_ready",
            "public_swarm_live_kaggle_ready",
            "external_runtime_verified",
            "public_swarm_alpha_rc_evidence_imported",
            "support_bundle_ready",
            "private_artifacts_cleaned",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        })
    else:
        codes.add("public_swarm_live_preview_evidence_import_blocked")
    if gpu_summary.get("ok"):
        codes.add("gpu_generation_evidence_import_ready")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_EVIDENCE_IMPORT,
        "target": args.target,
        "output_dir": str(output_dir),
        "live_preview": {
            "ready": ready,
            "developer_preview_ready": developer_ready,
            "alpha_rc_ready": alpha_rc_ready,
            "external_runtime_verified": alpha_rc_ready,
            "fresh_live_kaggle_run": False,
            "stage0_live_requeue_ready": "stage0_live_requeue_evidence_ready" in alpha_rc_codes,
            "stage1_live_requeue_ready": "stage1_live_requeue_evidence_ready" in alpha_rc_codes,
            "private_artifacts_cleaned": "public_swarm_alpha_private_artifacts_absent" in alpha_rc_codes,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
        },
        "steps": [],
        "payload_summaries": {
            "developer_preview": developer_summary,
            "public_swarm_alpha_rc": alpha_rc_summary,
            "gpu_generation_evidence": gpu_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": base_artifacts(output_dir, ok=ready),
        "safety": common_safety(mode=MODE_EVIDENCE_IMPORT, gpu_ready=bool(gpu_summary.get("ok")), external_verified=alpha_rc_ready, private_cleaned="public_swarm_alpha_private_artifacts_absent" in alpha_rc_codes),
        "operator_action": operator_actions(MODE_EVIDENCE_IMPORT),
        "limitations": limitations(),
    }
    report["artifacts"]["developer_preview_json"] = artifact_entry(
        developer_path,
        output_dir,
        kind="public_swarm_developer_preview",
        schema=DEVELOPER_PREVIEW_SCHEMA,
        ok=developer_payload.get("ok") if developer_payload else None,
    )
    report["artifacts"]["public_swarm_alpha_rc_json"] = artifact_entry(
        alpha_rc_path,
        output_dir,
        kind="public_swarm_inference_alpha_rc",
        schema=ALPHA_RC_SCHEMA,
        ok=alpha_rc_payload.get("ok") if alpha_rc_payload else None,
    )
    report["artifacts"]["gpu_generation_evidence_json"] = artifact_entry(
        Path(args.gpu_report),
        output_dir,
        kind="gpu_sharded_generation_beta",
        schema=GPU_GENERATION_SCHEMA,
        ok=bool(gpu_summary.get("ok")),
    )
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report)
    return report


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Live Preview RC report contained secret-like fragments"
    json_path = output_dir / "public_swarm_live_preview_rc.json"
    md_path = output_dir / "public_swarm_live_preview_rc.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    if "public_swarm_live_preview_rc_json" in artifacts:
        artifacts["public_swarm_live_preview_rc_json"]["present"] = True
    if "public_swarm_live_preview_rc_markdown" in artifacts:
        artifacts["public_swarm_live_preview_rc_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    preview = report.get("live_preview") if isinstance(report.get("live_preview"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Live Preview RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- ready: `{preview.get('ready')}`",
        f"- external_runtime_verified: `{preview.get('external_runtime_verified')}`",
        f"- fresh_live_kaggle_run: `{preview.get('fresh_live_kaggle_run')}`",
        f"- workload_type: `{preview.get('workload_type')}`",
        f"- output_dir: `{report.get('output_dir')}`",
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
    parser = argparse.ArgumentParser(description="Build Public Swarm Live Preview RC evidence.")
    parser.add_argument("mode", choices=[MODE_LOCAL_SMOKE, MODE_PACKAGE, MODE_LIVE_KAGGLE, MODE_EVIDENCE_IMPORT])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--ready-url", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--target", choices=["local", "kaggle"], default="kaggle")
    parser.add_argument("--miner-id-prefix", default="public-swarm-live-preview")
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--developer-preview-report", default=DEFAULT_DEVELOPER_PREVIEW_REPORT)
    parser.add_argument("--alpha-rc-report", default=DEFAULT_ALPHA_RC_REPORT)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--cpu-request-count", type=int, default=1)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--failure-mode", choices=sorted(FAILURE_MODES), default=FAILURE_KILL_STAGE0_AFTER_CLAIM)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--dataset-slug", default="")
    parser.add_argument("--dataset-title", default="CrowdTensor Public Swarm Live Preview RC")
    parser.add_argument("--kernel-slug-prefix", default="ct-live-preview")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor Public Swarm Live Preview RC")
    parser.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-live-private-artifacts", action="store_true")
    parser.add_argument("--keep-child-artifacts", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
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
    if args.port < 1 or args.base_port < 1:
        raise SystemExit("--port and --base-port must be positive")
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
    if args.mode == MODE_LIVE_KAGGLE and args.failure_mode != FAILURE_NONE:
        if args.victim_compute_seconds <= args.lease_seconds:
            args.victim_compute_seconds = args.lease_seconds + 30.0
        if args.requeue_timeout <= args.lease_seconds:
            args.requeue_timeout = args.lease_seconds + 45.0
    return args


def print_human(report: dict[str, Any]) -> None:
    preview = report.get("live_preview") if isinstance(report.get("live_preview"), dict) else {}
    print("CrowdTensor Public Swarm Live Preview RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {preview.get('ready')}")
    print(f"  external_runtime_verified: {preview.get('external_runtime_verified')}")
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
