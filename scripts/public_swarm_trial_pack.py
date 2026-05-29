#!/usr/bin/env python3
"""Build the Public Swarm v0.2 usable inference trial artifact.

The trial is the ordinary-user entrypoint over the current product surface. It
does not add a new runtime protocol; it orchestrates existing Coordinator-backed
serve/join/generate, Operator Preview, CPU fallback, and optional GPU retained
or live evidence into one shareable report.
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

import public_swarm_operator_preview_pack as operator_pack  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID  # noqa: E402


SCHEMA = "public_swarm_trial_v1"
PRODUCT_BETA_SCHEMA = "public_swarm_product_beta_v1"
OPERATOR_PREVIEW_SCHEMA = "public_swarm_operator_preview_v1"
GPU_GENERATION_SCHEMA = "gpu_sharded_generation_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
MODE_LOCAL_LOOPBACK = "local-loopback"
MODE_PACKAGE = "package"
MODE_LIVE_KAGGLE = "live-kaggle"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODES = [MODE_LOCAL_LOOPBACK, MODE_PACKAGE, MODE_LIVE_KAGGLE, MODE_EVIDENCE_IMPORT]
DEFAULT_OUTPUT_DIR = "dist/public-swarm-trial"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9400
DEFAULT_BASE_PORT = 9401
DEFAULT_RELEASE_BASE_PORT = 9410
DEFAULT_GPU_REPORT = operator_pack.DEFAULT_GPU_REPORT
DEFAULT_PRODUCT_BETA_REPORT = "dist/public-swarm-product-beta/public_swarm_product_beta.json"
DEFAULT_OPERATOR_PREVIEW_REPORT = "dist/public-swarm-operator-preview/public_swarm_operator_preview.json"

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
    allow_failure_payload: bool = False,
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
        if not allow_failure_payload:
            step["ok"] = bool(step.get("ok") and payload.get("ok"))
    if not step.get("ok"):
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:], secret_values)
    return step, payload


def find_int_key(value: Any, key_name: str) -> int:
    pending: list[Any] = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if isinstance(item, dict):
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            if key_name in item:
                try:
                    return int(item.get(key_name) or 0)
                except (TypeError, ValueError):
                    return 0
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return 0


def summarize_product_beta(payload: dict[str, Any], *, max_new_tokens: int) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    beta = payload.get("product_beta") if isinstance(payload.get("product_beta"), dict) else {}
    ready = bool(payload.get("ok") is True and payload.get("schema") == PRODUCT_BETA_SCHEMA and "public_swarm_product_beta_ready" in codes)
    generated_count = find_int_key(payload, "generated_token_count")
    generated_ready = bool(generated_count >= max_new_tokens or "public_swarm_generate_ready" in codes)
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "ready": ready,
        "diagnosis_codes": sorted(codes),
        "serve_ready": "serve_ready" in codes,
        "stage0_join_ready": "stage0_join_ready" in codes,
        "stage1_join_ready": "stage1_join_ready" in codes,
        "generate_ready": "generate_ready" in codes or "public_swarm_generate_ready" in codes,
        "generated_token_count": generated_count,
        "generated_token_count_ready": generated_ready,
        "cpu_fallback_ready": "cpu_fallback_ready" in codes or beta.get("cpu_fallback_ready"),
        "hf_dependencies_missing": "hf_dependencies_missing" in codes,
        "package_ready": "miner_join_pack_ready" in codes and "private_artifacts_local_only" in codes,
        "workload_type": beta.get("workload_type", WORKLOAD_TYPE),
    }


def summarize_operator_preview(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    preview = payload.get("operator_preview") if isinstance(payload.get("operator_preview"), dict) else {}
    ready = bool(payload.get("ok") is True and payload.get("schema") == OPERATOR_PREVIEW_SCHEMA and "public_swarm_operator_preview_ready" in codes)
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "ready": ready,
        "diagnosis_codes": sorted(codes),
        "cpu_fallback_ready": "cpu_fallback_ready" in codes or preview.get("cpu_fallback_ready"),
        "external_runtime_verified": "external_runtime_verified" in codes or preview.get("external_runtime_verified"),
        "external_runtime_blocked": "external_runtime_blocked" in codes or preview.get("external_runtime_blocked"),
        "retained_evidence_ready": "operator_preview_retained_evidence_ready" in codes or preview.get("retained_evidence_ready"),
        "package_ready": "operator_preview_package_ready" in codes or preview.get("package_ready"),
        "degraded": "developer_preview_degraded" in codes or preview.get("developer_preview_degraded"),
    }


def summarize_gpu_generation(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    ready = bool(payload.get("ok") is True and payload.get("schema") == GPU_GENERATION_SCHEMA and "gpu_sharded_generation_ready" in codes)
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "ready": ready,
        "diagnosis_codes": sorted(codes),
        "generated_token_count": generation.get("generated_token_count"),
        "multi_token_generation_ready": "multi_token_generation_ready" in codes or generation.get("multi_token_generation_ready"),
        "external_gpu_runtime_verified": "external_gpu_runtime_verified" in codes,
    }


def product_beta_command(args: argparse.Namespace, mode: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_product_beta_pack.py"),
        mode,
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
        "--prompt-text",
        args.prompt_text,
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
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return command


def operator_preview_command(args: argparse.Namespace, mode: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_operator_preview_pack.py"),
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
        "--release-base-port",
        str(args.release_base_port),
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
        "--live-stage0-report",
        args.live_stage0_report,
        "--live-stage1-report",
        args.live_stage1_report,
        "--release-readiness-report",
        args.release_readiness_report,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(max(2, args.max_new_tokens)),
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
        "--release-timeout-seconds",
        str(args.release_timeout_seconds),
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
    if args.allow_dirty_release:
        command.append("--allow-dirty-release")
    return command


def gpu_generation_command(args: argparse.Namespace, mode: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "gpu_sharded_generation_beta_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--request-count",
        "1",
        "--max-new-tokens",
        str(max(2, args.max_new_tokens)),
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--public-host",
        args.public_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--bind-host",
        args.bind_host,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if mode == "kaggle-auto":
        command.extend([
            "--kaggle-owner",
            args.kaggle_owner,
            "--dataset-title",
            args.dataset_title,
            "--kernel-title-prefix",
            args.kernel_title_prefix,
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
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.dataset_slug:
            command.extend(["--dataset-slug", args.dataset_slug])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        command.append("--inline-kernel-payload" if args.inline_kernel_payload else "--no-inline-kernel-payload")
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    return command


def run_product_beta(args: argparse.Namespace, *, output_dir: Path, mode: str, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_json_step(
        "public_swarm_product_beta",
        product_beta_command(args, mode, output_dir),
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 540.0,
    )


def run_operator_preview(args: argparse.Namespace, *, output_dir: Path, mode: str, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_json_step(
        "public_swarm_operator_preview",
        operator_preview_command(args, mode, output_dir),
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), float(args.release_timeout_seconds), 60.0) + 1260.0,
        allow_failure_payload=mode == MODE_LIVE_KAGGLE,
    )


def run_gpu_generation(args: argparse.Namespace, *, output_dir: Path, mode: str, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_json_step(
        "gpu_sharded_generation_beta",
        gpu_generation_command(args, mode, output_dir),
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.kaggle_status_timeout_seconds), 60.0) + 720.0,
        allow_failure_payload=True,
    )


def write_trial_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "SWARM_TRIAL.md"
    lines = [
        "# CrowdTensor Public Swarm v0.2 Usable Inference Trial",
        "",
        "This runbook is the ordinary-user trial path for the current Coordinator-backed preview.",
        "It is read-only, CPU-first, and not production Swarm Inference.",
        "",
        "## Local Trial",
        "",
        "```bash",
        "python -m pip install -e '.[dev]'",
        "crowdtensor swarm-trial local-loopback --max-new-tokens 2 --json",
        "```",
        "",
        "## Product Commands",
        "",
        "```bash",
        f"crowdtensor serve --public-host {args.public_host} --port {args.port} --json",
        f"crowdtensor join --coordinator-url http://{args.public_host}:{args.port} --stage stage0 --json",
        f"crowdtensor join --coordinator-url http://{args.public_host}:{args.port} --stage stage1 --json",
        f"crowdtensor generate --coordinator-url http://{args.public_host}:{args.port} --prompt-text 'CrowdTensor trial' --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "## Optional Live Proof",
        "",
        "```bash",
        f"crowdtensor swarm-trial live-kaggle --public-host {args.public_host} --kaggle-owner YOUR_KAGGLE_USERNAME --json",
        "```",
        "",
        "Share only the top-level JSON/Markdown and support bundle. Do not publish private env files, raw prompts, generated text, token ids, activations, or temporary Kaggle kernels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="public_swarm_trial_runbook")


def support_bundle_artifact(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    bundle = support_bundle.sanitize(redact_values({
        "schema": "public_swarm_trial_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "trial": report.get("trial") or {},
        "payload_summaries": report.get("payload_summaries") or {},
        "artifacts": report.get("artifacts") or {},
        "safety": report.get("safety") or {},
        "limitations": report.get("limitations") or [],
    }))
    path = output_dir / "support_bundle.json"
    write_json(path, bundle)
    return artifact_entry(path, output_dir, kind="public_swarm_trial_support_bundle", schema=str(bundle.get("schema")), ok=bundle.get("ok"))


def build_common_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mode: str,
    steps: list[dict[str, Any]],
    product_payload: dict[str, Any],
    operator_payload: dict[str, Any],
    gpu_payload: dict[str, Any],
    extra_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    product_summary = summarize_product_beta(product_payload, max_new_tokens=max(2, args.max_new_tokens))
    operator_summary = summarize_operator_preview(operator_payload)
    gpu_summary = summarize_gpu_generation(gpu_payload)
    product_codes = set(product_summary.get("diagnosis_codes") or [])
    operator_codes = set(operator_summary.get("diagnosis_codes") or [])
    gpu_codes = set(gpu_summary.get("diagnosis_codes") or [])
    cpu_ready = bool(product_summary.get("cpu_fallback_ready") or operator_summary.get("cpu_fallback_ready") or "cpu_fallback_ready" in product_codes | operator_codes)
    real_trial_ready = bool(
        product_summary.get("ready")
        and product_summary.get("serve_ready")
        and product_summary.get("stage0_join_ready")
        and product_summary.get("stage1_join_ready")
        and product_summary.get("generate_ready")
        and product_summary.get("generated_token_count_ready")
    )
    package_ready = bool(
        mode == MODE_PACKAGE
        and (
            product_summary.get("package_ready")
            or operator_summary.get("package_ready")
            or {"miner_join_pack_ready", "private_artifacts_local_only"} <= (product_codes | operator_codes)
        )
    )
    evidence_ready = bool(mode == MODE_EVIDENCE_IMPORT and operator_summary.get("ready"))
    live_ready = bool(mode == MODE_LIVE_KAGGLE and operator_summary.get("ready"))
    degraded_ready = bool(
        mode == MODE_LOCAL_LOOPBACK
        and not real_trial_ready
        and cpu_ready
        and (
            operator_summary.get("ready")
            or "operator_preview_cpu_fallback_user_path_ready" in operator_codes
            or "hf_dependencies_missing" in product_codes
        )
    )
    user_path_ready = bool(real_trial_ready or package_ready or evidence_ready or live_ready or degraded_ready)
    support_ready = bool(user_path_ready and (operator_summary.get("ready") or product_summary.get("ready") or cpu_ready))
    ready = bool(user_path_ready and support_ready)
    private_artifacts_cleaned = bool(
        "private_artifacts_cleaned" in (product_codes | operator_codes | gpu_codes)
        or mode in {MODE_LOCAL_LOOPBACK, MODE_EVIDENCE_IMPORT}
    )
    external_runtime_blocked = bool(operator_summary.get("external_runtime_blocked") or "external_runtime_blocked" in (operator_codes | gpu_codes))
    external_runtime_verified = bool(operator_summary.get("external_runtime_verified") or gpu_summary.get("external_gpu_runtime_verified"))
    token_rotation_required = bool("token_rotation_required" in (operator_codes | gpu_codes))
    codes = set(diagnosis_codes(product_payload, operator_payload, gpu_payload))
    if ready:
        codes.update({
            "public_swarm_trial_ready",
            "public_swarm_trial_user_path_ready",
            "support_bundle_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_libp2p",
            "not_dht",
            "not_nat_traversal",
            "not_gpu_pooling_marketplace",
            "not_large_model_serving",
        })
    else:
        codes.add("public_swarm_trial_blocked")
    if real_trial_ready:
        codes.update({
            "serve_join_generate_trial_ready",
            "serve_ready",
            "stage0_join_ready",
            "stage1_join_ready",
            "generate_ready",
            "generated_token_count_ready",
        })
    if mode == MODE_LOCAL_LOOPBACK and user_path_ready:
        codes.add("swarm_trial_local_loopback_ready")
    if package_ready:
        codes.update({"public_swarm_trial_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"})
        if "stage0_join_pack_ready" in product_codes or "stage0_join_pack_ready" in operator_codes:
            codes.add("stage0_join_pack_ready")
        if "stage1_join_pack_ready" in product_codes or "stage1_join_pack_ready" in operator_codes:
            codes.add("stage1_join_pack_ready")
    if evidence_ready:
        codes.add("public_swarm_trial_evidence_import_ready")
    if live_ready:
        codes.add("public_swarm_trial_live_kaggle_ready")
    if degraded_ready:
        codes.add("swarm_trial_degraded_cpu_fallback_ready")
    if cpu_ready:
        codes.add("cpu_fallback_ready")
    if gpu_summary.get("ready"):
        codes.add("gpu_generation_evidence_import_ready")
    if operator_summary.get("ready"):
        codes.add("operator_preview_import_ready")
    if external_runtime_verified:
        codes.add("external_runtime_verified")
    if external_runtime_blocked:
        codes.add("external_runtime_blocked")
    if "kaggle_kernels_deleted" in operator_codes | gpu_codes:
        codes.add("kaggle_kernels_deleted")
    if private_artifacts_cleaned:
        codes.add("private_artifacts_cleaned")
    if token_rotation_required:
        codes.add("token_rotation_required")
    artifacts = {
        "public_swarm_trial_json": artifact_entry(output_dir / "public_swarm_trial.json", output_dir, kind="public_swarm_trial", schema=SCHEMA, ok=ready),
        "public_swarm_trial_markdown": artifact_entry(output_dir / "public_swarm_trial.md", output_dir, kind="public_swarm_trial_markdown"),
        "product_beta_json": artifact_entry(output_dir / "product-beta" / "public_swarm_product_beta.json", output_dir, kind="public_swarm_product_beta", schema=PRODUCT_BETA_SCHEMA, ok=product_payload.get("ok") if product_payload else None),
        "operator_preview_json": artifact_entry(output_dir / "operator-preview" / "public_swarm_operator_preview.json", output_dir, kind="public_swarm_operator_preview", schema=OPERATOR_PREVIEW_SCHEMA, ok=operator_payload.get("ok") if operator_payload else None),
        "gpu_generation_json": artifact_entry(output_dir / "gpu-generation" / "gpu_sharded_generation_beta_evidence_import.json", output_dir, kind="gpu_sharded_generation_beta", schema=GPU_GENERATION_SCHEMA, ok=gpu_payload.get("ok") if gpu_payload else None),
        "gpu_generation_evidence_json": artifact_entry(Path(args.gpu_report), output_dir, kind="gpu_sharded_generation_beta_retained", schema=GPU_GENERATION_SCHEMA, ok=gpu_payload.get("ok") if gpu_payload else None),
    }
    if extra_artifacts:
        artifacts.update(extra_artifacts)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": mode,
        "backend": args.backend,
        "target": args.target,
        "output_dir": str(output_dir),
        "trial": {
            "ready": ready,
            "real_serve_join_generate_ready": real_trial_ready,
            "serve_join_generate_trial_ready": real_trial_ready,
            "stage0_join_ready": bool(product_summary.get("stage0_join_ready")),
            "stage1_join_ready": bool(product_summary.get("stage1_join_ready")),
            "generate_ready": bool(product_summary.get("generate_ready")),
            "generated_token_count_ready": bool(product_summary.get("generated_token_count_ready") or gpu_summary.get("multi_token_generation_ready")),
            "generated_token_count": product_summary.get("generated_token_count") or gpu_summary.get("generated_token_count") or 0,
            "degraded_cpu_fallback_ready": degraded_ready,
            "package_ready": package_ready,
            "evidence_import_ready": evidence_ready,
            "live_kaggle_ready": live_ready,
            "user_path_ready": user_path_ready,
            "support_bundle_ready": support_ready,
            "cpu_fallback_ready": cpu_ready,
            "gpu_generation_ready": bool(gpu_summary.get("ready")),
            "operator_preview_ready": bool(operator_summary.get("ready")),
            "external_runtime_verified": external_runtime_verified,
            "external_runtime_blocked": external_runtime_blocked,
            "private_artifacts_cleaned": private_artifacts_cleaned,
            "token_rotation_required": token_rotation_required,
            "max_new_tokens": args.max_new_tokens,
            "hf_model_id": args.hf_model_id,
            "workload_type": WORKLOAD_TYPE,
            "user_surface": ["swarm-trial", "serve", "join", "generate", "support bundle"],
        },
        "steps": steps,
        "payload_summaries": {
            "product_beta": product_summary,
            "operator_preview": operator_summary,
            "gpu_generation": gpu_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": artifacts,
        "safety": {
            "coordinator_backed_task_execution": True,
            "serve_join_generate_trial": real_trial_ready,
            "p2p_lite_discovery_only": True,
            "cpu_first_default": args.backend == "cpu",
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
            "not_training": True,
        },
        "operator_action": operator_actions(mode, degraded=degraded_ready),
        "limitations": limitations(),
    }
    return report


def operator_actions(mode: str, *, degraded: bool) -> list[str]:
    actions = [
        "Use crowdtensor swarm-trial as the top-level ordinary-user trial command.",
        "Share only top-level JSON/Markdown and support bundle artifacts.",
        "Do not publish private env files, temporary Kaggle kernels, raw prompts, generated text, token ids, activations, or runtime state.",
    ]
    if degraded:
        actions.append("Install optional HF dependencies with python -m pip install -e '.[hf]' to run the real tiny GPT serve/join/generate loop.")
    if mode == MODE_LIVE_KAGGLE:
        actions.append("Rotate generated Coordinator and Miner tokens after temporary public HTTP/Kaggle runs.")
    if mode == MODE_PACKAGE:
        actions.append("Use SWARM_TRIAL.md to hand the trial to another operator or host.")
    return actions


def limitations() -> list[str]:
    return [
        "Public Swarm v0.2 Usable Inference Trial is Coordinator-backed and read-only; it is not production Swarm Inference.",
        "The real route is tiny GPT serve/join/generate evidence with CPU fallback; it is not Hivemind/Petals-level serving.",
        "P2P-lite remains metadata discovery only; this is not libp2p, DHT, NAT traversal, decentralized security, or P2P execution.",
        "Optional CUDA/Kaggle evidence is a bounded proof, not a GPU pooling marketplace or large-model serving path.",
    ]


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report)
    report = redact_values(report)
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Trial report contained secret-like fragments"
    json_path = output_dir / "public_swarm_trial.json"
    md_path = output_dir / "public_swarm_trial.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_trial_json"]["present"] = True
    report["artifacts"]["public_swarm_trial_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    trial = report.get("trial") if isinstance(report.get("trial"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm v0.2 Usable Inference Trial",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- backend: `{report.get('backend')}`",
        f"- ready: `{trial.get('ready')}`",
        f"- real_serve_join_generate_ready: `{trial.get('real_serve_join_generate_ready')}`",
        f"- degraded_cpu_fallback_ready: `{trial.get('degraded_cpu_fallback_ready')}`",
        f"- cpu_fallback_ready: `{trial.get('cpu_fallback_ready')}`",
        f"- output_dir: `{report.get('output_dir')}`",
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


def build_local_loopback(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_beta(args, output_dir=output_dir / "product-beta", mode="local-loopback", runner=runner)
    operator_step, operator_payload = run_operator_preview(args, output_dir=output_dir / "operator-preview", mode="local-smoke", runner=runner)
    gpu_step, gpu_payload = run_gpu_generation(args, output_dir=output_dir / "gpu-generation", mode="evidence-import", runner=runner)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_LOCAL_LOOPBACK,
        steps=[product_step, operator_step, gpu_step],
        product_payload=product_payload,
        operator_payload=operator_payload,
        gpu_payload=gpu_payload,
    )


def build_package(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_beta(args, output_dir=output_dir / "product-beta", mode="package", runner=runner)
    operator_step, operator_payload = run_operator_preview(args, output_dir=output_dir / "operator-preview", mode="package", runner=runner)
    gpu_step, gpu_payload = run_gpu_generation(args, output_dir=output_dir / "gpu-generation", mode="evidence-import", runner=runner)
    runbook = write_trial_runbook(args, output_dir)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_PACKAGE,
        steps=[product_step, operator_step, gpu_step],
        product_payload=product_payload,
        operator_payload=operator_payload,
        gpu_payload=gpu_payload,
        extra_artifacts={"swarm_trial_runbook": runbook},
    )


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    del runner
    product_payload = load_json(args.product_beta_report)
    operator_payload = load_json(args.operator_preview_report)
    if not operator_payload:
        operator_payload = operator_pack.build_report(operator_pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "operator-preview"),
            "--gpu-report",
            args.gpu_report,
            "--live-stage0-report",
            args.live_stage0_report,
            "--live-stage1-report",
            args.live_stage1_report,
            "--release-readiness-report",
            args.release_readiness_report,
            "--json",
        ]))
    gpu_payload = load_json(args.gpu_report)
    if gpu_payload.get("schema") != GPU_GENERATION_SCHEMA:
        gpu_payload = {}
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_EVIDENCE_IMPORT,
        steps=[],
        product_payload=product_payload,
        operator_payload=operator_payload,
        gpu_payload=gpu_payload,
    )


def build_live_kaggle(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    operator_step, operator_payload = run_operator_preview(args, output_dir=output_dir / "operator-preview", mode="live-kaggle", runner=runner)
    if args.backend == "cuda":
        gpu_step, gpu_payload = run_gpu_generation(args, output_dir=output_dir / "gpu-generation", mode="kaggle-auto", runner=runner)
    else:
        gpu_step, gpu_payload = run_gpu_generation(args, output_dir=output_dir / "gpu-generation", mode="evidence-import", runner=runner)
    product_payload = load_json(args.product_beta_report)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_LIVE_KAGGLE,
        steps=[operator_step, gpu_step],
        product_payload=product_payload,
        operator_payload=operator_payload,
        gpu_payload=gpu_payload,
    )


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_LOCAL_LOOPBACK:
        report = build_local_loopback(args, output_dir=output_dir, runner=runner)
    elif args.mode == MODE_PACKAGE:
        report = build_package(args, output_dir=output_dir, runner=runner)
    elif args.mode == MODE_LIVE_KAGGLE:
        report = build_live_kaggle(args, output_dir=output_dir, runner=runner)
    else:
        report = build_evidence_import(args, output_dir=output_dir, runner=runner)
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
    parser = argparse.ArgumentParser(description="Build Public Swarm v0.2 usable inference trial evidence.")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--release-base-port", type=int, default=DEFAULT_RELEASE_BASE_PORT)
    parser.add_argument("--ready-url", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--target", choices=["local", "kaggle"], default="kaggle")
    parser.add_argument("--miner-id-prefix", default="public-swarm-trial")
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--product-beta-report", default=DEFAULT_PRODUCT_BETA_REPORT)
    parser.add_argument("--operator-preview-report", default=DEFAULT_OPERATOR_PREVIEW_REPORT)
    parser.add_argument("--developer-preview-report", default=operator_pack.DEFAULT_DEVELOPER_PREVIEW_REPORT)
    parser.add_argument("--alpha-rc-report", default=operator_pack.live_preview_pack.DEFAULT_ALPHA_RC_REPORT)
    parser.add_argument("--live-stage0-report", default=operator_pack.DEFAULT_LIVE_STAGE0_REPORT)
    parser.add_argument("--live-stage1-report", default=operator_pack.DEFAULT_LIVE_STAGE1_REPORT)
    parser.add_argument("--release-readiness-report", default=operator_pack.DEFAULT_RELEASE_READINESS_REPORT)
    parser.add_argument("--prompt-text", default="CrowdTensor swarm trial")
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--cpu-request-count", type=int, default=1)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="kill-stage0-after-claim")
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--dataset-slug", default="")
    parser.add_argument("--dataset-title", default="CrowdTensor Public Swarm v0.2 Trial")
    parser.add_argument("--kernel-slug-prefix", default="ct-swarm-trial")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor Swarm Trial")
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
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.cpu_request_count < 1 or args.cpu_request_count > 4:
        raise SystemExit("--cpu-request-count must be between 1 and 4")
    if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
        raise SystemExit("--external-llm-request-count must be between 1 and 4")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.base_port < 1 or args.port < 1 or args.release_base_port < 1:
        raise SystemExit("--base-port, --port, and --release-base-port must be positive")
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
    if args.mode == MODE_LIVE_KAGGLE and args.failure_mode != "none":
        if args.victim_compute_seconds <= args.lease_seconds:
            args.victim_compute_seconds = args.lease_seconds + 30.0
        if args.requeue_timeout <= args.lease_seconds:
            args.requeue_timeout = args.lease_seconds + 45.0
    return args


def print_human(report: dict[str, Any]) -> None:
    trial = report.get("trial") if isinstance(report.get("trial"), dict) else {}
    print("CrowdTensor Public Swarm v0.2 Usable Inference Trial")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {trial.get('ready')}")
    print(f"  real serve/join/generate: {trial.get('real_serve_join_generate_ready')}")
    print(f"  degraded CPU fallback: {trial.get('degraded_cpu_fallback_ready')}")
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
