#!/usr/bin/env python3
"""Build the Public Swarm Developer Preview artifact.

This layer is the user-facing preview over the current Coordinator-backed
product surface. It aggregates Product Beta, CPU fallback, optional retained GPU
generation evidence, external packaging, diagnostics, and safety boundaries.
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

import public_swarm_product_beta_pack as product_beta_pack  # noqa: E402
import support_bundle  # noqa: E402


SCHEMA = "public_swarm_developer_preview_v1"
PRODUCT_BETA_SCHEMA = "public_swarm_product_beta_v1"
GPU_GENERATION_SCHEMA = "gpu_sharded_generation_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-developer-preview"
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
    "sharded_inference_result",
    "inference_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
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
        step["ok"] = bool(step.get("ok") and payload.get("ok"))
    if not step.get("ok"):
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:], secret_values)
    return step, payload


def product_mode_for_preview(mode: str) -> str:
    if mode == "local":
        return "local-loopback"
    if mode in {"package", "external-existing"}:
        return mode
    raise ValueError(f"mode {mode} does not map to Product Beta")


def run_product_beta(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_product_beta_pack.py"),
        product_mode_for_preview(args.mode),
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
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    return run_json_step(
        "public_swarm_product_beta",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 420.0,
        secret_values=[args.observer_token, args.admin_token],
    )


def import_gpu_report(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(Path(args.gpu_report))
    if not payload:
        return {
            "ok": False,
            "present": False,
            "schema": "",
            "path": args.gpu_report,
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
        "path": args.gpu_report,
        "diagnosis_codes": sorted(codes),
        "generated_token_count": generated_count,
        "generated_text_hash": generated_hash,
        "raw_generated_text_public": raw_public,
    }


def summarize_product_beta(payload: dict[str, Any]) -> dict[str, Any]:
    product = payload.get("product_beta") if isinstance(payload.get("product_beta"), dict) else {}
    split = (payload.get("payload_summaries") or {}).get("real_llm_split_validation")
    if not isinstance(split, dict):
        split = {}
    assignment = split.get("stage_assignment") if isinstance(split.get("stage_assignment"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
        "ready": product.get("ready"),
        "mode_ready": product.get("mode_ready"),
        "support_bundle_ready": product.get("support_bundle_ready"),
        "privacy_ready": product.get("privacy_ready"),
        "workload_type": product.get("workload_type"),
        "hf_model_id": product.get("hf_model_id"),
        "max_new_tokens": product.get("max_new_tokens"),
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
            "Public Swarm Developer Preview artifacts summarize preview readiness "
            "with counts, hashes, route evidence, and support diagnostics only. "
            "Run `crowdtensor generate` in human mode to see a local answer."
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
            "This Developer Preview report is shareable preview evidence, not a local "
            "answer transcript; raw prompts, generated text, generated token ids, "
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
        "summary": "Share public_swarm_developer_preview.json/md and support_bundle.json; they contain hashes/counts and readiness evidence, not raw prompts or answers.",
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


def support_bundle_artifact(output_dir: Path, report: dict[str, Any], *, secret_values: list[str] | None = None) -> dict[str, Any]:
    bundle = support_bundle.sanitize(redact_values({
        "schema": "public_swarm_developer_preview_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": bool(report.get("ok")),
        "preview": {
            "schema": report.get("schema"),
            "mode": report.get("mode"),
            "ok": report.get("ok"),
            "diagnosis_codes": report.get("diagnosis_codes") or [],
        },
        "artifacts": report.get("artifacts") or {},
        "output_request": report.get("output_request"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety") or {},
        "limitations": report.get("limitations") or [],
    }, secret_values))
    path = output_dir / "support_bundle.json"
    write_json(path, bundle)
    return artifact_entry(path, output_dir, kind="public_swarm_developer_preview_support_bundle", schema=str(bundle.get("schema")), ok=bundle.get("ok"))


def build_runtime_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    secret_values = [args.observer_token, args.admin_token]
    product_dir = output_dir / "product-beta"
    step, product_payload = run_product_beta(args, output_dir=product_dir, runner=runner)
    product_codes = set(diagnosis_codes(product_payload))
    gpu_summary = import_gpu_report(args)
    gpu_ready = bool(gpu_summary.get("ok"))
    inherited_codes = set(product_codes)
    ready_codes: set[str] = set()
    mode_ready = False
    product_ready = bool(
        step.get("ok")
        and product_payload.get("schema") == PRODUCT_BETA_SCHEMA
        and product_payload.get("ok") is True
        and "public_swarm_product_beta_ready" in product_codes
    )
    if args.mode == "local":
        local_required = {
            "public_swarm_product_beta_user_path_ready",
            "serve_ready",
            "stage0_join_ready",
            "stage1_join_ready",
            "generate_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "private_artifacts_cleaned",
        }
        mode_ready = bool(product_ready and local_required <= product_codes)
        if mode_ready:
            ready_codes.update({
                "local_two_stage_generation_ready",
                "serve_join_generate_ready",
                "developer_preview_local_ready",
            })
    elif args.mode == "package":
        package_required = {"public_swarm_product_beta_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"}
        mode_ready = bool(product_ready and package_required <= product_codes)
        if mode_ready:
            ready_codes.update({"developer_preview_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"})
    else:
        external_required = {"external_runtime_verified", "remote_real_llm_sharded_existing_ready"}
        mode_ready = bool(product_ready and external_required <= product_codes)
        if mode_ready:
            ready_codes.update({"developer_preview_external_ready", "external_runtime_verified"})
    if gpu_ready:
        ready_codes.add("gpu_generation_evidence_import_ready")
    else:
        inherited_codes.add("gpu_generation_evidence_missing")
    if not product_ready:
        inherited_codes.discard("public_swarm_product_beta_ready")
        inherited_codes.discard("public_swarm_inference_beta_ready")
    ready = bool(mode_ready)
    common_codes = {
        "developer_preview_ready",
        "public_swarm_developer_preview_ready",
        "product_beta_ready",
        "support_bundle_ready",
        "cpu_fallback_ready",
        "local_cpu_inference_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    }
    codes = inherited_codes | ready_codes
    if ready:
        codes.update(common_codes)
    else:
        codes.add("developer_preview_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": args.mode,
        "target": args.target,
        "output_dir": str(output_dir),
        "developer_preview": {
            "ready": ready,
            "mode_ready": mode_ready,
            "product_beta_ready": product_ready,
            "gpu_generation_evidence_ready": gpu_ready,
            "cpu_fallback_ready": "cpu_fallback_ready" in product_codes,
            "support_bundle_ready": "support_bundle_ready" in product_codes,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "user_surface": ["preview", "serve", "join", "generate", "collect diagnostics"],
        },
        "steps": [step],
        "payload_summaries": {
            "public_swarm_product_beta": summarize_product_beta(product_payload),
            "gpu_generation_evidence": gpu_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "public_swarm_product_beta_json": artifact_entry(
                product_dir / "public_swarm_product_beta.json",
                output_dir,
                kind="public_swarm_product_beta",
                schema=PRODUCT_BETA_SCHEMA,
                ok=product_payload.get("ok") if product_payload else None,
            ),
            "gpu_generation_evidence_json": artifact_entry(
                Path(args.gpu_report),
                output_dir,
                kind="gpu_sharded_generation_beta",
                schema=GPU_GENERATION_SCHEMA,
                ok=gpu_ready,
            ),
        },
        "safety": safety_summary(gpu_ready=gpu_ready),
        "operator_action": [
            "Run crowdtensor preview local for a one-command localhost Developer Preview.",
            "Run crowdtensor preview package for two-machine or Kaggle stage join material.",
            "Run crowdtensor preview external-existing only after a controlled public Coordinator and external stage Miners are already running.",
            "Share only the generated JSON/Markdown and support bundle; keep private env files local and rotate temporary public tokens after live demos.",
        ],
        "limitations": limitations(),
    }
    report["output_request"] = output_request_summary()
    report["answer_scope"] = answer_scope_summary()
    report["shareable_summary"] = shareable_summary()
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report, secret_values=secret_values)
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def build_evidence_import(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    product_payload = load_json(Path(args.product_beta_report))
    product_codes = set(diagnosis_codes(product_payload))
    product_ready = bool(
        product_payload.get("schema") == PRODUCT_BETA_SCHEMA
        and product_payload.get("ok") is True
        and "public_swarm_product_beta_ready" in product_codes
    )
    gpu_summary = import_gpu_report(args)
    ready = product_ready
    codes = set(product_codes) | set(gpu_summary.get("diagnosis_codes") or [])
    if product_ready:
        codes.update({
            "developer_preview_ready",
            "public_swarm_developer_preview_ready",
            "developer_preview_evidence_import_ready",
            "product_beta_ready",
            "support_bundle_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        })
    else:
        codes.add("developer_preview_blocked")
    if gpu_summary.get("ok"):
        codes.add("gpu_generation_evidence_import_ready")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": "evidence-import",
        "target": args.target,
        "output_dir": str(output_dir),
        "developer_preview": {
            "ready": ready,
            "mode_ready": ready,
            "product_beta_ready": product_ready,
            "gpu_generation_evidence_ready": bool(gpu_summary.get("ok")),
            "workload_type": WORKLOAD_TYPE,
            "user_surface": ["preview", "serve", "join", "generate", "collect diagnostics"],
        },
        "steps": [],
        "payload_summaries": {
            "public_swarm_product_beta": summarize_product_beta(product_payload),
            "gpu_generation_evidence": gpu_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "public_swarm_product_beta_json": artifact_entry(
                Path(args.product_beta_report),
                output_dir,
                kind="public_swarm_product_beta",
                schema=PRODUCT_BETA_SCHEMA,
                ok=product_payload.get("ok") if product_payload else None,
            ),
            "gpu_generation_evidence_json": artifact_entry(
                Path(args.gpu_report),
                output_dir,
                kind="gpu_sharded_generation_beta",
                schema=GPU_GENERATION_SCHEMA,
                ok=bool(gpu_summary.get("ok")),
            ),
        },
        "safety": safety_summary(gpu_ready=bool(gpu_summary.get("ok"))),
        "operator_action": [
            "Use evidence-import only for retained, already redacted Product Beta and GPU generation reports.",
            "Run crowdtensor preview local for a fresh localhost proof when optional runtime dependencies are available.",
        ],
        "limitations": limitations(),
    }
    report["output_request"] = output_request_summary()
    report["answer_scope"] = answer_scope_summary()
    report["shareable_summary"] = shareable_summary()
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report)
    return persist_report(report, output_dir=output_dir)


def safety_summary(*, gpu_ready: bool) -> dict[str, Any]:
    return {
        "coordinator_backed_task_execution": True,
        "developer_preview_product_surface": True,
        "p2p_lite_discovery_only": True,
        "gpu_generation_evidence_imported": bool(gpu_ready),
        "tokens_public": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_payloads_redacted": True,
        "read_only_workload": WORKLOAD_TYPE,
        "not_production": True,
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
        "Public Swarm Developer Preview is Coordinator-backed and read-only; it is not production Swarm Inference.",
        "P2P-lite is discovery/routing metadata only; not libp2p, DHT, NAT traversal, decentralized security, or P2P execution.",
        "The default runtime path is tiny GPT split generation plus CPU fallback; not Hivemind/Petals-level serving or large-model public prompt serving.",
        "GPU generation is optional retained evidence import unless an explicit CUDA-capable live path is run; not a GPU pooling marketplace.",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    preview = report.get("developer_preview") if isinstance(report.get("developer_preview"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Developer Preview",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        f"- ready: `{preview.get('ready')}`",
        f"- workload_type: `{preview.get('workload_type')}`",
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
    ]
    if report.get("steps"):
        lines.extend(["## Steps", ""])
        for step in report.get("steps") or []:
            lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
        lines.append("")
    lines.extend(["## Artifacts", ""])
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def persist_report(report: dict[str, Any], *, output_dir: Path, secret_values: list[str] | None = None) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Developer Preview report contained secret-like fragments"
    json_path = output_dir / "public_swarm_developer_preview.json"
    markdown_path = output_dir / "public_swarm_developer_preview.md"
    report.setdefault("artifacts", {})
    report["artifacts"]["public_swarm_developer_preview_json"] = artifact_entry(
        json_path,
        output_dir,
        kind="public_swarm_developer_preview",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["public_swarm_developer_preview_markdown"] = artifact_entry(
        markdown_path,
        output_dir,
        kind="public_swarm_developer_preview_markdown",
    )
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_developer_preview_json"]["present"] = True
    report["artifacts"]["public_swarm_developer_preview_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Developer Preview evidence.")
    parser.add_argument("mode", choices=["local", "package", "external-existing", "evidence-import"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-port", type=int, default=9330)
    parser.add_argument("--port", type=int, default=9330)
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--target", choices=["local", "kaggle"], default="local")
    parser.add_argument("--miner-id-prefix", default="public-swarm-preview")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--product-beta-report", default="dist/public-swarm-product-beta/public_swarm_product_beta.json")
    parser.add_argument("--prompt-text", default="CrowdTensor developer preview")
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


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    if args.mode == "evidence-import":
        return build_evidence_import(args)
    return build_runtime_report(args, runner=runner)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
        answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
        shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
        print(f"Public Swarm Developer Preview ready: {report.get('ok')}")
        if output_request:
            print(f"  output_request: {output_request_text(output_request)}")
        if answer_scope:
            print(f"  answer_scope: {answer_scope_text(answer_scope)}")
        if shareable:
            print(f"  shareable: {shareable_summary_text(shareable)}")
        print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
