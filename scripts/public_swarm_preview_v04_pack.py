#!/usr/bin/env python3
"""Build the Public Swarm Inference Preview v0.4 artifact.

This is a release-preview aggregate over the current Coordinator-backed product
surface. It does not introduce P2P execution. It makes the external stage0 /
stage1 evidence, multi-token generation, stage latency, throughput, and memory
or VRAM summaries explicit in one redacted report.
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
import public_swarm_trial_pack as trial_pack  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID  # noqa: E402


SCHEMA = "public_swarm_preview_v04_v1"
PRODUCT_MVP_SCHEMA = "product_swarm_mvp_check_v1"
PRODUCT_BETA_SCHEMA = "public_swarm_product_beta_v1"
LIVE_PREVIEW_SCHEMA = "public_swarm_live_preview_rc_v1"
GPU_GENERATION_SCHEMA = "gpu_sharded_generation_beta_v1"
REMOTE_REAL_SCHEMA = "remote_real_llm_sharded_beta_v1"
REAL_LLM_EVIDENCE_SCHEMA = "real_llm_sharded_evidence_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
MODE_LOCAL_SMOKE = "local-smoke"
MODE_PACKAGE = "package"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODES = [MODE_LOCAL_SMOKE, MODE_PACKAGE, MODE_EVIDENCE_IMPORT]
DEFAULT_OUTPUT_DIR = "dist/public-swarm-preview-v04"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9440
DEFAULT_BASE_PORT = 9441
DEFAULT_GPU_REPORT = trial_pack.DEFAULT_GPU_REPORT
DEFAULT_LIVE_STAGE0_REPORT = operator_pack.DEFAULT_LIVE_STAGE0_REPORT
DEFAULT_LIVE_STAGE1_REPORT = operator_pack.DEFAULT_LIVE_STAGE1_REPORT
DEFAULT_PRODUCT_MVP_REPORT = "dist/product-swarm-mvp/product_swarm_mvp_check.json"
DEFAULT_PRODUCT_BETA_REPORT = trial_pack.DEFAULT_PRODUCT_BETA_REPORT
DEFAULT_OPTIONAL_MODEL_ID = "distilgpt2"

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


def nested_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    pending: list[Any] = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if isinstance(item, dict):
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            found.append(item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return found


def find_first_dict(value: Any, key: str) -> dict[str, Any]:
    for item in nested_dicts(value):
        nested = item.get(key)
        if isinstance(nested, dict):
            return nested
    return {}


def find_first_int(value: Any, key: str) -> int:
    for item in nested_dicts(value):
        if key in item:
            try:
                return int(item.get(key) or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def find_stage_summary(value: Any) -> dict[str, Any]:
    for item in nested_dicts(value):
        stage_summary = item.get("stage_summary")
        if isinstance(stage_summary, dict) and ("stage_0" in stage_summary or "stage_1" in stage_summary):
            return stage_summary
    return {}


def summarize_product_mvp(payload: dict[str, Any], *, required_tokens: int) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    performance = payload.get("performance") if isinstance(payload.get("performance"), dict) else {}
    resources = payload.get("runtime_resources") if isinstance(payload.get("runtime_resources"), dict) else {}
    generated_count = int(generation.get("generated_token_count") or find_first_int(payload, "generated_token_count"))
    ready = bool(payload.get("schema") == PRODUCT_MVP_SCHEMA and payload.get("ok") is True and "product_swarm_mvp_ready" in codes)
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "mode": payload.get("mode"),
        "hf_model_id": payload.get("hf_model_id"),
        "diagnosis_codes": sorted(codes),
        "generated_token_count": generated_count,
        "multi_token_generation_ready": bool(generated_count >= required_tokens),
        "stage_latency_ready": "stage_latency_ready" in codes or bool(performance.get("stage_latency_ready")),
        "throughput_summary_ready": "throughput_summary_ready" in codes or bool(performance.get("throughput_summary_ready")),
        "memory_or_vram_summary_ready": "memory_or_vram_summary_ready" in codes or bool(resources.get("memory_or_vram_summary_ready")),
        "performance": {
            "stage_total_seconds": performance.get("stage_total_seconds"),
            "generated_tokens_per_stage_second": performance.get("generated_tokens_per_stage_second"),
            "per_stage": performance.get("per_stage") if isinstance(performance.get("per_stage"), dict) else {},
        },
        "runtime_resources": resources,
    }


def summarize_optional_model(payload: dict[str, Any], *, optional_model_id: str, required_tokens: int) -> dict[str, Any]:
    summary = summarize_product_mvp(payload, required_tokens=required_tokens)
    strict_ready = bool(summary.get("ready") and summary.get("hf_model_id") in {optional_model_id, "distilgpt2", "gpt2"})
    summary["optional_model_id"] = optional_model_id
    summary["optional_model_ready"] = strict_ready
    return summary


def summarize_live_report(payload: dict[str, Any], *, expected_requeue_code: str) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    live = payload.get("live_preview") if isinstance(payload.get("live_preview"), dict) else {}
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    ready = bool(
        payload.get("schema") == LIVE_PREVIEW_SCHEMA
        and payload.get("ok") is True
        and "external_runtime_verified" in codes
        and "external_stage_requeue_ready" in codes
        and expected_requeue_code in codes
        and "kaggle_kernels_deleted" in codes
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "mode": payload.get("mode"),
        "failure_mode": payload.get("failure_mode"),
        "diagnosis_codes": sorted(codes),
        "external_runtime_verified": "external_runtime_verified" in codes or live.get("external_runtime_verified"),
        "external_stage_requeue_ready": "external_stage_requeue_ready" in codes,
        expected_requeue_code: expected_requeue_code in codes,
        "kaggle_kernels_deleted": "kaggle_kernels_deleted" in codes or live.get("kaggle_kernels_deleted"),
        "private_artifacts_cleaned": "private_artifacts_cleaned" in codes or live.get("private_artifacts_cleaned"),
        "token_rotation_required": "token_rotation_required" in codes or live.get("token_rotation_required"),
        "distinct_stage_miners": "distinct_stage_miners" in codes or session.get("distinct_stage_miners"),
        "stage_assignment_valid": "stage_assignment_valid" in codes or session.get("stage_assignment_valid"),
        "multi_token_generation_ready": "multi_token_generation_ready" in codes,
    }


def real_llm_evidence_path_from_gpu_report(path_value: str) -> Path | None:
    source = Path(path_value)
    if not source.is_file():
        return None
    payload = load_json(source)
    public_artifact = ((payload.get("artifacts") or {}).get("public_swarm_gpu_beta_json") or {}).get("path")
    if not public_artifact:
        return None
    public_path = source.parent / str(public_artifact)
    public_payload = load_json(public_path)
    remote_artifact = ((public_payload.get("artifacts") or {}).get("external_remote_real_llm_sharded_beta_json") or {}).get("path")
    if not remote_artifact:
        return None
    remote_path = public_path.parent / str(remote_artifact)
    remote_payload = load_json(remote_path)
    evidence_artifact = ((remote_payload.get("artifacts") or {}).get("remote_existing_real_llm_sharded_evidence_json") or {}).get("path")
    if not evidence_artifact:
        return None
    return remote_path.parent / str(evidence_artifact)


def summarize_gpu_report(path_value: str) -> dict[str, Any]:
    payload = load_json(path_value)
    codes = set(diagnosis_codes(payload))
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else find_first_dict(payload, "generation")
    evidence_path = real_llm_evidence_path_from_gpu_report(path_value)
    evidence_payload = load_json(evidence_path) if evidence_path is not None else {}
    stage_summary = find_stage_summary(evidence_payload)
    stage0 = stage_summary.get("stage_0") if isinstance(stage_summary.get("stage_0"), dict) else {}
    stage1 = stage_summary.get("stage_1") if isinstance(stage_summary.get("stage_1"), dict) else {}
    stage0_ms = float(stage0.get("elapsed_ms") or 0.0)
    stage1_ms = float(stage1.get("elapsed_ms") or 0.0)
    generated_count = int(generation.get("generated_token_count") or find_first_int(payload, "generated_token_count"))
    stage_total_seconds = round((stage0_ms + stage1_ms) / 1000.0, 6) if stage0_ms or stage1_ms else 0.0
    stage_latency_ready = bool(stage0_ms > 0 and stage1_ms > 0)
    throughput_ready = bool(generated_count > 0 and stage_total_seconds > 0)
    memory_ready = "stage_gpu_memory_reduced" in (codes | set(diagnosis_codes(evidence_payload)))
    ready = bool(
        payload.get("schema") == GPU_GENERATION_SCHEMA
        and payload.get("ok") is True
        and "multi_token_generation_ready" in codes
        and generated_count > 0
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "path": path_value,
        "diagnosis_codes": sorted(codes),
        "generated_token_count": generated_count,
        "generated_text_hash": generation.get("generated_text_hash"),
        "raw_generated_text_public": generation.get("raw_generated_text_public", False),
        "multi_token_generation_ready": bool("multi_token_generation_ready" in codes or generation.get("multi_token_generation_ready")),
        "evidence_path": str(evidence_path) if evidence_path is not None else "",
        "stage_latency": {
            "stage_latency_ready": stage_latency_ready,
            "stage0_elapsed_ms": round(stage0_ms, 3),
            "stage1_elapsed_ms": round(stage1_ms, 3),
            "stage_total_seconds": stage_total_seconds,
        },
        "throughput": {
            "throughput_summary_ready": throughput_ready,
            "generated_tokens_per_stage_second": round(generated_count / stage_total_seconds, 6) if throughput_ready else 0.0,
        },
        "memory_or_vram": {
            "memory_or_vram_summary_ready": memory_ready,
            "stage_gpu_memory_reduced": memory_ready,
            "stage0_parameter_count": stage0.get("stage_parameter_count"),
            "stage1_parameter_count": stage1.get("stage_parameter_count"),
            "full_model_parameter_count": stage0.get("full_model_parameter_count") or stage1.get("full_model_parameter_count"),
        },
    }


def support_bundle_artifact(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    bundle = support_bundle.sanitize(redact_values({
        "schema": "public_swarm_preview_v04_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "preview": report.get("preview") or {},
        "payload_summaries": report.get("payload_summaries") or {},
        "artifacts": report.get("artifacts") or {},
        "safety": report.get("safety") or {},
        "limitations": report.get("limitations") or [],
    }))
    path = output_dir / "support_bundle.json"
    write_json(path, bundle)
    return artifact_entry(path, output_dir, kind="public_swarm_preview_v04_support_bundle", schema=str(bundle.get("schema")), ok=bundle.get("ok"))


def base_artifacts(output_dir: Path, *, ok: bool) -> dict[str, Any]:
    return {
        "public_swarm_preview_v04_json": artifact_entry(output_dir / "public_swarm_preview_v04.json", output_dir, kind="public_swarm_preview_v04", schema=SCHEMA, ok=ok),
        "public_swarm_preview_v04_markdown": artifact_entry(output_dir / "public_swarm_preview_v04.md", output_dir, kind="public_swarm_preview_v04_markdown"),
    }


def common_safety(*, mode: str, external_verified: bool, package_mode: bool) -> dict[str, Any]:
    return {
        "coordinator_backed_task_execution": True,
        "serve_join_generate_product_surface": True,
        "package_mode": package_mode,
        "external_runtime_verified": external_verified,
        "tokens_public": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_payloads_redacted": True,
        "read_only_workload": WORKLOAD_TYPE,
        "tiny_gpt2_ci_fallback": True,
        "optional_distilgpt2_or_gpt2_path": True,
        "not_production": True,
        "not_p2p": True,
        "not_libp2p": True,
        "not_dht": True,
        "not_nat_traversal": True,
        "not_hivemind_or_petals_parity": True,
        "not_gpu_pooling_marketplace": True,
        "not_large_model_serving": True,
        "not_public_prompt_serving": True,
        "not_training": True,
        "mode": mode,
    }


def limitations() -> list[str]:
    return [
        "Public Swarm Preview v0.4 is Coordinator-backed and read-only; it is not production Swarm Inference.",
        "The default CI fallback remains tiny GPT. distilgpt2/gpt2 is an optional strict path and may require a larger host or Kaggle runtime.",
        "External evidence uses controlled two-stage or Kaggle Miners with temporary private artifacts and token rotation after live runs.",
        "This does not implement libp2p, DHT, NAT traversal, decentralized security, payment/tokenomics, training, or Hivemind/Petals-level serving.",
    ]


def operator_actions(mode: str, *, optional_ready: bool) -> list[str]:
    actions = [
        "Use the top-level v0.4 JSON/Markdown plus support bundle as the shareable redacted evidence.",
        "Do not publish raw prompts, generated text, token ids, activations, runtime state, private env files, or temporary Kaggle kernels.",
    ]
    if mode == MODE_PACKAGE:
        actions.append("Use PUBLIC_SWARM_PREVIEW_V04.md to run the two-machine or Kaggle path.")
    if not optional_ready:
        actions.append("Run the optional strict path with --run-optional-model --optional-model-id distilgpt2 or gpt2 on a host with optional [hf] dependencies.")
    actions.append("Rotate Coordinator and Miner tokens after every temporary public HTTP/Kaggle proof.")
    return actions


def write_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "PUBLIC_SWARM_PREVIEW_V04.md"
    lines = [
        "# CrowdTensor Public Swarm Inference Preview v0.4",
        "",
        "This runbook exercises the current Coordinator-backed, read-only two-stage generation path. It is not production Swarm Inference or P2P routing.",
        "",
        "## Local / Two-Machine Path",
        "",
        "```bash",
        "python -m pip install -e '.[dev,hf]'",
        f"crowdtensor public-swarm-product-beta package --target local --public-host {args.public_host} --port {args.port} --json",
        f"crowdtensor serve --public-host {args.public_host} --bind-host {args.bind_host} --port {args.port} --profile gpu-generation --hf-model-id {args.hf_model_id} --run --json",
        f"crowdtensor join --coordinator-url http://{args.public_host}:{args.port} --stage stage0 --hf-model-id {args.hf_model_id} --run --json",
        f"crowdtensor join --coordinator-url http://{args.public_host}:{args.port} --stage stage1 --hf-model-id {args.hf_model_id} --run --json",
        f"crowdtensor generate --coordinator-url http://{args.public_host}:{args.port} --prompt-text 'CrowdTensor preview v0.4' --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "## Kaggle CPU/GPU Path",
        "",
        "```bash",
        f"crowdtensor live-preview live-kaggle --public-host {args.public_host} --port {args.port} --base-port {args.base_port} --failure-mode kill-stage0-after-claim --json",
        f"crowdtensor live-preview live-kaggle --public-host {args.public_host} --port {args.port + 1} --base-port {args.base_port + 1} --failure-mode kill-stage1-after-claim --json",
        "crowdtensor gpu-generate kaggle-auto --kaggle-owner YOUR_KAGGLE_USERNAME --max-new-tokens 16 --json",
        "```",
        "",
        "## v0.4 Aggregated Evidence",
        "",
        "```bash",
        "crowdtensor preview-v04 evidence-import --json",
        "python scripts/public_swarm_preview_v04_check.py --mode evidence-import --json",
        "```",
        "",
        f"Optional larger-model strict path: add `--run-optional-model --optional-model-id {args.optional_model_id}` on a host with enough CPU/GPU memory.",
        "",
        "Share only the top-level JSON/Markdown and support bundle. Rotate all temporary tokens after public HTTP/Kaggle proofs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="public_swarm_preview_v04_runbook")


def product_mvp_command(args: argparse.Namespace, output_dir: Path, *, model_id: str, port: int, require_runtime: bool) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "product_swarm_mvp_check.py"),
        "--output-dir",
        str(output_dir),
        "--port",
        str(port),
        "--backend",
        args.backend,
        "--hf-model-id",
        model_id,
        "--prompt-text",
        args.prompt_text,
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
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if require_runtime:
        command.append("--require-hf-runtime")
    return command


def run_product_mvp(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_json_step(
        "product_swarm_mvp_tiny_gpt2",
        product_mvp_command(args, output_dir, model_id=args.hf_model_id, port=args.port, require_runtime=args.require_hf_runtime),
        runner=runner,
        timeout_seconds=max(float(args.generate_timeout), float(args.miner_timeout), 60.0) * max(2, args.max_new_tokens) + 180.0,
        allow_failure_payload=not args.require_hf_runtime,
    )


def run_optional_model(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    if not args.run_optional_model:
        return {
            "name": "product_swarm_mvp_optional_model",
            "ok": False,
            "skipped": True,
            "reason": "run_optional_model_false",
        }, {}
    return run_json_step(
        "product_swarm_mvp_optional_model",
        product_mvp_command(args, output_dir, model_id=args.optional_model_id, port=args.port + 1, require_runtime=True),
        runner=runner,
        timeout_seconds=max(float(args.generate_timeout), float(args.miner_timeout), 60.0) * max(2, args.max_new_tokens) + 420.0,
        allow_failure_payload=False,
    )


def build_common_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mode: str,
    steps: list[dict[str, Any]],
    product_payload: dict[str, Any],
    optional_payload: dict[str, Any],
    stage0_payload: dict[str, Any],
    stage1_payload: dict[str, Any],
    gpu_summary: dict[str, Any],
    product_beta_payload: dict[str, Any] | None = None,
    extra_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    product_summary = summarize_product_mvp(product_payload, required_tokens=max(2, args.max_new_tokens))
    optional_summary = summarize_optional_model(optional_payload, optional_model_id=args.optional_model_id, required_tokens=max(2, args.max_new_tokens))
    stage0_summary = summarize_live_report(stage0_payload, expected_requeue_code="live_stage0_requeue_ready")
    stage1_summary = summarize_live_report(stage1_payload, expected_requeue_code="live_stage1_requeue_ready")
    product_beta_codes = set(diagnosis_codes(product_beta_payload or {}))

    external_ready = bool(
        stage0_summary.get("ready")
        and stage1_summary.get("ready")
        and stage0_summary.get("external_runtime_verified")
        and stage1_summary.get("external_runtime_verified")
    )
    external_requeue_ready = bool(stage0_summary.get("live_stage0_requeue_ready") and stage1_summary.get("live_stage1_requeue_ready"))
    distinct_stage_miners = bool(
        stage0_summary.get("distinct_stage_miners")
        or stage1_summary.get("distinct_stage_miners")
        or "distinct_stage_miners" in set(gpu_summary.get("diagnosis_codes") or [])
    )
    stage_assignment_valid = bool(
        stage0_summary.get("stage_assignment_valid")
        or stage1_summary.get("stage_assignment_valid")
        or "stage_assignment_valid" in set(gpu_summary.get("diagnosis_codes") or [])
    )
    multi_token_ready = bool(
        product_summary.get("multi_token_generation_ready")
        or optional_summary.get("multi_token_generation_ready")
        or gpu_summary.get("multi_token_generation_ready")
        or stage0_summary.get("multi_token_generation_ready")
        or stage1_summary.get("multi_token_generation_ready")
    )
    stage_latency_ready = bool(product_summary.get("stage_latency_ready") or gpu_summary.get("stage_latency", {}).get("stage_latency_ready"))
    throughput_ready = bool(product_summary.get("throughput_summary_ready") or gpu_summary.get("throughput", {}).get("throughput_summary_ready"))
    memory_ready = bool(product_summary.get("memory_or_vram_summary_ready") or gpu_summary.get("memory_or_vram", {}).get("memory_or_vram_summary_ready"))
    tiny_fallback_ready = bool(product_summary.get("ready") or "product_swarm_mvp_degraded_ready" in set(product_summary.get("diagnosis_codes") or []))
    optional_ready = bool(optional_summary.get("optional_model_ready"))
    package_ready = bool(mode == MODE_PACKAGE and "miner_join_pack_ready" in product_beta_codes and "private_artifacts_local_only" in product_beta_codes)

    ready = bool(
        external_ready
        and external_requeue_ready
        and distinct_stage_miners
        and stage_assignment_valid
        and multi_token_ready
        and stage_latency_ready
        and throughput_ready
        and memory_ready
        and tiny_fallback_ready
        and (optional_ready or not args.require_optional_model_ready)
        and (mode != MODE_PACKAGE or package_ready)
    )

    codes = set(diagnosis_codes(product_payload, optional_payload, stage0_payload, stage1_payload, product_beta_payload or {}))
    codes.update(gpu_summary.get("diagnosis_codes") or [])
    if ready:
        codes.update({
            "public_swarm_preview_v04_ready",
            "external_two_stage_generation_ready",
            "external_stage_requeue_ready",
            "multi_token_generation_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "stage_latency_ready",
            "throughput_summary_ready",
            "memory_or_vram_summary_ready",
            "tiny_gpt2_ci_fallback_ready",
            "redacted_evidence_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_libp2p",
            "not_dht",
            "not_nat_traversal",
            "not_large_model_serving",
        })
    else:
        codes.add("public_swarm_preview_v04_blocked")
    if external_ready:
        codes.add("external_two_stage_generation_ready")
    if external_requeue_ready:
        codes.add("external_stage_requeue_ready")
    if multi_token_ready:
        codes.add("multi_token_generation_ready")
    if distinct_stage_miners:
        codes.add("distinct_stage_miners")
    if stage_assignment_valid:
        codes.add("stage_assignment_valid")
    if stage_latency_ready:
        codes.add("stage_latency_ready")
    if throughput_ready:
        codes.add("throughput_summary_ready")
    if memory_ready:
        codes.add("memory_or_vram_summary_ready")
    if tiny_fallback_ready:
        codes.add("tiny_gpt2_ci_fallback_ready")
    if optional_ready:
        codes.add("optional_distilgpt2_or_gpt2_strict_ready")
    else:
        codes.add("optional_distilgpt2_or_gpt2_path_available")
    if package_ready:
        codes.update({"kaggle_or_two_machine_runbook_ready", "miner_join_pack_ready", "private_artifacts_local_only"})
    if gpu_summary.get("ready"):
        codes.add("gpu_generation_evidence_import_ready")

    artifacts = base_artifacts(output_dir, ok=ready)
    artifacts.update({
        "product_swarm_mvp_json": artifact_entry(output_dir / "product-mvp" / "product_swarm_mvp_check.json", output_dir, kind="product_swarm_mvp_check", schema=PRODUCT_MVP_SCHEMA, ok=product_payload.get("ok") if product_payload else None),
        "optional_model_mvp_json": artifact_entry(output_dir / "optional-model-mvp" / "product_swarm_mvp_check.json", output_dir, kind="product_swarm_mvp_optional_model", schema=PRODUCT_MVP_SCHEMA, ok=optional_payload.get("ok") if optional_payload else None),
        "stage0_live_preview_json": artifact_entry(Path(args.live_stage0_report), output_dir, kind="public_swarm_live_preview_rc_stage0", schema=LIVE_PREVIEW_SCHEMA, ok=stage0_payload.get("ok") if stage0_payload else None),
        "stage1_live_preview_json": artifact_entry(Path(args.live_stage1_report), output_dir, kind="public_swarm_live_preview_rc_stage1", schema=LIVE_PREVIEW_SCHEMA, ok=stage1_payload.get("ok") if stage1_payload else None),
        "gpu_generation_json": artifact_entry(Path(args.gpu_report), output_dir, kind="gpu_sharded_generation_beta", schema=GPU_GENERATION_SCHEMA, ok=gpu_summary.get("ok") if gpu_summary else None),
    })
    if product_beta_payload is not None:
        artifacts["product_beta_package_json"] = artifact_entry(output_dir / "product-beta-package" / "public_swarm_product_beta.json", output_dir, kind="public_swarm_product_beta", schema=PRODUCT_BETA_SCHEMA, ok=product_beta_payload.get("ok"))
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
        "preview": {
            "ready": ready,
            "external_two_stage_generation_ready": external_ready,
            "external_stage_requeue_ready": external_requeue_ready,
            "multi_token_generation_ready": multi_token_ready,
            "distinct_stage_miners": distinct_stage_miners,
            "stage_assignment_valid": stage_assignment_valid,
            "stage_latency_ready": stage_latency_ready,
            "throughput_summary_ready": throughput_ready,
            "memory_or_vram_summary_ready": memory_ready,
            "tiny_gpt2_ci_fallback_ready": tiny_fallback_ready,
            "optional_model_ready": optional_ready,
            "optional_model_id": args.optional_model_id,
            "package_ready": package_ready,
            "gpu_generation_ready": bool(gpu_summary.get("ready")),
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "workload_type": WORKLOAD_TYPE,
            "user_surface": ["serve", "join stage0", "join stage1", "generate", "preview-v04"],
        },
        "performance": {
            "stage_latency": gpu_summary.get("stage_latency") or product_summary.get("performance", {}),
            "throughput": gpu_summary.get("throughput") or product_summary.get("performance", {}),
            "memory_or_vram": gpu_summary.get("memory_or_vram") or product_summary.get("runtime_resources", {}),
            "local_product_mvp": product_summary.get("performance"),
        },
        "steps": steps,
        "payload_summaries": {
            "product_mvp": product_summary,
            "optional_model": optional_summary,
            "stage0_live_preview": stage0_summary,
            "stage1_live_preview": stage1_summary,
            "gpu_generation": gpu_summary,
            "product_beta_package": {
                "schema": (product_beta_payload or {}).get("schema"),
                "ok": (product_beta_payload or {}).get("ok"),
                "diagnosis_codes": sorted(product_beta_codes),
                "package_ready": package_ready,
            },
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": artifacts,
        "safety": common_safety(mode=mode, external_verified=external_ready, package_mode=mode == MODE_PACKAGE),
        "operator_action": operator_actions(mode, optional_ready=optional_ready),
        "limitations": limitations(),
    }
    return report


def build_local_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_mvp(args, output_dir=output_dir / "product-mvp", runner=runner)
    optional_step, optional_payload = run_optional_model(args, output_dir=output_dir / "optional-model-mvp", runner=runner)
    stage0_payload = load_json(args.live_stage0_report)
    stage1_payload = load_json(args.live_stage1_report)
    gpu_summary = summarize_gpu_report(args.gpu_report)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_LOCAL_SMOKE,
        steps=[product_step, optional_step],
        product_payload=product_payload,
        optional_payload=optional_payload,
        stage0_payload=stage0_payload,
        stage1_payload=stage1_payload,
        gpu_summary=gpu_summary,
    )


def build_package(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_mvp(args, output_dir=output_dir / "product-mvp", runner=runner)
    optional_step, optional_payload = run_optional_model(args, output_dir=output_dir / "optional-model-mvp", runner=runner)
    stage0_payload = load_json(args.live_stage0_report)
    stage1_payload = load_json(args.live_stage1_report)
    gpu_summary = summarize_gpu_report(args.gpu_report)
    product_beta_step, product_beta_payload = run_json_step(
        "public_swarm_product_beta_package",
        [
            sys.executable,
            str(ROOT / "scripts" / "public_swarm_product_beta_pack.py"),
            "package",
            "--output-dir",
            str(output_dir / "product-beta-package"),
            "--target",
            args.target,
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
            "--hf-model-id",
            args.hf_model_id,
            "--gpu-report",
            args.gpu_report,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--cpu-timeout-seconds",
            str(args.cpu_timeout_seconds),
            "--json",
        ],
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 420.0,
    )
    runbook = write_runbook(args, output_dir)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_PACKAGE,
        steps=[product_step, optional_step, product_beta_step],
        product_payload=product_payload,
        optional_payload=optional_payload,
        stage0_payload=stage0_payload,
        stage1_payload=stage1_payload,
        gpu_summary=gpu_summary,
        product_beta_payload=product_beta_payload,
        extra_artifacts={"public_swarm_preview_v04_runbook": runbook},
    )


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    product_payload = load_json(args.product_mvp_report)
    stage0_payload = load_json(args.live_stage0_report)
    stage1_payload = load_json(args.live_stage1_report)
    optional_payload = load_json(args.optional_model_report)
    gpu_summary = summarize_gpu_report(args.gpu_report)
    runbook = write_runbook(args, output_dir)
    extra_artifacts = {"public_swarm_preview_v04_runbook": runbook}
    if args.product_mvp_report:
        extra_artifacts["product_swarm_mvp_source_json"] = artifact_entry(
            Path(args.product_mvp_report),
            output_dir,
            kind="product_swarm_mvp_check_source",
            schema=PRODUCT_MVP_SCHEMA,
            ok=product_payload.get("ok") if product_payload else None,
        )
    if args.optional_model_report:
        extra_artifacts["optional_model_mvp_source_json"] = artifact_entry(
            Path(args.optional_model_report),
            output_dir,
            kind="product_swarm_mvp_optional_source",
            schema=PRODUCT_MVP_SCHEMA,
            ok=optional_payload.get("ok") if optional_payload else None,
        )
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_EVIDENCE_IMPORT,
        steps=[],
        product_payload=product_payload,
        optional_payload=optional_payload,
        stage0_payload=stage0_payload,
        stage1_payload=stage1_payload,
        gpu_summary=gpu_summary,
        extra_artifacts=extra_artifacts,
    )


def render_markdown(report: dict[str, Any]) -> str:
    preview = report.get("preview") if isinstance(report.get("preview"), dict) else {}
    perf = report.get("performance") if isinstance(report.get("performance"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Inference Preview v0.4",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- backend: `{report.get('backend')}`",
        f"- ready: `{preview.get('ready')}`",
        f"- external_two_stage_generation_ready: `{preview.get('external_two_stage_generation_ready')}`",
        f"- external_stage_requeue_ready: `{preview.get('external_stage_requeue_ready')}`",
        f"- multi_token_generation_ready: `{preview.get('multi_token_generation_ready')}`",
        f"- stage_latency_ready: `{preview.get('stage_latency_ready')}`",
        f"- throughput_summary_ready: `{preview.get('throughput_summary_ready')}`",
        f"- memory_or_vram_summary_ready: `{preview.get('memory_or_vram_summary_ready')}`",
        f"- optional_model_ready: `{preview.get('optional_model_ready')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Performance",
        "",
        f"- stage_latency: `{json.dumps(perf.get('stage_latency') or {}, sort_keys=True)}`",
        f"- throughput: `{json.dumps(perf.get('throughput') or {}, sort_keys=True)}`",
        f"- memory_or_vram: `{json.dumps(perf.get('memory_or_vram') or {}, sort_keys=True)}`",
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


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report)
    report = support_bundle.sanitize(redact_values(report))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Preview v0.4 report contained secret-like fragments"
    json_path = output_dir / "public_swarm_preview_v04.json"
    md_path = output_dir / "public_swarm_preview_v04.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_preview_v04_json"]["present"] = True
    report["artifacts"]["public_swarm_preview_v04_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_LOCAL_SMOKE:
        report = build_local_smoke(args, output_dir=output_dir, runner=runner)
    elif args.mode == MODE_PACKAGE:
        report = build_package(args, output_dir=output_dir, runner=runner)
    else:
        report = build_evidence_import(args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir)


def default_kaggle_owner() -> str:
    return os.environ.get("KAGGLE_USERNAME", "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Inference Preview v0.4 evidence.")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--target", choices=["local", "kaggle"], default="kaggle")
    parser.add_argument("--miner-id-prefix", default="public-swarm-preview-v04")
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--optional-model-id", choices=["distilgpt2", "gpt2"], default=DEFAULT_OPTIONAL_MODEL_ID)
    parser.add_argument("--run-optional-model", action="store_true")
    parser.add_argument("--require-optional-model-ready", action="store_true")
    parser.add_argument("--require-hf-runtime", action="store_true")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--live-stage0-report", default=DEFAULT_LIVE_STAGE0_REPORT)
    parser.add_argument("--live-stage1-report", default=DEFAULT_LIVE_STAGE1_REPORT)
    parser.add_argument("--product-mvp-report", default=DEFAULT_PRODUCT_MVP_REPORT)
    parser.add_argument("--optional-model-report", default="")
    parser.add_argument("--product-beta-report", default=DEFAULT_PRODUCT_BETA_REPORT)
    parser.add_argument("--prompt-text", default="CrowdTensor preview v0.4")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--session-queue-timeout", type=float, default=45.0)
    parser.add_argument("--miner-timeout", type=float, default=240.0)
    parser.add_argument("--generate-timeout", type=float, default=240.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1 or args.base_port < 1:
        raise SystemExit("--port and --base-port must be positive")
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "cpu_timeout_seconds",
        "startup_timeout",
        "session_queue_timeout",
        "miner_timeout",
        "generate_timeout",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.require_optional_model_ready and not args.run_optional_model and not args.optional_model_report:
        raise SystemExit("--require-optional-model-ready requires --run-optional-model or --optional-model-report")
    return args


def print_human(report: dict[str, Any]) -> None:
    preview = report.get("preview") if isinstance(report.get("preview"), dict) else {}
    print("CrowdTensor Public Swarm Inference Preview v0.4")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {preview.get('ready')}")
    print(f"  external_two_stage_generation_ready: {preview.get('external_two_stage_generation_ready')}")
    print(f"  external_stage_requeue_ready: {preview.get('external_stage_requeue_ready')}")
    print(f"  stage_latency_ready: {preview.get('stage_latency_ready')}")
    print(f"  throughput_summary_ready: {preview.get('throughput_summary_ready')}")
    print(f"  memory_or_vram_summary_ready: {preview.get('memory_or_vram_summary_ready')}")
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
