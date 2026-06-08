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


def _shell_join(parts: list[str]) -> str:
    return " ".join(str(part) if " " not in str(part) else json.dumps(str(part)) for part in parts if str(part))


def command_entry(
    label: str,
    command: list[str],
    *,
    reason: str = "",
    requires_private_credentials: bool = False,
    side_effectful: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": _shell_join(command),
        "public_artifact_safe": True,
    }
    if reason:
        entry["reason"] = reason
    if requires_private_credentials:
        entry["requires_private_credentials"] = True
        entry["credential_note"] = (
            "Prepare private operator/Kaggle credentials and temporary runtime tokens before running this command; "
            "credential values stay out of public artifacts."
        )
    if side_effectful:
        entry["side_effectful"] = True
        entry["side_effect_note"] = (
            "This command can depend on side-effectful retained live evidence; delete temporary kernels and rotate "
            "tokens after fresh public HTTP/Kaggle collection."
        )
    return entry


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "public_swarm_preview_v04.md",
        "summary_json": output_dir / "public_swarm_preview_v04.json",
        "summary_markdown": output_dir / "public_swarm_preview_v04.md",
        "support_bundle": output_dir / "support_bundle.json",
    }
    present = sum(1 for path in paths.values() if path.is_file())
    return {
        **{name: str(path) for name, path in paths.items()},
        "artifact_count": len(paths),
        "present_artifact_count": present,
        "shareable_paths": [
            str(paths["summary_json"]),
            str(paths["summary_markdown"]),
            str(paths["support_bundle"]),
        ],
        "public_artifact_safe": True,
    }


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
            step["ok"] = bool(step["ok"] and payload.get("ok"))
    if not step.get("ok"):
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:], secret_values)
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


def preview_v04_command(args: argparse.Namespace, output_dir: Path, mode: str) -> list[str]:
    command = [
        "crowdtensor",
        "preview-v04",
        mode,
        "--output-dir",
        str(output_dir),
        "--backend",
        str(args.backend),
        "--public-host",
        str(args.public_host),
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gpu-report",
        str(args.gpu_report),
        "--live-stage0-report",
        str(args.live_stage0_report),
        "--live-stage1-report",
        str(args.live_stage1_report),
        "--product-mvp-report",
        str(args.product_mvp_report),
        "--product-beta-report",
        str(args.product_beta_report),
        "--optional-model-id",
        str(args.optional_model_id),
    ]
    if mode == MODE_PACKAGE:
        command.extend([
            "--target",
            str(args.target),
            "--miner-id-prefix",
            str(args.miner_id_prefix),
        ])
    if args.optional_model_report:
        command.extend(["--optional-model-report", str(args.optional_model_report)])
    if args.run_optional_model:
        command.append("--run-optional-model")
    if args.require_optional_model_ready:
        command.append("--require-optional-model-ready")
    if args.require_hf_runtime:
        command.append("--require-hf-runtime")
    if args.hf_model_id != DEFAULT_MODEL_ID:
        command.extend(["--hf-model-id", str(args.hf_model_id)])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", "HF_CACHE_DIR"])
    return command


def not_completed_items(report: dict[str, Any]) -> list[str]:
    preview = report.get("preview") if isinstance(report.get("preview"), dict) else {}
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    mode = str(report.get("mode") or "")
    items = [
        ("Public Swarm Preview v0.4 ready", report.get("ok") is True and preview.get("ready") is True),
        ("external two-stage generation ready", preview.get("external_two_stage_generation_ready")),
        ("external stage requeue ready", preview.get("external_stage_requeue_ready")),
        ("multi-token generation ready", preview.get("multi_token_generation_ready")),
        ("distinct stage miners", preview.get("distinct_stage_miners")),
        ("stage assignment valid", preview.get("stage_assignment_valid")),
        ("stage latency ready", preview.get("stage_latency_ready")),
        ("throughput summary ready", preview.get("throughput_summary_ready")),
        ("memory or VRAM summary ready", preview.get("memory_or_vram_summary_ready")),
        ("tiny GPT fallback ready", preview.get("tiny_gpt2_ci_fallback_ready")),
        ("redacted evidence ready", "redacted_evidence_ready" in set(report.get("diagnosis_codes") or [])),
        ("read-only workload boundary recorded", safety.get("read_only_workload") == WORKLOAD_TYPE),
    ]
    if mode == MODE_PACKAGE:
        items.append(("package join material ready", preview.get("package_ready")))
    if "optional_distilgpt2_or_gpt2_strict_ready" in set(report.get("diagnosis_codes") or []):
        items.append(("optional distilgpt2 or gpt2 strict path ready", preview.get("optional_model_ready")))
    return [label for label, ready in items if ready is not True]


def recommended_next_command(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    ready: bool,
    mode: str,
    not_completed: list[str],
) -> dict[str, Any]:
    if ready:
        return command_entry(
            "inspect Preview v0.4 evidence",
            artifact_command(output_dir, "public_swarm_preview_v04.md"),
            reason="review_artifacts",
        )
    if mode == MODE_PACKAGE:
        return command_entry(
            "run local Preview v0.4 smoke",
            preview_v04_command(args, output_dir, MODE_LOCAL_SMOKE),
            reason="verify_local_preview_path",
        )
    if mode == MODE_EVIDENCE_IMPORT:
        return command_entry(
            "refresh retained Preview v0.4 evidence",
            preview_v04_command(args, output_dir, MODE_EVIDENCE_IMPORT),
            reason="refresh_retained_evidence",
        )
    reason = "fix_preview_v04_blockers" if not_completed else "rerun_local_smoke"
    return command_entry(
        "retry local Preview v0.4 smoke",
        preview_v04_command(args, output_dir, MODE_LOCAL_SMOKE),
        reason=reason,
    )


def next_commands(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    ready: bool,
    mode: str,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    commands = [
        command_entry(
            "inspect shareable summary",
            artifact_command(output_dir, "public_swarm_preview_v04.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect support bundle",
            artifact_command(output_dir, "support_bundle.json", lines="1,220p"),
            reason="inspect_diagnostics",
        ),
    ]
    if mode == MODE_PACKAGE:
        commands.append(command_entry(
            "run local Preview v0.4 smoke",
            preview_v04_command(args, output_dir, MODE_LOCAL_SMOKE),
            reason="verify_local_preview_path",
        ))
    elif mode == MODE_EVIDENCE_IMPORT:
        commands.append(command_entry(
            "refresh retained Preview v0.4 evidence",
            preview_v04_command(args, output_dir, MODE_EVIDENCE_IMPORT),
            reason="refresh_retained_evidence",
        ))
    elif ready:
        commands.append(command_entry(
            "rerun local Preview v0.4 smoke",
            preview_v04_command(args, output_dir, MODE_LOCAL_SMOKE),
            reason="refresh_local_preview_path",
        ))
    else:
        commands.append(command_entry(
            "retry local Preview v0.4 smoke",
            preview_v04_command(args, output_dir, MODE_LOCAL_SMOKE),
            reason=str(recommended.get("reason") or "fix_preview_v04_blockers"),
        ))
    commands.append(command_entry(
        "refresh live stage evidence",
        [
            "crowdtensor",
            "live-preview",
            "live-kaggle",
            "--public-host",
            str(args.public_host),
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--failure-mode",
            "kill-stage0-after-claim",
        ],
        reason="refresh_side_effectful_live_evidence",
        requires_private_credentials=True,
        side_effectful=True,
    ))
    return commands


def user_status(
    *,
    ready: bool,
    mode: str,
    recommended: dict[str, Any],
    not_completed: list[str],
) -> dict[str, Any]:
    if ready:
        state = "ready"
        headline = "Public Swarm Preview v0.4 evidence is ready."
        next_step = "review_artifacts"
    elif mode == MODE_PACKAGE:
        state = "package-blocked"
        headline = "Preview v0.4 package did not satisfy the package readiness gate."
        next_step = "run_local_or_fix_package"
    else:
        state = "blocked"
        headline = "Public Swarm Preview v0.4 evidence needs attention."
        next_step = "fix_blockers"
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "not_completed_count": len(not_completed),
        "public_artifact_safe": True,
    }


def review_summary(
    report: dict[str, Any],
    *,
    output_dir: Path,
    recommended: dict[str, Any],
    not_completed: list[str],
) -> dict[str, Any]:
    codes = [str(code) for code in (report.get("diagnosis_codes") or [])]
    ready = bool(report.get("ok"))
    attention = "none" if ready else (not_completed[0] if not_completed else "public_swarm_preview_v04_blocked")
    return {
        "schema": "public_swarm_preview_v04_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "headline": "Public Swarm Preview v0.4 evidence is ready." if ready else "Public Swarm Preview v0.4 evidence needs attention.",
        "next_step": "review_artifacts" if ready else "fix_blockers",
        "inspect_first": str(output_dir / "public_swarm_preview_v04.md"),
        "support_bundle": str(output_dir / "support_bundle.json"),
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": "public_swarm_preview_v04_ready" if ready else (codes[0] if codes else "public_swarm_preview_v04_blocked"),
        "attention": attention,
        "attention_detail": "; ".join(not_completed[:5]),
        "not_completed_count": len(not_completed),
        "public_artifact_safe": True,
    }


def attach_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    missing = not_completed_items(report)
    recommended = recommended_next_command(
        args,
        output_dir,
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or args.mode),
        not_completed=missing,
    )
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = next_commands(
        args,
        output_dir,
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or args.mode),
        recommended=recommended,
    )
    report["user_status"] = user_status(
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or args.mode),
        recommended=recommended,
        not_completed=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        not_completed=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def ensure_user_guidance(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    existing_status = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    existing_ready = existing_status.get("state") == "ready"
    if (
        isinstance(report.get("recommended_next_command"), dict)
        and isinstance(report.get("next_commands"), list)
        and isinstance(report.get("user_status"), dict)
        and isinstance(report.get("review_summary"), dict)
        and existing_ready == bool(report.get("ok"))
    ):
        report.setdefault("artifact_summary", artifact_summary(output_dir))
        return report
    missing = not_completed_items(report)
    ready = bool(report.get("ok"))
    recommended = command_entry(
        "inspect Preview v0.4 evidence",
        artifact_command(output_dir, "public_swarm_preview_v04.md"),
        reason="review_artifacts" if ready else "review_missing_evidence",
    )
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = [
        command_entry(
            "inspect shareable summary",
            artifact_command(output_dir, "public_swarm_preview_v04.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect support bundle",
            artifact_command(output_dir, "support_bundle.json", lines="1,220p"),
            reason="inspect_diagnostics",
        ),
    ]
    report["user_status"] = user_status(
        ready=ready,
        mode=str(report.get("mode") or "unknown"),
        recommended=recommended,
        not_completed=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        not_completed=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Preview v0.4 artifacts summarize preview readiness with "
            "counts, hashes, retained/live evidence, performance summaries, and support diagnostics only. "
            "Run `crowdtensor generate` in human mode to see a local answer."
        ),
    }


def prompt_scope_summary(args: argparse.Namespace) -> dict[str, Any]:
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
            "This Public Swarm Preview v0.4 artifact records prompt source/count and "
            "placeholder safety only; raw prompt text is excluded from public JSON, "
            "Markdown, runbooks, and support bundles."
        ),
    }


def prompt_secret_values(args: argparse.Namespace) -> list[str]:
    return [str(args.prompt_text)] if getattr(args, "prompt_text", "") else []


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
            "This Public Swarm Preview v0.4 report is shareable preview evidence, not a local "
            "answer transcript; raw prompts, generated text, generated token ids, activations, "
            "leases, credentials, private env files, and runtime state are excluded."
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
            "Share public_swarm_preview_v04.json/md and support_bundle.json; they contain "
            "hashes/counts and readiness evidence, not raw prompts or answers."
        ),
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


def answer_scope_text(answer_scope: dict[str, Any]) -> str:
    return (
        f"state={answer_scope.get('scope_state') or 'unknown'} "
        f"terminal_only={bool(answer_scope.get('terminal_only'))} "
        f"visible_in_terminal={bool(answer_scope.get('visible_in_terminal'))} "
        f"saved_json={answer_scope.get('saved_json_display')} "
        f"saved_markdown={answer_scope.get('saved_markdown_display')} "
        f"public_artifact_safe={bool(answer_scope.get('public_artifact_safe'))}"
    )


def prompt_scope_note(prompt_scope: dict[str, Any]) -> str:
    return str(prompt_scope.get("summary") or "Public artifacts exclude raw prompt text.")


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


def support_bundle_artifact(output_dir: Path, report: dict[str, Any], *, secret_values: list[str] | None = None) -> dict[str, Any]:
    bundle = support_bundle.sanitize(redact_values({
        "schema": "public_swarm_preview_v04_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "preview": report.get("preview") or {},
        "payload_summaries": report.get("payload_summaries") or {},
        "artifacts": report.get("artifacts") or {},
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "user_status": report.get("user_status"),
        "review_summary": report.get("review_summary"),
        "recommended_next_command": report.get("recommended_next_command"),
        "next_commands": report.get("next_commands"),
        "artifact_summary": report.get("artifact_summary"),
        "not_completed": report.get("not_completed"),
        "safety": report.get("safety") or {},
        "limitations": report.get("limitations") or [],
    }, secret_values))
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
        f"crowdtensor generate --coordinator-url http://{args.public_host}:{args.port} --prompt-text '<your-private-prompt>' --max-new-tokens {args.max_new_tokens} --json",
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
        secret_values=prompt_secret_values(args),
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
        secret_values=prompt_secret_values(args),
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
    report["output_request"] = output_request_summary()
    report["prompt_scope"] = prompt_scope_summary(args)
    report["answer_scope"] = answer_scope_summary()
    report["shareable_summary"] = shareable_summary()
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
        secret_values=prompt_secret_values(args),
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
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    next_items = report.get("next_commands") if isinstance(report.get("next_commands"), list) else []
    perf = report.get("performance") if isinstance(report.get("performance"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
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
        "## Review",
        "",
        f"- state: `{review.get('state')}`",
        f"- status: `{user.get('headline')}`",
        f"- next step: `{review.get('next_step')}`",
        f"- inspect first: `{review.get('inspect_first')}`",
        f"- recommended: `{recommended.get('label')}` reason=`{recommended.get('reason')}`",
        f"- recommended command: `{recommended.get('command_line')}`",
        f"- not completed count: `{review.get('not_completed_count')}`",
        "",
        "## What To Do Next",
        "",
    ]
    if next_items:
        lines.extend(
            (
                f"- {item.get('label')}: `{item.get('command_line')}`"
                + (" (requires private credentials; see runbook)" if item.get("requires_private_credentials") else "")
                + (" (side-effectful live evidence)" if item.get("side_effectful") else "")
            )
            for item in next_items
            if isinstance(item, dict)
        )
    else:
        lines.append("- none")
    lines.extend([
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
        "## Artifact Summary",
        "",
        f"- inspect first: `{artifact_report.get('inspect_first')}`",
        f"- summary JSON: `{artifact_report.get('summary_json')}`",
        f"- summary Markdown: `{artifact_report.get('summary_markdown')}`",
        f"- support bundle: `{artifact_report.get('support_bundle')}`",
        f"- present: `{artifact_report.get('present_artifact_count')}` / `{artifact_report.get('artifact_count')}`",
        f"- public artifact safe: `{artifact_report.get('public_artifact_safe')}`",
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
        "## Not Completed",
        "",
    ])
    not_completed = report.get("not_completed") or []
    lines.extend(f"- {item}" for item in not_completed) if not_completed else lines.append("- none")
    lines.extend([
        "",
        "## Artifacts",
        "",
    ])
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def persist_report(report: dict[str, Any], *, output_dir: Path, secret_values: list[str] | None = None) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", {
        "source": "unknown",
        "prompt_count": 0,
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": False,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
    })
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report.setdefault("artifact_summary", artifact_summary(output_dir))
    report = ensure_user_guidance(report, output_dir=output_dir)
    report["artifacts"]["support_bundle_json"] = support_bundle_artifact(output_dir, report, secret_values=secret_values)
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Preview v0.4 report contained secret-like fragments"
        report = ensure_user_guidance(report, output_dir=output_dir)
    json_path = output_dir / "public_swarm_preview_v04.json"
    md_path = output_dir / "public_swarm_preview_v04.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_preview_v04_json"]["present"] = True
    report["artifacts"]["public_swarm_preview_v04_markdown"]["present"] = True
    report["artifact_summary"] = artifact_summary(output_dir)
    if isinstance(report.get("review_summary"), dict):
        report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
        report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    support_path = output_dir / "support_bundle.json"
    if support_path.is_file():
        support_payload = load_json(support_path)
        support_payload["artifacts"] = report.get("artifacts")
        support_payload["artifact_summary"] = report.get("artifact_summary")
        support_payload["review_summary"] = report.get("review_summary")
        support_payload["user_status"] = report.get("user_status")
        support_payload["recommended_next_command"] = report.get("recommended_next_command")
        support_payload["next_commands"] = report.get("next_commands")
        support_payload["not_completed"] = report.get("not_completed")
        write_json(support_path, support_bundle.sanitize(redact_values(support_payload, secret_values)))
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
    report = attach_user_guidance(report, args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir, secret_values=prompt_secret_values(args))


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
    user_status_value = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
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
    print(f"  optional_model_ready: {preview.get('optional_model_ready')}")
    if user_status_value:
        print(
            "  status: "
            f"state={user_status_value.get('state')} next_step={user_status_value.get('next_step')} "
            f"recommended={user_status_value.get('recommended_label')} not_completed={user_status_value.get('not_completed_count')} "
            f"public_artifact_safe={bool(user_status_value.get('public_artifact_safe'))}"
        )
    if review:
        print(
            "  review: "
            f"state={review.get('state')} next_step={review.get('next_step')} "
            f"attention={review.get('attention')} not_completed={review.get('not_completed_count')} "
            f"public_artifact_safe={bool(review.get('public_artifact_safe'))}"
        )
        print(
            "  review_next: "
            f"label={review.get('recommended_label')} reason={review.get('recommended_reason')} "
            f"command={review.get('next_command')}"
        )
        if review.get("inspect_first"):
            print(f"  inspect_first: {review.get('inspect_first')}")
    if recommended:
        print(
            "  recommended_next: "
            f"{recommended.get('label')} reason={recommended.get('reason')} {recommended.get('command_line')}"
        )
    if output_request:
        print(f"  output_request: {output_request_text(output_request)}")
        print(f"  output_request_note: {output_request_note(output_request)}")
    if prompt_scope:
        print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
        print(f"  prompt_scope_note: {prompt_scope_note(prompt_scope)}")
    if answer_scope:
        print(f"  answer_scope: {answer_scope_text(answer_scope)}")
        print(f"  answer_scope_note: {answer_scope_note(answer_scope)}")
    if shareable:
        print(f"  shareable: {shareable_summary_text(shareable)}")
    for index, item in enumerate((report.get("next_commands") or []), start=1):
        if not isinstance(item, dict):
            continue
        suffix = " side_effectful=True" if item.get("side_effectful") else ""
        print(f"  next[{index}] {item.get('label')}: {item.get('command_line')}{suffix}")
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    if artifact_report:
        print(
            "  artifacts: "
            f"present={artifact_report.get('present_artifact_count')}/{artifact_report.get('artifact_count')} "
            f"support={artifact_report.get('support_bundle')} "
            f"public_artifact_safe={bool(artifact_report.get('public_artifact_safe'))}"
        )
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
