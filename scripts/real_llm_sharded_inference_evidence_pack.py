#!/usr/bin/env python3
"""Run and package a local CPU-only real tiny-LLM sharded inference proof."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import sharded_inference_evidence_pack as base  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.protocol import (  # noqa: E402
    REAL_LLM_SHARDED_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
)
from crowdtensor.real_llm import DEFAULT_MODEL_ID, DEFAULT_PROMPTS, inspect_real_llm_artifact  # noqa: E402
from crowdtensor.real_llm import BACKEND_CPU as REAL_LLM_BACKEND_CPU  # noqa: E402
from crowdtensor.real_llm import BACKEND_CUDA as REAL_LLM_BACKEND_CUDA  # noqa: E402
from crowdtensor.real_llm import normalize_backend as normalize_real_llm_backend  # noqa: E402
from crowdtensor.real_llm import normalize_partition_mode as normalize_real_llm_partition_mode  # noqa: E402
from product_swarm_mvp_check import parse_prompt_texts_arg, read_prompt_texts_file  # noqa: E402


SCHEMA = "real_llm_sharded_evidence_v1"
OBSERVABILITY_SCHEMA = "real_llm_sharded_observability_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
DEFAULT_ADMIN_TOKEN = "real-llm-sharded-admin"
DEFAULT_OBSERVER_TOKEN = "real-llm-sharded-observer"


def shell_command(parts: list[Any]) -> str:
    return shlex.join([str(part) for part in parts])


def command_entry(
    label: str,
    command: list[Any],
    *,
    reason: str = "",
    side_effectful: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": shell_command(command),
        "public_artifact_safe": True,
        "side_effectful": bool(side_effectful),
    }
    if reason:
        entry["reason"] = reason
    return entry


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


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


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "real_llm_sharded_evidence.md",
        "summary_json": output_dir / "real_llm_sharded_evidence.json",
        "summary_markdown": output_dir / "real_llm_sharded_evidence.md",
        "support_bundle": output_dir / "support_bundle.json",
    }
    return {
        "schema": "real_llm_sharded_artifact_summary_v1",
        "inspect_first": str(paths["inspect_first"]),
        "summary_json": str(paths["summary_json"]),
        "summary_markdown": str(paths["summary_markdown"]),
        "support_bundle": str(paths["support_bundle"]),
        "shareable_paths": [str(path) for path in paths.values()],
        "artifact_count": len(paths),
        "present_artifact_count": sum(1 for path in paths.values() if path.is_file()),
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Open inspect_first first, then support_bundle for diagnostics. "
            "Artifacts contain model/stage readiness, hashes, and counts only."
        ),
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
            "Real LLM sharded evidence records tiny GPT split-inference readiness, token counts, "
            "and hashes only; raw prompts, generated text, token ids, intermediate tensors, and raw model scores are excluded."
        ),
    }


def prompt_scope_summary(args: argparse.Namespace | None = None) -> dict[str, Any]:
    prompt_count = len(DEFAULT_PROMPTS)
    source = "default-validation-prompts"
    if args is not None:
        if str(getattr(args, "prompt_texts_file", "") or ""):
            prompt_count = len(prompt_list_from_args(args))
            source = "local-prompt-file"
        elif str(getattr(args, "prompt_texts", "") or ""):
            prompt_count = len(prompt_list_from_args(args))
            source = "local-inline-prompt-batch"
    return {
        "source": source,
        "prompt_count": prompt_count,
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": source.startswith("local-"),
        "terminal_logs_local_private": True,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": True,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Prompt source and count are reported for operator context; raw prompt text and prompt file paths "
            "are kept out of public JSON, Markdown, and support bundles."
        ),
    }


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "hash-only",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "hash-count-redacted",
        "saved_markdown_display": "hash-count-redacted",
        "json_stdout_display": "hash-count-redacted",
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "The report proves split inference with decoded-token match and generated text hashes; "
            "it does not save or print the raw answer text."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "hash-only",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share real_llm_sharded_evidence.json/md and support_bundle.json; they include readiness, "
            "stage assignment, redaction status, and hashes/counts only."
        ),
    }


def configure_base_module() -> None:
    base.SCHEMA = SCHEMA
    base.OBSERVABILITY_SCHEMA = OBSERVABILITY_SCHEMA
    base.WORKLOAD_TYPE = WORKLOAD_TYPE
    base.DEFAULT_ADMIN_TOKEN = DEFAULT_ADMIN_TOKEN
    base.DEFAULT_OBSERVER_TOKEN = DEFAULT_OBSERVER_TOKEN
    base.create_session = create_session
    base.build_report = build_report
    base.render_markdown = render_markdown


def stage_capability_advertised(task: dict[str, Any], *, required_capability: str, stage_role: str) -> bool:
    capabilities = task.get("capabilities") if isinstance(task.get("capabilities"), dict) else {}
    advertised = capabilities.get("real_llm_sharded_stage_capabilities")
    if advertised is None:
        advertised_set: set[str] = set()
    elif isinstance(advertised, str):
        advertised_set = {advertised}
    else:
        advertised_set = {str(item) for item in advertised}
    role = str(capabilities.get("real_llm_sharded_stage_role") or "both").strip().lower()
    if required_capability in {REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY, REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY}:
        runtime = capabilities.get("real_llm_runtime") if isinstance(capabilities.get("real_llm_runtime"), dict) else {}
        return (
            str(runtime.get("adapter_kind") or "") == REAL_LLM_BACKEND_CUDA
            and (
                required_capability in advertised_set
                or REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY in advertised_set
            )
        )
    return (
        required_capability in advertised_set
        or REAL_LLM_SHARDED_BOTH_CAPABILITY in advertised_set
        or role in {stage_role, "both"}
    )


def create_session(args: argparse.Namespace) -> dict[str, Any]:
    resolved_backend = normalize_real_llm_backend(getattr(args, "real_llm_backend", REAL_LLM_BACKEND_CPU))
    payload: dict[str, Any] = {
        "request_count": args.request_count,
        "workload_type": WORKLOAD_TYPE,
        "backend": "cuda" if resolved_backend == REAL_LLM_BACKEND_CUDA else "cpu",
        "partition_mode": normalize_real_llm_partition_mode(getattr(args, "real_llm_partition_mode", "full")),
    }
    prompt_texts = prompt_list_from_args(args)
    if prompt_texts:
        payload["prompt_texts"] = prompt_texts
    return base.request_json(
        "POST",
        args.base_url,
        "/admin/inference-sessions",
        payload=payload,
        admin_token=args.admin_token,
        timeout=max(30.0, float(args.startup_timeout)),
    )


def prompt_list_from_args(args: argparse.Namespace) -> list[str]:
    prompt_list = getattr(args, "prompt_texts_list", None)
    if isinstance(prompt_list, list) and prompt_list:
        return [str(prompt) for prompt in prompt_list]
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    if prompt_texts_file:
        return read_prompt_texts_file(prompt_texts_file)
    return parse_prompt_texts_arg("", str(getattr(args, "prompt_texts", "") or ""))


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


def real_llm_sharded_command(args: argparse.Namespace, output_dir: Path) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        "real-llm-shard-infer",
        "--output-dir",
        str(output_dir),
        "--port",
        getattr(args, "port", 9880),
        "--request-count",
        getattr(args, "request_count", 1),
        "--max-new-tokens",
        getattr(args, "max_new_tokens", 1),
        "--failure-mode",
        getattr(args, "failure_mode", base.FAILURE_NONE),
        "--stage-mode",
        getattr(args, "stage_mode", "both"),
        "--hf-model-id",
        getattr(args, "hf_model_id", DEFAULT_MODEL_ID),
        "--real-llm-backend",
        getattr(args, "real_llm_backend", REAL_LLM_BACKEND_CPU),
        "--real-llm-partition-mode",
        getattr(args, "real_llm_partition_mode", "full"),
        "--json",
    ]
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", getattr(args, "hf_cache_dir")])
    if getattr(args, "prompt_texts_file", ""):
        command.extend(["--prompt-texts-file", getattr(args, "prompt_texts_file")])
    elif getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", "<local-prompts-redacted>"])
    if getattr(args, "require_distinct_stage_miners", False):
        command.append("--require-distinct-stage-miners")
    return command


def not_completed_items(report: dict[str, Any]) -> list[str]:
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []) if code)
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    artifact = report.get("artifact") if isinstance(report.get("artifact"), dict) else {}
    assignment = report.get("stage_assignment") if isinstance(report.get("stage_assignment"), dict) else {}
    requeue = (
        ((report.get("observability") if isinstance(report.get("observability"), dict) else {}).get("requeue_summary"))
        if isinstance((report.get("observability") if isinstance(report.get("observability"), dict) else {}).get("requeue_summary"), dict)
        else {}
    )
    items: list[tuple[str, Any]] = [
        ("real LLM sharded inference ready", report.get("ok") is True and "real_llm_sharded_ready" in codes),
        ("stage 0 accepted", "stage_0_accepted" in codes),
        ("stage 1 accepted", "stage_1_accepted" in codes),
        ("activation transport ready", "activation_transport_ready" in codes),
        ("real LLM artifact ready", "real_llm_artifact_ready" in codes),
        ("baseline match ready", "baseline_match" in codes),
        ("decoded tokens match ready", "decoded_tokens_match" in codes),
        ("generation complete", "generation_complete" in codes),
        ("generated token count reached target", int(generation.get("generated_token_count") or 0) >= int(generation.get("max_new_tokens") or 1)),
        ("raw generated text redacted", safety.get("generated_text_redacted") is True),
        ("generated token ids redacted", safety.get("generated_token_ids_redacted") is True),
        ("raw activation redacted", safety.get("raw_activation_redacted") is True),
        ("read-only safety boundary present", safety.get("read_only") is True),
        ("redaction safety present", safety.get("redaction_ok") is True),
        ("not production boundary present", safety.get("not_production") is True),
    ]
    if assignment.get("required_distinct_stage_miners"):
        items.extend([
            ("distinct stage miners ready", assignment.get("distinct_stage_miners") is True),
            ("stage assignment valid", assignment.get("stage_assignment_valid") is True),
        ])
    if artifact.get("partition_mode") == "stage_local":
        items.extend([
            ("stage-local partition ready", "stage_local_partition_ready" in codes),
            ("stage 0 partition loaded", "stage0_partition_loaded" in codes),
            ("stage 1 partition loaded", "stage1_partition_loaded" in codes),
            ("partition parameter split valid", "partition_parameter_split_valid" in codes),
        ])
    if artifact.get("backend") == REAL_LLM_BACKEND_CUDA:
        items.extend([
            ("CUDA runtime available", "cuda_runtime_available" in codes),
            ("HF Transformers CUDA backend ready", "hf_transformers_cuda_ready" in codes),
        ])
    if requeue.get("enabled"):
        items.append(("stage requeue ready", "stage_requeue_ready" in codes))
    for process in ((report.get("observability") or {}).get("processes") or []):
        if isinstance(process, dict) and process.get("ok") is not True and process.get("expected_failure") is not True:
            items.append((f"miner process {process.get('miner_id') or 'unknown'} passed", False))
    missing: list[str] = []
    for label, ready in items:
        if ready is not True:
            missing.append(label)
    return missing


def recommended_next_command(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    missing: list[str],
) -> dict[str, Any]:
    if report.get("ok") is True:
        return command_entry(
            "inspect real LLM sharded evidence",
            artifact_command(output_dir, "real_llm_sharded_evidence.md"),
            reason="review_artifacts",
        )
    return command_entry(
        "rerun real LLM sharded inference",
        real_llm_sharded_command(args, output_dir),
        reason="fix_real_llm_sharded_blockers" if missing else "rerun_real_llm_sharded_inference",
        side_effectful=True,
    )


def next_commands(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    commands = [
        command_entry(
            "inspect shareable summary",
            artifact_command(output_dir, "real_llm_sharded_evidence.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect support bundle",
            artifact_command(output_dir, "support_bundle.json"),
            reason="inspect_diagnostics",
        ),
    ]
    if recommended and all(item.get("command_line") != recommended.get("command_line") for item in commands):
        commands.append(dict(recommended))
    refresh = command_entry(
        "refresh real LLM sharded inference",
        real_llm_sharded_command(args, output_dir),
        reason="refresh_real_llm_sharded_inference",
        side_effectful=True,
    )
    if all(item.get("command_line") != refresh.get("command_line") for item in commands):
        commands.append(refresh)
    return commands


def user_status(report: dict[str, Any], *, recommended: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    artifact = report.get("artifact") if isinstance(report.get("artifact"), dict) else {}
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    if report.get("ok") is True and not missing:
        state = "ready"
        headline = "Real tiny-LLM sharded inference evidence is ready."
        next_step = "review_artifacts"
    else:
        state = "blocked"
        headline = "Real tiny-LLM sharded inference evidence needs attention."
        next_step = "fix_blockers"
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "backend": artifact.get("backend") or "unknown",
        "partition_mode": artifact.get("partition_mode") or "unknown",
        "generated_token_count": generation.get("generated_token_count"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "public_artifact_safe": True,
    }


def review_summary(
    report: dict[str, Any],
    *,
    output_dir: Path,
    recommended: dict[str, Any],
    missing: list[str],
) -> dict[str, Any]:
    artifacts = artifact_summary(output_dir)
    ready = bool(report.get("ok") is True and not missing)
    return {
        "schema": "real_llm_sharded_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "ready": ready,
        "next_step": "review_artifacts" if ready else "fix_real_llm_sharded_blockers",
        "inspect_first": artifacts["inspect_first"],
        "support_bundle": artifacts["support_bundle"],
        "recommended_next_command": recommended,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": (report.get("diagnosis_codes") or ["none"])[0],
        "not_completed_count": len(missing),
        "not_completed": list(missing),
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def support_bundle_payload(report: dict[str, Any]) -> dict[str, Any]:
    return support_bundle.sanitize({
        "schema": "real_llm_sharded_support_bundle_v1",
        "generated_at": base.utc_now(),
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "not_completed": report.get("not_completed"),
        "review_summary": report.get("review_summary"),
        "user_status": report.get("user_status"),
        "recommended_next_command": report.get("recommended_next_command"),
        "next_commands": report.get("next_commands"),
        "artifact_summary": report.get("artifact_summary"),
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "session": report.get("session"),
        "artifact": report.get("artifact"),
        "generation": report.get("generation"),
        "stage_assignment": report.get("stage_assignment"),
        "observability": report.get("observability"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
    })


def attach_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = Path.cwd()
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary(args))
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    missing = not_completed_items(report)
    report["not_completed"] = missing
    recommended = recommended_next_command(report, args, output_dir=output_dir, missing=missing)
    report["recommended_next_command"] = recommended
    report["next_commands"] = next_commands(report, args, output_dir=output_dir, recommended=recommended)
    report["user_status"] = user_status(report, recommended=recommended, missing=missing)
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def redacted_token_summary(token_ids: Any) -> dict[str, Any]:
    if not isinstance(token_ids, list):
        return {"count": 0, "redacted": True}
    return {"count": len(token_ids), "redacted": True}


def redacted_request_trace(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        sanitized.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "activation_hash": item.get("activation_hash"),
            "output_hash": item.get("output_hash"),
            "baseline_match": item.get("baseline_match"),
            "generation_step": item.get("generation_step"),
            "max_new_tokens": item.get("max_new_tokens"),
            "generated_token_count": item.get("generated_token_count"),
            "generated_text_hash": item.get("generated_text_hash"),
            "next_token_redacted": "next_token_id" in item or "next_token_text" in item,
        })
    return sanitized


def task_generation_step(task: dict[str, Any]) -> int:
    metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
    validation = task.get("validation") if isinstance(task.get("validation"), dict) else {}
    for source in (metadata, validation):
        try:
            return int(source.get("generation_step", 0))
        except (TypeError, ValueError):
            continue
    return 0


def latest_stage_task(tasks: list[dict[str, Any]], stage_id: int) -> dict[str, Any]:
    candidates = [
        task
        for task in tasks
        if int((task.get("workload_metadata") or {}).get("stage_id", -1)) == stage_id
    ]
    if not candidates:
        return {}
    return max(candidates, key=task_generation_step)


def build_report(
    *,
    args: argparse.Namespace,
    session: dict[str, Any],
    state: dict[str, Any],
    stage_processes: list[dict[str, Any]],
    requeue_summary: dict[str, Any],
    ledger_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    configure_base_module()
    session_id = str(session.get("session_id") or "")
    tasks = base.session_tasks(state, session_id)
    rows = list(ledger_rows) if ledger_rows is not None else (base.admin_results(args, limit=10).get("results") or [])
    stage0 = latest_stage_task(tasks, 0)
    stage1 = latest_stage_task(tasks, 1)
    stage0_validation = stage0.get("validation") or {}
    stage1_validation = stage1.get("validation") or {}
    raw_public = base.safe_state_text(state)
    forbidden_fragments = [
        "activation_results",
        "activation_result",
        "hidden_state",
        "input_ids",
        "logits",
        "sharded_inference_result",
        "inference_results",
        "inference_result",
        "CrowdTensor routes",
        "A miner returns",
    ]
    redaction_failures = [fragment for fragment in forbidden_fragments if fragment in raw_public]
    redaction_ok = not redaction_failures
    read_only = (
        state.get("model", {}).get("global_step") == 0
        and state.get("model_updates") == 0
        and not any(row.get("model_updated") for row in rows)
    )
    stage0_ok = stage0.get("status") == "completed" and stage0_validation.get("code") == "ok"
    stage1_ok = stage1.get("status") == "completed" and stage1_validation.get("code") == "ok"
    baseline_match = bool(stage1_validation.get("baseline_match"))
    decoded_tokens_match = bool(stage1_validation.get("decoded_tokens_match"))
    activation_ready = bool(
        stage0_validation.get("activation_transport_ready")
        and stage1_validation.get("activation_transport_ready")
    )
    stage0_miner_id = str(stage0.get("miner_id") or "")
    stage1_miner_id = str(stage1.get("miner_id") or "")
    distinct_stage_miners = bool(stage0_miner_id and stage1_miner_id and stage0_miner_id != stage1_miner_id)
    max_new_tokens = max(1, int(getattr(args, "max_new_tokens", session.get("max_new_tokens") or 1)))
    generated_token_ids = stage1_validation.get("generated_token_ids")
    generated_token_count = int(
        stage1_validation.get("generated_token_count")
        or (len(generated_token_ids) if isinstance(generated_token_ids, list) else 0)
    )
    if max_new_tokens == 1 and generated_token_count == 0 and decoded_tokens_match:
        generated_token_count = 1
    generation_step = task_generation_step(stage1)
    completed_generation_steps = len([
        task
        for task in tasks
        if int((task.get("workload_metadata") or {}).get("stage_id", -1)) == 1
        and task.get("status") == "completed"
        and (task.get("validation") or {}).get("code") == "ok"
    ])
    generation_complete = bool(generated_token_count >= max_new_tokens and completed_generation_steps >= max_new_tokens)
    multi_token_generation_ready = bool(max_new_tokens > 1 and generation_complete)
    artifact = {
        "schema": session.get("artifact_schema") or stage0_validation.get("artifact_schema") or stage1_validation.get("artifact_schema"),
        "artifact_hash": session.get("artifact_hash") or stage0_validation.get("artifact_hash") or stage1_validation.get("artifact_hash"),
        "model_id": session.get("model_id") or stage0_validation.get("model_id") or stage1_validation.get("model_id"),
        "backend": session.get("backend") or stage0_validation.get("backend") or stage1_validation.get("backend"),
        "partition_mode": (
            session.get("partition_mode")
            or stage0_validation.get("partition_mode")
            or stage1_validation.get("partition_mode")
            or normalize_real_llm_partition_mode(getattr(args, "real_llm_partition_mode", "full"))
        ),
        "split_index": session.get("split_index") or stage0_validation.get("split_index") or stage1_validation.get("split_index"),
        "num_hidden_layers": session.get("num_hidden_layers"),
        "hidden_size": session.get("hidden_size"),
        "loaded": bool(stage0_validation.get("real_llm_artifact_ready") and stage1_validation.get("real_llm_artifact_ready")),
        "stage0_parameter_count": stage0_validation.get("stage_parameter_count"),
        "stage1_parameter_count": stage1_validation.get("stage_parameter_count"),
        "full_model_parameter_count": (
            stage0_validation.get("full_model_parameter_count")
            or stage1_validation.get("full_model_parameter_count")
        ),
        "partition_parameter_split_valid": bool(
            stage0_validation.get("partition_parameter_split_valid")
            and stage1_validation.get("partition_parameter_split_valid")
        ),
        "stage_local_partition_ready": bool(
            stage0_validation.get("stage_local_partition_ready")
            and stage1_validation.get("stage_local_partition_ready")
        ),
    }
    partition_mode = normalize_real_llm_partition_mode(artifact.get("partition_mode"))
    stage0_partition_loaded = bool(stage0_validation.get("stage0_partition_loaded"))
    stage1_partition_loaded = bool(stage1_validation.get("stage1_partition_loaded"))
    partition_ready = bool(
        partition_mode != "stage_local"
        or (
            artifact["stage_local_partition_ready"]
            and artifact["partition_parameter_split_valid"]
            and stage0_partition_loaded
            and stage1_partition_loaded
        )
    )
    stage_assignment_valid = bool(
        stage0_miner_id
        and stage1_miner_id
        and stage_capability_advertised(
            stage0,
            required_capability=(
                REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY
                if artifact.get("backend") == REAL_LLM_BACKEND_CUDA
                else REAL_LLM_SHARDED_STAGE0_CAPABILITY
            ),
            stage_role="stage0",
        )
        and stage_capability_advertised(
            stage1,
            required_capability=(
                REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY
                if artifact.get("backend") == REAL_LLM_BACKEND_CUDA
                else REAL_LLM_SHARDED_STAGE1_CAPABILITY
            ),
            stage_role="stage1",
        )
    )
    codes = []
    if stage0_ok:
        codes.append("stage_0_accepted")
    if stage1_ok:
        codes.append("stage_1_accepted")
    if baseline_match:
        codes.append("baseline_match")
    if decoded_tokens_match:
        codes.append("decoded_tokens_match")
    if generation_complete:
        codes.append("generation_complete")
    if multi_token_generation_ready:
        codes.append("multi_token_generation_ready")
        codes.append("decoded_text_ready")
    if activation_ready:
        codes.append("activation_transport_ready")
    if artifact["loaded"]:
        codes.append("real_llm_artifact_ready")
    if partition_mode == "stage_local":
        if stage0_partition_loaded:
            codes.append("stage0_partition_loaded")
        if stage1_partition_loaded:
            codes.append("stage1_partition_loaded")
        if artifact["partition_parameter_split_valid"]:
            codes.append("partition_parameter_split_valid")
        if artifact["stage_local_partition_ready"]:
            codes.append("stage_local_partition_ready")
        if stage0_validation.get("stage_gpu_memory_reduced") or stage1_validation.get("stage_gpu_memory_reduced"):
            codes.append("stage_gpu_memory_reduced")
        if stage0_validation.get("stage_cpu_partition_ready") and stage1_validation.get("stage_cpu_partition_ready"):
            codes.append("stage_cpu_partition_ready")
    if artifact.get("backend") == REAL_LLM_BACKEND_CUDA:
        codes.extend(["cuda_runtime_available", "hf_transformers_cuda_ready"])
    if distinct_stage_miners:
        codes.append("distinct_stage_miners")
    if stage_assignment_valid:
        codes.append("stage_assignment_valid")
    if requeue_summary.get("enabled") and requeue_summary.get("rescued_result"):
        codes.append("stage_requeue_ready")
    distinct_requirement_met = not getattr(args, "require_distinct_stage_miners", False) or distinct_stage_miners
    if (
        stage0_ok
        and stage1_ok
        and baseline_match
        and decoded_tokens_match
        and generation_complete
        and activation_ready
        and artifact["loaded"]
        and partition_ready
        and read_only
        and redaction_ok
        and distinct_requirement_met
    ):
        codes.append("real_llm_sharded_ready")
    else:
        if not stage0_ok:
            codes.append("stage_0_missing")
        if not stage1_ok:
            codes.append("stage_1_missing")
        if not baseline_match:
            codes.append("baseline_mismatch")
        if not decoded_tokens_match:
            codes.append("decoded_tokens_mismatch")
        if not generation_complete:
            codes.append("multi_token_generation_incomplete")
        if not activation_ready:
            codes.append("activation_transport_failed")
        if not artifact["loaded"]:
            codes.append("real_llm_artifact_missing")
        if not partition_ready:
            codes.append("stage_local_partition_missing")
        if not distinct_requirement_met:
            codes.append("distinct_stage_miners_missing")
        if not read_only or not redaction_ok:
            codes.append("safety_failed")
    report = {
        "schema": SCHEMA,
        "generated_at": base.utc_now(),
        "ok": "real_llm_sharded_ready" in codes and (
            not requeue_summary.get("enabled") or "stage_requeue_ready" in codes
        ),
        "base_url": args.base_url,
        "workload_type": WORKLOAD_TYPE,
        "session": {
            "schema": session.get("schema"),
            "session_id": session_id,
            "stage_count": session.get("stage_count"),
            "stage_0_task_id": stage0.get("task_id") or session.get("stage_0_task_id"),
            "stage_1_task_id": stage1.get("task_id") or session.get("stage_1_task_id"),
            "request_count": session.get("request_count"),
            "artifact_hash": session.get("artifact_hash"),
            "model_id": session.get("model_id"),
            "prompt_request_count": session.get("prompt_request_count"),
            "partition_mode": artifact.get("partition_mode"),
            "max_new_tokens": max_new_tokens,
            "final_generation_step": generation_step,
        },
        "artifact": artifact,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "completed_generation_steps": completed_generation_steps,
            "final_generation_step": generation_step,
            "generated_token_count": generated_token_count,
            "generated_token_summary": redacted_token_summary(stage1_validation.get("generated_token_ids")),
            "generated_text_hash": stage1_validation.get("generated_text_hash"),
            "generated_text_redacted": bool(stage1_validation.get("generated_text") is not None),
            "multi_token_generation_ready": multi_token_generation_ready,
        },
        "stage_summary": {
            "stage_0": {
                "task_id": stage0.get("task_id"),
                "miner_id": stage0.get("miner_id"),
                "attempt": stage0.get("attempt"),
                "accepted": stage0_ok,
                "generation_step": task_generation_step(stage0),
                "activation_count": stage0_validation.get("activation_count"),
                "activation_bytes": stage0_validation.get("activation_bytes"),
                "activation_hashes": stage0_validation.get("activation_hashes"),
                "artifact_hash": stage0_validation.get("artifact_hash"),
                "backend": stage0_validation.get("backend"),
                "partition_mode": stage0_validation.get("partition_mode"),
                "stage_layer_range": stage0_validation.get("stage_layer_range"),
                "stage_parameter_count": stage0_validation.get("stage_parameter_count"),
                "full_model_parameter_count": stage0_validation.get("full_model_parameter_count"),
                "stage_parameter_fraction": stage0_validation.get("stage_parameter_fraction"),
                "partition_parameter_split_valid": stage0_validation.get("partition_parameter_split_valid"),
                "stage_local_partition_ready": stage0_validation.get("stage_local_partition_ready"),
                "stage0_partition_loaded": stage0_partition_loaded,
                "stage_gpu_memory_reduced": stage0_validation.get("stage_gpu_memory_reduced"),
                "stage_cpu_partition_ready": stage0_validation.get("stage_cpu_partition_ready"),
                "elapsed_ms": (stage0.get("metrics") or {}).get("elapsed_ms"),
            },
            "stage_1": {
                "task_id": stage1.get("task_id"),
                "miner_id": stage1.get("miner_id"),
                "attempt": stage1.get("attempt"),
                "accepted": stage1_ok,
                "generation_step": generation_step,
                "baseline_match": baseline_match,
                "decoded_tokens_match": decoded_tokens_match,
                "request_count": stage1_validation.get("request_count"),
                "generated_token_summary": redacted_token_summary(stage1_validation.get("generated_token_ids")),
                "generated_text_redacted": bool(stage1_validation.get("generated_text") is not None),
                "request_trace": redacted_request_trace(stage1_validation.get("request_trace")),
                "artifact_hash": stage1_validation.get("artifact_hash"),
                "backend": stage1_validation.get("backend"),
                "partition_mode": stage1_validation.get("partition_mode"),
                "baseline_device": stage1_validation.get("baseline_device"),
                "stage_layer_range": stage1_validation.get("stage_layer_range"),
                "stage_parameter_count": stage1_validation.get("stage_parameter_count"),
                "full_model_parameter_count": stage1_validation.get("full_model_parameter_count"),
                "stage_parameter_fraction": stage1_validation.get("stage_parameter_fraction"),
                "partition_parameter_split_valid": stage1_validation.get("partition_parameter_split_valid"),
                "stage_local_partition_ready": stage1_validation.get("stage_local_partition_ready"),
                "stage1_partition_loaded": stage1_partition_loaded,
                "stage_gpu_memory_reduced": stage1_validation.get("stage_gpu_memory_reduced"),
                "stage_cpu_partition_ready": stage1_validation.get("stage_cpu_partition_ready"),
                "elapsed_ms": (stage1.get("metrics") or {}).get("elapsed_ms"),
            },
        },
        "stage_assignment": {
            "mode": getattr(args, "stage_mode", "both"),
            "required_distinct_stage_miners": bool(getattr(args, "require_distinct_stage_miners", False)),
            "stage0_miner_id": stage0_miner_id,
            "stage1_miner_id": stage1_miner_id,
            "distinct_stage_miners": distinct_stage_miners,
            "stage_assignment_valid": stage_assignment_valid,
        },
        "observability": {
            "schema": OBSERVABILITY_SCHEMA,
            "stage_count": len(tasks),
            "generation_task_count": len(tasks),
            "accepted_ledger_rows": len(rows),
            "miner_ids": sorted({str(task.get("miner_id")) for task in tasks if task.get("miner_id")}),
            "processes": stage_processes,
            "requeue_summary": requeue_summary,
        },
        "diagnosis_codes": sorted(set(codes)),
        "safety": {
            "read_only": read_only,
            "redaction_ok": redaction_ok,
            "redaction_failures": redaction_failures,
            "raw_activation_redacted": "activation_results" not in raw_public and "hidden_state" not in raw_public,
            "generated_text_redacted": True,
            "generated_token_ids_redacted": True,
            "not_production": True,
            "gpu_backend_selected": artifact.get("backend") == REAL_LLM_BACKEND_CUDA,
            "stage_local_partition": partition_mode == "stage_local",
        },
        "limitations": [
            "Tiny Hugging Face GPT two-stage pipeline; optional CUDA backend is explicit and not production Swarm Inference",
            "Downloads or uses an operator-provided HF cache; not GGUF/llama.cpp, GPU pooling marketplace, or large-model serving",
            "Not P2P routing, NAT traversal, payments, or arbitrary public prompt serving",
        ],
    }
    return attach_user_guidance(report, args)


def render_markdown(report: dict[str, Any]) -> str:
    session = report.get("session") or {}
    stage = report.get("stage_summary") or {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    artifacts = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Real Tiny-LLM Sharded Inference Evidence",
        "",
        f"Schema: `{report.get('schema')}`",
        f"OK: `{report.get('ok')}`",
        f"Session: `{session.get('session_id')}`",
        f"Workload: `{report.get('workload_type')}`",
        f"Model: `{session.get('model_id')}`",
        f"Partition mode: `{session.get('partition_mode')}`",
        "",
        "## Review",
        "",
        f"- status: `{user.get('state')}` {user.get('headline') or ''}",
        f"- state: `{review.get('state')}`",
        f"- next step: `{review.get('next_step')}`",
        f"- inspect first: `{review.get('inspect_first')}`",
        f"- support bundle: `{review.get('support_bundle')}`",
        f"- recommended next: `{recommended.get('command_line') or 'none'}`",
        f"- public artifact safe: `{review.get('public_artifact_safe')}`",
        "",
        "## What To Do Next",
        "",
    ]
    for item in report.get("next_commands") or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('label')}: `{item.get('command_line')}`")
    if not report.get("next_commands"):
        lines.append("- none")
    lines.extend([
        "",
        "## Artifact Summary",
        "",
        f"- inspect first: `{artifacts.get('inspect_first')}`",
        f"- summary JSON: `{artifacts.get('summary_json')}`",
        f"- support bundle: `{artifacts.get('support_bundle')}`",
        f"- present artifacts: `{artifacts.get('present_artifact_count')}/{artifacts.get('artifact_count')}`",
        "",
        "## Not Completed",
        "",
    ])
    for item in report.get("not_completed") or []:
        lines.append(f"- {item}")
    if not report.get("not_completed"):
        lines.append("- none")
    lines.extend([
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- prompt scope: `{prompt_scope_text(prompt_scope)}`",
        f"- prompt scope note: {prompt_scope.get('summary') or 'Public artifacts exclude raw prompt text.'}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope.get('summary') or 'Public artifacts contain no local answer transcript or raw generated text.'}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []),
        "",
        "## Stages",
        "",
        f"- Stage 0: task `{(stage.get('stage_0') or {}).get('task_id')}`, miner `{(stage.get('stage_0') or {}).get('miner_id')}`, activations `{(stage.get('stage_0') or {}).get('activation_count')}`",
        f"- Stage 1: task `{(stage.get('stage_1') or {}).get('task_id')}`, miner `{(stage.get('stage_1') or {}).get('miner_id')}`, baseline match `{(stage.get('stage_1') or {}).get('baseline_match')}`",
        "",
        "CPU-only tiny Hugging Face GPT two-stage pipeline; not production Swarm Inference, GPU pooling, P2P routing, or GGUF/llama.cpp serving.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU-only real tiny-LLM sharded inference evidence.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--prompt-texts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--prompt-texts-file", default="", help="newline-delimited bounded batch of up to 4 prompts")
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--real-llm-backend",
        choices=["hf_transformers_cpu", "hf_transformers_cuda", "cpu", "cuda", "auto"],
        default=REAL_LLM_BACKEND_CPU,
    )
    parser.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--failure-mode", choices=sorted(base.FAILURE_MODES), default=base.FAILURE_NONE)
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--miner-prefix", default="real-llm-shard-miner")
    parser.add_argument("--invite-token-prefix", default="real-llm-sharded-token")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--compute-seconds", type=float, default=0.0)
    parser.add_argument("--victim-compute-seconds", type=float, default=15.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.2)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--miner-timeout", type=float, default=120.0)
    parser.add_argument("--stage-timeout", type=float, default=30.0)
    parser.add_argument("--claim-observe-timeout", type=float, default=10.0)
    parser.add_argument("--requeue-timeout", type=float, default=20.0)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    if args.failure_mode != base.FAILURE_NONE and args.max_new_tokens != 1:
        raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    prompt_texts_explicit = "--prompt-texts" in raw_argv or any(item.startswith("--prompt-texts=") for item in raw_argv)
    if args.prompt_texts_file and prompt_texts_explicit:
        raise SystemExit("real_llm_sharded_inference_evidence accepts either --prompt-texts or --prompt-texts-file, not both")
    try:
        if args.prompt_texts_file:
            args.prompt_texts_list = read_prompt_texts_file(args.prompt_texts_file)
            args.prompt_texts = ""
        else:
            args.prompt_texts_list = parse_prompt_texts_arg("", args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.requeue_timeout <= args.lease_seconds:
        args.requeue_timeout = args.lease_seconds + 5.0
    if args.victim_compute_seconds <= args.lease_seconds:
        args.victim_compute_seconds = args.lease_seconds + 3.0
    args.base_url = f"http://{args.host}:{args.port}"
    args.real_llm_backend = normalize_real_llm_backend(args.real_llm_backend)
    args.real_llm_partition_mode = normalize_real_llm_partition_mode(args.real_llm_partition_mode)
    inspect_real_llm_artifact(
        model_id=args.hf_model_id,
        cache_dir=args.hf_cache_dir,
        backend=args.real_llm_backend,
    )
    return args


configure_base_module()


def run_evidence(args: argparse.Namespace) -> dict[str, Any]:
    configure_base_module()
    if args.stage_mode == "split":
        args.require_distinct_stage_miners = True
    return base.run_evidence(args)


def main() -> None:
    try:
        args = parse_args()
        report = run_evidence(args)
        output_dir = Path(args.json_out).resolve().parent if args.json_out else Path.cwd()
        report = attach_user_guidance(report, args, output_dir=output_dir)
        report.setdefault("artifacts", {})
        report["artifacts"]["real_llm_sharded_evidence_json"] = artifact_entry(
            Path(args.json_out) if args.json_out else output_dir / "real_llm_sharded_evidence.json",
            output_dir,
            kind="real_llm_sharded_evidence",
            schema=SCHEMA,
            ok=report.get("ok"),
        )
        report["artifacts"]["real_llm_sharded_evidence_markdown"] = artifact_entry(
            Path(args.markdown_out) if args.markdown_out else output_dir / "real_llm_sharded_evidence.md",
            output_dir,
            kind="real_llm_sharded_evidence_markdown",
        )
        report["artifacts"]["support_bundle_json"] = artifact_entry(
            output_dir / "support_bundle.json",
            output_dir,
            kind="real_llm_sharded_support_bundle",
            schema="real_llm_sharded_support_bundle_v1",
        )
        base.write_json(report, args.json_out)
        if args.markdown_out:
            output = Path(args.markdown_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_markdown(report), encoding="utf-8")
        support_path = output_dir / "support_bundle.json"
        base.write_json(support_bundle_payload(report), str(support_path))
        if args.json_out:
            report["artifacts"]["real_llm_sharded_evidence_json"]["present"] = Path(args.json_out).is_file()
        if args.markdown_out:
            report["artifacts"]["real_llm_sharded_evidence_markdown"]["present"] = Path(args.markdown_out).is_file()
        report["artifacts"]["support_bundle_json"]["present"] = support_path.is_file()
        report["artifact_summary"] = artifact_summary(output_dir)
        base.write_json(support_bundle_payload(report), str(support_path))
        base.write_json(report, args.json_out)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
