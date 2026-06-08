#!/usr/bin/env python3
"""Run and package a local CPU-only micro-LLM pipeline-sharded inference proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import sharded_inference_evidence_pack as base  # noqa: E402
from crowdtensor.micro_llm_artifact import DEFAULT_PROMPTS, inspect_micro_llm_artifact  # noqa: E402
from crowdtensor.protocol import (  # noqa: E402
    MICRO_LLM_SHARDED_BOTH_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
)


SCHEMA = "micro_llm_sharded_evidence_v1"
OBSERVABILITY_SCHEMA = "micro_llm_sharded_observability_v1"
WORKLOAD_TYPE = "micro_llm_sharded_infer"
DEFAULT_ADMIN_TOKEN = "micro-llm-sharded-admin"
DEFAULT_OBSERVER_TOKEN = "micro-llm-sharded-observer"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"


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
    advertised = capabilities.get("micro_llm_sharded_stage_capabilities")
    if advertised is None:
        advertised_set: set[str] = set()
    elif isinstance(advertised, str):
        advertised_set = {advertised}
    else:
        advertised_set = {str(item) for item in advertised}
    role = str(capabilities.get("micro_llm_sharded_stage_role") or "both").strip().lower()
    return (
        required_capability in advertised_set
        or MICRO_LLM_SHARDED_BOTH_CAPABILITY in advertised_set
        or role in {stage_role, "both"}
    )


def create_session(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_count": args.request_count,
        "decode_steps": args.decode_steps,
        "workload_type": WORKLOAD_TYPE,
    }
    prompt_texts = [item for item in str(getattr(args, "prompt_texts", "") or "").split(",") if item]
    if prompt_texts:
        payload["prompt_texts"] = prompt_texts
    return base.request_json(
        "POST",
        args.base_url,
        "/admin/inference-sessions",
        payload=payload,
        admin_token=args.admin_token,
    )


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def redacted_micro_request_trace(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        output_text = item.get("output_text") or item.get("generated_text")
        token_ids = item.get("generated_token_ids")
        sanitized.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "activation_hash": item.get("activation_hash"),
            "output_hash": item.get("output_hash") or (stable_hash(output_text) if output_text else None),
            "baseline_match": item.get("baseline_match"),
            "decoded_tokens_match": item.get("decoded_tokens_match"),
            "generation_step": item.get("generation_step"),
            "generated_token_count": item.get("generated_token_count") or (len(token_ids) if isinstance(token_ids, list) else None),
            "generated_text_hash": stable_hash(output_text) if output_text else item.get("generated_text_hash"),
            "generated_text_redacted": bool(output_text),
            "generated_token_ids_redacted": isinstance(token_ids, list),
        })
    return sanitized


def generated_text_hash(value: Any) -> str | None:
    if value is None:
        return None
    return stable_hash(str(value))


def micro_artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "micro_llm_sharded_evidence.md",
        "summary_json": output_dir / "micro_llm_sharded_evidence.json",
        "summary_markdown": output_dir / "micro_llm_sharded_evidence.md",
        "support_bundle": output_dir / "support_bundle.json",
    }
    return {
        "schema": "micro_llm_sharded_artifact_summary_v1",
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
            "Artifacts contain stage readiness, decoded-token match, hashes, and counts only."
        ),
    }


def micro_prompt_scope_summary(args: argparse.Namespace | None = None, report: dict[str, Any] | None = None) -> dict[str, Any]:
    session = report.get("session") if isinstance(report, dict) and isinstance(report.get("session"), dict) else {}
    prompt_count = session.get("prompt_request_count") or session.get("request_count")
    source = "default-micro-llm-prompts"
    if args is not None and str(getattr(args, "prompt_texts", "") or ""):
        source = "local-inline-prompt-batch"
        prompt_count = len([item for item in str(getattr(args, "prompt_texts", "") or "").split(",") if item])
    try:
        prompt_count = int(prompt_count or 0)
    except (TypeError, ValueError):
        prompt_count = 0
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
            "Prompt source and count are reported for operator context; raw prompt text is kept out of public artifacts."
        ),
    }


def micro_output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Micro-LLM sharded evidence records decoded-token match, counts, and hashes only; "
            "raw prompts, generated text, token ids, and intermediate tensors are excluded."
        ),
    }


def micro_answer_scope_summary() -> dict[str, Any]:
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
            "The report proves micro-LLM split inference with decoded-token match and hashes; "
            "it does not save or print the raw answer text."
        ),
    }


def micro_shareable_summary() -> dict[str, Any]:
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
            "Share micro_llm_sharded_evidence.json/md and support_bundle.json; they include readiness, "
            "stage assignment, redaction status, and hashes/counts only."
        ),
    }


def micro_not_completed_items(report: dict[str, Any]) -> list[str]:
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []) if code)
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    artifact = report.get("artifact") if isinstance(report.get("artifact"), dict) else {}
    assignment = report.get("stage_assignment") if isinstance(report.get("stage_assignment"), dict) else {}
    requeue = (
        ((report.get("observability") if isinstance(report.get("observability"), dict) else {}).get("requeue_summary"))
        if isinstance((report.get("observability") if isinstance(report.get("observability"), dict) else {}).get("requeue_summary"), dict)
        else {}
    )
    items: list[tuple[str, Any]] = [
        ("micro-LLM sharded inference ready", report.get("ok") is True and "micro_llm_sharded_ready" in codes),
        ("stage 0 accepted", "stage_0_accepted" in codes),
        ("stage 1 accepted", "stage_1_accepted" in codes),
        ("activation transport ready", "activation_transport_ready" in codes),
        ("baseline match ready", "baseline_match" in codes),
        ("decoded tokens match ready", "decoded_tokens_match" in codes),
        ("raw generated text redacted", safety.get("generated_text_redacted") is True),
        ("generated token ids redacted", safety.get("generated_token_ids_redacted") is True),
        ("raw activation redacted", safety.get("raw_activation_redacted") is True),
        ("read-only safety boundary present", safety.get("read_only") is True),
        ("redaction safety present", safety.get("redaction_ok") is True),
        ("not production boundary present", safety.get("not_production") is True),
    ]
    if artifact.get("schema"):
        items.append(("micro-LLM artifact ready", "micro_llm_artifact_ready" in codes or artifact.get("loaded") is True))
    if assignment.get("required_distinct_stage_miners"):
        items.extend([
            ("distinct stage miners ready", assignment.get("distinct_stage_miners") is True),
            ("stage assignment valid", assignment.get("stage_assignment_valid") is True),
        ])
    if requeue.get("enabled"):
        items.append(("stage requeue ready", "stage_requeue_ready" in codes))
    for process in ((report.get("observability") or {}).get("processes") or []):
        if isinstance(process, dict) and process.get("ok") is not True and process.get("expected_failure") is not True:
            items.append((f"miner process {process.get('miner_id') or 'unknown'} passed", False))
    return [label for label, ready in items if ready is not True]


def micro_command_entry(label: str, command: list[Any], *, reason: str = "", side_effectful: bool = False) -> dict[str, Any]:
    entry = base.command_entry(label, command, reason=reason, side_effectful=side_effectful)
    return entry


def micro_sharded_command(args: argparse.Namespace, output_dir: Path) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        "micro-llm-shard-infer",
        "--output-dir",
        str(output_dir),
        "--port",
        getattr(args, "port", 9860),
        "--request-count",
        getattr(args, "request_count", 4),
        "--decode-steps",
        getattr(args, "decode_steps", 4),
        "--failure-mode",
        getattr(args, "failure_mode", base.FAILURE_NONE),
        "--stage-mode",
        getattr(args, "stage_mode", "both"),
        "--json",
    ]
    if getattr(args, "micro_llm_artifact", ""):
        command.extend(["--micro-llm-artifact", getattr(args, "micro_llm_artifact")])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", "<local-prompts-redacted>"])
    if getattr(args, "require_distinct_stage_miners", False):
        command.append("--require-distinct-stage-miners")
    return command


def micro_recommended_next_command(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    missing: list[str],
) -> dict[str, Any]:
    if report.get("ok") is True:
        return micro_command_entry(
            "inspect micro-LLM sharded evidence",
            base.artifact_command(output_dir, "micro_llm_sharded_evidence.md"),
            reason="review_artifacts",
        )
    return micro_command_entry(
        "rerun micro-LLM sharded inference",
        micro_sharded_command(args, output_dir),
        reason="fix_micro_llm_sharded_blockers" if missing else "rerun_micro_llm_sharded_inference",
        side_effectful=True,
    )


def micro_next_commands(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    commands = [
        micro_command_entry(
            "inspect shareable summary",
            base.artifact_command(output_dir, "micro_llm_sharded_evidence.md"),
            reason="review_artifacts",
        ),
        micro_command_entry(
            "inspect support bundle",
            base.artifact_command(output_dir, "support_bundle.json"),
            reason="inspect_diagnostics",
        ),
    ]
    if recommended and all(item.get("command_line") != recommended.get("command_line") for item in commands):
        commands.append(dict(recommended))
    refresh = micro_command_entry(
        "refresh micro-LLM sharded inference",
        micro_sharded_command(args, output_dir),
        reason="refresh_micro_llm_sharded_inference",
        side_effectful=True,
    )
    if all(item.get("command_line") != refresh.get("command_line") for item in commands):
        commands.append(refresh)
    return commands


def micro_user_status(report: dict[str, Any], *, recommended: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    if report.get("ok") is True and not missing:
        state = "ready"
        headline = "Micro-LLM sharded inference evidence is ready."
        next_step = "review_artifacts"
    else:
        state = "blocked"
        headline = "Micro-LLM sharded inference evidence needs attention."
        next_step = "fix_blockers"
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "generated_token_count": generation.get("generated_token_count"),
        "decode_steps": generation.get("decode_steps"),
        "public_artifact_safe": True,
    }


def micro_review_summary(
    report: dict[str, Any],
    *,
    output_dir: Path,
    recommended: dict[str, Any],
    missing: list[str],
) -> dict[str, Any]:
    artifacts = micro_artifact_summary(output_dir)
    ready = bool(report.get("ok") is True and not missing)
    return {
        "schema": "micro_llm_sharded_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "ready": ready,
        "next_step": "review_artifacts" if ready else "fix_micro_llm_sharded_blockers",
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


def micro_support_bundle_payload(report: dict[str, Any]) -> dict[str, Any]:
    return base.support_bundle.sanitize({
        "schema": "micro_llm_sharded_support_bundle_v1",
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


def attach_micro_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = Path.cwd()
    report["output_request"] = micro_output_request_summary()
    report["prompt_scope"] = micro_prompt_scope_summary(args, report)
    report["answer_scope"] = micro_answer_scope_summary()
    report["shareable_summary"] = micro_shareable_summary()
    missing = micro_not_completed_items(report)
    report["not_completed"] = missing
    recommended = micro_recommended_next_command(report, args, output_dir=output_dir, missing=missing)
    report["recommended_next_command"] = recommended
    report["next_commands"] = micro_next_commands(report, args, output_dir=output_dir, recommended=recommended)
    report["user_status"] = micro_user_status(report, recommended=recommended, missing=missing)
    report["review_summary"] = micro_review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = micro_artifact_summary(output_dir)
    return report


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
    stage_tasks = {int((task.get("workload_metadata") or {}).get("stage_id", -1)): task for task in tasks}
    rows = list(ledger_rows) if ledger_rows is not None else (base.admin_results(args, limit=10).get("results") or [])
    stage0 = stage_tasks.get(0, {})
    stage1 = stage_tasks.get(1, {})
    stage0_validation = stage0.get("validation") or {}
    stage1_validation = stage1.get("validation") or {}
    raw_public = base.safe_state_text(state)
    redaction_ok = all(fragment not in raw_public for fragment in [
        "activation_results",
        "activation_result",
        "hidden_state",
        "logits",
        "sharded_inference_result",
        "inference_results",
        "inference_result",
        '"lease_token": "<redacted>"',
    ])
    micro = state.get("model", {}).get("micro_transformer", {})
    artifact_summary = {
        "schema": micro.get("artifact_schema") or "",
        "artifact_id": micro.get("artifact_id") or "",
        "artifact_version": micro.get("artifact_version"),
        "artifact_hash": micro.get("artifact_hash") or stage1_validation.get("artifact_hash") or stage0_validation.get("artifact_hash"),
        "tokenizer_schema": micro.get("tokenizer_schema") or "",
        "loaded": bool(micro.get("artifact_schema") and micro.get("artifact_hash")),
    }
    read_only = (
        micro.get("version") == 0
        and micro.get("optimizer_step") == 0
        and state.get("model", {}).get("global_step") == 0
        and state.get("model_updates") == 0
        and not any(row.get("model_updated") or row.get("micro_transformer_updated") for row in rows)
    )
    stage0_ok = stage0.get("status") == "completed" and stage0_validation.get("code") == "ok"
    stage1_ok = stage1.get("status") == "completed" and stage1_validation.get("code") == "ok"
    baseline_match = bool(stage1_validation.get("baseline_match"))
    decoded_tokens_match = bool(stage1_validation.get("decoded_tokens_match"))
    activation_ready = bool(
        stage0_validation.get("activation_transport_ready")
        and stage1_validation.get("activation_transport_ready")
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
    if activation_ready:
        codes.append("activation_transport_ready")
    if requeue_summary.get("enabled") and requeue_summary.get("rescued_result"):
        codes.append("stage_requeue_ready")
    stage0_miner_id = str(stage0.get("miner_id") or "")
    stage1_miner_id = str(stage1.get("miner_id") or "")
    distinct_stage_miners = bool(stage0_miner_id and stage1_miner_id and stage0_miner_id != stage1_miner_id)
    stage_assignment_valid = bool(
        stage0_miner_id
        and stage1_miner_id
        and stage_capability_advertised(
            stage0,
            required_capability=MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
            stage_role="stage0",
        )
        and stage_capability_advertised(
            stage1,
            required_capability=MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
            stage_role="stage1",
        )
    )
    if distinct_stage_miners:
        codes.append("distinct_stage_miners")
    if stage_assignment_valid:
        codes.append("stage_assignment_valid")
    if artifact_summary["loaded"]:
        codes.append("artifact_loaded")
    if artifact_summary["schema"] == "micro_llm_artifact_v1":
        codes.append("micro_llm_artifact_ready")
    generated_text = stage1_validation.get("generated_text")
    generated_token_count = stage1_validation.get("generated_token_count")
    if generated_token_count is None and isinstance(generated_text, str):
        generated_token_count = len(generated_text.split())
    distinct_requirement_met = not getattr(args, "require_distinct_stage_miners", False) or distinct_stage_miners
    if stage0_ok and stage1_ok and baseline_match and decoded_tokens_match and activation_ready and read_only and redaction_ok:
        if distinct_requirement_met:
            codes.append("micro_llm_sharded_ready")
        else:
            codes.append("distinct_stage_miners_missing")
    else:
        if not stage0_ok:
            codes.append("stage_0_missing")
        if not stage1_ok:
            codes.append("stage_1_missing")
        if not baseline_match:
            codes.append("baseline_mismatch")
        if not decoded_tokens_match:
            codes.append("decoded_tokens_mismatch")
        if not activation_ready:
            codes.append("activation_transport_failed")
        if not read_only or not redaction_ok:
            codes.append("safety_failed")
    report = {
        "schema": SCHEMA,
        "generated_at": base.utc_now(),
        "ok": "micro_llm_sharded_ready" in codes and (
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
            "decode_steps": session.get("decode_steps"),
            "artifact_hash": session.get("artifact_hash"),
            "artifact_id": session.get("artifact_id"),
            "prompt_request_count": session.get("prompt_request_count"),
        },
        "artifact": artifact_summary,
        "generation": {
            "decode_steps": session.get("decode_steps"),
            "generated_token_count": generated_token_count,
            "generated_text_hash": generated_text_hash(generated_text),
            "generated_text_redacted": True,
            "generated_token_ids_redacted": True,
            "decoded_tokens_match": decoded_tokens_match,
        },
        "stage_summary": {
            "stage_0": {
                "task_id": stage0.get("task_id"),
                "miner_id": stage0.get("miner_id"),
                "attempt": stage0.get("attempt"),
                "accepted": stage0_ok,
                "activation_count": stage0_validation.get("activation_count"),
                "activation_bytes": stage0_validation.get("activation_bytes"),
                "activation_hashes": stage0_validation.get("activation_hashes"),
                "artifact_hash": stage0_validation.get("artifact_hash"),
                "elapsed_ms": (stage0.get("metrics") or {}).get("elapsed_ms"),
            },
            "stage_1": {
                "task_id": stage1.get("task_id"),
                "miner_id": stage1.get("miner_id"),
                "attempt": stage1.get("attempt"),
                "accepted": stage1_ok,
                "baseline_match": baseline_match,
                "decoded_tokens_match": decoded_tokens_match,
                "request_count": stage1_validation.get("request_count"),
                "decode_steps": stage1_validation.get("decode_steps"),
                "generated_token_count": generated_token_count,
                "generated_text_hash": generated_text_hash(generated_text),
                "generated_text_redacted": True,
                "request_trace": redacted_micro_request_trace(stage1_validation.get("request_trace")),
                "artifact_hash": stage1_validation.get("artifact_hash"),
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
            "accepted_ledger_rows": len(rows),
            "miner_ids": sorted({str(task.get("miner_id")) for task in tasks if task.get("miner_id")}),
            "processes": stage_processes,
            "requeue_summary": requeue_summary,
        },
        "diagnosis_codes": sorted(set(codes)),
        "safety": {
            "read_only": read_only,
            "redaction_ok": redaction_ok,
            "raw_activation_redacted": "activation_results" not in raw_public and "hidden_state" not in raw_public,
            "generated_text_redacted": True,
            "generated_token_ids_redacted": True,
            "not_production": True,
        },
        "limitations": [
            "CPU-only deterministic micro-LLM two-stage pipeline; not production Swarm Inference",
            "Not GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    }
    return attach_micro_user_guidance(report, args)


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
        "# CrowdTensor Micro-LLM Sharded Inference Evidence",
        "",
        f"Schema: `{report.get('schema')}`",
        f"OK: `{report.get('ok')}`",
        f"Session: `{session.get('session_id')}`",
        f"Workload: `{report.get('workload_type')}`",
        f"Decode steps: `{session.get('decode_steps')}`",
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
        f"- prompt scope: `{base.prompt_scope_text(prompt_scope)}`",
        f"- prompt scope note: {prompt_scope.get('summary') or 'Public artifacts exclude raw prompt text.'}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope.get('summary') or 'Public artifacts contain no raw generated text.'}",
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
        f"- Stage 1: task `{(stage.get('stage_1') or {}).get('task_id')}`, miner `{(stage.get('stage_1') or {}).get('miner_id')}`, baseline match `{(stage.get('stage_1') or {}).get('baseline_match')}`, decoded tokens match `{(stage.get('stage_1') or {}).get('decoded_tokens_match')}`",
        "",
        "CPU-only deterministic micro-LLM two-stage pipeline; not production Swarm Inference, GPU pooling, P2P routing, or GGUF/llama.cpp serving.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU-only micro-LLM sharded inference evidence.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9860)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--micro-llm-artifact", default="")
    parser.add_argument("--prompt-texts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--failure-mode", choices=sorted(base.FAILURE_MODES), default=base.FAILURE_NONE)
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--miner-prefix", default="micro-llm-shard-miner")
    parser.add_argument("--invite-token-prefix", default="micro-llm-sharded-token")
    parser.add_argument("--lease-seconds", type=float, default=5.0)
    parser.add_argument("--compute-seconds", type=float, default=0.0)
    parser.add_argument("--victim-compute-seconds", type=float, default=8.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--stage-timeout", type=float, default=10.0)
    parser.add_argument("--claim-observe-timeout", type=float, default=5.0)
    parser.add_argument("--requeue-timeout", type=float, default=10.0)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    if args.decode_steps < 1 or args.decode_steps > 4:
        raise SystemExit("--decode-steps must be between 1 and 4")
    if args.requeue_timeout <= args.lease_seconds:
        args.requeue_timeout = args.lease_seconds + 5.0
    if args.victim_compute_seconds <= args.lease_seconds:
        args.victim_compute_seconds = args.lease_seconds + 3.0
    args.base_url = f"http://{args.host}:{args.port}"
    if args.micro_llm_artifact:
        inspect_micro_llm_artifact(args.micro_llm_artifact)
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
        report = attach_micro_user_guidance(report, args, output_dir=output_dir)
        report.setdefault("artifacts", {})
        report["artifacts"]["micro_llm_sharded_evidence_json"] = base.artifact_entry(
            Path(args.json_out) if args.json_out else output_dir / "micro_llm_sharded_evidence.json",
            output_dir,
            kind="micro_llm_sharded_evidence",
            schema=SCHEMA,
            ok=report.get("ok"),
        )
        report["artifacts"]["micro_llm_sharded_evidence_markdown"] = base.artifact_entry(
            Path(args.markdown_out) if args.markdown_out else output_dir / "micro_llm_sharded_evidence.md",
            output_dir,
            kind="micro_llm_sharded_evidence_markdown",
        )
        report["artifacts"]["support_bundle_json"] = base.artifact_entry(
            output_dir / "support_bundle.json",
            output_dir,
            kind="micro_llm_sharded_support_bundle",
            schema="micro_llm_sharded_support_bundle_v1",
        )
        base.write_json(report, args.json_out)
        if args.markdown_out:
            output = Path(args.markdown_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_markdown(report), encoding="utf-8")
        support_path = output_dir / "support_bundle.json"
        base.write_json(micro_support_bundle_payload(report), str(support_path))
        if args.json_out:
            report["artifacts"]["micro_llm_sharded_evidence_json"]["present"] = Path(args.json_out).is_file()
        if args.markdown_out:
            report["artifacts"]["micro_llm_sharded_evidence_markdown"]["present"] = Path(args.markdown_out).is_file()
        report["artifacts"]["support_bundle_json"]["present"] = support_path.is_file()
        report["artifact_summary"] = micro_artifact_summary(output_dir)
        base.write_json(micro_support_bundle_payload(report), str(support_path))
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
