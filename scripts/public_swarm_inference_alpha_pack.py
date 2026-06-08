#!/usr/bin/env python3
"""Public Swarm Inference Alpha evidence wrapper.

This is the product-shaped aggregation layer above the existing real tiny GPT
split proofs.  It deliberately keeps the workload CPU-only and read-only while
combining the live Kaggle proof with the mandatory local stage requeue proof.
"""

from __future__ import annotations

import argparse
import json
import shlex
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
    "SOURCE_TARBALL_B64",
    "MINER_ENV_TEXT",
    '"prompt_text":',
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


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "generation_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Alpha artifacts summarize controlled tiny GPT split "
            "readiness, local/live requeue evidence, cleanup state, hashes/counts, "
            "and diagnostics only. They do not include answer text."
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
        "raw_generation_public": False,
        "generation_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Alpha report is shareable evidence, not a local answer "
            "transcript; raw prompts, answer text, token ids, activations, leases, "
            "credentials, private env files, and raw runtime state are excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "generation_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share public_swarm_inference_alpha.json/md; they contain readiness "
            "evidence, cleanup state, hashes, counts, and diagnostics, not raw "
            "prompts or answers."
        ),
    }


def prompt_scope_summary(report: dict[str, Any] | None = None) -> dict[str, Any]:
    session = report.get("session") if isinstance(report, dict) and isinstance(report.get("session"), dict) else {}
    try:
        prompt_count = int(session.get("request_count") or 0)
    except (TypeError, ValueError):
        prompt_count = 0
    return {
        "source": "imported-or-built-in-validation-prompts",
        "prompt_count": prompt_count,
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": False,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Alpha aggregates fixed or imported validation prompts; "
            "public artifacts record prompt source/count only and exclude raw prompt text."
        ),
    }


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


def shell_command(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def command_entry(
    label: str,
    command: list[Any],
    *,
    reason: str = "",
    requires_private_credentials: bool = False,
    side_effectful: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": shell_command(command),
        "public_artifact_safe": True,
    }
    if reason:
        entry["reason"] = reason
    if requires_private_credentials:
        entry["requires_private_credentials"] = True
        entry["credential_note"] = (
            "Use local private Kaggle/operator credentials when running this command; "
            "credential values are intentionally excluded from public artifacts."
        )
    if side_effectful:
        entry["side_effectful"] = True
    return entry


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


def prompt_scope_note(prompt_scope: dict[str, Any]) -> str:
    return str(
        prompt_scope.get("summary")
        or "Public artifacts record prompt source/count only and exclude raw prompt text."
    )


def answer_scope_note(answer_scope: dict[str, Any]) -> str:
    return str(
        answer_scope.get("summary")
        or "Public artifacts contain no local answer transcript or raw generated text."
    )


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


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "public_swarm_inference_alpha.md",
        "summary_json": output_dir / "public_swarm_inference_alpha.json",
        "summary_markdown": output_dir / "public_swarm_inference_alpha.md",
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


def public_swarm_alpha_command(args: argparse.Namespace, output_dir: Path, mode: str) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        "swarm-session",
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        getattr(args, "public_host", DEFAULT_PUBLIC_HOST),
        "--bind-host",
        getattr(args, "bind_host", "0.0.0.0"),
        "--port",
        str(getattr(args, "port", DEFAULT_PORT)),
        "--base-port",
        str(getattr(args, "base_port", DEFAULT_BASE_PORT)),
        "--miner-id-prefix",
        getattr(args, "miner_id_prefix", "public-swarm-alpha"),
        "--request-count",
        str(getattr(args, "request_count", 2)),
        "--hf-model-id",
        getattr(args, "hf_model_id", DEFAULT_MODEL_ID),
        "--failure-mode",
        getattr(args, "failure_mode", FAILURE_NONE),
        "--timeout-seconds",
        str(getattr(args, "timeout_seconds", 300.0)),
        "--remote-timeout-seconds",
        str(getattr(args, "remote_timeout_seconds", 300.0)),
        "--startup-timeout",
        str(getattr(args, "startup_timeout", 60.0)),
        "--process-exit-timeout",
        str(getattr(args, "process_exit_timeout", 10.0)),
        "--poll-interval",
        str(getattr(args, "poll_interval", 1.0)),
        "--http-timeout",
        str(getattr(args, "http_timeout", 30.0)),
        "--kaggle-push-timeout-seconds",
        str(getattr(args, "kaggle_push_timeout_seconds", 180.0)),
        "--kaggle-delete-timeout-seconds",
        str(getattr(args, "kaggle_delete_timeout_seconds", 120.0)),
        "--kaggle-status-timeout-seconds",
        str(getattr(args, "kaggle_status_timeout_seconds", 300.0)),
        "--kaggle-status-poll-interval",
        str(getattr(args, "kaggle_status_poll_interval", 5.0)),
        "--lease-seconds",
        str(getattr(args, "lease_seconds", 15.0)),
        "--compute-seconds",
        str(getattr(args, "compute_seconds", 0.2)),
        "--victim-compute-seconds",
        str(getattr(args, "victim_compute_seconds", 45.0)),
        "--heartbeat-interval",
        str(getattr(args, "heartbeat_interval", 0.1)),
        "--idle-sleep",
        str(getattr(args, "idle_sleep", 0.2)),
        "--claim-observe-timeout",
        str(getattr(args, "claim_observe_timeout", 180.0)),
        "--requeue-timeout",
        str(getattr(args, "requeue_timeout", 120.0)),
        "--max-request-attempts",
        str(getattr(args, "max_request_attempts", 240)),
    ]
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", "HF_CACHE_DIR"])
    if mode == MODE_LIVE_KAGGLE:
        command.extend([
            "--dataset-title",
            getattr(args, "dataset_title", "CrowdTensor Public Swarm Inference Alpha"),
            "--kernel-title-prefix",
            getattr(args, "kernel_title_prefix", "CrowdTensor Public Swarm Inference Alpha"),
            "--kaggle-owner",
            getattr(args, "kaggle_owner", "") or "KAGGLE_USERNAME",
        ])
        if getattr(args, "ready_url", ""):
            command.extend(["--ready-url", "READY_URL"])
        if getattr(args, "coordinator_url", ""):
            command.extend(["--coordinator-url", "COORDINATOR_URL"])
        if getattr(args, "dataset_slug", ""):
            command.extend(["--dataset-slug", getattr(args, "dataset_slug")])
        if getattr(args, "kernel_slug_prefix", ""):
            command.extend(["--kernel-slug-prefix", getattr(args, "kernel_slug_prefix")])
        if getattr(args, "inline_kernel_payload", True):
            command.append("--inline-kernel-payload")
        else:
            command.append("--no-inline-kernel-payload")
        if getattr(args, "skip_kaggle_cleanup", False):
            command.append("--skip-kaggle-cleanup")
        if getattr(args, "keep_live_private_artifacts", False):
            command.append("--keep-live-private-artifacts")
    if getattr(args, "keep_child_artifacts", False):
        command.append("--keep-child-artifacts")
    if getattr(args, "skip_local_requeue", False):
        command.append("--skip-local-requeue")
    command.append("--json")
    return command


def not_completed_items(report: dict[str, Any]) -> list[str]:
    session = report.get("session") if isinstance(report.get("session"), dict) else {}
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    codes = set(report.get("diagnosis_codes") or [])
    mode = str(report.get("mode") or "")
    existing = [str(item) for item in (report.get("not_completed") or []) if str(item)]
    items: list[tuple[str, Any]] = [
        ("Public Swarm Alpha ready", report.get("ok")),
        ("public session ready", "public_swarm_session_ready" in codes),
        ("local stage requeue ready", "local_stage_requeue_ready" in codes and session.get("local_stage_requeue_verified") is True),
        ("stage assignment valid", session.get("stage_assignment_valid") is True or "stage_assignment_valid" in codes),
        ("distinct stage miners", session.get("distinct_stage_miners") is True or "distinct_stage_miners" in codes),
        ("decoded tokens match", session.get("decoded_tokens_match") is True or "decoded_tokens_match" in codes),
    ]
    if mode == MODE_LIVE_KAGGLE:
        items.extend([
            ("live Kaggle proof ready", "public_swarm_live_kaggle_ready" in codes and safety.get("live_kaggle_verified") is True),
            ("external runtime verified", session.get("live_external_runtime_verified") is True or "external_runtime_verified" in codes),
            ("Kaggle kernels deleted", session.get("live_kaggle_kernels_deleted") is True or "kaggle_kernels_deleted" in codes),
        ])
        if str(report.get("failure_mode") or FAILURE_NONE) != FAILURE_NONE:
            items.append(("live external stage requeue ready", "external_stage_requeue_ready" in codes and session.get("live_stage_requeue_verified") is True))
    for step in report.get("steps") or []:
        if isinstance(step, dict) and step.get("ok") is not True:
            items.append((f"step {step.get('name') or 'step'} passed", False))
    missing = list(existing)
    seen = set(missing)
    for label, ready in items:
        if ready is True or label in seen:
            continue
        missing.append(label)
        seen.add(label)
    return missing


def recommended_next_command(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    missing: list[str],
) -> dict[str, Any]:
    mode = str(report.get("mode") or args.mode)
    if report.get("ok"):
        return command_entry(
            "inspect Public Swarm Alpha evidence",
            artifact_command(output_dir, "public_swarm_inference_alpha.md"),
            reason="review_artifacts",
        )
    if mode == MODE_LIVE_KAGGLE:
        return command_entry(
            "rerun Public Swarm Alpha live proof",
            public_swarm_alpha_command(args, output_dir, MODE_LIVE_KAGGLE),
            reason="fix_live_kaggle_or_requeue_blockers" if missing else "rerun_live_kaggle",
            requires_private_credentials=True,
            side_effectful=True,
        )
    return command_entry(
        "rerun Public Swarm Alpha local proof",
        public_swarm_alpha_command(args, output_dir, MODE_LOCAL_GENERATED),
        reason="fix_local_requeue_blockers" if missing else "rerun_local_generated",
        side_effectful=True,
    )


def next_commands(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = str(report.get("mode") or args.mode)
    commands = [
        command_entry(
            "inspect shareable summary",
            artifact_command(output_dir, "public_swarm_inference_alpha.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect support bundle",
            artifact_command(output_dir, "support_bundle.json", lines="1,220p"),
            reason="inspect_diagnostics",
        ),
    ]
    if report.get("ok"):
        commands.append(command_entry(
            f"refresh {mode} proof",
            public_swarm_alpha_command(args, output_dir, mode),
            reason="refresh_public_swarm_alpha",
            requires_private_credentials=mode == MODE_LIVE_KAGGLE,
            side_effectful=True,
        ))
    else:
        commands.append(dict(recommended))
    if mode != MODE_LIVE_KAGGLE:
        commands.append(command_entry(
            "run live Kaggle Alpha proof",
            public_swarm_alpha_command(args, output_dir, MODE_LIVE_KAGGLE),
            reason="promote_local_proof_to_live_public_alpha",
            requires_private_credentials=True,
            side_effectful=True,
        ))
    return commands


def user_status(*, ready: bool, mode: str, recommended: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    if ready:
        state = "ready"
        headline = "Public Swarm Inference Alpha evidence is ready."
        next_step = "review_artifacts"
    elif mode == MODE_LIVE_KAGGLE:
        state = "live-kaggle-blocked"
        headline = "Public Swarm Alpha live Kaggle proof needs attention."
        next_step = "fix_live_kaggle_or_requeue_blockers"
    else:
        state = "local-generated-blocked"
        headline = "Public Swarm Alpha local proof needs attention."
        next_step = "fix_local_requeue_blockers"
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "not_completed_count": len(missing),
        "public_artifact_safe": True,
    }


def review_summary(
    report: dict[str, Any],
    *,
    output_dir: Path,
    recommended: dict[str, Any],
    missing: list[str],
) -> dict[str, Any]:
    ready = bool(report.get("ok"))
    codes = [str(code) for code in (report.get("diagnosis_codes") or [])]
    return {
        "schema": "public_swarm_inference_alpha_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "headline": (
            "Public Swarm Inference Alpha evidence is ready."
            if ready
            else "Public Swarm Inference Alpha evidence needs attention."
        ),
        "mode": report.get("mode"),
        "next_step": "review_artifacts" if ready else "fix_blockers",
        "inspect_first": str(output_dir / "public_swarm_inference_alpha.md"),
        "support_bundle": str(output_dir / "support_bundle.json"),
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": "public_swarm_inference_alpha_ready" if ready else (codes[0] if codes else "public_swarm_inference_alpha_blocked"),
        "attention": "none" if ready else (missing[0] if missing else "public_swarm_inference_alpha_blocked"),
        "attention_detail": "; ".join(missing[:6]),
        "not_completed_count": len(missing),
        "public_artifact_safe": True,
    }


def attach_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    missing = not_completed_items(report)
    recommended = recommended_next_command(report, args, output_dir=output_dir, missing=missing)
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = next_commands(report, args, output_dir=output_dir, recommended=recommended)
    report["user_status"] = user_status(
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or args.mode),
        recommended=recommended,
        missing=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def ensure_user_guidance(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    if (
        isinstance(report.get("recommended_next_command"), dict)
        and isinstance(report.get("next_commands"), list)
        and isinstance(report.get("user_status"), dict)
        and isinstance(report.get("review_summary"), dict)
    ):
        report.setdefault("not_completed", not_completed_items(report))
        report.setdefault("artifact_summary", artifact_summary(output_dir))
        return report
    missing = not_completed_items(report)
    recommended = command_entry(
        "inspect Public Swarm Alpha evidence",
        artifact_command(output_dir, "public_swarm_inference_alpha.md"),
        reason="review_artifacts" if report.get("ok") else "review_missing_evidence",
    )
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = [
        command_entry("inspect shareable summary", artifact_command(output_dir, "public_swarm_inference_alpha.md"), reason="review_artifacts"),
        command_entry("inspect support bundle", artifact_command(output_dir, "support_bundle.json", lines="1,220p"), reason="inspect_diagnostics"),
    ]
    report["user_status"] = user_status(
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or ""),
        recommended=recommended,
        missing=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def support_bundle_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "public_swarm_inference_alpha_support_bundle_v1",
        "ok": report.get("ok"),
        "mode": report.get("mode"),
        "output_dir": report.get("output_dir"),
        "coordinator_url": report.get("coordinator_url"),
        "failure_mode": report.get("failure_mode"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "session": report.get("session"),
        "steps": report.get("steps"),
        "payload_summaries": report.get("payload_summaries"),
        "artifact_cleanup": report.get("artifact_cleanup"),
        "review_summary": report.get("review_summary"),
        "user_status": report.get("user_status"),
        "recommended_next_command": report.get("recommended_next_command"),
        "next_commands": report.get("next_commands"),
        "artifact_summary": report.get("artifact_summary"),
        "not_completed": report.get("not_completed"),
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
        "public_artifact_safe": True,
    }


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
    report = attach_user_guidance(report, args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir)


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary(report))
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report.setdefault("artifact_summary", artifact_summary(output_dir))
    report = ensure_user_guidance(report, output_dir=output_dir)
    report = support_bundle.sanitize(redact_values(report))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "public Swarm Inference Alpha report contained secret-like fragments"
    json_path = output_dir / "public_swarm_inference_alpha.json"
    md_path = output_dir / "public_swarm_inference_alpha.md"
    support_path = output_dir / "support_bundle.json"
    report.setdefault("artifacts", {})
    report["artifacts"]["support_bundle_json"] = artifact_entry(
        support_path,
        output_dir,
        kind="public_swarm_inference_alpha_support_bundle",
        schema="public_swarm_inference_alpha_support_bundle_v1",
        ok=report.get("ok"),
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    if isinstance(report.get("review_summary"), dict):
        report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
        report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    write_json(support_path, support_bundle_payload(report))
    if "artifacts" in report:
        report["artifacts"]["public_swarm_inference_alpha_json"]["present"] = True
        report["artifacts"]["public_swarm_inference_alpha_markdown"]["present"] = True
        report["artifacts"]["support_bundle_json"]["present"] = True
        report["artifact_summary"] = artifact_summary(output_dir)
        if isinstance(report.get("review_summary"), dict):
            report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
            report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
        write_json(json_path, report)
        md_path.write_text(render_markdown(report), encoding="utf-8")
        write_json(support_path, support_bundle_payload(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    session = report.get("session") if isinstance(report.get("session"), dict) else {}
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    next_items = report.get("next_commands") if isinstance(report.get("next_commands"), list) else []
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
                + (" side_effectful=`True`" if item.get("side_effectful") else "")
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
        f"- output request note: {output_request.get('summary') or 'Public artifacts summarize inference evidence only and do not include answer text.'}",
        f"- prompt scope: `{prompt_scope_text(prompt_scope)}`",
        f"- prompt scope note: {prompt_scope_note(prompt_scope)}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope_note(answer_scope)}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generation_public={shareable.get('raw_generation_public')} generation_ids_public={shareable.get('generation_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
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
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    print("CrowdTensor Public Swarm Inference Alpha")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  model: {session.get('model_id')}")
    print(f"  external runtime: {session.get('live_external_runtime_verified')}")
    print(f"  local requeue: {session.get('local_stage_requeue_verified')}")
    if output_request:
        print(f"  output_request: include_output={bool(output_request.get('include_output'))} raw_generation_public={bool(output_request.get('raw_generation_public'))} public_artifact_safe={bool(output_request.get('public_artifact_safe'))}")
        print(f"  output_request_note: {output_request.get('summary') or 'Public artifacts summarize inference evidence only and do not include answer text.'}")
    if prompt_scope:
        print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
        print(f"  prompt_scope_note: {prompt_scope_note(prompt_scope)}")
    if answer_scope:
        print(f"  answer_scope: {answer_scope.get('scope_state')}")
        print(f"  answer_scope_note: {answer_scope_note(answer_scope)}")
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
