#!/usr/bin/env python3
"""Build the CPU Inference Beta release-candidate evidence pack."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402


SCHEMA = "cpu_inference_beta_rc_v1"
KAGGLE_REAL_RUNTIME_SCHEMA = "kaggle_real_runtime_acceptance_v1"
REAL_LLM_LIVE_RC_SCHEMA = "real_llm_live_rc_v1"
Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = (
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "CROWDTENSOR_LLM_RUNTIME_API_KEY=",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    redacted = text
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: int,
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
        return (
            {
                "name": name,
                "ok": False,
                "returncode": None,
                "duration_seconds": round(time.monotonic() - started, 3),
                "error": "timeout",
            },
            {},
        )

    payload: dict[str, Any] = {}
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    else:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
        step["ok"] = bool(step.get("ok") and payload.get("ok"))
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:])
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:])
    return step, payload


def diagnosis_codes(*payloads: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
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


def payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "diagnosis_codes": diagnosis_codes(payload),
    }
    for key in ["mode", "workload", "target", "step_count", "scenario_id"]:
        if payload.get(key) is not None:
            summary[key] = payload.get(key)
    if isinstance(payload.get("target_environment"), dict):
        target = payload["target_environment"]
        summary["target_environment"] = {
            "name": target.get("name"),
            "remote_environment": target.get("remote_environment"),
            "kaggle_remote_miner_beta": target.get("kaggle_remote_miner_beta"),
            "gpu_tpu_workload_enabled": target.get("gpu_tpu_workload_enabled"),
            "miner_outbound_only": target.get("miner_outbound_only"),
        }
    if isinstance(payload.get("safety"), dict):
        safety = payload["safety"]
        summary["safety"] = {
            "read_only": safety.get("read_only"),
            "credential_redaction_checked": safety.get("token_redaction_checked"),
            "not_production": safety.get("not_production"),
            "not_p2p": safety.get("not_p2p"),
        }
    return summary


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def kaggle_real_runtime_summary(path_value: str) -> dict[str, Any]:
    if not path_value:
        return {
            "present": False,
            "ok": None,
            "schema": KAGGLE_REAL_RUNTIME_SCHEMA,
            "ready": False,
            "diagnosis_codes": [],
        }
    path = Path(path_value).resolve()
    payload = load_json_file(path)
    codes = diagnosis_codes(payload)
    acceptance = payload.get("acceptance_summary") if isinstance(payload.get("acceptance_summary"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    schema = payload.get("schema") or KAGGLE_REAL_RUNTIME_SCHEMA
    is_kaggle_runtime = schema == KAGGLE_REAL_RUNTIME_SCHEMA and payload.get("ok") is True and "kaggle_real_runtime_ready" in codes
    is_real_llm_live = schema == REAL_LLM_LIVE_RC_SCHEMA and payload.get("ok") is True and "kaggle_real_llm_sharded_ready" in codes
    stage_assignment: dict[str, Any] = {}
    if is_real_llm_live:
        verify = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        verify_payload = verify.get("verify") if isinstance(verify.get("verify"), dict) else {}
        evidence = verify_payload.get("remote_existing_real_llm_sharded_inference") if isinstance(verify_payload.get("remote_existing_real_llm_sharded_inference"), dict) else {}
        stage_assignment = evidence.get("stage_assignment") if isinstance(evidence.get("stage_assignment"), dict) else {}
    return {
        "present": path.is_file(),
        "path": str(path),
        "schema": schema,
        "ok": payload.get("ok"),
        "ready": bool(is_kaggle_runtime or is_real_llm_live),
        "diagnosis_codes": codes,
        "mode": payload.get("mode"),
        "coordinator_url": payload.get("coordinator_url"),
        "miner_id": payload.get("miner_id"),
        "task_id": acceptance.get("task_id"),
        "scenario_id": acceptance.get("scenario_id"),
        "request_count": acceptance.get("request_count"),
        "runtime_kind": "real_llm_live_rc" if is_real_llm_live else "kaggle_real_runtime_acceptance",
        "stage0_miner_id": stage_assignment.get("stage0_miner_id"),
        "stage1_miner_id": stage_assignment.get("stage1_miner_id"),
        "stage_assignment_valid": stage_assignment.get("stage_assignment_valid"),
        "distinct_stage_miners": stage_assignment.get("distinct_stage_miners"),
        "token_rotation_required": bool(safety.get("token_rotation_required")),
        "temporary_http": bool(safety.get("temporary_http")),
        "operator_env_excluded_from_kaggle": bool(safety.get("operator_env_excluded_from_kaggle") or is_real_llm_live),
    }


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
            "Use local private operator/runtime credentials when running this command; "
            "credential values are intentionally excluded from public artifacts."
        )
    if side_effectful:
        entry["side_effectful"] = True
    return entry


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "cpu_inference_beta_rc.md",
        "summary_json": output_dir / "cpu_inference_beta_rc.json",
        "summary_markdown": output_dir / "cpu_inference_beta_rc.md",
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


def cpu_beta_rc_command(args: argparse.Namespace, output_dir: Path) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        "cpu-infer",
        "--mode",
        "beta-rc",
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(getattr(args, "base_port", 9070)),
        "--request-count",
        str(getattr(args, "request_count", 2)),
        "--external-llm-request-count",
        str(getattr(args, "external_llm_request_count", 2)),
        "--scenario-id",
        getattr(args, "scenario_id", "route-baseline"),
        "--timeout-seconds",
        str(getattr(args, "timeout_seconds", 240)),
        "--remote-timeout-seconds",
        str(getattr(args, "remote_timeout_seconds", 60.0)),
        "--startup-timeout",
        str(getattr(args, "startup_timeout", 10.0)),
        "--verify-timeout",
        str(getattr(args, "verify_timeout", 40.0)),
        "--command-timeout",
        str(getattr(args, "command_timeout", 120.0)),
        "--two-machine-timeout-seconds",
        str(getattr(args, "two_machine_timeout_seconds", 240)),
        "--kaggle-timeout-seconds",
        str(getattr(args, "kaggle_timeout_seconds", 240)),
        "--kaggle-coordinator-url",
        "https://YOUR_COORDINATOR_HOST",
        "--kaggle-miner-id",
        getattr(args, "kaggle_miner_id", "kaggle-cpu-1"),
    ]
    if getattr(args, "kaggle_real_runtime_report", ""):
        command.extend(["--kaggle-real-runtime-report", getattr(args, "kaggle_real_runtime_report")])
    if getattr(args, "quick", False):
        command.append("--quick")
    command.append("--json")
    return command


def cpu_beta_rc_import_command(args: argparse.Namespace, output_dir: Path) -> list[Any]:
    command = cpu_beta_rc_command(args, output_dir)
    if "--kaggle-real-runtime-report" not in command:
        json_index = command.index("--json") if "--json" in command else len(command)
        command[json_index:json_index] = [
            "--kaggle-real-runtime-report",
            "dist/kaggle-real-runtime/kaggle_real_runtime_acceptance.json",
        ]
    return command


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "raw_external_llm_output_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "CPU Inference Beta RC artifacts summarize readiness, child evidence, "
            "Miner join artifacts, optional real-runtime import state, and diagnostics only. "
            "They do not include raw prompt text, generated text, external LLM output text, "
            "credentials, leases, or idempotency keys."
        ),
    }


def prompt_scope_summary(report: dict[str, Any]) -> dict[str, Any]:
    request_count = int(report.get("request_count") or 0)
    external_count = int(report.get("external_llm_request_count") or 0)
    return {
        "source": "built-in-fixed-scenarios",
        "prompt_count": request_count + external_count,
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
            "CPU Inference Beta RC uses fixed built-in scenarios across local, remote-loopback, "
            "two-machine rehearsal, and Kaggle Miner artifact checks. Public artifacts record "
            "counts/source only and exclude raw prompt text."
        ),
    }


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "summary-only",
        "saved_markdown_display": "summary-only",
        "json_stdout_display": "summary-only-json",
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "raw_external_llm_output_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This RC report is shareable operator evidence, not an answer transcript; "
            "raw inference payloads, generated text, credentials, leases, and idempotency keys are excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "raw_external_llm_output_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share cpu_inference_beta_rc.json/md and support_bundle.json; they contain "
            "RC readiness evidence and diagnostics, not raw prompts, outputs, or credentials."
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


def not_completed_items(report: dict[str, Any]) -> list[str]:
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []) if code)
    existing = [str(item) for item in (report.get("not_completed") or []) if str(item)]
    items: list[tuple[str, Any]] = [
        ("CPU Inference Beta RC ready", report.get("ok") is True and "cpu_inference_beta_rc_ready" in codes),
        ("local CPU inference ready", "local_cpu_inference_ready" in codes),
        ("remote loopback ready", "remote_loopback_ready" in codes),
        ("two-machine rehearsal ready", "two_machine_rehearsal_ready" in codes),
        ("Kaggle Remote Miner artifacts ready", "kaggle_remote_miner_artifacts_ready" in codes),
        ("Miner join pack ready", "miner_join_pack_ready" in codes),
        ("CPU Miner Beta ready", "cpu_miner_beta_ready" in codes),
        ("CPU-only safety boundary present", safety.get("cpu_only") is True),
        ("read-only safety boundary present", safety.get("read_only") is True),
        ("not production boundary present", safety.get("not_production") is True),
        ("not P2P boundary present", safety.get("not_p2p") is True),
        ("not GPU/TPU workload boundary present", safety.get("not_gpu_tpu_workload") is True),
    ]
    real_runtime = report.get("real_runtime_evidence") if isinstance(report.get("real_runtime_evidence"), dict) else {}
    if real_runtime.get("present") is True:
        items.append(("real runtime evidence ready", real_runtime.get("ready") is True))
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    for step in steps:
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
    if report.get("ok"):
        return command_entry(
            "inspect CPU Inference Beta RC evidence",
            artifact_command(output_dir, "cpu_inference_beta_rc.md"),
            reason="review_artifacts",
        )
    return command_entry(
        "rerun CPU Inference Beta RC",
        cpu_beta_rc_command(args, output_dir),
        reason="fix_cpu_inference_beta_rc_blockers" if missing else "rerun_cpu_inference_beta_rc",
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
            artifact_command(output_dir, "cpu_inference_beta_rc.md"),
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
            "refresh CPU Inference Beta RC proof",
            cpu_beta_rc_command(args, output_dir),
            reason="refresh_cpu_inference_beta_rc",
            side_effectful=True,
        ))
    else:
        commands.append(dict(recommended))
    commands.append(command_entry(
        "import retained real runtime evidence",
        cpu_beta_rc_import_command(args, output_dir),
        reason="import_real_runtime_evidence",
        side_effectful=True,
    ))
    return commands


def user_status(*, ready: bool, recommended: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    return {
        "state": "ready" if ready else "blocked",
        "headline": (
            "CPU Inference Beta RC evidence is ready."
            if ready
            else "CPU Inference Beta RC evidence needs attention."
        ),
        "next_step": "review_artifacts" if ready else "fix_beta_rc_blockers",
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
    codes = [str(code) for code in (report.get("diagnosis_codes") or []) if code]
    return {
        "schema": "cpu_inference_beta_rc_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "headline": "CPU Inference Beta RC evidence is ready." if ready else "CPU Inference Beta RC evidence needs attention.",
        "mode": report.get("mode"),
        "next_step": "review_artifacts" if ready else "fix_blockers",
        "inspect_first": str(output_dir / "cpu_inference_beta_rc.md"),
        "support_bundle": str(output_dir / "support_bundle.json"),
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": "cpu_inference_beta_rc_ready" if ready else (codes[0] if codes else "cpu_inference_beta_rc_failed"),
        "attention": "none" if ready else (missing[0] if missing else "cpu_inference_beta_rc_failed"),
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
        "schema": "cpu_inference_beta_rc_support_bundle_v1",
        "ok": report.get("ok"),
        "mode": report.get("mode"),
        "output_dir": report.get("output_dir"),
        "request_count": report.get("request_count"),
        "external_llm_request_count": report.get("external_llm_request_count"),
        "scenario_id": report.get("scenario_id"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "steps": report.get("steps"),
        "payload_summaries": report.get("payload_summaries"),
        "miner_join_pack": report.get("miner_join_pack"),
        "real_runtime_evidence": report.get("real_runtime_evidence"),
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


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}

    local_dir = output_dir / "local"
    remote_dir = output_dir / "remote-loopback"
    kaggle_dir = output_dir / "kaggle-remote-miner"
    manifest_dir = output_dir / "demo-manifest"

    local_step, local_payload = run_json_step(
        "local_cpu_inference",
        [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "cpu-infer",
            "--mode",
            "local",
            "--output-dir",
            str(local_dir),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--scenario-id",
            args.scenario_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ],
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    steps.append(local_step)
    payloads["local_cpu_inference"] = payload_summary(local_payload)

    remote_step, remote_payload = run_json_step(
        "remote_loopback_cpu_inference",
        [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "cpu-infer",
            "--mode",
            "remote-loopback",
            "--workload",
            "all",
            "--output-dir",
            str(remote_dir),
            "--base-port",
            str(args.base_port + 20),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--scenario-id",
            args.scenario_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--json",
        ],
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    steps.append(remote_step)
    payloads["remote_loopback_cpu_inference"] = payload_summary(remote_payload)

    two_machine_step, two_machine_payload = run_json_step(
        "two_machine_rehearsal",
        [
            sys.executable,
            str(ROOT / "scripts" / "remote_two_machine_beta_check.py"),
            "--workload",
            "all",
            "--base-port",
            str(args.base_port + 40),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--scenario-id",
            args.scenario_id,
            "--startup-timeout",
            str(args.startup_timeout),
            "--verify-timeout",
            str(args.verify_timeout),
            "--command-timeout",
            str(args.command_timeout),
            "--timeout-seconds",
            str(args.two_machine_timeout_seconds),
        ],
        runner=runner,
        timeout_seconds=args.two_machine_timeout_seconds,
    )
    steps.append(two_machine_step)
    payloads["two_machine_rehearsal"] = payload_summary(two_machine_payload)

    kaggle_prepare_step, kaggle_prepare_payload = run_json_step(
        "kaggle_remote_miner_prepare",
        [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "remote-demo",
            "prepare",
            "--target",
            "kaggle",
            "--coordinator-url",
            args.kaggle_coordinator_url,
            "--miner-id",
            args.kaggle_miner_id,
            "--output-dir",
            str(kaggle_dir),
            "--request-count",
            str(args.request_count),
            "--scenario-id",
            args.scenario_id,
            "--replace",
            "--json",
        ],
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    steps.append(kaggle_prepare_step)
    payloads["kaggle_remote_miner_prepare"] = payload_summary(kaggle_prepare_payload)

    kaggle_check_step, kaggle_check_payload = run_json_step(
        "kaggle_remote_miner_beta_check",
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_remote_miner_beta_check.py"),
            "--coordinator-url",
            args.kaggle_coordinator_url,
            "--miner-id",
            args.kaggle_miner_id,
            "--port",
            str(args.base_port + 60),
            "--request-count",
            str(args.request_count),
            "--scenario-id",
            args.scenario_id,
            "--startup-timeout",
            str(args.startup_timeout),
            "--verify-timeout",
            str(args.verify_timeout),
            "--command-timeout",
            str(args.command_timeout),
            "--timeout-seconds",
            str(args.kaggle_timeout_seconds),
        ],
        runner=runner,
        timeout_seconds=args.kaggle_timeout_seconds,
    )
    steps.append(kaggle_check_step)
    payloads["kaggle_remote_miner_beta_check"] = payload_summary(kaggle_check_payload)

    manifest_step, manifest_payload = run_json_step(
        "demo_manifest",
        [
            sys.executable,
            str(ROOT / "scripts" / "demo_manifest_check.py"),
            "--output-dir",
            str(manifest_dir),
            "--base-port",
            str(args.base_port + 80),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
        ],
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    steps.append(manifest_step)
    payloads["demo_manifest"] = payload_summary(manifest_payload)

    ok = all(bool(step.get("ok")) for step in steps)
    join_artifacts_ready = all(
        (kaggle_dir / name).is_file()
        for name in ["miner_join.sh", "MINER_JOIN.md", "miner.private.env", "kaggle_remote_miner.py"]
    )
    kaggle_real_summary = kaggle_real_runtime_summary(args.kaggle_real_runtime_report)
    codes = set(diagnosis_codes(local_payload, remote_payload, two_machine_payload, kaggle_prepare_payload, kaggle_check_payload, manifest_payload))
    if local_step.get("ok"):
        codes.add("local_cpu_inference_ready")
    if remote_step.get("ok"):
        codes.add("remote_loopback_ready")
    if two_machine_step.get("ok"):
        codes.add("two_machine_rehearsal_ready")
    if kaggle_prepare_step.get("ok"):
        codes.add("kaggle_remote_miner_artifacts_ready")
    if join_artifacts_ready:
        codes.add("miner_join_pack_ready")
        codes.add("cpu_miner_beta_ready")
    if kaggle_real_summary.get("ready"):
        codes.add("real_runtime_evidence_ready")
        if kaggle_real_summary.get("schema") == REAL_LLM_LIVE_RC_SCHEMA:
            codes.add("kaggle_real_llm_sharded_ready")
        else:
            codes.add("kaggle_real_runtime_ready")
    if ok:
        codes.add("cpu_inference_beta_rc_ready")
    else:
        codes.add("beta_rc_blocked")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": "beta-rc",
        "output_dir": str(output_dir),
        "request_count": args.request_count,
        "external_llm_request_count": args.external_llm_request_count,
        "scenario_id": args.scenario_id,
        "steps": steps,
        "payload_summaries": payloads,
        "miner_join_pack": {
            "schema": "miner_join_pack_v1",
            "ready": join_artifacts_ready,
            "target": "kaggle",
            "generated_files": ["miner.private.env", "miner_join.sh", "MINER_JOIN.md", "kaggle_remote_miner.py"],
            "operator_files_excluded": ["operator.private.env", "miner_registry.json"],
            "recommended_command": "bash miner_join.sh",
            "kaggle_command": "python kaggle_remote_miner.py --env-file miner.private.env",
        },
        "real_runtime_evidence": kaggle_real_summary,
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "local_cpu_inference_beta_json": artifact_entry(
                local_dir / "cpu_inference_beta.json",
                output_dir,
                kind="cpu_inference_beta",
                schema="cpu_inference_beta_v1",
            ),
            "remote_loopback_cpu_inference_beta_json": artifact_entry(
                remote_dir / "cpu_inference_beta.json",
                output_dir,
                kind="cpu_inference_beta",
                schema="cpu_inference_beta_v1",
            ),
            "kaggle_remote_miner_script": artifact_entry(
                kaggle_dir / "kaggle_remote_miner.py",
                output_dir,
                kind="kaggle_remote_miner_script",
            ),
            "kaggle_remote_miner_runbook": artifact_entry(
                kaggle_dir / "kaggle_remote_miner.md",
                output_dir,
                kind="kaggle_remote_miner_runbook",
            ),
            "miner_join_script": artifact_entry(
                kaggle_dir / "miner_join.sh",
                output_dir,
                kind="miner_join_script",
            ),
            "miner_join_runbook": artifact_entry(
                kaggle_dir / "MINER_JOIN.md",
                output_dir,
                kind="miner_join_runbook",
                schema="miner_join_pack_v1",
            ),
            "kaggle_remote_home_compute_demo_json": artifact_entry(
                kaggle_dir / "remote_home_compute_demo.json",
                output_dir,
                kind="remote_home_compute_demo",
                schema="remote_home_compute_demo_v1",
            ),
            "demo_manifest_json": artifact_entry(
                manifest_dir / "demo_manifest.json",
                output_dir,
                kind="demo_manifest",
                schema="demo_manifest_v1",
            ),
            "demo_manifest_markdown": artifact_entry(
                manifest_dir / "demo_manifest.md",
                output_dir,
                kind="demo_manifest_markdown",
            ),
            "kaggle_real_runtime_report": artifact_entry(
                Path(args.kaggle_real_runtime_report).resolve() if args.kaggle_real_runtime_report else output_dir / "kaggle_real_runtime_acceptance.json",
                output_dir,
                kind="kaggle_real_runtime_acceptance",
                schema=str(kaggle_real_summary.get("schema") or KAGGLE_REAL_RUNTIME_SCHEMA),
                ok=kaggle_real_summary.get("ok") if kaggle_real_summary.get("present") else None,
            ),
        },
        "safety": {
            "cpu_only": True,
            "read_only": True,
            "fixed_scenarios": True,
            "kaggle_outbound_miner_only": True,
            "operator_env_excluded_from_kaggle": True,
            "summary_excludes_raw_inference_payloads": True,
            "miner_join_pack_redacted": True,
            "real_runtime_evidence_imported": bool(kaggle_real_summary.get("present")),
            "not_production": True,
            "not_p2p": True,
            "not_gpu_tpu_workload": True,
            "not_arbitrary_prompt_serving": True,
        },
        "limitations": [
            "CPU Inference Beta RC evidence only; not production Swarm Inference.",
            "Remote checks use local loopback stand-ins unless an operator runs the documented real two-machine flow.",
            "Kaggle is prepared as an outbound temporary CPU Miner target; this does not enable GPU/TPU workloads.",
            "Real Kaggle runtime evidence is imported only when --kaggle-real-runtime-report points to an existing acceptance report.",
            "This does not provide model sharding, P2P routing, NAT traversal, public prompt serving, payments, or incentives.",
        ],
    }
    report["output_request"] = output_request_summary()
    report["prompt_scope"] = prompt_scope_summary(report)
    report["answer_scope"] = answer_scope_summary()
    report["shareable_summary"] = shareable_summary()
    report = attach_user_guidance(report, args, output_dir=output_dir)

    json_out = Path(args.json_out).resolve() if args.json_out else output_dir / "cpu_inference_beta_rc.json"
    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else output_dir / "cpu_inference_beta_rc.md"
    support_out = output_dir / "support_bundle.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    report["artifacts"]["cpu_inference_beta_rc_json"] = artifact_entry(
        json_out,
        output_dir,
        kind="cpu_inference_beta_rc",
        schema=SCHEMA,
        ok=ok,
    )
    report["artifacts"]["cpu_inference_beta_rc_markdown"] = artifact_entry(
        markdown_out,
        output_dir,
        kind="cpu_inference_beta_rc_markdown",
    )
    report["artifacts"]["support_bundle_json"] = artifact_entry(
        support_out,
        output_dir,
        kind="cpu_inference_beta_rc_support_bundle",
        schema="cpu_inference_beta_rc_support_bundle_v1",
        ok=report.get("ok"),
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    if isinstance(report.get("review_summary"), dict):
        report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
        report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
    report = support_bundle.sanitize(report)
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"sensitive_output_detected"})
        report["safety_error"] = "CPU Inference Beta RC report contained secret-like fragments"
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    support_out.write_text(json.dumps(support_bundle_payload(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifacts"]["cpu_inference_beta_rc_json"]["present"] = json_out.is_file()
    report["artifacts"]["cpu_inference_beta_rc_markdown"]["present"] = markdown_out.is_file()
    report["artifacts"]["support_bundle_json"]["present"] = support_out.is_file()
    report["artifact_summary"] = artifact_summary(output_dir)
    if isinstance(report.get("review_summary"), dict):
        report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
        report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    support_out.write_text(json.dumps(support_bundle_payload(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    next_items = report.get("next_commands") if isinstance(report.get("next_commands"), list) else []
    lines = [
        "# CrowdTensor CPU Inference Beta RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
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
        f"- output request note: {output_request.get('summary') or 'Public artifacts summarize CPU inference RC evidence only and do not include answer text.'}",
        f"- prompt scope: `{prompt_scope_text(prompt_scope)}`",
        f"- prompt scope note: {prompt_scope.get('summary') or 'Public artifacts record prompt source/count only.'}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope.get('summary') or 'Public artifacts contain no local answer transcript.'}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generation_public={shareable.get('raw_generation_public')} raw_external_llm_output_public={shareable.get('raw_external_llm_output_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
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
        "## Steps",
        "",
    ])
    for step in report.get("steps") or []:
        lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
    lines.extend(["", "## Diagnosis", ""])
    lines.append(", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`")
    lines.extend(["", "## Not Completed", ""])
    not_completed = report.get("not_completed") or []
    lines.extend(f"- {item}" for item in not_completed) if not_completed else lines.append("- none")
    lines.extend(["", "## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CPU Inference Beta RC evidence pack.")
    parser.add_argument("--output-dir", default="dist/cpu-infer-beta-rc")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--base-port", type=int, default=9070)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--external-llm-request-count", type=int, default=2)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--remote-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--verify-timeout", type=float, default=40.0)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--two-machine-timeout-seconds", type=int, default=240)
    parser.add_argument("--kaggle-timeout-seconds", type=int, default=240)
    parser.add_argument("--kaggle-coordinator-url", default="https://YOUR_COORDINATOR_HOST")
    parser.add_argument("--kaggle-miner-id", default="kaggle-cpu-1")
    parser.add_argument("--kaggle-real-runtime-report", default="")
    parser.add_argument("--quick", action="store_true", help="use one request per workload for fast CI validation")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.request_count = min(args.request_count, 1)
        args.external_llm_request_count = min(args.external_llm_request_count, 1)
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.external_llm_request_count < 1:
        raise SystemExit("--external-llm-request-count must be at least 1")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be at least 1")
    if args.remote_timeout_seconds < 0:
        raise SystemExit("--remote-timeout-seconds must be non-negative")
    if args.startup_timeout < 0 or args.verify_timeout < 0 or args.command_timeout <= 0:
        raise SystemExit("--startup-timeout/--verify-timeout must be non-negative and --command-timeout positive")
    if args.two_machine_timeout_seconds < 1 or args.kaggle_timeout_seconds < 1:
        raise SystemExit("--two-machine-timeout-seconds and --kaggle-timeout-seconds must be at least 1")
    return args


def print_human(report: dict[str, Any]) -> None:
    user_status_report = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    print("CrowdTensor CPU Inference Beta RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    if user_status_report:
        print(
            "  status: "
            f"{user_status_report.get('state')} "
            f"next={user_status_report.get('next_step')} "
            f"recommended={user_status_report.get('recommended_label')}"
        )
    if review:
        print(
            "  review: "
            f"state={review.get('state')} next={review.get('next_step')} "
            f"inspect={review.get('inspect_first')} attention={review.get('attention')}"
        )
        print(f"  review_next: {review.get('next_command') or 'none'}")
    if recommended:
        print(
            "  recommended_next: "
            f"{recommended.get('label')} reason={recommended.get('reason')} {recommended.get('command_line')}"
        )
    if prompt_scope:
        print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
        if prompt_scope.get("summary"):
            print(f"  prompt_scope_note: {prompt_scope.get('summary')}")
    if output_request:
        print(
            "  output_request: "
            f"include_output={output_request.get('include_output')} "
            f"raw_prompt_public={output_request.get('raw_prompt_public')} "
            f"raw_generated_text_public={output_request.get('raw_generation_public')} "
            f"raw_external_llm_output_public={output_request.get('raw_external_llm_output_public')} "
            f"public_artifact_safe={output_request.get('public_artifact_safe')}"
        )
    if answer_scope:
        print(
            "  answer_scope: "
            f"state={answer_scope.get('scope_state')} "
            f"terminal_only={answer_scope.get('terminal_only')} "
            f"saved_json={answer_scope.get('saved_json_display')} "
            f"saved_markdown={answer_scope.get('saved_markdown_display')} "
            f"raw_generated_text_public={answer_scope.get('raw_generation_public')} "
            f"raw_external_llm_output_public={answer_scope.get('raw_external_llm_output_public')} "
            f"public_artifact_safe={answer_scope.get('public_artifact_safe')}"
        )
        if answer_scope.get("summary"):
            print(f"  answer_scope_note: {answer_scope.get('summary')}")
    if shareable:
        print(
            "  shareable: "
            f"saved_artifacts={shareable.get('saved_artifacts_public_safe')} "
            f"raw_prompt_public={shareable.get('raw_prompt_public')} "
            f"raw_generated_text_public={shareable.get('raw_generation_public')} "
            f"raw_external_llm_output_public={shareable.get('raw_external_llm_output_public')} "
            f"answer_scope_state={shareable.get('answer_scope_state')} "
            f"terminal_only={shareable.get('local_answer_terminal_only')} "
            f"public_artifact_safe={shareable.get('public_artifact_safe')}"
        )
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for index, item in enumerate((report.get("next_commands") or []), start=1):
        if not isinstance(item, dict):
            continue
        suffix = ""
        if item.get("requires_private_credentials"):
            suffix += " (requires private credentials)"
        if item.get("side_effectful"):
            suffix += " side_effectful=True"
        print(f"  next[{index}] {item.get('label')}: {item.get('command_line')}{suffix}")
    if artifact_report:
        print(
            "  artifacts: "
            f"present={artifact_report.get('present_artifact_count')}/{artifact_report.get('artifact_count')} "
            f"support={artifact_report.get('support_bundle')} "
            f"public_artifact_safe={bool(artifact_report.get('public_artifact_safe'))}"
        )
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


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
