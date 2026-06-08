#!/usr/bin/env python3
"""Build the CPU-only remote pipeline-sharded inference Beta report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
import sharded_inference_evidence_pack as sharded_pack  # noqa: E402


SCHEMA = "remote_sharded_inference_beta_v1"
WORKLOAD_TYPE = "sharded_model_bundle_infer"
FAILURE_NONE = "none"
FAILURE_KILL_STAGE_AFTER_CLAIM = "kill-stage-after-claim"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "lease_token",
    "idempotency_key",
    "sharded_inference_result",
    "activation_results",
    "activation_result",
    "logits",
    "inference_results",
    "inference_result",
    "Bearer ",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


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


def redact_text(text: str, secret_values: list[str] | None = None) -> str:
    redacted = text
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
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:], secret_values)
    return step, payload


def diagnosis_codes(*payloads: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
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


def report_kind() -> str:
    if SCHEMA == "remote_micro_llm_sharded_beta_v1":
        return "micro-llm"
    if SCHEMA == "remote_real_llm_sharded_beta_v1":
        return "real-llm"
    return "model-bundle"


def report_title() -> str:
    kind = report_kind()
    if kind == "micro-llm":
        return "Remote micro-LLM sharded inference Beta"
    if kind == "real-llm":
        return "Remote real tiny-LLM sharded inference Beta"
    return "Remote sharded inference Beta"


def report_slug() -> str:
    if SCHEMA == "remote_micro_llm_sharded_beta_v1":
        return "remote_micro_llm_sharded_beta"
    if SCHEMA == "remote_real_llm_sharded_beta_v1":
        return "remote_real_llm_sharded_beta"
    return "remote_sharded_inference_beta"


def local_command_name() -> str:
    kind = report_kind()
    if kind == "micro-llm":
        return "micro-llm-shard-infer"
    if kind == "real-llm":
        return "real-llm-shard-infer"
    return "shard-infer"


def beta_command_name() -> str:
    kind = report_kind()
    if kind == "micro-llm":
        return "micro-llm-shard-infer-beta"
    if kind == "real-llm":
        return "real-llm-shard-infer-beta"
    return "shard-infer-beta"


def report_file_names() -> tuple[str, str]:
    slug = report_slug()
    return f"{slug}.json", f"{slug}.md"


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


def command_entry(
    label: str,
    command: list[Any] | str,
    *,
    reason: str,
    side_effectful: bool = False,
    public_artifact_safe: bool = True,
) -> dict[str, Any]:
    if isinstance(command, str):
        command_line = command
    else:
        command_line = " ".join(str(part) for part in command)
    return {
        "label": label,
        "command_line": command_line,
        "reason": reason,
        "side_effectful": side_effectful,
        "public_artifact_safe": public_artifact_safe,
    }


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    json_name, md_name = report_file_names()
    paths = {
        "inspect_first": output_dir / md_name,
        "summary_json": output_dir / json_name,
        "summary_markdown": output_dir / md_name,
        "support_bundle": output_dir / "support_bundle.json",
    }
    kind = report_kind()
    if kind == "real-llm":
        detail = "tiny HF stage routing, baseline/decoded-token checks, hashes, redaction status, and diagnostics only"
    elif kind == "micro-llm":
        detail = "micro-LLM stage routing, decoded-token checks, hashes, redaction status, and diagnostics only"
    else:
        detail = "model-bundle route/stage readiness, activation hashes, baseline match, redaction status, and diagnostics only"
    return {
        "schema": f"{report_slug()}_artifact_summary_v1",
        "inspect_first": str(paths["inspect_first"]),
        "summary_json": str(paths["summary_json"]),
        "summary_markdown": str(paths["summary_markdown"]),
        "support_bundle": str(paths["support_bundle"]),
        "shareable_paths": [str(path) for path in paths.values()],
        "artifact_count": len(paths),
        "present_artifact_count": sum(1 for path in paths.values() if path.is_file()),
        "raw_prompt_public": False,
        "raw_result_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": f"Open inspect_first first, then support_bundle for diagnostics. Artifacts contain {detail}.",
    }


def output_request_summary() -> dict[str, Any]:
    kind = report_kind()
    if kind == "real-llm":
        summary = (
            "Remote real tiny-LLM Beta records routing and validation summaries only; raw prompts, raw generated text, "
            "token ids, and intermediate activations are excluded from public artifacts."
        )
    elif kind == "micro-llm":
        summary = (
            "Remote micro-LLM Beta records fixed prompt/decode validation summaries only; raw prompts, raw generated text, "
            "token ids, and intermediate activations are excluded from public artifacts."
        )
    else:
        summary = (
            "Remote sharded model-bundle Beta records route readiness and validation summaries only; raw prompts, "
            "raw outputs, and intermediate tensors are excluded."
        )
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_result_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": summary,
    }


def _first_session_summary(report: dict[str, Any]) -> dict[str, Any]:
    summaries = report.get("payload_summaries") if isinstance(report.get("payload_summaries"), dict) else {}
    for summary in summaries.values():
        if not isinstance(summary, dict):
            continue
        session = summary.get("session") if isinstance(summary.get("session"), dict) else {}
        if session:
            return session
    return {}


def prompt_scope_summary(report: dict[str, Any]) -> dict[str, Any]:
    session = _first_session_summary(report)
    try:
        prompt_count = int(session.get("prompt_request_count") or session.get("request_count") or report.get("request_count") or 0)
    except (TypeError, ValueError):
        prompt_count = 0
    kind = report_kind()
    source = "fixed-model-bundle-scenario"
    if kind == "micro-llm":
        source = "fixed-micro-llm-prompt-scenario"
    elif kind == "real-llm":
        source = "operator-provided-or-default-tiny-hf-prompts"
    return {
        "source": source,
        "prompt_count": prompt_count,
        "scenario_id": session.get("scenario_id") or report.get("scenario_id"),
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": True,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Public artifacts identify prompt source, scenario, and counts only; raw prompt text is not included."
        ),
    }


def prompt_scope_text(prompt_scope: dict[str, Any]) -> str:
    return (
        f"source={prompt_scope.get('source') or 'unknown'} "
        f"count={prompt_scope.get('prompt_count')} "
        f"scenario_id={prompt_scope.get('scenario_id') or 'none'} "
        f"inline_prompt_text={bool(prompt_scope.get('inline_prompt_text'))} "
        f"terminal_next_commands_local_private={bool(prompt_scope.get('terminal_next_commands_local_private'))} "
        f"saved_artifacts_prompt_placeholders={bool(prompt_scope.get('saved_artifacts_prompt_placeholders'))} "
        f"prompt_file_path_public={bool(prompt_scope.get('prompt_file_path_public'))} "
        f"raw_prompt_public={bool(prompt_scope.get('raw_prompt_public'))} "
        f"public_artifact_safe={bool(prompt_scope.get('public_artifact_safe'))}"
    )


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "evidence-only",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "validation-summary-only",
        "saved_markdown_display": "validation-summary-only",
        "json_stdout_display": "validation-summary-only",
        "raw_result_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "The Beta report proves stage routing and validation; it does not save or print raw model outputs."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    json_name, md_name = report_file_names()
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_result_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "evidence-only",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            f"Share {json_name}, {md_name}, and support_bundle.json; they contain readiness, hashes, "
            "redaction status, and diagnostics only."
        ),
    }


def sharded_payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    stage = payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "diagnosis_codes": diagnosis_codes(payload),
        "session": {
            "schema": session.get("schema"),
            "session_id": session.get("session_id"),
            "stage_count": session.get("stage_count"),
            "stage_0_task_id": session.get("stage_0_task_id"),
            "stage_1_task_id": session.get("stage_1_task_id"),
            "request_count": session.get("request_count"),
            "scenario_id": session.get("scenario_id"),
        },
        "stage_summary": {
            "stage_0": stage.get("stage_0") or {},
            "stage_1": stage.get("stage_1") or {},
        },
        "safety": {
            "read_only": safety.get("read_only"),
            "redaction_ok": safety.get("redaction_ok"),
            "raw_activation_redacted": safety.get("raw_activation_redacted"),
            "not_production": safety.get("not_production"),
        },
    }


def build_local(args: argparse.Namespace, *, runner: Runner) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    local_dir = output_dir / "local-shard-infer"
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "shard-infer",
        "--output-dir",
        str(local_dir),
        "--port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    step, payload = run_json_step(
        "local_sharded_inference",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    return {
        "mode": "local",
        "steps": [step],
        "payload_summaries": {"local_sharded_inference": sharded_payload_summary(payload)},
        "artifacts": {
            "local_sharded_inference_cli_summary": artifact_entry(
                local_dir / "sharded_inference_cli_summary.json",
                output_dir,
                kind="sharded_inference_cli_summary",
                schema="sharded_inference_cli_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "local_sharded_inference_evidence_json": artifact_entry(
                local_dir / "sharded_inference_evidence.json",
                output_dir,
                kind="sharded_inference_evidence",
                schema="sharded_inference_evidence_v1",
            ),
        },
        "diagnosis_codes": diagnosis_codes(payload),
    }


def build_remote_loopback(args: argparse.Namespace, *, runner: Runner) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    loopback_dir = output_dir / "remote-loopback-shard-infer"
    evidence_json = loopback_dir / "sharded_inference_evidence.json"
    evidence_md = loopback_dir / "sharded_inference_evidence.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "sharded_inference_evidence_pack.py"),
        "--port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--miner-prefix",
        "remote-shard-miner",
        "--invite-token-prefix",
        "remote-sharded-token",
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    step, payload = run_json_step(
        "remote_loopback_sharded_inference",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    return {
        "mode": "remote-loopback",
        "steps": [step],
        "payload_summaries": {"remote_loopback_sharded_inference": sharded_payload_summary(payload)},
        "artifacts": {
            "remote_loopback_sharded_inference_evidence_json": artifact_entry(
                evidence_json,
                output_dir,
                kind="sharded_inference_evidence",
                schema="sharded_inference_evidence_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "remote_loopback_sharded_inference_evidence_markdown": artifact_entry(
                evidence_md,
                output_dir,
                kind="sharded_inference_evidence_markdown",
            ),
        },
        "diagnosis_codes": diagnosis_codes(payload),
    }


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    admin_token: str = "",
    observer_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    return sharded_pack.request_json(
        method,
        base_url,
        path,
        payload=payload,
        admin_token=admin_token,
        observer_token=observer_token,
        timeout=timeout,
    )


def wait_for_remote_completion(args: argparse.Namespace, session_id: str) -> tuple[bool, dict[str, Any]]:
    deadline = time.monotonic() + args.remote_timeout_seconds
    last_state: dict[str, Any] = {}
    max_new_tokens = max(1, int(getattr(args, "max_new_tokens", 1)))
    while time.monotonic() < deadline:
        last_state = request_json(
            "GET",
            args.coordinator_url,
            "/state",
            observer_token=args.observer_token,
            timeout=args.http_timeout,
        )
        tasks = sharded_pack.session_tasks(last_state, session_id)
        completed_stage0 = {
            int((task.get("workload_metadata") or {}).get("generation_step", 0))
            for task in tasks
            if int((task.get("workload_metadata") or {}).get("stage_id", -1)) == 0
            and task.get("status") == "completed"
        }
        completed_stage1 = {
            int((task.get("workload_metadata") or {}).get("generation_step", 0))
            for task in tasks
            if int((task.get("workload_metadata") or {}).get("stage_id", -1)) == 1
            and task.get("status") == "completed"
        }
        expected = set(range(max_new_tokens))
        if expected.issubset(completed_stage0) and expected.issubset(completed_stage1):
            return True, last_state
        time.sleep(args.poll_interval)
    return False, last_state


def admin_results_for_session(args: argparse.Namespace, session_id: str) -> list[dict[str, Any]]:
    query = urlencode({
        "status": "accepted",
        "workload_type": WORKLOAD_TYPE,
        "session_id": session_id,
        "limit": 10,
    })
    payload = request_json(
        "GET",
        args.coordinator_url,
        f"/admin/results?{query}",
        admin_token=args.admin_token,
        timeout=args.http_timeout,
    )
    rows = payload.get("results") if isinstance(payload, dict) else []
    return rows if isinstance(rows, list) else []


def build_remote_existing(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    existing_dir = output_dir / "remote-existing-shard-infer"
    evidence_json = existing_dir / "sharded_inference_evidence.json"
    evidence_md = existing_dir / "sharded_inference_evidence.md"
    started = time.monotonic()
    step: dict[str, Any] = {
        "name": "remote_existing_sharded_inference",
        "ok": False,
        "returncode": None,
    }
    try:
        session = request_json(
            "POST",
            args.coordinator_url,
            "/admin/inference-sessions",
            payload={
                "request_count": args.request_count,
                "scenario_id": args.scenario_id,
                "workload_type": WORKLOAD_TYPE,
            },
            admin_token=args.admin_token,
            timeout=args.http_timeout,
        )
        session_id = str(session.get("session_id") or "")
        completed, state = wait_for_remote_completion(args, session_id)
        rows = admin_results_for_session(args, session_id) if session_id else []
        report_args = argparse.Namespace(
            base_url=args.coordinator_url,
            admin_token=args.admin_token,
            observer_token=args.observer_token,
            stage_mode=args.stage_mode,
            require_distinct_stage_miners=args.require_distinct_stage_miners,
        )
        evidence = sharded_pack.build_report(
            args=report_args,
            session=session,
            state=state,
            stage_processes=[],
            requeue_summary={
                "enabled": False,
                "failure_mode": FAILURE_NONE,
                "victim_stage_id": None,
                "victim_task_id": "",
                "rescue_miner_id": "",
                "lease_expired": False,
                "rescued_result": False,
                "victim_result_accepted": False,
            },
            ledger_rows=rows,
        )
        evidence_json.parent.mkdir(parents=True, exist_ok=True)
        evidence_json.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evidence_md.write_text(sharded_pack.render_markdown(evidence), encoding="utf-8")
        step["ok"] = bool(completed and evidence.get("ok"))
        step["payload_schema"] = evidence.get("schema")
        step["payload_ok"] = evidence.get("ok")
        if not completed:
            step["error"] = "remote_timeout_waiting_for_stages"
    except Exception as exc:  # pragma: no cover - exercised through check behavior
        evidence = {}
        step["error"] = str(exc)
    step["duration_seconds"] = round(time.monotonic() - started, 3)
    return {
        "mode": "remote-existing",
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "steps": [step],
        "payload_summaries": {"remote_existing_sharded_inference": sharded_payload_summary(evidence)},
        "artifacts": {
            "remote_existing_sharded_inference_evidence_json": artifact_entry(
                evidence_json,
                output_dir,
                kind="sharded_inference_evidence",
                schema="sharded_inference_evidence_v1",
                ok=evidence.get("ok") if evidence else None,
            ),
            "remote_existing_sharded_inference_evidence_markdown": artifact_entry(
                evidence_md,
                output_dir,
                kind="sharded_inference_evidence_markdown",
            ),
        },
        "diagnosis_codes": diagnosis_codes(evidence),
    }


def mode_ready_code(mode: str) -> str:
    if mode == "remote-loopback":
        return "remote_sharded_loopback_ready"
    if mode == "remote-existing":
        return "remote_sharded_existing_ready"
    return "local_sharded_inference_ready"


def base_ready_code() -> str:
    kind = report_kind()
    if kind == "micro-llm":
        return "micro_llm_sharded_ready"
    if kind == "real-llm":
        return "real_llm_sharded_ready"
    return "sharded_inference_ready"


def wrapper_ready_code() -> str:
    if SCHEMA == "remote_micro_llm_sharded_beta_v1":
        return "remote_micro_llm_sharded_ready"
    if SCHEMA == "remote_real_llm_sharded_beta_v1":
        return "remote_real_llm_sharded_ready"
    return "remote_sharded_inference_ready"


def wrapper_failed_code() -> str:
    if SCHEMA == "remote_micro_llm_sharded_beta_v1":
        return "remote_micro_llm_sharded_failed"
    if SCHEMA == "remote_real_llm_sharded_beta_v1":
        return "remote_real_llm_sharded_failed"
    return "remote_sharded_inference_failed"


def beta_command(args: argparse.Namespace, output_dir: Path) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        beta_command_name(),
        "--mode",
        getattr(args, "mode", "remote-loopback"),
        "--output-dir",
        str(output_dir),
        "--base-port",
        getattr(args, "base_port", 9830),
        "--request-count",
        getattr(args, "request_count", 4),
        "--failure-mode",
        getattr(args, "failure_mode", FAILURE_NONE),
        "--stage-mode",
        getattr(args, "stage_mode", "both"),
        "--json",
    ]
    if getattr(args, "scenario_id", "") and report_kind() == "model-bundle":
        command.extend(["--scenario-id", getattr(args, "scenario_id")])
    if getattr(args, "decode_steps", None) is not None:
        command.extend(["--decode-steps", getattr(args, "decode_steps")])
    if getattr(args, "max_new_tokens", None) is not None:
        command.extend(["--max-new-tokens", getattr(args, "max_new_tokens")])
    if getattr(args, "hf_model_id", ""):
        command.extend(["--hf-model-id", getattr(args, "hf_model_id")])
    if getattr(args, "real_llm_backend", ""):
        command.extend(["--real-llm-backend", getattr(args, "real_llm_backend")])
    if getattr(args, "real_llm_partition_mode", ""):
        command.extend(["--real-llm-partition-mode", getattr(args, "real_llm_partition_mode")])
    if getattr(args, "require_distinct_stage_miners", False):
        command.append("--require-distinct-stage-miners")
    if getattr(args, "micro_llm_artifact", ""):
        command.extend(["--micro-llm-artifact", str(getattr(args, "micro_llm_artifact"))])
    if getattr(args, "prompt_texts_file", ""):
        command.extend(["--prompt-texts-file", "<prompt-file>"])
    elif getattr(args, "prompt_texts", "") and report_kind() in {"micro-llm", "real-llm"}:
        command.extend(["--prompt-texts", "<redacted-prompts>"])
    if getattr(args, "mode", "") == "remote-existing":
        if getattr(args, "coordinator_url", ""):
            command.extend(["--coordinator-url", getattr(args, "coordinator_url")])
        command.extend(["--observer-token", "<observer-token>", "--admin-token", "<admin-token>"])
    return command


def local_inference_command(args: argparse.Namespace, output_dir: Path) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        local_command_name(),
        "--output-dir",
        str(output_dir / "local-rerun"),
        "--port",
        getattr(args, "base_port", 9830),
        "--request-count",
        getattr(args, "request_count", 4),
        "--failure-mode",
        getattr(args, "failure_mode", FAILURE_NONE),
        "--stage-mode",
        getattr(args, "stage_mode", "both"),
        "--json",
    ]
    if getattr(args, "scenario_id", "") and report_kind() == "model-bundle":
        command.extend(["--scenario-id", getattr(args, "scenario_id")])
    if getattr(args, "decode_steps", None) is not None:
        command.extend(["--decode-steps", getattr(args, "decode_steps")])
    if getattr(args, "max_new_tokens", None) is not None:
        command.extend(["--max-new-tokens", getattr(args, "max_new_tokens")])
    if getattr(args, "hf_model_id", ""):
        command.extend(["--hf-model-id", getattr(args, "hf_model_id")])
    if getattr(args, "real_llm_backend", ""):
        command.extend(["--real-llm-backend", getattr(args, "real_llm_backend")])
    if getattr(args, "real_llm_partition_mode", ""):
        command.extend(["--real-llm-partition-mode", getattr(args, "real_llm_partition_mode")])
    if getattr(args, "require_distinct_stage_miners", False):
        command.append("--require-distinct-stage-miners")
    if getattr(args, "micro_llm_artifact", ""):
        command.extend(["--micro-llm-artifact", str(getattr(args, "micro_llm_artifact"))])
    if getattr(args, "prompt_texts_file", ""):
        command.extend(["--prompt-texts-file", "<prompt-file>"])
    elif getattr(args, "prompt_texts", "") and report_kind() in {"micro-llm", "real-llm"}:
        command.extend(["--prompt-texts", "<redacted-prompts>"])
    return command


def not_completed_items(report: dict[str, Any], *, output_dir: Path) -> list[str]:
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []) if code)
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    items: list[tuple[str, Any]] = [
        (f"{report_slug()} ready", report.get("ok") is True and wrapper_ready_code() in codes),
        ("mode readiness ready", mode_ready_code(str(report.get("mode") or "")) in codes),
        ("child sharded evidence ready", base_ready_code() in codes),
        ("all wrapper steps passed", all(bool(step.get("ok")) for step in report.get("steps") or [])),
        ("CPU-only default boundary present", safety.get("cpu_only_default") is True),
        ("read-only workload boundary present", bool(safety.get("read_only_workload"))),
        ("activation payload redaction present", safety.get("activation_payloads_redacted") is True),
        ("captured output redaction present", safety.get("captured_output_redacted") is True),
        ("not production boundary present", safety.get("not_production") is True),
        ("summary JSON artifact present", (artifacts.get(f"{report_slug()}_json") or {}).get("present") is True),
        ("summary Markdown artifact present", (artifacts.get(f"{report_slug()}_markdown") or {}).get("present") is True),
        ("support bundle artifact present", (output_dir / "support_bundle.json").is_file()),
    ]
    if str(report.get("failure_mode") or "") != FAILURE_NONE:
        items.append(("stage requeue ready", "stage_requeue_ready" in codes))
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
    _, md_name = report_file_names()
    if report.get("ok") is True and not missing:
        return command_entry(
            f"inspect {report_title()} evidence",
            artifact_command(output_dir, md_name),
            reason="review_artifacts",
        )
    return command_entry(
        f"rerun {report_title()}",
        beta_command(args, output_dir),
        reason="fix_remote_sharded_beta_blockers" if missing else "rerun_remote_sharded_beta",
        side_effectful=True,
    )


def next_commands(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    _, md_name = report_file_names()
    commands = [
        command_entry(
            "inspect shareable summary",
            artifact_command(output_dir, md_name),
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
        f"refresh {report_title()}",
        beta_command(args, output_dir),
        reason="refresh_remote_sharded_beta",
        side_effectful=True,
    )
    if all(item.get("command_line") != refresh.get("command_line") for item in commands):
        commands.append(refresh)
    if report.get("mode") != "local":
        local = command_entry(
            f"compare with local {local_command_name()}",
            local_inference_command(args, output_dir),
            reason="compare_local_cpu_baseline",
            side_effectful=True,
        )
        if all(item.get("command_line") != local.get("command_line") for item in commands):
            commands.append(local)
    return commands


def user_status(report: dict[str, Any], *, recommended: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    mode = str(report.get("mode") or "unknown")
    if report.get("ok") is True and not missing:
        state = "ready"
        headline = f"{report_title()} evidence is ready."
        next_step = "review_artifacts"
    else:
        state = "blocked"
        headline = f"{report_title()} evidence needs attention."
        next_step = "fix_blockers"
    if mode == "remote-loopback":
        proof_level = "local-loopback-remote-stand-in"
    elif mode == "remote-existing":
        proof_level = "external-existing-runtime"
    else:
        proof_level = "local-cpu"
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "mode": mode,
        "proof_level": proof_level,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
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
        "schema": f"{report_slug()}_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "ready": ready,
        "next_step": "review_artifacts" if ready else "fix_remote_sharded_beta_blockers",
        "inspect_first": artifacts["inspect_first"],
        "support_bundle": artifacts["support_bundle"],
        "recommended_next_command": recommended,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": (report.get("diagnosis_codes") or ["none"])[0],
        "not_completed_count": len(missing),
        "not_completed": list(missing),
        "mode": report.get("mode"),
        "proof_level": (report.get("user_status") or {}).get("proof_level"),
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_result_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def support_bundle_payload(report: dict[str, Any]) -> dict[str, Any]:
    return support_bundle.sanitize({
        "schema": f"{report_slug()}_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": report.get("ok"),
        "mode": report.get("mode"),
        "proof_level": (report.get("user_status") or {}).get("proof_level"),
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
        "steps": report.get("steps"),
        "payload_summaries": report.get("payload_summaries"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
    })


def default_limitations() -> list[str]:
    kind = report_kind()
    if kind == "micro-llm":
        return [
            "CPU-only deterministic micro-LLM two-stage pipeline-sharded inference Beta; not production Swarm Inference",
            "Uses fixed tiny Transformer requests and activation hashes; not GGUF/llama.cpp or large-model serving",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, payments, or arbitrary prompt serving",
        ]
    if kind == "real-llm":
        return [
            "Tiny Hugging Face GPT two-stage pipeline-sharded inference Beta; optional CUDA backend is explicit and not production Swarm Inference",
            "Uses a tiny HF model and activation hashes; not GGUF/llama.cpp, GPU pooling marketplace, or large-model serving",
            "Does not provide P2P routing, NAT traversal, payments, or arbitrary public prompt serving",
        ]
    return [
        "CPU-only two-stage pipeline-sharded inference Beta; not production Swarm Inference",
        "Uses fixed model-bundle requests and activation hashes; not real LLM sharding",
        "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, payments, or arbitrary prompt serving",
    ]


def default_recommended_next_commands() -> list[str]:
    command = beta_command_name()
    if report_kind() == "micro-llm":
        return [
            f"crowdtensor {command} --mode remote-loopback --json",
            f"crowdtensor {command} --mode remote-loopback --stage-mode split --require-distinct-stage-miners --json",
        ]
    if report_kind() == "real-llm":
        return [
            f"crowdtensor {command} --mode remote-loopback --stage-mode split --json",
            "python3 scripts/remote_real_llm_sharded_beta_check.py --mode remote-loopback",
        ]
    return [
        f"crowdtensor {command} --mode remote-loopback --json",
        f"crowdtensor {command} --mode remote-loopback --failure-mode kill-stage-after-claim --json",
    ]


def attach_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    report["output_request"] = output_request_summary()
    report["prompt_scope"] = prompt_scope_summary(report)
    report["answer_scope"] = answer_scope_summary()
    report["shareable_summary"] = shareable_summary()
    report["artifact_summary"] = artifact_summary(output_dir)
    missing = not_completed_items(report, output_dir=output_dir)
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
    return report


def finalize_report_artifacts(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    json_out: Path,
    markdown_out: Path,
    support_path: Path,
    json_artifact_key: str,
    markdown_artifact_key: str,
    support_schema: str,
) -> dict[str, Any]:
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    support_path.parent.mkdir(parents=True, exist_ok=True)
    report["artifacts"][json_artifact_key]["present"] = True
    report["artifacts"][markdown_artifact_key]["present"] = True
    report["artifacts"]["support_bundle_json"] = artifact_entry(
        support_path,
        output_dir,
        kind="support_bundle",
        schema=support_schema,
        ok=report.get("ok"),
    )
    report["artifacts"]["support_bundle_json"]["present"] = support_path.is_file()
    report = attach_user_guidance(report, args, output_dir=output_dir)
    report = support_bundle.sanitize(redact_values(report, secret_values))
    support_path.write_text(json.dumps(support_bundle_payload(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifacts"]["support_bundle_json"] = artifact_entry(
        support_path,
        output_dir,
        kind="support_bundle",
        schema=support_schema,
        ok=report.get("ok"),
    )
    report["artifacts"]["support_bundle_json"]["present"] = True
    report = attach_user_guidance(report, args, output_dir=output_dir)
    report = support_bundle.sanitize(redact_values(report, secret_values))
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    support_path.write_text(json.dumps(support_bundle_payload(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    artifacts = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        f"# CrowdTensor {report_title()}",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- proof_level: `{user.get('proof_level')}`",
        f"- failure_mode: `{report.get('failure_mode')}`",
        f"- output_dir: `{report.get('output_dir')}`",
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
        f"- answer scope note: {answer_scope.get('summary') or 'Public artifacts contain no raw result.'}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Steps",
        "",
    ])
    for step in report.get("steps") or []:
        lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
    lines.extend([
        "",
        "## Boundaries",
        "",
    ])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "local":
        body = build_local(args, runner=runner)
    elif args.mode == "remote-loopback":
        body = build_remote_loopback(args, runner=runner)
    elif args.mode == "remote-existing":
        body = build_remote_existing(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")

    steps = body.get("steps") or []
    ok = all(bool(step.get("ok")) for step in steps)
    codes = set(body.get("diagnosis_codes") or [])
    if ok:
        codes.add(mode_ready_code(args.mode))
        codes.add(wrapper_ready_code())
    else:
        codes.add(wrapper_failed_code())
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": body.get("mode"),
        "output_dir": str(output_dir),
        "request_count": args.request_count,
        "scenario_id": args.scenario_id,
        "failure_mode": args.failure_mode,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "coordinator_url": body.get("coordinator_url"),
        "steps": steps,
        "payload_summaries": body.get("payload_summaries") or {},
        "artifacts": body.get("artifacts") or {},
        "diagnosis_codes": sorted(codes),
        "safety": {
            "cpu_only_default": True,
            "read_only_workload": WORKLOAD_TYPE,
            "activation_payloads_redacted": True,
            "captured_output_redacted": True,
            "not_production": True,
        },
        "limitations": default_limitations(),
        "recommended_next_commands": default_recommended_next_commands(),
    }
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    json_out = Path(args.json_out).resolve() if args.json_out else output_dir / report_file_names()[0]
    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else output_dir / report_file_names()[1]
    support_path = output_dir / "support_bundle.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    support_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_json_key = f"{report_slug()}_json"
    artifact_markdown_key = f"{report_slug()}_markdown"
    report["artifacts"][artifact_json_key] = artifact_entry(
        json_out,
        output_dir,
        kind=report_slug(),
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"][artifact_markdown_key] = artifact_entry(
        markdown_out,
        output_dir,
        kind=f"{report_slug()}_markdown",
    )
    report["artifacts"]["support_bundle_json"] = artifact_entry(
        support_path,
        output_dir,
        kind="support_bundle",
        schema=f"{report_slug()}_support_bundle_v1",
    )
    report = attach_user_guidance(report, args, output_dir=output_dir)
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"sensitive_output_detected"})
        report["safety_error"] = "remote sharded inference report contained secret-like fragments"
        report = attach_user_guidance(report, args, output_dir=output_dir)
        report = support_bundle.sanitize(redact_values(report, secret_values))

    return finalize_report_artifacts(
        report,
        args,
        output_dir=output_dir,
        json_out=json_out,
        markdown_out=markdown_out,
        support_path=support_path,
        json_artifact_key=artifact_json_key,
        markdown_artifact_key=artifact_markdown_key,
        support_schema=f"{report_slug()}_support_bundle_v1",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CPU-only remote pipeline-sharded inference Beta report.")
    parser.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing"], default="remote-loopback")
    parser.add_argument("--output-dir", default="dist/remote-sharded-inference")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--base-port", type=int, default=9830)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument(
        "--failure-mode",
        choices=[
            FAILURE_NONE,
            FAILURE_KILL_STAGE_AFTER_CLAIM,
            FAILURE_KILL_STAGE0_AFTER_CLAIM,
            FAILURE_KILL_STAGE1_AFTER_CLAIM,
        ],
        default=FAILURE_NONE,
    )
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--remote-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be at least 1")
    if args.remote_timeout_seconds <= 0:
        raise SystemExit("--remote-timeout-seconds must be positive")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")
    if args.http_timeout <= 0:
        raise SystemExit("--http-timeout must be positive")
    if args.mode == "remote-existing":
        missing = [
            name for name in ["coordinator_url", "observer_token", "admin_token"]
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
        if args.failure_mode != FAILURE_NONE:
            raise SystemExit("remote-existing does not orchestrate failure-mode kills; use remote-loopback for requeue proof")
    return args


def print_human(report: dict[str, Any]) -> None:
    user_status_value = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifacts = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    print(f"CrowdTensor {report_title()}")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    if user_status_value:
        print(
            "  status: "
            f"state={user_status_value.get('state')} "
            f"next={user_status_value.get('next_step')} "
            f"proof_level={user_status_value.get('proof_level')} "
            f"headline={user_status_value.get('headline')}"
        )
    if review:
        print(
            "  review: "
            f"state={review.get('state')} "
            f"next={review.get('next_step')} "
            f"inspect={review.get('inspect_first')} "
            f"recommended={review.get('recommended_label')} "
            f"primary={review.get('primary_code')} "
            f"not_completed={review.get('not_completed_count')} "
            f"public_artifact_safe={bool(review.get('public_artifact_safe'))}"
        )
    if recommended:
        print(f"  recommended_next: {recommended.get('command_line')}")
    for index, item in enumerate((report.get("next_commands") or []), start=1):
        if isinstance(item, dict):
            print(f"  next[{index}] {item.get('label')}: {item.get('command_line')}")
    if artifacts:
        print(
            "  artifacts: "
            f"inspect={artifacts.get('inspect_first')} "
            f"present={artifacts.get('present_artifact_count')}/{artifacts.get('artifact_count')} "
            f"support={artifacts.get('support_bundle')} "
            f"public_artifact_safe={bool(artifacts.get('public_artifact_safe'))}"
        )
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    if output_request:
        print(
            "  output_request: "
            f"include_output={bool(output_request.get('include_output'))} "
            f"raw_prompt_public={bool(output_request.get('raw_prompt_public'))} "
            f"raw_generated_text_public={bool(output_request.get('raw_generated_text_public'))} "
            f"generated_token_ids_public={bool(output_request.get('generated_token_ids_public'))} "
            f"public_artifact_safe={bool(output_request.get('public_artifact_safe'))}"
        )
    if prompt_scope:
        print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
    if answer_scope:
        print(
            "  answer_scope: "
            f"state={answer_scope.get('scope_state')} "
            f"saved_json={answer_scope.get('saved_json_display')} "
            f"saved_markdown={answer_scope.get('saved_markdown_display')} "
            f"raw_generated_text_public={bool(answer_scope.get('raw_generated_text_public'))} "
            f"generated_token_ids_public={bool(answer_scope.get('generated_token_ids_public'))} "
            f"public_artifact_safe={bool(answer_scope.get('public_artifact_safe'))}"
        )
    if shareable:
        print(
            "  shareable: "
            f"saved_artifacts={bool(shareable.get('saved_artifacts_public_safe'))} "
            f"raw_prompt_public={bool(shareable.get('raw_prompt_public'))} "
            f"raw_generated_text_public={bool(shareable.get('raw_generated_text_public'))} "
            f"generated_token_ids_public={bool(shareable.get('generated_token_ids_public'))} "
            f"answer_scope_state={shareable.get('answer_scope_state')} "
            f"local_answer_terminal_only={bool(shareable.get('local_answer_terminal_only'))}"
        )
    for item in report.get("not_completed") or []:
        print(f"  not_completed: {item}")
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
