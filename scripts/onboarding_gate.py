#!/usr/bin/env python3
"""Fresh clone onboarding gate for CrowdTensorD Alpha."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "onboarding_gate_v1"
USER_INFER_SMOKE_VALIDATION_SCHEMA = "user_infer_smoke_validation_v1"
USER_INFER_SMOKE_PROMPT = "CrowdTensor onboarding local infer smoke prompt"
SECRET_FRAGMENTS = (
    USER_INFER_SMOKE_PROMPT,
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
    "demo-manifest-token",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def redact_text(text: str) -> str:
    redacted = text
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_script(venv_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def command_display(command: list[str], *, output_dir: Path, venv_dir: Path) -> list[str]:
    replacements = {
        str(ROOT.resolve()): "<repo>",
        str(output_dir.resolve()): "<output_dir>",
        str(venv_dir.resolve()): "<venv>",
    }
    displayed: list[str] = []
    for part in command:
        item = str(part)
        for source, target in replacements.items():
            item = item.replace(source, target)
        displayed.append(redact_text(item))
    return displayed


def diagnosis_from_payload(payload: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for code in payload.get("diagnosis_codes") or []:
        if isinstance(code, str):
            codes.add(code)
    status = payload.get("release_status")
    if isinstance(status, dict):
        for code in status.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
    summary = payload.get("diagnosis_summary")
    if isinstance(summary, dict):
        for code in summary.get("codes") or []:
            if isinstance(code, str):
                codes.add(code)
    return sorted(codes)


def run_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    cwd: Path,
    timeout_seconds: int,
    output_dir: Path,
    venv_dir: Path,
    expect_json: bool = False,
    require_payload_ok: bool = False,
    stdin_text: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=stdin_text if stdin_text else None,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        step = {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
            "command": command_display(command, output_dir=output_dir, venv_dir=venv_dir),
        }
        if stdin_text:
            step["stdin_provided"] = True
            step["stdin_redacted"] = True
        return (step, {})

    elapsed = round(time.monotonic() - started, 3)
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": elapsed,
        "command": command_display(command, output_dir=output_dir, venv_dir=venv_dir),
        "stdout_line_count": len(completed.stdout.splitlines()),
        "stderr_line_count": len(completed.stderr.splitlines()),
    }
    if stdin_text:
        step["stdin_provided"] = True
        step["stdin_redacted"] = True
    payload: dict[str, Any] = {}
    if expect_json:
        try:
            payload = json_from_stdout(completed.stdout)
        except ValueError as exc:
            step["ok"] = False
            step["error"] = str(exc)
        else:
            step["payload_schema"] = payload.get("schema")
            step["payload_ok"] = payload.get("ok")
            codes = diagnosis_from_payload(payload)
            if codes:
                step["diagnosis_codes"] = codes
            if require_payload_ok and payload.get("ok") is not True:
                step["ok"] = False
                step["error"] = "payload_not_ok"
    if not step["ok"]:
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1000:])
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1000:])
    return step, payload


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


def skip_step(name: str, reason: str) -> dict[str, Any]:
    return {"name": name, "ok": False, "skipped": True, "reason": reason}


def validate_user_infer_smoke(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "infer_summary.json"
    markdown_path = output_dir / "infer_summary.md"
    errors: list[str] = []
    report: dict[str, Any] = {}
    markdown = ""
    if not summary_path.is_file():
        errors.append("infer_summary_json_missing")
    else:
        try:
            report = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("infer_summary_json_invalid")
    if not markdown_path.is_file():
        errors.append("infer_summary_markdown_missing")
    else:
        markdown = markdown_path.read_text(encoding="utf-8")

    verdict = report.get("inference_verdict") if isinstance(report.get("inference_verdict"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    output_display = report.get("output_display") if isinstance(report.get("output_display"), dict) else {}
    gpu_status = report.get("gpu_status") if isinstance(report.get("gpu_status"), dict) else {}
    shareable_terminal = (
        report.get("shareable_terminal") if isinstance(report.get("shareable_terminal"), dict) else {}
    )
    shareable_summary = (
        report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    )
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    diagnosis_codes = [str(code) for code in report.get("diagnosis_codes") or [] if isinstance(code, str)]

    if report:
        if report.get("schema") != "crowdtensor_infer_cli_v1":
            errors.append("infer_schema_mismatch")
        if report.get("ok") is not True:
            errors.append("infer_not_ok")
        for code in ["crowdtensor_infer_ready", "user_friendly_infer_ready"]:
            if code not in diagnosis_codes:
                errors.append(f"missing_{code}")
        if verdict.get("schema") != "crowdtensor_inference_verdict_v1":
            errors.append("verdict_schema_mismatch")
        if verdict.get("state") != "completed" or verdict.get("completed") is not True:
            errors.append("verdict_not_completed")
        if verdict.get("answer_scope_state") != "shareable-terminal-redacted":
            errors.append("verdict_answer_scope_mismatch")
        if verdict.get("answer_visible_in_terminal") is not False:
            errors.append("verdict_answer_visibility_mismatch")
        if verdict.get("evidence_level") != "local-cpu-loopback":
            errors.append("verdict_evidence_level_mismatch")
        if verdict.get("executed_where") != "local-cpu":
            errors.append("verdict_executed_where_mismatch")
        if verdict.get("gpu_state") != "local-cpu-only":
            errors.append("verdict_gpu_state_mismatch")
        if verdict.get("fresh_kaggle_gpu_verified") is not False:
            errors.append("verdict_fresh_kaggle_gpu_mismatch")
        if verdict.get("public_artifact_safe") is not True:
            errors.append("verdict_public_artifact_safe_mismatch")
        if answer_scope.get("scope_state") != "shareable-terminal-redacted":
            errors.append("answer_scope_mismatch")
        if answer_scope.get("visible_in_terminal") is not False:
            errors.append("answer_scope_visible_mismatch")
        if answer_scope.get("public_artifact_safe") is not True:
            errors.append("answer_scope_public_artifact_safe_mismatch")
        if output_display.get("terminal_display") != "shareable-terminal-redacted":
            errors.append("output_display_terminal_mismatch")
        if output_display.get("terminal_text_available") is not False:
            errors.append("output_display_terminal_text_mismatch")
        if output_display.get("saved_artifact_display") != "hash-only":
            errors.append("output_display_saved_artifact_mismatch")
        if output_display.get("public_artifact_safe") is not True:
            errors.append("output_display_public_artifact_safe_mismatch")
        if gpu_status.get("state") != "local-cpu-only":
            errors.append("gpu_status_state_mismatch")
        if gpu_status.get("fresh_kaggle_gpu_attempted") is not False:
            errors.append("gpu_status_fresh_kaggle_attempt_mismatch")
        if gpu_status.get("fresh_kaggle_gpu_verified") is not False:
            errors.append("gpu_status_fresh_kaggle_verified_mismatch")
        if shareable_terminal.get("enabled") is not True:
            errors.append("shareable_terminal_not_enabled")
        if shareable_terminal.get("prompt_sources_redacted") is not True:
            errors.append("shareable_terminal_prompt_sources_mismatch")
        if shareable_terminal.get("answer_text_redacted") is not True:
            errors.append("shareable_terminal_answer_text_mismatch")
        if shareable_terminal.get("public_artifact_safe") is not True:
            errors.append("shareable_terminal_public_artifact_safe_mismatch")
        for key in ["raw_prompt_public", "raw_generated_text_public", "generated_token_ids_public"]:
            if safety.get(key) is not False:
                errors.append(f"safety_{key}_mismatch")
            if shareable_summary.get(key) is not False:
                errors.append(f"shareable_summary_{key}_mismatch")

    combined = (json.dumps(report, sort_keys=True) if report else "") + "\n" + markdown
    for fragment in [
        USER_INFER_SMOKE_PROMPT,
        '"generated_token_ids": [',
        '"input_ids":',
        '"hidden_state":',
        '"logits":',
        '"activation_results":',
        "lease_token",
        "idempotency_key",
        "Bearer ",
    ]:
        if fragment in combined:
            errors.append(f"artifact_leaked_{fragment[:24]}")

    return {
        "schema": USER_INFER_SMOKE_VALIDATION_SCHEMA,
        "ok": not errors,
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "inference_verdict": verdict,
        "answer_scope_state": answer_scope.get("scope_state") or "",
        "output_display": output_display.get("terminal_display") or "",
        "gpu_state": gpu_status.get("state") or "",
        "fresh_kaggle_gpu_attempted": bool(gpu_status.get("fresh_kaggle_gpu_attempted")),
        "fresh_kaggle_gpu_verified": bool(gpu_status.get("fresh_kaggle_gpu_verified")),
        "shareable_terminal": shareable_terminal,
        "diagnosis_codes": ["user_infer_smoke_ready"] if not errors else ["user_infer_smoke_failed"],
        "errors": errors,
        "public_artifact_safe": not errors,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Onboarding Gate",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- ok: `{payload.get('ok')}`",
        f"- mode: `{payload.get('mode')}`",
        f"- output_dir: `{payload.get('output_dir')}`",
        "",
        "## Steps",
        "",
    ]
    for step in payload.get("steps") or []:
        state = "skipped" if step.get("skipped") else step.get("ok")
        lines.append(f"- `{step.get('name')}`: `{state}`")
    lines.extend([
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in payload.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Boundaries",
        "",
    ])
    for item in payload.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload), encoding="utf-8")


def command_plan(args: argparse.Namespace, *, venv_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    crowdtensor = str(venv_script(venv_dir, "crowdtensor"))
    return [
        {
            "name": "create_venv",
            "command": [sys.executable, "-m", "venv", str(venv_dir)],
            "expect_json": False,
            "diagnosis": "venv_create_failed",
        },
        {
            "name": "install_package",
            "command": [str(venv_python(venv_dir)), "-m", "pip", "install", "-e", ".[dev,hf]"],
            "expect_json": False,
            "diagnosis": "install_failed",
        },
        {
            "name": "crowdtensor_help",
            "command": [crowdtensor, "--help"],
            "expect_json": False,
            "diagnosis": "console_script_failed",
        },
        {
            "name": "crowdtensord_help",
            "command": [str(venv_script(venv_dir, "crowdtensord")), "--help"],
            "expect_json": False,
            "diagnosis": "console_script_failed",
        },
        {
            "name": "crowdtensor_miner_help",
            "command": [str(venv_script(venv_dir, "crowdtensor-miner")), "--help"],
            "expect_json": False,
            "diagnosis": "console_script_failed",
        },
        {
            "name": "user_friendly_inference_frontdoor",
            "command": [
                str(venv_python(venv_dir)),
                str(ROOT / "scripts" / "user_friendly_inference_frontdoor_check.py"),
                "--output-dir",
                str(output_dir / "user-friendly-inference-frontdoor"),
                "--json",
            ],
            "expect_json": True,
            "diagnosis": "user_friendly_inference_frontdoor_failed",
        },
        {
            "name": "user_infer_smoke",
            "command": [
                crowdtensor,
                "infer",
                "--prompt-stdin",
                "--shareable-terminal",
                "--output-dir",
                str(output_dir / "user-infer-smoke"),
                "--max-new-tokens",
                "2",
                "--timeout-seconds",
                str(args.timeout_seconds),
            ],
            "expect_json": False,
            "stdin_text": USER_INFER_SMOKE_PROMPT,
            "validation": "user_infer_smoke",
            "diagnosis": "user_infer_smoke_failed",
        },
        {
            "name": "local_proof",
            "command": [
                crowdtensor,
                "local-proof",
                "--output-dir",
                str(output_dir / "local-proof"),
                "--base-port",
                str(args.base_port),
                "--request-count",
                str(args.request_count),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--json",
            ],
            "expect_json": True,
            "diagnosis": "local_proof_failed",
        },
        {
            "name": "home_infer",
            "command": [
                crowdtensor,
                "home-infer",
                "--output-dir",
                str(output_dir / "home-infer"),
                "--port",
                str(args.base_port + 10),
                "--request-count",
                str(args.request_count),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--json",
            ],
            "expect_json": True,
            "diagnosis": "home_infer_failed",
        },
        {
            "name": "llm_infer_mock",
            "command": [
                crowdtensor,
                "llm-infer",
                "--mock",
                "--output-dir",
                str(output_dir / "llm-infer"),
                "--port",
                str(args.base_port + 20),
                "--request-count",
                str(args.external_llm_request_count),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--json",
            ],
            "expect_json": True,
            "diagnosis": "llm_infer_failed",
        },
        {
            "name": "cpu_infer_beta",
            "command": [
                crowdtensor,
                "cpu-infer",
                "--mode",
                "local",
                "--output-dir",
                str(output_dir / "cpu-infer"),
                "--base-port",
                str(args.base_port + 25),
                "--request-count",
                str(args.request_count),
                "--external-llm-request-count",
                str(args.external_llm_request_count),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--json",
            ],
            "expect_json": True,
            "diagnosis": "cpu_infer_failed",
        },
        {
            "name": "release_ready_smoke",
            "command": [
                crowdtensor,
                "release-ready",
                "--allow-dirty",
                "--output-dir",
                str(output_dir / "release-ready"),
                "--base-port",
                str(args.base_port + 30),
                "--request-count",
                str(args.request_count),
                "--external-llm-request-count",
                str(args.external_llm_request_count),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--json",
            ],
            "expect_json": True,
            "diagnosis": "release_ready_failed",
        },
    ]


def build_onboarding_gate(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_onboarding_venv_"))
    steps: list[dict[str, Any]] = []
    diagnosis: set[str] = set()
    payload_summaries: dict[str, dict[str, Any]] = {}
    failed = False
    venv_removed = False

    try:
        for spec in command_plan(args, venv_dir=venv_dir, output_dir=output_dir):
            if failed:
                steps.append(skip_step(str(spec["name"]), "previous_step_failed"))
                continue
            step, payload = run_step(
                str(spec["name"]),
                list(spec["command"]),
                runner=runner,
                cwd=ROOT,
                timeout_seconds=args.timeout_seconds,
                output_dir=output_dir,
                venv_dir=venv_dir,
                expect_json=bool(spec["expect_json"]),
                require_payload_ok=bool(spec["expect_json"]),
                stdin_text=str(spec.get("stdin_text") or ""),
            )
            if spec.get("validation") == "user_infer_smoke":
                validation = validate_user_infer_smoke(output_dir / "user-infer-smoke")
                step["validation_schema"] = validation.get("schema")
                step["validation_ok"] = validation.get("ok")
                step["validation"] = validation
                validation_codes = diagnosis_from_payload(validation)
                if validation_codes:
                    step["diagnosis_codes"] = sorted(set(step.get("diagnosis_codes") or []) | set(validation_codes))
                payload = validation
                if validation.get("ok") is not True:
                    step["ok"] = False
                    step["error"] = "validation_not_ok"
            steps.append(step)
            if payload:
                payload_summaries[str(spec["name"])] = {
                    "schema": payload.get("schema"),
                    "ok": payload.get("ok"),
                    "diagnosis_codes": diagnosis_from_payload(payload),
                }
            if not step.get("ok"):
                diagnosis.add(str(spec["diagnosis"]))
                failed = True
    finally:
        if not args.keep_venv:
            shutil.rmtree(venv_dir, ignore_errors=True)
            venv_removed = True

    if not failed:
        diagnosis.add("onboarding_ready")

    artifacts = {
        "local_proof_summary": artifact_entry(
            output_dir / "local-proof" / "local_proof_summary.json",
            output_dir,
            kind="local_proof_summary",
            schema="local_proof_summary_v1",
        ),
        "home_inference_cli_summary": artifact_entry(
            output_dir / "home-infer" / "home_inference_cli_summary.json",
            output_dir,
            kind="home_inference_cli_summary",
            schema="home_inference_cli_v1",
        ),
        "llm_inference_cli_summary": artifact_entry(
            output_dir / "llm-infer" / "llm_inference_cli_summary.json",
            output_dir,
            kind="llm_inference_cli_summary",
            schema="llm_inference_cli_v1",
        ),
        "cpu_inference_beta_json": artifact_entry(
            output_dir / "cpu-infer" / "cpu_inference_beta.json",
            output_dir,
            kind="cpu_inference_beta",
            schema="cpu_inference_beta_v1",
        ),
        "user_friendly_inference_frontdoor_check": artifact_entry(
            output_dir / "user-friendly-inference-frontdoor" / "user_friendly_inference_frontdoor_check.json",
            output_dir,
            kind="user_friendly_inference_frontdoor_check",
            schema="user_friendly_inference_frontdoor_check_v1",
        ),
        "user_infer_smoke_summary": artifact_entry(
            output_dir / "user-infer-smoke" / "infer_summary.json",
            output_dir,
            kind="crowdtensor_infer_summary",
            schema="crowdtensor_infer_cli_v1",
        ),
        "user_infer_smoke_markdown": artifact_entry(
            output_dir / "user-infer-smoke" / "infer_summary.md",
            output_dir,
            kind="crowdtensor_infer_summary_markdown",
        ),
        "release_readiness_json": artifact_entry(
            output_dir / "release-ready" / "release_readiness.json",
            output_dir,
            kind="release_readiness",
            schema="release_readiness_v1",
        ),
    }

    json_out = Path(args.json_out).resolve() if args.json_out else output_dir / "onboarding_gate.json"
    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else None
    summary = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": not failed and all(bool(step.get("ok")) for step in steps if not step.get("skipped")),
        "mode": "quick" if args.quick else "full",
        "root": str(ROOT),
        "output_dir": str(output_dir),
        "json_out": str(json_out),
        "markdown_out": str(markdown_out) if markdown_out else "",
        "python": {
            "host_executable": sys.executable,
            "host_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "venv": {
            "path": str(venv_dir),
            "clean_environment": True,
            "uses_system_site_packages": False,
            "kept": bool(args.keep_venv),
            "removed": venv_removed,
        },
        "install": {
            "editable": True,
            "extras": ["dev", "hf"],
            "command": "python -m pip install -e .[dev,hf]",
            "pep668_safe": True,
        },
        "request_count": args.request_count,
        "external_llm_request_count": args.external_llm_request_count,
        "base_port": args.base_port,
        "steps": steps,
        "payload_summaries": payload_summaries,
        "artifacts": artifacts,
        "diagnosis_codes": sorted(diagnosis),
        "safety": {
            "captured_failure_tails_redacted": True,
            "summary_excludes_raw_inference_payloads": True,
            "summary_excludes_raw_external_llm_outputs": True,
            "user_infer_smoke_uses_shareable_terminal": True,
            "fresh_kaggle_gpu_attempted": False,
            "fresh_kaggle_gpu_verified": False,
            "writes_reports_under_tmp_by_default": str(output_dir).startswith("/tmp"),
            "not_production": True,
        },
        "limitations": [
            "Alpha onboarding gate for CPU-only local proof paths; not production Swarm Inference",
            "Does not prove arbitrary prompt serving, real LLM serving, GPU pooling, WebGPU shards, P2P routing, or incentives",
            "The external LLM path uses deterministic --mock unless an operator explicitly configures a local runtime elsewhere",
        ],
        "recommended_next_commands": [
            "crowdtensor local-proof --json",
            "printf %s '<prompt>' | crowdtensor infer --prompt-stdin --shareable-terminal --max-new-tokens 2",
            "crowdtensor cpu-infer --mode local --json",
            "crowdtensor home-infer --scenario-id route-baseline --json",
            "crowdtensor llm-infer --mock --json",
            "python scripts/user_friendly_inference_frontdoor_check.py --json",
            "crowdtensor release-ready --json",
        ],
    }

    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary["diagnosis_codes"] = sorted(set(summary["diagnosis_codes"]) | {"sensitive_output_detected"})
        summary["safety_error"] = "onboarding gate summary contained secret-like fragments"
        summary["secret_like_fragments"] = ["<redacted>" for _ in leaks]

    write_json(summary, json_out)
    summary["artifacts"]["onboarding_gate_json"] = artifact_entry(
        json_out,
        output_dir,
        kind="onboarding_gate",
        schema=SCHEMA,
        ok=summary.get("ok"),
    )
    if markdown_out:
        write_markdown(summary, markdown_out)
        summary["artifacts"]["onboarding_gate_markdown"] = artifact_entry(
            markdown_out,
            output_dir,
            kind="onboarding_gate_markdown",
        )
    write_json(summary, json_out)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the documented fresh-clone CrowdTensor onboarding path.")
    parser.add_argument("--output-dir", default="/tmp/crowdtensor_onboarding_gate")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--base-port", type=int, default=8940)
    parser.add_argument("--request-count", type=int, default=None)
    parser.add_argument("--external-llm-request-count", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--quick", action="store_true", help="use one request per runtime proof for CI/onboarding smoke")
    parser.add_argument("--keep-venv", action="store_true", help="keep the temporary virtualenv for debugging")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    args.request_count = args.request_count if args.request_count is not None else (1 if args.quick else 4)
    args.external_llm_request_count = (
        args.external_llm_request_count if args.external_llm_request_count is not None else (1 if args.quick else 3)
    )
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.external_llm_request_count < 1:
        raise SystemExit("--external-llm-request-count must be at least 1")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be at least 1")
    return args


def print_human(summary: dict[str, Any]) -> None:
    print("CrowdTensor onboarding gate")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  mode: {summary.get('mode')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    for step in summary.get("steps") or []:
        state = "skipped" if step.get("skipped") else step.get("ok")
        print(f"  step {step.get('name')}: {state}")
    print(f"  json: {summary.get('json_out')}")


def main() -> None:
    args = parse_args()
    summary = build_onboarding_gate(args)
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print_human(summary)
    raise SystemExit(0 if summary.get("ok") else 1)


if __name__ == "__main__":
    main()
