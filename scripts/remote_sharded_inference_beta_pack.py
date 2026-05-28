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


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Remote Sharded Inference Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- failure_mode: `{report.get('failure_mode')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Steps",
        "",
    ]
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
        codes.add("remote_sharded_inference_ready")
    else:
        codes.add("remote_sharded_inference_failed")
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
        "limitations": [
            "CPU-only two-stage pipeline-sharded inference Beta; not production Swarm Inference",
            "Uses fixed model-bundle requests and activation hashes; not real LLM sharding",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, payments, or arbitrary prompt serving",
        ],
        "recommended_next_commands": [
            "crowdtensor shard-infer-beta --mode remote-loopback --json",
            "crowdtensor shard-infer-beta --mode remote-loopback --failure-mode kill-stage-after-claim --json",
        ],
    }
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"sensitive_output_detected"})
        report["safety_error"] = "remote sharded inference report contained secret-like fragments"

    json_out = Path(args.json_out).resolve() if args.json_out else output_dir / "remote_sharded_inference_beta.json"
    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else output_dir / "remote_sharded_inference_beta.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    report["artifacts"]["remote_sharded_inference_beta_json"] = artifact_entry(
        json_out,
        output_dir,
        kind="remote_sharded_inference_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["remote_sharded_inference_beta_markdown"] = artifact_entry(
        markdown_out,
        output_dir,
        kind="remote_sharded_inference_beta_markdown",
    )
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["remote_sharded_inference_beta_json"]["present"] = True
    report["artifacts"]["remote_sharded_inference_beta_markdown"]["present"] = True
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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
    print("CrowdTensor remote sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
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
