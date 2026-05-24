#!/usr/bin/env python3
"""High-level two-machine home-compute remote Miner demo wrapper."""

from __future__ import annotations

import argparse
import json
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


SCHEMA = "remote_home_compute_demo_v1"
WORKLOAD_TYPE = "model_bundle_infer"
ROUTE_NAME = "remote_python_model_bundle_infer"
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")


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
    if payload:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:], secret_values)
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def summarize_runbook(payload: dict[str, Any]) -> dict[str, Any]:
    demo = payload.get("demo") if isinstance(payload.get("demo"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "coordinator_url": demo.get("coordinator_url"),
        "workload_type": demo.get("workload_type"),
        "route": demo.get("route"),
        "request_count": demo.get("request_count"),
        "scenario_id": demo.get("scenario_id"),
        "scenario_schema": demo.get("scenario_schema"),
        "registry_hashed": safety.get("registry_hashed"),
        "public_artifact_redacted": safety.get("public_artifact_redacted"),
    }


def summarize_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.get("session_request") if isinstance(payload.get("session_request"), dict) else {}
    evidence = payload.get("evidence_summary") if isinstance(payload.get("evidence_summary"), dict) else {}
    observability = payload.get("observability_summary") if isinstance(payload.get("observability_summary"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "diagnosis_codes": payload.get("diagnosis_codes") or [],
        "session_created": session.get("created"),
        "task_id": session.get("task_id"),
        "scenario_id": (payload.get("scenario") or {}).get("scenario_id"),
        "request_count": session.get("request_count"),
        "evidence_schema": evidence.get("schema"),
        "evidence_ok": evidence.get("ok"),
        "observability_schema": observability.get("schema"),
        "accepted_results": (observability.get("work_queue") or {}).get("accepted_results"),
        "requests_per_second": (observability.get("inference") or {}).get("requests_per_second"),
    }


def diagnosis_codes(mode: str, *, step_ok: bool, payload: dict[str, Any]) -> list[str]:
    if mode == "prepare":
        return ["remote_home_compute_prepare_ready"] if step_ok and payload.get("ok") else ["remote_home_compute_prepare_failed"]
    codes = [str(code) for code in payload.get("diagnosis_codes") or [] if isinstance(code, str)]
    if payload.get("ok") is True and "remote_home_compute_ready" not in codes:
        codes.append("remote_home_compute_ready")
    if not codes:
        codes.append("remote_home_compute_failed")
    return sorted(set(codes))


def render_markdown(payload: dict[str, Any]) -> str:
    demo = payload.get("demo") or {}
    runbook = payload.get("runbook_summary") or {}
    acceptance = payload.get("acceptance_summary") or {}
    lines = [
        "# CrowdTensor Remote Home-Compute Demo",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Mode: `{payload.get('mode')}`",
        f"Route: `{demo.get('route')}`",
        f"Workload: `{demo.get('workload_type')}`",
        f"Scenario: `{demo.get('scenario_id')}`",
        "",
        "## Summary",
        "",
        f"- Runbook OK: `{runbook.get('ok')}`",
        f"- Acceptance OK: `{acceptance.get('ok')}`",
        f"- Session created: `{acceptance.get('session_created')}`",
        f"- Task ID: `{acceptance.get('task_id')}`",
        f"- Evidence OK: `{acceptance.get('evidence_ok')}`",
        f"- Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in sorted((payload.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def build_prepare(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runbook_json = output_dir / "remote_demo_runbook.json"
    runbook_md = output_dir / "remote_demo_runbook.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_demo_runbook_pack.py"),
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--json-out",
        str(runbook_json),
        "--markdown-out",
        str(runbook_md),
    ]
    if args.replace:
        command.append("--replace")
    step, runbook = run_json_step("remote_demo_prepare", command, runner=runner, timeout_seconds=args.timeout_seconds)
    step["ok"] = bool(step.get("ok") and runbook.get("ok"))
    return build_report(
        args=args,
        mode="prepare",
        step=step,
        runbook=runbook,
        acceptance={},
        output_dir=output_dir,
        write_outputs=True,
    )


def build_verify(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    acceptance_json = output_dir / "remote_demo_acceptance.json"
    acceptance_md = output_dir / "remote_demo_acceptance.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_demo_acceptance_pack.py"),
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--output-dir",
        str(output_dir),
        "--json-out",
        str(acceptance_json),
        "--markdown-out",
        str(acceptance_md),
    ]
    if args.create_session:
        command.append("--create-session")
    step, acceptance = run_json_step(
        "remote_demo_verify",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=[args.observer_token, args.admin_token],
    )
    step["ok"] = bool(step.get("ok") and acceptance.get("ok"))
    runbook = load_json(output_dir / "remote_demo_runbook.json")
    return build_report(
        args=args,
        mode="verify",
        step=step,
        runbook=runbook,
        acceptance=acceptance,
        output_dir=output_dir,
        secret_values=[args.observer_token, args.admin_token],
        write_outputs=True,
    )


def build_report(
    *,
    args: argparse.Namespace,
    mode: str,
    step: dict[str, Any],
    runbook: dict[str, Any],
    acceptance: dict[str, Any],
    output_dir: Path,
    write_outputs: bool,
    secret_values: list[str] | None = None,
) -> dict[str, Any]:
    ok = bool(step.get("ok") and (runbook.get("ok") if mode == "prepare" else acceptance.get("ok")))
    summary_json = output_dir / "remote_home_compute_demo.json"
    summary_md = output_dir / "remote_home_compute_demo.md"
    artifacts = {
        "remote_home_compute_demo_json": artifact_entry(summary_json, output_dir, kind="remote_home_compute_demo", schema=SCHEMA, ok=ok),
        "remote_home_compute_demo_markdown": artifact_entry(summary_md, output_dir, kind="remote_home_compute_demo_markdown"),
        "remote_demo_runbook_json": artifact_entry(output_dir / "remote_demo_runbook.json", output_dir, kind="remote_demo_runbook", schema="remote_demo_runbook_v1"),
        "remote_demo_runbook_markdown": artifact_entry(output_dir / "remote_demo_runbook.md", output_dir, kind="remote_demo_runbook_markdown"),
        "operator_private_env": artifact_entry(output_dir / "operator.private.env", output_dir, kind="private_env"),
        "miner_private_env": artifact_entry(output_dir / "miner.private.env", output_dir, kind="private_env"),
        "remote_demo_acceptance_json": artifact_entry(output_dir / "remote_demo_acceptance.json", output_dir, kind="remote_demo_acceptance", schema="remote_demo_acceptance_v1"),
        "remote_demo_acceptance_markdown": artifact_entry(output_dir / "remote_demo_acceptance.md", output_dir, kind="remote_demo_acceptance_markdown"),
        "remote_compute_evidence_json": artifact_entry(output_dir / "remote_compute_evidence.json", output_dir, kind="remote_compute_evidence", schema="remote_compute_evidence_v1"),
        "support_bundle_json": artifact_entry(output_dir / "support_bundle.json", output_dir, kind="support_bundle", schema="support_bundle_v1"),
    }
    runbook_summary = summarize_runbook(runbook)
    acceptance_summary = summarize_acceptance(acceptance)
    scenario = acceptance.get("scenario") or (runbook.get("demo") or {})
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": mode,
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "miner_id": args.miner_id,
        "demo": {
            "workload_type": WORKLOAD_TYPE,
            "route": ROUTE_NAME,
            "request_count": args.request_count,
            "scenario_id": scenario.get("scenario_id") or args.scenario_id,
            "scenario_schema": scenario.get("scenario_schema") or "model_bundle_inference_scenario_v1",
        },
        "step": step,
        "runbook_summary": runbook_summary,
        "acceptance_summary": acceptance_summary,
        "diagnosis_codes": diagnosis_codes(mode, step_ok=bool(step.get("ok")), payload=acceptance if mode == "verify" else runbook),
        "artifacts": artifacts,
        "safety": {
            "public_artifact_redacted": True,
            "private_env_files": ["operator.private.env", "miner.private.env"],
            "summary_excludes_plaintext_tokens": True,
            "summary_excludes_raw_inference_payloads": True,
            "raw_state_dump_in_report": False,
            "read_only_workload": WORKLOAD_TYPE,
            "requires_tls_or_vpn": True,
            "not_production": True,
        },
        "limitations": [
            "Controlled two-machine CPU demo; not production Swarm Inference",
            "Requires operator-provided TLS, VPN, tunnel, or reachable private network for real two-machine use",
            "Does not implement P2P/NAT traversal, GPU pooling, WebGPU model shards, arbitrary prompt serving, or incentives",
        ],
        "recommended_next_commands": [
            "crowdtensor remote-demo prepare --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --json",
            "source dist/remote-home-compute/operator.private.env",
            "crowdtensor remote-demo verify --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --json",
        ],
    }
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    allowed_fragments = {
        "CROWDTENSOR_OBSERVER_TOKEN",
        "CROWDTENSOR_ADMIN_TOKEN",
        "CROWDTENSOR_MINER_TOKEN",
    }
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded and fragment not in allowed_fragments]
    leaks.extend(secret for secret in secret_values or [] if secret and secret in encoded)
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"sensitive_output_detected"})
        report["safety_error"] = "remote home-compute demo report contained secret-like fragments"
    if write_outputs:
        write_json(report, str(summary_json))
        write_markdown(report, str(summary_md))
        report["artifacts"]["remote_home_compute_demo_json"] = artifact_entry(summary_json, output_dir, kind="remote_home_compute_demo", schema=SCHEMA, ok=report.get("ok"))
        report["artifacts"]["remote_home_compute_demo_markdown"] = artifact_entry(summary_md, output_dir, kind="remote_home_compute_demo_markdown")
        write_json(report, str(summary_json))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or verify the CrowdTensor remote home-compute demo.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common: dict[str, Any] = {"coordinator_url": "http://127.0.0.1:8787", "miner_id": "remote-linux-1"}
    prepare = subparsers.add_parser("prepare", help="Create the remote home-compute runbook and private env files.")
    prepare.add_argument("--coordinator-url", default=common["coordinator_url"])
    prepare.add_argument("--miner-id", default=common["miner_id"])
    prepare.add_argument("--output-dir", default="dist/remote-home-compute")
    prepare.add_argument("--request-count", type=int, default=4)
    prepare.add_argument("--scenario-id", default="route-baseline")
    prepare.add_argument("--timeout-seconds", type=float, default=180.0)
    prepare.add_argument("--replace", action="store_true")
    prepare.add_argument("--json", action="store_true")
    verify = subparsers.add_parser("verify", help="Create and verify a read-only remote model_bundle_infer session.")
    verify.add_argument("--coordinator-url", required=True)
    verify.add_argument("--miner-id", required=True)
    verify.add_argument("--observer-token", required=True)
    verify.add_argument("--admin-token", required=True)
    verify.add_argument("--output-dir", default="dist/remote-home-compute")
    verify.add_argument("--request-count", type=int, default=4)
    verify.add_argument("--scenario-id", default="route-baseline")
    verify.add_argument("--timeout-seconds", type=float, default=180.0)
    verify.add_argument("--remote-timeout-seconds", type=float, default=120.0)
    verify.add_argument("--poll-interval", type=float, default=2.0)
    verify.add_argument("--create-session", dest="create_session", action="store_true", default=True)
    verify.add_argument("--no-create-session", dest="create_session", action="store_false")
    verify.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if getattr(args, "remote_timeout_seconds", 0) < 0:
        raise SystemExit("--remote-timeout-seconds must be non-negative")
    if getattr(args, "poll_interval", 1) <= 0:
        raise SystemExit("--poll-interval must be positive")
    args.coordinator_url = args.coordinator_url.rstrip("/")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        payload = build_prepare(args)
    elif args.mode == "verify":
        payload = build_verify(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"CrowdTensor remote home-compute demo {args.mode}")
        print(f"  ok: {payload.get('ok')}")
        print(f"  schema: {payload.get('schema')}")
        print(f"  output: {payload.get('output_dir')}")
        print(f"  diagnosis: {', '.join(payload.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if payload.get("ok") else 1)


if __name__ == "__main__":
    main()
