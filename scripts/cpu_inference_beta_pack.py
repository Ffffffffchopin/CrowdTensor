#!/usr/bin/env python3
"""Build the CPU-only inference Beta aggregate report."""

from __future__ import annotations

import argparse
import json
import os
import platform
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


SCHEMA = "cpu_inference_beta_v1"
Runner = Callable[..., subprocess.CompletedProcess[str]]
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
MODEL_BUNDLE_WORKLOAD = "model-bundle"
EXTERNAL_LLM_WORKLOAD = "external-llm"


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
    for fragment in [
        "lease_token",
        "idempotency_key",
        "inference_results",
        "external_llm_results",
        "output_text",
        "Bearer ",
    ]:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    return value


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: int,
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

    elapsed = round(time.monotonic() - started, 3)
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": elapsed,
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
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:], secret_values)
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:], secret_values)
    return step, payload


def diagnosis_codes(*payloads: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
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
    for key in [
        "mode",
        "workload",
        "evidence_schema",
        "observability_schema",
        "acceptance_schema",
        "scenario_id",
    ]:
        if payload.get(key) is not None:
            summary[key] = payload.get(key)
    if isinstance(payload.get("route"), dict):
        route = payload["route"]
        summary["route"] = {
            "name": route.get("name"),
            "target": route.get("target"),
            "workload": route.get("workload"),
            "confidence": route.get("confidence"),
            "usable_now": route.get("usable_now"),
        }
    if isinstance(payload.get("inference"), dict):
        inference = payload["inference"]
        summary["inference"] = {
            "request_count": inference.get("request_count"),
            "completion_count": inference.get("completion_count"),
            "request_trace_count": inference.get("request_trace_count"),
            "requests_per_second": inference.get("requests_per_second"),
            "read_only": inference.get("read_only"),
            "redaction_ok": inference.get("redaction_ok"),
        }
    acceptance = payload.get("acceptance_summary")
    if isinstance(acceptance, dict):
        summary["acceptance"] = {
            "schema": acceptance.get("schema"),
            "task_id": acceptance.get("task_id"),
            "evidence_schema": acceptance.get("evidence_schema"),
            "observability_schema": acceptance.get("observability_schema"),
        }
    return summary


def workload_list(value: str) -> list[str]:
    if value == "all":
        return [MODEL_BUNDLE_WORKLOAD, EXTERNAL_LLM_WORKLOAD]
    return [value]


def external_llm_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.mock:
        flags.append("--mock")
    if args.llm_runtime_cmd:
        flags.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    if args.llm_runtime_url:
        flags.extend(["--llm-runtime-url", args.llm_runtime_url])
    if args.llm_runtime_api_key:
        flags.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    if args.llm_runtime_model_id:
        flags.extend(["--llm-runtime-model-id", args.llm_runtime_model_id])
    if args.llm_runtime_timeout:
        flags.extend(["--llm-runtime-timeout", str(args.llm_runtime_timeout)])
    return flags


def build_local(args: argparse.Namespace, *, runner: Runner) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    home_dir = output_dir / "local-home-infer"
    llm_dir = output_dir / "local-llm-infer"
    steps: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}

    home_cmd = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "home-infer",
        "--output-dir",
        str(home_dir),
        "--port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    home_step, home_payload = run_json_step(
        "home_infer",
        home_cmd,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    home_step["ok"] = bool(home_step.get("ok") and home_payload.get("ok"))
    steps.append(home_step)
    payloads["home_infer"] = payload_summary(home_payload)

    llm_cmd = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "llm-infer",
        "--mock",
        "--output-dir",
        str(llm_dir),
        "--port",
        str(args.base_port + 10),
        "--request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    llm_step, llm_payload = run_json_step(
        "llm_infer_mock",
        llm_cmd,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    llm_step["ok"] = bool(llm_step.get("ok") and llm_payload.get("ok"))
    steps.append(llm_step)
    payloads["llm_infer_mock"] = payload_summary(llm_payload)

    return {
        "mode": "local",
        "steps": steps,
        "payload_summaries": payloads,
        "artifacts": {
            "home_inference_cli_summary": artifact_entry(
                home_dir / "home_inference_cli_summary.json",
                output_dir,
                kind="home_inference_cli_summary",
                schema="home_inference_cli_v1",
                ok=home_payload.get("ok") if home_payload else None,
            ),
            "home_compute_evidence_json": artifact_entry(
                home_dir / "home_compute_evidence.json",
                output_dir,
                kind="home_compute_evidence",
                schema="home_compute_evidence_v1",
            ),
            "llm_inference_cli_summary": artifact_entry(
                llm_dir / "llm_inference_cli_summary.json",
                output_dir,
                kind="llm_inference_cli_summary",
                schema="llm_inference_cli_v1",
                ok=llm_payload.get("ok") if llm_payload else None,
            ),
            "external_llm_evidence_json": artifact_entry(
                llm_dir / "external_llm_evidence.json",
                output_dir,
                kind="external_llm_evidence",
                schema="external_llm_evidence_v1",
            ),
        },
        "diagnosis_codes": diagnosis_codes(home_payload, llm_payload),
    }


def build_remote_loopback(args: argparse.Namespace, *, runner: Runner) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for index, workload in enumerate(workload_list(args.workload)):
        request_count = args.external_llm_request_count if workload == EXTERNAL_LLM_WORKLOAD else args.request_count
        command = [
            sys.executable,
            str(ROOT / "scripts" / "remote_home_compute_demo_check.py"),
            "--port",
            str(args.base_port + index),
            "--workload",
            workload,
            "--request-count",
            str(request_count),
            "--scenario-id",
            args.scenario_id,
            "--verify-timeout",
            str(args.remote_timeout_seconds),
            "--command-timeout",
            str(args.timeout_seconds),
        ]
        step, payload = run_json_step(
            f"remote_loopback_{workload.replace('-', '_')}",
            command,
            runner=runner,
            timeout_seconds=args.timeout_seconds,
        )
        step["ok"] = bool(step.get("ok") and payload.get("ok"))
        steps.append(step)
        payloads[workload] = payload_summary(payload)
    return {
        "mode": "remote-loopback",
        "workload": args.workload,
        "steps": steps,
        "payload_summaries": payloads,
        "artifacts": {},
        "diagnosis_codes": diagnosis_codes(*[payload for payload in payloads.values() if isinstance(payload, dict)]),
    }


def remote_demo_base(args: argparse.Namespace, action: str, workload: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "remote-demo",
        action,
        "--workload",
        workload,
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.external_llm_request_count if workload == EXTERNAL_LLM_WORKLOAD else args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    return command


def add_existing_auth_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    command.extend([
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
    ])
    return command


def build_remote_existing(args: argparse.Namespace, *, runner: Runner) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    steps: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    secret_values = [
        args.observer_token,
        args.admin_token,
        args.llm_runtime_url,
        args.llm_runtime_api_key,
    ]
    for workload in workload_list(args.workload):
        workload_dir = output_dir / f"remote-existing-{workload}"
        doctor_cmd = add_existing_auth_flags(remote_demo_base(args, "doctor", workload, workload_dir), args)
        doctor_cmd.append("--json")
        doctor_step, doctor_payload = run_json_step(
            f"remote_existing_{workload.replace('-', '_')}_doctor",
            doctor_cmd,
            runner=runner,
            timeout_seconds=args.timeout_seconds,
            secret_values=secret_values,
        )
        doctor_step["ok"] = bool(doctor_step.get("ok") and doctor_payload.get("ok"))
        steps.append(doctor_step)
        payloads[f"{workload}_doctor"] = payload_summary(doctor_payload)

        verify_cmd = add_existing_auth_flags(remote_demo_base(args, "verify", workload, workload_dir), args)
        verify_cmd.extend([
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--poll-interval",
            str(args.poll_interval),
            "--json",
        ])
        if workload == EXTERNAL_LLM_WORKLOAD:
            verify_cmd.extend(external_llm_flags(args))
        verify_step, verify_payload = run_json_step(
            f"remote_existing_{workload.replace('-', '_')}_verify",
            verify_cmd,
            runner=runner,
            timeout_seconds=args.timeout_seconds,
            secret_values=secret_values,
        )
        verify_step["ok"] = bool(verify_step.get("ok") and verify_payload.get("ok"))
        steps.append(verify_step)
        payloads[f"{workload}_verify"] = payload_summary(verify_payload)

        collect_cmd = add_existing_auth_flags(remote_demo_base(args, "collect", workload, workload_dir), args)
        acceptance = verify_payload.get("acceptance_summary") if isinstance(verify_payload.get("acceptance_summary"), dict) else {}
        task_id = str(acceptance.get("task_id") or "")
        if task_id:
            collect_cmd.extend(["--task-id", task_id])
        collect_cmd.append("--json")
        if workload == EXTERNAL_LLM_WORKLOAD:
            collect_cmd.extend(external_llm_flags(args))
        collect_step, collect_payload = run_json_step(
            f"remote_existing_{workload.replace('-', '_')}_collect",
            collect_cmd,
            runner=runner,
            timeout_seconds=args.timeout_seconds,
            secret_values=secret_values,
        )
        collect_step["ok"] = bool(collect_step.get("ok") and collect_payload.get("ok"))
        steps.append(collect_step)
        payloads[f"{workload}_collect"] = payload_summary(collect_payload)
        artifacts[f"{workload}_remote_home_compute_demo_json"] = artifact_entry(
            workload_dir / "remote_home_compute_demo.json",
            output_dir,
            kind="remote_home_compute_demo",
            schema="remote_home_compute_demo_v1",
        )
    return {
        "mode": "remote-existing",
        "workload": args.workload,
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "miner_id": args.miner_id,
        "steps": steps,
        "payload_summaries": payloads,
        "artifacts": artifacts,
        "diagnosis_codes": diagnosis_codes(*[payload for payload in payloads.values() if isinstance(payload, dict)]),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor CPU Inference Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Steps",
        "",
    ]
    for step in report.get("steps") or []:
        lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
    lines.extend([
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
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
        body = build_remote_existing(args, runner=runner)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")

    steps = body.get("steps") or []
    ok = all(bool(step.get("ok")) for step in steps)
    codes = sorted(set(body.get("diagnosis_codes") or []))
    codes.append("cpu_inference_beta_ready" if ok else "cpu_inference_beta_failed")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": body.get("mode"),
        "output_dir": str(output_dir),
        "request_count": args.request_count,
        "external_llm_request_count": args.external_llm_request_count,
        "scenario_id": args.scenario_id,
        "workload": body.get("workload", args.workload),
        "coordinator_url": body.get("coordinator_url"),
        "miner_id": body.get("miner_id"),
        "steps": steps,
        "payload_summaries": body.get("payload_summaries") or {},
        "artifacts": body.get("artifacts") or {},
        "diagnosis_codes": sorted(set(codes)),
        "safety": {
            "cpu_only_default": True,
            "read_only_workloads": ["model_bundle_infer", "external_llm_infer"],
            "captured_output_redacted": True,
            "summary_excludes_raw_inference_payloads": True,
            "summary_excludes_raw_external_llm_outputs": True,
            "not_production": True,
        },
        "limitations": [
            "CPU-only Beta inference proof; not production Swarm Inference",
            "Fixed scenarios and fixed prompt runtime evidence; not arbitrary public prompt serving",
            "Does not provide GPU pooling, WebGPU model shards, P2P routing, NAT traversal, payments, or incentives",
        ],
        "recommended_next_commands": [
            "crowdtensor cpu-infer --mode local --json",
            "crowdtensor remote-demo prepare --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --json",
            "crowdtensor remote-demo doctor --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --json",
        ],
    }
    report = support_bundle.sanitize(redact_values(report, [
        getattr(args, "observer_token", ""),
        getattr(args, "admin_token", ""),
        getattr(args, "llm_runtime_url", ""),
        getattr(args, "llm_runtime_api_key", ""),
    ]))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [
        fragment for fragment in [
            "lease_token",
            "idempotency_key",
            "inference_results",
            "external_llm_results",
            "output_text",
            "Bearer ",
        ]
        if fragment in encoded
    ]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"sensitive_output_detected"})
        report["safety_error"] = "cpu inference beta report contained secret-like fragments"

    json_out = Path(args.json_out).resolve() if args.json_out else output_dir / "cpu_inference_beta.json"
    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else output_dir / "cpu_inference_beta.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["cpu_inference_beta_json"] = artifact_entry(
        json_out,
        output_dir,
        kind="cpu_inference_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["cpu_inference_beta_markdown"] = artifact_entry(
        markdown_out,
        output_dir,
        kind="cpu_inference_beta_markdown",
    )
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CPU-only CrowdTensor inference Beta report.")
    parser.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing"], default="local")
    parser.add_argument("--output-dir", default="dist/cpu-infer")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--base-port", type=int, default=8970)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--external-llm-request-count", type=int, default=3)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--workload", choices=["model-bundle", "external-llm", "all"], default="all")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id", default="remote-linux-1")
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--remote-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--mock", action="store_true", help="use mock external LLM runtime for remote external-llm mode")
    parser.add_argument("--llm-runtime-cmd", default="")
    parser.add_argument("--llm-runtime-url", default="")
    parser.add_argument("--llm-runtime-api-key", default="")
    parser.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    parser.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
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
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")
    if args.llm_runtime_cmd and args.llm_runtime_url:
        raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
    if args.llm_runtime_timeout <= 0:
        raise SystemExit("--llm-runtime-timeout must be positive")
    if args.mode == "remote-existing":
        missing = [
            name for name in ["coordinator_url", "observer_token", "admin_token"]
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    return args


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor CPU inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
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
