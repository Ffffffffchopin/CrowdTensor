#!/usr/bin/env python3
"""Build the CPU Inference Beta release-candidate evidence pack."""

from __future__ import annotations

import argparse
import json
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
        "recommended_next_commands": [
            "crowdtensor cpu-infer --mode beta-rc --kaggle-real-runtime-report dist/kaggle-real-runtime/kaggle_real_runtime_acceptance.json --json",
            "crowdtensor remote-demo prepare --target kaggle --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id kaggle-cpu-1 --json",
            "python3 scripts/cpu_inference_beta_rc_check.py --quick",
        ],
    }

    json_out = Path(args.json_out).resolve() if args.json_out else output_dir / "cpu_inference_beta_rc.json"
    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else output_dir / "cpu_inference_beta_rc.md"
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
    report = support_bundle.sanitize(report)
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"sensitive_output_detected"})
        report["safety_error"] = "CPU Inference Beta RC report contained secret-like fragments"
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["cpu_inference_beta_rc_json"]["present"] = json_out.is_file()
    report["artifacts"]["cpu_inference_beta_rc_markdown"]["present"] = markdown_out.is_file()
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor CPU Inference Beta RC",
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
    lines.extend(["", "## Diagnosis", ""])
    lines.append(", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`")
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
    print("CrowdTensor CPU Inference Beta RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
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
