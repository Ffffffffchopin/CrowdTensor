#!/usr/bin/env python3
"""Run the CrowdTensorD runtime acceptance smoke pack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOKEN_ENV_SCRIPTS = {
    "browser_miner_chaos.py",
    "browser_miner_smoke.py",
    "browser_probe_smoke.py",
    "capability_ledger_check.py",
    "chaos_runner.py",
    "external_llm_http_adapter_smoke.py",
    "external_llm_evidence_check.py",
    "external_llm_inference_smoke.py",
    "home_compute_demo_check.py",
    "home_compute_evidence_check.py",
    "micro_transformer_smoke.py",
    "inference_session_demo.py",
    "model_bundle_inference_smoke.py",
    "model_bundle_smoke.py",
    "multi_miner_scenario_sweep_check.py",
    "operator_control_check.py",
    "remote_compute_evidence_check.py",
    "remote_miner_readiness_check.py",
    "replay_audit_check.py",
    "runtime_contract_check.py",
    "trust_quarantine_check.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tail_text(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def safe_summary_json(payload: dict[str, Any]) -> dict[str, Any]:
    safe_keys = [
        "ok",
        "schema",
        "demo",
        "route",
        "route_confidence",
        "workload",
        "workload_type",
        "adapter_kind",
        "request_count",
        "completion_count",
        "output_chars",
        "request_trace_count",
        "requests_per_second",
        "execution_mode",
        "failure_mode",
        "scenario_ids",
        "distinct_miner_count",
        "diagnosis",
        "diagnosis_codes",
    ]
    summary = {key: payload.get(key) for key in safe_keys if key in payload}
    for nested_key in ["selected_workload", "route_decision", "inference_summary", "wait_summary"]:
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            summary[nested_key] = {
                key: nested.get(key)
                for key in [
                    "name",
                    "status",
                    "target",
                    "workload",
                    "confidence",
                    "usable_now",
                    "request_count",
                    "request_trace_count",
                    "ok",
                    "attempts",
                    "elapsed_seconds",
                ]
                if key in nested
            }
    diagnosis = payload.get("diagnosis")
    if isinstance(diagnosis, dict):
        summary["diagnosis"] = {
            key: diagnosis.get(key)
            for key in ["primary_code", "severity", "summary"]
            if key in diagnosis
        }
    return summary


def parse_summary_json(stdout: str) -> dict[str, Any] | None:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return safe_summary_json(payload)


def diagnosis_codes_from_summary(summary: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    raw_codes = summary.get("diagnosis_codes")
    if isinstance(raw_codes, list):
        codes.extend(str(code) for code in raw_codes if code)
    diagnosis = summary.get("diagnosis")
    if isinstance(diagnosis, dict) and diagnosis.get("primary_code"):
        codes.append(str(diagnosis["primary_code"]))
    elif isinstance(diagnosis, str):
        codes.append(diagnosis)
    return sorted(set(codes))


def diagnosis_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    by_check: dict[str, list[str]] = {}
    failed_checks: list[str] = []
    all_codes: list[str] = []
    for check in checks:
        name = str(check.get("name") or "")
        if check.get("ok") is not True:
            failed_checks.append(name)
        codes = [str(code) for code in check.get("diagnosis_codes") or [] if code]
        if codes:
            by_check[name] = codes
            all_codes.extend(codes)
    return {
        "codes": sorted(set(all_codes)),
        "by_check": by_check,
        "failed_checks": failed_checks,
    }


def check_command(
    script_name: str,
    *,
    host: str,
    port: int,
    state_dir: Path,
    admin_token: str,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / script_name),
        "--host",
        host,
        "--port",
        str(port),
        "--state-dir",
        str(state_dir),
    ]
    if script_name in {
        "operator_control_check.py",
        "home_compute_demo_check.py",
        "home_compute_evidence_check.py",
        "admin_inference_session_check.py",
    }:
        command.extend(["--admin-token", admin_token])
    if script_name == "inference_session_demo.py":
        command.append("--json")
    if script_name == "multi_miner_scenario_sweep_check.py":
        command.extend(["--execution-mode", "concurrent"])
        command.extend(["--json-out", str(state_dir / "multi_miner_scenario_sweep.json")])
    return command


def multi_miner_requeue_check_command(
    *,
    host: str,
    port: int,
    state_dir: Path,
    admin_token: str,
) -> list[str]:
    command = check_command(
        "multi_miner_scenario_sweep_check.py",
        host=host,
        port=port,
        state_dir=state_dir,
        admin_token=admin_token,
    )
    command.extend(["--failure-mode", "kill-after-claim"])
    json_index = command.index("--json-out")
    command[json_index + 1] = str(state_dir / "multi_miner_requeue.json")
    return command


def browser_check_command(
    script_name: str,
    *,
    host: str,
    port: int,
    web_port: int | None,
    state_dir: Path,
    browser: str,
    headful: bool,
    timeout: float,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / script_name),
        "--host",
        host,
    ]
    if script_name == "webrtc_smoke.py":
        command.extend([
            "--port",
            str(port),
            "--timeout",
            str(timeout),
        ])
    else:
        command.extend([
            "--coordinator-port",
            str(port),
            "--web-port",
            str(web_port),
            "--state-dir",
            str(state_dir),
        ])
        if script_name != "runtime_contract_check.py":
            command.extend(["--timeout", str(timeout)])
    if browser:
        command.extend(["--browser", browser])
    if headful:
        command.append("--headful")
    return command


def token_env_for(script_name: str, args: argparse.Namespace) -> dict[str, str]:
    if script_name not in TOKEN_ENV_SCRIPTS:
        return {"miner_token": "", "observer_token": ""}
    return {"miner_token": args.miner_token, "observer_token": args.observer_token}


def selected_checks(args: argparse.Namespace, state_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    specs = [
        ("readiness", "readiness_check.py", args.base_port, args.skip_readiness),
        ("runtime_matrix", "runtime_matrix_check.py", args.base_port + 34, args.skip_runtime_matrix),
        ("home_compute_demo", "home_compute_demo_check.py", args.base_port + 35, args.skip_home_compute_demo),
        ("home_compute_evidence", "home_compute_evidence_check.py", args.base_port + 36, args.skip_home_compute_evidence),
        ("api_contract", "api_contract_check.py", args.base_port + 1, args.skip_api_contract),
        ("chaos", "chaos_runner.py", args.base_port + 2, args.skip_chaos),
        ("trust_quarantine", "trust_quarantine_check.py", args.base_port + 3, args.skip_trust),
        ("replay_audit", "replay_audit_check.py", args.base_port + 4, args.skip_replay_audit),
        ("operator_control", "operator_control_check.py", args.base_port + 5, args.skip_operator),
        ("micro_transformer", "micro_transformer_smoke.py", args.base_port + 6, args.skip_micro_transformer),
        ("model_bundle", "model_bundle_smoke.py", args.base_port + 7, args.skip_model_bundle),
        ("result_idempotency", "result_idempotency_check.py", args.base_port + 8, args.skip_result_idempotency),
        ("result_ledger", "result_ledger_check.py", args.base_port + 9, args.skip_result_ledger),
        ("miner_resilience", "miner_resilience_check.py", args.base_port + 10, args.skip_miner_resilience),
        ("miner_auth", "miner_auth_check.py", args.base_port + 11, args.skip_miner_auth),
        ("observer_auth", "observer_auth_check.py", args.base_port + 12, args.skip_observer_auth),
        ("miner_registry_auth", "miner_registry_auth_check.py", args.base_port + 13, args.skip_miner_registry_auth),
        ("token_hash_auth", "token_hash_auth_check.py", args.base_port + 14, args.skip_token_hash_auth),
        ("outer_optimizer", "outer_optimizer_check.py", args.base_port + 15, args.skip_outer_optimizer),
        ("compressed_error_feedback", "compressed_error_feedback_check.py", args.base_port + 16, args.skip_compressed_error_feedback),
        ("delta_transport_negotiation", "delta_transport_negotiation_check.py", args.base_port + 17, args.skip_delta_transport_negotiation),
        ("model_bundle_inference", "model_bundle_inference_smoke.py", args.base_port + 30, args.skip_model_bundle_inference),
        ("inference_session_demo", "inference_session_demo.py", args.base_port + 31, args.skip_inference_session_demo),
        ("inference_session_client", "inference_session_client_check.py", args.base_port + 39, args.skip_inference_session_client),
        ("external_llm_inference", "external_llm_inference_smoke.py", args.base_port + 32, args.skip_external_llm_inference),
        ("external_llm_http_adapter", "external_llm_http_adapter_smoke.py", args.base_port + 33, args.skip_external_llm_http_adapter),
        ("external_llm_evidence", "external_llm_evidence_check.py", args.base_port + 42, args.skip_external_llm_evidence),
        ("admin_inference_session", "admin_inference_session_check.py", args.base_port + 38, args.skip_admin_inference_session),
    ]
    if args.include_remote_miner:
        specs.append((
            "remote_miner",
            "remote_miner_readiness_check.py",
            args.base_port + 18,
            args.skip_remote_miner,
        ))
    if args.include_remote_evidence:
        specs.append((
            "remote_compute_evidence",
            "remote_compute_evidence_check.py",
            args.base_port + 37,
            args.skip_remote_evidence,
        ))
    if args.include_multi_miner_sweep:
        specs.append((
            "multi_miner_scenario_sweep",
            "multi_miner_scenario_sweep_check.py",
            args.base_port + 40,
            args.skip_multi_miner_sweep,
        ))
    if args.include_multi_miner_requeue:
        specs.append((
            "multi_miner_requeue",
            "multi_miner_scenario_sweep_check.py",
            args.base_port + 41,
            args.skip_multi_miner_requeue,
        ))
    for name, script_name, port, skipped in specs:
        if skipped:
            continue
        state_dir = state_root / name
        token_env = token_env_for(script_name, args)
        checks.append({
            "name": name,
            "port": port,
            "state_dir": str(state_dir),
            "miner_token": token_env["miner_token"],
            "observer_token": token_env["observer_token"],
            "command": check_command(
                script_name,
                host=args.host,
                port=port,
                state_dir=state_dir,
                admin_token=args.admin_token,
            ),
        })
        if name == "multi_miner_requeue":
            checks[-1]["command"] = multi_miner_requeue_check_command(
                host=args.host,
                port=port,
                state_dir=state_dir,
                admin_token=args.admin_token,
            )
    if args.include_browser:
        browser_specs = [
            ("webrtc", "webrtc_smoke.py", args.base_port + 19, None, args.skip_webrtc),
            ("runtime_contract", "runtime_contract_check.py", args.base_port + 20, args.base_port + 21, args.skip_runtime_contract),
            ("browser_miner", "browser_miner_smoke.py", args.base_port + 22, args.base_port + 23, args.skip_browser_miner),
            ("browser_probe", "browser_probe_smoke.py", args.base_port + 24, args.base_port + 25, args.skip_browser_probe),
            ("capability_ledger", "capability_ledger_check.py", args.base_port + 26, args.base_port + 27, args.skip_capability_ledger),
        ]
        if args.include_browser_chaos:
            browser_specs.append((
                "browser_chaos",
                "browser_miner_chaos.py",
                args.base_port + 28,
                args.base_port + 29,
                args.skip_browser_chaos,
            ))
        for name, script_name, port, web_port, skipped in browser_specs:
            if skipped:
                continue
            state_dir = state_root / name
            token_env = token_env_for(script_name, args)
            checks.append({
                "name": name,
                "port": port,
                "web_port": web_port,
                "state_dir": str(state_dir),
                "miner_token": token_env["miner_token"],
                "observer_token": token_env["observer_token"],
                "command": browser_check_command(
                    script_name,
                    host=args.host,
                    port=port,
                    web_port=web_port,
                    state_dir=state_dir,
                    browser=args.browser,
                    headful=args.headful,
                    timeout=args.browser_timeout,
                ),
            })
    return checks


def run_check(check: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    Path(check["state_dir"]).mkdir(parents=True, exist_ok=True)
    started = utc_now()
    start_time = time.monotonic()
    result = {
        "name": check["name"],
        "port": check["port"],
        "state_dir": check["state_dir"],
        "command": check["command"],
        "started_at": started,
        "ok": False,
    }
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if check.get("miner_token"):
        env["CROWDTENSOR_MINER_TOKEN"] = check["miner_token"]
    if check.get("observer_token"):
        env["CROWDTENSOR_OBSERVER_TOKEN"] = check["observer_token"]
    try:
        completed = subprocess.run(
            check["command"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        result.update({
            "finished_at": utc_now(),
            "duration_seconds": round(time.monotonic() - start_time, 3),
            "returncode": None,
            "error": f"timed out after {timeout_seconds:.1f}s",
            "stdout": tail_text(exc.stdout or ""),
            "stderr": tail_text(exc.stderr or ""),
        })
        return result

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    result.update({
        "finished_at": utc_now(),
        "duration_seconds": round(time.monotonic() - start_time, 3),
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": tail_text(stdout),
        "stderr": tail_text(stderr),
    })
    summary_json = parse_summary_json(stdout)
    if summary_json is not None:
        result["summary_json"] = summary_json
        codes = diagnosis_codes_from_summary(summary_json)
        if codes:
            result["diagnosis_codes"] = codes
        for key in ["route", "schema", "workload", "request_count", "adapter_kind", "completion_count", "output_chars"]:
            if key in summary_json:
                result[key] = summary_json[key]
    if completed.returncode == 0:
        result["summary"] = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    else:
        result["error"] = stderr.strip().splitlines()[-1] if stderr.strip() else stdout.strip().splitlines()[-1] if stdout.strip() else "check failed"
    return result


def build_report(started_at: str, finished_at: str, duration_seconds: float, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ok": all(check.get("ok") for check in checks) if checks else False,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration_seconds, 3),
        "diagnosis_summary": diagnosis_summary(checks),
        "checks": checks,
    }


def write_report(report: dict[str, Any], path: str) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrowdTensorD runtime acceptance smoke pack.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8910)
    parser.add_argument("--state-root", default="")
    parser.add_argument("--admin-token", default="local-admin")
    parser.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    parser.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    parser.add_argument("--report", default="")
    parser.add_argument("--check-timeout", type=float, default=120.0)
    parser.add_argument("--include-browser", action="store_true")
    parser.add_argument("--include-browser-chaos", action="store_true")
    parser.add_argument("--include-micro-transformer", action="store_true")
    parser.add_argument("--include-remote-miner", action="store_true")
    parser.add_argument("--include-remote-evidence", action="store_true")
    parser.add_argument("--include-multi-miner-sweep", action="store_true")
    parser.add_argument("--include-multi-miner-requeue", action="store_true")
    parser.add_argument("--browser", default="")
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--browser-timeout", type=float, default=20.0)
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--skip-runtime-matrix", action="store_true")
    parser.add_argument("--skip-home-compute-demo", action="store_true")
    parser.add_argument("--skip-home-compute-evidence", action="store_true")
    parser.add_argument("--skip-api-contract", action="store_true")
    parser.add_argument("--skip-chaos", action="store_true")
    parser.add_argument("--skip-operator", action="store_true")
    parser.add_argument("--skip-replay-audit", action="store_true")
    parser.add_argument("--skip-trust", action="store_true")
    parser.add_argument("--skip-micro-transformer", action="store_true")
    parser.add_argument("--skip-model-bundle", action="store_true")
    parser.add_argument("--skip-model-bundle-inference", action="store_true")
    parser.add_argument("--skip-inference-session-demo", action="store_true")
    parser.add_argument("--skip-inference-session-client", action="store_true")
    parser.add_argument("--skip-admin-inference-session", action="store_true")
    parser.add_argument("--skip-external-llm-inference", action="store_true")
    parser.add_argument("--skip-external-llm-http-adapter", action="store_true")
    parser.add_argument("--skip-external-llm-evidence", action="store_true")
    parser.add_argument("--skip-result-idempotency", action="store_true")
    parser.add_argument("--skip-result-ledger", action="store_true")
    parser.add_argument("--skip-miner-resilience", action="store_true")
    parser.add_argument("--skip-miner-auth", action="store_true")
    parser.add_argument("--skip-observer-auth", action="store_true")
    parser.add_argument("--skip-miner-registry-auth", action="store_true")
    parser.add_argument("--skip-token-hash-auth", action="store_true")
    parser.add_argument("--skip-outer-optimizer", action="store_true")
    parser.add_argument("--skip-compressed-error-feedback", action="store_true")
    parser.add_argument("--skip-delta-transport-negotiation", action="store_true")
    parser.add_argument("--skip-remote-miner", action="store_true")
    parser.add_argument("--skip-remote-evidence", action="store_true")
    parser.add_argument("--skip-multi-miner-sweep", action="store_true")
    parser.add_argument("--skip-multi-miner-requeue", action="store_true")
    parser.add_argument("--skip-webrtc", action="store_true")
    parser.add_argument("--skip-runtime-contract", action="store_true")
    parser.add_argument("--skip-browser-miner", action="store_true")
    parser.add_argument("--skip-browser-probe", action="store_true")
    parser.add_argument("--skip-capability-ledger", action="store_true")
    parser.add_argument("--skip-browser-chaos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.state_root:
        state_root = Path(args.state_root)
        state_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_acceptance_")
        state_root = Path(temp_dir.name)

    started_at = utc_now()
    start_time = time.monotonic()
    try:
        checks = [
            run_check(check, timeout_seconds=args.check_timeout)
            for check in selected_checks(args, state_root)
        ]
        finished_at = utc_now()
        report = build_report(started_at, finished_at, time.monotonic() - start_time, checks)
        if args.report:
            write_report(report, args.report)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 1)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
