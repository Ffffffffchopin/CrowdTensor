#!/usr/bin/env python3
"""CI-safe check for the CPU inference Beta aggregate path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_FRAGMENTS = [
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
    "observer-secret",
    "admin-secret",
    "runtime-secret",
]


def run_json(command: list[str], *, timeout: float) -> dict:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed([line for line in completed.stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"command emitted no JSON: {' '.join(command)}\nstdout:\n{completed.stdout}")


def validate_report(payload: dict, *, mode: str, workload: str) -> list[str]:
    if payload.get("schema") != "cpu_inference_beta_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected CPU inference Beta report: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != mode:
        raise SystemExit(f"unexpected mode: {payload.get('mode')}")
    if mode != "local" and payload.get("workload") != workload:
        raise SystemExit(f"unexpected workload: {payload.get('workload')}")
    if "cpu_inference_beta_ready" not in payload.get("diagnosis_codes", []):
        raise SystemExit(f"missing readiness diagnosis: {payload.get('diagnosis_codes')}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"CPU inference Beta report leaked sensitive fragment: {fragment}")
    if not (payload.get("artifacts") or {}).get("cpu_inference_beta_json", {}).get("present"):
        raise SystemExit("CPU inference Beta JSON artifact was not reported present")
    scope_errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        scope_errors.append("output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        scope_errors.append("output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generation_public") is not False:
        scope_errors.append("output_request_raw_generation_public_mismatch")
    if output_request.get("raw_external_llm_output_public") is not False:
        scope_errors.append("output_request_raw_external_llm_output_public_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        scope_errors.append("output_request_public_artifact_safe_mismatch")
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    if prompt_scope.get("source") != "built-in-fixed-scenarios":
        scope_errors.append("prompt_scope_source_mismatch")
    if prompt_scope.get("inline_prompt_text") is not False:
        scope_errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not False:
        scope_errors.append("prompt_scope_terminal_next_commands_local_private_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        scope_errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        scope_errors.append("prompt_scope_public_artifact_safe_mismatch")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        scope_errors.append("answer_scope_state_mismatch")
    if answer_scope.get("raw_generation_public") is not False:
        scope_errors.append("answer_scope_raw_generation_public_mismatch")
    if answer_scope.get("raw_external_llm_output_public") is not False:
        scope_errors.append("answer_scope_raw_external_llm_output_public_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        scope_errors.append("answer_scope_public_artifact_safe_mismatch")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        scope_errors.append("shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        scope_errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generation_public") is not False:
        scope_errors.append("shareable_raw_generation_public_mismatch")
    if shareable.get("raw_external_llm_output_public") is not False:
        scope_errors.append("shareable_raw_external_llm_output_public_mismatch")
    if scope_errors:
        raise SystemExit(f"CPU inference Beta scope contract failed: {scope_errors}")
    guidance_errors: list[str] = []
    user_status = payload.get("user_status") if isinstance(payload.get("user_status"), dict) else {}
    review = payload.get("review_summary") if isinstance(payload.get("review_summary"), dict) else {}
    recommended = payload.get("recommended_next_command") if isinstance(payload.get("recommended_next_command"), dict) else {}
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    artifact_summary = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    not_completed = payload.get("not_completed") if isinstance(payload.get("not_completed"), list) else []
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    support = artifacts.get("support_bundle_json") if isinstance(artifacts.get("support_bundle_json"), dict) else {}
    if user_status.get("public_artifact_safe") is not True:
        guidance_errors.append("user_status_public_artifact_safe_mismatch")
    if user_status.get("state") != "ready":
        guidance_errors.append("user_status_ready_state_mismatch")
    if review.get("schema") != "cpu_inference_beta_review_summary_v1":
        guidance_errors.append("review_summary_schema_mismatch")
    if review.get("state") != "ready":
        guidance_errors.append("review_summary_ready_state_mismatch")
    if not isinstance(recommended.get("command_line"), str) or not recommended.get("command_line"):
        guidance_errors.append("recommended_next_command_missing")
    if recommended.get("public_artifact_safe") is not True:
        guidance_errors.append("recommended_next_command_public_artifact_safe_mismatch")
    if len(next_commands) < 2:
        guidance_errors.append("next_commands_missing")
    if not_completed:
        guidance_errors.append("ready_report_not_completed")
    if artifact_summary.get("public_artifact_safe") is not True:
        guidance_errors.append("artifact_summary_public_artifact_safe_mismatch")
    if artifact_summary.get("present_artifact_count") != artifact_summary.get("artifact_count"):
        guidance_errors.append("artifact_summary_present_count_mismatch")
    if support.get("present") is not True:
        guidance_errors.append("support_bundle_missing")
    if guidance_errors:
        raise SystemExit(f"CPU inference Beta guidance contract failed: {guidance_errors}")
    return guidance_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CPU inference Beta aggregate check.")
    parser.add_argument("--mode", choices=["local", "remote-loopback"], default="local")
    parser.add_argument("--workload", choices=["model-bundle", "external-llm", "all"], default="all")
    parser.add_argument("--base-port", type=int, default=8970)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--external-llm-request-count", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_cpu_infer_beta_") as temp:
        output_dir = Path(temp) / "cpu-infer"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "cpu_inference_beta_pack.py"),
            "--mode",
            args.mode,
            "--workload",
            args.workload,
            "--output-dir",
            str(output_dir),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--timeout-seconds",
            str(int(args.timeout_seconds)),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--json",
        ]
        payload = run_json(command, timeout=args.timeout_seconds)
        guidance_errors = validate_report(payload, mode=args.mode, workload=args.workload)
        print(json.dumps({
            "ok": True,
            "schema": "cpu_inference_beta_check_v1",
            "beta_schema": payload.get("schema"),
            "mode": payload.get("mode"),
            "workload": payload.get("workload"),
            "step_count": len(payload.get("steps") or []),
            "diagnosis_codes": payload.get("diagnosis_codes"),
            "guidance_errors": guidance_errors,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
