#!/usr/bin/env python3
"""CI-safe check for Public Swarm Inference Alpha RC evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "public_swarm_inference_alpha_rc_check_v1"
SECRET_FRAGMENTS = (
    "lease_token",
    "idempotency_key",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "real_llm_sharded_result",
    "generated_text",
    "generated_token_ids",
    "Bearer ",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
)


def run_json(command: list[str], *, timeout: float) -> dict[str, Any]:
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


def assert_no_sensitive_output(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"Public Swarm Inference Alpha RC check leaked sensitive fragment: {fragment}")


def validate_report(payload: dict[str, Any], *, mode: str) -> list[str]:
    if payload.get("schema") != "public_swarm_inference_alpha_rc_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected Public Swarm Inference Alpha RC report: {json.dumps(payload, sort_keys=True)}")
    codes = payload.get("diagnosis_codes") or []
    if mode == "evidence-import":
        for code in [
            "public_swarm_inference_alpha_rc_ready",
            "public_swarm_alpha_rc_evidence_imported",
            "stage0_live_requeue_evidence_ready",
            "stage1_live_requeue_evidence_ready",
            "public_swarm_live_requeue_evidence_ready",
            "public_swarm_live_requeue_summary_ready",
            "public_swarm_alpha_private_artifacts_absent",
        ]:
            if code not in codes:
                raise SystemExit(f"missing RC readiness diagnosis {code}: {codes}")
        imported = payload.get("imported_reports") or {}
        for stage in ["stage0", "stage1"]:
            stage_summary = imported.get(stage) or {}
            if stage_summary.get("ready") is not True:
                raise SystemExit(f"{stage} retained live evidence is not ready: {stage_summary}")
            requeue = stage_summary.get("live_requeue_summary") or {}
            for key in ["claim_observed", "victim_kernel_deleted", "rescued_result"]:
                if requeue.get(key) is not True:
                    raise SystemExit(f"{stage} missing live requeue summary {key}: {requeue}")
            if not requeue.get("lease_expired"):
                raise SystemExit(f"{stage} missing live requeue summary lease_expired: {requeue}")
            if requeue.get("victim_result_accepted") is not False:
                raise SystemExit(f"{stage} victim result was not rejected: {requeue}")
        summary = imported.get("summary") or {}
        if summary.get("ready") is not True:
            raise SystemExit(f"live requeue summary is not ready: {summary}")
    else:
        for code in ["public_swarm_alpha_rc_local_smoke_ready", "public_swarm_alpha_rc_contract_ready"]:
            if code not in codes:
                raise SystemExit(f"missing local smoke readiness diagnosis {code}: {codes}")
    safety = payload.get("safety") or {}
    for key in ["cpu_only", "not_production", "not_p2p", "not_large_model_serving"]:
        if safety.get(key) is not True:
            raise SystemExit(f"missing safety flag {key}: {safety}")
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        raise SystemExit(f"output_request include_output mismatch: {output_request}")
    if output_request.get("raw_prompt_public") is not False:
        raise SystemExit(f"output_request raw_prompt_public mismatch: {output_request}")
    if output_request.get("raw_generation_public") is not False:
        raise SystemExit(f"output_request raw_generation_public mismatch: {output_request}")
    if output_request.get("generation_ids_public") is not False:
        raise SystemExit(f"output_request generation_ids_public mismatch: {output_request}")
    if output_request.get("local_output_display_only") is not False:
        raise SystemExit(f"output_request local_output_display_only mismatch: {output_request}")
    if output_request.get("public_artifact_safe") is not True:
        raise SystemExit(f"output_request public_artifact_safe mismatch: {output_request}")
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    if prompt_scope.get("source") != "imported-or-built-in-validation-prompts":
        raise SystemExit(f"prompt_scope source mismatch: {prompt_scope}")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 0:
        raise SystemExit(f"prompt_scope count mismatch: {prompt_scope}")
    if prompt_scope.get("inline_prompt_text") is not False:
        raise SystemExit(f"prompt_scope inline_prompt_text mismatch: {prompt_scope}")
    if prompt_scope.get("terminal_next_commands_local_private") is not False:
        raise SystemExit(f"prompt_scope terminal_next_commands_local_private mismatch: {prompt_scope}")
    if prompt_scope.get("terminal_logs_local_private") is not False:
        raise SystemExit(f"prompt_scope terminal_logs_local_private mismatch: {prompt_scope}")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        raise SystemExit(f"prompt_scope saved_artifacts_prompt_placeholders mismatch: {prompt_scope}")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        raise SystemExit(f"prompt_scope saved_artifacts_public_safe mismatch: {prompt_scope}")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not False:
        raise SystemExit(f"prompt_scope prefer_prompt_file_or_stdin mismatch: {prompt_scope}")
    if prompt_scope.get("prompt_file_path_public") is not False:
        raise SystemExit(f"prompt_scope prompt_file_path_public mismatch: {prompt_scope}")
    if prompt_scope.get("raw_prompt_public") is not False:
        raise SystemExit(f"prompt_scope raw_prompt_public mismatch: {prompt_scope}")
    if prompt_scope.get("public_artifact_safe") is not True:
        raise SystemExit(f"prompt_scope public_artifact_safe mismatch: {prompt_scope}")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        raise SystemExit(f"answer_scope state mismatch: {answer_scope}")
    if answer_scope.get("visible_in_terminal") is not False:
        raise SystemExit(f"answer_scope visible_in_terminal mismatch: {answer_scope}")
    if answer_scope.get("terminal_only") is not False:
        raise SystemExit(f"answer_scope terminal_only mismatch: {answer_scope}")
    if answer_scope.get("saved_json_display") != "hash-only":
        raise SystemExit(f"answer_scope saved_json_display mismatch: {answer_scope}")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        raise SystemExit(f"answer_scope saved_markdown_display mismatch: {answer_scope}")
    if answer_scope.get("raw_prompt_public") is not False:
        raise SystemExit(f"answer_scope raw_prompt_public mismatch: {answer_scope}")
    if answer_scope.get("raw_generation_public") is not False:
        raise SystemExit(f"answer_scope raw_generation_public mismatch: {answer_scope}")
    if answer_scope.get("generation_ids_public") is not False:
        raise SystemExit(f"answer_scope generation_ids_public mismatch: {answer_scope}")
    if answer_scope.get("public_artifact_safe") is not True:
        raise SystemExit(f"answer_scope public_artifact_safe mismatch: {answer_scope}")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        raise SystemExit(f"shareable saved_artifacts_public_safe mismatch: {shareable}")
    if shareable.get("raw_prompt_public") is not False:
        raise SystemExit(f"shareable raw_prompt_public mismatch: {shareable}")
    if shareable.get("raw_generation_public") is not False:
        raise SystemExit(f"shareable raw_generation_public mismatch: {shareable}")
    if shareable.get("generation_ids_public") is not False:
        raise SystemExit(f"shareable generation_ids_public mismatch: {shareable}")
    if shareable.get("local_output_display_only") is not False:
        raise SystemExit(f"shareable local_output_display_only mismatch: {shareable}")
    if shareable.get("answer_scope_state") != "no-local-answer":
        raise SystemExit(f"shareable answer_scope_state mismatch: {shareable}")
    if shareable.get("local_answer_terminal_only") is not False:
        raise SystemExit(f"shareable local_answer_terminal_only mismatch: {shareable}")
    if shareable.get("public_artifact_safe") is not True:
        raise SystemExit(f"shareable public_artifact_safe mismatch: {shareable}")
    artifacts = payload.get("artifacts") or {}
    for name in ["public_swarm_inference_alpha_rc_json", "public_swarm_inference_alpha_rc_markdown"]:
        artifact = artifacts.get(name) or {}
        if artifact.get("present") is not True:
            raise SystemExit(f"missing RC artifact {name}: {artifact}")
    guidance_errors: list[str] = []
    user_status = payload.get("user_status") if isinstance(payload.get("user_status"), dict) else {}
    review = payload.get("review_summary") if isinstance(payload.get("review_summary"), dict) else {}
    recommended = payload.get("recommended_next_command") if isinstance(payload.get("recommended_next_command"), dict) else {}
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    artifact_summary = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    not_completed = payload.get("not_completed") if isinstance(payload.get("not_completed"), list) else []
    support = artifacts.get("support_bundle_json") if isinstance(artifacts.get("support_bundle_json"), dict) else {}
    if user_status.get("public_artifact_safe") is not True:
        guidance_errors.append("user_status_public_artifact_safe_mismatch")
    if payload.get("ok") is True and user_status.get("state") != "ready":
        guidance_errors.append("user_status_ready_state_mismatch")
    if review.get("schema") != "public_swarm_inference_alpha_rc_review_summary_v1":
        guidance_errors.append("review_summary_schema_mismatch")
    if payload.get("ok") is True and review.get("state") != "ready":
        guidance_errors.append("review_summary_ready_state_mismatch")
    if not isinstance(review.get("inspect_first"), str) or not review.get("inspect_first"):
        guidance_errors.append("review_summary_inspect_first_missing")
    if not isinstance(review.get("support_bundle"), str) or not review.get("support_bundle"):
        guidance_errors.append("review_summary_support_bundle_missing")
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
        raise SystemExit(f"RC guidance contract failed: {guidance_errors}")
    assert_no_sensitive_output(payload)
    return guidance_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Swarm Inference Alpha RC evidence.")
    parser.add_argument("--mode", choices=["evidence-import", "local-smoke"], default="evidence-import")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stage0-report", default="")
    parser.add_argument("--stage1-report", default="")
    parser.add_argument("--summary-report", default="")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_public_swarm_alpha_rc_") as temp:
        output_dir = Path(args.output_dir) if args.output_dir else Path(temp) / "alpha-rc"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "public_swarm_inference_alpha_rc_pack.py"),
            "--mode",
            args.mode,
            "--output-dir",
            str(output_dir),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ]
        if args.stage0_report:
            command.extend(["--stage0-report", args.stage0_report])
        if args.stage1_report:
            command.extend(["--stage1-report", args.stage1_report])
        if args.summary_report:
            command.extend(["--summary-report", args.summary_report])
        payload = run_json(command, timeout=args.timeout_seconds + 30.0)
        guidance_errors = validate_report(payload, mode=args.mode)
        print(json.dumps({
            "ok": True,
            "schema": SCHEMA,
            "mode": args.mode,
            "rc_schema": payload.get("schema"),
            "diagnosis_codes": payload.get("diagnosis_codes"),
            "artifact_count": len(payload.get("artifacts") or {}),
            "guidance_errors": guidance_errors,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
