#!/usr/bin/env python3
"""CI-safe check for the real Internet Swarm Inference Alpha path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "real_llm_internet_alpha_check_v1"

SECRET_FRAGMENTS = [
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "activation_result",
    "hidden_state",
    "input_ids",
    "logits",
    "inference_results",
    "inference_result",
    "sharded_inference_result",
    "real_llm_sharded_result",
    "output_text",
    "Bearer ",
    "SOURCE_TARBALL_B64",
    "MINER_ENV_TEXT",
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
    "CrowdTensor routes",
    "A miner returns",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
]

REQUIRED_CODES = {
    "real_llm_internet_alpha_ready",
    "real_llm_live_rc_ready",
    "real_llm_stage_requeue_ready",
    "stage_requeue_ready",
    "remote_real_llm_sharded_ready",
    "real_llm_artifact_ready",
    "stage_0_accepted",
    "stage_1_accepted",
    "activation_transport_ready",
    "baseline_match",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("command emitted no JSON object")


def run_json(command: list[str], *, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json_from_stdout(completed.stdout)


def assert_no_sensitive_output(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"real LLM Internet Alpha check leaked sensitive fragment: {fragment}")


def output_scope_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if "answer" not in str(output_request.get("summary") or ""):
        errors.append("output_request_summary")
    if "raw prompt" not in str(prompt_scope.get("summary") or ""):
        errors.append("prompt_scope_summary")
    if "answer transcript" not in str(answer_scope.get("summary") or ""):
        errors.append("answer_scope_summary")
    if output_request.get("include_output") is not False:
        errors.append("output_request_include_output")
    for key in ["raw_prompt_public", "raw_generated_text_public", "generated_token_ids_public"]:
        if output_request.get(key) is not False:
            errors.append(f"output_request_{key}")
    if output_request.get("public_artifact_safe") is not True:
        errors.append("output_request_public_artifact_safe")
    if prompt_scope.get("source") != "built-in-default-prompts":
        errors.append("prompt_scope_source")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_count")
    if prompt_scope.get("inline_prompt_text") is not False:
        errors.append("prompt_scope_inline_prompt_text")
    if prompt_scope.get("terminal_next_commands_local_private") is not False:
        errors.append("prompt_scope_terminal_next_commands_local_private")
    if prompt_scope.get("terminal_logs_local_private") is not False:
        errors.append("prompt_scope_terminal_logs_local_private")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_artifacts_prompt_placeholders")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("prompt_scope_raw_prompt_public")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append("prompt_scope_public_artifact_safe")
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append("answer_scope_state")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append("answer_visible_in_terminal")
    if answer_scope.get("terminal_only") is not False:
        errors.append("answer_terminal_only")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append("answer_saved_json_display")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append("answer_saved_markdown_display")
    for key in ["raw_prompt_public", "raw_generated_text_public", "generated_token_ids_public"]:
        if answer_scope.get(key) is not False:
            errors.append(f"answer_{key}")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_public_artifact_safe")
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts")
    for key in ["raw_prompt_public", "raw_generated_text_public", "generated_token_ids_public"]:
        if shareable.get(key) is not False:
            errors.append(f"shareable_{key}")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append("shareable_local_answer_terminal_only")
    if shareable.get("public_artifact_safe") is not True:
        errors.append("shareable_public_artifact_safe")
    return errors


def guidance_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    user_status = payload.get("user_status") if isinstance(payload.get("user_status"), dict) else {}
    review = payload.get("review_summary") if isinstance(payload.get("review_summary"), dict) else {}
    recommended = payload.get("recommended_next_command") if isinstance(payload.get("recommended_next_command"), dict) else {}
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    artifact_summary = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    not_completed = payload.get("not_completed") if isinstance(payload.get("not_completed"), list) else []
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    support = artifacts.get("support_bundle_json") if isinstance(artifacts.get("support_bundle_json"), dict) else {}
    if user_status.get("public_artifact_safe") is not True:
        errors.append("user_status_public_artifact_safe")
    if payload.get("ok") is True and user_status.get("state") != "ready":
        errors.append("user_status_ready_state")
    if review.get("schema") != "real_llm_internet_alpha_review_summary_v1":
        errors.append("review_schema")
    if payload.get("ok") is True and review.get("state") != "ready":
        errors.append("review_ready_state")
    if not isinstance(review.get("inspect_first"), str) or not review.get("inspect_first"):
        errors.append("review_inspect_first")
    if not isinstance(review.get("support_bundle"), str) or not review.get("support_bundle"):
        errors.append("review_support_bundle")
    if not isinstance(recommended.get("command_line"), str) or not recommended.get("command_line"):
        errors.append("recommended_command")
    if recommended.get("public_artifact_safe") is not True:
        errors.append("recommended_public_artifact_safe")
    if len(next_commands) < 2:
        errors.append("next_commands")
    if not_completed:
        errors.append("ready_report_not_completed")
    if artifact_summary.get("public_artifact_safe") is not True:
        errors.append("artifact_summary_public_artifact_safe")
    if artifact_summary.get("present_artifact_count") != artifact_summary.get("artifact_count"):
        errors.append("artifact_summary_present_count")
    if support.get("present") is not True:
        errors.append("support_bundle_present")
    return errors


def validate_local_generated(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "real_llm_internet_alpha_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected real LLM Internet Alpha report: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != "local-generated":
        raise SystemExit(f"unexpected mode: {payload.get('mode')}")
    codes = set(payload.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        raise SystemExit(f"missing readiness diagnosis: {missing}")
    if "external_runtime_verified" in codes:
        raise SystemExit("local generated Internet Alpha check must not claim external_runtime_verified")
    runtime = payload.get("runtime_classification") if isinstance(payload.get("runtime_classification"), dict) else {}
    if runtime.get("local_generated_stage_upload_standins") is not True:
        raise SystemExit(f"local generated runtime classification missing: {runtime}")
    if runtime.get("external_runtime_verified") is not False:
        raise SystemExit(f"local generated check must not claim external runtime: {runtime}")
    if runtime.get("stage_requeue_verified") is not True:
        raise SystemExit(f"stage requeue verification missing: {runtime}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in [
        "read_only",
        "cpu_only_workload",
        "summary_excludes_plaintext_tokens",
        "raw_activation_redacted",
        "local_requeue_verified",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    ]:
        if safety.get(key) is not True:
            raise SystemExit(f"safety flag {key} must be true: {safety}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in [
        "real_llm_internet_alpha_json",
        "real_llm_internet_alpha_markdown",
        "support_bundle_json",
        "real_llm_live_rc_json",
        "kill_stage0_after_claim_remote_real_llm_sharded_beta_json",
        "kill_stage1_after_claim_remote_real_llm_sharded_beta_json",
        "kill_stage0_after_claim_real_llm_sharded_evidence_json",
        "kill_stage1_after_claim_real_llm_sharded_evidence_json",
    ]:
        if (artifacts.get(name) or {}).get("present") is not True:
            raise SystemExit(f"missing artifact {name}: {artifacts.get(name)}")
    summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    for name in ["live_rc", "stage0_requeue", "stage1_requeue"]:
        summary = summaries.get(name) if isinstance(summaries.get(name), dict) else {}
        if summary.get("ok") is not True:
            raise SystemExit(f"payload summary {name} was not ok: {summary}")
    scope_errors = output_scope_errors(payload)
    if scope_errors:
        raise SystemExit(f"output scope errors: {scope_errors}")
    guide_errors = guidance_errors(payload)
    if guide_errors:
        raise SystemExit(f"guidance errors: {guide_errors}")
    assert_no_sensitive_output(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real Internet Swarm Inference Alpha check.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--port", type=int, default=9186)
    parser.add_argument("--base-port", type=int, default=9188)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.output_dir:
            base_dir = Path(args.output_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_real_llm_internet_alpha_")
            base_dir = Path(temp_dir.name)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "real_llm_internet_alpha_pack.py"),
            "--mode",
            "local-generated",
            "--output-dir",
            str(base_dir / "real-llm-internet-alpha"),
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--hf-model-id",
            args.hf_model_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ]
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        payload = run_json(command, timeout=max(args.timeout_seconds, 240.0) * 3 + 120.0)
        validate_local_generated(payload)
        print(json.dumps({
            "schema": SCHEMA,
            "ok": True,
            "output_dir": str(base_dir),
            "mode": payload.get("mode"),
            "diagnosis_codes": payload.get("diagnosis_codes") or [],
            "request_count": args.request_count,
            "hf_model_id": args.hf_model_id,
            "guidance_errors": [],
            "output_scope_errors": [],
            "artifacts": {
                "real_llm_internet_alpha_json": str(base_dir / "real-llm-internet-alpha" / "real_llm_internet_alpha.json"),
                "real_llm_internet_alpha_markdown": str(base_dir / "real-llm-internet-alpha" / "real_llm_internet_alpha.md"),
                "support_bundle_json": str(base_dir / "real-llm-internet-alpha" / "support_bundle.json"),
            },
        }, sort_keys=True))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
