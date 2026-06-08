#!/usr/bin/env python3
"""CI-safe checks for the GPU sharded multi-token generation Beta wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import gpu_sharded_generation_beta_pack as pack  # noqa: E402


SCHEMA = "gpu_sharded_generation_beta_check_v1"
REQUIRED_CODES = {
    "gpu_sharded_generation_ready",
    "multi_token_generation_ready",
    "public_swarm_gpu_beta_ready",
    "hf_transformers_cuda_ready",
    "stage0_partition_loaded",
    "stage1_partition_loaded",
    "partition_parameter_split_valid",
    "stage_local_partition_ready",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
}
SECRET_FRAGMENTS = {
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed(payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cmd"], 0, stdout=json.dumps(payload) + "\n", stderr="")


def option_value(command: list[str], option: str, default: str = "") -> str:
    if option not in command:
        return default
    index = command.index(option) + 1
    if index >= len(command):
        return default
    return command[index]


def synthetic_public_gpu_report(max_new_tokens: int, *, mode: str = "local-loopback") -> dict[str, Any]:
    codes = sorted(REQUIRED_CODES | {
        "read_only_workload",
        "not_production",
        "not_p2p",
        "gpu_loopback_generation_ready" if mode != "kaggle-auto" else "gpu_multi_machine_generation_ready",
    })
    if mode == "kaggle-auto":
        codes.extend([
            "public_swarm_gpu_beta_kaggle_auto_ready",
            "external_gpu_runtime_verified",
            "kaggle_kernels_deleted",
        ])
    return {
        "schema": "public_swarm_gpu_inference_beta_v1",
        "ok": True,
        "mode": mode,
        "diagnosis_codes": codes,
        "payload_summaries": {
            "real_llm_internet_beta": {
                "schema": "real_llm_internet_beta_v1",
                "ok": True,
                "generation": {
                    "max_new_tokens": max_new_tokens,
                    "generated_token_count": max_new_tokens,
                    "generated_text_hash": "sha256:synthetic-generation",
                    "generated_text_redacted": True,
                    "multi_token_generation_ready": True,
                },
                "external_alpha": {
                    "schema": "real_llm_internet_alpha_v1",
                    "ok": True,
                    "generation": {
                        "max_new_tokens": max_new_tokens,
                        "generated_token_count": max_new_tokens,
                        "generated_text_hash": "sha256:synthetic-generation",
                        "generated_text_redacted": True,
                        "multi_token_generation_ready": True,
                    },
                },
            },
        },
    }


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    if "public_swarm_gpu_inference_beta_pack.py" not in " ".join(command):
        raise AssertionError(command)
    max_new_tokens = int(option_value(command, "--max-new-tokens", "4"))
    script_index = next(
        (index for index, item in enumerate(command) if item.endswith("public_swarm_gpu_inference_beta_pack.py")),
        -1,
    )
    mode = command[script_index + 1] if script_index >= 0 and script_index + 1 < len(command) else "local-loopback"
    return completed(synthetic_public_gpu_report(max_new_tokens, mode=mode))


def validate_report(report: dict[str, Any], *, max_new_tokens: int) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ready")
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    if generation.get("generated_token_count") != max_new_tokens:
        errors.append("generated_token_count_mismatch")
    if generation.get("multi_token_generation_ready") is not True:
        errors.append("multi_token_generation_not_ready")
    codes = set(report.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        errors.append(f"missing_codes:{','.join(missing)}")
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append("output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        errors.append("output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append("output_request_raw_generated_text_public_mismatch")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append("output_request_generated_token_ids_public_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        errors.append("output_request_public_artifact_safe_mismatch")
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    if prompt_scope.get("source") != "imported-or-built-in-validation-prompts":
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 0:
        errors.append("prompt_scope_count_mismatch")
    if prompt_scope.get("inline_prompt_text") is not False:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not False:
        errors.append("prompt_scope_terminal_next_commands_local_private_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not False:
        errors.append("prompt_scope_terminal_logs_local_private_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_artifacts_prompt_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not False:
        errors.append("prompt_scope_prefer_prompt_file_or_stdin_mismatch")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append("prompt_scope_prompt_file_path_public_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append("prompt_scope_public_artifact_safe_mismatch")
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append("answer_scope_state_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append("answer_scope_visible_in_terminal_mismatch")
    if answer_scope.get("terminal_only") is not False:
        errors.append("answer_scope_terminal_only_mismatch")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append("answer_scope_saved_json_display_mismatch")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append("answer_scope_saved_markdown_display_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_mismatch")
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generated_text_public") is not False:
        errors.append("shareable_raw_generated_text_public_mismatch")
    if shareable.get("generated_token_ids_public") is not False:
        errors.append("shareable_generated_token_ids_public_mismatch")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state_mismatch")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append("shareable_local_answer_terminal_only_mismatch")
    encoded = json.dumps(report, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    if pack.has_raw_generation_payload(report):
        errors.append("raw_generation_payload_present")
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    own_json = artifacts.get("gpu_sharded_generation_beta_json") if isinstance(artifacts.get("gpu_sharded_generation_beta_json"), dict) else {}
    if own_json.get("present") is not True:
        errors.append("summary_artifact_missing")
    own_markdown = artifacts.get("gpu_sharded_generation_beta_markdown") if isinstance(artifacts.get("gpu_sharded_generation_beta_markdown"), dict) else {}
    if own_markdown.get("present") is not True:
        errors.append("markdown_artifact_missing")
    support_artifact = artifacts.get("support_bundle_json") if isinstance(artifacts.get("support_bundle_json"), dict) else {}
    if support_artifact.get("present") is not True:
        errors.append("support_bundle_artifact_missing")
    mode = str(report.get("mode") or "")
    provenance = report.get("runtime_provenance") if isinstance(report.get("runtime_provenance"), dict) else {}
    if provenance.get("schema") != "gpu_generation_runtime_provenance_v1":
        errors.append("runtime_provenance_schema_mismatch")
    if provenance.get("mode") != mode:
        errors.append("runtime_provenance_mode_mismatch")
    if provenance.get("public_artifact_safe") is not True:
        errors.append("runtime_provenance_public_artifact_safe_mismatch")
    if mode == "evidence-import":
        if provenance.get("proof_level") != "retained-evidence-import":
            errors.append("runtime_provenance_evidence_import_level_mismatch")
        if provenance.get("evidence_import") is not True:
            errors.append("runtime_provenance_evidence_import_flag_mismatch")
        if provenance.get("fresh_kaggle_gpu_attempted") is not False:
            errors.append("runtime_provenance_fresh_kaggle_attempt_mismatch")
        if not provenance.get("imported_evidence_path"):
            errors.append("runtime_provenance_imported_path_missing")
    elif mode == "local-loopback":
        if provenance.get("proof_level") != "local-loopback":
            errors.append("runtime_provenance_local_loopback_level_mismatch")
        if provenance.get("local_loopback_attempted") is not True:
            errors.append("runtime_provenance_local_loopback_flag_mismatch")
        if provenance.get("fresh_kaggle_gpu_attempted") is not False:
            errors.append("runtime_provenance_local_fresh_kaggle_attempt_mismatch")
    elif mode == "kaggle-auto":
        if provenance.get("proof_level") != "fresh-kaggle-gpu":
            errors.append("runtime_provenance_kaggle_level_mismatch")
        if provenance.get("fresh_kaggle_gpu_attempted") is not True:
            errors.append("runtime_provenance_kaggle_attempt_mismatch")
        if provenance.get("fresh_kaggle_gpu_verified") is not True:
            errors.append("runtime_provenance_kaggle_verified_mismatch")
    user_status = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    if user_status.get("state") not in {
        "evidence-import-ready",
        "local-loopback-ready",
        "fresh-kaggle-gpu-ready",
    }:
        errors.append("user_status_state_mismatch")
    if user_status.get("public_artifact_safe") is not True:
        errors.append("user_status_public_artifact_safe_mismatch")
    if not user_status.get("headline"):
        errors.append("user_status_headline_missing")
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    if review.get("schema") != "gpu_sharded_generation_beta_review_summary_v1":
        errors.append("review_summary_schema_mismatch")
    if review.get("state") != "ready":
        errors.append("review_summary_state_mismatch")
    if review.get("ready") is not True:
        errors.append("review_summary_ready_mismatch")
    if review.get("runtime_proof_level") != provenance.get("proof_level"):
        errors.append("review_summary_runtime_proof_level_mismatch")
    if review.get("public_artifact_safe") is not True:
        errors.append("review_summary_public_artifact_safe_mismatch")
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    if not recommended.get("command_line"):
        errors.append("recommended_next_command_missing")
    if recommended.get("public_artifact_safe") is not True:
        errors.append("recommended_next_command_public_artifact_safe_mismatch")
    next_commands = report.get("next_commands") if isinstance(report.get("next_commands"), list) else []
    if len([item for item in next_commands if isinstance(item, dict) and item.get("command_line")]) < 3:
        errors.append("next_commands_missing")
    not_completed = report.get("not_completed") if isinstance(report.get("not_completed"), list) else []
    if not_completed:
        errors.append(f"not_completed_present:{','.join(str(item) for item in not_completed[:5])}")
    artifact_summary = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    if artifact_summary.get("schema") != "gpu_sharded_generation_beta_artifact_summary_v1":
        errors.append("artifact_summary_schema_mismatch")
    if artifact_summary.get("public_artifact_safe") is not True:
        errors.append("artifact_summary_public_artifact_safe_mismatch")
    if not artifact_summary.get("inspect_first") or not artifact_summary.get("support_bundle"):
        errors.append("artifact_summary_paths_missing")
    if artifact_summary.get("artifact_count") != artifact_summary.get("present_artifact_count"):
        errors.append("artifact_summary_present_count_mismatch")
    return errors


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_gpu_generation_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.gpu_report).resolve() if args.gpu_report else output_dir / "synthetic_public_gpu_report.json"
    if args.gpu_report:
        report_args = [
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        generated_source = False
    else:
        write_json(source, synthetic_public_gpu_report(args.max_new_tokens))
        report_args = [
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        generated_source = True
    imported = pack.build_report(pack.parse_args(report_args))
    errors = validate_report(imported, max_new_tokens=args.max_new_tokens)

    loopback_report = None
    loopback_errors: list[str] = []
    if args.include_wrapper_check:
        loopback_report = pack.build_report(pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir / "wrapper"),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]), runner=fake_runner)
        loopback_errors = validate_report(loopback_report, max_new_tokens=args.max_new_tokens)

    ok = not errors and not loopback_errors
    result = {
        "schema": SCHEMA,
        "ok": ok,
        "output_dir": str(output_dir),
        "max_new_tokens": args.max_new_tokens,
        "generated_source": generated_source,
        "checks": {
            "evidence_import": {
                "ok": not errors,
                "errors": errors,
                "report_schema": imported.get("schema"),
                "generated_token_count": (imported.get("generation") or {}).get("generated_token_count"),
            },
            "wrapper": {
                "ok": not loopback_errors,
                "enabled": bool(args.include_wrapper_check),
                "errors": loopback_errors,
                "report_schema": loopback_report.get("schema") if loopback_report else None,
            },
        },
        "diagnosis_codes": [
            "gpu_sharded_generation_beta_check_ready" if ok else "gpu_sharded_generation_beta_check_blocked",
        ],
        "artifacts": {
            "source_report": str(source),
            "imported_report": str(output_dir / "gpu_sharded_generation_beta_evidence_import.json"),
        },
    }
    write_json(output_dir / "gpu_sharded_generation_beta_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CI-safe GPU sharded generation Beta checks.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--gpu-report", default="")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--include-wrapper-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    return args


def main() -> None:
    result = run_check(parse_args())
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
