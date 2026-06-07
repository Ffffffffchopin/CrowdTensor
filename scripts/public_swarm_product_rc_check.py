#!/usr/bin/env python3
"""Validate the Public Swarm Product RC wrapper."""

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

import public_swarm_product_rc_pack as pack  # noqa: E402


SCHEMA = "public_swarm_product_rc_check_v1"


def completed(payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cmd"], 0, stdout=json.dumps(payload) + "\n", stderr="")


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "session_protocol_check.py" in joined:
        return completed({
            "schema": "session_protocol_check_v1",
            "ok": True,
            "route": {"usable_now": True},
            "diagnosis_codes": ["session_protocol_ready"],
        })
    if "p2p_lite_discovery_check.py" in joined:
        return completed({
            "schema": "p2p_lite_discovery_check_v1",
            "ok": True,
            "cpu_route": {"ok": True},
            "cuda_route": {"ok": True},
            "diagnosis_codes": ["p2p_lite_discovery_ready"],
        })
    raise AssertionError(command)


def synthetic_gpu_report(path: Path, max_new_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "public_swarm_gpu_inference_beta_v1",
        "ok": True,
        "mode": "kaggle-auto",
        "diagnosis_codes": [
            "public_swarm_gpu_beta_ready",
            "public_swarm_gpu_beta_kaggle_auto_ready",
            "external_gpu_runtime_verified",
            "kaggle_kernels_deleted",
            "hf_transformers_cuda_ready",
            "stage0_partition_loaded",
            "stage1_partition_loaded",
            "partition_parameter_split_valid",
            "stage_local_partition_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "multi_token_generation_ready",
        ],
        "payload_summaries": {
            "gpu": {
                "generation": {
                    "max_new_tokens": max_new_tokens,
                    "generated_token_count": max_new_tokens,
                    "generated_text_hash": "sha256:synthetic",
                    "generated_text_redacted": True,
                    "multi_token_generation_ready": True,
                }
            }
        },
    }), encoding="utf-8")


def output_scope_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
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
    if answer_scope.get("raw_prompt_public") is not False:
        errors.append("answer_scope_raw_prompt_public_mismatch")
    if answer_scope.get("raw_generated_text_public") is not False:
        errors.append("answer_scope_raw_generated_text_public_mismatch")
    if answer_scope.get("generated_token_ids_public") is not False:
        errors.append("answer_scope_generated_token_ids_public_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_mismatch")
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generated_text_public") is not False:
        errors.append("shareable_raw_generated_text_public_mismatch")
    if shareable.get("generated_token_ids_public") is not False:
        errors.append("shareable_generated_token_ids_public_mismatch")
    if shareable.get("local_output_display_only") is not False:
        errors.append("shareable_local_output_display_only_mismatch")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state_mismatch")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append("shareable_local_answer_terminal_only_mismatch")
    return errors


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_public_swarm_product_check_")).resolve()
    gpu_report = output_dir / "synthetic_gpu_report.json"
    synthetic_gpu_report(gpu_report, args.max_new_tokens)
    report = pack.build_report(pack.parse_args([
        "--output-dir",
        str(output_dir / "rc"),
        "--gpu-report",
        str(gpu_report),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]), runner=fake_runner)
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("rc_not_ready")
    codes = set(report.get("diagnosis_codes") or [])
    for code in [
        "public_swarm_product_rc_ready",
        "coordinator_product_surface_ready",
        "session_protocol_ready",
        "p2p_lite_discovery_ready",
        "gpu_generation_evidence_import_ready",
    ]:
        if code not in codes:
            errors.append(f"missing_code:{code}")
    scope_errors = output_scope_errors(report)
    errors.extend(scope_errors)
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    rc_json = artifacts.get("public_swarm_product_rc_json") if isinstance(artifacts.get("public_swarm_product_rc_json"), dict) else {}
    rc_markdown = artifacts.get("public_swarm_product_rc_markdown") if isinstance(artifacts.get("public_swarm_product_rc_markdown"), dict) else {}
    if rc_json.get("present") is not True:
        errors.append("public_swarm_product_rc_json_missing")
    if rc_markdown.get("present") is not True:
        errors.append("public_swarm_product_rc_markdown_missing")
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "output_dir": str(output_dir),
        "errors": errors,
        "rc_schema": report.get("schema"),
        "rc_ok": report.get("ok"),
        "diagnosis_codes": ["public_swarm_product_rc_check_ready"] if not errors else ["public_swarm_product_rc_check_blocked"],
        "artifacts": {
            "rc_report": str(output_dir / "rc" / "public_swarm_product_rc.json"),
        },
    }
    (output_dir / "public_swarm_product_rc_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Swarm Product RC wrapper.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Public Swarm Product RC check ready: {result.get('ok')}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
