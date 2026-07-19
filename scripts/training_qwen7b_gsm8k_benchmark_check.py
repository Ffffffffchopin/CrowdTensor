#!/usr/bin/env python3
"""Validate one isolated Qwen2.5-7B GSM8K benchmark live report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import MODEL_ID, MODEL_REVISION
from scripts.training_cuda_kaggle_common import public_safety_errors
from scripts.training_qwen7b_gsm8k_benchmark_live_probe import SCHEMA


def _validate_pass(value: dict[str, Any], errors: list[str], name: str) -> None:
    records = list(value.get("records") or [])
    count = int(value.get("example_count") or 0)
    correct = sum(bool(item.get("normalized_exact_match")) for item in records)
    valid = sum(bool(item.get("answer_valid")) for item in records)
    strict = sum(bool(item.get("strict_exact_match")) for item in records)
    if (
        count != 128
        or len(records) != count
        or len({int(item.get("example_index", -1)) for item in records}) != count
        or int(value.get("normalized_exact_match_count") or 0) != correct
        or int(value.get("valid_answer_count") or 0) != valid
        or int(value.get("strict_exact_match_count") or 0) != strict
        or abs(float(value.get("normalized_exact_match") or 0.0) - correct / count)
        > 1e-12
        or value.get("records_hash") != stable_hash(records)
    ):
        errors.append("benchmark_pass_invalid:" + name)


def check(path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except BaseException as exc:
        return {
            "schema": "crowdtensor_qwen7b_gsm8k_benchmark_check_v1",
            "ok": False,
            "benchmark_ready": False,
            "errors": ["benchmark_report_load_failed:" + type(exc).__name__],
            "error_count": 1,
        }
    if report.get("schema") != SCHEMA:
        errors.append("benchmark_schema_mismatch")
    if report.get("content_hash") != stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    ):
        errors.append("benchmark_content_hash_mismatch")
    mode = str(report.get("mode") or "")
    worker = dict(report.get("worker") or {})
    if (
        mode not in {"base", "adapter", "both"}
        or worker.get("mode") != mode
        or worker.get("model_id") != MODEL_ID
        or worker.get("model_revision") != MODEL_REVISION
        or worker.get("kaggle_kernel") is not True
        or int(worker.get("cuda_device_count") or 0) < 2
        or int(worker.get("benchmark_example_count") or 0) != 128
        or worker.get("public_artifact_safe") is not True
    ):
        errors.append("benchmark_worker_contract_invalid")
    if worker.get("content_hash") != stable_hash(
        {key: value for key, value in worker.items() if key != "content_hash"}
    ):
        errors.append("benchmark_worker_content_hash_mismatch")
    passes = dict(worker.get("passes") or {})
    for name in ({"base"} if mode == "base" else {"adapter"} if mode == "adapter" else {"base", "adapter"}):
        _validate_pass(dict(passes.get(name) or {}), errors, name)
    if mode in {"adapter", "both"} and (
        report.get("adapter_input_materialized") is not True
        or (report.get("dataset_attachment_preflight") or {}).get("ready") is not True
        or worker.get("standard_peft_reload_verified") is not True
        or worker.get("input_hashes_verified") is not True
        or set(worker.get("input_match_counts") or {})
        != {
            "qwen7b_gsm8k_benchmark_private.json",
            "qwen7b_gsm8k_validation_private.json",
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        or any(
            int(value.get("hash_match_count") or 0) < 1
            for value in (worker.get("input_match_counts") or {}).values()
        )
        or not str(worker.get("adapter_file_hash") or "").startswith("sha256:")
    ):
        errors.append("benchmark_standard_peft_reload_invalid")
    if mode == "both":
        base = dict(passes.get("base") or {})
        adapter = dict(passes.get("adapter") or {})
        if (
            worker.get("benchmark_prompt_hash") == ""
            or worker.get("benchmark_gold_hash") == ""
            or [
                (item.get("example_index"), item.get("prompt_hash"), item.get("gold_hash"))
                for item in base.get("records") or []
            ]
            != [
                (item.get("example_index"), item.get("prompt_hash"), item.get("gold_hash"))
                for item in adapter.get("records") or []
            ]
            or int((passes.get("base_validation") or {}).get("sequence_count") or 0) < 8
            or int((passes.get("adapter_validation") or {}).get("sequence_count") or 0) < 8
        ):
            errors.append("benchmark_same_runtime_alignment_invalid")
    cleanup = dict(report.get("cleanup") or {})
    if (
        cleanup.get("live_resources_left_running") is not False
        or cleanup.get("kernel_deleted") is not True
        or cleanup.get("private_dataset_deleted") is not True
        or cleanup.get("private_runtime_removed") is not True
    ):
        errors.append("benchmark_cleanup_invalid")
    if any(
        report.get(key) is not False
        for key in (
            "raw_text_public",
            "token_ids_public",
            "generated_text_public",
            "gold_answers_public",
            "credentials_public",
            "credential_paths_public",
            "private_paths_public",
        )
    ):
        errors.append("benchmark_public_safety_flags_invalid")
    safety = public_safety_errors(report)
    if safety:
        errors.append("benchmark_public_safety_scan_failed")
    ready_claim = bool(
        report.get("ok") is True
        and report.get("live_run_performed") is True
        and worker.get("ok") is True
        and not report.get("blockers")
    )
    if require_ready and not ready_claim:
        errors.append("benchmark_readiness_required")
    if ready_claim and errors:
        errors.append("benchmark_false_ready_claim")
    return {
        "schema": "crowdtensor_qwen7b_gsm8k_benchmark_check_v1",
        "ok": not errors,
        "benchmark_ready": ready_claim and not errors,
        "public_artifact_safe": not safety,
        "errors": errors,
        "error_count": len(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report, require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
