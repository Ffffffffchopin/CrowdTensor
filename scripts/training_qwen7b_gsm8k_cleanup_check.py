#!/usr/bin/env python3
"""Strictly validate a Qwen7B GSM8K private-payload cleanup audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import stable_hash
from scripts.training_cuda_kaggle_common import public_safety_errors
from scripts.training_qwen7b_gsm8k_cleanup import PRIVATE_PAYLOADS, SCHEMA


def check(path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except BaseException as exc:
        return {
            "schema": "crowdtensor_qwen7b_gsm8k_cleanup_check_v1",
            "ok": False,
            "cleanup_ready": False,
            "errors": ["cleanup_report_load_failed:" + type(exc).__name__],
            "error_count": 1,
        }
    if report.get("schema") != SCHEMA:
        errors.append("cleanup_schema_mismatch")
    if report.get("content_hash") != stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    ):
        errors.append("cleanup_content_hash_mismatch")
    hashes = dict(report.get("evidence_hashes") or {})
    if set(hashes) != {
        "dataset_manifest",
        "additional_dataset_manifests",
        "training_report",
        "baseline_report",
        "post_benchmark_report",
    } or any(not str(value).startswith("sha256:") for value in hashes.values()):
        errors.append("cleanup_evidence_hashes_invalid")
    payloads = dict(report.get("private_payloads") or {})
    if (
        not set(PRIVATE_PAYLOADS).issubset(payloads)
        or len(payloads) != int(report.get("dataset_manifest_count") or 0) * 2
        or any(
        value.get("present_before_cleanup") is not True
        or value.get("hash_verified_before_cleanup") is not True
        or value.get("removed") is not True
        or value.get("path_public") is not False
        or value.get("raw_content_public") is not False
        for value in payloads.values()
        )
    ):
        errors.append("cleanup_private_payload_evidence_invalid")
    required_true = (
        "remove_requested",
        "all_private_payloads_removed",
        "dataset_transient_directories_absent",
        "runtime_private_directories_absent",
        "training_live_cleanup_verified",
        "baseline_live_cleanup_verified",
        "post_benchmark_live_cleanup_verified",
    )
    if any(report.get(key) is not True for key in required_true):
        errors.append("cleanup_required_evidence_incomplete")
    if report.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_remain")
    if any(
        report.get(key) is not False
        for key in (
            "raw_text_public",
            "token_ids_public",
            "gold_answers_public",
            "credentials_public",
            "credential_paths_public",
            "private_paths_public",
        )
    ):
        errors.append("cleanup_public_safety_flags_invalid")
    safety = public_safety_errors(report)
    if safety:
        errors.append("cleanup_public_safety_scan_failed")
    ready_claim = bool(
        report.get("ok") is True
        and report.get("cleanup_ready") is True
        and not report.get("blockers")
    )
    if require_ready and not ready_claim:
        errors.append("cleanup_readiness_required")
    if ready_claim and errors:
        errors.append("cleanup_false_ready_claim")
    return {
        "schema": "crowdtensor_qwen7b_gsm8k_cleanup_check_v1",
        "ok": not errors,
        "cleanup_ready": ready_claim and not errors,
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
