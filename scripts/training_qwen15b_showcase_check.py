#!/usr/bin/env python3
"""Check the public contract for a Qwen training showcase artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_qwen15b_training_showcase_v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("showcase artifact must be an object")
    return value


def check_report(report: dict[str, Any], *, require_ready: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema_invalid")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_not_safe")
    if report.get("content_hash"):
        expected = "sha256:" + hashlib.sha256(
            json.dumps(
                {key: value for key, value in report.items() if key != "content_hash"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if report.get("content_hash") != expected:
            errors.append("content_hash_invalid")
    model = dict(report.get("model") or {})
    dataset = dict(report.get("dataset") or {})
    training = dict(report.get("training") or {})
    evaluation = dict(report.get("evaluation") or {})
    continuation = dict(report.get("elastic_continuation") or {})
    cleanup = dict(report.get("cleanup") or {})
    if model.get("model_id") != "Qwen/Qwen2.5-1.5B":
        errors.append("model_not_pinned_qwen15b")
    if model.get("model_revision") != "8faed761d45a263340a0528343f099c05c9a4323":
        errors.append("model_revision_invalid")
    if int(model.get("parameter_count") or 0) < 1_000_000_000:
        errors.append("model_parameter_count_too_small")
    if not dataset.get("dataset_id") or not dataset.get("dataset_revision"):
        errors.append("dataset_revision_missing")
    if dataset.get("dataset_id") != "Salesforce/wikitext" or dataset.get(
        "dataset_revision"
    ) != "b08601e04326c79dfdd32d625aee71d232d685c3":
        errors.append("dataset_source_not_pinned")
    steps = int(training.get("target_optimizer_steps") or 0)
    replacement = int(training.get("replacement_step") or 0)
    tokens = int(training.get("training_token_count") or 0)
    if steps < 128:
        errors.append("training_steps_below_showcase_floor")
    if tokens < 65536:
        errors.append("training_token_budget_below_showcase_floor")
    if not 0 < replacement < steps:
        errors.append("replacement_boundary_invalid")
    before = float(evaluation.get("before_validation_loss") or 0.0)
    after = float(evaluation.get("after_validation_loss") or 0.0)
    relative = float(evaluation.get("relative_validation_loss_improvement") or 0.0)
    if not (math.isfinite(before) and math.isfinite(after) and before > 0 and after > 0):
        errors.append("validation_loss_invalid")
    if not evaluation.get("validation_loss_reduced"):
        errors.append("validation_loss_not_reduced")
    if relative < 0.01:
        errors.append("validation_improvement_below_one_percent")
    if evaluation.get("evaluation_verified") is not True:
        errors.append("evaluation_not_verified")
    if evaluation.get("standard_peft_cpu_load") is not True:
        errors.append("standard_peft_cpu_reload_missing")
    if evaluation.get("standard_peft_cuda_load") is not True:
        errors.append("standard_peft_cuda_reload_missing")
    adapter = dict(report.get("adapter") or {})
    if adapter.get("standard_peft_format") is not True:
        errors.append("standard_peft_export_missing")
    if adapter.get("archive_verified") is not True:
        errors.append("standard_peft_archive_not_verified")
    required_continuation = (
        "old_kernels_deleted_before_replacement",
        "zero_miner_pause_verified",
        "new_miner_checkpoint_restore_verified",
        "exactly_once_contiguous_steps",
    )
    for key in required_continuation:
        if continuation.get(key) is not True:
            errors.append(f"continuation_{key}_missing")
    if int(continuation.get("old_generation_kernel_count") or 0) != 2:
        errors.append("old_generation_kernel_count_invalid")
    if int(continuation.get("replacement_generation_kernel_count") or 0) != 2:
        errors.append("replacement_generation_kernel_count_invalid")
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_remaining")
    if cleanup.get("all_four_kernels_deleted") is not True:
        errors.append("cleanup_kernel_deletion_missing")
    for key in (
        "private_runtime_removed",
        "rendezvous_payloads_removed",
        "coordinator_stopped",
        "tunnel_stopped",
    ):
        if cleanup.get(key) is not True:
            errors.append(f"cleanup_{key}_missing")
    if report.get("gates") and not all(
        value is True for value in dict(report["gates"]).values()
    ):
        errors.append("gate_summary_not_all_true")
    if require_ready and report.get("showcase_ready") is not True:
        errors.append("showcase_not_ready")
    result = {
        "schema": "crowdtensor_qwen15b_training_showcase_check_v1",
        "ok": not errors,
        "showcase_ready": report.get("showcase_ready") is True and not errors,
        "error_count": len(errors),
        "errors": sorted(set(errors)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check_report(_load(args.report), require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "training_qwen15b_showcase_check "
            f"ok={result['ok']} errors={','.join(result['errors']) or 'none'}"
        )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
