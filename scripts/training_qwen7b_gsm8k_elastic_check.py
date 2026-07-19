#!/usr/bin/env python3
"""Validate real two-generation Qwen2.5-7B GSM8K training evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
)
from scripts.training_cuda_kaggle_common import public_safety_errors
from scripts.training_qwen7b_gsm8k_elastic_live_probe import SCHEMA


def check(path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except BaseException as exc:
        return {
            "schema": "crowdtensor_qwen7b_gsm8k_elastic_check_v1",
            "ok": False,
            "training_ready": False,
            "errors": ["training_report_load_failed:" + type(exc).__name__],
            "error_count": 1,
        }
    if report.get("schema") != SCHEMA:
        errors.append("training_schema_mismatch")
    if report.get("content_hash") != stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    ):
        errors.append("training_content_hash_mismatch")
    if (
        report.get("model_id") != MODEL_ID
        or report.get("model_revision") != MODEL_REVISION
        or int(report.get("parameter_count") or 0) != MODEL_PARAMETER_COUNT
        or report.get("dataset_id") != DATASET_ID
        or report.get("dataset_revision") != DATASET_REVISION
    ):
        errors.append("training_source_identity_invalid")
    budget = dict(report.get("training_budget") or {})
    if (
        int(report.get("target_steps") or 0) < 256
        or int(report.get("replacement_step") or 0) != 128
        or int(report.get("microbatches_per_step") or 0) != 4
        or int(budget.get("training_non_padding_token_count") or 0) < 262_144
        or int(budget.get("training_supervised_token_count") or 0) < 1
    ):
        errors.append("training_budget_invalid")
    roles = {"kernel_a": [0, 1], "kernel_b": [2, 3]}
    for name in ("old_generation", "new_generation"):
        generation = dict(report.get(name) or {})
        workers = list(generation.get("worker_reports") or [])
        if (
            generation.get("ok") is not True
            or int(generation.get("maximum_running_kernel_count") or 0) != 2
            or generation.get("all_kernels_deleted") is not True
            or len(workers) != 2
            or len(generation.get("checkpoint_bundles") or []) != 2
            or not all(
                item.get("verified") is True
                for item in generation.get("checkpoint_bundles") or []
            )
        ):
            errors.append("training_generation_invalid:" + name)
            continue
        seen_roles = set()
        for artifact in workers:
            worker = dict(artifact.get("worker") or {})
            role = str(worker.get("role") or "")
            seen_roles.add(role)
            shards = list(worker.get("stage_shards") or [])
            if (
                artifact.get("kaggle_kernel") is not True
                or artifact.get("cuda_available") is not True
                or int(artifact.get("cuda_device_count") or 0) < 2
                or worker.get("ok") is not True
                or worker.get("model_id") != MODEL_ID
                or worker.get("model_revision") != MODEL_REVISION
                or int(worker.get("parameter_count") or 0) != MODEL_PARAMETER_COUNT
                or worker.get("stage_ids") != roles.get(role)
                or len(shards) != 2
                or not all(
                    shard.get("multi_file_source") is True
                    and shard.get("stage_selective_loading") is True
                    and shard.get("full_model_file_downloaded") is False
                    for shard in shards
                )
                or worker.get("base_weights_frozen") is not True
                or worker.get("positive_lora_gradient_norms") is not True
                or worker.get("all_segment_barriers_committed") is not True
            ):
                errors.append("training_worker_invalid:" + name + ":" + role)
        if seen_roles != set(roles):
            errors.append("training_worker_roles_invalid:" + name)
    evidence = dict(report.get("evidence") or {})
    required_evidence = {
        "old_generation_live_verified",
        "all_old_miners_offline",
        "old_kernels_deleted_before_replacement",
        "bounded_no_miner_pause_verified",
        "new_generation_live_verified",
        "new_miners_restore_step4_verified",
        "entirely_new_miner_sessions_verified",
        "stage_reassignment_to_new_miners_verified",
        "exactly_once_optimizer_commits_verified",
        "automatic_pause_wake_verified",
        "central_checkpoint_independent_of_old_kernels",
        "final_target_step_completed",
        "four_distinct_kernel_refs_verified",
        "real_cuda_only_verified",
        "rendezvous_full_pipeline_verified",
        "verified",
    }
    if any(evidence.get(key) is not True for key in required_evidence):
        errors.append("training_elastic_evidence_incomplete")
    adapter = dict(report.get("adapter") or {})
    if (
        adapter.get("verified") is not True
        or adapter.get("standard_peft_layout") is not True
        or adapter.get("base_model_verified") is not True
        or adapter.get("model_revision_verified") is not True
        or not str(adapter.get("archive_hash") or "").startswith("sha256:")
        or adapter.get("isolated_benchmark_required") is not True
    ):
        errors.append("training_adapter_export_invalid")
    cleanup = dict(report.get("cleanup") or {})
    if (
        cleanup.get("live_resources_left_running") is not False
        or any(
            cleanup.get(key) is not True
            for key in (
                "all_four_kernels_deleted",
                "coordinator_stopped",
                "tunnel_stopped",
                "private_runtime_removed",
                "rendezvous_payloads_removed",
                "uncommitted_checkpoint_blobs_removed",
            )
        )
    ):
        errors.append("training_cleanup_invalid")
    if any(
        report.get(key) is not False
        for key in (
            "mock_runtime_used",
            "cpu_fallback_used",
            "tiny_or_random_model_used",
            "physical_multi_host_verified",
            "full_parameter_training_claimed",
            "raw_training_text_public",
            "token_ids_public",
            "credentials_public",
            "credential_paths_public",
            "private_paths_public",
        )
    ):
        errors.append("training_boundary_or_safety_flags_invalid")
    safety = public_safety_errors(report)
    if safety:
        errors.append("training_public_safety_scan_failed")
    ready_claim = bool(
        report.get("ok") is True
        and report.get("training_ready") is True
        and report.get("live_run_performed") is True
        and not report.get("blockers")
    )
    if require_ready and not ready_claim:
        errors.append("training_readiness_required")
    if ready_claim and errors:
        errors.append("training_false_ready_claim")
    return {
        "schema": "crowdtensor_qwen7b_gsm8k_elastic_check_v1",
        "ok": not errors,
        "training_ready": ready_claim and not errors,
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
