#!/usr/bin/env python3
"""Validate the full-offline Elastic Volunteer Training live artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_qwen15b_elastic_live_probe_v1"
REQUIRED_EVIDENCE = (
    "old_generation_live_verified",
    "old_generation_steps_1_to_4_verified",
    "midpoint_step4_committed",
    "all_old_miners_offline",
    "old_kernels_deleted_before_replacement",
    "bounded_no_miner_pause_verified",
    "new_generation_live_verified",
    "new_generation_steps_5_to_8_verified",
    "new_miners_restore_step4_verified",
    "entirely_new_miner_sessions_verified",
    "stage_reassignment_to_new_miners_verified",
    "exactly_once_optimizer_commits_verified",
    "automatic_pause_wake_verified",
    "central_checkpoint_independent_of_old_kernels",
    "final_step8_completed",
    "final_peft_export_evaluation_verified",
    "four_distinct_kernel_refs_verified",
    "real_cuda_only_verified",
    "rendezvous_full_pipeline_verified",
)


def _workers(generation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item.get("worker") or {})
        for item in generation.get("worker_reports") or []
        if isinstance(item, dict)
    ]


def _step_sequences(workers: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {
        str(worker.get("role") or ""): [
            int(item.get("step") or 0)
            for item in (worker.get("runtime") or {}).get("step_reports") or []
        ]
        for worker in workers
    }


def _safety_errors(value: dict[str, Any]) -> list[str]:
    encoded = json.dumps(value, sort_keys=True)
    lowered = encoded.lower()
    errors = []
    forbidden = (
        "kaggle_key",
        "kaggle_api_token",
        "authorization:",
        "bearer ",
        "cookie:",
        '"payload_b64":',
        '"activation_gradient":',
        '"raw_training_text":',
        "/root/",
        "/tmp/",
        "/home/",
        "trycloudflare.com",
        "elastic-miner-",
    )
    errors.extend(f"public_forbidden_fragment:{item}" for item in forbidden if item in lowered)
    token_pattern = re.compile(r"(?:kga|eyj)[a-z0-9._=-]{16,}", re.IGNORECASE)
    if token_pattern.search(encoded):
        errors.append("public_token_like_value")
    for key in (
        "credentials_public",
        "credential_paths_public",
        "coordinator_url_public",
        "session_tokens_public",
        "assignment_tokens_public",
        "private_paths_public",
        "activation_values_public",
        "gradient_values_public",
        "checkpoint_tensor_values_public",
        "adapter_tensor_values_public",
        "token_ids_public",
        "raw_training_text_public",
    ):
        if value.get(key) is not False:
            errors.append(f"public_safety_flag_invalid:{key}")
    if value.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_false")
    return errors


def check_report(value: dict[str, Any], *, require_ready: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("elastic_live_schema_invalid")
    errors.extend(_safety_errors(value))
    if value.get("mock_runtime_used") is not False:
        errors.append("elastic_live_mock_runtime_used")
    if value.get("cpu_fallback_used") is not False:
        errors.append("elastic_live_cpu_fallback_used")
    if value.get("tiny_or_random_model_used") is not False:
        errors.append("elastic_live_wrong_model_used")
    if value.get("requested_model") != "Qwen/Qwen2.5-1.5B":
        errors.append("elastic_live_model_invalid")
    if int(value.get("target_steps") or 0) != 8:
        errors.append("elastic_live_target_steps_invalid")

    old = value.get("old_generation") if isinstance(value.get("old_generation"), dict) else {}
    new = value.get("new_generation") if isinstance(value.get("new_generation"), dict) else {}
    midpoint = value.get("midpoint_status") if isinstance(value.get("midpoint_status"), dict) else {}
    final = value.get("final_status") if isinstance(value.get("final_status"), dict) else {}
    evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
    cleanup = value.get("cleanup") if isinstance(value.get("cleanup"), dict) else {}
    old_workers = _workers(old)
    new_workers = _workers(new)
    old_steps = _step_sequences(old_workers)
    new_steps = _step_sequences(new_workers)
    pause = value.get("full_offline_pause") if isinstance(value.get("full_offline_pause"), dict) else {}
    pause_observations = list(pause.get("observations") or [])
    epoch_by_id = {
        int(item.get("epoch_id", -1)): item for item in final.get("epochs") or []
    }
    committed_step5_assignments = [
        item
        for item in final.get("assignments") or []
        if epoch_by_id.get(int(item.get("epoch_id", -1)), {}).get("target_step") == 5
        and epoch_by_id.get(int(item.get("epoch_id", -1)), {}).get("state") == "committed"
        and item.get("state") == "completed"
    ]
    new_miner_ids = {
        worker.get("elastic_client", {}).get("miner_id_hash") for worker in new_workers
    }
    old_miner_ids = {
        worker.get("elastic_client", {}).get("miner_id_hash") for worker in old_workers
    }
    old_session_hashes = {
        item.get("miner_session_hash")
        for item in final.get("miners") or []
        if item.get("miner_id_hash") in old_miner_ids
    }
    new_session_hashes = {
        item.get("miner_session_hash")
        for item in final.get("miners") or []
        if item.get("miner_id_hash") in new_miner_ids
    }
    step4_archives = {
        int(item.get("stage_id", -1)): str(item.get("archive_hash") or "")
        for item in final.get("events") or []
        if item.get("operation") == "stage_checkpoint_submitted"
        and int(item.get("target_step") or 0) == 4
    }
    restored_archives = {
        int(item.get("stage_id", -1)): str(item.get("archive_hash") or "")
        for worker in new_workers
        for item in worker.get("central_checkpoint_restore") or []
    }
    new_role_b = next(
        (worker for worker in new_workers if worker.get("role") == "kernel_b"), {}
    )
    rendezvous = value.get("rendezvous") if isinstance(value.get("rendezvous"), dict) else {}

    derived = {
        "old_generation_present": len(old_workers) == 2
        and {worker.get("role") for worker in old_workers} == {"kernel_a", "kernel_b"},
        "new_generation_present": len(new_workers) == 2
        and {worker.get("role") for worker in new_workers} == {"kernel_a", "kernel_b"},
        "old_steps_valid": old_steps
        == {"kernel_a": [1, 2, 3, 4], "kernel_b": [1, 2, 3, 4]},
        "new_steps_valid": new_steps
        == {"kernel_a": [5, 6, 7, 8], "kernel_b": [5, 6, 7, 8]},
        "old_worker_barriers_valid": len(old_workers) == 2
        and all(
            len(worker.get("barrier_commits") or []) == 4
            and all(item.get("barrier_committed") is True for item in worker["barrier_commits"])
            for worker in old_workers
        ),
        "new_worker_barriers_valid": len(new_workers) == 2
        and all(
            len(worker.get("barrier_commits") or []) == 4
            and all(item.get("barrier_committed") is True for item in worker["barrier_commits"])
            for worker in new_workers
        ),
        "midpoint_pause_valid": int(midpoint.get("committed_step") or 0) == 4
        and midpoint.get("runtime_state") == "paused_waiting_for_miners"
        and midpoint.get("zero_live_miners") is True
        and int(midpoint.get("live_miner_count") or 0) == 0,
        "pause_window_valid": float(pause.get("observed_seconds") or 0) >= 5.0
        and len(pause_observations) >= 2
        and pause.get("new_kernel_launched_during_pause") is False
        and all(
            item.get("runtime_state") == "paused_waiting_for_miners"
            and int(item.get("committed_step") or 0) == 4
            and int(item.get("live_miner_count", -1)) == 0
            for item in pause_observations
        ),
        "final_commit_ledger_valid": int(final.get("committed_step") or 0) == 8
        and final.get("runtime_state") == "completed"
        and int(final.get("optimizer_commit_count") or 0) == 8
        and final.get("committed_steps") == list(range(1, 9))
        and final.get("committed_steps_contiguous") is True,
        "central_restore_valid": len(new_workers) == 2
        and all(
            worker.get("central_checkpoint_restore_verified") is True
            and len(worker.get("central_checkpoint_restore") or []) == 2
            and {
                int(item.get("global_step") or 0)
                for item in worker.get("central_checkpoint_restore") or []
            }
            == {4}
            and worker.get("fresh_checkpoint_directory_before_restore") is True
            and worker.get("old_kernel_local_checkpoint_dependency") is False
            for worker in new_workers
        ),
        "central_restore_hashes_valid": len(step4_archives) == 4
        and restored_archives == step4_archives,
        "replacement_sessions_valid": len(old_session_hashes) == 2
        and len(new_session_hashes) == 2
        and old_session_hashes.isdisjoint(new_session_hashes),
        "replacement_stage_assignment_valid": len(committed_step5_assignments) == 4
        and {int(item.get("stage_id", -1)) for item in committed_step5_assignments}
        == {0, 1, 2, 3}
        and all(
            item.get("miner_id_hash") in new_miner_ids
            for item in committed_step5_assignments
        ),
        "peft_evaluation_valid": new_role_b.get("export", {}).get(
            "standard_peft_format"
        )
        is True
        and new_role_b.get("evaluation", {}).get("evaluation_verified") is True,
        "real_kaggle_cuda_valid": len(old.get("worker_reports") or []) == 2
        and len(new.get("worker_reports") or []) == 2
        and all(
            item.get("kaggle_kernel") is True
            and item.get("cuda_available") is True
            and int(item.get("cuda_device_count") or 0) >= 2
            and item.get("ok") is True
            for item in [
                *(old.get("worker_reports") or []),
                *(new.get("worker_reports") or []),
            ]
        ),
        "rendezvous_optimizer_events_valid": len(
            [
                item
                for item in rendezvous.get("events") or []
                if item.get("run_kind") == "elastic"
                and item.get("operation") == "optimizer_step"
            ]
        )
        == 32,
        "generation_kernel_identity_valid": len(old.get("kernel_ref_hashes") or []) == 2
        and len(new.get("kernel_ref_hashes") or []) == 2
        and set(old.get("kernel_ref_hashes") or []).isdisjoint(
            set(new.get("kernel_ref_hashes") or [])
        ),
        "generation_cleanup_valid": old.get("all_kernels_deleted") is True
        and new.get("all_kernels_deleted") is True
        and float(old.get("deleted_at_epoch") or 0)
        <= float(new.get("launched_at_epoch") or 0),
        "global_cleanup_valid": cleanup.get("all_four_kernels_deleted") is True
        and cleanup.get("coordinator_stopped") is True
        and cleanup.get("tunnel_stopped") is True
        and cleanup.get("private_runtime_removed") is True
        and cleanup.get("rendezvous_payloads_removed") is True
        and cleanup.get("uncommitted_checkpoint_blobs_removed") is True
        and cleanup.get("live_resources_left_running") is False,
    }
    for name, passed in derived.items():
        if not passed and (require_ready or value.get("elastic_volunteer_training_ready") is True):
            errors.append(f"elastic_live_gate_failed:{name}")
    missing_evidence = [key for key in REQUIRED_EVIDENCE if evidence.get(key) is not True]
    if missing_evidence and (require_ready or value.get("elastic_volunteer_training_ready") is True):
        errors.extend(f"elastic_live_evidence_false:{key}" for key in missing_evidence)
    derived_ready = bool(
        all(derived.values())
        and not missing_evidence
        and evidence.get("verified") is True
        and value.get("ok") is True
        and not value.get("blockers")
        and not _safety_errors(value)
    )
    if value.get("elastic_volunteer_training_ready") is True and not derived_ready:
        errors.append("elastic_live_ready_overclaimed")
    if require_ready and value.get("elastic_volunteer_training_ready") is not True:
        errors.append("elastic_volunteer_training_not_ready")
    if require_ready and not derived_ready:
        errors.append("elastic_live_strict_acceptance_incomplete")
    return {
        "schema": "crowdtensor_qwen15b_elastic_live_check_v1",
        "ok": not errors,
        "require_ready": bool(require_ready),
        "elastic_volunteer_training_ready": bool(derived_ready),
        "derived_gates": derived,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    value = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = check_report(value, require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "training_qwen15b_elastic_check "
            f"ok={result['ok']} ready={result['elastic_volunteer_training_ready']} "
            f"errors={result['error_count']}"
        )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
