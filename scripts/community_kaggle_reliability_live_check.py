#!/usr/bin/env python3
"""Strict checker for the bounded Kaggle CPU+GPU Community reliability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from scripts.community_kaggle_reliability_live_probe import SCHEMA
from scripts.community_live_gate_ledger_amend import (
    AMENDED_MAXIMUM,
    AMENDMENT_FIELDS,
    AMENDMENT_SCOPE,
    ORIGINAL_MAXIMUM,
)


def check(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("community_reliability_live_schema_invalid")
    for field in (
        "ok",
        "live_run_performed",
        "real_open_model_weights",
        "worker_replacement_verified",
        "coordinator_restart_verified",
        "checkpoint_recovery_verified",
        "ledger_exactly_once_verified",
        "finite_update_verified",
        "adapter_reload_verified",
        "monitoring_verified",
        "cleanup_verified",
        "public_artifact_safe",
    ):
        if value.get(field) is not True:
            errors.append("community_reliability_live_" + field + "_missing")
    if value.get("node_scope") != "Kaggle logical multi-node":
        errors.append("community_reliability_live_node_scope_invalid")
    if value.get("physical_multi_machine_verified") is not False:
        errors.append("community_reliability_live_physical_multi_machine_overclaim")
    if value.get("random_or_synthetic_weights_used") is not False:
        errors.append("community_reliability_live_synthetic_weights_invalid")
    if (
        int(value.get("live_gate_attempt") or 0) != AMENDED_MAXIMUM
        or int(value.get("maximum_full_live_gates") or 0) != AMENDED_MAXIMUM
    ):
        errors.append("community_reliability_live_attempt_bound_invalid")
    authorization = value.get("live_gate_amendment") if isinstance(
        value.get("live_gate_amendment"), dict
    ) else {}
    if (
        set(authorization) != AMENDMENT_FIELDS | {"verified"}
        or authorization.get("verified") is not True
        or int(authorization.get("old_maximum_full_live_gates") or 0)
        != ORIGINAL_MAXIMUM
        or int(authorization.get("new_maximum_full_live_gates") or 0)
        != AMENDED_MAXIMUM
        or authorization.get("scope") != AMENDMENT_SCOPE
        or not str(authorization.get("approval_statement_hash") or "").startswith(
            "sha256:"
        )
    ):
        errors.append("community_reliability_live_authorization_invalid")
    duration = float(value.get("duration_seconds") or 0.0)
    if duration <= 0 or duration > 2700 or int(value.get("maximum_gate_seconds") or 0) != 2700:
        errors.append("community_reliability_live_duration_bound_invalid")
    steps = [int(item) for item in value.get("committed_step_ids") or []]
    if steps != list(range(1, 101)):
        errors.append("community_reliability_live_100_contiguous_steps_missing")
    if sorted(value.get("providers") or []) != ["kaggle_cpu", "kaggle_cuda"]:
        errors.append("community_reliability_live_provider_coverage_invalid")
    if int(value.get("logical_kernel_count") or 0) != 2 or int(value.get("logical_miner_count") or 0) < 3:
        errors.append("community_reliability_live_logical_node_coverage_invalid")
    install = value.get("clean_install") if isinstance(value.get("clean_install"), dict) else {}
    if (
        install.get("verified") is not True
        or install.get("fresh_install_root_per_kernel") is not True
        or install.get("fresh_install_kind") != "pip_target"
        or install.get("workspace_import_used") is not False
    ):
        errors.append("community_reliability_live_clean_install_invalid")
    replacement = value.get("worker_replacement") if isinstance(value.get("worker_replacement"), dict) else {}
    if (
        replacement.get("verified") is not True
        or int(replacement.get("replacement_after_step") or 0) != 30
        or int(replacement.get("restored_checkpoint_step") or 0) < 30
        or replacement.get("optimizer_state_restored") is not True
        or replacement.get("old_worker_id_hash") == replacement.get("replacement_worker_id_hash")
    ):
        errors.append("community_reliability_live_replacement_invalid")
    restart = value.get("coordinator_restart") if isinstance(value.get("coordinator_restart"), dict) else {}
    if (
        restart.get("verified") is not True
        or restart.get("restart_barrier_verified") is not True
        or int(restart.get("restart_at_committed_step") or 0) < 50
        or int(restart.get("generation_after") or 0) != int(restart.get("generation_before") or 0) + 1
        or restart.get("same_committed_step_after_restart") is not True
    ):
        errors.append("community_reliability_live_coordinator_restart_invalid")
    kernels = value.get("kernel_evidence") if isinstance(value.get("kernel_evidence"), list) else []
    if (
        {item.get("kernel_role") for item in kernels} != {"stage0", "stage1"}
        or {item.get("backend") for item in kernels} != {"cpu", "cuda"}
        or any(
            item.get("ok") is not True
            or item.get("wheel_clean_install") is not True
            or item.get("model_stack_import_verified") is not True
            for item in kernels
        )
    ):
        errors.append("community_reliability_live_kernel_evidence_invalid")
    second_model = value.get("second_model_live") if isinstance(value.get("second_model_live"), dict) else {}
    if (
        second_model.get("verified") is not True
        or int(second_model.get("logical_stage_count") or 0) != 2
        or second_model.get("devices") != ["cuda", "cuda"]
        or second_model.get("adapter_reload_verified") is not True
    ):
        errors.append("community_reliability_live_second_model_dual_gpu_invalid")
    export = value.get("export") if isinstance(value.get("export"), dict) else {}
    reload = value.get("reload") if isinstance(value.get("reload"), dict) else {}
    if export.get("standard_peft_format") is not True or int(export.get("adapter_tensor_count") or 0) <= 0:
        errors.append("community_reliability_live_export_invalid")
    if reload.get("adapter_reload_verified") is not True or reload.get("independent_process_reload") is not True:
        errors.append("community_reliability_live_reload_invalid")
    benchmark = value.get("benchmark") if isinstance(value.get("benchmark"), dict) else {}
    if (
        float(benchmark.get("steps_per_second") or 0.0) <= 0
        or float(benchmark.get("p50_step_seconds") or 0.0) <= 0
        or float(benchmark.get("p95_step_seconds") or 0.0) <= 0
        or int(benchmark.get("checkpoint_count") or 0) != 2
        or int(benchmark.get("checkpoint_write_count") or 0) < 6
        or int(benchmark.get("checkpoint_bytes") or 0) <= 0
        or int(benchmark.get("forward_payload_count") or 0) != 100
        or int(benchmark.get("forward_payload_bytes") or 0) <= 0
        or int(benchmark.get("backward_payload_count") or 0) != 100
        or int(benchmark.get("backward_payload_bytes") or 0) <= 0
        or benchmark.get("transfer_payloads_private") is not True
        or benchmark.get("resource_scope")
        != "one Kaggle GPU Kernel plus one Kaggle CPU Kernel"
    ):
        errors.append("community_reliability_live_benchmark_invalid")
    tpu = value.get("tpu") if isinstance(value.get("tpu"), dict) else {}
    if tpu.get("required") is not False or int(tpu.get("acquisition_windows_used") or 0) > 2:
        errors.append("community_reliability_live_tpu_optional_bound_invalid")
    cleanup = value.get("cleanup") if isinstance(value.get("cleanup"), dict) else {}
    if (
        cleanup.get("all_remote_kernels_deleted") is not True
        or cleanup.get("coordinator_stopped") is not True
        or cleanup.get("tunnel_stopped") is not True
        or cleanup.get("private_runtime_removed") is not True
        or cleanup.get("live_resources_left_running") is not False
    ):
        errors.append("community_reliability_live_cleanup_invalid")
    acceptance = value.get("acceptance") if isinstance(value.get("acceptance"), dict) else {}
    if acceptance.get("ok") is not True:
        errors.append("community_reliability_live_acceptance_invalid")
    privacy = scan_public_value(value)
    if privacy["ok"] is not True:
        errors.append("community_reliability_live_public_safety_invalid")
    return {
        "schema": "crowdtensor_community_kaggle_short_reliability_live_check_v1",
        "ok": not errors,
        "errors": sorted(set(errors)),
        "step_count": len(steps),
        "duration_seconds": duration,
        "public_safety": privacy,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report)
    print(json.dumps(result, sort_keys=True) if args.json else f"ok={result['ok']} errors={len(result['errors'])}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
