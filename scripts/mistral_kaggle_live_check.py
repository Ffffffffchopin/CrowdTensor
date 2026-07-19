#!/usr/bin/env python3
"""Strict checker for the Mistral CPU+CUDA heterogeneous live gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash


SCHEMA = "crowdtensor_mistral_kaggle_heterogeneous_live_v1"
MODEL_ID = "Locutusque/TinyMistral-248M-v2"
MODEL_REVISION = "0f57b17cb317bb322c7c1466b669c681f80c058f"
REQUIRED_STEPS = list(range(1, 9))
HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _hash(value: Any) -> bool:
    return bool(HASH_RE.fullmatch(str(value or "")))


def _workers(kernels: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    for kernel in kernels:
        if kernel.get("kernel_role") == role:
            return [
                dict(item)
                for item in kernel.get("worker_reports") or []
                if isinstance(item, dict)
            ]
    return []


def check_report(report: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        return {
            "ok": False,
            "schema": "crowdtensor_mistral_kaggle_live_check_v1",
            "errors": ["mistral_live_schema_invalid"],
            "error_count": 1,
            "mistral_live_verified": False,
        }
    expected_hash = stable_hash(
        {key: item for key, item in report.items() if key != "content_hash"}
    )
    if report.get("content_hash") != expected_hash:
        errors.append("mistral_live_content_hash_invalid")
    if report.get("ok") is not True or report.get("live_run_performed") is not True:
        errors.append("mistral_live_execution_missing")
    if report.get("node_scope") != "Kaggle logical multi-node":
        errors.append("mistral_live_node_scope_invalid")
    if report.get("accepted_providers") != ["kaggle_cpu", "kaggle_cuda"]:
        errors.append("mistral_live_provider_coverage_invalid")
    model = dict(report.get("model") or {})
    if not all(
        (
            model.get("adapter_id") == "mistral_lora_v1",
            model.get("family") == "mistral",
            model.get("architecture") == "MistralForCausalLM",
            model.get("model_id") == MODEL_ID,
            model.get("model_revision") == MODEL_REVISION,
            model.get("license") == "apache-2.0",
            int(model.get("parameter_count") or 0) == 248_024_064,
            model.get("real_trained_weights") is True,
            model.get("random_or_synthetic_weights_used") is False,
        )
    ):
        errors.append("mistral_live_model_identity_invalid")
    plugin = dict(report.get("plugin_installation") or {})
    if not all(
        plugin.get(field) is True
        for field in (
            "core_wheel_hash_verified",
            "adapter_wheel_hash_verified",
            "both_wheels_installed_in_fresh_environment",
            "entry_point_plugin_discovered",
            "workspace_import_not_used",
        )
    ) or plugin.get("registration_kind") != "entry_point_plugin":
        errors.append("mistral_live_plugin_installation_invalid")
    final = dict(report.get("final_status") or {})
    if not all(
        (
            final.get("model_adapter_id") == "mistral_lora_v1",
            final.get("model_id") == MODEL_ID,
            int(final.get("target_steps") or 0) == 8,
            final.get("committed_step_ids") == REQUIRED_STEPS,
            final.get("strictly_contiguous_steps") is True,
            final.get("finite_losses") is True,
            int(final.get("ledger_entry_count") or 0) == 8,
            final.get("completed") is True,
        )
    ):
        errors.append("mistral_live_atomic_training_invalid")
    kernels = [
        dict(item)
        for item in report.get("kernel_evidence") or []
        if isinstance(item, dict)
    ]
    roles = {(item.get("kernel_role"), item.get("backend")) for item in kernels}
    if len(kernels) != 2 or roles != {("stage0", "cuda"), ("stage1", "cpu")}:
        errors.append("mistral_live_kernel_topology_invalid")
    if any(
        item.get("ok") is not True
        or item.get("node_scope") != "Kaggle logical multi-node"
        or item.get("both_wheels_installed_in_fresh_environment") is not True
        or item.get("adapter_plugin_discovered") is not True
        for item in kernels
    ):
        errors.append("mistral_live_kernel_evidence_invalid")
    cuda_kernels = [item for item in kernels if item.get("backend") == "cuda"]
    if len(cuda_kernels) != 1 or int(cuda_kernels[0].get("cuda_device_count") or 0) < 2:
        errors.append("mistral_live_t4x2_kernel_invalid")
    stage0 = _workers(kernels, "stage0")
    stage1 = _workers(kernels, "stage1")
    if len(stage0) != 2 or len(stage1) != 1:
        errors.append("mistral_live_worker_count_invalid")
    all_workers = stage0 + stage1
    if any(
        item.get("ok") is not True
        or item.get("model_adapter_id") != "mistral_lora_v1"
        or item.get("model_id") != MODEL_ID
        or item.get("model_revision") != MODEL_REVISION
        or item.get("real_model_weights_loaded") is not True
        or item.get("adapter_updated") is not True
        or (item.get("stage_runtime") or {}).get("family") != "mistral"
        or (item.get("stage_runtime") or {}).get("architecture")
        != "MistralForCausalLM"
        for item in all_workers
    ):
        errors.append("mistral_live_worker_training_invalid")
    old_steps = [
        int(item.get("step") or 0)
        for item in (stage0[0].get("step_events") or [])
    ] if stage0 else []
    replacement_steps = [
        int(item.get("step") or 0)
        for item in (stage0[1].get("step_events") or [])
    ] if len(stage0) > 1 else []
    if not all(
        (
            bool(old_steps),
            max(old_steps, default=0) == 4,
            int(stage0[0].get("last_committed_step") or 0) == 4 if stage0 else False,
            bool(replacement_steps),
            min(replacement_steps, default=0) == 5,
            max(replacement_steps, default=0) == 8,
        )
    ):
        errors.append("mistral_live_replacement_step_boundary_invalid")
    replacement = dict(report.get("gpu_worker_replacement") or {})
    if not all(
        (
            replacement.get("verified") is True,
            replacement.get("after_step") == 4,
            _hash(replacement.get("old_worker_id_hash")),
            _hash(replacement.get("new_worker_id_hash")),
            replacement.get("old_worker_id_hash")
            != replacement.get("new_worker_id_hash"),
            replacement.get("checkpoint_restored") is True,
            replacement.get("restored_checkpoint_step") == 4,
            replacement.get("optimizer_state_restored") is True,
        )
    ):
        errors.append("mistral_live_gpu_worker_replacement_invalid")
    checkpoints = dict(report.get("checkpoints") or {})
    if checkpoints.get("steps_by_role") != {
        "stage0": [4, 8],
        "stage1": [4, 8],
    } or not all(
        checkpoints.get(field) is True
        for field in (
            "adapter_state_saved",
            "adam_state_saved",
            "hash_integrity_verified",
            "final_stage_checkpoints_present",
        )
    ):
        errors.append("mistral_live_checkpoint_contract_invalid")
    transfer = dict(report.get("cross_device_transfer") or {})
    if not all(
        (
            transfer.get("activation_gradient_transfer_verified") is True,
            int(transfer.get("forward_activation_count") or 0) == 8,
            int(transfer.get("backward_gradient_count") or 0) == 8,
            transfer.get("safetensors_serialization") is True,
            transfer.get("all_payload_hashes_verified") is True,
            transfer.get("payload_values_public") is False,
        )
    ):
        errors.append("mistral_live_cross_device_transfer_invalid")
    exported = dict(report.get("export") or {})
    reload = dict(report.get("reload") or {})
    if not all(
        (
            exported.get("adapter_id") == "mistral_lora_v1",
            exported.get("standard_peft_format") is True,
            exported.get("stage_adapter_key_overlap") is False,
            int(exported.get("adapter_tensor_count") or 0) > 0,
            _hash(exported.get("adapter_file_hash")),
            reload.get("adapter_id") == "mistral_lora_v1",
            reload.get("independent_process_reload") is True,
            reload.get("adapter_reload_verified") is True,
            reload.get("reload_logits_finite") is True,
        )
    ):
        errors.append("mistral_live_export_reload_invalid")
    cleanup = dict(report.get("cleanup") or {})
    if not all(
        (
            cleanup.get("all_remote_kernels_deleted") is True,
            cleanup.get("coordinator_stopped") is True,
            cleanup.get("tunnel_stopped") is True,
            cleanup.get("private_runtime_removed") is True,
            cleanup.get("live_resources_left_running") is False,
            report.get("cleanup_verified") is True,
        )
    ):
        errors.append("mistral_live_cleanup_invalid")
    ledger = dict(report.get("attempt_ledger") or {})
    if not all(
        (
            ledger.get("schema") == "crowdtensor_mistral_kaggle_live_gate_ledger_v1",
            int(ledger.get("attempt") or 0) in {1, 2},
            int(ledger.get("maximum_attempts") or 0) == 2,
            ledger.get("community_maturity_ledger_modified") is False,
        )
    ):
        errors.append("mistral_live_attempt_ledger_invalid")
    unsupported = dict(report.get("unsupported_claims") or {})
    if any(value is not False for value in unsupported.values()) or set(unsupported) != {
        "arbitrary_mistral_models_supported",
        "full_parameter_training_verified",
        "mistral_7b_live_verified",
        "physical_multi_machine_verified",
        "production_sla_verified",
    }:
        errors.append("mistral_live_unsupported_claim_invalid")
    safety = scan_public_value(
        {key: item for key, item in report.items() if key != "public_safety"}
    )
    if safety["ok"] is not True or report.get("public_artifact_safe") is not True:
        errors.append("mistral_live_public_safety_invalid")
    result = {
        "schema": "crowdtensor_mistral_kaggle_live_check_v1",
        "ok": not errors,
        "errors": sorted(set(errors)),
        "error_count": len(set(errors)),
        "mistral_live_verified": not errors,
        "generated_or_training_tensor_values_public": False,
        "public_artifact_safe": True,
    }
    result["content_hash"] = stable_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    result = check_report(report)
    print(json.dumps(result, sort_keys=True) if args.json else f"ok={result['ok']} errors={result['error_count']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
