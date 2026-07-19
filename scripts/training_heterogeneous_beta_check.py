#!/usr/bin/env python3
"""Independently validate heterogeneous CPU/GPU Training Beta evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_heterogeneous_training_beta_v1"
CHECK_SCHEMA = "crowdtensor_heterogeneous_training_beta_check_v1"
MODEL_ID = "Qwen/Qwen2.5-7B"
MODEL_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
REQUIRED_STAGES = set(range(5))
REQUIRED_STEPS = list(range(1, 7))
REQUIRED_CHECKPOINT_COMPONENTS = {
    "adapter",
    "optimizer",
    "lr_scheduler",
    "grad_scaler",
    "rng",
    "manifest",
}
REQUIRED_GATES = {
    "fixed_revision_manifest_verified",
    "same_job_cpu_cuda_verified",
    "single_gpu_miner_verified",
    "dynamic_resource_placement_verified",
    "six_contiguous_steps_verified",
    "cross_device_forward_backward_verified",
    "stage_replacement_resume_verified",
    "finite_real_lora_update_verified",
    "peft_export_reload_forward_verified",
    "regression_suite_verified",
    "cleanup_verified",
    "public_safety_verified",
}
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("heterogeneous_training_beta_report_invalid")
    return value


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _hash(value: Any) -> bool:
    return bool(HASH_RE.fullmatch(str(value or "")))


def _assignments(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value or [] if isinstance(item, dict)]


def _workers(report: dict[str, Any]) -> list[dict[str, Any]]:
    return _assignments(report.get("worker_evidence"))


def _worker_ids(workers: list[dict[str, Any]], role: str) -> set[str]:
    return {
        str(item.get("miner_id_hash") or "")
        for item in workers
        if str(item.get("role") or "") == role
    }


def _public_safety_errors(report: dict[str, Any]) -> list[str]:
    encoded = json.dumps(report, sort_keys=True, ensure_ascii=True)
    lowered = encoded.lower()
    patterns = {
        "absolute_private_path": r"/(?:root|tmp|home|kaggle)/(?!working(?:\"|/))",
        "bearer_header": r"bearer\s+[a-z0-9._=-]+",
        "authorization_header": r"authorization\s*[:=]",
        "cookie_value": r"(?:set-)?cookie\s*[:=]",
        "kaggle_secret": r"kaggle_(?:key|api_token)\s*[:=]",
        "raw_prompt": r'"(?:prompt|raw_training_text|generated_text)"\s*:',
        "tensor_payload": r'"(?:payload_b64|hidden_b64|tensor_values|token_ids)"\s*:',
        "private_url": r'"coordinator_url"\s*:',
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, lowered)]


def build_acceptance_gates(report: dict[str, Any]) -> dict[str, bool]:
    topology = dict(report.get("kernel_topology") or {})
    placement = dict(report.get("placement_evidence") or {})
    training = dict(report.get("training_evidence") or {})
    transport = dict(report.get("tensor_transport_evidence") or {})
    replacement = dict(report.get("replacement_evidence") or {})
    exported = dict(report.get("export_evidence") or {})
    cleanup = dict(report.get("cleanup") or {})
    regression = dict(report.get("regression_summary") or {})
    workers = _workers(report)
    initial = _assignments(placement.get("initial_assignments"))
    replacement_assignments = _assignments(
        placement.get("replacement_assignments")
    )
    initial_stage_ids = {int(item.get("stage_id", -1)) for item in initial}
    initial_device_types = {str(item.get("device_type") or "") for item in initial}
    old_id = str(replacement.get("old_miner_id_hash") or "")
    new_id = str(replacement.get("replacement_miner_id_hash") or "")
    all_worker_steps = sorted(
        {
            int(step)
            for worker in workers
            for step in worker.get("committed_steps") or []
        }
    )
    fixed_revision = bool(
        report.get("live_run_performed") is True
        and report.get("model_id") == MODEL_ID
        and report.get("model_revision") == MODEL_REVISION
        and int(report.get("parameter_count") or 0) >= 7_000_000_000
        and int(report.get("stage_count") or 0) == 5
        and int(report.get("target_steps") or 0) == 6
        and _hash(report.get("training_manifest_hash"))
    )
    same_job = bool(
        report.get("same_job_training_verified") is True
        and _hash(report.get("job_id_hash"))
        and _hash(report.get("run_id_hash"))
        and initial_stage_ids == REQUIRED_STAGES
        and initial_device_types == {"cpu", "cuda"}
        and int(topology.get("pure_cpu_miner_count") or 0) >= 1
        and int(topology.get("initial_single_gpu_miner_count") or 0) >= 4
        and any(
            item.get("pure_cpu_miner") is True and item.get("gpu_count") == 0
            for item in workers
        )
    )
    single_gpu = bool(
        int(topology.get("physical_gpu_count") or 0) >= 4
        and int(topology.get("gpu_kernel_count") or 0) == 2
        and len(
            [
                item
                for item in workers
                if item.get("single_gpu_miner") is True
                and int(item.get("gpu_count") or 0) == 1
            ]
        )
        >= 5
    )
    placement_verified = bool(
        initial_stage_ids == REQUIRED_STAGES
        and {int(item.get("stage_id", -1)) for item in replacement_assignments}
        == REQUIRED_STAGES
        and all(item.get("resource_fit_verified") is True for item in initial)
        and all(
            item.get("resource_fit_verified") is True
            for item in replacement_assignments
        )
        and placement.get("auditable_scores_present") is True
        and placement.get("memory_reserve_enforced") is True
        and placement.get("performance_and_network_cost_used") is True
        and int(placement.get("replacement_generation") or 0)
        > int(placement.get("initial_generation") or 0)
    )
    six_steps = bool(
        training.get("committed_steps") == REQUIRED_STEPS
        and all_worker_steps == REQUIRED_STEPS
        and training.get("committed_steps_contiguous") is True
        and int(training.get("optimizer_commit_count") or 0) == 6
        and training.get("duplicate_committed_steps") == []
        and training.get("missing_committed_steps") == []
        and training.get("atomic_global_commit_verified") is True
        and set(training.get("checkpoint_components") or [])
        == REQUIRED_CHECKPOINT_COMPONENTS
    )
    cross_device = bool(
        transport.get("format") == "safetensors"
        and transport.get("pickle_deserialization_allowed") is False
        and int(transport.get("forward_activation_count") or 0) >= 24
        and int(transport.get("backward_gradient_count") or 0) >= 24
        and int(transport.get("cuda_to_cpu_activation_count") or 0) >= 6
        and int(transport.get("cpu_to_cuda_gradient_count") or 0) >= 6
        and transport.get("all_checksums_verified") is True
        and transport.get("chunking_verified") is True
        and transport.get("finite_retry_verified") is True
        and transport.get("idempotent_delivery_verified") is True
        and transport.get("stale_generation_rejected") is True
        and transport.get("duplicate_message_deduplicated") is True
    )
    replacement_verified = bool(
        _hash(old_id)
        and _hash(new_id)
        and old_id != new_id
        and replacement.get("removed_after_committed_step") == 3
        and replacement.get("trainable_stage_removed") is True
        and replacement.get("pause_or_incomplete_placement_observed") is True
        and replacement.get("rebalance_verified") is True
        and replacement.get("replacement_checkpoint_restore_verified") is True
        and int(replacement.get("replacement_steps_completed") or 0) >= 3
        and old_id in _worker_ids(workers, "gpu_old")
        and new_id in _worker_ids(workers, "gpu_replacement")
    )
    numerical = bool(
        int(training.get("finite_loss_count") or 0) >= 6
        and int(training.get("non_finite_loss_count") or 0) == 0
        and set(training.get("positive_gradient_stage_ids") or []) == REQUIRED_STAGES
        and set(training.get("changed_lora_stage_ids") or []) == REQUIRED_STAGES
        and all(
            worker.get("positive_lora_gradient_norms") is True
            and worker.get("optimizer_and_scheduler_steps_applied") is True
            for worker in workers
            if int(worker.get("steps_completed") or 0) > 0
        )
    )
    export_verified = bool(
        exported.get("standard_peft_format") is True
        and exported.get("all_five_stages_present") is True
        and exported.get("adapter_reload_verified") is True
        and exported.get("forward_inference_verified") is True
        and exported.get("finite_logits_verified") is True
        and exported.get("model_binding_verified") is True
        and _hash(exported.get("adapter_file_hash"))
    )
    regression_verified = bool(
        int(regression.get("failed") or 0) == 0
        and int(regression.get("passed") or 0) >= 1
        and regression.get("legacy_training_regression_included") is True
        and regression.get("heterogeneous_training_tests_included") is True
    )
    cleanup_verified = bool(
        cleanup.get("all_remote_kernels_deleted") is True
        and cleanup.get("temporary_private_packages_removed") is True
        and cleanup.get("coordinator_stopped") is True
        and cleanup.get("tunnel_stopped") is True
        and cleanup.get("tensor_payloads_removed") is True
        and cleanup.get("temporary_credentials_removed") is True
        and cleanup.get("live_resources_left_running") is False
    )
    safety_verified = bool(
        report.get("public_artifact_safe") is True
        and not _public_safety_errors(report)
        and all(
            report.get(key) is False
            for key in (
                "credential_values_public",
                "credential_paths_public",
                "coordinator_url_public",
                "raw_training_text_public",
                "token_ids_public",
                "activation_values_public",
                "gradient_values_public",
                "checkpoint_tensor_values_public",
                "adapter_tensor_values_public",
                "private_paths_public",
            )
        )
    )
    return {
        "fixed_revision_manifest_verified": fixed_revision,
        "same_job_cpu_cuda_verified": same_job,
        "single_gpu_miner_verified": single_gpu,
        "dynamic_resource_placement_verified": placement_verified,
        "six_contiguous_steps_verified": six_steps,
        "cross_device_forward_backward_verified": cross_device,
        "stage_replacement_resume_verified": replacement_verified,
        "finite_real_lora_update_verified": numerical,
        "peft_export_reload_forward_verified": export_verified,
        "regression_suite_verified": regression_verified,
        "cleanup_verified": cleanup_verified,
        "public_safety_verified": safety_verified,
    }


def check(report_path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    report = _load(report_path)
    errors: list[str] = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    require(report.get("schema") == SCHEMA, "heterogeneous_beta_schema_invalid")
    stored_hash = str(report.get("content_hash") or "")
    unhashed = {key: value for key, value in report.items() if key != "content_hash"}
    require(
        stored_hash == _stable_hash(unhashed),
        "heterogeneous_beta_content_hash_invalid",
    )
    stored_gates = dict(report.get("acceptance_gates") or {})
    require(
        set(stored_gates) == REQUIRED_GATES,
        "heterogeneous_beta_gate_set_invalid",
    )
    derived_gates = build_acceptance_gates(report)
    require(
        stored_gates == derived_gates,
        "heterogeneous_beta_gate_derivation_mismatch",
    )
    safety_errors = _public_safety_errors(report)
    require(not safety_errors, "heterogeneous_beta_public_safety_scan_failed")
    require(
        report.get("blockers") == sorted(set(report.get("blockers") or [])),
        "heterogeneous_beta_blockers_not_canonical",
    )
    derived_ready = bool(
        all(derived_gates.values())
        and report.get("blockers") == []
        and report.get("live_run_performed") is True
    )
    require(
        report.get("heterogeneous_training_beta_ready") is derived_ready,
        "heterogeneous_beta_ready_claim_invalid",
    )
    if require_ready:
        for name, value in derived_gates.items():
            require(value is True, f"heterogeneous_beta_gate_failed:{name}")
        require(not report.get("blockers"), "heterogeneous_beta_blockers_present")
        require(derived_ready, "heterogeneous_training_beta_not_ready")
    ready = bool(not errors and derived_ready)
    return {
        "schema": CHECK_SCHEMA,
        "ok": not errors,
        "heterogeneous_training_beta_ready": ready,
        "error_count": len(errors),
        "errors": errors,
        "public_safety_errors": safety_errors,
        "report_content_hash": stored_hash,
        "public_artifact_safe": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report, require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "training_heterogeneous_beta_check "
            f"ok={result['ok']} ready={result['heterogeneous_training_beta_ready']} "
            f"errors={result['error_count']}"
        )
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
