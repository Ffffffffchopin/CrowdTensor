#!/usr/bin/env python3
"""Independently validate CPU/CUDA/Kaggle JAX-TPU Training Beta evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_heterogeneous_training_tpu_beta_v1"
LIVE_SCHEMA = "crowdtensor_heterogeneous_training_tpu_beta_live_probe_v1"
CHECK_SCHEMA = "crowdtensor_heterogeneous_training_tpu_beta_check_v1"
MODEL_ID = "Qwen/Qwen2.5-7B"
MODEL_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
MANIFEST_SCHEMA = "crowdtensor_heterogeneous_training_manifest_v2"
REQUIRED_STEPS = [1, 2, 3, 4, 5, 6]
REQUIRED_STAGES = {0, 1, 2, 3, 4}
REQUIRED_DEVICE_ORDER = ["cuda", "cuda", "jax_tpu", "cuda", "cpu"]
REQUIRED_PROVIDERS = ["kaggle_cpu", "kaggle_cuda", "kaggle_jax_tpu"]
REQUIRED_GATES = {
    "fixed_tpu_manifest_verified",
    "same_job_three_accelerators_verified",
    "real_v5e8_jax_training_verified",
    "tpu_aware_dynamic_placement_verified",
    "six_atomic_steps_verified",
    "cross_framework_transport_verified",
    "finite_real_lora_updates_verified",
    "tpu_restart_checkpoint_restore_verified",
    "backend_complete_checkpoints_verified",
    "peft_export_cpu_reload_verified",
    "regression_suite_verified",
    "cleanup_verified",
    "public_safety_verified",
}
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("heterogeneous_training_tpu_beta_report_invalid")
    return value


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _hash(value: Any) -> bool:
    return bool(HASH_RE.fullmatch(str(value or "")))


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value or [] if isinstance(item, dict)]


def public_safety_errors(report: dict[str, Any]) -> list[str]:
    encoded = json.dumps(report, sort_keys=True, ensure_ascii=True)
    lowered = encoded.lower()
    patterns = {
        "absolute_private_path": r"/(?:root|tmp|home|kaggle)/(?!working(?:\"|/))",
        "bearer_header": r"bearer\s+[a-z0-9._=-]+",
        "authorization_header": r"authorization\s*[:=]",
        "cookie_value": r"(?:set-)?cookie\s*[:=]",
        "kaggle_secret": r"kaggle_(?:key|api_token)\s*[:=]",
        "raw_training_data": r'"(?:prompt|raw_training_text|generated_text)"\s*:',
        "tensor_payload": r'"(?:payload_b64|hidden_b64|tensor_values|token_ids)"\s*:',
        "private_url": r'"coordinator_url"\s*:',
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, lowered)]


def build_acceptance_gates(report: dict[str, Any]) -> dict[str, bool]:
    placement = dict(report.get("placement_evidence") or {})
    training = dict(report.get("training_evidence") or {})
    tpu = dict(report.get("tpu_training_evidence") or {})
    transport = dict(report.get("tensor_transport_evidence") or {})
    recovery = dict(report.get("tpu_recovery_evidence") or {})
    checkpoints = dict(report.get("checkpoint_evidence") or {})
    exported = dict(report.get("export_evidence") or {})
    regression = dict(report.get("regression_summary") or {})
    cleanup = dict(report.get("cleanup") or {})
    initial = _rows(placement.get("initial_assignments"))
    replacement = _rows(placement.get("replacement_assignments"))
    tpu_steps = [int(item) for item in tpu.get("committed_steps") or []]

    fixed_manifest = bool(
        report.get("live_run_performed") is True
        and report.get("model_id") == MODEL_ID
        and report.get("model_revision") == MODEL_REVISION
        and report.get("training_manifest_schema") == MANIFEST_SCHEMA
        and _hash(report.get("training_manifest_hash"))
        and int(report.get("parameter_count") or 0) >= 7_000_000_000
        and report.get("stage_boundaries")
        == [[0, 7], [7, 14], [14, 20], [20, 26], [26, 28]]
        and int(report.get("sequence_length") or 0) == 8
        and int(report.get("microbatch_size") or 0) == 1
        and int(report.get("target_steps") or 0) == 6
    )
    same_job = bool(
        report.get("same_job_training_verified") is True
        and _hash(report.get("job_id_hash"))
        and _hash(report.get("run_id_hash"))
        and report.get("provider_coverage") == REQUIRED_PROVIDERS
        and [int(item.get("stage_id", -1)) for item in initial] == list(range(5))
        and [str(item.get("device_type") or "") for item in initial]
        == REQUIRED_DEVICE_ORDER
        and len({str(item.get("job_id_hash") or "") for item in initial}) == 1
        and len({str(item.get("run_id_hash") or "") for item in initial}) == 1
    )
    real_tpu = bool(
        tpu.get("execution_provider") == "kaggle"
        and tpu.get("runtime_backend") == "jax_tpu"
        and tpu.get("accelerator_type") == "TPU v5e"
        and int(tpu.get("stage_id") or -1) == 2
        and [int(tpu.get("layer_start") or -1), int(tpu.get("layer_end") or -1)]
        == [14, 20]
        and int(tpu.get("jax_tpu_device_count") or 0) == 8
        and tpu.get("jax_mesh_shape") == [8]
        and tpu.get("all_mesh_devices_used") is True
        and tpu.get("parameter_sharding") == "named_mesh_model_axis"
        and tpu.get("stage_selective_real_weights") is True
        and tpu.get("full_model_loaded") is False
        and tpu.get("compute_dtype") == "bfloat16"
        and tpu.get("forward_executed") is True
        and tpu.get("backward_executed") is True
        and tpu.get("optimizer_executed") is True
        and tpu_steps == REQUIRED_STEPS
        and float(tpu.get("positive_lora_gradient_min") or 0.0) > 0.0
        and _hash(tpu.get("adapter_hash_before"))
        and _hash(tpu.get("adapter_hash_after"))
        and tpu.get("adapter_hash_before") != tpu.get("adapter_hash_after")
        and float(tpu.get("compile_latency_ms") or 0.0) > 0.0
        and int(tpu.get("steady_profile_sample_count") or 0) >= 1
    )
    dynamic_placement = bool(
        {int(item.get("stage_id", -1)) for item in initial} == REQUIRED_STAGES
        and {int(item.get("stage_id", -1)) for item in replacement}
        == REQUIRED_STAGES
        and all(item.get("resource_fit_verified") is True for item in initial)
        and all(item.get("resource_fit_verified") is True for item in replacement)
        and placement.get("hbm_reserve_enforced") is True
        and placement.get("tpu_compile_cost_considered") is True
        and placement.get("tpu_steady_state_cost_considered") is True
        and placement.get("network_and_load_cost_considered") is True
        and int(placement.get("replacement_generation") or 0)
        > int(placement.get("initial_generation") or 0)
    )
    six_steps = bool(
        training.get("committed_steps") == REQUIRED_STEPS
        and training.get("committed_steps_contiguous") is True
        and training.get("duplicate_committed_steps") == []
        and training.get("missing_committed_steps") == []
        and int(training.get("optimizer_commit_count") or 0) == 6
        and training.get("atomic_global_commit_verified") is True
        and set(training.get("updated_stage_ids") or []) == REQUIRED_STAGES
    )
    cross_transport = bool(
        transport.get("format") == "safetensors"
        and transport.get("pickle_deserialization_allowed") is False
        and transport.get("jax_array_conversion_verified") is True
        and int(transport.get("forward_activation_count") or 0) >= 24
        and int(transport.get("backward_gradient_count") or 0) >= 24
        and int(transport.get("cuda_to_tpu_activation_count") or 0) >= 6
        and int(transport.get("tpu_to_cuda_activation_count") or 0) >= 6
        and int(transport.get("cuda_to_tpu_gradient_count") or 0) >= 6
        and int(transport.get("tpu_to_cuda_gradient_count") or 0) >= 6
        and int(transport.get("cuda_to_cpu_activation_count") or 0) >= 6
        and int(transport.get("cpu_to_cuda_gradient_count") or 0) >= 6
        and transport.get("all_checksums_verified") is True
        and transport.get("chunking_verified") is True
        and transport.get("finite_retry_verified") is True
        and transport.get("idempotent_delivery_verified") is True
        and transport.get("stale_generation_rejected") is True
    )
    numerical = bool(
        int(training.get("finite_loss_count") or 0) >= 6
        and int(training.get("non_finite_loss_count") or 0) == 0
        and set(training.get("positive_gradient_stage_ids") or []) == REQUIRED_STAGES
        and set(training.get("changed_lora_stage_ids") or []) == REQUIRED_STAGES
        and training.get("all_optimizer_steps_real") is True
        and training.get("random_or_synthetic_weights_used") is False
        and training.get("fake_gradients_used") is False
    )
    recovery_verified = bool(
        recovery.get("tpu_removed_after_committed_step") == 3
        and recovery.get("same_tpu_kernel_runtime_retained") is True
        and _hash(recovery.get("old_tpu_miner_id_hash"))
        and _hash(recovery.get("replacement_tpu_miner_id_hash"))
        and recovery.get("old_tpu_miner_id_hash")
        != recovery.get("replacement_tpu_miner_id_hash")
        and recovery.get("pause_or_incomplete_placement_observed") is True
        and recovery.get("step3_tpu_checkpoint_restored") is True
        and recovery.get("restored_global_step") == 3
        and recovery.get("replacement_committed_steps") == [4, 5, 6]
        and recovery.get("old_generation_result_rejected") is True
        and recovery.get("rebalance_verified") is True
    )
    checkpoint_verified = bool(
        checkpoints.get("all_five_stage_archives_valid") is True
        and checkpoints.get("atomic_checkpoint_barrier_verified") is True
        and set(checkpoints.get("stage_ids") or []) == REQUIRED_STAGES
        and checkpoints.get("pytorch_components_complete") is True
        and checkpoints.get("tpu_runtime_backend") == "jax_tpu"
        and checkpoints.get("tpu_optimizer_state_present") is True
        and checkpoints.get("tpu_scheduler_state_present") is True
        and checkpoints.get("tpu_jax_prng_state_present") is True
        and checkpoints.get("tpu_grad_scaler_applicable") is False
        and checkpoints.get("tpu_pickle_deserialization_allowed") is False
        and checkpoints.get("all_component_hashes_verified") is True
    )
    export_verified = bool(
        exported.get("standard_peft_format") is True
        and int(exported.get("adapter_tensor_count") or 0) == 392
        and exported.get("layer_indexes") == list(range(28))
        and exported.get("cpu_reload_verified") is True
        and exported.get("finite_full_stagewise_forward_verified") is True
        and exported.get("model_binding_verified") is True
        and _hash(exported.get("adapter_file_hash"))
    )
    regression_verified = bool(
        int(regression.get("failed") or 0) == 0
        and int(regression.get("passed") or 0) >= 145
        and regression.get("legacy_cpu_cuda_tests_included") is True
        and regression.get("jax_tpu_tests_included") is True
        and regression.get("public_safety_tests_included") is True
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
        and not public_safety_errors(report)
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
        "fixed_tpu_manifest_verified": fixed_manifest,
        "same_job_three_accelerators_verified": same_job,
        "real_v5e8_jax_training_verified": real_tpu,
        "tpu_aware_dynamic_placement_verified": dynamic_placement,
        "six_atomic_steps_verified": six_steps,
        "cross_framework_transport_verified": cross_transport,
        "finite_real_lora_updates_verified": numerical,
        "tpu_restart_checkpoint_restore_verified": recovery_verified,
        "backend_complete_checkpoints_verified": checkpoint_verified,
        "peft_export_cpu_reload_verified": export_verified,
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

    require(report.get("schema") == SCHEMA, "heterogeneous_tpu_beta_schema_invalid")
    stored_hash = str(report.get("content_hash") or "")
    require(
        stored_hash
        == _stable_hash({key: value for key, value in report.items() if key != "content_hash"}),
        "heterogeneous_tpu_beta_content_hash_invalid",
    )
    stored_gates = dict(report.get("acceptance_gates") or {})
    require(set(stored_gates) == REQUIRED_GATES, "heterogeneous_tpu_beta_gate_set_invalid")
    derived = build_acceptance_gates(report)
    require(stored_gates == derived, "heterogeneous_tpu_beta_gate_derivation_mismatch")
    safety_errors = public_safety_errors(report)
    require(not safety_errors, "heterogeneous_tpu_beta_public_safety_scan_failed")
    runtime_diagnosis = report.get("runtime_diagnosis")
    if runtime_diagnosis is not None:
        require(
            isinstance(runtime_diagnosis, dict)
            and runtime_diagnosis.get("schema")
            == "crowdtensor_heterogeneous_training_tpu_runtime_diagnosis_v1",
            "heterogeneous_tpu_beta_runtime_diagnosis_schema_invalid",
        )
        diagnosis = dict(runtime_diagnosis) if isinstance(runtime_diagnosis, dict) else {}
        diagnosis_hash = str(diagnosis.pop("content_hash", ""))
        diagnosis.pop("diagnosis_report_hash", None)
        require(
            diagnosis_hash == _stable_hash(diagnosis),
            "heterogeneous_tpu_beta_runtime_diagnosis_content_hash_invalid",
        )
        require(
            _hash(runtime_diagnosis.get("source_live_report_hash")),
            "heterogeneous_tpu_beta_runtime_diagnosis_source_hash_invalid",
        )
        require(
            runtime_diagnosis.get("public_artifact_safe") is True,
            "heterogeneous_tpu_beta_runtime_diagnosis_not_public_safe",
        )
        require(
            not (
                runtime_diagnosis.get("terminal_worker_report_retrieved") is False
                and runtime_diagnosis.get("root_cause_confirmed") is True
            ),
            "heterogeneous_tpu_beta_runtime_diagnosis_overclaims_root_cause",
        )
    stage_diagnostic = report.get("stage_diagnostic_summary")
    if stage_diagnostic is not None:
        require(
            isinstance(stage_diagnostic, dict)
            and stage_diagnostic.get("schema")
            == "crowdtensor_heterogeneous_training_tpu_stage_diagnostic_summary_v1",
            "heterogeneous_tpu_beta_stage_diagnostic_schema_invalid",
        )
        diagnostic = (
            dict(stage_diagnostic) if isinstance(stage_diagnostic, dict) else {}
        )
        require(
            _hash(diagnostic.get("diagnostic_report_hash")),
            "heterogeneous_tpu_beta_stage_diagnostic_report_hash_invalid",
        )
        require(
            diagnostic.get("diagnostic_only") is True
            and diagnostic.get("full_training_gate_evidence") is False
            and diagnostic.get("same_job_three_accelerator_evidence") is False
            and diagnostic.get("live_gate_ledger_modified") is False,
            "heterogeneous_tpu_beta_stage_diagnostic_boundary_invalid",
        )
        require(
            diagnostic.get("requested_accelerator") == "tpuV5e8"
            and int(diagnostic.get("stage_id") or -1) == 2,
            "heterogeneous_tpu_beta_stage_diagnostic_target_invalid",
        )
        require(
            (diagnostic.get("cleanup") or {}).get("live_resources_left_running")
            is False,
            "heterogeneous_tpu_beta_stage_diagnostic_cleanup_incomplete",
        )
    compile_import = report.get("tpu_compile_latency_evidence")
    if compile_import is not None:
        require(
            isinstance(compile_import, dict)
            and compile_import.get("schema")
            == "crowdtensor_heterogeneous_training_tpu_compile_latency_import_v1",
            "heterogeneous_tpu_beta_compile_import_schema_invalid",
        )
        evidence = dict(compile_import) if isinstance(compile_import, dict) else {}
        statuses = _rows(evidence.get("worker_statuses"))
        recovery = dict(report.get("tpu_recovery_evidence") or {})
        expected_ids = {
            "tpu_old": str(recovery.get("old_tpu_miner_id_hash") or ""),
            "tpu_replacement": str(
                recovery.get("replacement_tpu_miner_id_hash") or ""
            ),
        }
        status_roles = [str(item.get("deployment_role") or "") for item in statuses]
        latencies = [float(item.get("compile_latency_ms") or 0.0) for item in statuses]
        imported_latency = float(evidence.get("compile_latency_ms") or 0.0)
        require(
            _hash(evidence.get("source_kernel_report_hash"))
            and evidence.get("training_manifest_hash")
            == report.get("training_manifest_hash")
            and evidence.get("worker_status_count") == 2
            and status_roles == ["tpu_old", "tpu_replacement"]
            and all(
                item.get("miner_id_hash") == expected_ids[item["deployment_role"]]
                and _hash(item.get("miner_id_hash"))
                and item.get("runtime_backend") == "jax_tpu"
                and int(item.get("stage_id") or -1) == 2
                and math.isfinite(float(item.get("compile_latency_ms") or 0.0))
                and float(item.get("compile_latency_ms") or 0.0) > 0.0
                and int(item.get("jax_mesh_device_count") or 0) == 8
                and item.get("jax_mesh_shape") == [8]
                and item.get("all_mesh_devices_used") is True
                and item.get("forward_output_sharding_explicit") is True
                and item.get("backward_output_sharding_explicit") is True
                and item.get("boundary_output_replicated") is True
                and item.get("tensor_values_public") is False
                and item.get("public_artifact_safe") is True
                for item in statuses
            )
            and math.isfinite(imported_latency)
            and imported_latency == max(latencies or [0.0])
            and imported_latency
            == float(
                (report.get("tpu_training_evidence") or {}).get(
                    "compile_latency_ms"
                )
                or 0.0
            )
            and evidence.get("measurement_recomputed") is False
            and evidence.get(
                "measurement_recovered_from_retained_worker_status"
            )
            is True
            and evidence.get("credential_values_public") is False
            and evidence.get("private_paths_public") is False
            and evidence.get("tensor_values_public") is False
            and evidence.get("public_artifact_safe") is True,
            "heterogeneous_tpu_beta_compile_import_invalid",
        )
        source_evidence = dict(report.get("source_evidence") or {})
        require(
            source_evidence.get("tpu_kernel_report_hash")
            == evidence.get("source_kernel_report_hash")
            and source_evidence.get("omitted_live_compile_measurement_recovered")
            is True
            and source_evidence.get("compile_measurement_recomputed") is False,
            "heterogeneous_tpu_beta_compile_import_source_invalid",
        )
    resources = report.get("bounded_resource_summary")
    if resources is not None:
        require(
            isinstance(resources, dict)
            and resources.get("schema")
            == "crowdtensor_heterogeneous_training_tpu_resource_summary_v1",
            "heterogeneous_tpu_beta_resource_summary_schema_invalid",
        )
        resource_summary = dict(resources) if isinstance(resources, dict) else {}
        unlimited_kinds: set[str] = set()
        for field, kind, duration in (
            ("tpu_acquisition", "tpu_acquisition_window", 43200.0),
            ("live_gate", "six_step_live_gate", 21600.0),
        ):
            ledger = resource_summary.get(field)
            require(
                isinstance(ledger, dict),
                f"heterogeneous_tpu_beta_{field}_ledger_missing",
            )
            summary = dict(ledger) if isinstance(ledger, dict) else {}
            mode = str(summary.get("attempt_limit_mode") or "bounded")
            require(
                mode in {"bounded", "unlimited_authorized"},
                f"heterogeneous_tpu_beta_{field}_limit_mode_invalid",
            )
            if mode != "unlimited_authorized":
                continue
            unlimited_kinds.add(kind)
            authorizations = list(summary.get("attempt_authorizations") or [])
            valid_authorizations = [
                item
                for item in authorizations
                if isinstance(item, dict)
                and item.get("schema")
                == "crowdtensor_heterogeneous_training_attempt_authorization_v1"
                and item.get("kind") == kind
                and item.get("mode") == "unlimited_authorized"
                and _hash(item.get("authorization_id_hash"))
                and item.get("authorization_identifier_public") is False
                and float(item.get("max_attempt_duration_seconds") or 0.0)
                == duration
                and item.get("attempt_duration_remains_bounded") is True
                and item.get("credential_values_public") is False
                and item.get("public_artifact_safe") is True
            ]
            require(
                int(summary.get("attempt_limit") or 0) == 0
                and summary.get("unlimited_attempts_authorized") is True
                and summary.get("attempt_limit_reached") is False
                and len(authorizations) == 1
                and len(valid_authorizations) == 1,
                f"heterogeneous_tpu_beta_{field}_unlimited_authorization_invalid",
            )
        if unlimited_kinds == {
            "tpu_acquisition_window",
            "six_step_live_gate",
        }:
            resume = dict(report.get("resume_contract") or {})
            require(
                resume.get("current_goal_acquisition_boundary_exhausted") is False
                and resume.get("current_goal_live_gate_boundary_exhausted") is False
                and resume.get("next_action")
                == "start_next_bounded_tpu_window_then_full_live_gate",
                "heterogeneous_tpu_beta_unlimited_resume_contract_invalid",
            )
            require(
                "heterogeneous_tpu_training_live_gate_limit_reached"
                not in (report.get("blockers") or []),
                "heterogeneous_tpu_beta_superseded_live_limit_blocker_present",
            )
    require(
        report.get("blockers") == sorted(set(report.get("blockers") or [])),
        "heterogeneous_tpu_beta_blockers_not_canonical",
    )
    ready_claim = bool(
        report.get("live_run_performed") is True
        and all(derived.values())
        and not report.get("blockers")
    )
    require(
        report.get("heterogeneous_training_tpu_beta_ready") is ready_claim,
        "heterogeneous_tpu_beta_ready_claim_invalid",
    )
    if require_ready:
        for name, value in derived.items():
            require(value is True, f"heterogeneous_tpu_beta_gate_failed:{name}")
        require(not report.get("blockers"), "heterogeneous_tpu_beta_blockers_present")
        require(ready_claim, "heterogeneous_training_tpu_beta_not_ready")
    return {
        "schema": CHECK_SCHEMA,
        "ok": not errors,
        "heterogeneous_training_tpu_beta_ready": bool(not errors and ready_claim),
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
    print(json.dumps(result, sort_keys=True) if args.json else result)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
