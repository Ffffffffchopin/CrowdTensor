"""Core technology handoff RC helpers.

This layer aggregates the Large-Model Shard Alpha and Inference RC evidence into
a stable handoff package for the control, user-facing, and operator/economics
layers.  It does not broaden the runtime claim beyond the evidence supplied by
the Inference RC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crowdtensor import large_model_inference_rc as inference_rc
from crowdtensor import large_model_shard as alpha


HANDOFF_SCHEMA = "core_technology_handoff_rc_v1"
HANDOFF_SUPPORT_BUNDLE_SCHEMA = "core_technology_handoff_rc_support_bundle_v1"
HANDOFF_CHECK_SCHEMA = "core_technology_handoff_rc_check_v1"
DEPLOYMENT_RUNBOOK_SCHEMA = "core_technology_deployment_runbook_v1"
NEXT_LAYER_CONTRACT_SCHEMA = "core_technology_next_layer_contract_v1"
ADAPTER_CONFORMANCE_SCHEMA = "core_technology_adapter_conformance_v1"
TEST_GATE_SCHEMA = "core_technology_test_gate_summary_v1"


def stable_hash_payload(value: Any) -> str:
    return alpha.stable_hash_payload(value)


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    return alpha.artifact_entry(path, output_dir, kind=kind, schema=schema, ok=ok)


def artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    count = len(artifacts)
    present = sum(1 for item in artifacts.values() if isinstance(item, dict) and item.get("present"))
    return {
        "schema": "core_technology_handoff_rc_artifact_summary_v1",
        "artifact_count": count,
        "present_artifact_count": present,
        "public_artifact_safe": True,
        "support_bundle": artifacts.get("support_bundle_json", {}).get("path") if artifacts else "",
        "inspect_first": artifacts.get("summary_markdown", {}).get("path") if artifacts else "",
    }


def public_redaction_errors(value: Any) -> list[str]:
    return alpha.public_redaction_errors(value)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_hash(report: dict[str, Any]) -> str:
    if not report:
        return ""
    return stable_hash_payload(report)


def _combined_diagnosis_codes(report: dict[str, Any]) -> set[str]:
    codes: set[str] = set(str(item) for item in _list(report.get("diagnosis_codes")) if item)
    payload_summaries = _dict(report.get("payload_summaries"))
    for payload in payload_summaries.values():
        if not isinstance(payload, dict):
            continue
        codes.update(str(item) for item in _list(payload.get("diagnosis_codes")) if item)
        live_rc = _dict(payload.get("live_rc"))
        codes.update(str(item) for item in _list(live_rc.get("diagnosis_codes")) if item)
    return codes


def _first_generation(report: dict[str, Any]) -> dict[str, Any]:
    generation = _dict(report.get("generation"))
    if generation:
        return generation
    payload_summaries = _dict(report.get("payload_summaries"))
    for key in ["external_alpha", "alpha", "live_rc"]:
        payload = _dict(payload_summaries.get(key))
        generation = _dict(payload.get("generation"))
        if generation:
            return generation
        live_rc = _dict(payload.get("live_rc"))
        generation = _dict(live_rc.get("generation"))
        if generation:
            return generation
    return {}


def _first_workload(report: dict[str, Any]) -> dict[str, Any]:
    workload = _dict(report.get("workload"))
    if workload:
        return workload
    payload_summaries = _dict(report.get("payload_summaries"))
    for key in ["external_alpha", "alpha", "live_rc"]:
        payload = _dict(payload_summaries.get(key))
        workload = _dict(payload.get("workload"))
        if workload:
            return workload
    return {}


def _first_runtime_classification(report: dict[str, Any]) -> dict[str, Any]:
    runtime = _dict(report.get("runtime_classification"))
    if runtime:
        return runtime
    payload_summaries = _dict(report.get("payload_summaries"))
    for key in ["external_alpha", "alpha", "live_rc"]:
        payload = _dict(payload_summaries.get(key))
        runtime = _dict(payload.get("runtime_classification"))
        if runtime:
            return runtime
    return {}


def _kaggle_gpu_stage_summary(report: dict[str, Any]) -> dict[str, Any]:
    package = _dict(_dict(report.get("payload_summaries")).get("kaggle_package"))
    stages = [stage for stage in _list(package.get("stages")) if isinstance(stage, dict)]
    stage_summaries = [
        {
            "stage": stage.get("stage"),
            "role": stage.get("role"),
            "gpu_accelerator_enabled": bool(stage.get("gpu_accelerator_enabled")),
            "cuda_preflight_present": bool(stage.get("cuda_preflight_present")),
            "hf_runtime_enabled": bool(stage.get("hf_runtime_enabled")),
            "real_llm_stage_role_present": bool(stage.get("real_llm_stage_role_present")),
            "backend": stage.get("real_llm_backend"),
            "execution_mode": stage.get("real_llm_execution_mode"),
            "partition_mode": stage.get("real_llm_partition_mode"),
        }
        for stage in stages
    ]
    return {
        "schema": "core_technology_kaggle_gpu_stage_summary_v1",
        "stage_count": len(stage_summaries),
        "all_stages_gpu_accelerator_enabled": bool(stage_summaries) and all(stage.get("gpu_accelerator_enabled") for stage in stage_summaries),
        "all_stages_cuda_preflight_present": bool(stage_summaries) and all(stage.get("cuda_preflight_present") for stage in stage_summaries),
        "all_stages_hf_runtime_enabled": bool(stage_summaries) and all(stage.get("hf_runtime_enabled") for stage in stage_summaries),
        "all_stages_role_present": bool(stage_summaries) and all(stage.get("real_llm_stage_role_present") for stage in stage_summaries),
        "stages": stage_summaries,
    }


def summarize_stage_selective_live_report(
    report: dict[str, Any],
    *,
    model_family: str,
    minimum_generated_tokens: int,
    require_multi_token: bool,
) -> dict[str, Any]:
    workload = _first_workload(report)
    generation = _first_generation(report)
    runtime = _first_runtime_classification(report)
    kaggle_gpu_stages = _kaggle_gpu_stage_summary(report)
    codes = _combined_diagnosis_codes(report)
    model_id = str(workload.get("hf_model_id") or workload.get("model_id") or "")
    generated_token_count = _as_int(generation.get("generated_token_count"))
    max_new_tokens = _as_int(generation.get("max_new_tokens") or workload.get("max_new_tokens"))
    redaction_errors = public_redaction_errors(report)
    checks = {
        "schema_ok": report.get("schema") == "real_llm_internet_beta_v1",
        "source_ok": report.get("ok") is True,
        "model_family_match": model_family.lower() in model_id.lower(),
        "external_runtime_verified": bool(runtime.get("external_runtime_verified")) or "external_runtime_verified" in codes,
        "kaggle_notebook_verified": bool(runtime.get("kaggle_notebook_verified")) or "kaggle_auto_ready" in codes,
        "kaggle_gpu_stage_package_ready": bool(
            kaggle_gpu_stages.get("all_stages_gpu_accelerator_enabled")
            and kaggle_gpu_stages.get("all_stages_cuda_preflight_present")
            and kaggle_gpu_stages.get("all_stages_hf_runtime_enabled")
            and kaggle_gpu_stages.get("all_stages_role_present")
        ),
        "kaggle_kernels_deleted": "kaggle_kernels_deleted" in codes,
        "stage_assignment_valid": "stage_assignment_valid" in codes,
        "distinct_stage_miners": "distinct_stage_miners" in codes,
        "stage_selective_execution": workload.get("real_llm_execution_mode") == "stage_selective_hf",
        "stage_local_partition": workload.get("real_llm_partition_mode") == "stage_local",
        "split_stage_mode": workload.get("stage_mode") == "split",
        "generation_complete": "generation_complete" in codes,
        "generated_token_target_met": generated_token_count >= minimum_generated_tokens,
        "public_artifact_safe": not redaction_errors,
    }
    if require_multi_token:
        checks["multi_token_generation_ready"] = bool(generation.get("multi_token_generation_ready")) or "multi_token_generation_ready" in codes
    verified = all(checks.values())
    return {
        "schema": "core_technology_stage_selective_live_report_summary_v1",
        "verified": verified,
        "source_schema": report.get("schema"),
        "source_report_hash": _source_hash(report),
        "model_id": model_id,
        "model_family": model_family,
        "backend": workload.get("real_llm_backend"),
        "execution_mode": workload.get("real_llm_execution_mode"),
        "partition_mode": workload.get("real_llm_partition_mode"),
        "stage_mode": workload.get("stage_mode"),
        "external_runtime_verified": checks["external_runtime_verified"],
        "kaggle_notebook_verified": checks["kaggle_notebook_verified"],
        "kaggle_gpu_stage_package": kaggle_gpu_stages,
        "kaggle_kernels_deleted": checks["kaggle_kernels_deleted"],
        "stage_assignment_valid": checks["stage_assignment_valid"],
        "distinct_stage_miners": checks["distinct_stage_miners"],
        "generated_token_count": generated_token_count,
        "max_new_tokens": max_new_tokens,
        "multi_token_generation_ready": bool(generation.get("multi_token_generation_ready")) or "multi_token_generation_ready" in codes,
        "public_artifact_safe": checks["public_artifact_safe"],
        "redaction_error_count": len(redaction_errors),
        "checks": checks,
        "missing_requirements": [name for name, ok in checks.items() if not ok],
    }


def summarize_stage_selective_plan_report(report: dict[str, Any]) -> dict[str, Any]:
    model_summaries: list[dict[str, Any]] = []
    target_stage_count = _as_int(report.get("target_stage_count"))
    for item in _list(report.get("model_plans")):
        if not isinstance(item, dict):
            continue
        n_stage = _dict(item.get("n_stage_partition_plan") or _dict(item.get("execution_support")).get("n_stage_partition_plan"))
        stage_plans = [stage for stage in _list(n_stage.get("stage_plans")) if isinstance(stage, dict)]
        stage_count = _as_int(n_stage.get("stage_count") or item.get("target_stage_count"))
        target_stage_count = max(target_stage_count, stage_count)
        stage_weight_bytes = [
            _as_int(stage.get("estimated_stage_weight_bytes_fp32"))
            for stage in stage_plans
            if stage.get("estimated_stage_weight_bytes_fp32") is not None
        ]
        model_summaries.append({
            "model_id": item.get("model_id"),
            "parameter_count_estimate": _as_int(item.get("parameter_count_estimate")),
            "target_stage_count": stage_count,
            "n_stage_plan_ready": bool(item.get("n_stage_plan_ready") or n_stage.get("ready")),
            "two_stage_plan_ready": bool(item.get("two_stage_plan_ready")),
            "dual_kaggle_kernel_fit_estimate": bool(item.get("dual_kaggle_kernel_fit_estimate")),
            "two_stage_practical_fit_with_overhead_guard": bool(item.get("two_stage_practical_fit_with_overhead_guard")),
            "n_stage_max_stage_weight_gb_fp16_estimate": _as_float(item.get("n_stage_max_stage_weight_gb_fp16_estimate")),
            "two_stage_max_stage_weight_gb_fp16_estimate": _as_float(item.get("two_stage_max_stage_weight_gb_fp16_estimate")),
            "stage_count": stage_count,
            "stage_plan_count": len(stage_plans),
            "stage_ranges_valid": bool(n_stage.get("stage_ranges_valid")),
            "loads_only_stage_weight_keys": bool(stage_plans) and all(stage.get("loads_only_stage_weight_keys") is True for stage in stage_plans),
            "max_stage_weight_bytes_fp32": max(stage_weight_bytes) if stage_weight_bytes else 0,
        })
    redaction_errors = public_redaction_errors(report)
    codes = set(str(item) for item in _list(report.get("diagnosis_codes")) if item)
    seven_b_ready = any("7b" in str(item.get("model_id") or "").lower() and item.get("n_stage_plan_ready") for item in model_summaries)
    fourteen_b_ready = any("14b" in str(item.get("model_id") or "").lower() and item.get("n_stage_plan_ready") for item in model_summaries)
    checks = {
        "schema_ok": report.get("schema") == "large_model_stage_selective_plan_v1",
        "source_ok": report.get("ok") is True,
        "n_stage_code_ready": "large_model_n_stage_partition_plan_ready" in codes,
        "target_stage_count_gt_two": target_stage_count > 2,
        "seven_b_plan_ready": seven_b_ready,
        "fourteen_b_plan_ready": fourteen_b_ready,
        "stage_weight_scope_ready": bool(model_summaries) and all(item.get("loads_only_stage_weight_keys") for item in model_summaries),
        "public_artifact_safe": not redaction_errors,
    }
    return {
        "schema": "core_technology_stage_selective_plan_summary_v1",
        "n_stage_partition_plan_ready": all(checks.values()),
        "source_schema": report.get("schema"),
        "source_report_hash": _source_hash(report),
        "target_stage_count": target_stage_count,
        "kaggle_gpu_memory_gb": _as_float(report.get("kaggle_gpu_memory_gb")),
        "model_plan_count": len(model_summaries),
        "model_plan_summaries": model_summaries,
        "limitations": _list(report.get("limitations")),
        "public_artifact_safe": checks["public_artifact_safe"],
        "redaction_error_count": len(redaction_errors),
        "checks": checks,
        "missing_requirements": [name for name, ok in checks.items() if not ok],
    }


def summarize_stage_selective_performance_report(report: dict[str, Any]) -> dict[str, Any]:
    performance = _dict(report.get("performance"))
    memory = _dict(performance.get("memory"))
    latency = _dict(performance.get("latency"))
    throughput = _dict(performance.get("throughput"))
    failure_recovery = _dict(performance.get("failure_recovery"))
    redaction_errors = public_redaction_errors(report)
    checks = {
        "schema_ok": report.get("schema") == "real_llm_sharded_evidence_v1",
        "source_ok": report.get("ok") is True,
        "performance_schema_ok": performance.get("schema") == "real_llm_sharded_performance_summary_v1",
        "memory_report_ready": bool(memory),
        "stage_weight_download_scope_ready": bool(memory.get("stage_weight_downloads_only_stage_files")),
        "latency_report_ready": bool(latency),
        "throughput_report_ready": bool(throughput),
        "failure_recovery_report_ready": bool(failure_recovery),
        "public_artifact_safe": performance.get("public_artifact_safe") is True and not redaction_errors,
    }
    return {
        "schema": "core_technology_stage_selective_performance_summary_v1",
        "performance_report_ready": all(checks.values()),
        "source_schema": report.get("schema"),
        "source_report_hash": _source_hash(report),
        "memory": {
            "full_model_parameter_count": _as_int(memory.get("full_model_parameter_count")),
            "stage0_parameter_count": _as_int(memory.get("stage0_parameter_count")),
            "stage1_parameter_count": _as_int(memory.get("stage1_parameter_count")),
            "stage0_loaded_tensor_bytes": _as_int(memory.get("stage0_loaded_tensor_bytes")),
            "stage1_loaded_tensor_bytes": _as_int(memory.get("stage1_loaded_tensor_bytes")),
            "stage0_weight_download_scope": memory.get("stage0_weight_download_scope"),
            "stage1_weight_download_scope": memory.get("stage1_weight_download_scope"),
            "stage0_weight_download_file_count": _as_int(memory.get("stage0_weight_download_file_count")),
            "stage1_weight_download_file_count": _as_int(memory.get("stage1_weight_download_file_count")),
            "stage_weight_downloads_only_stage_files": bool(memory.get("stage_weight_downloads_only_stage_files")),
            "stage_gpu_memory_reduced": bool(memory.get("stage_gpu_memory_reduced")),
        },
        "latency": {
            "effective_elapsed_seconds": _as_float(latency.get("effective_elapsed_seconds")),
            "total_stage_elapsed_ms": _as_float(latency.get("total_stage_elapsed_ms")),
            "stage0_elapsed_ms": _as_float(latency.get("stage0_elapsed_ms")),
            "stage1_elapsed_ms": _as_float(latency.get("stage1_elapsed_ms")),
        },
        "throughput": {
            "generated_token_count": _as_int(throughput.get("generated_token_count")),
            "max_new_tokens": _as_int(throughput.get("max_new_tokens")),
            "completed_generation_steps": _as_int(throughput.get("completed_generation_steps")),
            "tokens_per_second_effective": _as_float(throughput.get("tokens_per_second_effective")),
        },
        "failure_recovery": {
            "requeue_observed": bool(failure_recovery.get("requeue_observed")),
            "stage0_attempt": _as_int(failure_recovery.get("stage0_attempt")),
            "stage1_attempt": _as_int(failure_recovery.get("stage1_attempt")),
        },
        "public_artifact_safe": checks["public_artifact_safe"],
        "redaction_error_count": len(redaction_errors),
        "checks": checks,
        "missing_requirements": [name for name, ok in checks.items() if not ok],
    }


def build_large_model_stage_selective_evidence_summary(
    *,
    seven_b_live_report: dict[str, Any] | None = None,
    fourteen_b_live_report: dict[str, Any] | None = None,
    stage_selective_plan_report: dict[str, Any] | None = None,
    stage_selective_performance_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seven_b_summary = summarize_stage_selective_live_report(
        seven_b_live_report or {},
        model_family="7b",
        minimum_generated_tokens=2,
        require_multi_token=True,
    )
    fourteen_b_summary = summarize_stage_selective_live_report(
        fourteen_b_live_report or {},
        model_family="14b",
        minimum_generated_tokens=1,
        require_multi_token=False,
    )
    plan_summary = summarize_stage_selective_plan_report(stage_selective_plan_report or {})
    performance_summary = summarize_stage_selective_performance_report(stage_selective_performance_report or {})
    checks = {
        "seven_b_multi_token_verified": bool(seven_b_summary.get("verified")),
        "fourteen_b_dual_kaggle_verified": bool(fourteen_b_summary.get("verified")),
        "n_stage_partition_plan_ready": bool(plan_summary.get("n_stage_partition_plan_ready")),
        "stage_selective_performance_report_ready": bool(performance_summary.get("performance_report_ready")),
    }
    ready = all(checks.values())
    diagnosis_codes = []
    if checks["seven_b_multi_token_verified"]:
        diagnosis_codes.append("core_technology_7b_multi_token_verified")
    if checks["fourteen_b_dual_kaggle_verified"]:
        diagnosis_codes.append("core_technology_14b_dual_kaggle_verified")
    if checks["n_stage_partition_plan_ready"]:
        diagnosis_codes.append("core_technology_n_stage_partition_plan_ready")
    if checks["stage_selective_performance_report_ready"]:
        diagnosis_codes.append("core_technology_stage_selective_performance_report_ready")
    if ready:
        diagnosis_codes.append("core_technology_large_model_alpha_ready")
        diagnosis_codes.append("core_technology_live_kaggle_stage_selective_evidence_ready")
    else:
        diagnosis_codes.append("core_technology_large_model_stage_selective_evidence_incomplete")
    return {
        "schema": "core_technology_large_model_stage_selective_evidence_v1",
        "core_technology_large_model_alpha_ready": ready,
        "evidence_scope": "live-kaggle-stage-selective" if ready else "not-complete",
        "seven_b_live": seven_b_summary,
        "fourteen_b_live": fourteen_b_summary,
        "n_stage_partition": plan_summary,
        "stage_selective_performance": performance_summary,
        "checks": checks,
        "not_completed": [name for name, ok in checks.items() if not ok],
        "limitations": [
            "Live runtime evidence is the controlled two-Kaggle-stage path; N-stage is a planner/runtime abstraction until live N-stage scheduling is wired.",
            "This is not production P2P, not arbitrary public prompt serving, and not unbounded GPU pooling.",
            "Kaggle evidence is temporary external Miner validation and remains subject to account/kernel limits.",
        ],
        "diagnosis_codes": diagnosis_codes,
    }


def build_deployment_runbook(
    *,
    output_dir: Path,
    inference_report: dict[str, Any],
    large_model_stage_selective_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runner = inference_report.get("runner_result") if isinstance(inference_report.get("runner_result"), dict) else {}
    runtime_probe = inference_report.get("runtime_adapter_probe") if isinstance(inference_report.get("runtime_adapter_probe"), dict) else {}
    blockers = list(inference_report.get("blockers") or [])
    max_new_tokens = int(runner.get("max_new_tokens") or inference_rc.DEFAULT_RC_MAX_NEW_TOKENS)
    runbook = {
        "schema": DEPLOYMENT_RUNBOOK_SCHEMA,
        "output_dir": str(output_dir),
        "ready": True,
        "local_fixture": {
            "command": "crowdtensor large-model-shard-rc --mode fixture --json",
            "ci_safe": True,
            "real_runtime_verified": False,
            "purpose": "Validate contracts, artifacts, redaction, planner, runner, benchmark, and serving hooks without a GGUF runtime.",
        },
        "local_real_runtime": {
            "command": (
                "crowdtensor large-model-shard-rc --mode real "
                "--model-path /models/llama-7b.Q4_K_M.gguf "
                "--rpc-endpoint http://127.0.0.1:50052 "
                f"--max-new-tokens {min(max_new_tokens, inference_rc.MAX_REAL_RUN_TOKENS)} --real-timeout-seconds 1200 --json"
            ),
            "requires": [
                "local GGUF model file",
                "llama-cli on PATH",
                "reachable controlled llama.cpp RPC worker or --start-workers setup",
                "enough CPU/RAM/GPU/VRAM for the selected partition",
            ],
            "timeout_seconds_max": inference_rc.MAX_REAL_RUN_TIMEOUT_SECONDS,
            "max_new_tokens_max": inference_rc.MAX_REAL_RUN_TOKENS,
        },
        "lan_vpn_two_worker_runtime": {
            "worker_commands": [
                item.get("command_line")
                for item in (inference_report.get("runtime_adapter") or {}).get("worker_commands", [])
                if isinstance(item, dict) and item.get("command_line")
            ],
            "client_command": (inference_report.get("runtime_adapter") or {}).get("client_command_line", ""),
            "controlled_network_only": True,
            "not_public_rpc_safe": True,
        },
        "import_retained_evidence": {
            "command": (
                "crowdtensor large-model-shard-rc "
                "--real-run-report /secure/private/large_model_real_run.json "
                "--real-benchmark-report /secure/private/large_model_benchmark.json --json"
            ),
            "real_run_required_fields": [
                "ttft_ms",
                "tokens_per_second",
                "wall_time_seconds",
                "generated_token_count",
                "output_digest",
            ],
            "benchmark_import_supplements_metrics_only": True,
        },
        "import_stage_selective_live_evidence": {
            "ready": bool((large_model_stage_selective_evidence or {}).get("core_technology_large_model_alpha_ready")),
            "command": (
                "crowdtensor core-tech-handoff "
                "--seven-b-live-report /public-safe/real_llm_internet_beta_7b.json "
                "--fourteen-b-live-report /public-safe/real_llm_internet_beta_14b.json "
                "--stage-selective-plan-report /public-safe/large_model_stage_selective_plan.json "
                "--stage-selective-performance-report /public-safe/real_llm_sharded_evidence_14b.json --json"
            ),
            "required_public_safe_reports": [
                "7B stage-selective Kaggle live report with multi-token generation",
                "14B stage-selective Kaggle live report",
                "N-stage partition planning report",
                "memory/latency/throughput/failure-recovery performance report",
            ],
            "claim_boundary": "controlled live Kaggle stage-selective Alpha evidence, not production P2P/GPU pooling",
        },
        "troubleshooting": {
            "blockers": blockers,
            "runtime_probe_codes": runtime_probe.get("diagnosis_codes") or [],
            "operator_actions": [
                "Install llama.cpp client/server binaries or pass their explicit paths.",
                "Provide a local GGUF model path; do not auto-download large models.",
                "Start controlled local/LAN/VPN RPC workers and verify endpoint reachability.",
                "Import a public-safe real-run report when external runtime proof already exists.",
            ],
        },
        "cleanup": {
            "process_leak_check": "ps -eo pid,comm,args | rg -i 'llama|rpc-server|large_model_inference|large-model-shard-rc' || true",
            "clean_artifacts": "crowdtensor clean-artifacts --dry-run",
            "runtime_processes_started_by_runner_must_be_terminated": True,
        },
        "diagnosis_codes": ["core_technology_deployment_runbook_ready"],
    }
    runbook["runbook_hash"] = stable_hash_payload(runbook)
    return runbook


def build_next_layer_contract(
    *,
    inference_report: dict[str, Any],
    large_model_stage_selective_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    serving = inference_report.get("serving_readiness_hooks") if isinstance(inference_report.get("serving_readiness_hooks"), dict) else {}
    partition = inference_report.get("partition_manifest") if isinstance(inference_report.get("partition_manifest"), dict) else {}
    benchmark = inference_report.get("benchmark") if isinstance(inference_report.get("benchmark"), dict) else {}
    correctness = inference_report.get("correctness_summary") if isinstance(inference_report.get("correctness_summary"), dict) else {}
    contract = {
        "schema": NEXT_LAYER_CONTRACT_SCHEMA,
        "ready": True,
        "control_layer": {
            "stable_entrypoints": [
                "crowdtensor large-model-shard-rc",
                "crowdtensor large-model-shard --stage-selective-plan",
                "crowdtensor real-llm-internet-beta --mode kaggle-auto",
                "scripts/large_model_inference_rc_pack.py",
                "scripts/core_technology_handoff_pack.py",
            ],
            "route_health_schema": serving.get("health_aware_route_metadata_schema"),
            "runner_result_schema": inference_rc.RUNNER_RESULT_SCHEMA,
            "blocker_codes_source": "core_technology_handoff_rc_v1.blockers",
            "schedule_inputs": [
                "partition_manifest.assignments",
                "partition_manifest.tensor_split_plan",
                "device_profile.devices",
                "runtime_adapter_probe.rpc_endpoint_health",
            ],
        },
        "user_layer": {
            "safe_status_fields": [
                "ok",
                "real_runtime_verified",
                "real_7b_runtime_verified",
                "core_technology_large_model_alpha_ready",
                "mode",
                "diagnosis_codes",
                "blockers",
            ],
            "streaming_event_schema": serving.get("streaming_event_schema"),
            "bounded_batch_request_schema": serving.get("bounded_batch_request_schema"),
            "answer_visibility": "public artifacts expose digests and readiness only; local generated text belongs to a human runtime command.",
        },
        "permissions_trust_billing_layer": {
            "core_signals": [
                "runtime_backend",
                "model_id",
                "partition_hash",
                "runner_result.real_runtime_verified",
                "benchmark.tokens_per_second",
                "benchmark.wall_time_seconds",
                "correctness_summary.output_digest",
                "route_health.healthy",
                "process_cleanup.completed",
                "large_model_stage_selective_evidence.seven_b_live.generated_token_count",
                "large_model_stage_selective_evidence.fourteen_b_live.generated_token_count",
                "large_model_stage_selective_evidence.stage_selective_performance.throughput.tokens_per_second_effective",
            ],
            "not_implemented_here": [
                "accounts",
                "billing",
                "trust scores",
                "incentives",
                "staking",
                "slashing",
            ],
        },
        "sample_control_request": {
            "schema": "core_technology_control_request_v1",
            "mode": inference_report.get("mode"),
            "partition_hash": partition.get("partition_hash"),
            "max_new_tokens": (serving.get("bounded_batch_request") or {}).get("max_new_tokens"),
            "timeout_seconds": (serving.get("bounded_batch_request") or {}).get("timeout_seconds"),
            "cancel_requested": False,
            "raw_prompt_public": False,
        },
        "performance_contract": {
            "measurement_kind": benchmark.get("measurement_kind"),
            "real_runtime_verified": bool(benchmark.get("real_runtime_verified")),
            "ttft_ms": benchmark.get("ttft_ms"),
            "tokens_per_second": benchmark.get("tokens_per_second"),
            "wall_time_seconds": benchmark.get("wall_time_seconds"),
        },
        "large_model_stage_selective_contract": {
            "schema": "core_technology_large_model_stage_selective_contract_v1",
            "ready": bool((large_model_stage_selective_evidence or {}).get("core_technology_large_model_alpha_ready")),
            "seven_b_multi_token_verified": bool(((large_model_stage_selective_evidence or {}).get("checks") or {}).get("seven_b_multi_token_verified")),
            "fourteen_b_dual_kaggle_verified": bool(((large_model_stage_selective_evidence or {}).get("checks") or {}).get("fourteen_b_dual_kaggle_verified")),
            "n_stage_partition_plan_ready": bool(((large_model_stage_selective_evidence or {}).get("checks") or {}).get("n_stage_partition_plan_ready")),
            "stage_selective_performance_report_ready": bool(((large_model_stage_selective_evidence or {}).get("checks") or {}).get("stage_selective_performance_report_ready")),
            "public_runtime_boundary": "Coordinator-backed controlled Kaggle Miner Alpha/Beta evidence; not production P2P, not arbitrary public prompt serving.",
        },
        "correctness_contract": {
            "generated_token_count": correctness.get("generated_token_count"),
            "output_digest": correctness.get("output_digest"),
            "baseline_comparison": correctness.get("baseline_comparison"),
            "generated_token_ids_public": False,
        },
        "diagnosis_codes": ["core_technology_next_layer_contract_ready"],
    }
    contract["contract_hash"] = stable_hash_payload(contract)
    return contract


def build_adapter_conformance(*, inference_report: dict[str, Any]) -> dict[str, Any]:
    adapter_interface = inference_report.get("adapter_interface") if isinstance(inference_report.get("adapter_interface"), dict) else {}
    descriptors = adapter_interface.get("descriptors") if isinstance(adapter_interface.get("descriptors"), list) else []
    descriptor_checks = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        status = descriptor.get("status")
        descriptor_checks.append({
            "adapter_kind": descriptor.get("adapter_kind"),
            "status": status,
            "has_capability_contract": bool(descriptor.get("capabilities") or descriptor.get("operator_action")),
            "unsupported_diagnostic_ready": bool(status == "supported" or "unsupported_runtime_backend" in (descriptor.get("diagnosis_codes") or [])),
            "conformant": bool(
                descriptor.get("adapter_kind")
                and status in {"supported", "unsupported"}
                and (descriptor.get("capabilities") or descriptor.get("operator_action"))
                and (status == "supported" or "unsupported_runtime_backend" in (descriptor.get("diagnosis_codes") or []))
            ),
        })
    conformance = {
        "schema": ADAPTER_CONFORMANCE_SCHEMA,
        "ready": bool(descriptor_checks and all(item.get("conformant") for item in descriptor_checks)),
        "selected_runtime_backend": adapter_interface.get("selected_runtime_backend"),
        "selected_supported": bool(adapter_interface.get("selected_supported")),
        "descriptor_checks": descriptor_checks,
        "future_runtime_backends": [
            item.get("adapter_kind")
            for item in descriptor_checks
            if item.get("status") == "unsupported"
        ],
        "diagnosis_codes": ["core_technology_adapter_conformance_ready"]
        + ([] if descriptor_checks and all(item.get("conformant") for item in descriptor_checks) else ["core_technology_adapter_conformance_failed"]),
    }
    conformance["conformance_hash"] = stable_hash_payload(conformance)
    return conformance


def build_test_gate_summary(*, mode: str, full_pytest: bool = False) -> dict[str, Any]:
    commands = [
        "python -m py_compile crowdtensor/core_technology_handoff.py scripts/core_technology_handoff_pack.py scripts/core_technology_handoff_check.py",
        "python scripts/core_technology_handoff_check.py --mode fixture --json",
        "python scripts/large_model_inference_rc_check.py --mode fixture --json",
        "python -m pytest tests/test_core_technology_handoff.py tests/test_large_model_inference_rc.py tests/test_large_model_shard_alpha.py -q",
    ]
    if full_pytest:
        commands.append("python -m pytest -q")
    summary = {
        "schema": TEST_GATE_SCHEMA,
        "ready": True,
        "mode": mode,
        "ci_safe": True,
        "commands": commands,
        "full_pytest_requested": bool(full_pytest),
        "coverage": [
            "CLI/API entry",
            "adapter interface",
            "unsupported adapters",
            "device profile import/export",
            "planner v2",
            "runner plan/fixture/import",
            "real mode validation and timeout constraints",
            "benchmark v2",
            "correctness",
            "serving hooks",
            "deployment/runbook artifact generation",
            "aggregate handoff report",
            "redaction",
            "backward compatibility",
        ],
        "diagnosis_codes": ["core_technology_test_gates_ready"],
    }
    summary["test_gate_hash"] = stable_hash_payload(summary)
    return summary


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    stage_evidence = report.get("large_model_stage_selective_evidence") if isinstance(report.get("large_model_stage_selective_evidence"), dict) else {}
    return {
        "schema": HANDOFF_SUPPORT_BUNDLE_SCHEMA,
        "report_schema": report.get("schema"),
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "real_runtime_verified": bool(report.get("real_runtime_verified")),
        "real_7b_runtime_verified": bool(report.get("real_7b_runtime_verified")),
        "core_technology_large_model_alpha_ready": bool(report.get("core_technology_large_model_alpha_ready")),
        "large_model_stage_selective_checks": stage_evidence.get("checks") if stage_evidence else {},
        "large_model_stage_selective_not_completed": stage_evidence.get("not_completed") if stage_evidence else [],
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "blockers": report.get("blockers") or [],
        "artifact_summary": report.get("artifact_summary"),
        "public_artifact_safe": bool((report.get("safety") or {}).get("public_artifact_safe")),
    }


def build_handoff_report(
    *,
    output_dir: Path,
    mode: str,
    inference_report: dict[str, Any],
    deployment_runbook: dict[str, Any],
    next_layer_contract: dict[str, Any],
    adapter_conformance: dict[str, Any],
    test_gate_summary: dict[str, Any],
    large_model_stage_selective_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    real_verified = bool(inference_report.get("real_runtime_verified"))
    stage_selective_evidence = large_model_stage_selective_evidence or {}
    stage_selective_ready = bool(stage_selective_evidence.get("core_technology_large_model_alpha_ready"))
    blockers = list(inference_report.get("blockers") or [])
    if not real_verified:
        for item in [
            "core_technology_real_7b_runtime_not_verified",
            "external_real_runtime_resources_required",
        ]:
            if item not in blockers:
                blockers.append(item)
    elif not bool(inference_report.get("real_7b_runtime_verified")):
        if "core_technology_real_7b_runtime_not_verified" not in blockers:
            blockers.append("core_technology_real_7b_runtime_not_verified")
    codes = [
        "core_technology_handoff_rc_ready",
        "core_technology_stable_entrypoint_ready",
        "core_technology_inference_rc_imported",
        "core_technology_deployment_runbook_ready",
        "core_technology_next_layer_contract_ready",
        "core_technology_adapter_conformance_ready",
        "core_technology_test_gates_ready",
        "core_technology_public_artifact_redaction_ready",
    ]
    for source in [inference_report, deployment_runbook, next_layer_contract, adapter_conformance, test_gate_summary]:
        codes.extend(source.get("diagnosis_codes") or [])
    codes.extend(stage_selective_evidence.get("diagnosis_codes") or [])
    if real_verified:
        codes.append("core_technology_real_runtime_verified")
    else:
        codes.extend([
            "core_technology_real_runtime_not_verified",
            "core_technology_handoff_fixture_or_import_ready",
        ])
    seen: set[str] = set()
    diagnosis_codes = [code for code in codes if not (code in seen or seen.add(code))]
    report = {
        "schema": HANDOFF_SCHEMA,
        "ok": bool(
            inference_report.get("ok")
            and deployment_runbook.get("ready")
            and next_layer_contract.get("ready")
            and adapter_conformance.get("ready")
            and test_gate_summary.get("ready")
        ),
        "mode": mode,
        "output_dir": str(output_dir),
        "stable_entrypoints": [
            "crowdtensor large-model-shard-rc",
            "crowdtensor large-model-shard --stage-selective-plan",
            "crowdtensor real-llm-internet-beta --mode kaggle-auto",
            "crowdtensor core-tech-handoff",
            "scripts/core_technology_handoff_pack.py",
            "scripts/core_technology_handoff_check.py",
        ],
        "real_runtime_verified": real_verified,
        "real_7b_runtime_verified": bool(inference_report.get("real_7b_runtime_verified")),
        "core_technology_large_model_alpha_ready": stage_selective_ready,
        "capability_summary": {
            "can_plan_large_model_sharding": True,
            "can_run_ci_safe_fixture": True,
            "can_import_real_runtime_evidence": True,
            "can_import_live_stage_selective_kaggle_evidence": True,
            "can_attempt_controlled_real_runtime": True,
            "can_export_next_layer_contract": True,
            "can_support_control_layer_development": True,
            "can_support_user_layer_development": True,
            "can_support_permissions_trust_billing_layer_development": True,
            "requires_external_runtime_for_real_7b_claim": not bool(inference_report.get("real_7b_runtime_verified")),
            "seven_b_multi_token_verified": bool((stage_selective_evidence.get("checks") or {}).get("seven_b_multi_token_verified")),
            "fourteen_b_dual_kaggle_verified": bool((stage_selective_evidence.get("checks") or {}).get("fourteen_b_dual_kaggle_verified")),
            "n_stage_partition_plan_ready": bool((stage_selective_evidence.get("checks") or {}).get("n_stage_partition_plan_ready")),
            "stage_selective_performance_report_ready": bool((stage_selective_evidence.get("checks") or {}).get("stage_selective_performance_report_ready")),
        },
        "evidence_scope": (
            "live-kaggle-stage-selective-handoff"
            if stage_selective_ready
            else ("real-runtime" if real_verified else "fixture-diagnostic-handoff")
        ),
        "inference_rc_report": inference_report,
        "alpha_evidence": inference_report.get("alpha_report"),
        "large_model_stage_selective_evidence": stage_selective_evidence or None,
        "adapter_interface": inference_report.get("adapter_interface"),
        "runtime_probe": inference_report.get("runtime_adapter_probe"),
        "device_profile": inference_report.get("device_profile"),
        "partition_planner": inference_report.get("partition_manifest"),
        "runner_result": inference_report.get("runner_result"),
        "benchmark": inference_report.get("benchmark"),
        "correctness_summary": inference_report.get("correctness_summary"),
        "serving_hooks": inference_report.get("serving_readiness_hooks"),
        "deployment_runbook": deployment_runbook,
        "next_layer_integration_contract": next_layer_contract,
        "adapter_conformance": adapter_conformance,
        "test_gate_summary": test_gate_summary,
        "blockers": [item for index, item in enumerate(blockers) if item and item not in blockers[:index]],
        "handoff_answers": {
            "what_core_can_do": [
                "Build CI-safe large-model sharding plans and evidence.",
                "Probe llama.cpp/GGUF/RPC runtime prerequisites.",
                "Profile devices from local probes or JSON imports.",
                "Plan layer placement, tensor split, KV reservation, and memory estimates.",
                "Run fixture/plan/real/import runner paths with benchmark and correctness evidence.",
                "Import live stage-selective Kaggle proof summaries for 7B multi-token and 14B dual-kernel validation.",
                "Export N-stage partition planning and stage-selective memory/latency/throughput/failure-recovery summaries.",
                "Expose serving-readiness hooks and next-layer integration contracts.",
            ],
            "real_verified": bool(real_verified),
            "fixture_diagnostic_or_import": "fixture/diagnostic unless real runner or real-run import is supplied",
            "control_layer_call": "Use core_technology_handoff_rc_v1.next_layer_integration_contract.control_layer and stable_entrypoints.",
            "user_layer_call": "Use safe status fields, streaming event schema, and bounded batch schema; do not expose raw generated text from public artifacts.",
            "permissions_trust_billing_dependencies": next_layer_contract.get("permissions_trust_billing_layer", {}).get("core_signals", []),
            "external_runtime_future_work": [
                "Provide real GGUF model files and controlled llama.cpp/vLLM/SGLang/TensorRT/Petals-like runtimes.",
                "Run bounded 7B/13B/70B external proofs on real consumer devices.",
                "Migrate the proven two-stage live runtime path to the N-stage scheduler/runtime abstraction.",
                "Optimize production throughput only after real runtime evidence exists.",
            ],
        },
        "boundary": {
            "core_technology_only": True,
            "inference_only": True,
            "not_training_or_finetuning": True,
            "not_permissions_accounts_billing": True,
            "not_incentives_staking_slashing": True,
            "not_public_p2p_nat_traversal": True,
            "not_production_petals_hivemind": True,
            "not_gpu_marketplace": True,
            "not_arbitrary_public_prompt_serving": True,
            "not_unbounded_kaggle_kernel_pooling": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
            "lease_material_public": False,
            "idempotency_material_public": False,
        },
        "diagnosis_codes": diagnosis_codes,
    }
    report["handoff_hash"] = stable_hash_payload({
        "schema": report["schema"],
        "inference": inference_report.get("schema"),
        "contract": next_layer_contract.get("contract_hash"),
        "runbook": deployment_runbook.get("runbook_hash"),
        "adapter": adapter_conformance.get("conformance_hash"),
        "large_model_stage_selective_evidence": stage_selective_evidence.get("schema"),
        "large_model_stage_selective_ready": stage_selective_ready,
    })
    errors = public_redaction_errors(report)
    if errors:
        report["ok"] = False
        report.setdefault("errors", []).extend(errors)
        report["safety"]["public_artifact_safe"] = False
        if "core_technology_public_artifact_redaction_failed" not in report["diagnosis_codes"]:
            report["diagnosis_codes"].append("core_technology_public_artifact_redaction_failed")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    stage_evidence = report.get("large_model_stage_selective_evidence") if isinstance(report.get("large_model_stage_selective_evidence"), dict) else {}
    stage_checks = stage_evidence.get("checks") if isinstance(stage_evidence.get("checks"), dict) else {}
    seven_b = stage_evidence.get("seven_b_live") if isinstance(stage_evidence.get("seven_b_live"), dict) else {}
    fourteen_b = stage_evidence.get("fourteen_b_live") if isinstance(stage_evidence.get("fourteen_b_live"), dict) else {}
    n_stage = stage_evidence.get("n_stage_partition") if isinstance(stage_evidence.get("n_stage_partition"), dict) else {}
    performance = stage_evidence.get("stage_selective_performance") if isinstance(stage_evidence.get("stage_selective_performance"), dict) else {}
    throughput = performance.get("throughput") if isinstance(performance.get("throughput"), dict) else {}
    memory = performance.get("memory") if isinstance(performance.get("memory"), dict) else {}
    latency = performance.get("latency") if isinstance(performance.get("latency"), dict) else {}
    lines = [
        "# CrowdTensor Core Technology Handoff RC",
        "",
        f"- Schema: `{report.get('schema')}`",
        f"- OK: `{bool(report.get('ok'))}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Real runtime verified: `{bool(report.get('real_runtime_verified'))}`",
        f"- Real 7B runtime verified: `{bool(report.get('real_7b_runtime_verified'))}`",
        f"- Stage-selective large-model Alpha ready: `{bool(report.get('core_technology_large_model_alpha_ready'))}`",
        f"- Evidence scope: `{report.get('evidence_scope')}`",
        f"- Output: `{report.get('output_dir')}`",
        "",
        "## Stable Entrypoints",
        "",
    ]
    for item in report.get("stable_entrypoints") or []:
        lines.append(f"- `{item}`")
    lines.extend([
        "",
        "## Stage-Selective Evidence",
        "",
        f"- 7B multi-token verified: `{bool(stage_checks.get('seven_b_multi_token_verified'))}` tokens=`{seven_b.get('generated_token_count')}`",
        f"- 14B dual-Kaggle verified: `{bool(stage_checks.get('fourteen_b_dual_kaggle_verified'))}` tokens=`{fourteen_b.get('generated_token_count')}`",
        f"- N-stage partition ready: `{bool(stage_checks.get('n_stage_partition_plan_ready'))}` stages=`{n_stage.get('target_stage_count')}`",
        f"- Performance report ready: `{bool(stage_checks.get('stage_selective_performance_report_ready'))}` token_per_second=`{throughput.get('tokens_per_second_effective')}`",
        f"- Stage weight download scope: stage0=`{memory.get('stage0_weight_download_scope')}` stage1=`{memory.get('stage1_weight_download_scope')}`",
        f"- Latency effective seconds: `{latency.get('effective_elapsed_seconds')}`",
    ])
    for item in stage_evidence.get("not_completed") or []:
        lines.append(f"- Stage-selective missing: `{item}`")
    lines.extend(["", "## Handoff Answers", ""])
    answers = report.get("handoff_answers") if isinstance(report.get("handoff_answers"), dict) else {}
    for item in answers.get("what_core_can_do") or []:
        lines.append(f"- {item}")
    lines.extend([
        "",
        f"- Control layer: {answers.get('control_layer_call')}",
        f"- User layer: {answers.get('user_layer_call')}",
        f"- Fixture/import scope: {answers.get('fixture_diagnostic_or_import')}",
        "",
        "## Blockers",
        "",
    ])
    for item in report.get("blockers") or ["none"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Diagnosis Codes", ""])
    for code in report.get("diagnosis_codes") or []:
        lines.append(f"- `{code}`")
    lines.append("")
    return "\n".join(lines)
