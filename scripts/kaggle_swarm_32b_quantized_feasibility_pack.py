#!/usr/bin/env python3
"""Build Kaggle Swarm 32B quantized feasibility RC evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_swarm_production_like_validation_pack as prod_like  # noqa: E402


SCHEMA = "kaggle_swarm_32b_quantized_feasibility_v1"
SUPPORT_BUNDLE_SCHEMA = "kaggle_swarm_32b_quantized_feasibility_support_bundle_v1"
STAGE_PACKAGE_PLAN_SCHEMA = "kaggle_swarm_32b_stage_package_plan_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-swarm-32b-quantized-feasibility"
DEFAULT_PRODUCTION_LIKE_REPORT = (
    "dist/gpu-swarm-production-like-validation-goal-r1/"
    "gpu_swarm_production_like_validation.json"
)
DEFAULT_CORE_STATUS_REPORT = prod_like.DEFAULT_CORE_STATUS_REPORT
DEFAULT_LARGE_MODEL_KAGGLE_REPORT = (
    "dist/large-model-kaggle-stage-selective-hf-7b-manual-rope-20260616/"
    "large_model_kaggle_validation.json"
)
DEFAULT_FRESH_32B_LIVE_PROBE_REPORT = (
    "dist/kaggle-32b-quantized-live-probe-summary/"
    "kaggle_32b_quantized_live_experiment_summary.json"
)
DEFAULT_FRESH_32B_STAGE_OWNED_LOADING_PROBE_REPORT = (
    "dist/kaggle-32b-stage-owned-safetensors-probe-awq-live-r3-clone/"
    "kaggle_32b_stage_owned_safetensors_probe.json"
)
DEFAULT_FRESH_32B_ACTIVATION_DECODE_PROBE_REPORT = (
    "dist/kaggle-32b-upper-bound-crossing-live-20260620-r3/"
    "kaggle_32b_stage_owned_activation_decode_probe.json"
)
EXECUTION_MODES = ("fixture", "evidence-import", "package", "external-existing", "kaggle-auto")
RUNTIME_ADAPTERS = (
    "hf-awq-stage-selective-kaggle",
    "gguf-llama-cpp-cuda",
    "hf-stage-selective-cuda",
    "hf-bitsandbytes",
    "exllama-v2",
    "vllm",
    "sglang",
)
BOUNDARIES = {
    "not_production": True,
    "not_p2p_nat_traversal": True,
    "not_arbitrary_public_prompt_serving": True,
    "not_billing": True,
    "not_unbounded_gpu_pooling": True,
    "not_32b_batch_requeue_or_production_success_by_default": True,
    "not_fresh_kaggle_run_by_default": True,
}
SENSITIVE_FRAGMENTS = prod_like.SENSITIVE_FRAGMENTS + (
    "KAGGLE_KEY=",
    "KAGGLE_USERNAME=",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "SOURCE_TARBALL_B64",
    "MINER_ENV_TEXT",
    "INLINE_KERNEL_PAYLOAD_B64",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def load_optional_report(path: Path) -> dict[str, Any]:
    return load_json(path) if path.is_file() else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def stable_hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    errors = [fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded]
    return sorted(set(errors))


def source_summary(path: Path, report: dict[str, Any], *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "present": path.is_file(),
        "schema": report.get("schema", ""),
        "ok": report.get("ok") is True,
        "sha256": sha256_file(path) if path.is_file() else "",
        "public_artifact_safe": True,
    }


def build_fresh_32b_live_probe_summary(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = _dict(report.get("aggregate"))
    experiments = []
    for item in _list(report.get("experiments")):
        if not isinstance(item, dict):
            continue
        experiments.append({
            "backend": item.get("backend"),
            "fresh_kaggle_run_performed": item.get("fresh_kaggle_run_performed") is True,
            "gpu_count": _int(item.get("gpu_count")),
            "gpu_names": [str(name) for name in _list(item.get("gpu_names"))],
            "downloaded_file_count": _int(item.get("downloaded_file_count")),
            "downloaded_mb": _int(item.get("downloaded_mb")),
            "probe_stage": item.get("probe_stage"),
            "probe_success": item.get("probe_success") is True,
            "one_token_generation_verified": item.get("one_token_generation_verified") is True,
            "cuda_build_ok": item.get("cuda_build_ok") is True,
            "cuda_build_duration_seconds": item.get("cuda_build_duration_seconds"),
            "kernel_deleted": item.get("kernel_deleted") is True,
            "private_package_removed": item.get("private_package_removed") is True,
            "public_artifact_safe": True,
        })
    fresh_runs = _int(aggregate.get("fresh_kaggle_runs"))
    return {
        "schema": "kaggle_swarm_32b_fresh_live_probe_summary_v1",
        "present": bool(report),
        "source_schema": report.get("schema", ""),
        "source_ok": report.get("ok") is True,
        "conclusion": report.get("conclusion", ""),
        "fresh_kaggle_run_performed": fresh_runs > 0,
        "fresh_kaggle_runs": fresh_runs,
        "gpu_hardware_verified": aggregate.get("gpu_hardware_verified") is True,
        "q2k_all_splits_downloaded": aggregate.get("q2k_all_splits_downloaded") is True,
        "largest_downloaded_mb": _int(aggregate.get("largest_downloaded_mb")),
        "cuda_source_build_verified": aggregate.get("cuda_source_build_verified") is True,
        "one_token_generation_verified": aggregate.get("one_token_generation_verified") is True,
        "kaggle_terminal_status": aggregate.get("kaggle_terminal_status", ""),
        "kaggle_log_signal": aggregate.get("kaggle_log_signal", ""),
        "blocked_at": aggregate.get("blocked_at", ""),
        "all_kernels_deleted": aggregate.get("all_kernels_deleted") is True,
        "all_private_packages_removed": aggregate.get("all_private_packages_removed") is True,
        "experiments": experiments,
        "public_artifact_safe": True,
    }


def build_fresh_32b_stage_owned_loading_probe_summary(report: dict[str, Any]) -> dict[str, Any]:
    stage_summaries = []
    for item in _list(report.get("stage_summaries")):
        if not isinstance(item, dict):
            continue
        stage_summaries.append({
            "stage_id": item.get("stage_id"),
            "stage_ok": item.get("stage_ok") is True,
            "gpu_verified": item.get("gpu_verified") is True,
            "gpu_count": _int(item.get("gpu_count")),
            "gpu_names": [str(name) for name in _list(item.get("gpu_names"))],
            "stage_layer_range": list(item.get("stage_layer_range") or []),
            "assigned_weight_key_count": _int(item.get("assigned_weight_key_count")),
            "assigned_weight_file_count": _int(item.get("assigned_weight_file_count")),
            "downloaded_file_count": _int(item.get("downloaded_file_count")),
            "loaded_weight_key_count": _int(item.get("loaded_weight_key_count")),
            "loaded_tensor_gb": _float(item.get("loaded_tensor_gb")),
            "materialize_clone_requested": item.get("materialize_clone_requested") is True,
            "materialized_weight_key_count": _int(item.get("materialized_weight_key_count")),
            "materialized_tensor_gb": _float(item.get("materialized_tensor_gb")),
            "retained_tensor_gb": _float(item.get("retained_tensor_gb")),
            "loads_only_stage_weight_keys": item.get("loads_only_stage_weight_keys") is True,
            "cross_stage_weight_keys_loaded": item.get("cross_stage_weight_keys_loaded") is True,
            "stage_weight_downloads_only_stage_files": item.get("stage_weight_downloads_only_stage_files") is True,
            "public_artifact_safe": True,
        })
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    ready = report.get("stage_owned_quantized_32b_loading_ready") is True
    return {
        "schema": "kaggle_swarm_32b_stage_owned_loading_probe_summary_v1",
        "present": bool(report),
        "source_schema": report.get("schema", ""),
        "source_ok": report.get("ok") is True,
        "fresh_kaggle_run_performed": report.get("fresh_kaggle_run_performed") is True,
        "stage_owned_quantized_32b_loading_ready": ready,
        "gpu_hardware_verified": report.get("gpu_hardware_verified") is True,
        "coverage_ready": report.get("coverage_ready") is True,
        "stage_owned_download_scope_ready": report.get("stage_owned_download_scope_ready") is True,
        "loads_only_stage_weight_keys_ready": report.get("loads_only_stage_weight_keys_ready") is True,
        "all_stage_reports_downloaded": report.get("all_stage_reports_downloaded") is True,
        "all_stage_owned_loading_ready": report.get("all_stage_owned_loading_ready") is True,
        "stage_count": _int(_dict(report.get("runtime")).get("stage_count")),
        "stage_ids": [int(value) for value in _list(_dict(report.get("runtime")).get("stage_ids")) if isinstance(value, int)],
        "model_repo": _dict(report.get("model")).get("repo", ""),
        "quantization_format": _dict(report.get("model")).get("quantization_format", ""),
        "one_token_generation_verified": _dict(report.get("runtime")).get("one_token_generation_verified") is True,
        "stage_owned_loading_only": _dict(report.get("runtime")).get("stage_owned_loading_only") is True,
        "all_kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "all_private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "stage_summaries": stage_summaries,
        "public_artifact_safe": True,
    }


def build_fresh_32b_activation_decode_probe_summary(report: dict[str, Any]) -> dict[str, Any]:
    stage_summaries = []
    for item in _list(report.get("stage_summaries")):
        if not isinstance(item, dict):
            continue
        stage_summaries.append({
            "mode": item.get("mode"),
            "stage_id": item.get("stage_id"),
            "stage_ok": item.get("ok") is True,
            "gpu_verified": item.get("gpu_verified") is True,
            "gpu_count": _int(item.get("gpu_count")),
            "gpu_names": [str(name) for name in _list(item.get("gpu_names"))],
            "stage_layer_range": list(item.get("stage_layer_range") or []),
            "assigned_weight_key_count": _int(item.get("assigned_weight_key_count")),
            "assigned_weight_file_count": _int(item.get("assigned_weight_file_count")),
            "loaded_weight_key_count": _int(item.get("loaded_weight_key_count")),
            "loaded_tensor_gb": _float(item.get("loaded_tensor_gb")),
            "awq_stage_model_prepared": item.get("awq_stage_model_prepared") is True,
            "activation_ready": item.get("activation_ready") is True,
            "stage1_decode_ready": item.get("stage1_decode_ready") is True,
            "generated_token_count": _int(item.get("generated_token_count")),
            "activation_hash": str(item.get("activation_hash") or ""),
            "output_hash": str(item.get("output_hash") or ""),
            "diagnosis_codes": [str(code) for code in _list(item.get("diagnosis_codes"))],
            "public_artifact_safe": True,
        })
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    safety = _dict(report.get("safety"))
    comparison = _dict(report.get("comparison"))
    single_kernel = _dict(report.get("single_kernel_baseline"))
    single_metrics = _dict(single_kernel.get("metrics"))
    return {
        "schema": "kaggle_swarm_32b_activation_decode_probe_summary_v1",
        "present": bool(report),
        "source_schema": report.get("schema", ""),
        "source_ok": report.get("ok") is True,
        "fresh_kaggle_run_performed": report.get("fresh_kaggle_run_performed") is True,
        "execution_mode": str(report.get("execution_mode") or ""),
        "coordinator_direct_management_verified": report.get("coordinator_direct_management_verified") is True,
        "upper_bound_crossing_verified": report.get("upper_bound_crossing_verified") is True
        or comparison.get("upper_bound_crossing_verified") is True,
        "cross_kernel_activation_decode_verified": report.get("cross_kernel_activation_decode_verified") is True,
        "one_token_generation_verified": report.get("one_token_generation_verified") is True,
        "multi_token_decode_verified": report.get("multi_token_decode_verified") is True,
        "stage_owned_awq_runtime_verified": report.get("stage_owned_awq_runtime_verified") is True,
        "activation_handoff_verified": report.get("activation_handoff_verified") is True,
        "gpu_hardware_verified": bool(stage_summaries) and all(item["gpu_verified"] for item in stage_summaries),
        "all_stage_reports_ready": bool(stage_summaries) and all(item["stage_ok"] for item in stage_summaries),
        "generated_token_count": max(
            [_int(report.get("generated_token_count"))]
            + [_int(item.get("generated_token_count")) for item in stage_summaries]
        ),
        "max_new_tokens": _int(report.get("max_new_tokens")),
        "stage_count": _int(_dict(report.get("model")).get("stage_count")),
        "split_index": _int(_dict(report.get("model")).get("split_index")),
        "model_repo": _dict(report.get("model")).get("repo", ""),
        "quantization": _dict(report.get("model")).get("quantization", ""),
        "all_kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "all_private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "private_activation_removed": lifecycle.get("private_activation_removed") is True,
        "two_kernel_generated_token_count": _int(comparison.get("two_kernel_generated_token_count")),
        "two_kernel_stability": str(comparison.get("two_kernel_stability") or ""),
        "two_kernel_stage_latency": _dict(comparison.get("two_kernel_stage_latency")),
        "two_kernel_stage_memory": _dict(comparison.get("two_kernel_stage_memory")),
        "single_kernel_attempted": comparison.get("single_kernel_attempted") is True,
        "single_kernel_ok": comparison.get("single_kernel_ok") is True,
        "single_kernel_generated_token_count": _int(comparison.get("single_kernel_generated_token_count")),
        "single_kernel_wall_time_seconds": _float(comparison.get("single_kernel_wall_time_seconds")),
        "single_kernel_tokens_per_second": _float(comparison.get("single_kernel_tokens_per_second")),
        "single_kernel_stage_memory": _dict(comparison.get("single_kernel_stage_memory")),
        "single_kernel_stability": str(comparison.get("single_kernel_stability") or ""),
        "single_kernel_blockers": [str(item) for item in _list(comparison.get("single_kernel_blockers"))],
        "single_kernel_stage0_loaded_tensor_gb": _float(_dict(single_kernel.get("stage0")).get("loaded_tensor_gb")),
        "single_kernel_stage1_loaded_tensor_gb": _float(_dict(single_kernel.get("stage1")).get("loaded_tensor_gb")),
        "single_kernel_metrics": {
            "generated_token_count": _int(single_metrics.get("generated_token_count")),
            "wall_time_seconds": _float(single_metrics.get("wall_time_seconds")),
            "tokens_per_second": _float(single_metrics.get("tokens_per_second")),
            "public_artifact_safe": True,
        },
        "activation_public": safety.get("activation_public") is True,
        "hidden_state_public": safety.get("hidden_state_public") is True,
        "generated_token_ids_public": safety.get("generated_token_ids_public") is True,
        "stage_summaries": stage_summaries,
        "public_artifact_safe": True,
    }


def fixture_production_like_report() -> dict[str, Any]:
    return {
        "schema": "gpu_swarm_production_like_validation_v1",
        "ok": True,
        "gpu_swarm_production_like_validation_ready": True,
        "production_like_workload_ready": True,
        "largest_successful_model_tier": "14b",
        "largest_attempted_model_tier": "32b",
        "multi_token_decode_ready": True,
        "batch_or_multi_request_ready": True,
        "stage_requeue_or_failure_recovery_ready": True,
        "fresh_gpu_run_performed": False,
        "external_runtime_verified": False,
        "retained_evidence_imported": True,
        "production_like_workload": {
            "generated_token_count": 16,
            "request_count": 2,
            "stage_requeue_or_failure_recovery_ready": True,
            "latency_throughput_summary_ready": True,
        },
        "larger_model_attempt": {
            "candidate_model_id": "Qwen/Qwen2.5-32B-Instruct",
            "largest_attempted_model_tier": "32b",
        },
        "safety": {"public_artifact_safe": True},
    }


def fixture_core_status_report() -> dict[str, Any]:
    return {
        "schema": "core_technology_validation_status_v1",
        "ok": True,
        "core_validation_ready": True,
        "largest_successful_tier": "14b",
        "handoff_stage_selective_evidence": {
            "seven_b_multi_token_verified": True,
            "seven_b_generated_token_count": 2,
            "fourteen_b_dual_kaggle_verified": True,
            "fourteen_b_generated_token_count": 1,
            "n_stage_partition_plan_ready": True,
            "stage_selective_performance_report_ready": True,
        },
        "seven_b_eight_b_evidence": {
            "real_7b_runtime_verified": True,
            "generated_token_count": 2,
            "memory_peak_mb": 14608,
        },
    }


def fixture_large_model_kaggle_report() -> dict[str, Any]:
    return {
        "schema": "large_model_kaggle_validation_v1",
        "ok": True,
        "real_7b_runtime_verified": True,
        "core_validation_ready": True,
        "gpu_runtime_verified": True,
        "largest_successful_tier": "7b",
        "hardware": {
            "gpu_count": 2,
            "gpu_names": ["NVIDIA Tesla T4", "NVIDIA Tesla T4"],
            "gpu_memory_total_mb": [15360, 15360],
        },
        "diagnosis_codes": [
            "large_model_kaggle_gpu_hardware_verified",
            "large_model_kaggle_gpu_runtime_verified",
            "large_model_7b_runtime_verified",
        ],
    }


def build_retained_evidence_summary(
    *,
    production_like: dict[str, Any],
    core_status: dict[str, Any],
    large_kaggle: dict[str, Any],
    execution_mode: str,
) -> dict[str, Any]:
    handoff = _dict(core_status.get("handoff_stage_selective_evidence"))
    seven = _dict(core_status.get("seven_b_eight_b_evidence"))
    prod_workload = _dict(production_like.get("production_like_workload"))
    retained_imported = bool(
        execution_mode in {"evidence-import", "package", "external-existing", "kaggle-auto"}
        and production_like.get("ok") is True
        and core_status.get("ok") is True
        and large_kaggle.get("ok") is True
    )
    return {
        "schema": "kaggle_swarm_32b_retained_evidence_summary_v1",
        "retained_evidence_imported": retained_imported,
        "production_like_ready": production_like.get("gpu_swarm_production_like_validation_ready") is True,
        "retained_largest_successful_model_tier": (
            production_like.get("largest_successful_model_tier")
            or core_status.get("largest_successful_tier")
            or large_kaggle.get("largest_successful_tier")
            or ""
        ),
        "retained_7b_kaggle_runtime_verified": bool(
            large_kaggle.get("real_7b_runtime_verified")
            or seven.get("real_7b_runtime_verified")
        ),
        "retained_14b_stage_selective_verified": bool(
            production_like.get("largest_successful_model_tier") == "14b"
            or core_status.get("largest_successful_tier") == "14b"
            or handoff.get("fourteen_b_dual_kaggle_verified")
        ),
        "retained_stage_owned_loading_ready": bool(
            handoff.get("n_stage_partition_plan_ready")
            or _dict(production_like.get("production_like_workload")).get("stage_owned_weight_loading_ready")
        ),
        "retained_one_token_generation_ready": bool(
            large_kaggle.get("real_7b_runtime_verified")
            or handoff.get("fourteen_b_generated_token_count")
        ),
        "retained_multi_token_generation_ready": bool(
            production_like.get("multi_token_decode_ready")
            and _int(prod_workload.get("generated_token_count")) >= 16
        ),
        "retained_batch_or_sequential_ready": bool(
            production_like.get("batch_or_multi_request_ready")
            and _int(prod_workload.get("request_count")) >= 2
        ),
        "retained_requeue_ready": bool(
            production_like.get("stage_requeue_or_failure_recovery_ready")
            or prod_workload.get("stage_requeue_or_failure_recovery_ready")
        ),
        "fresh_kaggle_run_performed": False,
        "external_runtime_verified": False,
        "public_artifact_safe": True,
    }


def build_kaggle_profile(args: argparse.Namespace, large_kaggle: dict[str, Any]) -> dict[str, Any]:
    hardware = _dict(large_kaggle.get("hardware"))
    gpu_names = hardware.get("gpu_names") if isinstance(hardware.get("gpu_names"), list) else []
    observed_gpu_count = _int(hardware.get("gpu_count"))
    observed_memory = hardware.get("gpu_memory_total_mb") if isinstance(hardware.get("gpu_memory_total_mb"), list) else []
    return {
        "schema": "kaggle_swarm_gpu_profile_assumption_v1",
        "assumed_gpu_type": args.kaggle_gpu_type,
        "assumed_gpu_count": args.gpu_count,
        "available_vram_per_gpu_mb": args.available_vram_per_gpu_mb,
        "available_total_vram_mb": args.gpu_count * args.available_vram_per_gpu_mb,
        "simultaneous_gpu_kernel_limit": args.simultaneous_kaggle_gpu_kernel_limit,
        "observed_retained_gpu_count": observed_gpu_count,
        "observed_retained_gpu_names": [str(item) for item in gpu_names],
        "observed_retained_gpu_memory_mb": [_int(item) for item in observed_memory],
        "quota_boundary": "Kaggle GPU quota and accelerator availability are account and time dependent; this RC does not bypass quota.",
        "runtime_boundary": "Kaggle notebooks are treated as temporary external stage Miners, not production serving infrastructure.",
        "internet_boundary": "Model download/build steps require Kaggle internet access and can fail independently of CrowdTensor routing.",
        "profile_source": "retained Kaggle evidence plus explicit CLI assumptions",
        "public_artifact_safe": True,
    }


def build_quantized_runtime_plan(args: argparse.Namespace) -> dict[str, Any]:
    adapters = [
        {
            "adapter": "hf-awq-stage-selective-kaggle",
            "repo_status": "fresh-cross-kernel-32b-awq-activation-decode-proof-retained",
            "quantized_32b_stage_status": "integrated",
            "selected": args.runtime_adapter == "hf-awq-stage-selective-kaggle",
            "public_artifact_safe": True,
        },
        {
            "adapter": "gguf-llama-cpp-cuda",
            "repo_status": "partial-large-model-kaggle-validation-path-exists",
            "quantized_32b_stage_status": "not-integrated-as-coordinator-backed-kaggle-stage-miner",
            "selected": args.runtime_adapter == "gguf-llama-cpp-cuda",
            "public_artifact_safe": True,
        },
        {
            "adapter": "hf-stage-selective-cuda",
            "repo_status": "retained-7b-14b-stage-selective-path-exists",
            "quantized_32b_stage_status": "not-quantized-32b-runtime",
            "selected": args.runtime_adapter == "hf-stage-selective-cuda",
            "public_artifact_safe": True,
        },
        {
            "adapter": "hf-bitsandbytes",
            "repo_status": "candidate-only",
            "quantized_32b_stage_status": "not-implemented-for-stage-owned-kaggle-miners",
            "selected": args.runtime_adapter == "hf-bitsandbytes",
            "public_artifact_safe": True,
        },
        {
            "adapter": "exllama-v2",
            "repo_status": "candidate-only",
            "quantized_32b_stage_status": "not-implemented",
            "selected": args.runtime_adapter == "exllama-v2",
            "public_artifact_safe": True,
        },
        {
            "adapter": "vllm",
            "repo_status": "candidate-only",
            "quantized_32b_stage_status": "not-integrated-for-cross-kernel-stage-sharding",
            "selected": args.runtime_adapter == "vllm",
            "public_artifact_safe": True,
        },
        {
            "adapter": "sglang",
            "repo_status": "candidate-only",
            "quantized_32b_stage_status": "not-integrated-for-cross-kernel-stage-sharding",
            "selected": args.runtime_adapter == "sglang",
            "public_artifact_safe": True,
        },
    ]
    selected = next((item for item in adapters if item["selected"]), adapters[0])
    integrated = selected["quantized_32b_stage_status"] == "integrated"
    return {
        "schema": "kaggle_swarm_32b_quantized_runtime_plan_v1",
        "candidate_32b_model_selected": True,
        "candidate_model_id": args.candidate_model_id,
        "candidate_model_tier": args.candidate_model_tier,
        "candidate_parameter_count_b": args.candidate_parameter_count_b,
        "quantized_runtime_plan_ready": True,
        "quantization": {
            "format": args.quantized_format,
            "weight_bits": args.quantization_bits,
            "metadata_overhead_percent": args.quantization_metadata_overhead_percent,
            "assumption": "public feasibility estimate only; no model weights are downloaded by default",
            "public_artifact_safe": True,
        },
        "selected_runtime_adapter": selected["adapter"],
        "selected_runtime_adapter_integrated_for_32b_kaggle_swarm": integrated,
        "runtime_adapter_candidates": adapters,
        "runtime_adapter_blocker": "" if integrated else "quantized_32b_stage_runtime_not_integrated",
        "public_artifact_safe": True,
    }


def build_stage_partition_plan(
    args: argparse.Namespace,
    *,
    kaggle_profile: dict[str, Any],
    runtime_plan: dict[str, Any],
) -> dict[str, Any]:
    stage_count = max(1, args.stage_count)
    params = _float(args.candidate_parameter_count_b)
    bits = _float(args.quantization_bits)
    weight_mb_total = int(params * 1_000_000_000 * (bits / 8.0) / (1024 * 1024))
    metadata_mb_total = int(weight_mb_total * (_float(args.quantization_metadata_overhead_percent) / 100.0))
    total_weight_with_metadata_mb = weight_mb_total + metadata_mb_total
    weight_mb_per_stage = int(total_weight_with_metadata_mb / stage_count)
    kv_total_mb = int(
        args.context_length
        * args.candidate_layer_count
        * args.candidate_hidden_size
        * 2
        * args.kv_cache_bytes_per_element
        / (1024 * 1024)
    )
    kv_mb_per_stage = int(kv_total_mb / stage_count)
    activation_mb_per_token = (
        args.candidate_hidden_size
        * args.activation_bytes_per_element
        / (1024 * 1024)
    )
    activation_transfer_mb_per_request = activation_mb_per_token * args.target_max_new_tokens * max(1, args.batch_request_target)
    required_vram_per_stage_mb = (
        weight_mb_per_stage
        + kv_mb_per_stage
        + args.runtime_overhead_mb_per_stage
        + args.fragmentation_margin_mb_per_stage
        + args.package_overhead_mb_per_stage
    )
    available_per_gpu = _int(kaggle_profile.get("available_vram_per_gpu_mb"))
    gpu_count = _int(kaggle_profile.get("assumed_gpu_count"))
    simultaneous_limit = _int(kaggle_profile.get("simultaneous_gpu_kernel_limit"))
    memory_feasible = required_vram_per_stage_mb <= available_per_gpu
    topology_feasible = stage_count <= gpu_count and stage_count <= simultaneous_limit
    return {
        "schema": "kaggle_swarm_32b_stage_partition_plan_v1",
        "stage_partition_plan_ready": True,
        "stage_count": stage_count,
        "stage_roles": [f"stage{index}" for index in range(stage_count)],
        "partition_mode": "pipeline-stage-owned-quantized-weights",
        "runtime_adapter": runtime_plan.get("selected_runtime_adapter"),
        "per_stage_memory_estimate_ready": True,
        "memory_estimate": {
            "quantized_weight_mb_total": weight_mb_total,
            "quantization_metadata_mb_total": metadata_mb_total,
            "weight_plus_metadata_mb_total": total_weight_with_metadata_mb,
            "weight_plus_metadata_mb_per_stage": weight_mb_per_stage,
            "kv_cache_mb_total": kv_total_mb,
            "kv_cache_mb_per_stage": kv_mb_per_stage,
            "runtime_overhead_mb_per_stage": args.runtime_overhead_mb_per_stage,
            "fragmentation_margin_mb_per_stage": args.fragmentation_margin_mb_per_stage,
            "package_overhead_mb_per_stage": args.package_overhead_mb_per_stage,
            "required_vram_mb_per_stage": required_vram_per_stage_mb,
            "available_vram_mb_per_gpu": available_per_gpu,
            "margin_mb_per_stage": available_per_gpu - required_vram_per_stage_mb,
            "memory_feasible_on_assumed_profile": memory_feasible,
            "public_artifact_safe": True,
        },
        "activation_transfer_estimate_ready": True,
        "activation_transfer_estimate": {
            "hidden_size": args.candidate_hidden_size,
            "activation_bytes_per_element": args.activation_bytes_per_element,
            "activation_mb_per_token": round(activation_mb_per_token, 6),
            "target_max_new_tokens": args.target_max_new_tokens,
            "batch_request_target": args.batch_request_target,
            "estimated_stage_handoff_mb_per_request": round(activation_transfer_mb_per_request, 6),
            "raw_activations_public": False,
            "public_artifact_safe": True,
        },
        "kaggle_topology_feasible": topology_feasible,
        "stage_package_count_required": stage_count,
        "public_artifact_safe": True,
    }


def build_topology_plan(args: argparse.Namespace, *, partition: dict[str, Any]) -> dict[str, Any]:
    stage_roles = [str(item) for item in _list(partition.get("stage_roles"))]
    stages = []
    for role in stage_roles:
        stages.append({
            "stage": role,
            "kernel_role": "private-kaggle-gpu-stage-miner",
            "accelerator": args.kaggle_gpu_type,
            "requires_public_coordinator_url": True,
            "requires_private_runtime_token": True,
            "package_metadata_only": True,
            "inline_payload_public": False,
            "public_artifact_safe": True,
        })
    topology_ready = bool(stages and len(stages) <= args.simultaneous_kaggle_gpu_kernel_limit)
    return {
        "schema": "kaggle_swarm_multi_kernel_topology_v1",
        "kaggle_multi_kernel_topology_ready": topology_ready,
        "coordinator": {
            "role": "operator-owned-public-or-tunneled-coordinator",
            "runs_on_kaggle": False,
            "public_endpoint_required_for_kaggle_miners": True,
            "public_artifact_safe": True,
        },
        "stage_miners": stages,
        "simultaneous_gpu_kernel_limit": args.simultaneous_kaggle_gpu_kernel_limit,
        "kaggle_stage_package_plan_ready": bool(stages),
        "private_package_payloads_written": False,
        "package_plan_note": "This RC writes public metadata only; private Kaggle source payloads must be generated in a separate temporary local-only step.",
        "public_artifact_safe": True,
    }


def build_stage_package_plan(args: argparse.Namespace, *, topology: dict[str, Any], partition: dict[str, Any]) -> dict[str, Any]:
    stages = []
    for stage in _list(topology.get("stage_miners")):
        if isinstance(stage, dict):
            stage_name = str(stage.get("stage"))
            stages.append({
                "stage": stage_name,
                "package_slug_hint": f"{args.package_slug_prefix}-{stage_name}",
                "contains_private_payload": False,
                "contains_inline_kernel_payload": False,
                "contains_private_env": False,
                "metadata_hash": stable_hash_payload({
                    "stage": stage_name,
                    "candidate_model_id": args.candidate_model_id,
                    "runtime_adapter": args.runtime_adapter,
                    "quantized_format": args.quantized_format,
                }),
                "public_artifact_safe": True,
            })
    return {
        "schema": STAGE_PACKAGE_PLAN_SCHEMA,
        "kaggle_stage_package_plan_ready": bool(stages),
        "candidate_model_id": args.candidate_model_id,
        "candidate_model_tier": args.candidate_model_tier,
        "runtime_adapter": args.runtime_adapter,
        "quantized_format": args.quantized_format,
        "stage_count": partition.get("stage_count"),
        "stage_packages": stages,
        "private_artifact_cleanup_plan": {
            "schema": "kaggle_swarm_32b_private_cleanup_plan_v1",
            "cleanup_required_after_fresh_run": True,
            "delete_private_kaggle_kernels": True,
            "delete_local_private_payloads": True,
            "rotate_runtime_tokens": True,
            "public_artifact_safe": True,
        },
        "public_artifact_safe": True,
    }


def build_evidence_validation(
    *,
    args: argparse.Namespace,
    retained: dict[str, Any],
    runtime_plan: dict[str, Any],
    partition: dict[str, Any],
    topology: dict[str, Any],
    fresh_probe: dict[str, Any],
    stage_owned_probe: dict[str, Any],
    activation_decode_probe: dict[str, Any],
) -> dict[str, Any]:
    memory = _dict(partition.get("memory_estimate"))
    memory_feasible = memory.get("memory_feasible_on_assumed_profile") is True
    runtime_integrated = runtime_plan.get("selected_runtime_adapter_integrated_for_32b_kaggle_swarm") is True
    topology_ready = topology.get("kaggle_multi_kernel_topology_ready") is True
    stage_owned_loading_verified = stage_owned_probe.get("stage_owned_quantized_32b_loading_ready") is True
    activation_decode_verified = activation_decode_probe.get("cross_kernel_activation_decode_verified") is True
    activation_decode_one_token = activation_decode_probe.get("one_token_generation_verified") is True
    activation_decode_multi_token = activation_decode_probe.get("multi_token_decode_verified") is True
    coordinator_direct = activation_decode_probe.get("coordinator_direct_management_verified") is True
    upper_bound_crossing = activation_decode_probe.get("upper_bound_crossing_verified") is True
    stage_owned_loading_feasible = bool(
        stage_owned_loading_verified
        or activation_decode_verified
        or (memory_feasible and runtime_integrated and topology_ready)
    )
    fresh_kaggle_run_performed = bool(
        (
            args.fresh_kaggle_run_performed
            and args.execution_mode in {"external-existing", "kaggle-auto"}
        )
        or fresh_probe.get("fresh_kaggle_run_performed") is True
        or stage_owned_probe.get("fresh_kaggle_run_performed") is True
        or activation_decode_probe.get("fresh_kaggle_run_performed") is True
    )
    one_token_verified = bool(
        fresh_probe.get("one_token_generation_verified") is True
        or activation_decode_one_token
    )
    generation_feasible = bool(stage_owned_loading_feasible and one_token_verified)
    return {
        "schema": "kaggle_swarm_32b_evidence_validation_plan_v1",
        "stage_owned_loading_feasible": stage_owned_loading_feasible,
        "fresh_32b_stage_owned_loading_verified": stage_owned_loading_verified,
        "fresh_32b_activation_decode_verified": activation_decode_verified,
        "fresh_32b_activation_handoff_verified": activation_decode_probe.get("activation_handoff_verified") is True,
        "fresh_32b_stage_owned_awq_runtime_verified": activation_decode_probe.get("stage_owned_awq_runtime_verified") is True,
        "fresh_32b_activation_decode_generated_token_count": _int(activation_decode_probe.get("generated_token_count")),
        "fresh_32b_activation_decode_private_activation_removed": activation_decode_probe.get("private_activation_removed") is True,
        "fresh_32b_multi_token_decode_verified": activation_decode_multi_token,
        "fresh_32b_coordinator_direct_management_verified": coordinator_direct,
        "fresh_32b_upper_bound_crossing_verified": upper_bound_crossing,
        "fresh_32b_single_kernel_baseline_attempted": activation_decode_probe.get("single_kernel_attempted") is True,
        "fresh_32b_single_kernel_baseline_ok": activation_decode_probe.get("single_kernel_ok") is True,
        "fresh_32b_single_kernel_generated_token_count": _int(activation_decode_probe.get("single_kernel_generated_token_count")),
        "fresh_32b_single_kernel_wall_time_seconds": _float(activation_decode_probe.get("single_kernel_wall_time_seconds")),
        "fresh_32b_stage_owned_gpu_hardware_verified": stage_owned_probe.get("gpu_hardware_verified") is True,
        "fresh_32b_stage_owned_download_scope_ready": stage_owned_probe.get("stage_owned_download_scope_ready") is True,
        "fresh_32b_loads_only_stage_weight_keys_ready": stage_owned_probe.get("loads_only_stage_weight_keys_ready") is True,
        "fresh_32b_stage_owned_clone_verified": all(
            isinstance(item, dict)
            and item.get("materialize_clone_requested") is True
            and _int(item.get("materialized_weight_key_count")) == _int(item.get("assigned_weight_key_count"))
            for item in _list(stage_owned_probe.get("stage_summaries"))
        ) if stage_owned_probe.get("stage_summaries") else False,
        "one_token_generation_feasible": generation_feasible,
        "multi_token_generation_feasible": bool(generation_feasible and activation_decode_multi_token and coordinator_direct),
        "coordinator_direct_management_feasible": bool(generation_feasible and coordinator_direct),
        "upper_bound_crossing_feasible": bool(generation_feasible and coordinator_direct and upper_bound_crossing),
        "batch_or_sequential_request_feasible": False,
        "stage_requeue_feasible": False,
        "retained_stage_owned_loading_ready": retained.get("retained_stage_owned_loading_ready") is True,
        "retained_one_token_generation_ready": retained.get("retained_one_token_generation_ready") is True,
        "retained_multi_token_generation_ready": retained.get("retained_multi_token_generation_ready") is True,
        "retained_batch_or_sequential_ready": retained.get("retained_batch_or_sequential_ready") is True,
        "retained_requeue_ready": retained.get("retained_requeue_ready") is True,
        "fresh_kaggle_run_performed": fresh_kaggle_run_performed,
        "fresh_32b_gpu_hardware_verified": fresh_probe.get("gpu_hardware_verified") is True,
        "fresh_32b_q2k_all_splits_downloaded": fresh_probe.get("q2k_all_splits_downloaded") is True,
        "fresh_32b_cuda_source_build_verified": fresh_probe.get("cuda_source_build_verified") is True,
        "fresh_32b_blocked_at": fresh_probe.get("blocked_at", ""),
        "fresh_32b_kaggle_log_signal": fresh_probe.get("kaggle_log_signal", ""),
        "external_runtime_verified": bool(fresh_kaggle_run_performed and generation_feasible),
        "public_artifact_safe": True,
    }


def build_blocker_details(
    *,
    args: argparse.Namespace,
    runtime_plan: dict[str, Any],
    partition: dict[str, Any],
    topology: dict[str, Any],
    evidence: dict[str, Any],
    fresh_probe: dict[str, Any],
    stage_owned_probe: dict[str, Any],
    activation_decode_probe: dict[str, Any],
) -> dict[str, Any]:
    memory = _dict(partition.get("memory_estimate"))
    activation_decode_verified = activation_decode_probe.get("cross_kernel_activation_decode_verified") is True
    runtime_blocked = bool(
        runtime_plan.get("selected_runtime_adapter_integrated_for_32b_kaggle_swarm") is not True
        and not activation_decode_verified
    )
    memory_blocked = memory.get("memory_feasible_on_assumed_profile") is not True
    topology_blocked = topology.get("kaggle_multi_kernel_topology_ready") is not True
    stage_owned_loading_verified = stage_owned_probe.get("stage_owned_quantized_32b_loading_ready") is True
    missing_live = evidence.get("fresh_kaggle_run_performed") is not True
    fresh_generation_blocked = bool(
        evidence.get("fresh_kaggle_run_performed") is True
        and evidence.get("one_token_generation_feasible") is not True
    )
    fresh_download_measured = fresh_probe.get("q2k_all_splits_downloaded") is True
    fresh_cuda_build_measured = fresh_probe.get("cuda_source_build_verified") is True
    blockers = {
        "runtime_adapter": {
            "blocked": runtime_blocked,
            "reason": runtime_plan.get("runtime_adapter_blocker") if runtime_blocked else "",
            "operator_action": (
                "integrate stage-owned 32B activation/decode execution on top of verified AWQ safetensors loading"
                if runtime_blocked and stage_owned_loading_verified
                else "integrate a quantized 32B stage-owned runtime adapter for Kaggle Miners"
                if runtime_blocked
                else ""
            ),
            "public_artifact_safe": True,
        },
        "vram": {
            "blocked": bool(memory_blocked and not stage_owned_loading_verified),
            "required_vram_mb_per_stage": memory.get("required_vram_mb_per_stage"),
            "available_vram_mb_per_gpu": memory.get("available_vram_mb_per_gpu"),
            "margin_mb_per_stage": memory.get("margin_mb_per_stage"),
            "operator_action": "reduce runtime overhead, increase stage count, or use larger-VRAM GPUs" if memory_blocked and not stage_owned_loading_verified else "",
            "public_artifact_safe": True,
        },
        "model_format": {
            "blocked": bool(fresh_download_measured is not True and not stage_owned_loading_verified),
            "reason": "32b_quantized_model_artifact_not_materialized_in_repo" if fresh_download_measured is not True and not stage_owned_loading_verified else "",
            "operator_action": "prepare or reference a public-safe quantized model artifact manifest" if fresh_download_measured is not True and not stage_owned_loading_verified else "",
            "public_artifact_safe": True,
        },
        "kaggle_quota": {
            "blocked": missing_live,
            "reason": "fresh_kaggle_gpu_run_not_performed" if missing_live else "",
            "operator_action": "run kaggle-auto only with available GPU quota and cleanup/token rotation" if missing_live else "",
            "public_artifact_safe": True,
        },
        "download_build_time": {
            "blocked": bool(
                missing_live
                or (
                    fresh_download_measured is not True
                    and not stage_owned_loading_verified
                )
                or (
                    fresh_cuda_build_measured is not True
                    and not stage_owned_loading_verified
                )
            ),
            "reason": "fresh_model_download_and_runtime_build_not_measured" if missing_live else (
                "fresh_32b_download_or_cuda_build_not_verified"
                if (
                    fresh_download_measured is not True
                    or (fresh_cuda_build_measured is not True and not stage_owned_loading_verified)
                )
                else ""
            ),
            "operator_action": "measure bounded download/build time in a fresh private Kaggle run" if missing_live else "",
            "public_artifact_safe": True,
        },
        "fresh_32b_generation": {
            "blocked": fresh_generation_blocked,
            "reason": (
                f"fresh_32b_{fresh_probe.get('blocked_at') or 'generation'}_blocked_by_{fresh_probe.get('kaggle_log_signal') or 'runtime_failure'}"
                if fresh_generation_blocked
                else ""
            ),
            "operator_action": "implement cross-kernel 32B AWQ activation/decode execution now that stage-owned loading is verified" if fresh_generation_blocked else "",
            "public_artifact_safe": True,
        },
        "activation_transfer": {
            "blocked": False,
            "reason": "estimated_only_no_raw_activation_payloads_public",
            "operator_action": "measure live stage handoff bytes after 1-token path is available",
            "public_artifact_safe": True,
        },
        "stage_partitioning": {
            "blocked": topology_blocked,
            "reason": "stage_count_exceeds_assumed_kaggle_kernel_limit" if topology_blocked else "",
            "operator_action": "use a permitted stage count and keep distinct stage Miners" if topology_blocked else "",
            "public_artifact_safe": True,
        },
        "missing_live_hardware": {
            "blocked": bool(missing_live or evidence.get("external_runtime_verified") is not True),
            "reason": (
                "fresh_kaggle_run_not_performed"
                if missing_live
                else "fresh_32b_generation_external_runtime_not_verified"
                if evidence.get("external_runtime_verified") is not True
                else ""
            ),
            "operator_action": "verify an already running public Coordinator plus private Kaggle stage Miners" if missing_live else "",
            "public_artifact_safe": True,
        },
    }
    if evidence.get("upper_bound_crossing_feasible") is True:
        blocked_reason = ""
    elif evidence.get("multi_token_generation_feasible") is True:
        blocked_reason = ""
    elif evidence.get("one_token_generation_feasible") is True:
        blocked_reason = ""
    elif stage_owned_loading_verified and runtime_blocked:
        blocked_reason = "quantized_32b_stage_owned_loading_verified_generation_runtime_missing"
    elif runtime_blocked and memory_blocked:
        blocked_reason = "quantized_32b_runtime_and_memory_blocked_on_assumed_kaggle_profile"
    elif runtime_blocked:
        blocked_reason = "quantized_32b_runtime_adapter_not_integrated"
    elif memory_blocked:
        blocked_reason = "quantized_32b_memory_margin_negative_on_assumed_kaggle_profile"
    elif fresh_generation_blocked:
        blocked_reason = "fresh_kaggle_32b_one_token_generation_killed"
    elif missing_live:
        blocked_reason = "fresh_kaggle_32b_runtime_not_verified"
    else:
        blocked_reason = ""
    if evidence.get("upper_bound_crossing_feasible") is True:
        verdict = "feasible_32b_upper_bound_crossing_rc"
    elif evidence.get("multi_token_generation_feasible") is True:
        verdict = "feasible_32b_multitoken_coordinator_rc"
    elif evidence.get("one_token_generation_feasible") is True:
        verdict = "feasible_32b_one_token_cross_kernel_rc"
    else:
        verdict = "feasible_for_fresh_attempt" if not blocked_reason else "blocked_current_repo_or_kaggle_profile"
    return {
        "schema": "kaggle_swarm_32b_blocker_details_v1",
        "feasibility_verdict": verdict,
        "blocked_reason": blocked_reason,
        "blocker_details": blockers,
        "hard_limits": {
            "max_fresh_model_attempts": args.max_fresh_model_attempts,
            "max_requeue_attempts": args.max_requeue_attempts,
            "single_attempt_timeout_minutes": args.max_attempt_timeout_minutes,
            "public_artifact_safe": True,
        },
        "public_artifact_safe": True,
    }


def artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "kaggle_swarm_32b_quantized_feasibility_artifact_summary_v1",
        "artifact_count": len(artifacts),
        "present_artifact_count": sum(1 for item in artifacts.values() if item.get("present")),
        "inspect_first": (artifacts.get("summary_markdown") or {}).get("path", ""),
        "support_bundle": (artifacts.get("support_bundle_json") or {}).get("path", ""),
        "stage_package_plan": (artifacts.get("stage_package_plan_json") or {}).get("path", ""),
        "public_artifact_safe": True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    runtime = _dict(report.get("quantized_runtime_plan"))
    partition = _dict(report.get("stage_partition_plan"))
    memory = _dict(partition.get("memory_estimate"))
    blockers = _dict(report.get("blocker_details"))
    lines = [
        "# Kaggle Swarm 32B Quantized Feasibility RC",
        "",
        f"- ready: `{report.get('kaggle_swarm_32b_quantized_feasibility_ready')}`",
        f"- verdict: `{report.get('feasibility_verdict')}`",
        f"- blocked reason: `{report.get('blocked_reason')}`",
        f"- execution mode: `{report.get('execution_mode')}`",
        f"- fresh Kaggle run performed: `{report.get('fresh_kaggle_run_performed')}`",
        f"- external runtime verified: `{report.get('external_runtime_verified')}`",
        f"- retained evidence imported: `{report.get('retained_evidence_imported')}`",
        f"- largest feasible model tier: `{report.get('largest_feasible_model_tier')}`",
        f"- largest attempted model tier: `{report.get('largest_attempted_model_tier')}`",
        "",
        "## Candidate",
        "",
        f"- model: `{runtime.get('candidate_model_id')}`",
        f"- tier: `{runtime.get('candidate_model_tier')}`",
        f"- runtime adapter: `{runtime.get('selected_runtime_adapter')}`",
        f"- quantized format: `{_dict(runtime.get('quantization')).get('format')}`",
        "",
        "## Stage Plan",
        "",
        f"- stage count: `{partition.get('stage_count')}`",
        f"- required VRAM per stage MB: `{memory.get('required_vram_mb_per_stage')}`",
        f"- available VRAM per GPU MB: `{memory.get('available_vram_mb_per_gpu')}`",
        f"- margin MB per stage: `{memory.get('margin_mb_per_stage')}`",
        f"- activation transfer estimate ready: `{report.get('activation_transfer_estimate_ready')}`",
        "",
        "## Blockers",
        "",
    ]
    for name, item in sorted(blockers.items()):
        if isinstance(item, dict):
            lines.append(f"- {name}: blocked=`{item.get('blocked')}` reason=`{item.get('reason')}` action=`{item.get('operator_action')}`")
    lines.extend(["", "## Next Operator Actions", ""])
    for item in report.get("next_operator_actions") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Boundaries", ""])
    for name, value in sorted(_dict(report.get("boundaries")).items()):
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Diagnosis", "", "- " + ", ".join(report.get("diagnosis_codes") or []), ""])
    return "\n".join(lines)


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "generated_at": report.get("generated_at"),
        "ok": report.get("ok") is True,
        "kaggle_swarm_32b_quantized_feasibility_ready": report.get("kaggle_swarm_32b_quantized_feasibility_ready") is True,
        "feasibility_verdict": report.get("feasibility_verdict"),
        "blocked_reason": report.get("blocked_reason"),
        "largest_feasible_model_tier": report.get("largest_feasible_model_tier"),
        "largest_attempted_model_tier": report.get("largest_attempted_model_tier"),
        "execution_mode": report.get("execution_mode"),
        "fresh_kaggle_run_performed": report.get("fresh_kaggle_run_performed") is True,
        "external_runtime_verified": report.get("external_runtime_verified") is True,
        "retained_evidence_imported": report.get("retained_evidence_imported") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "artifact_summary": report.get("artifact_summary") or {},
        "blocker_details": report.get("blocker_details") or {},
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    production_like_path = Path(args.production_like_report)
    core_status_path = Path(args.core_status_report)
    large_kaggle_path = Path(args.large_model_kaggle_report)
    fresh_probe_path = Path(args.fresh_32b_live_probe_report)
    stage_owned_probe_path = Path(args.fresh_32b_stage_owned_loading_probe_report)
    activation_decode_probe_path = Path(args.fresh_32b_activation_decode_probe_report)
    if args.execution_mode == "fixture":
        production_like_report = fixture_production_like_report()
        core_status_report = fixture_core_status_report()
        large_kaggle_report = fixture_large_model_kaggle_report()
        fresh_probe_report: dict[str, Any] = {}
        stage_owned_probe_report: dict[str, Any] = {}
        activation_decode_probe_report: dict[str, Any] = {}
    else:
        production_like_report = load_optional_report(production_like_path)
        core_status_report = load_optional_report(core_status_path)
        large_kaggle_report = load_optional_report(large_kaggle_path)
        fresh_probe_report = load_optional_report(fresh_probe_path)
        stage_owned_probe_report = load_optional_report(stage_owned_probe_path)
        activation_decode_probe_report = load_optional_report(activation_decode_probe_path)
    fresh_probe = build_fresh_32b_live_probe_summary(fresh_probe_report)
    stage_owned_probe = build_fresh_32b_stage_owned_loading_probe_summary(stage_owned_probe_report)
    activation_decode_probe = build_fresh_32b_activation_decode_probe_summary(activation_decode_probe_report)

    retained = build_retained_evidence_summary(
        production_like=production_like_report,
        core_status=core_status_report,
        large_kaggle=large_kaggle_report,
        execution_mode=args.execution_mode,
    )
    kaggle_profile = build_kaggle_profile(args, large_kaggle_report)
    runtime_plan = build_quantized_runtime_plan(args)
    partition = build_stage_partition_plan(args, kaggle_profile=kaggle_profile, runtime_plan=runtime_plan)
    topology = build_topology_plan(args, partition=partition)
    stage_package_plan = build_stage_package_plan(args, topology=topology, partition=partition)
    evidence = build_evidence_validation(
        args=args,
        retained=retained,
        runtime_plan=runtime_plan,
        partition=partition,
        topology=topology,
        fresh_probe=fresh_probe,
        stage_owned_probe=stage_owned_probe,
        activation_decode_probe=activation_decode_probe,
    )
    blockers = build_blocker_details(
        args=args,
        runtime_plan=runtime_plan,
        partition=partition,
        topology=topology,
        evidence=evidence,
        fresh_probe=fresh_probe,
        stage_owned_probe=stage_owned_probe,
        activation_decode_probe=activation_decode_probe,
    )

    if evidence.get("upper_bound_crossing_feasible"):
        retained_largest = "32b-quantized-4stage-upper-bound-rc"
    elif evidence.get("multi_token_generation_feasible"):
        retained_largest = "32b-quantized-2token-rc"
    elif evidence.get("one_token_generation_feasible"):
        retained_largest = "32b-quantized-1token"
    else:
        retained_largest = retained.get("retained_largest_successful_model_tier") or "unknown"
    fresh_kaggle_run_performed = evidence.get("fresh_kaggle_run_performed") is True
    external_runtime_verified = evidence.get("external_runtime_verified") is True
    ready = bool(
        runtime_plan.get("candidate_32b_model_selected")
        and runtime_plan.get("quantized_runtime_plan_ready")
        and topology.get("kaggle_multi_kernel_topology_ready")
        and partition.get("stage_partition_plan_ready")
        and partition.get("per_stage_memory_estimate_ready")
        and partition.get("activation_transfer_estimate_ready")
        and stage_package_plan.get("kaggle_stage_package_plan_ready")
        and blockers.get("feasibility_verdict")
        and args.max_fresh_model_attempts <= 2
        and args.max_requeue_attempts <= 1
        and args.max_attempt_timeout_minutes <= 60
    )
    diagnosis_codes = {
        "kaggle_swarm_32b_quantized_feasibility_ready" if ready else "kaggle_swarm_32b_quantized_feasibility_blocked",
        "candidate_32b_model_selected",
        "quantized_runtime_plan_ready",
        "kaggle_multi_kernel_topology_ready" if topology.get("kaggle_multi_kernel_topology_ready") else "kaggle_multi_kernel_topology_blocked",
        "stage_partition_plan_ready",
        "per_stage_memory_estimate_ready",
        "activation_transfer_estimate_ready",
        "kaggle_stage_package_plan_ready" if stage_package_plan.get("kaggle_stage_package_plan_ready") else "kaggle_stage_package_plan_missing",
        "stage_owned_loading_feasible" if evidence.get("stage_owned_loading_feasible") else "stage_owned_loading_blocked",
        "one_token_generation_feasible" if evidence.get("one_token_generation_feasible") else "one_token_generation_blocked",
        "multi_token_generation_feasible" if evidence.get("multi_token_generation_feasible") else "multi_token_generation_blocked",
        "batch_or_sequential_request_feasible" if evidence.get("batch_or_sequential_request_feasible") else "batch_or_sequential_request_blocked",
        "stage_requeue_feasible" if evidence.get("stage_requeue_feasible") else "stage_requeue_blocked_until_generation_ready",
        "retained_evidence_imported" if retained.get("retained_evidence_imported") else "retained_evidence_not_imported",
        "fresh_kaggle_run_performed" if fresh_kaggle_run_performed else "fresh_kaggle_run_not_performed",
        "external_runtime_verified" if external_runtime_verified else "external_runtime_not_verified",
        "kaggle_swarm_32b_public_artifact_redaction_ready",
    }
    if fresh_probe.get("present"):
        diagnosis_codes.add("fresh_32b_live_probe_imported")
        diagnosis_codes.add("fresh_32b_gpu_hardware_verified" if fresh_probe.get("gpu_hardware_verified") else "fresh_32b_gpu_hardware_not_verified")
        diagnosis_codes.add("fresh_32b_q2k_all_splits_downloaded" if fresh_probe.get("q2k_all_splits_downloaded") else "fresh_32b_q2k_download_not_verified")
        diagnosis_codes.add("fresh_32b_cuda_source_build_verified" if fresh_probe.get("cuda_source_build_verified") else "fresh_32b_cuda_source_build_not_verified")
        if fresh_probe.get("one_token_generation_verified"):
            diagnosis_codes.add("fresh_32b_one_token_generation_verified")
        elif not evidence.get("one_token_generation_feasible"):
            diagnosis_codes.add("fresh_32b_one_token_generation_blocked")
        else:
            diagnosis_codes.add("fresh_32b_legacy_live_probe_one_token_blocked")
        if fresh_probe.get("kaggle_log_signal") == "Killed":
            diagnosis_codes.add("fresh_32b_kaggle_killed")
    if stage_owned_probe.get("present"):
        diagnosis_codes.add("fresh_32b_stage_owned_loading_probe_imported")
        diagnosis_codes.add(
            "fresh_32b_stage_owned_loading_verified"
            if stage_owned_probe.get("stage_owned_quantized_32b_loading_ready")
            else "fresh_32b_stage_owned_loading_not_verified"
        )
        diagnosis_codes.add(
            "fresh_32b_stage_owned_tensor_clone_verified"
            if evidence.get("fresh_32b_stage_owned_clone_verified")
            else "fresh_32b_stage_owned_tensor_clone_not_verified"
        )
        diagnosis_codes.add(
            "fresh_32b_loads_only_stage_weight_keys"
            if stage_owned_probe.get("loads_only_stage_weight_keys_ready")
            else "fresh_32b_stage_weight_key_scope_not_verified"
        )
    if activation_decode_probe.get("present"):
        diagnosis_codes.add("fresh_32b_activation_decode_probe_imported")
        diagnosis_codes.add(
            "fresh_32b_cross_kernel_activation_decode_verified"
            if activation_decode_probe.get("cross_kernel_activation_decode_verified")
            else "fresh_32b_cross_kernel_activation_decode_not_verified"
        )
        diagnosis_codes.add(
            "fresh_32b_one_token_generation_verified"
            if activation_decode_probe.get("one_token_generation_verified")
            else "fresh_32b_one_token_generation_blocked"
        )
        diagnosis_codes.add(
            "fresh_32b_multi_token_decode_verified"
            if activation_decode_probe.get("multi_token_decode_verified")
            else "fresh_32b_multi_token_decode_blocked"
        )
        diagnosis_codes.add(
            "fresh_32b_coordinator_direct_management_verified"
            if activation_decode_probe.get("coordinator_direct_management_verified")
            else "fresh_32b_coordinator_direct_management_missing"
        )
        diagnosis_codes.add(
            "fresh_32b_single_kernel_baseline_completed"
            if activation_decode_probe.get("single_kernel_ok")
            else "fresh_32b_single_kernel_baseline_missing_or_failed"
        )
        diagnosis_codes.add(
            "fresh_32b_upper_bound_crossing_verified"
            if activation_decode_probe.get("upper_bound_crossing_verified")
            else "fresh_32b_upper_bound_crossing_not_verified"
        )
        diagnosis_codes.add(
            "fresh_32b_private_activation_removed"
            if activation_decode_probe.get("private_activation_removed")
            else "fresh_32b_private_activation_cleanup_missing"
        )
    if evidence.get("upper_bound_crossing_feasible") is True:
        next_operator_actions = [
            "move the 4-stage upper-bound crossing proof into the production Coordinator/Miner task API",
            "add KV-cache reuse so 32B multi-token decode does not recompute the full prompt prefix each token",
            "add bounded batch/sequential request validation for the 4-stage 32B AWQ runtime",
            "add stage requeue/rescue validation across all four 32B stages",
            "run a stronger memory-pressure crossing proof with longer context after the slot-count proof is retained",
        ]
    elif evidence.get("multi_token_generation_feasible") is True:
        next_operator_actions = [
            "move the temporary 32B proof Coordinator protocol into the production Coordinator/Miner task API",
            "add KV-cache reuse so multi-token 32B decode does not recompute the full prompt prefix each token",
            "add bounded batch/sequential request validation for the 32B AWQ runtime",
            "add stage requeue/rescue validation for the 32B AWQ runtime",
            "keep comparing cross-kernel vs single-kernel T4x2 latency, memory, and stability before claiming a scale advantage",
        ]
    elif evidence.get("one_token_generation_feasible") is True:
        next_operator_actions = [
            "extend the retained 32B AWQ cross-kernel path from 1-token to bounded multi-token decode",
            "move the local orchestrator-mediated activation handoff behind the Coordinator/Miner protocol",
            "measure live activation transfer bytes, latency, and tokens/sec for the 32B path",
            "add stage requeue/rescue validation for the 32B AWQ runtime",
            "keep private Kaggle kernels, activation payloads, and tokens cleaned up after each fresh proof",
        ]
    else:
        next_operator_actions = [
            "integrate cross-kernel 32B AWQ activation/decode execution before claiming 32B generation",
            "reduce per-stage memory or use more/larger GPUs if the memory margin remains negative",
            "run package mode to inspect public-safe stage package metadata before any private Kaggle launch",
            "run external-existing or kaggle-auto only with available Kaggle GPU quota, cleanup, and token rotation",
            "measure live activation transfer after the 1-token 32B path succeeds",
            "use the retained AWQ safetensors stage-owned loading proof instead of repeating single-kernel full-model attempts",
        ]

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "kaggle_swarm_32b_quantized_feasibility_ready": ready,
        "candidate_32b_model_selected": runtime_plan.get("candidate_32b_model_selected") is True,
        "quantized_runtime_plan_ready": runtime_plan.get("quantized_runtime_plan_ready") is True,
        "kaggle_multi_kernel_topology_ready": topology.get("kaggle_multi_kernel_topology_ready") is True,
        "stage_partition_plan_ready": partition.get("stage_partition_plan_ready") is True,
        "per_stage_memory_estimate_ready": partition.get("per_stage_memory_estimate_ready") is True,
        "activation_transfer_estimate_ready": partition.get("activation_transfer_estimate_ready") is True,
        "kaggle_stage_package_plan_ready": stage_package_plan.get("kaggle_stage_package_plan_ready") is True,
        "stage_owned_loading_feasible": evidence.get("stage_owned_loading_feasible") is True,
        "one_token_generation_feasible": evidence.get("one_token_generation_feasible") is True,
        "multi_token_generation_feasible": evidence.get("multi_token_generation_feasible") is True,
        "coordinator_direct_management_feasible": evidence.get("coordinator_direct_management_feasible") is True,
        "upper_bound_crossing_feasible": evidence.get("upper_bound_crossing_feasible") is True,
        "batch_or_sequential_request_feasible": evidence.get("batch_or_sequential_request_feasible") is True,
        "stage_requeue_feasible": evidence.get("stage_requeue_feasible") is True,
        "largest_feasible_model_tier": retained_largest,
        "largest_attempted_model_tier": args.candidate_model_tier,
        "feasibility_verdict": blockers.get("feasibility_verdict"),
        "blocked_reason": blockers.get("blocked_reason"),
        "blocker_details": blockers.get("blocker_details"),
        "hard_limits": blockers.get("hard_limits"),
        "execution_mode": args.execution_mode,
        "fresh_kaggle_run_performed": fresh_kaggle_run_performed,
        "external_runtime_verified": external_runtime_verified,
        "retained_evidence_imported": retained.get("retained_evidence_imported") is True,
        "public_artifact_safe": True,
        "output_dir": str(output_dir),
        "kaggle_gpu_profile": kaggle_profile,
        "quantized_runtime_plan": runtime_plan,
        "stage_partition_plan": partition,
        "kaggle_multi_kernel_topology": topology,
        "kaggle_stage_package_plan": stage_package_plan,
        "evidence_validation": evidence,
        "retained_evidence_summary": retained,
        "fresh_32b_live_probe_summary": fresh_probe,
        "fresh_32b_stage_owned_loading_probe_summary": stage_owned_probe,
        "fresh_32b_activation_decode_probe_summary": activation_decode_probe,
        "source_reports": {
            "production_like_validation": source_summary(production_like_path, production_like_report, kind="gpu_swarm_production_like_validation") if production_like_report else {"present": False, "public_artifact_safe": True},
            "core_status": source_summary(core_status_path, core_status_report, kind="core_technology_validation_status") if core_status_report else {"present": False, "public_artifact_safe": True},
            "large_model_kaggle": source_summary(large_kaggle_path, large_kaggle_report, kind="large_model_kaggle_validation") if large_kaggle_report else {"present": False, "public_artifact_safe": True},
            "fresh_32b_live_probe": source_summary(fresh_probe_path, fresh_probe_report, kind="kaggle_32b_quantized_live_experiment_summary") if fresh_probe_report else {"present": False, "public_artifact_safe": True},
            "fresh_32b_stage_owned_loading_probe": source_summary(stage_owned_probe_path, stage_owned_probe_report, kind="kaggle_32b_stage_owned_safetensors_probe") if stage_owned_probe_report else {"present": False, "public_artifact_safe": True},
            "fresh_32b_activation_decode_probe": source_summary(activation_decode_probe_path, activation_decode_probe_report, kind="kaggle_32b_stage_owned_activation_decode_probe") if activation_decode_probe_report else {"present": False, "public_artifact_safe": True},
        },
        "mode_truth": {
            "fixture": args.execution_mode == "fixture",
            "evidence_import": args.execution_mode == "evidence-import",
            "package": args.execution_mode == "package",
            "external_existing": args.execution_mode == "external-existing",
            "kaggle_auto": args.execution_mode == "kaggle-auto",
            "fresh_kaggle_run_performed": fresh_kaggle_run_performed,
            "retained_evidence_imported": retained.get("retained_evidence_imported") is True,
        },
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_states_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "model_cache_private_paths_public": False,
            "kaggle_credentials_public": False,
            "api_keys_public": False,
            "coordinator_tokens_public": False,
            "lease_material_public": False,
            "idempotency_material_public": False,
            "private_env_written": False,
            "registry_written": False,
            "inline_kaggle_payload_public": False,
            "runtime_private_state_public": False,
            "private_package_payloads_written": False,
            "report_public_leak_paths": [],
        },
        "boundaries": dict(BOUNDARIES),
        "next_operator_actions": next_operator_actions,
        "diagnosis_codes": sorted(diagnosis_codes),
        "errors": [] if ready else ["kaggle_swarm_32b_quantized_feasibility_not_ready"],
    }
    leaks = public_redaction_errors(report)
    report["safety"]["report_public_leak_paths"] = leaks
    report["public_artifact_safe"] = not leaks
    report["safety"]["public_artifact_safe"] = not leaks
    if leaks:
        report["ok"] = False
        report["kaggle_swarm_32b_quantized_feasibility_ready"] = False
        report["errors"] = sorted(set(_list(report.get("errors")) + ["public_redaction_failed"]))
        report["diagnosis_codes"] = sorted(set(_list(report.get("diagnosis_codes")) + ["kaggle_swarm_32b_public_artifact_redaction_failed"]))

    summary_json = output_dir / "kaggle_swarm_32b_quantized_feasibility.json"
    summary_md = output_dir / "KAGGLE_SWARM_32B_QUANTIZED_FEASIBILITY.md"
    support_json = output_dir / "support_bundle.json"
    stage_package_json = output_dir / "kaggle_stage_package_plan.json"
    artifacts = {
        "summary_json": artifact_entry(summary_json, output_dir, kind="kaggle_swarm_32b_quantized_feasibility", schema=SCHEMA, ok=report.get("ok")),
        "summary_markdown": artifact_entry(summary_md, output_dir, kind="kaggle_swarm_32b_quantized_feasibility_markdown"),
        "support_bundle_json": artifact_entry(support_json, output_dir, kind="kaggle_swarm_32b_quantized_feasibility_support_bundle", schema=SUPPORT_BUNDLE_SCHEMA, ok=report.get("ok")),
        "stage_package_plan_json": artifact_entry(stage_package_json, output_dir, kind="kaggle_swarm_32b_stage_package_plan", schema=STAGE_PACKAGE_PLAN_SCHEMA, ok=stage_package_plan.get("kaggle_stage_package_plan_ready")),
    }
    if production_like_report and production_like_path.is_file():
        artifacts["production_like_validation_json"] = artifact_entry(production_like_path.resolve(), output_dir, kind="gpu_swarm_production_like_validation", schema=production_like_report.get("schema", ""), ok=production_like_report.get("ok"))
    if core_status_report and core_status_path.is_file():
        artifacts["core_status_json"] = artifact_entry(core_status_path.resolve(), output_dir, kind="core_technology_validation_status", schema=core_status_report.get("schema", ""), ok=core_status_report.get("ok"))
    if large_kaggle_report and large_kaggle_path.is_file():
        artifacts["large_model_kaggle_json"] = artifact_entry(large_kaggle_path.resolve(), output_dir, kind="large_model_kaggle_validation", schema=large_kaggle_report.get("schema", ""), ok=large_kaggle_report.get("ok"))
    if fresh_probe_report and fresh_probe_path.is_file():
        artifacts["fresh_32b_live_probe_json"] = artifact_entry(fresh_probe_path.resolve(), output_dir, kind="kaggle_32b_quantized_live_experiment_summary", schema=fresh_probe_report.get("schema", ""), ok=fresh_probe_report.get("ok"))
    if stage_owned_probe_report and stage_owned_probe_path.is_file():
        artifacts["fresh_32b_stage_owned_loading_probe_json"] = artifact_entry(stage_owned_probe_path.resolve(), output_dir, kind="kaggle_32b_stage_owned_safetensors_probe", schema=stage_owned_probe_report.get("schema", ""), ok=stage_owned_probe_report.get("ok"))
    if activation_decode_probe_report and activation_decode_probe_path.is_file():
        artifacts["fresh_32b_activation_decode_probe_json"] = artifact_entry(activation_decode_probe_path.resolve(), output_dir, kind="kaggle_32b_stage_owned_activation_decode_probe", schema=activation_decode_probe_report.get("schema", ""), ok=activation_decode_probe_report.get("ok"))
    report["artifacts"] = artifacts
    report["artifact_summary"] = artifact_summary(artifacts)
    write_json(stage_package_json, stage_package_plan)
    write_json(summary_json, report)
    summary_md.write_text(render_markdown(report), encoding="utf-8")
    write_json(support_json, build_support_bundle(report))
    for name, path in [
        ("summary_json", summary_json),
        ("summary_markdown", summary_md),
        ("support_bundle_json", support_json),
        ("stage_package_plan_json", stage_package_json),
    ]:
        artifacts[name] = artifact_entry(path, output_dir, kind=artifacts[name]["kind"], schema=artifacts[name].get("schema", ""), ok=artifacts[name].get("ok"))
    report["artifact_summary"] = artifact_summary(artifacts)
    write_json(summary_json, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kaggle Swarm 32B quantized feasibility RC evidence.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="evidence-import")
    parser.add_argument("--production-like-report", default=DEFAULT_PRODUCTION_LIKE_REPORT)
    parser.add_argument("--core-status-report", default=DEFAULT_CORE_STATUS_REPORT)
    parser.add_argument("--large-model-kaggle-report", default=DEFAULT_LARGE_MODEL_KAGGLE_REPORT)
    parser.add_argument("--fresh-32b-live-probe-report", default=DEFAULT_FRESH_32B_LIVE_PROBE_REPORT)
    parser.add_argument("--fresh-32b-stage-owned-loading-probe-report", default=DEFAULT_FRESH_32B_STAGE_OWNED_LOADING_PROBE_REPORT)
    parser.add_argument("--fresh-32b-activation-decode-probe-report", default=DEFAULT_FRESH_32B_ACTIVATION_DECODE_PROBE_REPORT)
    parser.add_argument("--candidate-model-id", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--candidate-model-tier", default="32b-quantized")
    parser.add_argument("--candidate-parameter-count-b", type=float, default=32.5)
    parser.add_argument("--candidate-hidden-size", type=int, default=5120)
    parser.add_argument("--candidate-layer-count", type=int, default=64)
    parser.add_argument("--quantized-format", default="AWQ-safetensors")
    parser.add_argument("--quantization-bits", type=float, default=4.0)
    parser.add_argument("--quantization-metadata-overhead-percent", type=float, default=10.0)
    parser.add_argument("--runtime-adapter", choices=RUNTIME_ADAPTERS, default="hf-awq-stage-selective-kaggle")
    parser.add_argument("--kaggle-gpu-type", default="NVIDIA_TESLA_T4_X2")
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--available-vram-per-gpu-mb", type=int, default=15360)
    parser.add_argument("--simultaneous-kaggle-gpu-kernel-limit", type=int, default=2)
    parser.add_argument("--stage-count", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--kv-cache-bytes-per-element", type=int, default=2)
    parser.add_argument("--activation-bytes-per-element", type=int, default=2)
    parser.add_argument("--runtime-overhead-mb-per-stage", type=int, default=3072)
    parser.add_argument("--fragmentation-margin-mb-per-stage", type=int, default=1024)
    parser.add_argument("--package-overhead-mb-per-stage", type=int, default=512)
    parser.add_argument("--target-max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-request-target", type=int, default=2)
    parser.add_argument("--max-fresh-model-attempts", type=int, default=2)
    parser.add_argument("--max-requeue-attempts", type=int, default=1)
    parser.add_argument("--max-attempt-timeout-minutes", type=int, default=60)
    parser.add_argument("--fresh-kaggle-run-performed", action="store_true")
    parser.add_argument("--package-slug-prefix", default="ct-32b-q")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.execution_mode == "evidence-import":
        for attr in ["production_like_report", "core_status_report", "large_model_kaggle_report"]:
            if not Path(getattr(args, attr)).is_file():
                raise SystemExit(f"--{attr.replace('_', '-')} must point to an existing JSON file")
        if args.fresh_32b_live_probe_report and not Path(args.fresh_32b_live_probe_report).is_file():
            default_fresh = Path(DEFAULT_FRESH_32B_LIVE_PROBE_REPORT)
            requested = Path(args.fresh_32b_live_probe_report)
            if requested != default_fresh:
                raise SystemExit("--fresh-32b-live-probe-report must point to an existing JSON file when provided")
        if args.fresh_32b_stage_owned_loading_probe_report and not Path(args.fresh_32b_stage_owned_loading_probe_report).is_file():
            default_stage_owned = Path(DEFAULT_FRESH_32B_STAGE_OWNED_LOADING_PROBE_REPORT)
            requested = Path(args.fresh_32b_stage_owned_loading_probe_report)
            if requested != default_stage_owned:
                raise SystemExit("--fresh-32b-stage-owned-loading-probe-report must point to an existing JSON file when provided")
        if args.fresh_32b_activation_decode_probe_report and not Path(args.fresh_32b_activation_decode_probe_report).is_file():
            default_activation_decode = Path(DEFAULT_FRESH_32B_ACTIVATION_DECODE_PROBE_REPORT)
            requested = Path(args.fresh_32b_activation_decode_probe_report)
            if requested != default_activation_decode:
                raise SystemExit("--fresh-32b-activation-decode-probe-report must point to an existing JSON file when provided")
    for name in [
        "candidate_parameter_count_b",
        "quantization_bits",
        "quantization_metadata_overhead_percent",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in [
        "candidate_hidden_size",
        "candidate_layer_count",
        "gpu_count",
        "available_vram_per_gpu_mb",
        "simultaneous_kaggle_gpu_kernel_limit",
        "stage_count",
        "context_length",
        "kv_cache_bytes_per_element",
        "activation_bytes_per_element",
        "runtime_overhead_mb_per_stage",
        "fragmentation_margin_mb_per_stage",
        "package_overhead_mb_per_stage",
    ]:
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.target_max_new_tokens < 1 or args.target_max_new_tokens > 128:
        raise SystemExit("--target-max-new-tokens must be between 1 and 128")
    if args.batch_request_target < 1 or args.batch_request_target > 16:
        raise SystemExit("--batch-request-target must be between 1 and 16")
    if args.max_fresh_model_attempts > 2:
        raise SystemExit("--max-fresh-model-attempts must be <= 2")
    if args.max_requeue_attempts > 1:
        raise SystemExit("--max-requeue-attempts must be <= 1")
    if args.max_attempt_timeout_minutes < 1 or args.max_attempt_timeout_minutes > 60:
        raise SystemExit("--max-attempt-timeout-minutes must be between 1 and 60")
    if args.stage_count > args.simultaneous_kaggle_gpu_kernel_limit and args.execution_mode == "kaggle-auto":
        raise SystemExit("--stage-count must not exceed --simultaneous-kaggle-gpu-kernel-limit in kaggle-auto mode")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(render_markdown(report))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
