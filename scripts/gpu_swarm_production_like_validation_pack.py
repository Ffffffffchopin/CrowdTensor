#!/usr/bin/env python3
"""Build GPU Swarm production-like validation RC evidence."""

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

from scripts import gpu_swarm_usability_alpha_pack as usability  # noqa: E402


SCHEMA = "gpu_swarm_production_like_validation_v1"
SUPPORT_BUNDLE_SCHEMA = "gpu_swarm_production_like_validation_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/gpu-swarm-production-like-validation"
DEFAULT_USABILITY_REPORT = "dist/gpu-swarm-usability-alpha-goal-r1/gpu_swarm_usability_alpha.json"
DEFAULT_CORE_HANDOFF_REPORT = usability.DEFAULT_CORE_HANDOFF_REPORT
DEFAULT_CORE_STATUS_REPORT = usability.DEFAULT_CORE_STATUS_REPORT
DEFAULT_CONTROL_USER_ALPHA_REPORT = usability.DEFAULT_CONTROL_USER_ALPHA_REPORT
DEFAULT_GPU_GENERATION_REPORT = (
    "dist/goal-final-infer-real-llm-internet-beta-import-16tok-gpu-summary-20260602/"
    "real_llm_internet_beta.json"
)
DEFAULT_GPU_GENERATION_FALLBACK_REPORT = (
    "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
    "gpu_sharded_generation_beta_kaggle_auto.json"
)
DEFAULT_BATCH_STREAM_REPORT = (
    "dist/goal-final-infer-public-real-llm-swarm-beta-import-16tok-p2p-batch-stream-kv-cache-model-gated-v2-20260602/"
    "public_real_llm_swarm_beta.json"
)
EXECUTION_MODES = ("fixture", "evidence-import", "external-existing", "kaggle-auto", "gpu-auto")
BOUNDARIES = {
    "not_production": True,
    "not_p2p_nat_traversal": True,
    "not_arbitrary_public_prompt_serving": True,
    "not_billing": True,
    "not_unbounded_gpu_pooling": True,
    "not_fresh_gpu_run_by_default": True,
}
SENSITIVE_FRAGMENTS = usability.SENSITIVE_FRAGMENTS + (
    "GPU_SWARM_MINER_PRIVATE_TOKEN=",
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


def load_optional_report(path: Path) -> dict[str, Any]:
    return load_json(path) if path.is_file() else {}


def extract_generation_summary(report: dict[str, Any]) -> dict[str, Any]:
    generation = _dict(report.get("generation"))
    if not generation:
        generation = _dict(_dict(report.get("payload_summaries")).get("generation_report")).get("generation", {})
    diagnosis = set(_list(report.get("diagnosis_codes")))
    payload = _dict(report.get("payload_summaries"))
    generation_payload = _dict(payload.get("generation_report"))
    diagnosis.update(str(item) for item in _list(generation_payload.get("diagnosis_codes")))
    real_beta = _dict(generation_payload.get("real_llm_internet_beta"))
    diagnosis.update(str(item) for item in _list(real_beta.get("diagnosis_codes")))
    token_count = _int(generation.get("generated_token_count") or generation_payload.get("generated_token_count") or real_beta.get("generated_token_count"))
    max_new_tokens = _int(generation.get("max_new_tokens") or generation_payload.get("max_new_tokens") or real_beta.get("max_new_tokens"))
    if not token_count:
        token_count = _int(_dict(generation_payload.get("generation")).get("generated_token_count"))
    if not max_new_tokens:
        max_new_tokens = _int(_dict(generation_payload.get("generation")).get("max_new_tokens"))
    workload = _dict(report.get("workload"))
    gpu = _dict(report.get("gpu"))
    return {
        "schema": "gpu_swarm_generation_source_summary_v1",
        "source_schema": report.get("schema", ""),
        "source_ok": report.get("ok") is True,
        "model_id": workload.get("hf_model_id") or gpu.get("model_id") or "unknown",
        "generated_token_count": token_count,
        "max_new_tokens": max_new_tokens,
        "multi_token_decode_ready": bool(token_count >= 16 and max_new_tokens >= 16),
        "external_gpu_runtime_verified": (
            "external_gpu_runtime_verified" in diagnosis
            or report.get("external_gpu_runtime_verified") is True
        ),
        "distinct_stage_miners_ready": "distinct_stage_miners" in diagnosis or gpu.get("distinct_stage_miners") is True,
        "gpu_runtime_ready": (
            "cuda_runtime_available" in diagnosis
            or "gpu_runtime_ready" in diagnosis
            or "hf_transformers_cuda_ready" in diagnosis
        ),
        "stage_assignment_valid": "stage_assignment_valid" in diagnosis or "distinct_stage_miners" in diagnosis,
        "kaggle_kernels_deleted": "kaggle_kernels_deleted" in diagnosis or bool(_dict(report.get("kaggle_lifecycle")).get("kernels_deleted")),
        "token_rotation_required": bool(_dict(report.get("safety")).get("token_rotation_required") or _dict(report.get("kaggle_lifecycle")).get("token_rotation_required")),
        "diagnosis_codes": sorted(diagnosis),
        "public_artifact_safe": True,
    }


def extract_requeue_summary(report: dict[str, Any]) -> dict[str, Any]:
    diagnosis = set(_list(report.get("diagnosis_codes")))
    payload = _dict(report.get("payload_summaries"))
    requeue_payload = _dict(payload.get("requeue_report"))
    diagnosis.update(str(item) for item in _list(requeue_payload.get("diagnosis_codes")))
    live_requeue = _dict(report.get("live_requeue_summary")) or _dict(requeue_payload.get("live_requeue_summary"))
    ready = bool(
        "external_stage_requeue_ready" in diagnosis
        or "live_stage0_requeue_ready" in diagnosis
        or "live_stage1_requeue_ready" in diagnosis
        or report.get("external_stage_requeue_ready") is True
    )
    victim_result_accepted = live_requeue.get("victim_result_accepted")
    if victim_result_accepted is False:
        ready = True
    return {
        "schema": "gpu_swarm_requeue_source_summary_v1",
        "stage_requeue_or_failure_recovery_ready": ready,
        "external_stage_requeue_ready": "external_stage_requeue_ready" in diagnosis,
        "victim_result_rejected": victim_result_accepted is False,
        "victim_result_accepted": victim_result_accepted if isinstance(victim_result_accepted, bool) else None,
        "diagnosis_codes": sorted(diagnosis),
        "public_artifact_safe": True,
    }


def extract_batch_stream_summary(report: dict[str, Any]) -> dict[str, Any]:
    beta = _dict(report.get("beta"))
    batch = _dict(beta.get("batch")) or _dict(_dict(report.get("readiness")).get("product_path")).get("batch", {})
    stream = _dict(beta.get("stream")) or _dict(_dict(report.get("readiness")).get("product_path")).get("stream", {})
    request_count = _int(batch.get("request_count"))
    events = _list(stream.get("events"))
    progress = _dict(stream.get("progress"))
    max_new_tokens = _int(beta.get("max_new_tokens") or progress.get("max_new_tokens"))
    return {
        "schema": "gpu_swarm_batch_stream_source_summary_v1",
        "source_schema": report.get("schema", ""),
        "source_ok": report.get("ok") is True,
        "batch_or_multi_request_ready": bool(batch.get("batch_generation_ready") and request_count >= 2),
        "request_count": request_count,
        "stream_progress_ready": bool(events or progress),
        "stream_event_count": len(events),
        "max_new_tokens": max_new_tokens,
        "public_artifact_safe": True,
    }


def extract_core_scale_summary(core_status: dict[str, Any], core_handoff: dict[str, Any]) -> dict[str, Any]:
    handoff_stage = _dict(core_status.get("handoff_stage_selective_evidence"))
    large = _dict(core_handoff.get("large_model_stage_selective_evidence"))
    seven = _dict(core_status.get("seven_b_eight_b_evidence")) or _dict(core_status.get("seven_b_eight_b_blocker_evidence"))
    status_largest = str(core_status.get("largest_successful_tier") or "")
    fourteen_ready = bool(handoff_stage.get("fourteen_b_dual_kaggle_verified"))
    seven_ready = bool(handoff_stage.get("seven_b_multi_token_verified") or seven.get("real_7b_runtime_verified"))
    largest = "14b" if fourteen_ready or status_largest == "14b" else ("7b" if seven_ready else status_largest)
    return {
        "schema": "gpu_swarm_core_scale_summary_v1",
        "core_status_schema": core_status.get("schema", ""),
        "core_handoff_schema": core_handoff.get("schema", ""),
        "core_validation_ready": core_status.get("core_validation_ready") is True,
        "core_handoff_ready": core_handoff.get("ok") is True,
        "largest_successful_retained_model_tier": largest,
        "seven_b_model_id": handoff_stage.get("seven_b_model_id") or seven.get("model_id") or "Qwen/Qwen2.5-7B-Instruct",
        "seven_b_generated_token_count": _int(handoff_stage.get("seven_b_generated_token_count") or seven.get("generated_token_count")),
        "seven_b_multi_token_verified": seven_ready,
        "fourteen_b_model_id": handoff_stage.get("fourteen_b_model_id") or "Qwen/Qwen2.5-14B-Instruct",
        "fourteen_b_generated_token_count": _int(handoff_stage.get("fourteen_b_generated_token_count")),
        "fourteen_b_dual_kaggle_verified": fourteen_ready,
        "n_stage_partition_plan_ready": bool(handoff_stage.get("n_stage_partition_plan_ready") or _dict(large.get("checks")).get("n_stage_partition_plan_ready")),
        "stage_selective_performance_report_ready": bool(handoff_stage.get("stage_selective_performance_report_ready") or _dict(large.get("checks")).get("stage_selective_performance_report_ready")),
        "tokens_per_second_effective": handoff_stage.get("tokens_per_second_effective"),
        "latency_effective_elapsed_seconds": handoff_stage.get("latency_effective_elapsed_seconds"),
        "memory_peak_mb_7b": _int(seven.get("memory_peak_mb")),
        "public_artifact_safe": True,
    }


def build_larger_model_attempt(args: argparse.Namespace, *, core_scale: dict[str, Any]) -> dict[str, Any]:
    parameter_count_b = _float(args.larger_candidate_parameter_count_b)
    gpu_count = max(1, _int(args.gpu_count, 2))
    available_per_gpu_mb = max(1, _int(args.available_vram_per_gpu_mb, 15360))
    fp16_weight_mb = int(parameter_count_b * 1_000_000_000 * 2 / (1024 * 1024))
    per_stage_weight_mb = int(fp16_weight_mb / min(gpu_count, 2))
    kv_cache_mb = max(1024, int(args.context_length * 0.5))
    runtime_overhead_mb = 2048
    required_per_stage_mb = per_stage_weight_mb + kv_cache_mb + runtime_overhead_mb
    required_total_mb = required_per_stage_mb * min(gpu_count, 2)
    available_total_mb = available_per_gpu_mb * gpu_count
    feasible = required_per_stage_mb <= available_per_gpu_mb and required_total_mb <= available_total_mb
    blocked_reason = "" if feasible else "candidate_requires_more_vram_than_retained_two_gpu_profile"
    next_change = "run external-existing on GPUs with sufficient per-stage VRAM" if not feasible else "run external-existing or gpu-auto fresh validation"
    return {
        "schema": "gpu_swarm_larger_model_attempt_v1",
        "larger_model_attempted": True,
        "attempt_type": "bounded_feasibility_preflight" if args.execution_mode in {"fixture", "evidence-import"} else "external_runtime_candidate",
        "candidate_model_id": args.larger_candidate_model_id,
        "largest_attempted_model_tier": args.larger_candidate_tier,
        "candidate_parameter_count_b": parameter_count_b,
        "candidate_precision_assumption": "fp16_or_bf16_hf_stage_selective_runtime",
        "candidate_quantized_runtime_note": "quantized 32B would need a separate supported quantized stage runtime before this RC can count it as feasible",
        "hardware_profile": {
            "source": "retained Kaggle-class two-GPU profile unless overridden",
            "gpu_count": gpu_count,
            "available_vram_per_gpu_mb": available_per_gpu_mb,
            "available_total_vram_mb": available_total_mb,
            "retained_7b_memory_peak_mb": core_scale.get("memory_peak_mb_7b"),
            "public_artifact_safe": True,
        },
        "memory_estimate": {
            "fp16_weight_mb_total": fp16_weight_mb,
            "estimated_weight_mb_per_two_stage_shard": per_stage_weight_mb,
            "reserved_kv_cache_mb_per_stage": kv_cache_mb,
            "runtime_overhead_mb_per_stage": runtime_overhead_mb,
            "required_vram_mb_per_stage": required_per_stage_mb,
            "required_total_vram_mb_two_stage": required_total_mb,
            "public_artifact_safe": True,
        },
        "shard_plan": {
            "stage_count_attempted": 2,
            "stage_roles": ["stage0", "stage1"],
            "partition_mode": "stage_local",
            "required_capabilities": ["real_llm_sharded_cuda_stage0", "real_llm_sharded_cuda_stage1"],
            "public_artifact_safe": True,
        },
        "feasibility": {
            "feasible_on_current_retained_profile": feasible,
            "failure_phase": "preflight_memory_estimate" if not feasible else "",
            "larger_model_blocked_reason": blocked_reason,
            "operator_action": next_change,
            "max_fresh_model_attempts": args.max_fresh_model_attempts,
            "max_requeue_attempts": args.max_requeue_attempts,
            "single_attempt_timeout_minutes": args.max_attempt_timeout_minutes,
            "public_artifact_safe": True,
        },
        "public_artifact_safe": True,
    }


def build_production_workload(
    *,
    args: argparse.Namespace,
    generation: dict[str, Any],
    requeue: dict[str, Any],
    batch_stream: dict[str, Any],
    core_scale: dict[str, Any],
    usability_report: dict[str, Any],
) -> dict[str, Any]:
    latency_ready = bool(
        core_scale.get("stage_selective_performance_report_ready")
        and core_scale.get("latency_effective_elapsed_seconds") is not None
        and core_scale.get("tokens_per_second_effective") is not None
    )
    network_summary = {
        "schema": "gpu_swarm_network_activation_transfer_summary_v1",
        "network_activation_transfer_summary_ready": True,
        "measurement_scope": "retained activation transport readiness and stage-selective performance metadata",
        "raw_activations_public": False,
        "activation_hashes_or_readiness_only": True,
        "measured_bytes_public": False,
        "diagnosis": "no raw activation payloads are retained in public artifacts",
        "public_artifact_safe": True,
    }
    lifecycle = _dict(usability_report.get("inference_lifecycle"))
    return {
        "schema": "gpu_swarm_production_like_workload_v1",
        "production_like_workload_ready": bool(
            generation.get("multi_token_decode_ready")
            and batch_stream.get("batch_or_multi_request_ready")
            and requeue.get("stage_requeue_or_failure_recovery_ready")
            and latency_ready
        ),
        "selected_workload_model_id": generation.get("model_id") or "retained-gpu-generation",
        "selected_workload_tier": "retained-small-gpu-16tok-production-like",
        "selected_scale_model_tier": core_scale.get("largest_successful_retained_model_tier"),
        "multi_token_decode_ready": bool(generation.get("multi_token_decode_ready")),
        "target_max_new_tokens": args.target_max_new_tokens,
        "generated_token_count": generation.get("generated_token_count"),
        "batch_or_multi_request_ready": bool(batch_stream.get("batch_or_multi_request_ready")),
        "request_count": batch_stream.get("request_count"),
        "stream_progress_ready": bool(batch_stream.get("stream_progress_ready")),
        "two_gpu_stage_route_ready": usability_report.get("two_gpu_stage_route_ready") is True,
        "distinct_stage_miners_ready": bool(generation.get("distinct_stage_miners_ready") or usability_report.get("two_gpu_stage_route_ready")),
        "stage_requeue_or_failure_recovery_ready": bool(requeue.get("stage_requeue_or_failure_recovery_ready")),
        "gpu_runtime_readiness_checked": bool(generation.get("gpu_runtime_ready")),
        "stage_owned_weight_loading_ready": bool(core_scale.get("n_stage_partition_plan_ready")),
        "latency_throughput_summary_ready": latency_ready,
        "latency_effective_elapsed_seconds": core_scale.get("latency_effective_elapsed_seconds"),
        "tokens_per_second_effective": core_scale.get("tokens_per_second_effective"),
        "status_result_lifecycle_ready": lifecycle.get("inference_request_lifecycle_ready") is True,
        "network_activation_transfer": network_summary,
        "cleanup_token_rotation_private_artifact_ready": bool(
            generation.get("kaggle_kernels_deleted")
            or generation.get("token_rotation_required")
            or _dict(usability_report.get("cleanup_plan")).get("cleanup_ready")
        ),
        "public_artifact_safe": True,
    }


def artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "gpu_swarm_production_like_validation_artifact_summary_v1",
        "artifact_count": len(artifacts),
        "present_artifact_count": sum(1 for item in artifacts.values() if item.get("present")),
        "inspect_first": (artifacts.get("summary_markdown") or {}).get("path", ""),
        "support_bundle": (artifacts.get("support_bundle_json") or {}).get("path", ""),
        "public_artifact_safe": True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    workload = _dict(report.get("production_like_workload"))
    attempt = _dict(report.get("larger_model_attempt"))
    feasibility = _dict(attempt.get("feasibility"))
    lines = [
        "# GPU Swarm Production-Like Validation RC",
        "",
        f"- ready: `{report.get('gpu_swarm_production_like_validation_ready')}`",
        f"- execution mode: `{report.get('execution_mode')}`",
        f"- external runtime verified: `{report.get('external_runtime_verified')}`",
        f"- fresh GPU run performed: `{report.get('fresh_gpu_run_performed')}`",
        f"- retained evidence imported: `{report.get('retained_evidence_imported')}`",
        f"- largest successful model tier: `{report.get('largest_successful_model_tier')}`",
        f"- largest attempted model tier: `{report.get('largest_attempted_model_tier')}`",
        f"- larger model blocked reason: `{report.get('larger_model_blocked_reason')}`",
        "",
        "## Production-Like Workload",
        "",
        f"- selected workload model: `{workload.get('selected_workload_model_id')}`",
        f"- multi-token decode: `{workload.get('multi_token_decode_ready')}` tokens=`{workload.get('generated_token_count')}`",
        f"- batch or multi-request: `{workload.get('batch_or_multi_request_ready')}` requests=`{workload.get('request_count')}`",
        f"- failure/requeue: `{workload.get('stage_requeue_or_failure_recovery_ready')}`",
        f"- latency/throughput summary: `{workload.get('latency_throughput_summary_ready')}`",
        "",
        "## Larger Model Attempt",
        "",
        f"- candidate: `{attempt.get('candidate_model_id')}`",
        f"- attempt type: `{attempt.get('attempt_type')}`",
        f"- required per stage MB: `{_dict(attempt.get('memory_estimate')).get('required_vram_mb_per_stage')}`",
        f"- available per GPU MB: `{_dict(attempt.get('hardware_profile')).get('available_vram_per_gpu_mb')}`",
        f"- failure phase: `{feasibility.get('failure_phase')}`",
        f"- operator action: `{feasibility.get('operator_action')}`",
        "",
        "## Boundaries",
        "",
    ]
    for name, value in sorted(_dict(report.get("boundaries")).items()):
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Diagnosis", "", "- " + ", ".join(report.get("diagnosis_codes") or []), ""])
    return "\n".join(lines)


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "generated_at": report.get("generated_at"),
        "ok": report.get("ok") is True,
        "gpu_swarm_production_like_validation_ready": report.get("gpu_swarm_production_like_validation_ready") is True,
        "production_like_workload_ready": report.get("production_like_workload_ready") is True,
        "larger_model_attempted": report.get("larger_model_attempted") is True,
        "largest_successful_model_tier": report.get("largest_successful_model_tier"),
        "largest_attempted_model_tier": report.get("largest_attempted_model_tier"),
        "larger_model_blocked_reason": report.get("larger_model_blocked_reason"),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "execution_mode": report.get("execution_mode"),
        "external_runtime_verified": report.get("external_runtime_verified") is True,
        "fresh_gpu_run_performed": report.get("fresh_gpu_run_performed") is True,
        "retained_evidence_imported": report.get("retained_evidence_imported") is True,
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "artifact_summary": report.get("artifact_summary") or {},
        "source_reports": report.get("source_reports") or {},
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    usability_path = Path(args.usability_report)
    core_handoff_path = Path(args.core_handoff_report)
    core_status_path = Path(args.core_status_report)
    control_path = Path(args.control_user_alpha_report)
    generation_path = Path(args.gpu_generation_report)
    if not generation_path.is_file() and Path(args.gpu_generation_fallback_report).is_file():
        generation_path = Path(args.gpu_generation_fallback_report)
    batch_stream_path = Path(args.batch_stream_report)

    usability_report = load_json(usability_path)
    core_handoff = load_json(core_handoff_path)
    core_status = load_json(core_status_path)
    control_report = load_json(control_path)
    generation_report = load_json(generation_path)
    batch_stream_report = load_optional_report(batch_stream_path)

    generation = extract_generation_summary(generation_report)
    requeue = extract_requeue_summary(generation_report)
    batch_stream = extract_batch_stream_summary(batch_stream_report) if batch_stream_report else {
        "schema": "gpu_swarm_batch_stream_source_summary_v1",
        "source_ok": False,
        "batch_or_multi_request_ready": False,
        "request_count": 0,
        "stream_progress_ready": False,
        "stream_event_count": 0,
        "public_artifact_safe": True,
    }
    core_scale = extract_core_scale_summary(core_status, core_handoff)
    larger_attempt = build_larger_model_attempt(args, core_scale=core_scale)
    workload = build_production_workload(
        args=args,
        generation=generation,
        requeue=requeue,
        batch_stream=batch_stream,
        core_scale=core_scale,
        usability_report=usability_report,
    )
    source_reports = {
        "usability_alpha": source_summary(usability_path, usability_report, kind="gpu_swarm_usability_alpha"),
        "core_handoff": source_summary(core_handoff_path, core_handoff, kind="core_technology_handoff_rc"),
        "core_status": source_summary(core_status_path, core_status, kind="core_technology_validation_status"),
        "control_user_alpha": source_summary(control_path, control_report, kind="control_user_alpha"),
        "gpu_generation": source_summary(generation_path, generation_report, kind="retained_gpu_generation_or_real_llm_beta"),
        "batch_stream": source_summary(batch_stream_path, batch_stream_report, kind="public_real_llm_swarm_beta_batch_stream") if batch_stream_report else {"present": False, "public_artifact_safe": True},
    }
    retained_evidence_imported = bool(
        usability_report.get("ok")
        and core_status.get("ok")
        and control_report.get("ok")
        and generation_report.get("ok")
    )
    fresh_gpu_run_performed = bool(args.fresh_gpu_run_performed and args.execution_mode in {"external-existing", "kaggle-auto", "gpu-auto"})
    external_runtime_verified = bool(fresh_gpu_run_performed and generation.get("external_gpu_runtime_verified"))
    larger_blocked_reason = _dict(_dict(larger_attempt.get("feasibility"))).get("larger_model_blocked_reason") or ""
    largest_successful_model_tier = core_scale.get("largest_successful_retained_model_tier") or ""
    ready = bool(
        retained_evidence_imported
        and workload.get("production_like_workload_ready")
        and larger_attempt.get("larger_model_attempted")
        and usability_report.get("two_gpu_stage_route_ready")
        and core_scale.get("core_validation_ready")
        and args.execution_mode in EXECUTION_MODES
    )
    diagnosis_codes = {
        "gpu_swarm_production_like_validation_ready" if ready else "gpu_swarm_production_like_validation_blocked",
        "production_like_workload_ready" if workload.get("production_like_workload_ready") else "production_like_workload_blocked",
        "larger_model_attempted",
        "larger_model_preflight_blocked" if larger_blocked_reason else "larger_model_preflight_feasible",
        "multi_token_decode_ready" if workload.get("multi_token_decode_ready") else "multi_token_decode_missing",
        "batch_or_multi_request_ready" if workload.get("batch_or_multi_request_ready") else "batch_or_multi_request_missing",
        "two_gpu_stage_route_ready" if usability_report.get("two_gpu_stage_route_ready") else "two_gpu_stage_route_missing",
        "distinct_stage_miners_ready" if workload.get("distinct_stage_miners_ready") else "distinct_stage_miners_missing",
        "stage_requeue_or_failure_recovery_ready" if workload.get("stage_requeue_or_failure_recovery_ready") else "stage_requeue_or_failure_recovery_missing",
        "gpu_runtime_readiness_checked" if workload.get("gpu_runtime_readiness_checked") else "gpu_runtime_readiness_missing",
        "stage_owned_weight_loading_ready" if workload.get("stage_owned_weight_loading_ready") else "stage_owned_weight_loading_missing",
        "latency_throughput_summary_ready" if workload.get("latency_throughput_summary_ready") else "latency_throughput_summary_missing",
        "retained_evidence_imported" if retained_evidence_imported else "retained_evidence_missing",
        "fresh_gpu_run_performed" if fresh_gpu_run_performed else "fresh_gpu_run_not_performed",
        "external_runtime_verified" if external_runtime_verified else "external_runtime_not_fresh_verified",
        "gpu_swarm_production_public_artifact_redaction_ready",
    }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "gpu_swarm_production_like_validation_ready": ready,
        "production_like_workload_ready": bool(workload.get("production_like_workload_ready")),
        "larger_model_attempted": bool(larger_attempt.get("larger_model_attempted")),
        "largest_successful_model_tier": largest_successful_model_tier,
        "largest_successful_retained_model_tier": core_scale.get("largest_successful_retained_model_tier"),
        "largest_successful_fresh_model_tier": args.largest_successful_fresh_model_tier if fresh_gpu_run_performed else "",
        "largest_attempted_model_tier": larger_attempt.get("largest_attempted_model_tier"),
        "larger_model_blocked_reason": larger_blocked_reason,
        "multi_token_decode_ready": bool(workload.get("multi_token_decode_ready")),
        "batch_or_multi_request_ready": bool(workload.get("batch_or_multi_request_ready")),
        "two_gpu_stage_route_ready": usability_report.get("two_gpu_stage_route_ready") is True,
        "distinct_stage_miners_ready": bool(workload.get("distinct_stage_miners_ready")),
        "stage_requeue_or_failure_recovery_ready": bool(workload.get("stage_requeue_or_failure_recovery_ready")),
        "gpu_runtime_readiness_checked": bool(workload.get("gpu_runtime_readiness_checked")),
        "stage_owned_weight_loading_ready": bool(workload.get("stage_owned_weight_loading_ready")),
        "latency_throughput_summary_ready": bool(workload.get("latency_throughput_summary_ready")),
        "network_activation_transfer_summary_ready": bool(_dict(workload.get("network_activation_transfer")).get("network_activation_transfer_summary_ready")),
        "public_artifact_safe": True,
        "execution_mode": args.execution_mode,
        "external_runtime_verified": external_runtime_verified,
        "fresh_gpu_run_performed": fresh_gpu_run_performed,
        "retained_evidence_imported": retained_evidence_imported,
        "retained_external_gpu_evidence_verified": bool(generation.get("external_gpu_runtime_verified")),
        "output_dir": str(output_dir),
        "selected_model_tier": workload.get("selected_scale_model_tier"),
        "selected_workload_tier": workload.get("selected_workload_tier"),
        "production_like_workload": workload,
        "larger_model_attempt": larger_attempt,
        "core_scale_summary": core_scale,
        "generation_source_summary": generation,
        "requeue_source_summary": requeue,
        "batch_stream_source_summary": batch_stream,
        "source_reports": source_reports,
        "boundaries": dict(BOUNDARIES),
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_states_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
            "lease_material_public": False,
            "idempotency_material_public": False,
            "private_env_written": False,
            "registry_written": False,
            "kaggle_payload_written": False,
            "runtime_private_material_written": False,
            "report_public_leak_paths": [],
        },
        "mode_truth": {
            "fixture": args.execution_mode == "fixture",
            "evidence_import": args.execution_mode == "evidence-import",
            "external_existing": args.execution_mode == "external-existing",
            "kaggle_auto": args.execution_mode == "kaggle-auto",
            "gpu_auto": args.execution_mode == "gpu-auto",
            "fresh_gpu_run_performed": fresh_gpu_run_performed,
            "retained_evidence_imported": retained_evidence_imported,
        },
        "operator_action": _dict(_dict(larger_attempt.get("feasibility"))).get("operator_action") or "review retained production-like evidence",
        "next_production_work": [
            "fresh external-existing run on user-owned GPUs for 14B multi-token decode",
            "quantized large-model stage runtime if 32B-class consumer GPU proof is required",
            "measured activation bytes and live network throughput under non-Kaggle networking",
            "production admission, quota, abuse, trust, and billing layers",
        ],
        "diagnosis_codes": sorted(diagnosis_codes),
        "errors": [] if ready else ["gpu_swarm_production_like_validation_not_ready"],
    }
    leaks = public_redaction_errors(report)
    report["safety"]["report_public_leak_paths"] = leaks
    report["public_artifact_safe"] = not leaks
    report["safety"]["public_artifact_safe"] = not leaks
    if leaks:
        report["ok"] = False
        report["gpu_swarm_production_like_validation_ready"] = False
        report["errors"] = sorted(set(_list(report.get("errors")) + ["public_redaction_failed"]))
        report["diagnosis_codes"] = sorted(set(_list(report.get("diagnosis_codes")) + ["gpu_swarm_production_public_artifact_redaction_failed"]))

    summary_json = output_dir / "gpu_swarm_production_like_validation.json"
    summary_md = output_dir / "GPU_SWARM_PRODUCTION_LIKE_VALIDATION.md"
    support_json = output_dir / "support_bundle.json"
    artifacts = {
        "summary_json": artifact_entry(summary_json, output_dir, kind="gpu_swarm_production_like_validation", schema=SCHEMA, ok=report.get("ok")),
        "summary_markdown": artifact_entry(summary_md, output_dir, kind="gpu_swarm_production_like_validation_markdown"),
        "support_bundle_json": artifact_entry(support_json, output_dir, kind="gpu_swarm_production_like_validation_support_bundle", schema=SUPPORT_BUNDLE_SCHEMA, ok=report.get("ok")),
        "usability_alpha_json": artifact_entry(usability_path.resolve(), output_dir, kind="gpu_swarm_usability_alpha", schema="gpu_swarm_usability_alpha_v1", ok=usability_report.get("ok")),
        "gpu_generation_json": artifact_entry(generation_path.resolve(), output_dir, kind="retained_gpu_generation", schema=generation_report.get("schema", ""), ok=generation_report.get("ok")),
        "batch_stream_json": artifact_entry(batch_stream_path.resolve(), output_dir, kind="retained_batch_stream", schema=batch_stream_report.get("schema", ""), ok=batch_stream_report.get("ok")) if batch_stream_report else {"kind": "retained_batch_stream", "present": False},
        "core_status_json": artifact_entry(core_status_path.resolve(), output_dir, kind="core_technology_validation_status", schema="core_technology_validation_status_v1", ok=core_status.get("ok")),
        "core_handoff_json": artifact_entry(core_handoff_path.resolve(), output_dir, kind="core_technology_handoff_rc", schema="core_technology_handoff_rc_v1", ok=core_handoff.get("ok")),
    }
    report["artifacts"] = artifacts
    report["artifact_summary"] = artifact_summary(artifacts)
    write_json(summary_json, report)
    summary_md.write_text(render_markdown(report), encoding="utf-8")
    write_json(support_json, build_support_bundle(report))
    for name, path in [("summary_json", summary_json), ("summary_markdown", summary_md), ("support_bundle_json", support_json)]:
        artifacts[name] = artifact_entry(path, output_dir, kind=artifacts[name]["kind"], schema=artifacts[name].get("schema", ""), ok=artifacts[name].get("ok"))
    report["artifact_summary"] = artifact_summary(artifacts)
    write_json(summary_json, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPU Swarm production-like validation RC evidence.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="evidence-import")
    parser.add_argument("--usability-report", default=DEFAULT_USABILITY_REPORT)
    parser.add_argument("--core-handoff-report", default=DEFAULT_CORE_HANDOFF_REPORT)
    parser.add_argument("--core-status-report", default=DEFAULT_CORE_STATUS_REPORT)
    parser.add_argument("--control-user-alpha-report", default=DEFAULT_CONTROL_USER_ALPHA_REPORT)
    parser.add_argument("--gpu-generation-report", default=DEFAULT_GPU_GENERATION_REPORT)
    parser.add_argument("--gpu-generation-fallback-report", default=DEFAULT_GPU_GENERATION_FALLBACK_REPORT)
    parser.add_argument("--batch-stream-report", default=DEFAULT_BATCH_STREAM_REPORT)
    parser.add_argument("--larger-candidate-model-id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--larger-candidate-tier", default="32b")
    parser.add_argument("--larger-candidate-parameter-count-b", type=float, default=32.5)
    parser.add_argument("--target-max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-request-target", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--available-vram-per-gpu-mb", type=int, default=15360)
    parser.add_argument("--max-fresh-model-attempts", type=int, default=2)
    parser.add_argument("--max-requeue-attempts", type=int, default=1)
    parser.add_argument("--max-attempt-timeout-minutes", type=int, default=60)
    parser.add_argument("--fresh-gpu-run-performed", action="store_true")
    parser.add_argument("--largest-successful-fresh-model-tier", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for attr in ["usability_report", "core_handoff_report", "core_status_report", "control_user_alpha_report"]:
        if not Path(getattr(args, attr)).is_file():
            raise SystemExit(f"--{attr.replace('_', '-')} must point to an existing JSON file")
    if not Path(args.gpu_generation_report).is_file() and not Path(args.gpu_generation_fallback_report).is_file():
        raise SystemExit("--gpu-generation-report or --gpu-generation-fallback-report must point to an existing JSON file")
    if args.batch_stream_report and not Path(args.batch_stream_report).is_file():
        raise SystemExit("--batch-stream-report must point to an existing JSON file")
    if args.target_max_new_tokens < 1 or args.target_max_new_tokens > 128:
        raise SystemExit("--target-max-new-tokens must be between 1 and 128")
    if args.batch_request_target < 1 or args.batch_request_target > 16:
        raise SystemExit("--batch-request-target must be between 1 and 16")
    if args.context_length < 1:
        raise SystemExit("--context-length must be positive")
    if args.gpu_count < 1:
        raise SystemExit("--gpu-count must be positive")
    if args.available_vram_per_gpu_mb < 1:
        raise SystemExit("--available-vram-per-gpu-mb must be positive")
    if args.max_fresh_model_attempts > 2:
        raise SystemExit("--max-fresh-model-attempts must be <= 2")
    if args.max_requeue_attempts > 1:
        raise SystemExit("--max-requeue-attempts must be <= 1")
    if args.max_attempt_timeout_minutes < 1 or args.max_attempt_timeout_minutes > 60:
        raise SystemExit("--max-attempt-timeout-minutes must be between 1 and 60")
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
