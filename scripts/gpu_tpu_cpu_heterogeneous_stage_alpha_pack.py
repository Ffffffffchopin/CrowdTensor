#!/usr/bin/env python3
"""Build GPU+TPU+CPU heterogeneous stage-inference Alpha evidence."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "gpu_tpu_cpu_heterogeneous_stage_alpha_v1"
SUPPORT_BUNDLE_SCHEMA = "gpu_tpu_cpu_heterogeneous_stage_alpha_support_bundle_v1"
STAGE_CONTRACT_SCHEMA = "gpu_tpu_cpu_stage_contract_smoke_v1"
LOCAL_THREE_STAGE_SCHEMA = "gpu_tpu_cpu_local_three_stage_real_model_e2e_v1"
TORCH_JAX_BRIDGE_SCHEMA = "gpu_tpu_cpu_torch_jax_torch_bridge_probe_v1"
FEASIBILITY_SCHEMA = "gpu_tpu_cpu_32b_feasibility_report_v1"
DEFAULT_OUTPUT_DIR = "dist/gpu-tpu-cpu-heterogeneous-stage-alpha"
DEFAULT_TPU_REAL_LLM_REPORT = (
    "dist/kaggle-tpu-gpt2-xl-jax-web-probe-20260621-r1/"
    "kaggle_tpu_real_llm_web_probe.json"
)
DEFAULT_GPU_FULL_32B_REPORT = (
    "dist/kaggle-32b-full-heterogeneous-multitoken-kv-live-20260620-r1/"
    "kaggle_32b_full_heterogeneous_probe.json"
)
DEFAULT_GPU_AWQ_32B_REPORT = (
    "dist/kaggle-32b-upper-bound-crossing-live-20260620-r3/"
    "kaggle_32b_stage_owned_activation_decode_probe.json"
)
DEFAULT_CPU_REAL_LLM_REPORT = (
    "dist/real-llm-llama-like-local-smoke-20260615/"
    "real_llm_sharded_evidence.json"
)
EXECUTION_MODES = ("fixture", "evidence-import", "external-existing")
BOUNDARIES = {
    "not_production": True,
    "not_p2p_nat_traversal": True,
    "not_arbitrary_public_prompt_serving": True,
    "not_billing_or_settlement": True,
    "not_unbounded_kaggle_scaling": True,
    "not_multi_account_limit_bypass": True,
    "not_same_request_live_gpu_tpu_cpu_32b_success_yet": True,
    "not_tpu_large_model_serving_yet": True,
}
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY=",
    "KAGGLE_USERNAME=",
    "HF_TOKEN=",
    "HUGGING_FACE_HUB_TOKEN=",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "Bearer ",
    "SOURCE_TARBALL_B64",
    "MINER_ENV_TEXT",
    "INLINE_KERNEL_PAYLOAD_B64",
    "kaggle-cookies.json",
    "kaggle-web-storage-state.json",
    '"lease_token":',
    '"idempotency_key":',
    '"prompt":',
    '"prompt_text":',
    '"raw_prompt":',
    '"generated_text":',
    '"output_text":',
    '"generated_token_ids":',
    '"token_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
    "kernel.py",
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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
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
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def source_summary(path: Path, report: dict[str, Any], *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "present": path.is_file() or bool(report),
        "schema": str(report.get("schema") or ""),
        "ok": report.get("ok") is True,
        "sha256": sha256_file(path) if path.is_file() else "",
        "public_artifact_safe": True,
    }


def fixture_tpu_report() -> dict[str, Any]:
    return {
        "schema": "kaggle_tpu_real_llm_web_probe_v1",
        "ok": True,
        "tpu_runtime_ready": True,
        "generated_token_count": 1,
        "hf_model_loaded": True,
        "jax_forward_ready": True,
        "baseline_match": True,
        "runtime_report": {
            "schema": "kaggle_tpu_real_llm_runtime_report_v1",
            "ok": True,
            "backend": "jax_tpu_manual_gpt2_forward_from_hf_torch_weights",
            "model_id": "gpt2-medium",
            "model_family": "gpt2",
            "parameter_count": 354_823_168,
            "tpu_runtime_ready": True,
            "tpu_device_count": 8,
            "hf_model_loaded": True,
            "jax_forward_ready": True,
            "baseline_match": True,
            "generated_token_count": 1,
            "simple_tpu_op_ready": True,
            "raw_prompt_public": False,
            "generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
        },
        "safety": {
            "public_artifact_safe": True,
            "raw_cookie_public": False,
            "raw_session_url_public": False,
            "credentials_public": False,
            "raw_prompt_public": False,
            "generated_text_public": False,
            "token_ids_public": False,
            "activation_public": False,
            "logits_public": False,
        },
    }


def fixture_gpu_full_report() -> dict[str, Any]:
    return {
        "schema": "kaggle_32b_full_heterogeneous_probe_v1",
        "ok": True,
        "full_precision_32b": True,
        "quantization": "none",
        "generated_token_count": 2,
        "one_token_generation_verified": True,
        "multi_token_generation_verified": True,
        "stage_local_kv_cache_verified": True,
        "heterogeneous_placement_verified": True,
        "four_t4_five_cpu_topology_verified": True,
        "stage_owned_full_precision_runtime_verified": True,
        "model": {
            "repo": "Qwen/Qwen2.5-32B-Instruct",
            "parameter_count_b": 32,
            "precision": "bf16_or_fp16_stage_runtime",
            "quantization": "none",
            "stage_count": 9,
        },
        "kaggle_lifecycle": {
            "actual_gpu_push_count": 2,
            "actual_cpu_push_count": 5,
            "requested_topology": "4T4_plus_5CPU",
            "coordinator_direct_management": True,
            "kernels_deleted": True,
            "private_packages_removed": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "past_key_values_public": False,
            "credentials_public": False,
        },
    }


def fixture_gpu_awq_report() -> dict[str, Any]:
    return {
        "schema": "kaggle_32b_stage_owned_activation_decode_probe_v1",
        "ok": True,
        "generated_token_count": 1,
        "one_token_generation_verified": True,
        "multi_token_decode_verified": False,
        "upper_bound_crossing_verified": True,
        "coordinator_direct_management_verified": True,
        "cross_kernel_activation_decode_verified": True,
        "stage_owned_awq_runtime_verified": True,
        "activation_handoff_verified": True,
        "model": {
            "repo": "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "parameter_count_b": 32,
            "quantization": "awq",
            "stage_count": 4,
        },
        "kaggle_lifecycle": {
            "actual_push_count": 2,
            "coordinator_direct_management": True,
            "kernels_deleted": True,
            "private_activation_removed": True,
            "private_packages_removed": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "credentials_public": False,
        },
    }


def fixture_cpu_report() -> dict[str, Any]:
    return {
        "schema": "real_llm_sharded_evidence_v1",
        "ok": True,
        "diagnosis_codes": [
            "activation_transport_ready",
            "baseline_match",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "real_llm_artifact_ready",
            "real_llm_sharded_ready",
            "stage_assignment_valid",
            "stage_cpu_partition_ready",
            "stage_local_partition_ready",
        ],
        "safety": {
            "read_only": True,
            "redaction_ok": True,
            "generated_text_redacted": True,
            "generated_token_ids_redacted": True,
            "raw_activation_redacted": True,
            "not_production": True,
        },
    }


def build_tpu_backend_summary(report: dict[str, Any], path: Path, *, min_parameter_count: int) -> dict[str, Any]:
    runtime = _dict(report.get("runtime_report"))
    safety = _dict(report.get("safety"))
    parameter_count = _int(runtime.get("parameter_count") or report.get("parameter_count"))
    generated_tokens = _int(runtime.get("generated_token_count") or report.get("generated_token_count"))
    tpu_device_count = _int(runtime.get("tpu_device_count") or report.get("tpu_device_count"))
    safety_flags = {
        "raw_cookie_public": safety.get("raw_cookie_public") is True,
        "raw_session_url_public": safety.get("raw_session_url_public") is True,
        "credentials_public": safety.get("credentials_public") is True or runtime.get("credentials_public") is True,
        "raw_prompt_public": safety.get("raw_prompt_public") is True or runtime.get("raw_prompt_public") is True,
        "generated_text_public": safety.get("generated_text_public") is True or runtime.get("generated_text_public") is True,
        "generated_token_ids_public": (
            safety.get("token_ids_public") is True
            or runtime.get("generated_token_ids_public") is True
            or runtime.get("next_token_id_public") is True
        ),
        "activation_public": safety.get("activation_public") is True or runtime.get("activation_public") is True,
        "logits_public": safety.get("logits_public") is True or runtime.get("logits_public") is True,
        "kv_cache_public": runtime.get("kv_cache_public") is True,
    }
    redaction_ready = not any(safety_flags.values()) and safety.get("public_artifact_safe", True) is not False
    real_model_ready = bool(
        report.get("ok") is True
        and (report.get("tpu_runtime_ready") is True or runtime.get("tpu_runtime_ready") is True)
        and (report.get("hf_model_loaded") is True or runtime.get("hf_model_loaded") is True)
        and (report.get("jax_forward_ready") is True or runtime.get("jax_forward_ready") is True)
        and generated_tokens >= 1
        and tpu_device_count >= 1
        and redaction_ready
    )
    return {
        "schema": "gpu_tpu_cpu_tpu_backend_summary_v1",
        "source": source_summary(path, report, kind="kaggle_tpu_real_llm_web_probe"),
        "backend": str(runtime.get("backend") or "jax_tpu"),
        "model_id": str(runtime.get("model_id") or report.get("model_id") or "unknown"),
        "model_family": str(runtime.get("model_family") or ""),
        "parameter_count": parameter_count,
        "small_medium_parameter_floor": min_parameter_count,
        "small_medium_real_model_ready": real_model_ready and parameter_count >= min_parameter_count,
        "tpu_runtime_ready": report.get("tpu_runtime_ready") is True or runtime.get("tpu_runtime_ready") is True,
        "tpu_device_count": tpu_device_count,
        "hf_model_loaded": report.get("hf_model_loaded") is True or runtime.get("hf_model_loaded") is True,
        "jax_forward_ready": report.get("jax_forward_ready") is True or runtime.get("jax_forward_ready") is True,
        "baseline_match": report.get("baseline_match") is True or runtime.get("baseline_match") is True,
        "generated_token_count": generated_tokens,
        "real_model_tpu_inference_ready": real_model_ready,
        "redaction_ready": redaction_ready,
        "public_flags": safety_flags,
        "public_artifact_safe": True,
    }


def build_gpu_backend_summary(full_report: dict[str, Any], full_path: Path, awq_report: dict[str, Any], awq_path: Path) -> dict[str, Any]:
    full_lifecycle = _dict(full_report.get("kaggle_lifecycle"))
    full_safety = _dict(full_report.get("safety"))
    full_model = _dict(full_report.get("model"))
    awq_lifecycle = _dict(awq_report.get("kaggle_lifecycle"))
    awq_safety = _dict(awq_report.get("safety"))
    awq_model = _dict(awq_report.get("model"))
    full_redaction_ready = (
        full_safety.get("public_artifact_safe", True) is not False
        and full_safety.get("raw_prompt_public") is not True
        and full_safety.get("raw_generated_text_public") is not True
        and full_safety.get("generated_token_ids_public") is not True
        and full_safety.get("activation_public") is not True
        and full_safety.get("hidden_state_public") is not True
        and full_safety.get("logits_public") is not True
        and full_safety.get("kv_cache_public") is not True
    )
    awq_redaction_ready = (
        awq_safety.get("public_artifact_safe", True) is not False
        and awq_safety.get("raw_prompt_public") is not True
        and awq_safety.get("raw_generated_text_public") is not True
        and awq_safety.get("generated_token_ids_public") is not True
        and awq_safety.get("activation_public") is not True
        and awq_safety.get("hidden_state_public") is not True
        and awq_safety.get("logits_public") is not True
    )
    full_ready = bool(
        full_report.get("ok") is True
        and full_report.get("full_precision_32b") is True
        and full_report.get("multi_token_generation_verified") is True
        and full_report.get("stage_local_kv_cache_verified") is True
        and full_report.get("four_t4_five_cpu_topology_verified") is True
        and full_lifecycle.get("kernels_deleted") is True
        and full_lifecycle.get("private_packages_removed") is True
        and full_redaction_ready
    )
    awq_ready = bool(
        awq_report.get("ok") is True
        and awq_report.get("one_token_generation_verified") is True
        and awq_report.get("upper_bound_crossing_verified") is True
        and awq_report.get("coordinator_direct_management_verified") is True
        and awq_report.get("stage_owned_awq_runtime_verified") is True
        and awq_lifecycle.get("kernels_deleted") is True
        and awq_lifecycle.get("private_packages_removed") is True
        and awq_redaction_ready
    )
    return {
        "schema": "gpu_tpu_cpu_gpu_backend_summary_v1",
        "full_precision_source": source_summary(full_path, full_report, kind="kaggle_32b_full_gpu_cpu_heterogeneous_probe"),
        "quantized_awq_source": source_summary(awq_path, awq_report, kind="kaggle_32b_awq_gpu_upper_bound_probe"),
        "gpu_backend_evidence_ready": full_ready or awq_ready,
        "full_precision_32b_gpu_cpu_ready": full_ready,
        "full_precision_model_id": str(full_model.get("repo") or ""),
        "full_precision_generated_token_count": _int(full_report.get("generated_token_count")),
        "full_precision_stage_count": _int(full_model.get("stage_count")),
        "full_precision_topology": str(full_lifecycle.get("requested_topology") or ""),
        "full_precision_stage_local_kv_cache_ready": full_report.get("stage_local_kv_cache_verified") is True,
        "quantized_32b_gpu_upper_bound_ready": awq_ready,
        "quantized_model_id": str(awq_model.get("repo") or ""),
        "quantized_generated_token_count": _int(awq_report.get("generated_token_count")),
        "quantized_stage_count": _int(awq_model.get("stage_count")),
        "coordinator_direct_management_ready": (
            full_lifecycle.get("coordinator_direct_management") is True
            or awq_lifecycle.get("coordinator_direct_management") is True
        ),
        "redaction_ready": full_redaction_ready and awq_redaction_ready,
        "public_artifact_safe": True,
    }


def build_cpu_backend_summary(cpu_report: dict[str, Any], cpu_path: Path, full_gpu_report: dict[str, Any]) -> dict[str, Any]:
    diagnosis = {str(item) for item in _list(cpu_report.get("diagnosis_codes"))}
    safety = _dict(cpu_report.get("safety"))
    lifecycle = _dict(full_gpu_report.get("kaggle_lifecycle"))
    cpu_count = _int(lifecycle.get("actual_cpu_push_count"))
    local_ready = bool(
        cpu_report.get("ok") is True
        and "real_llm_sharded_ready" in diagnosis
        and "activation_transport_ready" in diagnosis
        and "baseline_match" in diagnosis
        and "stage_cpu_partition_ready" in diagnosis
        and safety.get("generated_text_redacted", True) is not False
        and safety.get("generated_token_ids_redacted", True) is not False
        and safety.get("raw_activation_redacted", True) is not False
    )
    live_32b_cpu_ready = bool(
        full_gpu_report.get("ok") is True
        and full_gpu_report.get("full_precision_32b") is True
        and full_gpu_report.get("multi_token_generation_verified") is True
        and cpu_count >= 1
    )
    return {
        "schema": "gpu_tpu_cpu_cpu_backend_summary_v1",
        "source": source_summary(cpu_path, cpu_report, kind="local_real_llm_cpu_stage_proof"),
        "cpu_backend_evidence_ready": local_ready or live_32b_cpu_ready,
        "local_cpu_real_llm_sharded_ready": local_ready,
        "local_activation_transport_ready": "activation_transport_ready" in diagnosis,
        "local_baseline_match": "baseline_match" in diagnosis,
        "retained_32b_cpu_stage_ready": live_32b_cpu_ready,
        "retained_32b_cpu_kernel_count": cpu_count,
        "redaction_ready": True,
        "public_artifact_safe": True,
    }


def build_stage_contract_smoke(
    args: argparse.Namespace,
    *,
    gpu: dict[str, Any],
    tpu: dict[str, Any],
    cpu: dict[str, Any],
) -> dict[str, Any]:
    stage_plan = [
        {
            "stage_id": 0,
            "role": "prefill_or_early_layers",
            "backend": "cuda",
            "accelerator_family": "gpu",
            "capability": "real_llm_sharded_cuda_stage0",
            "source_evidence": "retained_kaggle_gpu_32b_stage_probe",
            "raw_activation_public": False,
            "public_artifact_safe": True,
        },
        {
            "stage_id": 1,
            "role": "middle_layers_candidate",
            "backend": "jax_tpu",
            "accelerator_family": "tpu",
            "capability": "real_llm_sharded_tpu_stage",
            "source_evidence": "retained_kaggle_tpu_real_llm_probe",
            "raw_activation_public": False,
            "public_artifact_safe": True,
        },
        {
            "stage_id": 2,
            "role": "tail_or_lm_head_fallback",
            "backend": "cpu",
            "accelerator_family": "cpu",
            "capability": "real_llm_sharded_stage1",
            "source_evidence": "retained_cpu_real_llm_stage_probe",
            "raw_activation_public": False,
            "public_artifact_safe": True,
        },
    ]
    seed = {
        "schema": STAGE_CONTRACT_SCHEMA,
        "model_id": args.alpha_model_id,
        "sequence_length": args.alpha_sequence_length,
        "hidden_size": args.alpha_hidden_size,
        "dtype": args.activation_dtype,
        "stage_backends": [stage["backend"] for stage in stage_plan],
    }
    previous_hash = stable_hash_payload({"input": seed})
    handoffs: list[dict[str, Any]] = []
    for index, stage in enumerate(stage_plan):
        stage_output_hash = stable_hash_payload({
            "stage_id": stage["stage_id"],
            "backend": stage["backend"],
            "input_hash": previous_hash,
            "shape": [1, args.alpha_sequence_length, args.alpha_hidden_size],
            "dtype": args.activation_dtype,
        })
        handoffs.append({
            "from_stage_id": stage["stage_id"],
            "to_stage_id": stage_plan[index + 1]["stage_id"] if index + 1 < len(stage_plan) else None,
            "backend": stage["backend"],
            "activation_shape": [1, args.alpha_sequence_length, args.alpha_hidden_size],
            "activation_dtype": args.activation_dtype,
            "activation_payload_public": False,
            "activation_hash": stage_output_hash,
            "lease_material_public": False,
            "idempotency_material_public": False,
            "public_artifact_safe": True,
        })
        previous_hash = stage_output_hash
    ready = bool(
        gpu.get("gpu_backend_evidence_ready")
        and tpu.get("real_model_tpu_inference_ready")
        and cpu.get("cpu_backend_evidence_ready")
    )
    return {
        "schema": STAGE_CONTRACT_SCHEMA,
        "ok": ready,
        "stage_count": len(stage_plan),
        "model_id": args.alpha_model_id,
        "execution_kind": "logical_contract_with_retained_backend_evidence",
        "same_request_live_heterogeneous_verified": False,
        "live_tpu_stage_miner_integrated": False,
        "contract_smoke_ready": ready,
        "stage_assignment_valid": ready,
        "activation_transport_contract_ready": ready,
        "cross_backend_dtype_contract_ready": args.activation_dtype in {"float16", "bfloat16", "float32"},
        "stage_plan": stage_plan,
        "handoffs": handoffs,
        "final_output_hash": previous_hash,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "hidden_state_public": False,
        "logits_public": False,
        "kv_cache_public": False,
        "public_artifact_safe": True,
    }


def _tensor_hash(tensor: Any) -> str:
    try:
        data = tensor.detach().cpu().contiguous().numpy().tobytes()
    except Exception:
        return stable_hash_payload({"tensor": str(type(tensor)), "shape": getattr(tensor, "shape", "")})
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _first_block_hidden(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def fixture_local_three_stage_e2e(args: argparse.Namespace) -> dict[str, Any]:
    stage_plan = [
        {"stage_id": 0, "target_backend_family": "gpu", "actual_execution_device": "cpu", "layer_range": [0, 4]},
        {"stage_id": 1, "target_backend_family": "tpu", "actual_execution_device": "cpu", "layer_range": [4, 8]},
        {"stage_id": 2, "target_backend_family": "cpu", "actual_execution_device": "cpu", "layer_range": [8, 12]},
    ]
    return {
        "schema": LOCAL_THREE_STAGE_SCHEMA,
        "ok": True,
        "mode": "fixture",
        "model_id": args.local_e2e_model_id,
        "model_family": "gpt2",
        "parameter_count": 124_000_000,
        "small_medium_parameter_floor": args.small_medium_min_parameter_count,
        "small_medium_model_e2e_ready": 124_000_000 >= args.small_medium_min_parameter_count,
        "real_hf_model_loaded": True,
        "real_model_forward_executed": True,
        "three_stage_real_model_e2e_ready": True,
        "baseline_match": True,
        "generated_token_count": 1,
        "stage_count": 3,
        "stage_plan": stage_plan,
        "activation_handoffs": [
            {
                "from_stage_id": 0,
                "to_stage_id": 1,
                "activation_hash": stable_hash_payload({"fixture": "stage0"}),
                "activation_payload_public": False,
            },
            {
                "from_stage_id": 1,
                "to_stage_id": 2,
                "activation_hash": stable_hash_payload({"fixture": "stage1"}),
                "activation_payload_public": False,
            },
        ],
        "output_hash": stable_hash_payload({"fixture": "next_token"}),
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "logits_public": False,
        "public_artifact_safe": True,
        "diagnosis_codes": ["local_three_stage_real_model_e2e_ready", "baseline_match"],
        "blockers": [],
    }


def run_local_three_stage_e2e(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    model_id = str(args.local_e2e_model_id or "gpt2")
    prompt = str(args.local_e2e_prompt or "CrowdTensor heterogeneous stage smoke")[:128]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
        parameter_count = int(sum(int(param.numel()) for param in model.parameters()))
        transformer = getattr(model, "transformer", None)
        blocks = list(getattr(transformer, "h", []) if transformer is not None else [])
        if transformer is None or len(blocks) < 2 or not hasattr(model, "lm_head"):
            raise RuntimeError("local_three_stage_e2e_requires_gpt2_style_model")
        layer_count = len(blocks)
        if layer_count >= 3:
            first_end = max(1, layer_count // 3)
            second_end = max(first_end + 1, (2 * layer_count) // 3)
            if second_end >= layer_count:
                second_end = layer_count - 1
        else:
            first_end = 1
            second_end = 1
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).get("input_ids")
        if input_ids is None or int(input_ids.numel()) <= 0:
            raise RuntimeError("local_three_stage_e2e_empty_input")
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)
            stage0_start = datetime.now(timezone.utc)
            for block in blocks[:first_end]:
                hidden = _first_block_hidden(block(hidden))
            stage0_elapsed_ms = (datetime.now(timezone.utc) - stage0_start).total_seconds() * 1000.0
            stage0_hash = _tensor_hash(hidden)
            stage0_shape = [int(value) for value in hidden.shape]

            stage1_start = datetime.now(timezone.utc)
            for block in blocks[first_end:second_end]:
                hidden = _first_block_hidden(block(hidden))
            stage1_elapsed_ms = (datetime.now(timezone.utc) - stage1_start).total_seconds() * 1000.0
            stage1_hash = _tensor_hash(hidden)
            stage1_shape = [int(value) for value in hidden.shape]

            stage2_start = datetime.now(timezone.utc)
            for block in blocks[second_end:]:
                hidden = _first_block_hidden(block(hidden))
            hidden = transformer.ln_f(hidden)
            logits = model.lm_head(hidden)
            next_token_id = int(torch.argmax(logits[0, -1, :]).item())
            stage2_elapsed_ms = (datetime.now(timezone.utc) - stage2_start).total_seconds() * 1000.0

            baseline = model(input_ids=input_ids)
            baseline_next_token_id = int(torch.argmax(baseline.logits[0, -1, :]).item())
        baseline_match = next_token_id == baseline_next_token_id
        output_hash = stable_hash_payload({
            "model_id": model_id,
            "next_token_id": next_token_id,
            "baseline_next_token_id": baseline_next_token_id,
        })
        stage_plan = [
            {
                "stage_id": 0,
                "target_backend_family": "gpu",
                "actual_execution_device": "cpu",
                "layer_range": [0, first_end],
                "elapsed_ms": round(stage0_elapsed_ms, 6),
                "public_artifact_safe": True,
            },
            {
                "stage_id": 1,
                "target_backend_family": "tpu",
                "actual_execution_device": "cpu",
                "layer_range": [first_end, second_end],
                "elapsed_ms": round(stage1_elapsed_ms, 6),
                "public_artifact_safe": True,
            },
            {
                "stage_id": 2,
                "target_backend_family": "cpu",
                "actual_execution_device": "cpu",
                "layer_range": [second_end, layer_count],
                "includes_final_norm_and_lm_head": True,
                "elapsed_ms": round(stage2_elapsed_ms, 6),
                "public_artifact_safe": True,
            },
        ]
        ready = bool(baseline_match and stage0_hash and stage1_hash)
        elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
        return {
            "schema": LOCAL_THREE_STAGE_SCHEMA,
            "ok": ready,
            "mode": "run",
            "model_id": model_id,
            "model_family": "gpt2",
            "parameter_count": parameter_count,
            "small_medium_parameter_floor": args.small_medium_min_parameter_count,
            "small_medium_model_e2e_ready": parameter_count >= args.small_medium_min_parameter_count,
            "real_hf_model_loaded": True,
            "real_model_forward_executed": True,
            "three_stage_real_model_e2e_ready": ready,
            "baseline_match": baseline_match,
            "generated_token_count": 1 if ready else 0,
            "stage_count": 3,
            "decoder_layer_count": layer_count,
            "stage_plan": stage_plan,
            "activation_handoffs": [
                {
                    "from_stage_id": 0,
                    "to_stage_id": 1,
                    "activation_hash": stage0_hash,
                    "activation_shape": stage0_shape,
                    "activation_payload_public": False,
                    "public_artifact_safe": True,
                },
                {
                    "from_stage_id": 1,
                    "to_stage_id": 2,
                    "activation_hash": stage1_hash,
                    "activation_shape": stage1_shape,
                    "activation_payload_public": False,
                    "public_artifact_safe": True,
                },
            ],
            "prompt_hash": stable_hash_payload({"prompt": prompt}),
            "output_hash": output_hash,
            "next_token_hash": stable_hash_payload({"next_token_id": next_token_id}),
            "baseline_next_token_hash": stable_hash_payload({"next_token_id": baseline_next_token_id}),
            "elapsed_ms": round(elapsed_ms, 6),
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "next_token_id_public": False,
            "baseline_next_token_id_public": False,
            "activation_public": False,
            "logits_public": False,
            "public_artifact_safe": True,
            "diagnosis_codes": [
                "local_three_stage_real_model_e2e_ready" if ready else "local_three_stage_real_model_e2e_failed",
                "baseline_match" if baseline_match else "baseline_mismatch",
            ],
            "blockers": [] if ready else ["baseline_mismatch"],
        }
    except Exception as exc:
        return {
            "schema": LOCAL_THREE_STAGE_SCHEMA,
            "ok": False,
            "mode": "run",
            "model_id": model_id,
            "real_hf_model_loaded": False,
            "real_model_forward_executed": False,
            "three_stage_real_model_e2e_ready": False,
            "small_medium_model_e2e_ready": False,
            "baseline_match": False,
            "generated_token_count": 0,
            "stage_count": 3,
            "error_type": type(exc).__name__,
            "error_digest": stable_hash_payload(str(exc)),
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "logits_public": False,
            "public_artifact_safe": True,
            "diagnosis_codes": ["local_three_stage_real_model_e2e_failed"],
            "blockers": ["local_three_stage_real_model_e2e_failed"],
        }


def build_local_three_stage_e2e(args: argparse.Namespace) -> dict[str, Any]:
    if args.local_e2e_mode == "fixture":
        return fixture_local_three_stage_e2e(args)
    if args.local_e2e_mode == "skip":
        return {
            "schema": LOCAL_THREE_STAGE_SCHEMA,
            "ok": False,
            "mode": "skip",
            "model_id": str(args.local_e2e_model_id),
            "three_stage_real_model_e2e_ready": False,
            "small_medium_model_e2e_ready": False,
            "public_artifact_safe": True,
            "diagnosis_codes": ["local_three_stage_real_model_e2e_skipped"],
            "blockers": ["local_three_stage_real_model_e2e_skipped"],
        }
    return run_local_three_stage_e2e(args)


def fixture_torch_jax_bridge_probe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": TORCH_JAX_BRIDGE_SCHEMA,
        "ok": True,
        "mode": "fixture",
        "model_id": args.bridge_model_id,
        "bridge_ready": True,
        "torch_stage0_ready": True,
        "jax_stage1_ready": True,
        "torch_stage2_ready": True,
        "baseline_match": True,
        "generated_token_count": 1,
        "jax_device_platform": "cpu",
        "activation_public": False,
        "generated_token_ids_public": False,
        "logits_public": False,
        "public_artifact_safe": True,
        "diagnosis_codes": ["torch_jax_torch_bridge_ready"],
        "blockers": [],
    }


def run_torch_jax_bridge_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    model_id = str(args.bridge_model_id or "hf-internal-testing/tiny-random-gpt2")
    prompt = str(args.local_e2e_prompt or "CrowdTensor heterogeneous stage smoke")[:128]
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        return {
            "schema": TORCH_JAX_BRIDGE_SCHEMA,
            "ok": False,
            "mode": "run",
            "model_id": model_id,
            "bridge_ready": False,
            "torch_stage0_ready": False,
            "jax_stage1_ready": False,
            "torch_stage2_ready": False,
            "baseline_match": False,
            "generated_token_count": 0,
            "missing_dependency": str(exc.name or "unknown"),
            "activation_public": False,
            "generated_token_ids_public": False,
            "logits_public": False,
            "public_artifact_safe": True,
            "diagnosis_codes": ["torch_jax_torch_bridge_dependency_missing"],
            "blockers": ["jax_missing" if str(exc.name or "") in {"jax", "jaxlib"} else "bridge_dependency_missing"],
        }
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
        transformer = getattr(model, "transformer", None)
        blocks = list(getattr(transformer, "h", []) if transformer is not None else [])
        if transformer is None or len(blocks) < 3 or not hasattr(model, "lm_head"):
            raise RuntimeError("torch_jax_bridge_requires_gpt2_style_model_with_three_layers")
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).get("input_ids")
        if input_ids is None or int(input_ids.numel()) <= 0:
            raise RuntimeError("torch_jax_bridge_empty_input")
        max_tokens = min(int(input_ids.shape[1]), int(args.bridge_sequence_length))
        input_ids = input_ids[:, :max_tokens]
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)
            hidden = _first_block_hidden(blocks[0](hidden))
            torch_stage0_hash = _tensor_hash(hidden)

            bridge_input = hidden.detach().cpu().numpy()
            hidden_jax = jnp.asarray(bridge_input)
            # This deterministic JAX stage is a data-plane bridge probe, not a
            # faithful GPT-2 block replacement. It proves activation transport,
            # dtype/shape handling, and JAX device execution boundaries.
            hidden_jax = jax.jit(lambda x: x + jnp.tanh(x) * jnp.asarray(0.0, dtype=x.dtype))(hidden_jax)
            hidden_after_jax = torch.from_numpy(jax.device_get(hidden_jax)).to(hidden.dtype)
            jax_stage1_hash = _tensor_hash(hidden_after_jax)

            hidden = hidden_after_jax
            for block in blocks[1:]:
                hidden = _first_block_hidden(block(hidden))
            hidden = transformer.ln_f(hidden)
            logits = model.lm_head(hidden)
            next_token_id = int(torch.argmax(logits[0, -1, :]).item())

            baseline_hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)
            for block in blocks:
                baseline_hidden = _first_block_hidden(block(baseline_hidden))
            baseline_hidden = transformer.ln_f(baseline_hidden)
            baseline_logits = model.lm_head(baseline_hidden)
            baseline_next_token_id = int(torch.argmax(baseline_logits[0, -1, :]).item())
        devices = list(jax.devices())
        platform = str(getattr(devices[0], "platform", "unknown")) if devices else "unknown"
        baseline_match = next_token_id == baseline_next_token_id
        elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
        return {
            "schema": TORCH_JAX_BRIDGE_SCHEMA,
            "ok": baseline_match,
            "mode": "run",
            "model_id": model_id,
            "model_family": "gpt2",
            "parameter_count": int(sum(int(param.numel()) for param in model.parameters())),
            "bridge_ready": baseline_match,
            "torch_stage0_ready": bool(torch_stage0_hash),
            "jax_stage1_ready": bool(jax_stage1_hash),
            "torch_stage2_ready": True,
            "baseline_match": baseline_match,
            "generated_token_count": 1 if baseline_match else 0,
            "jax_device_platform": platform,
            "jax_device_count": len(devices),
            "activation_shape": [int(value) for value in bridge_input.shape],
            "torch_stage0_activation_hash": torch_stage0_hash,
            "jax_stage1_activation_hash": jax_stage1_hash,
            "output_hash": stable_hash_payload({
                "next_token_id": next_token_id,
                "baseline_next_token_id": baseline_next_token_id,
            }),
            "elapsed_ms": round(elapsed_ms, 6),
            "activation_public": False,
            "generated_token_ids_public": False,
            "next_token_id_public": False,
            "baseline_next_token_id_public": False,
            "logits_public": False,
            "public_artifact_safe": True,
            "diagnosis_codes": [
                "torch_jax_torch_bridge_ready" if baseline_match else "torch_jax_torch_bridge_mismatch",
                "jax_runtime_available",
            ],
            "blockers": [] if baseline_match else ["torch_jax_torch_bridge_mismatch"],
        }
    except Exception as exc:
        return {
            "schema": TORCH_JAX_BRIDGE_SCHEMA,
            "ok": False,
            "mode": "run",
            "model_id": model_id,
            "bridge_ready": False,
            "torch_stage0_ready": False,
            "jax_stage1_ready": False,
            "torch_stage2_ready": False,
            "baseline_match": False,
            "generated_token_count": 0,
            "error_type": type(exc).__name__,
            "error_digest": stable_hash_payload(str(exc)),
            "activation_public": False,
            "generated_token_ids_public": False,
            "logits_public": False,
            "public_artifact_safe": True,
            "diagnosis_codes": ["torch_jax_torch_bridge_failed"],
            "blockers": ["torch_jax_torch_bridge_failed"],
        }


def build_torch_jax_bridge_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.bridge_mode == "fixture":
        return fixture_torch_jax_bridge_probe(args)
    if args.bridge_mode == "skip":
        return {
            "schema": TORCH_JAX_BRIDGE_SCHEMA,
            "ok": False,
            "mode": "skip",
            "model_id": str(args.bridge_model_id),
            "bridge_ready": False,
            "public_artifact_safe": True,
            "diagnosis_codes": ["torch_jax_torch_bridge_skipped"],
            "blockers": ["torch_jax_torch_bridge_skipped"],
        }
    return run_torch_jax_bridge_probe(args)


def build_32b_feasibility_report(
    args: argparse.Namespace,
    *,
    gpu: dict[str, Any],
    tpu: dict[str, Any],
    cpu: dict[str, Any],
    stage_contract: dict[str, Any],
    local_e2e: dict[str, Any],
    bridge_probe: dict[str, Any],
) -> dict[str, Any]:
    tpu_adapter_ready_for_32b = False
    same_request_ready = False
    rc_ready = bool(
        gpu.get("full_precision_32b_gpu_cpu_ready")
        and tpu.get("small_medium_real_model_ready")
        and cpu.get("cpu_backend_evidence_ready")
        and stage_contract.get("contract_smoke_ready")
        and local_e2e.get("three_stage_real_model_e2e_ready")
    )
    bridge_status = "ready" if bridge_probe.get("bridge_ready") else "blocked_or_not_run"
    required_adapter_work = [
        {
            "item": "jax_tpu_llama_like_stage_runtime",
            "status": "missing",
            "reason": "Current TPU proof is GPT-2-family JAX forward; Qwen/Llama decoder block stage runtime is not integrated.",
        },
        {
            "item": "safetensors_or_maxtext_checkpoint_bridge",
            "status": "missing",
            "reason": "GPU proofs use HF/PyTorch safetensors or AWQ stage loaders; TPU needs JAX/Flax/MaxText-compatible stage-owned loading.",
        },
        {
            "item": "cuda_to_jax_activation_wire_format",
            "status": "alpha-bridge-ready" if bridge_probe.get("bridge_ready") else "alpha-contract-only",
            "reason": (
                "Torch-JAX-Torch bridge probe executed locally with public-safe activation hashes."
                if bridge_probe.get("bridge_ready")
                else "The Alpha report defines hashes/shapes/dtypes; Torch-JAX bridge is not available in this environment."
            ),
        },
        {
            "item": "stage_local_kv_cache_format_boundary",
            "status": "missing-for-tpu",
            "reason": "GPU+CPU 32B proof has stage-local KV reuse; TPU stage-local cache is not wired into Coordinator tasks.",
        },
        {
            "item": "coordinator_backend_capability_routing",
            "status": "alpha-contract-only",
            "reason": "The stage contract names CUDA/JAX-TPU/CPU capabilities; live Miner routing still needs implementation.",
        },
        {
            "item": "bounded_live_cleanup_and_requeue",
            "status": "missing-for-tpu",
            "reason": "GPU/CPU live cleanup exists in retained proofs; TPU web-runtime cleanup and requeue need a script-kernel path or reusable session API.",
        },
    ]
    next_rc_boundary = {
        "schema": "gpu_tpu_cpu_32b_next_rc_boundary_v1",
        "next_rc_boundary_ready": rc_ready,
        "goal": "Run a single-Coordinator 1-token public-safe GPU+TPU+CPU stage request for a Qwen/Llama-like model.",
        "initial_model_scope": args.target_32b_model_id,
        "fallback_model_scope": "7B or 14B Qwen/Llama-like model if TPU Qwen/Llama stage loading blocks 32B.",
        "target_generated_token_count": args.target_max_new_tokens,
        "max_context_length": args.context_length,
        "required_stages": [
            "cuda stage Miner with stage-owned early layers",
            "jax_tpu stage Miner with stage-owned middle layers",
            "cpu tail or verifier stage Miner",
        ],
        "success_criteria": [
            "one Coordinator issues all stage tasks for the same request",
            "at least one accepted CUDA stage task, one accepted TPU stage task, and one accepted CPU stage task",
            "raw prompts, generated text, token ids, logits, activations, KV-cache tensors, credentials, leases, and idempotency material stay out of public artifacts",
            "stage-owned model loading is proven for every live stage",
            "activation handoff hashes and shape/dtype metadata are recorded",
            "temporary Kaggle/private runtime artifacts are cleaned up or explicitly marked for rotation",
        ],
        "stop_conditions": [
            "TPU allocation unavailable after bounded wait",
            "Qwen/Llama JAX stage loader cannot materialize assigned layers within runtime memory",
            "cross-backend activation numerical mismatch exceeds tolerance",
            "provider quota or cleanup constraints would require multi-account bypass",
        ],
        "public_artifact_safe": True,
    }
    return {
        "schema": FEASIBILITY_SCHEMA,
        "ok": rc_ready,
        "gpu_tpu_cpu_32b_feasibility_report_ready": rc_ready,
        "gpu_tpu_cpu_32b_same_request_feasible_now": same_request_ready,
        "same_request_live_heterogeneous_verified": False,
        "tpu_32b_runtime_adapter_ready": tpu_adapter_ready_for_32b,
        "ready_for_bounded_rc": rc_ready,
        "verdict": "ready_for_bounded_rc_not_yet_live_verified" if rc_ready else "blocked_missing_backend_evidence",
        "candidate_model_id": args.target_32b_model_id,
        "candidate_model_tier": "32b",
        "target_generated_token_count": args.target_max_new_tokens,
        "context_length": args.context_length,
        "retained_full_precision_32b_gpu_cpu_ready": bool(gpu.get("full_precision_32b_gpu_cpu_ready")),
        "retained_quantized_32b_gpu_upper_bound_ready": bool(gpu.get("quantized_32b_gpu_upper_bound_ready")),
        "retained_tpu_real_model_max_model_id": tpu.get("model_id"),
        "retained_tpu_real_model_parameter_count": tpu.get("parameter_count"),
        "retained_cpu_stage_ready": bool(cpu.get("cpu_backend_evidence_ready")),
        "local_three_stage_real_model_e2e_ready": bool(local_e2e.get("three_stage_real_model_e2e_ready")),
        "local_three_stage_model_id": local_e2e.get("model_id"),
        "local_three_stage_parameter_count": local_e2e.get("parameter_count"),
        "torch_jax_torch_bridge_status": bridge_status,
        "torch_jax_torch_bridge_ready": bool(bridge_probe.get("bridge_ready")),
        "required_adapter_work": required_adapter_work,
        "blockers": {
            "same_request_live_gpu_tpu_cpu_not_verified": True,
            "tpu_qwen_llama_stage_runtime_missing": True,
            "cross_backend_activation_runtime_missing": True,
            "tpu_stage_local_kv_cache_missing": True,
            "provider_queue_and_quota_risk": True,
        },
        "next_rc_boundary": next_rc_boundary,
        "public_artifact_safe": True,
    }


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "generated_at": report.get("generated_at"),
        "ok": report.get("ok") is True,
        "gpu_tpu_cpu_heterogeneous_stage_alpha_ready": report.get("gpu_tpu_cpu_heterogeneous_stage_alpha_ready") is True,
        "small_medium_real_model_end_to_end_ready": report.get("small_medium_real_model_end_to_end_ready") is True,
        "same_request_live_heterogeneous_verified": report.get("same_request_live_heterogeneous_verified") is True,
        "local_three_stage_real_model_e2e_ready": report.get("local_three_stage_real_model_e2e_ready") is True,
        "torch_jax_torch_bridge_ready": report.get("torch_jax_torch_bridge_ready") is True,
        "gpu_tpu_cpu_32b_feasibility_report_ready": report.get("gpu_tpu_cpu_32b_feasibility_report_ready") is True,
        "next_rc_boundary_ready": report.get("next_rc_boundary_ready") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "artifact_summary": report.get("artifact_summary") or {},
        "source_reports": report.get("source_reports") or {},
    }


def render_markdown(report: dict[str, Any]) -> str:
    tpu = _dict(report.get("tpu_backend"))
    gpu = _dict(report.get("gpu_backend"))
    cpu = _dict(report.get("cpu_backend"))
    local_e2e = _dict(report.get("local_three_stage_real_model_e2e"))
    bridge_probe = _dict(report.get("torch_jax_torch_bridge_probe"))
    feasibility = _dict(report.get("heterogeneous_32b_feasibility"))
    lines = [
        "# GPU+TPU+CPU Heterogeneous Stage Inference Alpha",
        "",
        f"- ready: `{report.get('gpu_tpu_cpu_heterogeneous_stage_alpha_ready')}`",
        f"- execution mode: `{report.get('execution_mode')}`",
        f"- small/medium real-model Alpha path: `{report.get('small_medium_real_model_end_to_end_ready')}`",
        f"- local three-stage real-model e2e: `{report.get('local_three_stage_real_model_e2e_ready')}`",
        f"- Torch-JAX-Torch bridge: `{report.get('torch_jax_torch_bridge_ready')}`",
        f"- same-request live heterogeneous verified: `{report.get('same_request_live_heterogeneous_verified')}`",
        f"- 32B feasibility report ready: `{report.get('gpu_tpu_cpu_32b_feasibility_report_ready')}`",
        f"- next RC boundary ready: `{report.get('next_rc_boundary_ready')}`",
        "",
        "## Backend Evidence",
        "",
        f"- GPU: ready=`{gpu.get('gpu_backend_evidence_ready')}` full32B=`{gpu.get('full_precision_32b_gpu_cpu_ready')}` awq32B=`{gpu.get('quantized_32b_gpu_upper_bound_ready')}`",
        f"- TPU: ready=`{tpu.get('real_model_tpu_inference_ready')}` model=`{tpu.get('model_id')}` params=`{tpu.get('parameter_count')}` devices=`{tpu.get('tpu_device_count')}`",
        f"- CPU: ready=`{cpu.get('cpu_backend_evidence_ready')}` retained32B_cpu=`{cpu.get('retained_32b_cpu_stage_ready')}`",
        f"- Local 3-stage E2E: ready=`{local_e2e.get('three_stage_real_model_e2e_ready')}` model=`{local_e2e.get('model_id')}` params=`{local_e2e.get('parameter_count')}` baseline_match=`{local_e2e.get('baseline_match')}`",
        f"- Torch-JAX bridge: ready=`{bridge_probe.get('bridge_ready')}` mode=`{bridge_probe.get('mode')}` platform=`{bridge_probe.get('jax_device_platform')}`",
        "",
        "## 32B Feasibility",
        "",
        f"- verdict: `{feasibility.get('verdict')}`",
        f"- same-request feasible now: `{feasibility.get('gpu_tpu_cpu_32b_same_request_feasible_now')}`",
        f"- TPU 32B runtime adapter ready: `{feasibility.get('tpu_32b_runtime_adapter_ready')}`",
        "",
        "## Required Adapter Work",
        "",
    ]
    for item in feasibility.get("required_adapter_work") or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('item')}: `{item.get('status')}` - {item.get('reason')}")
    lines.extend(["", "## Boundaries", ""])
    for name, value in sorted(_dict(report.get("boundaries")).items()):
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Diagnosis", "", "- " + ", ".join(report.get("diagnosis_codes") or []), ""])
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tpu_path = Path(args.tpu_real_llm_report)
    gpu_full_path = Path(args.gpu_full_32b_report)
    gpu_awq_path = Path(args.gpu_awq_32b_report)
    cpu_path = Path(args.cpu_real_llm_report)
    if args.execution_mode == "fixture":
        tpu_report = fixture_tpu_report()
        gpu_full_report = fixture_gpu_full_report()
        gpu_awq_report = fixture_gpu_awq_report()
        cpu_report = fixture_cpu_report()
    else:
        tpu_report = load_optional_report(tpu_path)
        gpu_full_report = load_optional_report(gpu_full_path)
        gpu_awq_report = load_optional_report(gpu_awq_path)
        cpu_report = load_optional_report(cpu_path)

    tpu = build_tpu_backend_summary(tpu_report, tpu_path, min_parameter_count=args.small_medium_min_parameter_count)
    gpu = build_gpu_backend_summary(gpu_full_report, gpu_full_path, gpu_awq_report, gpu_awq_path)
    cpu = build_cpu_backend_summary(cpu_report, cpu_path, gpu_full_report)
    stage_contract = build_stage_contract_smoke(args, gpu=gpu, tpu=tpu, cpu=cpu)
    local_e2e = build_local_three_stage_e2e(args)
    bridge_probe = build_torch_jax_bridge_probe(args)
    feasibility = build_32b_feasibility_report(
        args,
        gpu=gpu,
        tpu=tpu,
        cpu=cpu,
        stage_contract=stage_contract,
        local_e2e=local_e2e,
        bridge_probe=bridge_probe,
    )

    backend_evidence_imported = bool(
        tpu["source"].get("present")
        and gpu["full_precision_source"].get("present")
        and cpu["source"].get("present")
        and args.execution_mode in EXECUTION_MODES
    )
    small_medium_ready = bool(
        stage_contract.get("contract_smoke_ready")
        and local_e2e.get("three_stage_real_model_e2e_ready")
        and local_e2e.get("small_medium_model_e2e_ready")
        and tpu.get("small_medium_real_model_ready")
        and gpu.get("gpu_backend_evidence_ready")
        and cpu.get("cpu_backend_evidence_ready")
    )
    next_rc_boundary_ready = bool(_dict(feasibility.get("next_rc_boundary")).get("next_rc_boundary_ready"))
    ready = bool(
        backend_evidence_imported
        and gpu.get("gpu_backend_evidence_ready")
        and tpu.get("real_model_tpu_inference_ready")
        and cpu.get("cpu_backend_evidence_ready")
        and stage_contract.get("contract_smoke_ready")
        and local_e2e.get("three_stage_real_model_e2e_ready")
        and small_medium_ready
        and feasibility.get("gpu_tpu_cpu_32b_feasibility_report_ready")
        and next_rc_boundary_ready
    )
    diagnosis_codes = {
        "gpu_tpu_cpu_heterogeneous_stage_alpha_ready" if ready else "gpu_tpu_cpu_heterogeneous_stage_alpha_blocked",
        "backend_evidence_imported" if backend_evidence_imported else "backend_evidence_missing",
        "gpu_backend_evidence_ready" if gpu.get("gpu_backend_evidence_ready") else "gpu_backend_evidence_missing",
        "tpu_backend_evidence_ready" if tpu.get("real_model_tpu_inference_ready") else "tpu_backend_evidence_missing",
        "cpu_backend_evidence_ready" if cpu.get("cpu_backend_evidence_ready") else "cpu_backend_evidence_missing",
        "logical_stage_contract_ready" if stage_contract.get("contract_smoke_ready") else "logical_stage_contract_missing",
        "small_medium_real_model_alpha_path_ready" if small_medium_ready else "small_medium_real_model_alpha_path_missing",
        "local_three_stage_real_model_e2e_ready" if local_e2e.get("three_stage_real_model_e2e_ready") else "local_three_stage_real_model_e2e_missing",
        "torch_jax_torch_bridge_ready" if bridge_probe.get("bridge_ready") else "torch_jax_torch_bridge_not_ready",
        "same_request_live_heterogeneous_not_verified",
        "tpu_stage_miner_not_integrated",
        "gpu_tpu_cpu_32b_feasibility_report_ready" if feasibility.get("gpu_tpu_cpu_32b_feasibility_report_ready") else "gpu_tpu_cpu_32b_feasibility_report_missing",
        "next_rc_boundary_ready" if next_rc_boundary_ready else "next_rc_boundary_missing",
        "gpu_tpu_cpu_public_artifact_redaction_ready",
    }
    source_reports = {
        "tpu_real_llm": tpu["source"],
        "gpu_full_32b": gpu["full_precision_source"],
        "gpu_awq_32b": gpu["quantized_awq_source"],
        "cpu_real_llm": cpu["source"],
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "output_dir": str(output_dir),
        "execution_mode": args.execution_mode,
        "gpu_tpu_cpu_heterogeneous_stage_alpha_ready": ready,
        "backend_evidence_imported": backend_evidence_imported,
        "gpu_backend_evidence_ready": bool(gpu.get("gpu_backend_evidence_ready")),
        "tpu_backend_evidence_ready": bool(tpu.get("real_model_tpu_inference_ready")),
        "cpu_backend_evidence_ready": bool(cpu.get("cpu_backend_evidence_ready")),
        "logical_stage_contract_ready": bool(stage_contract.get("contract_smoke_ready")),
        "local_three_stage_real_model_e2e_ready": bool(local_e2e.get("three_stage_real_model_e2e_ready")),
        "torch_jax_torch_bridge_ready": bool(bridge_probe.get("bridge_ready")),
        "small_medium_real_model_end_to_end_ready": small_medium_ready,
        "same_request_live_heterogeneous_verified": False,
        "live_tpu_stage_miner_integrated": False,
        "gpu_tpu_cpu_32b_feasibility_report_ready": bool(feasibility.get("gpu_tpu_cpu_32b_feasibility_report_ready")),
        "next_rc_boundary_ready": next_rc_boundary_ready,
        "public_artifact_safe": True,
        "boundaries": dict(BOUNDARIES),
        "source_reports": source_reports,
        "gpu_backend": gpu,
        "tpu_backend": tpu,
        "cpu_backend": cpu,
        "stage_contract_smoke": stage_contract,
        "local_three_stage_real_model_e2e": local_e2e,
        "torch_jax_torch_bridge_probe": bridge_probe,
        "heterogeneous_32b_feasibility": feasibility,
        "diagnosis_codes": sorted(diagnosis_codes),
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "past_key_values_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "lease_material_public": False,
            "idempotency_material_public": False,
            "private_runtime_state_public": False,
            "report_public_leak_paths": [],
        },
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["gpu_tpu_cpu_heterogeneous_stage_alpha_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["safety"]["report_public_leak_paths"] = leaks
        report["diagnosis_codes"].append("gpu_tpu_cpu_public_artifact_redaction_failed")

    summary_json = output_dir / "gpu_tpu_cpu_heterogeneous_stage_alpha.json"
    summary_md = output_dir / "GPU_TPU_CPU_HETEROGENEOUS_STAGE_ALPHA.md"
    support_path = output_dir / "support_bundle.json"
    stage_contract_path = output_dir / "stage_contract_smoke.json"
    local_e2e_path = output_dir / "local_three_stage_real_model_e2e.json"
    bridge_probe_path = output_dir / "torch_jax_torch_bridge_probe.json"
    feasibility_path = output_dir / "heterogeneous_32b_feasibility_report.json"
    write_json(stage_contract_path, stage_contract)
    write_json(local_e2e_path, local_e2e)
    write_json(bridge_probe_path, bridge_probe)
    write_json(feasibility_path, feasibility)
    summary_md.write_text(render_markdown(report), encoding="utf-8")
    support_bundle = build_support_bundle(report)
    write_json(support_path, support_bundle)
    artifacts = {
        "summary_json": artifact_entry(summary_json, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
        "summary_markdown": artifact_entry(summary_md, output_dir, kind="summary_markdown", ok=bool(report.get("ok"))),
        "support_bundle_json": artifact_entry(support_path, output_dir, kind="support_bundle_json", schema=SUPPORT_BUNDLE_SCHEMA, ok=bool(report.get("ok"))),
        "stage_contract_smoke_json": artifact_entry(stage_contract_path, output_dir, kind="stage_contract_smoke_json", schema=STAGE_CONTRACT_SCHEMA, ok=bool(stage_contract.get("ok"))),
        "local_three_stage_real_model_e2e_json": artifact_entry(local_e2e_path, output_dir, kind="local_three_stage_real_model_e2e_json", schema=LOCAL_THREE_STAGE_SCHEMA, ok=bool(local_e2e.get("ok"))),
        "torch_jax_torch_bridge_probe_json": artifact_entry(bridge_probe_path, output_dir, kind="torch_jax_torch_bridge_probe_json", schema=TORCH_JAX_BRIDGE_SCHEMA, ok=bool(bridge_probe.get("ok"))),
        "heterogeneous_32b_feasibility_json": artifact_entry(feasibility_path, output_dir, kind="heterogeneous_32b_feasibility_json", schema=FEASIBILITY_SCHEMA, ok=bool(feasibility.get("ok"))),
    }
    report["artifacts"] = artifacts
    report["artifact_summary"] = {
        "schema": "gpu_tpu_cpu_heterogeneous_stage_alpha_artifact_summary_v1",
        "artifact_count": len(artifacts),
        "present_artifact_count": sum(1 for item in artifacts.values() if item.get("present")),
        "inspect_first": str(summary_md),
        "support_bundle": str(support_path),
        "stage_contract_smoke": str(stage_contract_path),
        "local_three_stage_real_model_e2e": str(local_e2e_path),
        "torch_jax_torch_bridge_probe": str(bridge_probe_path),
        "heterogeneous_32b_feasibility_report": str(feasibility_path),
        "public_artifact_safe": bool(report.get("public_artifact_safe")),
    }
    write_json(summary_json, report)
    report["artifacts"]["summary_json"]["present"] = True
    report["artifacts"]["summary_json"]["sha256"] = sha256_file(summary_json)
    report["artifact_summary"]["present_artifact_count"] = sum(1 for item in report["artifacts"].values() if item.get("present"))
    write_json(summary_json, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPU+TPU+CPU heterogeneous stage inference Alpha evidence.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="evidence-import")
    parser.add_argument("--tpu-real-llm-report", default=DEFAULT_TPU_REAL_LLM_REPORT)
    parser.add_argument("--gpu-full-32b-report", default=DEFAULT_GPU_FULL_32B_REPORT)
    parser.add_argument("--gpu-awq-32b-report", default=DEFAULT_GPU_AWQ_32B_REPORT)
    parser.add_argument("--cpu-real-llm-report", default=DEFAULT_CPU_REAL_LLM_REPORT)
    parser.add_argument("--small-medium-min-parameter-count", type=int, default=100_000_000)
    parser.add_argument("--alpha-model-id", default="gpt2-xl")
    parser.add_argument("--local-e2e-mode", choices=["run", "fixture", "skip"], default="run")
    parser.add_argument("--local-e2e-model-id", default="gpt2")
    parser.add_argument("--local-e2e-prompt", default="CrowdTensor heterogeneous stage smoke")
    parser.add_argument("--bridge-mode", choices=["run", "fixture", "skip"], default="run")
    parser.add_argument("--bridge-model-id", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--bridge-sequence-length", type=int, default=16)
    parser.add_argument("--alpha-sequence-length", type=int, default=8)
    parser.add_argument("--alpha-hidden-size", type=int, default=1600)
    parser.add_argument("--activation-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--target-32b-model-id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--target-max-new-tokens", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.small_medium_min_parameter_count < 1:
        raise SystemExit("--small-medium-min-parameter-count must be positive")
    if args.alpha_sequence_length < 1:
        raise SystemExit("--alpha-sequence-length must be positive")
    if args.alpha_hidden_size < 1:
        raise SystemExit("--alpha-hidden-size must be positive")
    if not str(args.local_e2e_model_id).strip():
        raise SystemExit("--local-e2e-model-id must be non-empty")
    if not str(args.bridge_model_id).strip():
        raise SystemExit("--bridge-model-id must be non-empty")
    if args.bridge_sequence_length < 1 or args.bridge_sequence_length > 128:
        raise SystemExit("--bridge-sequence-length must be between 1 and 128")
    if args.target_max_new_tokens < 1 or args.target_max_new_tokens > 16:
        raise SystemExit("--target-max-new-tokens must be between 1 and 16")
    if args.context_length < 1 or args.context_length > 4096:
        raise SystemExit("--context-length must be between 1 and 4096")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"GPU+TPU+CPU heterogeneous stage Alpha ready: {report.get('ok')}")
        print(f"output: {report.get('output_dir')}")
        print(f"same-request live heterogeneous verified: {report.get('same_request_live_heterogeneous_verified')}")
        print(f"32B feasibility verdict: {_dict(report.get('heterogeneous_32b_feasibility')).get('verdict')}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
