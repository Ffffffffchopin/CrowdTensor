#!/usr/bin/env python3
"""Build GPU+TPU+CPU 32B heterogeneous stage-inference RC evidence."""

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

from scripts import gpu_tpu_cpu_heterogeneous_stage_alpha_pack as alpha_pack  # noqa: E402


SCHEMA = "gpu_tpu_cpu_32b_heterogeneous_rc_v1"
SUPPORT_BUNDLE_SCHEMA = "gpu_tpu_cpu_32b_heterogeneous_rc_support_bundle_v1"
ACTIVATION_PROTOCOL_SCHEMA = "gpu_tpu_cpu_32b_activation_protocol_v1"
LIVE_PROOF_SCHEMA = "gpu_tpu_cpu_32b_same_request_live_proof_v1"
TPU_ALLOCATION_SUMMARY_SCHEMA = "gpu_tpu_cpu_32b_tpu_allocation_attempt_summary_v1"
TPU_WEB_ACTIVE_EVENT_SUMMARY_SCHEMA = "gpu_tpu_cpu_32b_tpu_web_active_event_summary_v1"
TPU_STAGE_ADAPTER_PLAN_SUMMARY_SCHEMA = "gpu_tpu_cpu_32b_tpu_stage_adapter_plan_summary_v1"
TPU_STAGE_RUNTIME_PROBE_SUMMARY_SCHEMA = "gpu_tpu_cpu_32b_tpu_stage_runtime_probe_summary_v1"
TPU_STAGE_32B_LOADER_PROBE_SUMMARY_SCHEMA = "gpu_tpu_cpu_32b_tpu_stage_loader_probe_summary_v1"
RUNTIME_BRIDGE_SUMMARY_SCHEMA = "gpu_tpu_cpu_32b_runtime_bridge_summary_v1"
DEFAULT_OUTPUT_DIR = "dist/gpu-tpu-cpu-32b-heterogeneous-rc"
DEFAULT_ALPHA_REPORT = (
    "dist/gpu-tpu-cpu-heterogeneous-stage-alpha-20260622-r3-cli/"
    "gpu_tpu_cpu_heterogeneous_stage_alpha.json"
)
EXECUTION_MODES = ("fixture", "evidence-import", "external-existing")
LIVE_PROOF_MODES = ("none", "fixture-success", "fixture-fallback", "external")
TARGET_32B_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MIN_32B_PARAMETER_COUNT = 30_000_000_000

SENSITIVE_FRAGMENTS = alpha_pack.SENSITIVE_FRAGMENTS + (
    "COOKIE",
    "Set-Cookie",
    "Authorization:",
    "KAGGLE_SESSION",
    "X-CrowdTensor-32B-Token",
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


def fixture_alpha_report() -> dict[str, Any]:
    args = alpha_pack.parse_args([
        "--output-dir",
        str(Path("/tmp/crowdtensor_gpu_tpu_cpu_32b_rc_fixture_alpha")),
        "--execution-mode",
        "fixture",
        "--local-e2e-mode",
        "fixture",
        "--bridge-mode",
        "fixture",
    ])
    return alpha_pack.build_report(args)


def fixture_live_success_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": LIVE_PROOF_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model_id": args.target_32b_model_id,
        "model_parameter_count": MIN_32B_PARAMETER_COUNT + 2_000_000_000,
        "model_tier": "32b",
        "generated_token_count": args.target_max_new_tokens,
        "context_length": args.context_length,
        "gpu_tpu_cpu_32b_same_request_verified": True,
        "live_tpu_stage_miner_integrated": True,
        "tpu_32b_runtime_adapter_ready": True,
        "fallback_model_used": False,
        "stage_local_kv_cache_verified": True,
        "accepted_stage_tasks": [
            {"stage_id": 0, "backend": "cuda", "accepted": True, "stage_owned_model_loaded": True},
            {"stage_id": 1, "backend": "jax_tpu", "accepted": True, "stage_owned_model_loaded": True},
            {"stage_id": 2, "backend": "cpu", "accepted": True, "stage_owned_model_loaded": True},
        ],
        "stage_task_counts": {"cuda": 1, "jax_tpu": 1, "cpu": 1},
        "activation_handoffs": [
            {
                "from_backend": "cuda",
                "to_backend": "jax_tpu",
                "activation_hash": stable_hash_payload({"hop": "cuda_to_jax_tpu", "fixture": True}),
                "activation_shape": [1, args.context_length, 5120],
                "activation_dtype": "bfloat16",
                "activation_layout": "batch_seq_hidden",
                "activation_payload_public": False,
            },
            {
                "from_backend": "jax_tpu",
                "to_backend": "cpu",
                "activation_hash": stable_hash_payload({"hop": "jax_tpu_to_cpu", "fixture": True}),
                "activation_shape": [1, args.context_length, 5120],
                "activation_dtype": "bfloat16",
                "activation_layout": "batch_seq_hidden",
                "activation_payload_public": False,
            },
        ],
        "runtime_device_summary": {"cuda_gpu_count": 1, "tpu_device_count": 8, "cpu_stage_count": 1},
        "cleanup": {
            "private_runtime_artifacts_cleaned": True,
            "temporary_kaggle_kernels_deleted": True,
            "token_rotation_required": False,
        },
        "safety": default_safety_flags(),
    }


def fixture_fallback_report(args: argparse.Namespace) -> dict[str, Any]:
    report = fixture_live_success_report(args)
    report.update({
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "model_parameter_count": 14_700_000_000,
        "model_tier": "14b",
        "gpu_tpu_cpu_32b_same_request_verified": False,
        "fallback_model_used": True,
        "tpu_32b_runtime_adapter_ready": False,
    })
    return report


def default_safety_flags() -> dict[str, bool]:
    return {
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
    }


def build_alpha_import(args: argparse.Namespace, alpha_report: dict[str, Any], alpha_path: Path) -> dict[str, Any]:
    feasibility = _dict(alpha_report.get("heterogeneous_32b_feasibility"))
    return {
        "schema": "gpu_tpu_cpu_32b_rc_alpha_import_v1",
        "source": source_summary(alpha_path, alpha_report, kind="gpu_tpu_cpu_heterogeneous_stage_alpha"),
        "alpha_ready": alpha_report.get("gpu_tpu_cpu_heterogeneous_stage_alpha_ready") is True,
        "gpu_backend_evidence_ready": alpha_report.get("gpu_backend_evidence_ready") is True,
        "tpu_backend_evidence_ready": alpha_report.get("tpu_backend_evidence_ready") is True,
        "cpu_backend_evidence_ready": alpha_report.get("cpu_backend_evidence_ready") is True,
        "local_three_stage_real_model_e2e_ready": alpha_report.get("local_three_stage_real_model_e2e_ready") is True,
        "alpha_32b_feasibility_ready": alpha_report.get("gpu_tpu_cpu_32b_feasibility_report_ready") is True,
        "alpha_same_request_live_verified": alpha_report.get("same_request_live_heterogeneous_verified") is True,
        "alpha_live_tpu_stage_miner_integrated": alpha_report.get("live_tpu_stage_miner_integrated") is True,
        "alpha_verdict": str(feasibility.get("verdict") or ""),
        "target_model_id": args.target_32b_model_id,
        "public_artifact_safe": True,
    }


def build_tpu_stage_adapter_plan_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    mapping = _dict(report.get("mapping"))
    shape_protocol = _dict(report.get("shape_protocol"))
    tpu_stage = _dict(report.get("tpu_stage"))
    stage_local_kv_cache = _dict(shape_protocol.get("stage_local_kv_cache"))
    activation_metadata = _dict(shape_protocol.get("activation_metadata"))
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    return {
        "schema": TPU_STAGE_ADAPTER_PLAN_SUMMARY_SCHEMA,
        "source": source_summary(path, report, kind="gpu_tpu_qwen_stage_adapter_plan"),
        "adapter_plan_present": bool(report),
        "checkpoint_bridge_plan_ready": report.get("checkpoint_bridge_plan_ready") is True,
        "stage_owned_tpu_loader_plan_ready": report.get("stage_owned_tpu_loader_plan_ready") is True,
        "qwen_llama_like_stage_runtime_planned": report.get("qwen_llama_like_stage_runtime_planned") is True,
        "tpu_32b_runtime_adapter_ready": report.get("tpu_32b_runtime_adapter_ready") is True,
        "jax_tpu_runtime_execution_ready": report.get("jax_tpu_runtime_execution_ready") is True,
        "same_request_live_heterogeneous_verified": report.get("same_request_live_heterogeneous_verified") is True,
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "decoder_layer_count": _int(report.get("decoder_layer_count")),
        "hidden_size": _int(report.get("hidden_size")),
        "tpu_stage": {
            "stage_id": _int(tpu_stage.get("stage_id")),
            "backend": str(tpu_stage.get("backend") or ""),
            "layer_range": list(tpu_stage.get("layer_range") or []),
            "layer_count": _int(tpu_stage.get("layer_count")),
            "stage_owned_middle_layers": tpu_stage.get("stage_owned_middle_layers") is True,
        },
        "assigned_key_count": _int(mapping.get("assigned_key_count")),
        "assigned_file_count": _int(mapping.get("assigned_file_count")),
        "mapped_key_count": _int(mapping.get("mapped_key_count")),
        "unsupported_key_count": _int(mapping.get("unsupported_key_count")),
        "all_assigned_keys_mapped": mapping.get("all_assigned_keys_mapped") is True,
        "activation_metadata": {
            "shape": list(activation_metadata.get("shape") or []),
            "dtype": str(activation_metadata.get("dtype") or ""),
            "layout": str(activation_metadata.get("layout") or ""),
            "bytes_per_token": _int(activation_metadata.get("bytes_per_token")),
            "transport_requires_hash": activation_metadata.get("transport_requires_hash") is True,
        },
        "stage_local_kv_cache_metadata": {
            "planned": bool(stage_local_kv_cache),
            "layer_count": _int(stage_local_kv_cache.get("layer_count")),
            "estimated_kv_bytes_per_token": _int(stage_local_kv_cache.get("estimated_kv_bytes_per_token")),
            "kv_payload_public": stage_local_kv_cache.get("kv_payload_public") is True,
        },
        "blockers": blockers,
        "public_artifact_safe": report.get("public_artifact_safe") is True if report else True,
    }


def build_tpu_stage_runtime_probe_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    stage = _dict(report.get("jax_tpu_stage"))
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    blocked_reason = str(report.get("blocked_reason") or "")
    if blocked_reason and blocked_reason not in blockers:
        blockers.insert(0, blocked_reason)
    return {
        "schema": TPU_STAGE_RUNTIME_PROBE_SUMMARY_SCHEMA,
        "source": source_summary(path, report, kind="kaggle_tpu_qwen_stage_runtime_probe"),
        "runtime_probe_present": bool(report),
        "selected_accelerator": str(report.get("selected_accelerator") or ""),
        "stage_profile": str(report.get("stage_profile") or ""),
        "tpu_runtime_ready": report.get("tpu_runtime_ready") is True,
        "qwen_like_stage_runtime_ready": report.get("qwen_like_stage_runtime_ready") is True,
        "qwen32b_single_layer_runtime_ready": report.get("qwen32b_single_layer_runtime_ready") is True,
        "tpu_32b_runtime_adapter_ready": report.get("tpu_32b_runtime_adapter_ready") is True,
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True,
        "shape_metadata": _dict(report.get("shape_metadata") or stage.get("shape_metadata")),
        "stage_input_hash": str(report.get("stage_input_hash") or stage.get("stage_input_hash") or ""),
        "stage_output_hash": str(report.get("stage_output_hash") or stage.get("stage_output_hash") or ""),
        "blockers": blockers,
        "bounded_probe_blocked": bool(report) and report.get("qwen_like_stage_runtime_ready") is not True,
        "kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "public_artifact_safe": _dict(report.get("safety")).get("public_artifact_safe") is True if report else True,
    }


def build_tpu_stage_loader_probe_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    blocked_reason = str(report.get("blocked_reason") or "")
    if blocked_reason and blocked_reason not in blockers:
        blockers.insert(0, blocked_reason)
    return {
        "schema": TPU_STAGE_32B_LOADER_PROBE_SUMMARY_SCHEMA,
        "source": source_summary(path, report, kind="kaggle_tpu_32b_stage_owned_loader_probe"),
        "loader_probe_present": bool(report),
        "model_repo": str(report.get("model_repo") or ""),
        "stage_layer_range": list(report.get("stage_layer_range") or []),
        "stage_owned_header_verified": report.get("stage_owned_header_verified") is True,
        "partial_tensor_to_tpu_verified": report.get("partial_tensor_to_tpu_verified") is True,
        "full_stage_owned_tpu_loader_ready": report.get("full_stage_owned_tpu_loader_ready") is True,
        "tpu_32b_runtime_adapter_ready": report.get("tpu_32b_runtime_adapter_ready") is True,
        "assigned_weight_key_count": _int(report.get("assigned_weight_key_count")),
        "assigned_weight_file_count": _int(report.get("assigned_weight_file_count")),
        "present_stage_key_count": _int(report.get("present_stage_key_count")),
        "missing_stage_key_count": _int(report.get("missing_stage_key_count")),
        "selected_tensor_key_hash": str(report.get("selected_tensor_key_hash") or ""),
        "selected_tensor_value_hash": str(report.get("selected_tensor_value_hash") or ""),
        "selected_tensor_tpu_summary_hash": str(report.get("selected_tensor_tpu_summary_hash") or ""),
        "selected_tensor_shape": list(report.get("selected_tensor_shape") or []),
        "selected_tensor_dtype": str(report.get("selected_tensor_dtype") or ""),
        "selected_tensor_bytes": _int(report.get("selected_tensor_bytes")),
        "tpu_device_count": _int(report.get("tpu_device_count")),
        "blockers": blockers,
        "bounded_probe_blocked": bool(report) and report.get("full_stage_owned_tpu_loader_ready") is not True,
        "kernels_deleted": lifecycle.get("kernels_deleted") is True if report else True,
        "private_packages_removed": lifecycle.get("private_packages_removed") is True if report else True,
        "public_artifact_safe": _dict(report.get("safety")).get("public_artifact_safe") is True if report else True,
    }


def build_stage_runtime_matrix(
    alpha_report: dict[str, Any],
    tpu_stage_adapter_plan: dict[str, Any],
    tpu_stage_runtime_probe: dict[str, Any],
    tpu_stage_loader_probe: dict[str, Any],
) -> dict[str, Any]:
    gpu = _dict(alpha_report.get("gpu_backend"))
    tpu = _dict(alpha_report.get("tpu_backend"))
    cpu = _dict(alpha_report.get("cpu_backend"))
    adapter_plan_ready = tpu_stage_adapter_plan.get("checkpoint_bridge_plan_ready") is True
    loader_plan_ready = tpu_stage_adapter_plan.get("stage_owned_tpu_loader_plan_ready") is True
    runtime_probe_ready = tpu_stage_runtime_probe.get("qwen_like_stage_runtime_ready") is True
    qwen32b_single_layer_ready = tpu_stage_runtime_probe.get("qwen32b_single_layer_runtime_ready") is True
    runtime_ready = (
        tpu_stage_adapter_plan.get("tpu_32b_runtime_adapter_ready") is True
        or tpu_stage_runtime_probe.get("tpu_32b_runtime_adapter_ready") is True
        or tpu_stage_loader_probe.get("tpu_32b_runtime_adapter_ready") is True
    )
    kv_metadata = _dict(tpu_stage_adapter_plan.get("stage_local_kv_cache_metadata"))
    missing_items: list[str] = []
    if not runtime_probe_ready and not runtime_ready:
        missing_items.append("jax_tpu_llama_like_stage_runtime")
    if runtime_probe_ready and not runtime_ready:
        missing_items.append("full_32b_tpu_stage_owned_runtime_not_verified")
    if not adapter_plan_ready:
        missing_items.append("safetensors_or_maxtext_checkpoint_bridge")
    if not runtime_ready and not tpu_stage_adapter_plan.get("jax_tpu_runtime_execution_ready"):
        missing_items.append("jax_tpu_runtime_execution_not_performed")
    if not (runtime_ready or tpu_stage_runtime_probe.get("stage_local_kv_cache_verified") is True):
        missing_items.append("tpu_stage_local_kv_cache_format")
    return {
        "schema": "gpu_tpu_cpu_32b_stage_runtime_matrix_v1",
        "cuda_gpu_stage": {
            "backend": "cuda",
            "stage_runtime_family": "hf_pytorch_stage_owned_safetensors",
            "qwen_llama_like_stage_loading_ready": gpu.get("full_precision_32b_gpu_cpu_ready") is True,
            "stage_local_kv_cache_ready": gpu.get("full_precision_stage_local_kv_cache_ready") is True,
            "activation_handoff_evidence_ready": gpu.get("quantized_32b_gpu_upper_bound_ready") is True,
            "source_model_id": str(gpu.get("full_precision_model_id") or ""),
        },
        "jax_tpu_stage": {
            "backend": "jax_tpu",
            "stage_runtime_family": "jax_tpu_stage_owned_decoder",
            "real_model_tpu_runtime_ready": tpu.get("real_model_tpu_inference_ready") is True,
            "retained_tpu_model_id": str(tpu.get("model_id") or ""),
            "retained_tpu_parameter_count": _int(tpu.get("parameter_count")),
            "qwen_llama_like_stage_loading_ready": loader_plan_ready,
            "checkpoint_bridge_plan_ready": adapter_plan_ready,
            "stage_owned_tpu_loader_plan_ready": loader_plan_ready,
            "adapter_model_repo": str(tpu_stage_adapter_plan.get("model_repo") or ""),
            "adapter_layer_range": list(_dict(tpu_stage_adapter_plan.get("tpu_stage")).get("layer_range") or []),
            "adapter_assigned_key_count": _int(tpu_stage_adapter_plan.get("assigned_key_count")),
            "adapter_assigned_file_count": _int(tpu_stage_adapter_plan.get("assigned_file_count")),
            "adapter_unsupported_key_count": _int(tpu_stage_adapter_plan.get("unsupported_key_count")),
            "adapter_all_assigned_keys_mapped": tpu_stage_adapter_plan.get("all_assigned_keys_mapped") is True,
            "qwen_like_stage_runtime_probe_ready": runtime_probe_ready,
            "qwen32b_single_layer_runtime_probe_ready": qwen32b_single_layer_ready,
            "stage_runtime_probe_profile": str(tpu_stage_runtime_probe.get("stage_profile") or ""),
            "stage_runtime_probe_shape_metadata": _dict(tpu_stage_runtime_probe.get("shape_metadata")),
            "tpu_32b_runtime_adapter_ready": runtime_ready,
            "stage_owned_32b_header_verified": tpu_stage_loader_probe.get("stage_owned_header_verified") is True,
            "stage_owned_32b_partial_tensor_to_tpu_verified": tpu_stage_loader_probe.get("partial_tensor_to_tpu_verified") is True,
            "stage_owned_32b_full_loader_ready": tpu_stage_loader_probe.get("full_stage_owned_tpu_loader_ready") is True,
            "stage_owned_32b_loader_selected_tensor_hash": str(tpu_stage_loader_probe.get("selected_tensor_tpu_summary_hash") or ""),
            "jax_tpu_runtime_execution_ready": tpu_stage_adapter_plan.get("jax_tpu_runtime_execution_ready") is True,
            "stage_local_kv_cache_plan_ready": kv_metadata.get("planned") is True and kv_metadata.get("kv_payload_public") is False,
            "stage_local_kv_cache_probe_ready": tpu_stage_runtime_probe.get("stage_local_kv_cache_verified") is True,
            "stage_local_kv_cache_ready": runtime_ready,
            "missing_items": missing_items,
        },
        "cpu_tail_or_verifier_stage": {
            "backend": "cpu",
            "stage_runtime_family": "hf_pytorch_cpu_tail_or_verifier",
            "cpu_stage_ready": cpu.get("cpu_backend_evidence_ready") is True,
            "tail_or_verifier_ready": cpu.get("local_cpu_real_llm_sharded_ready") is True or cpu.get("retained_32b_cpu_stage_ready") is True,
            "retained_32b_cpu_stage_ready": cpu.get("retained_32b_cpu_stage_ready") is True,
        },
        "public_artifact_safe": True,
    }


def build_activation_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": ACTIVATION_PROTOCOL_SCHEMA,
        "protocol_ready": True,
        "same_request_live_handoffs_verified": False,
        "target_model_id": args.target_32b_model_id,
        "context_length": args.context_length,
        "hops": [
            {
                "from_backend": "cuda",
                "to_backend": "jax_tpu",
                "shape_metadata_required": True,
                "dtype_metadata_required": True,
                "layout_metadata_required": True,
                "activation_hash_required": True,
                "activation_payload_public": False,
                "allowed_dtypes": ["float16", "bfloat16", "float32"],
                "canonical_layout": "batch_seq_hidden",
            },
            {
                "from_backend": "jax_tpu",
                "to_backend": "cpu",
                "shape_metadata_required": True,
                "dtype_metadata_required": True,
                "layout_metadata_required": True,
                "activation_hash_required": True,
                "activation_payload_public": False,
                "allowed_dtypes": ["float16", "bfloat16", "float32"],
                "canonical_layout": "batch_seq_hidden",
            },
        ],
        "public_artifact_safe": True,
    }


def load_live_proof(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    path = Path(args.live_same_request_report)
    if args.live_proof_mode == "fixture-success":
        return fixture_live_success_report(args), path
    if args.live_proof_mode == "fixture-fallback":
        return fixture_fallback_report(args), path
    if args.live_proof_mode == "external":
        return load_optional_report(path), path
    return {}, path


def build_tpu_allocation_attempt_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    blocked_reason = str(report.get("blocked_reason") or "")
    if blocked_reason and blocked_reason not in blockers:
        blockers.insert(0, blocked_reason)
    tpu_runtime_ready = report.get("tpu_runtime_ready") is True
    allocation_blocked = bool(report) and not tpu_runtime_ready
    return {
        "schema": TPU_ALLOCATION_SUMMARY_SCHEMA,
        "source": source_summary(path, report, kind="kaggle_tpu_llm_probe"),
        "bounded_tpu_allocation_attempted": bool(report),
        "selected_accelerator": str(report.get("selected_accelerator") or ""),
        "tpu_runtime_ready": tpu_runtime_ready,
        "synthetic_llm_ready": report.get("synthetic_llm_ready") is True,
        "blocked_reason": blocked_reason,
        "blockers": blockers,
        "bounded_allocation_blocked": allocation_blocked,
        "kaggle_tpu_kernel_queued_timeout": "kaggle_tpu_kernel_queued_timeout" in blockers,
        "kaggle_tpu_accelerator_accepted": "kaggle_tpu_accelerator_accepted" in set(report.get("diagnosis_codes") or []),
        "kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "public_artifact_safe": True,
    }


def build_tpu_web_active_event_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    blocked_reason = str(report.get("blocked_reason") or "")
    if not blocked_reason and blockers:
        blocked_reason = str(blockers[0])
    if bool(report) and report.get("running") is not True and not blocked_reason:
        blocked_reason = "kaggle_web_tpu_runtime_not_allocated"
    if blocked_reason and blocked_reason not in blockers:
        blockers.insert(0, blocked_reason)
    running = report.get("running") is True or report.get("tpu_runtime_ready") is True
    queue_seen = report.get("queue_seen") is True
    attempted = bool(report)
    public_safe = report.get("public_artifact_safe") is True if report else True
    return {
        "schema": TPU_WEB_ACTIVE_EVENT_SUMMARY_SCHEMA,
        "source": source_summary(path, report, kind="kaggle_tpu_web_active_event"),
        "web_active_event_attempted": attempted,
        "notebook_url_public": str(report.get("notebook_url_public") or ""),
        "logged_in": report.get("logged_in") is True,
        "queue_seen": queue_seen,
        "queue_positions_public": [str(item) for item in _list(report.get("queue_positions_public"))],
        "tpu_runtime_ready": running,
        "running": running,
        "bounded_wait_seconds": _int(report.get("bounded_wait_seconds")),
        "poll_count": _int(report.get("poll_count")),
        "blocked_reason": "" if running else blocked_reason,
        "blockers": [] if running else blockers,
        "bounded_allocation_blocked": attempted and not running,
        "cleanup_not_required": True,
        "public_artifact_safe": public_safe,
    }


def build_runtime_bridge_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    blocked_reason = str(report.get("blocked_reason") or "")
    if blocked_reason and blocked_reason not in blockers:
        blockers.insert(0, blocked_reason)
    return {
        "schema": RUNTIME_BRIDGE_SUMMARY_SCHEMA,
        "source": source_summary(path, report, kind="gpu_tpu_cpu_same_request_runtime_bridge_probe"),
        "runtime_bridge_present": bool(report),
        "same_request_runtime_bridge_verified": report.get("same_request_runtime_bridge_verified") is True,
        "same_request_32b_model_verified": report.get("same_request_32b_model_verified") is True,
        "not_32b_weight_success": report.get("not_32b_weight_success") is True if report else True,
        "gpu_tpu_cpu_32b_same_request_verified": report.get("gpu_tpu_cpu_32b_same_request_verified") is True,
        "generated_token_count": _int(report.get("generated_token_count")),
        "accepted_stage_backends": [str(item) for item in _list(report.get("accepted_stage_backends"))],
        "activation_handoff_count": _int(report.get("activation_handoff_count")),
        "runtime_device_summary": _dict(report.get("runtime_device_summary")),
        "blocked_reason": blocked_reason,
        "blockers": blockers,
        "public_artifact_safe": _dict(report.get("safety")).get("public_artifact_safe") is True if report else True,
    }


def summarize_live_proof(args: argparse.Namespace, live_report: dict[str, Any], live_path: Path) -> dict[str, Any]:
    stage_counts = _dict(live_report.get("stage_task_counts"))
    accepted = [item for item in _list(live_report.get("accepted_stage_tasks")) if isinstance(item, dict) and item.get("accepted") is True]
    accepted_backends = {str(item.get("backend") or "") for item in accepted}
    handoffs = [item for item in _list(live_report.get("activation_handoffs")) if isinstance(item, dict)]
    model_parameter_count = _int(live_report.get("model_parameter_count"))
    model_id = str(live_report.get("model_id") or "")
    is_32b_class = model_parameter_count >= MIN_32B_PARAMETER_COUNT or "32b" in model_id.lower()
    generated = _int(live_report.get("generated_token_count"))
    cleanup = _dict(live_report.get("cleanup"))
    safety = _dict(live_report.get("safety"))
    public_safe = live_report.get("public_artifact_safe") is True and safety_flags_safe(safety)
    live_success = bool(
        live_report.get("schema") == LIVE_PROOF_SCHEMA
        and live_report.get("ok") is True
        and live_report.get("gpu_tpu_cpu_32b_same_request_verified") is True
        and live_report.get("live_tpu_stage_miner_integrated") is True
        and live_report.get("tpu_32b_runtime_adapter_ready") is True
        and live_report.get("fallback_model_used") is not True
        and is_32b_class
        and generated >= args.target_max_new_tokens
        and {"cuda", "jax_tpu", "cpu"}.issubset(accepted_backends)
        and _int(stage_counts.get("cuda")) >= 1
        and _int(stage_counts.get("jax_tpu")) >= 1
        and _int(stage_counts.get("cpu")) >= 1
        and len(handoffs) >= 2
        and all(str(item.get("activation_hash") or "").startswith("sha256:") for item in handoffs)
        and all(item.get("activation_payload_public") is False for item in handoffs)
        and public_safe
    )
    fallback_used = live_report.get("fallback_model_used") is True or (bool(live_report) and not is_32b_class)
    fallback_ready = bool(
        live_report.get("schema") == LIVE_PROOF_SCHEMA
        and live_report.get("ok") is True
        and fallback_used
        and live_report.get("live_tpu_stage_miner_integrated") is True
        and generated >= args.target_max_new_tokens
        and {"cuda", "jax_tpu", "cpu"}.issubset(accepted_backends)
        and public_safe
    )
    blockers: list[str] = []
    if not live_report:
        blockers.append("same_request_live_proof_missing")
    elif live_report.get("schema") != LIVE_PROOF_SCHEMA:
        blockers.append("live_proof_schema_mismatch")
    if bool(live_report) and not is_32b_class:
        blockers.append("live_proof_not_32b_class")
    if live_report and live_report.get("live_tpu_stage_miner_integrated") is not True:
        blockers.append("live_tpu_stage_miner_missing")
    if live_report and live_report.get("tpu_32b_runtime_adapter_ready") is not True:
        blockers.append("tpu_32b_runtime_adapter_missing")
    if live_report and not {"cuda", "jax_tpu", "cpu"}.issubset(accepted_backends):
        blockers.append("required_accelerator_stage_task_missing")
    if live_report and generated < args.target_max_new_tokens:
        blockers.append("generated_token_count_below_target")
    if live_report and not public_safe:
        blockers.append("public_artifact_safety_missing")
    if fallback_ready and "live_proof_not_32b_class" not in blockers:
        blockers.append("fallback_model_only_32b_not_verified")

    return {
        "schema": "gpu_tpu_cpu_32b_same_request_live_summary_v1",
        "source": source_summary(live_path, live_report, kind="gpu_tpu_cpu_32b_same_request_live_proof"),
        "live_proof_present": bool(live_report),
        "live_proof_mode": args.live_proof_mode,
        "same_request_verified": live_success,
        "fallback_live_proof_ready": fallback_ready,
        "fallback_model_used": fallback_used,
        "model_id": model_id,
        "model_parameter_count": model_parameter_count,
        "is_32b_class": is_32b_class,
        "generated_token_count": generated,
        "target_generated_token_count": args.target_max_new_tokens,
        "live_tpu_stage_miner_integrated": live_report.get("live_tpu_stage_miner_integrated") is True,
        "tpu_32b_runtime_adapter_ready": live_report.get("tpu_32b_runtime_adapter_ready") is True,
        "stage_local_kv_cache_verified": live_report.get("stage_local_kv_cache_verified") is True,
        "accepted_stage_backends": sorted(accepted_backends),
        "stage_task_counts": {
            "cuda": _int(stage_counts.get("cuda")),
            "jax_tpu": _int(stage_counts.get("jax_tpu")),
            "cpu": _int(stage_counts.get("cpu")),
        },
        "activation_handoff_count": len(handoffs),
        "activation_handoff_hashes": [
            {
                "from_backend": str(item.get("from_backend") or ""),
                "to_backend": str(item.get("to_backend") or ""),
                "activation_hash": str(item.get("activation_hash") or ""),
                "activation_payload_public": item.get("activation_payload_public") is True,
            }
            for item in handoffs
        ],
        "runtime_device_summary": _dict(live_report.get("runtime_device_summary")),
        "cleanup": {
            "private_runtime_artifacts_cleaned": cleanup.get("private_runtime_artifacts_cleaned") is True,
            "temporary_kaggle_kernels_deleted": cleanup.get("temporary_kaggle_kernels_deleted") is True,
            "token_rotation_required": cleanup.get("token_rotation_required") is True,
        },
        "blockers": blockers,
        "public_artifact_safe": public_safe if live_report else True,
    }


def safety_flags_safe(safety: dict[str, Any]) -> bool:
    if not safety:
        return False
    for key, expected in default_safety_flags().items():
        if safety.get(key) is not expected:
            return False
    return True


def build_blocker_report(
    args: argparse.Namespace,
    *,
    alpha_import: dict[str, Any],
    stage_runtime_matrix: dict[str, Any],
    live_summary: dict[str, Any],
    tpu_allocation_attempt: dict[str, Any],
    tpu_web_active_event: dict[str, Any],
    tpu_stage_adapter_plan: dict[str, Any],
    tpu_stage_runtime_probe: dict[str, Any],
    tpu_stage_loader_probe: dict[str, Any],
    runtime_bridge: dict[str, Any],
) -> dict[str, Any]:
    tpu_stage = _dict(stage_runtime_matrix.get("jax_tpu_stage"))
    resolved_runtime_blockers: set[str] = set()
    if (
        live_summary.get("same_request_verified") is True
        or live_summary.get("tpu_32b_runtime_adapter_ready") is True
        or tpu_stage.get("tpu_32b_runtime_adapter_ready") is True
        or tpu_stage_loader_probe.get("tpu_32b_runtime_adapter_ready") is True
    ):
        resolved_runtime_blockers.update(
            {
                "jax_tpu_llama_like_stage_runtime",
                "jax_tpu_runtime_execution_not_performed",
                "full_32b_tpu_stage_owned_runtime_not_verified",
                "full_stage_owned_tpu_loader_not_executed",
                "tpu_32b_runtime_adapter_missing",
            }
        )
    raw_blockers = (
        _list(live_summary.get("blockers"))
        + _list(tpu_stage.get("missing_items"))
        + _list(tpu_stage_adapter_plan.get("blockers"))
        + _list(tpu_stage_runtime_probe.get("blockers"))
        + _list(tpu_stage_loader_probe.get("blockers"))
    )
    blockers = list(
        dict.fromkeys(
            str(item)
            for item in raw_blockers
            if item and str(item) not in resolved_runtime_blockers
        )
    )
    if tpu_allocation_attempt.get("bounded_allocation_blocked") is True:
        allocation_reason = str(tpu_allocation_attempt.get("blocked_reason") or "tpu_runtime_allocation_blocked")
        blockers.insert(0, allocation_reason)
        for item in _list(tpu_allocation_attempt.get("blockers")):
            if item:
                blockers.append(str(item))
    if tpu_web_active_event.get("bounded_allocation_blocked") is True:
        web_reason = str(tpu_web_active_event.get("blocked_reason") or "kaggle_web_tpu_runtime_not_allocated")
        blockers.insert(0, web_reason)
        for item in _list(tpu_web_active_event.get("blockers")):
            if item:
                blockers.append(str(item))
    if live_summary.get("fallback_live_proof_ready") is True:
        blockers.append("fallback_model_only_32b_not_verified")
    if runtime_bridge.get("runtime_bridge_present") is True:
        for item in _list(runtime_bridge.get("blockers")):
            if item:
                blockers.append(str(item))
        if (
            runtime_bridge.get("same_request_runtime_bridge_verified") is True
            and runtime_bridge.get("gpu_tpu_cpu_32b_same_request_verified") is not True
        ):
            blockers.append("runtime_bridge_only_32b_weight_success_missing")
    if not blockers and live_summary.get("same_request_verified") is not True:
        blockers.append("same_request_live_gpu_tpu_cpu_32b_not_verified")
    return {
        "schema": "gpu_tpu_cpu_32b_rc_blocker_report_v1",
        "blocked": live_summary.get("same_request_verified") is not True,
        "blocked_reason": "" if live_summary.get("same_request_verified") is True else str(blockers[0]),
        "blockers": sorted(set(str(item) for item in blockers if item)),
        "blocker_classification": {
            "same_request_live_proof_missing": "live evidence" if "same_request_live_proof_missing" in blockers else "",
            "tpu_32b_runtime_adapter_missing": "TPU runtime adapter" if "tpu_32b_runtime_adapter_missing" in blockers or not tpu_stage.get("tpu_32b_runtime_adapter_ready") else "",
            "jax_tpu_llama_like_stage_runtime": "TPU model runtime" if "jax_tpu_llama_like_stage_runtime" in blockers else "",
            "jax_tpu_runtime_execution_not_performed": "TPU runtime execution" if "jax_tpu_runtime_execution_not_performed" in blockers else "",
            "safetensors_or_maxtext_checkpoint_bridge": "checkpoint conversion" if "safetensors_or_maxtext_checkpoint_bridge" in blockers else "",
            "fallback_model_only_32b_not_verified": "fallback boundary" if "fallback_model_only_32b_not_verified" in blockers else "",
            "tpu_runtime_allocation_blocked": "TPU allocation" if (
                tpu_allocation_attempt.get("bounded_allocation_blocked") is True
                or tpu_web_active_event.get("bounded_allocation_blocked") is True
            ) else "",
            "tpu_stage_adapter_plan": "metadata-only bridge plan" if tpu_stage_adapter_plan.get("checkpoint_bridge_plan_ready") is True else "",
            "qwen_like_stage_runtime_probe": "TPU stage runtime probe" if tpu_stage_runtime_probe.get("qwen_like_stage_runtime_ready") is True else "",
            "full_32b_tpu_stage_owned_runtime_not_verified": "32B TPU stage runtime scope" if "full_32b_tpu_stage_owned_runtime_not_verified" in blockers else "",
            "runtime_bridge_only_32b_weight_success_missing": "runtime bridge boundary" if "runtime_bridge_only_32b_weight_success_missing" in blockers else "",
            "stage_owned_32b_partial_loader": "partial 32B TPU loader evidence" if tpu_stage_loader_probe.get("partial_tensor_to_tpu_verified") is True else "",
            "full_stage_owned_tpu_loader_not_executed": "full 32B TPU loader boundary" if "full_stage_owned_tpu_loader_not_executed" in blockers else "",
        },
        "minimum_next_fix": [
            "Implement a JAX/TPU Qwen-or-Llama-like decoder stage runtime.",
            "Execute the stage-owned safetensors-to-JAX/MaxText checkpoint bridge inside an allocated TPU runtime.",
            "Run a bounded same-request Coordinator proof with accepted cuda, jax_tpu, and cpu stage tasks.",
            "Keep raw prompts, generated text, token ids, activations, logits, KV-cache, credentials, cookies, leases, and private runtime state out of public artifacts.",
        ],
        "alpha_ready": alpha_import.get("alpha_ready") is True,
        "public_artifact_safe": True,
    }


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "ok": report.get("ok") is True,
        "rc_ready": report.get("gpu_tpu_cpu_32b_heterogeneous_rc_ready") is True,
        "rc_success": report.get("gpu_tpu_cpu_32b_bounded_rc_success") is True,
        "same_request_verified": report.get("gpu_tpu_cpu_32b_same_request_verified") is True,
        "blocked_reason": report.get("blocked_reason") or "",
        "diagnosis_codes": list(report.get("diagnosis_codes") or []),
        "source_reports": report.get("source_reports") or {},
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    live_summary = _dict(report.get("live_same_request_summary"))
    blocker = _dict(report.get("blocker_report"))
    lines = [
        "# GPU+TPU+CPU 32B Heterogeneous RC",
        "",
        f"- report ready: `{report.get('gpu_tpu_cpu_32b_heterogeneous_rc_ready')}`",
        f"- bounded RC success: `{report.get('gpu_tpu_cpu_32b_bounded_rc_success')}`",
        f"- same-request 32B verified: `{report.get('gpu_tpu_cpu_32b_same_request_verified')}`",
        f"- live TPU stage Miner integrated: `{report.get('live_tpu_stage_miner_integrated')}`",
        f"- TPU 32B runtime adapter ready: `{report.get('tpu_32b_runtime_adapter_ready')}`",
        f"- TPU stage adapter plan ready: `{report.get('tpu_stage_adapter_plan_ready')}`",
        f"- TPU checkpoint bridge plan ready: `{report.get('tpu_checkpoint_bridge_plan_ready')}`",
        f"- TPU Qwen-like stage runtime probe ready: `{report.get('tpu_qwen_like_stage_runtime_probe_ready')}`",
        f"- fallback model used: `{report.get('fallback_model_used')}`",
        f"- stage-local KV cache verified: `{report.get('stage_local_kv_cache_verified')}`",
        f"- public artifact safe: `{report.get('public_artifact_safe')}`",
        "",
        "## Live Evidence",
        "",
        f"- live proof present: `{live_summary.get('live_proof_present')}`",
        f"- model: `{live_summary.get('model_id')}`",
        f"- generated tokens: `{live_summary.get('generated_token_count')}`",
        f"- accepted backends: `{', '.join(live_summary.get('accepted_stage_backends') or [])}`",
        "",
        "## Blockers",
        "",
        f"- blocked reason: `{blocker.get('blocked_reason') or report.get('blocked_reason')}`",
    ]
    for item in blocker.get("blockers") or []:
        lines.append(f"- `{item}`")
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- This is a bounded RC artifact. It must not be used to claim production serving.",
        "- Fixture, queue, fallback, or local smoke evidence cannot be represented as 32B same-request success.",
        "- Public artifacts redact raw prompts, generated text, token ids, activations, hidden states, logits, KV-cache, credentials, cookies, leases, idempotency material, and private runtime state.",
        "",
    ])
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = Path(args.alpha_report)
    if args.execution_mode == "fixture":
        alpha_report = fixture_alpha_report()
    else:
        alpha_report = load_optional_report(alpha_path)
    live_report, live_path = load_live_proof(args)
    tpu_attempt_path = Path(args.tpu_allocation_attempt_report)
    tpu_attempt_report = load_optional_report(tpu_attempt_path) if str(args.tpu_allocation_attempt_report or "").strip() else {}
    tpu_adapter_path = Path(args.tpu_stage_adapter_plan_report)
    tpu_adapter_report = load_optional_report(tpu_adapter_path) if str(args.tpu_stage_adapter_plan_report or "").strip() else {}
    tpu_runtime_path = Path(args.tpu_stage_runtime_probe_report)
    tpu_runtime_report = load_optional_report(tpu_runtime_path) if str(args.tpu_stage_runtime_probe_report or "").strip() else {}
    tpu_loader_path = Path(args.tpu_stage_loader_probe_report)
    tpu_loader_report = load_optional_report(tpu_loader_path) if str(args.tpu_stage_loader_probe_report or "").strip() else {}
    tpu_web_path = Path(args.tpu_web_active_event_report)
    tpu_web_report = load_optional_report(tpu_web_path) if str(args.tpu_web_active_event_report or "").strip() else {}
    runtime_bridge_path = Path(args.runtime_bridge_report)
    runtime_bridge_report = load_optional_report(runtime_bridge_path) if str(args.runtime_bridge_report or "").strip() else {}
    alpha_import = build_alpha_import(args, alpha_report, alpha_path)
    tpu_stage_adapter_plan = build_tpu_stage_adapter_plan_summary(tpu_adapter_path, tpu_adapter_report)
    tpu_stage_runtime_probe = build_tpu_stage_runtime_probe_summary(tpu_runtime_path, tpu_runtime_report)
    tpu_stage_loader_probe = build_tpu_stage_loader_probe_summary(tpu_loader_path, tpu_loader_report)
    stage_runtime_matrix = build_stage_runtime_matrix(alpha_report, tpu_stage_adapter_plan, tpu_stage_runtime_probe, tpu_stage_loader_probe)
    activation_protocol = build_activation_protocol(args)
    live_summary = summarize_live_proof(args, live_report, live_path)
    tpu_allocation_attempt = build_tpu_allocation_attempt_summary(tpu_attempt_path, tpu_attempt_report)
    tpu_web_active_event = build_tpu_web_active_event_summary(tpu_web_path, tpu_web_report)
    runtime_bridge = build_runtime_bridge_summary(runtime_bridge_path, runtime_bridge_report)
    activation_protocol["same_request_live_handoffs_verified"] = live_summary.get("same_request_verified") is True
    blocker_report = build_blocker_report(
        args,
        alpha_import=alpha_import,
        stage_runtime_matrix=stage_runtime_matrix,
        live_summary=live_summary,
        tpu_allocation_attempt=tpu_allocation_attempt,
        tpu_web_active_event=tpu_web_active_event,
        tpu_stage_adapter_plan=tpu_stage_adapter_plan,
        tpu_stage_runtime_probe=tpu_stage_runtime_probe,
        tpu_stage_loader_probe=tpu_stage_loader_probe,
        runtime_bridge=runtime_bridge,
    )
    rc_success = live_summary.get("same_request_verified") is True
    tpu_32b_runtime_adapter_ready = (
        live_summary.get("tpu_32b_runtime_adapter_ready") is True
        or tpu_stage_adapter_plan.get("tpu_32b_runtime_adapter_ready") is True
        or tpu_stage_runtime_probe.get("tpu_32b_runtime_adapter_ready") is True
        or tpu_stage_loader_probe.get("tpu_32b_runtime_adapter_ready") is True
    )
    fallback_used = live_summary.get("fallback_live_proof_ready") is True or live_summary.get("fallback_model_used") is True
    rc_ready = bool(
        alpha_import.get("alpha_ready")
        and alpha_import.get("gpu_backend_evidence_ready")
        and alpha_import.get("tpu_backend_evidence_ready")
        and alpha_import.get("cpu_backend_evidence_ready")
        and activation_protocol.get("protocol_ready")
        and blocker_report.get("public_artifact_safe")
    )
    blocked_reason = "" if rc_success else str(blocker_report.get("blocked_reason") or "gpu_tpu_cpu_32b_rc_not_verified")
    tpu_runtime_allocation_attempted = (
        tpu_allocation_attempt.get("bounded_tpu_allocation_attempted") is True
        or tpu_web_active_event.get("web_active_event_attempted") is True
    )
    tpu_runtime_allocation_ready = (
        tpu_allocation_attempt.get("tpu_runtime_ready") is True
        or tpu_web_active_event.get("tpu_runtime_ready") is True
    )
    tpu_runtime_allocation_blocked = bool(
        tpu_runtime_allocation_attempted
        and not tpu_runtime_allocation_ready
        and (
            tpu_allocation_attempt.get("bounded_allocation_blocked") is True
            or tpu_web_active_event.get("bounded_allocation_blocked") is True
        )
    )
    diagnosis_codes = {
        "gpu_tpu_cpu_32b_heterogeneous_rc_report_ready" if rc_ready else "gpu_tpu_cpu_32b_heterogeneous_rc_report_blocked",
        "gpu_tpu_cpu_32b_bounded_rc_success" if rc_success else "gpu_tpu_cpu_32b_bounded_rc_not_success",
        "gpu_tpu_cpu_32b_same_request_verified" if rc_success else "gpu_tpu_cpu_32b_same_request_not_verified",
        "live_tpu_stage_miner_integrated" if live_summary.get("live_tpu_stage_miner_integrated") else "live_tpu_stage_miner_not_integrated",
        "tpu_32b_runtime_adapter_ready" if tpu_32b_runtime_adapter_ready else "tpu_32b_runtime_adapter_not_ready",
        "fallback_model_used" if fallback_used else "fallback_model_not_used",
        "stage_local_kv_cache_verified" if live_summary.get("stage_local_kv_cache_verified") else "stage_local_kv_cache_not_verified_for_32b",
        "tpu_stage_adapter_plan_ready" if tpu_stage_adapter_plan.get("checkpoint_bridge_plan_ready") else "tpu_stage_adapter_plan_missing",
        "tpu_qwen_like_stage_runtime_probe_ready" if tpu_stage_runtime_probe.get("qwen_like_stage_runtime_ready") else "tpu_qwen_like_stage_runtime_probe_missing",
        "tpu_web_active_event_running" if tpu_web_active_event.get("tpu_runtime_ready") else (
            "tpu_web_active_event_blocked" if tpu_web_active_event.get("bounded_allocation_blocked") else "tpu_web_active_event_not_imported"
        ),
        "same_request_runtime_bridge_ready" if runtime_bridge.get("same_request_runtime_bridge_verified") else (
            "same_request_runtime_bridge_blocked" if runtime_bridge.get("runtime_bridge_present") else "same_request_runtime_bridge_not_imported"
        ),
        "gpu_tpu_cpu_32b_blocker_report_ready" if not rc_success else "gpu_tpu_cpu_32b_blocker_report_empty_success",
        "gpu_tpu_cpu_32b_public_artifact_redaction_ready",
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": rc_ready,
        "output_dir": str(output_dir),
        "execution_mode": args.execution_mode,
        "live_proof_mode": args.live_proof_mode,
        "target_model_id": args.target_32b_model_id,
        "target_generated_token_count": args.target_max_new_tokens,
        "context_length": args.context_length,
        "gpu_tpu_cpu_32b_heterogeneous_rc_ready": rc_ready,
        "gpu_tpu_cpu_32b_bounded_rc_success": rc_success,
        "gpu_tpu_cpu_32b_same_request_verified": rc_success,
        "live_tpu_stage_miner_integrated": live_summary.get("live_tpu_stage_miner_integrated") is True,
        "fallback_model_used": fallback_used,
        "tpu_32b_runtime_adapter_ready": tpu_32b_runtime_adapter_ready,
        "stage_local_kv_cache_verified": live_summary.get("stage_local_kv_cache_verified") is True,
        "tpu_runtime_allocation_attempted": tpu_runtime_allocation_attempted,
        "tpu_runtime_allocation_ready": tpu_runtime_allocation_ready,
        "tpu_runtime_allocation_blocked": tpu_runtime_allocation_blocked,
        "tpu_stage_adapter_plan_ready": tpu_stage_adapter_plan.get("checkpoint_bridge_plan_ready") is True,
        "tpu_checkpoint_bridge_plan_ready": tpu_stage_adapter_plan.get("checkpoint_bridge_plan_ready") is True,
        "tpu_stage_owned_loader_plan_ready": tpu_stage_adapter_plan.get("stage_owned_tpu_loader_plan_ready") is True,
        "tpu_qwen_like_stage_runtime_probe_ready": tpu_stage_runtime_probe.get("qwen_like_stage_runtime_ready") is True,
        "tpu_qwen32b_single_layer_runtime_probe_ready": tpu_stage_runtime_probe.get("qwen32b_single_layer_runtime_ready") is True,
        "external_runtime_verified": bool(rc_success and args.execution_mode == "external-existing" and args.live_proof_mode == "external"),
        "blocked_reason": blocked_reason,
        "public_artifact_safe": True,
        "source_reports": {
            "alpha": alpha_import.get("source"),
            "live_same_request": live_summary.get("source"),
            "tpu_allocation_attempt": tpu_allocation_attempt.get("source"),
            "tpu_web_active_event": tpu_web_active_event.get("source"),
            "runtime_bridge": runtime_bridge.get("source"),
            "tpu_stage_adapter_plan": tpu_stage_adapter_plan.get("source"),
            "tpu_stage_runtime_probe": tpu_stage_runtime_probe.get("source"),
            "tpu_stage_loader_probe": tpu_stage_loader_probe.get("source"),
        },
        "alpha_import": alpha_import,
        "stage_runtime_matrix": stage_runtime_matrix,
        "activation_protocol": activation_protocol,
        "live_same_request_summary": live_summary,
        "tpu_allocation_attempt_summary": tpu_allocation_attempt,
        "tpu_web_active_event_summary": tpu_web_active_event,
        "runtime_bridge_summary": runtime_bridge,
        "tpu_stage_adapter_plan_summary": tpu_stage_adapter_plan,
        "tpu_stage_runtime_probe_summary": tpu_stage_runtime_probe,
        "tpu_stage_loader_probe_summary": tpu_stage_loader_probe,
        "blocker_report": blocker_report,
        "diagnosis_codes": sorted(diagnosis_codes),
        "safety": default_safety_flags(),
        "boundaries": {
            "not_production_serving": True,
            "not_p2p_nat_traversal": True,
            "not_billing_or_settlement": True,
            "not_multi_account_limit_bypass": True,
            "fallback_is_not_32b_success": True,
            "queue_evidence_is_not_runtime_success": True,
            "fixture_is_not_live_success": True,
        },
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["gpu_tpu_cpu_32b_heterogeneous_rc_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["safety"]["report_public_leak_paths"] = leaks
        report["diagnosis_codes"].append("gpu_tpu_cpu_32b_public_artifact_redaction_failed")

    summary_json = output_dir / "gpu_tpu_cpu_32b_heterogeneous_rc.json"
    summary_md = output_dir / "GPU_TPU_CPU_32B_HETEROGENEOUS_RC.md"
    support_path = output_dir / "support_bundle.json"
    runtime_path = output_dir / "stage_runtime_matrix.json"
    activation_path = output_dir / "activation_protocol.json"
    live_summary_path = output_dir / "live_same_request_summary.json"
    tpu_attempt_summary_path = output_dir / "tpu_allocation_attempt_summary.json"
    tpu_web_summary_path = output_dir / "tpu_web_active_event_summary.json"
    runtime_bridge_summary_path = output_dir / "runtime_bridge_summary.json"
    tpu_adapter_summary_path = output_dir / "tpu_stage_adapter_plan_summary.json"
    tpu_runtime_summary_path = output_dir / "tpu_stage_runtime_probe_summary.json"
    tpu_loader_summary_path = output_dir / "tpu_stage_loader_probe_summary.json"
    blocker_path = output_dir / "blocker_report.json"
    write_json(runtime_path, stage_runtime_matrix)
    write_json(activation_path, activation_protocol)
    write_json(live_summary_path, live_summary)
    write_json(tpu_attempt_summary_path, tpu_allocation_attempt)
    write_json(tpu_web_summary_path, tpu_web_active_event)
    write_json(runtime_bridge_summary_path, runtime_bridge)
    write_json(tpu_adapter_summary_path, tpu_stage_adapter_plan)
    write_json(tpu_runtime_summary_path, tpu_stage_runtime_probe)
    write_json(tpu_loader_summary_path, tpu_stage_loader_probe)
    write_json(blocker_path, blocker_report)
    summary_md.write_text(render_markdown(report), encoding="utf-8")
    support_bundle = build_support_bundle(report)
    write_json(support_path, support_bundle)
    artifacts = {
        "summary_json": artifact_entry(summary_json, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
        "summary_markdown": artifact_entry(summary_md, output_dir, kind="summary_markdown", ok=bool(report.get("ok"))),
        "support_bundle_json": artifact_entry(support_path, output_dir, kind="support_bundle_json", schema=SUPPORT_BUNDLE_SCHEMA, ok=bool(report.get("ok"))),
        "stage_runtime_matrix_json": artifact_entry(runtime_path, output_dir, kind="stage_runtime_matrix_json", schema="gpu_tpu_cpu_32b_stage_runtime_matrix_v1", ok=True),
        "activation_protocol_json": artifact_entry(activation_path, output_dir, kind="activation_protocol_json", schema=ACTIVATION_PROTOCOL_SCHEMA, ok=True),
        "live_same_request_summary_json": artifact_entry(live_summary_path, output_dir, kind="live_same_request_summary_json", schema="gpu_tpu_cpu_32b_same_request_live_summary_v1", ok=bool(live_summary.get("same_request_verified"))),
        "tpu_allocation_attempt_summary_json": artifact_entry(tpu_attempt_summary_path, output_dir, kind="tpu_allocation_attempt_summary_json", schema=TPU_ALLOCATION_SUMMARY_SCHEMA, ok=bool(tpu_allocation_attempt.get("tpu_runtime_ready"))),
        "tpu_web_active_event_summary_json": artifact_entry(tpu_web_summary_path, output_dir, kind="tpu_web_active_event_summary_json", schema=TPU_WEB_ACTIVE_EVENT_SUMMARY_SCHEMA, ok=bool(tpu_web_active_event.get("tpu_runtime_ready"))),
        "runtime_bridge_summary_json": artifact_entry(runtime_bridge_summary_path, output_dir, kind="runtime_bridge_summary_json", schema=RUNTIME_BRIDGE_SUMMARY_SCHEMA, ok=bool(runtime_bridge.get("same_request_runtime_bridge_verified"))),
        "tpu_stage_adapter_plan_summary_json": artifact_entry(tpu_adapter_summary_path, output_dir, kind="tpu_stage_adapter_plan_summary_json", schema=TPU_STAGE_ADAPTER_PLAN_SUMMARY_SCHEMA, ok=bool(tpu_stage_adapter_plan.get("checkpoint_bridge_plan_ready"))),
        "tpu_stage_runtime_probe_summary_json": artifact_entry(tpu_runtime_summary_path, output_dir, kind="tpu_stage_runtime_probe_summary_json", schema=TPU_STAGE_RUNTIME_PROBE_SUMMARY_SCHEMA, ok=bool(tpu_stage_runtime_probe.get("qwen_like_stage_runtime_ready"))),
        "tpu_stage_loader_probe_summary_json": artifact_entry(tpu_loader_summary_path, output_dir, kind="tpu_stage_loader_probe_summary_json", schema=TPU_STAGE_32B_LOADER_PROBE_SUMMARY_SCHEMA, ok=bool(tpu_stage_loader_probe.get("full_stage_owned_tpu_loader_ready"))),
        "blocker_report_json": artifact_entry(blocker_path, output_dir, kind="blocker_report_json", schema="gpu_tpu_cpu_32b_rc_blocker_report_v1", ok=True),
    }
    report["artifacts"] = artifacts
    report["artifact_summary"] = {
        "schema": "gpu_tpu_cpu_32b_heterogeneous_rc_artifact_summary_v1",
        "artifact_count": len(artifacts),
        "present_artifact_count": sum(1 for item in artifacts.values() if item.get("present")),
        "inspect_first": str(summary_md),
        "support_bundle": str(support_path),
        "blocker_report": str(blocker_path),
        "public_artifact_safe": bool(report.get("public_artifact_safe")),
    }
    write_json(summary_json, report)
    report["artifacts"]["summary_json"]["present"] = True
    report["artifacts"]["summary_json"]["sha256"] = sha256_file(summary_json)
    report["artifact_summary"]["present_artifact_count"] = sum(1 for item in report["artifacts"].values() if item.get("present"))
    write_json(summary_json, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPU+TPU+CPU 32B heterogeneous stage inference RC evidence.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="evidence-import")
    parser.add_argument("--alpha-report", default=DEFAULT_ALPHA_REPORT)
    parser.add_argument("--live-proof-mode", choices=LIVE_PROOF_MODES, default="none")
    parser.add_argument("--live-same-request-report", default="")
    parser.add_argument("--tpu-allocation-attempt-report", default="")
    parser.add_argument("--tpu-web-active-event-report", default="")
    parser.add_argument("--runtime-bridge-report", default="")
    parser.add_argument("--tpu-stage-adapter-plan-report", default="")
    parser.add_argument("--tpu-stage-runtime-probe-report", default="")
    parser.add_argument("--tpu-stage-loader-probe-report", default="")
    parser.add_argument("--target-32b-model-id", default=TARGET_32B_MODEL_ID)
    parser.add_argument("--target-max-new-tokens", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.execution_mode in {"evidence-import", "external-existing"} and args.alpha_report and not Path(args.alpha_report).is_file():
        raise SystemExit("--alpha-report must point to an existing JSON file")
    if args.live_proof_mode == "external" and not str(args.live_same_request_report).strip():
        raise SystemExit("--live-same-request-report is required with --live-proof-mode external")
    if args.live_proof_mode == "external" and not Path(args.live_same_request_report).is_file():
        raise SystemExit("--live-same-request-report must point to an existing JSON file")
    if args.tpu_allocation_attempt_report and not Path(args.tpu_allocation_attempt_report).is_file():
        raise SystemExit("--tpu-allocation-attempt-report must point to an existing JSON file")
    if args.tpu_web_active_event_report and not Path(args.tpu_web_active_event_report).is_file():
        raise SystemExit("--tpu-web-active-event-report must point to an existing JSON file")
    if args.runtime_bridge_report and not Path(args.runtime_bridge_report).is_file():
        raise SystemExit("--runtime-bridge-report must point to an existing JSON file")
    if args.tpu_stage_adapter_plan_report and not Path(args.tpu_stage_adapter_plan_report).is_file():
        raise SystemExit("--tpu-stage-adapter-plan-report must point to an existing JSON file")
    if args.tpu_stage_runtime_probe_report and not Path(args.tpu_stage_runtime_probe_report).is_file():
        raise SystemExit("--tpu-stage-runtime-probe-report must point to an existing JSON file")
    if args.tpu_stage_loader_probe_report and not Path(args.tpu_stage_loader_probe_report).is_file():
        raise SystemExit("--tpu-stage-loader-probe-report must point to an existing JSON file")
    if args.target_max_new_tokens < 1 or args.target_max_new_tokens > 16:
        raise SystemExit("--target-max-new-tokens must be between 1 and 16")
    if args.context_length < 1 or args.context_length > 4096:
        raise SystemExit("--context-length must be between 1 and 4096")
    if not str(args.target_32b_model_id).strip():
        raise SystemExit("--target-32b-model-id must be non-empty")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"GPU+TPU+CPU 32B heterogeneous RC report ready: {report.get('ok')}")
        print(f"output: {report.get('output_dir')}")
        print(f"same-request 32B verified: {report.get('gpu_tpu_cpu_32b_same_request_verified')}")
        print(f"blocked reason: {report.get('blocked_reason')}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
