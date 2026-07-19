#!/usr/bin/env python3
"""Build the dense Qwen GPU+TPU+CPU frontier status artifact."""

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

from scripts import kaggle_dense_model_source_resolver as resolver  # noqa: E402
from scripts import qwen_dense_jax_tpu_stage_adapter_smoke as adapter_smoke  # noqa: E402


SCHEMA = "three_accelerator_dense_qwen_frontier_v1"
SUPPORT_BUNDLE_SCHEMA = "three_accelerator_dense_qwen_frontier_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/three-accelerator-dense-qwen-frontier"
DEFAULT_32B_BRIDGE_REPORT = (
    "dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r6-existing-session-4token/"
    "gpu_tpu_cpu_same_request_runtime_bridge_probe.json"
)
DEFAULT_GPU_CPU_DENSE_FALLBACK_REPORT = (
    "dist/kaggle-32b-full-heterogeneous-multitoken-kv-live-20260620-r1/"
    "kaggle_32b_full_heterogeneous_probe.json"
)
DEFAULT_TPU_DENSE_LOADER_REPORT = (
    "dist/kaggle-tpu-32b-stage-owned-loader-probe-web-live-20260623-r3-full-21-layer-real/"
    "kaggle_tpu_32b_stage_owned_loader_probe.json"
)
DEFAULT_KAGGLE_MODEL_ATTACH_PROBE_REPORT = ""
DEFAULT_DENSE_ADAPTER_REPORT = ""
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "jupyter-proxy",
    "operator.private.env",
    "miner.private.env",
    "kernel.py",
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"lease_token":',
    '"idempotency_key":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def parameter_value(value: str) -> float:
    return resolver.parameter_class_value(value)


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def source_summary(path: Path, report: dict[str, Any], *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "present": path.is_file(),
        "schema": str(report.get("schema") or ""),
        "ok": report.get("ok") is True,
        "sha256": sha256_file(path) if path.is_file() else "",
        "public_artifact_safe": bool(report.get("public_artifact_safe") is True or _dict(report.get("safety")).get("public_artifact_safe") is True),
    }


def summarize_32b_bridge(path: Path) -> dict[str, Any]:
    report = load_json(path)
    backends = set(str(item) for item in _list(report.get("accepted_stage_backends")))
    generated = _int(report.get("generated_token_count"))
    ready = bool(
        report.get("schema") == "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1"
        and report.get("ok") is True
        and report.get("same_request_runtime_bridge_verified") is True
        and report.get("gpu_tpu_cpu_32b_same_request_verified") is True
        and report.get("same_request_32b_model_verified") is True
        and {"cuda", "jax_tpu", "cpu"}.issubset(backends)
        and generated >= 1
    )
    return {
        "schema": "three_accelerator_dense_qwen_32b_bridge_import_v1",
        "source": source_summary(path, report, kind="gpu_tpu_cpu_32b_same_request_bridge"),
        "model_id": str(report.get("target_model_id") or "Qwen/Qwen2.5-32B-Instruct"),
        "parameter_class": "32b",
        "quantization": "none",
        "dense_full_precision": True,
        "generated_token_count": generated,
        "same_request_verified": ready,
        "all_three_accelerators_same_request_verified": ready,
        "accepted_stage_backends": sorted(backends),
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True,
        "public_artifact_safe": True,
    }


def summarize_gpu_cpu_dense_fallback(path: Path) -> dict[str, Any]:
    report = load_json(path)
    stage_summaries = [item for item in _list(report.get("stage_summaries")) if isinstance(item, dict)]
    resource_kinds = sorted({str(item.get("resource_kind") or "") for item in stage_summaries if item.get("resource_kind")})
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    safety = _dict(report.get("safety"))
    generated = _int(report.get("generated_token_count"))
    ready = bool(
        report.get("schema") == "kaggle_32b_full_heterogeneous_probe_v1"
        and report.get("ok") is True
        and report.get("quantization") == "none"
        and report.get("full_precision_32b") is True
        and report.get("four_t4_five_cpu_topology_verified") is True
        and report.get("stage_owned_full_precision_runtime_verified") is True
        and report.get("multi_token_generation_verified") is True
        and report.get("stage_local_kv_cache_verified") is True
        and generated >= 1
        and {"gpu", "cpu"}.issubset(set(resource_kinds))
        and lifecycle.get("kernels_deleted") is True
        and lifecycle.get("private_packages_removed") is True
        and safety.get("public_artifact_safe") is True
    )
    return {
        "schema": "three_accelerator_dense_qwen_gpu_cpu_fallback_import_v1",
        "source": source_summary(path, report, kind="gpu_cpu_32b_dense_fallback"),
        "model_id": str(_dict(report.get("model")).get("repo") or "Qwen/Qwen2.5-32B-Instruct"),
        "parameter_class": "32b",
        "quantization": "none",
        "dense_full_precision": report.get("full_precision_32b") is True,
        "generated_token_count": generated,
        "gpu_cpu_dense_fallback_verified": ready,
        "gpu_stage_runtime_ready": "gpu" in resource_kinds,
        "cpu_stage_runtime_ready": "cpu" in resource_kinds,
        "resource_kinds": resource_kinds,
        "stage_count": _int(_dict(report.get("model")).get("stage_count")),
        "stage_task_counts": report.get("stage_task_counts") or {},
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True,
        "multi_token_generation_verified": report.get("multi_token_generation_verified") is True,
        "temporary_kaggle_kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "temporary_private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "public_artifact_safe": True,
    }


def summarize_tpu_dense_loader(path: Path) -> dict[str, Any]:
    report = load_json(path)
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    safety = _dict(report.get("safety"))
    runtime = _dict(report.get("runtime_report"))
    model_repo = str(report.get("model_repo") or runtime.get("model_repo") or "Qwen/Qwen2.5-32B-Instruct")
    loaded_gb = report.get("loaded_execution_tensor_gb", runtime.get("loaded_execution_tensor_gb", 0))
    ready = bool(
        report.get("schema") == "kaggle_tpu_32b_stage_owned_loader_probe_v1"
        and report.get("ok") is True
        and report.get("full_stage_owned_tpu_loader_ready") is True
        and report.get("tpu_32b_runtime_adapter_ready") is True
        and _int(report.get("executed_layer_count")) >= 1
        and _int(report.get("loaded_execution_tensor_key_count")) >= 1
        and report.get("stage_local_kv_cache_verified") is True
        and _int(report.get("tpu_device_count")) >= 1
        and safety.get("public_artifact_safe") is True
    )
    return {
        "schema": "three_accelerator_dense_qwen_tpu_loader_import_v1",
        "source": source_summary(path, report, kind="tpu_dense_qwen_32b_stage_loader"),
        "model_id": model_repo,
        "parameter_class": "32b",
        "quantization": "none",
        "dense_full_precision": True,
        "tpu_runtime_ready": ready,
        "tpu_jax_qwen_stage_runtime_ready": ready,
        "full_stage_owned_tpu_loader_ready": report.get("full_stage_owned_tpu_loader_ready") is True,
        "tpu_32b_runtime_adapter_ready": report.get("tpu_32b_runtime_adapter_ready") is True,
        "stage_owned_header_verified": report.get("stage_owned_header_verified") is True,
        "partial_tensor_to_tpu_verified": report.get("partial_tensor_to_tpu_verified") is True,
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True,
        "stage_layer_range": report.get("stage_layer_range") or runtime.get("stage_layer_range") or [],
        "executed_layer_count": _int(report.get("executed_layer_count")),
        "full_stage_layer_count": _int(report.get("full_stage_layer_count")),
        "loaded_execution_tensor_key_count": _int(report.get("loaded_execution_tensor_key_count")),
        "loaded_execution_tensor_gb": loaded_gb,
        "tpu_device_count": _int(report.get("tpu_device_count")),
        "tpu_device_kind": str(report.get("tpu_device_kind") or ""),
        "temporary_kaggle_kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "temporary_private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "public_artifact_safe": True,
    }


def summarize_kaggle_model_attach_probe(path: Path) -> dict[str, Any]:
    report = load_json(path)
    runtime = _dict(report.get("runtime_report"))
    stage_plan = _dict(runtime.get("stage_plan"))
    cleanup = _dict(report.get("cleanup_status"))
    ready = bool(
        report.get("schema") == "kaggle_model_attach_probe_v1"
        and report.get("ok") is True
        and report.get("kaggle_model_attach_probe_ready") is True
        and report.get("kaggle_model_attach_used") is True
        and runtime.get("ok") is True
        and runtime.get("path_present") is True
        and runtime.get("config_json_present") is True
        and runtime.get("weight_index_present") is True
        and _int(runtime.get("safetensors_file_count")) >= 1
        and cleanup.get("temporary_kaggle_kernel_deleted") is True
        and cleanup.get("temporary_private_package_removed") is True
        and report.get("public_artifact_safe") is True
    )
    return {
        "schema": "three_accelerator_dense_qwen_kaggle_model_attach_import_v1",
        "source": source_summary(path, report, kind="kaggle_model_attach_probe"),
        "parameter_class": str(report.get("parameter_class") or ""),
        "hf_repo": str(report.get("hf_repo") or ""),
        "model_source": str(report.get("model_source") or ""),
        "expected_attached_path": str(report.get("expected_attached_path") or ""),
        "kaggle_model_attach_probe_ready": ready,
        "kaggle_model_attach_used": ready,
        "parameter_class_value_b": parameter_value(str(report.get("parameter_class") or "")),
        "path_present": runtime.get("path_present") is True,
        "config_json_present": runtime.get("config_json_present") is True,
        "weight_index_present": runtime.get("weight_index_present") is True,
        "tokenizer_json_present": runtime.get("tokenizer_json_present") is True,
        "safetensors_file_count": _int(runtime.get("safetensors_file_count")),
        "weight_index_key_count": _int(runtime.get("weight_index_key_count")),
        "weight_index_file_count": _int(runtime.get("weight_index_file_count")),
        "model_type": str(runtime.get("model_type") or ""),
        "torch_dtype": str(runtime.get("torch_dtype") or ""),
        "dense_full_precision": runtime.get("quantization_config_present") is not True,
        "quantization_config_present": runtime.get("quantization_config_present") is True,
        "stage_plan_requested": bool(report.get("stage_plan_requested") is True or runtime.get("stage_plan_enabled") is True),
        "stage_owned_preflight_verified": runtime.get("stage_owned_preflight_verified") is True,
        "stage_plan_schema": str(stage_plan.get("schema") or ""),
        "stage_plan_stage_count": _int(stage_plan.get("stage_count")),
        "stage_plan_backends": [str(item) for item in _list(stage_plan.get("stage_backends"))],
        "stage_plan_num_hidden_layers": _int(stage_plan.get("num_hidden_layers")),
        "stage_plan_hidden_size": _int(stage_plan.get("hidden_size")),
        "stage_plan_assigned_key_count_total": _int(stage_plan.get("assigned_key_count_total")),
        "stage_plan_present_key_count_total": _int(stage_plan.get("present_key_count_total")),
        "stage_plan_assigned_file_count_total": _int(stage_plan.get("assigned_file_count_total")),
        "stage_plan_total_logical_tensor_gb": stage_plan.get("total_planned_logical_tensor_gb", 0),
        "stage_plan_max_stage_logical_tensor_gb": stage_plan.get("max_stage_planned_logical_tensor_gb", 0),
        "stage_plan_stage_summaries": [
            {
                "stage_id": _int(item.get("stage_id")),
                "backend": str(item.get("backend") or ""),
                "layer_range": item.get("layer_range") or [],
                "assigned_key_count": _int(item.get("assigned_key_count")),
                "present_key_count": _int(item.get("present_key_count")),
                "missing_key_count": _int(item.get("missing_key_count")),
                "assigned_file_count": _int(item.get("assigned_file_count")),
                "logical_tensor_gb": item.get("logical_tensor_gb", 0),
                "stage_owned_header_verified": item.get("stage_owned_header_verified") is True,
            }
            for item in _list(stage_plan.get("stage_plans"))
            if isinstance(item, dict)
        ],
        "temporary_kaggle_kernel_deleted": cleanup.get("temporary_kaggle_kernel_deleted") is True,
        "temporary_private_package_removed": cleanup.get("temporary_private_package_removed") is True,
        "public_artifact_safe": True,
        "blockers": report.get("blocker_codes") or [],
    }


def summarize_model_sources(report: dict[str, Any]) -> dict[str, Any]:
    candidates = [item for item in _list(report.get("candidates")) if isinstance(item, dict)]
    dense = [item for item in candidates if item.get("full_precision_dense_candidate") is True]
    largest = max(dense, key=lambda item: parameter_value(str(item.get("parameter_class") or "")), default={})
    return {
        "schema": "three_accelerator_dense_qwen_model_sources_v1",
        "source_schema": report.get("schema"),
        "source_ok": report.get("ok") is True,
        "resolver_ready": report.get("kaggle_dense_model_source_resolver_ready") is True,
        "kaggle_model_attach_available": report.get("kaggle_model_attach_available") is True,
        "kaggle_model_attach_used": report.get("kaggle_model_attach_used") is True,
        "candidate_count": len(candidates),
        "dense_candidate_count": len(dense),
        "largest_dense_attach_candidate": {
            "parameter_class": largest.get("parameter_class", ""),
            "hf_repo": largest.get("hf_repo", ""),
            "kaggle_kernel_model_source": largest.get("kaggle_kernel_model_source", ""),
            "attached_runtime_path": largest.get("attached_runtime_path", ""),
            "resolved_attached_runtime_path": largest.get("resolved_attached_runtime_path", ""),
        },
        "candidates": [
            {
                "parameter_class": item.get("parameter_class"),
                "hf_repo": item.get("hf_repo"),
                "framework": item.get("framework"),
                "instance_slug": item.get("instance_slug"),
                "version_number": item.get("version_number"),
                "license_name": item.get("license_name"),
                "total_uncompressed_gb": item.get("total_uncompressed_gb"),
                "kaggle_kernel_model_source": item.get("kaggle_kernel_model_source"),
                "attached_runtime_path": item.get("attached_runtime_path"),
                "legacy_attached_runtime_path": item.get("legacy_attached_runtime_path"),
                "resolved_attached_runtime_path": item.get("resolved_attached_runtime_path"),
                "metadata_ready": item.get("metadata_ready") is True,
                "full_precision_dense_candidate": item.get("full_precision_dense_candidate") is True,
                "attach_can_avoid_runtime_download": item.get("attach_can_avoid_runtime_download") is True,
                "attach_path_present": item.get("attach_path_present") is True,
                "runtime_disk_download_required": item.get("runtime_disk_download_required"),
            }
            for item in candidates
        ],
        "blockers": report.get("blockers") or [],
        "public_artifact_safe": True,
    }


def summarize_adapter(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "three_accelerator_dense_qwen_tpu_adapter_v1",
        "source_schema": report.get("schema"),
        "source_ok": report.get("ok") is True,
        "torch_reference_forward_ready": report.get("torch_reference_forward_ready") is True,
        "jax_runtime_execution_ready": report.get("jax_runtime_execution_ready") is True,
        "tpu_runtime_ready": report.get("tpu_runtime_ready") is True,
        "tpu_jax_qwen_stage_runtime_ready": report.get("tpu_jax_qwen_stage_runtime_ready") is True,
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True,
        "dense_full_precision_only": report.get("dense_full_precision_only") is True,
        "quantized_weight_adapter_used": report.get("quantized_weight_adapter_used") is True,
        "shape_metadata": report.get("shape_metadata") or {},
        "qwen_components_exercised": report.get("qwen_components_exercised") or {},
        "blockers": report.get("blockers") or [],
        "public_artifact_safe": True,
    }


def combine_adapter_and_tpu_loader(adapter: dict[str, Any], tpu_loader: dict[str, Any]) -> dict[str, Any]:
    combined = dict(adapter)
    retained_ready = tpu_loader.get("tpu_jax_qwen_stage_runtime_ready") is True
    combined["retained_real_tpu_dense_loader_ready"] = retained_ready
    combined["retained_real_tpu_dense_loader_schema"] = tpu_loader.get("schema")
    combined["retained_real_tpu_dense_loader_model_id"] = tpu_loader.get("model_id")
    combined["retained_real_tpu_dense_loader_stage_layer_range"] = tpu_loader.get("stage_layer_range") or []
    combined["retained_real_tpu_dense_loader_executed_layer_count"] = _int(tpu_loader.get("executed_layer_count"))
    combined["retained_real_tpu_dense_loader_loaded_tensor_gb"] = tpu_loader.get("loaded_execution_tensor_gb", 0)
    combined["retained_real_tpu_dense_loader_tpu_device_count"] = _int(tpu_loader.get("tpu_device_count"))
    combined["tpu_runtime_ready"] = bool(adapter.get("tpu_runtime_ready") is True or retained_ready)
    combined["tpu_jax_qwen_stage_runtime_ready"] = bool(
        adapter.get("tpu_jax_qwen_stage_runtime_ready") is True or retained_ready
    )
    combined["stage_local_kv_cache_verified"] = bool(
        adapter.get("stage_local_kv_cache_verified") is True
        or tpu_loader.get("stage_local_kv_cache_verified") is True
    )
    blockers = {
        str(item)
        for item in _list(adapter.get("blockers"))
        if item
        and not (
            retained_ready
            and str(item)
            in {"jax_execution_not_requested", "tpu_runtime_not_requested", "tpu_runtime_not_available"}
        )
    }
    if not combined["tpu_jax_qwen_stage_runtime_ready"]:
        blockers.add("tpu_dense_qwen_jax_stage_runtime_not_verified")
    combined["blockers"] = sorted(blockers)
    return combined


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "ok": report.get("ok") is True,
        "generated_at": report.get("generated_at"),
        "report_schema": report.get("schema"),
        "canonical_summary": {
            "largest_dense_model_attempted": report.get("largest_dense_model_attempted"),
            "largest_dense_model_attach_candidate": report.get("largest_dense_model_attach_candidate"),
            "largest_dense_model_attached": report.get("largest_dense_model_attached"),
            "largest_dense_model_stage_preflighted": report.get("largest_dense_model_stage_preflighted"),
            "largest_dense_model_loaded": report.get("largest_dense_model_loaded"),
            "largest_dense_model_1token_decoded": report.get("largest_dense_model_1token_decoded"),
            "all_three_accelerators_same_request_verified": report.get("all_three_accelerators_same_request_verified"),
            "tpu_jax_qwen_stage_runtime_ready": report.get("tpu_jax_qwen_stage_runtime_ready"),
            "blocker_codes": report.get("blocker_codes") or [],
        },
        "public_artifact_safe": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolver_report = resolver.build_report(
        resolver.parse_args(
            [
                "--output-dir",
                str(output_dir / "model-source-resolver"),
                "--kaggle-input-root",
                args.kaggle_input_root,
                "--hf-timeout-seconds",
                str(args.hf_timeout_seconds),
                *([] if not args.fetch_hf_metadata else ["--fetch-hf-metadata"]),
            ]
        )
    )
    adapter_report_path = Path(args.dense_adapter_report) if str(args.dense_adapter_report or "").strip() else Path()
    if adapter_report_path.is_file():
        adapter_report = load_json(adapter_report_path)
        adapter_artifact_path = adapter_report_path
    else:
        adapter_report = adapter_smoke.build_report(
            adapter_smoke.parse_args(
                [
                    "--output-dir",
                    str(output_dir / "adapter-smoke"),
                    "--sequence-length",
                    str(args.adapter_sequence_length),
                    *([] if not args.run_jax_adapter else ["--run-jax"]),
                    *([] if not args.require_tpu_adapter else ["--require-tpu"]),
                ]
            )
        )
        adapter_artifact_path = output_dir / "adapter-smoke" / "qwen_dense_jax_tpu_stage_adapter_smoke.json"
    bridge = summarize_32b_bridge(Path(args.baseline_32b_bridge_report))
    gpu_cpu_fallback = summarize_gpu_cpu_dense_fallback(Path(args.gpu_cpu_dense_fallback_report))
    tpu_loader = summarize_tpu_dense_loader(Path(args.tpu_dense_loader_report))
    attach_probe = summarize_kaggle_model_attach_probe(Path(args.kaggle_model_attach_probe_report))
    sources = summarize_model_sources(resolver_report)
    if attach_probe.get("kaggle_model_attach_used") is True:
        sources["kaggle_model_attach_used"] = True
        sources["kaggle_model_attach_current_runtime_count"] = max(_int(sources.get("kaggle_model_attach_current_runtime_count")), 1)
        sources["largest_dense_live_attach"] = {
            "parameter_class": attach_probe.get("parameter_class", ""),
            "hf_repo": attach_probe.get("hf_repo", ""),
            "model_source": attach_probe.get("model_source", ""),
            "attached_runtime_path": attach_probe.get("expected_attached_path", ""),
        }
        for item in _list(sources.get("candidates")):
            if isinstance(item, dict) and item.get("parameter_class") == attach_probe.get("parameter_class"):
                item["live_attach_verified"] = True
                item["live_attached_runtime_path"] = attach_probe.get("expected_attached_path", "")
                if attach_probe.get("stage_owned_preflight_verified") is True:
                    item["live_stage_plan_verified"] = True
    adapter = combine_adapter_and_tpu_loader(summarize_adapter(adapter_report), tpu_loader)
    dense_candidates = [item for item in _list(sources.get("candidates")) if item.get("full_precision_dense_candidate") is True]
    largest_attempted = max(dense_candidates, key=lambda item: parameter_value(str(item.get("parameter_class") or "")), default={})
    decoded = bridge.get("parameter_class") if bridge.get("all_three_accelerators_same_request_verified") else ""
    loaded_classes = []
    if bridge.get("same_request_verified") is True:
        loaded_classes.append(str(bridge.get("parameter_class") or ""))
    if gpu_cpu_fallback.get("gpu_cpu_dense_fallback_verified") is True:
        loaded_classes.append(str(gpu_cpu_fallback.get("parameter_class") or ""))
    largest_loaded = max(loaded_classes, key=parameter_value, default="")
    largest_attached = str(attach_probe.get("parameter_class") or "") if attach_probe.get("kaggle_model_attach_used") is True else ""
    largest_stage_preflighted = (
        str(attach_probe.get("parameter_class") or "")
        if attach_probe.get("stage_owned_preflight_verified") is True
        else ""
    )
    same_request_32b_ready = bool(
        bridge.get("all_three_accelerators_same_request_verified") is True
        and adapter.get("tpu_jax_qwen_stage_runtime_ready") is True
    )
    same_request_ready = bool(
        same_request_32b_ready
        and sources.get("kaggle_model_attach_available") is True
        and sources.get("kaggle_model_attach_used") is True
        and decoded == str(largest_attempted.get("parameter_class") or "")
    )
    blockers = set()
    if not bridge.get("all_three_accelerators_same_request_verified"):
        blockers.add("retained_32b_three_accelerator_dense_decode_missing")
    if adapter.get("tpu_jax_qwen_stage_runtime_ready") is not True:
        blockers.add("tpu_dense_qwen_jax_stage_runtime_not_verified")
    if adapter.get("torch_reference_forward_ready") is not True:
        blockers.add("dense_qwen_torch_reference_forward_missing")
    if sources.get("kaggle_model_attach_available") is not True:
        blockers.add("kaggle_dense_model_attach_source_missing")
    if sources.get("kaggle_model_attach_used") is not True:
        blockers.add("kaggle_model_attach_not_mounted_in_current_runtime")
    if gpu_cpu_fallback.get("gpu_cpu_dense_fallback_verified") is not True:
        blockers.add("gpu_cpu_dense_fallback_not_verified")
    if decoded != str(largest_attempted.get("parameter_class") or ""):
        blockers.add("larger_than_32b_dense_decode_not_verified")
        if largest_stage_preflighted == str(largest_attempted.get("parameter_class") or ""):
            blockers.add("larger_dense_live_stage_load_not_verified_after_stage_preflight")
        elif largest_attached == str(largest_attempted.get("parameter_class") or ""):
            blockers.add("larger_dense_same_request_decode_not_verified_after_model_attach")
    blockers.update(str(item) for item in _list(adapter.get("blockers")) if item)
    for item in _list(sources.get("blockers")):
        blocker = str(item)
        if sources.get("kaggle_model_attach_used") is True and blocker in {
            "kaggle_model_attach_paths_not_present_in_current_runtime",
            "kaggle_model_attach_not_mounted_in_current_runtime",
        }:
            continue
        if blocker:
            blockers.add(blocker)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "three_accelerator_dense_qwen_frontier_ready": True,
        "output_dir": str(output_dir),
        "goal_scope": {
            "single_account_kaggle_only": True,
            "requires_gpu_tpu_cpu_same_request_for_success": True,
            "dense_full_precision_main_path": True,
            "quantized_large_model_main_path_allowed": False,
            "kaggle_models_attach_preferred": True,
        },
        "largest_dense_model_attempted": str(largest_attempted.get("parameter_class") or ""),
        "largest_dense_model_attach_candidate": str(largest_attempted.get("parameter_class") or ""),
        "largest_dense_model_attached": largest_attached,
        "largest_dense_model_stage_preflighted": largest_stage_preflighted,
        "largest_dense_model_loaded": largest_loaded,
        "largest_dense_model_1token_decoded": decoded,
        "all_three_accelerators_same_request_verified": bool(bridge.get("all_three_accelerators_same_request_verified")),
        "same_request_dense_32b_success": same_request_32b_ready,
        "same_request_dense_frontier_success": same_request_ready,
        "kaggle_model_attach_used": bool(sources.get("kaggle_model_attach_used")),
        "kaggle_model_attach_available": bool(sources.get("kaggle_model_attach_available")),
        "tpu_jax_qwen_stage_runtime_ready": bool(adapter.get("tpu_jax_qwen_stage_runtime_ready")),
        "gpu_stage_runtime_ready": bool(
            (bridge.get("all_three_accelerators_same_request_verified") and "cuda" in set(bridge.get("accepted_stage_backends") or []))
            or gpu_cpu_fallback.get("gpu_stage_runtime_ready") is True
        ),
        "cpu_stage_runtime_ready": bool(
            (bridge.get("all_three_accelerators_same_request_verified") and "cpu" in set(bridge.get("accepted_stage_backends") or []))
            or gpu_cpu_fallback.get("cpu_stage_runtime_ready") is True
        ),
        "generated_token_count": _int(bridge.get("generated_token_count")),
        "model_sources": sources,
        "kaggle_model_attach_probe": attach_probe,
        "tpu_dense_qwen_adapter": adapter,
        "retained_tpu_dense_qwen_stage": tpu_loader,
        "tpu_dense_qwen_adapter_source": source_summary(
            adapter_artifact_path,
            adapter_report,
            kind="qwen_dense_jax_tpu_stage_adapter_smoke",
        ),
        "baseline_32b_three_accelerator": bridge,
        "baseline_32b_gpu_cpu_dense_fallback": gpu_cpu_fallback,
        "bounded_experiment_ladder": [
            {
                "parameter_class": item.get("parameter_class"),
                "hf_repo": item.get("hf_repo"),
                "model_attach_source": item.get("kaggle_kernel_model_source"),
                "metadata_ready": item.get("metadata_ready"),
                "attach_can_avoid_runtime_download": item.get("attach_can_avoid_runtime_download"),
                "attach_path_present": item.get("attach_path_present"),
                "live_attach_verified": item.get("live_attach_verified") is True,
                "live_stage_plan_verified": item.get("live_stage_plan_verified") is True,
                "live_attached_runtime_path": item.get("live_attached_runtime_path", ""),
                "loaded_verified": item.get("parameter_class") == largest_loaded,
                "one_token_decode_verified": item.get("parameter_class") == decoded,
                "blocked_reason": "" if item.get("parameter_class") == decoded else (
                    "dense_live_stage_load_not_verified_after_stage_preflight"
                    if item.get("live_stage_plan_verified") is True
                    else
                    "dense_same_request_decode_not_verified_after_model_attach"
                    if item.get("live_attach_verified") is True
                    else "dense_same_request_decode_not_verified_for_this_size"
                ),
            }
            for item in dense_candidates
        ],
        "frontier_failure_stage": (
            ""
            if same_request_ready
            else "larger_dense_live_stage_load_not_verified_after_stage_preflight"
            if largest_stage_preflighted == str(largest_attempted.get("parameter_class") or "")
            else "larger_dense_same_request_decode_not_verified_after_model_attach"
            if largest_attached == str(largest_attempted.get("parameter_class") or "")
            else "larger_dense_model_attach_not_verified"
            if largest_attached != str(largest_attempted.get("parameter_class") or "")
            else "unknown"
        ),
        "cleanup_status": {
            "temporary_kaggle_kernels_created": False,
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "note": "This pack did not create Kaggle resources; imported live reports must carry their own cleanup evidence.",
        },
        "blocker_codes": sorted(blockers),
        "diagnosis_codes": [
            "three_accelerator_dense_qwen_frontier_ready",
            "retained_32b_three_accelerator_decode_verified" if bridge.get("all_three_accelerators_same_request_verified") else "retained_32b_three_accelerator_decode_missing",
            "same_request_dense_32b_success" if same_request_32b_ready else "same_request_dense_32b_missing",
            "tpu_dense_qwen_adapter_ready" if adapter.get("tpu_jax_qwen_stage_runtime_ready") else "tpu_dense_qwen_adapter_not_ready",
            "kaggle_dense_model_attach_available" if sources.get("kaggle_model_attach_available") else "kaggle_dense_model_attach_missing",
            "larger_than_32b_dense_decode_not_verified" if decoded != str(largest_attempted.get("parameter_class") or "") else "largest_dense_attempt_decode_verified",
        ],
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
            "private_runtime_state_public": False,
            "private_kaggle_payload_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
        "limitations": [
            "A dense model attach candidate is not a decode proof.",
            "A stage-owned safetensors header preflight is not a live weight load or decode proof.",
            "A Kaggle Models attach source or metadata record is not counted as loaded unless a live loader/decode proof exists.",
            "The retained 32B same-request proof is the current largest dense GPU+TPU+CPU decode proof.",
            "A larger dense Qwen model requires live TPU/JAX stage execution plus same-request activation handoff before it can be counted as decoded.",
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["three_accelerator_dense_qwen_frontier_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blocker_codes"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "three_accelerator_dense_qwen_frontier.json"
    support_path = output_dir / "support_bundle.json"
    write_json(summary_path, report)
    support = build_support_bundle(report)
    write_json(support_path, support)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
        "support_bundle_json": artifact_entry(support_path, output_dir, kind="support_bundle_json", schema=SUPPORT_BUNDLE_SCHEMA, ok=bool(support.get("ok"))),
        "model_source_resolver_json": artifact_entry(output_dir / "model-source-resolver" / "kaggle_dense_model_source_resolver.json", output_dir, kind="model_source_resolver_json", schema=resolver.SCHEMA, ok=bool(resolver_report.get("ok"))),
        "adapter_smoke_json": artifact_entry(adapter_artifact_path, output_dir, kind="adapter_smoke_json", schema=adapter_smoke.SCHEMA, ok=bool(adapter_report.get("ok"))),
        "gpu_cpu_dense_fallback_json": artifact_entry(Path(args.gpu_cpu_dense_fallback_report), output_dir, kind="gpu_cpu_dense_fallback_json", schema="kaggle_32b_full_heterogeneous_probe_v1", ok=bool(gpu_cpu_fallback.get("gpu_cpu_dense_fallback_verified"))),
        "tpu_dense_loader_json": artifact_entry(Path(args.tpu_dense_loader_report), output_dir, kind="tpu_dense_loader_json", schema="kaggle_tpu_32b_stage_owned_loader_probe_v1", ok=bool(tpu_loader.get("tpu_jax_qwen_stage_runtime_ready"))),
        "kaggle_model_attach_probe_json": artifact_entry(Path(args.kaggle_model_attach_probe_report), output_dir, kind="kaggle_model_attach_probe_json", schema="kaggle_model_attach_probe_v1", ok=bool(attach_probe.get("kaggle_model_attach_probe_ready"))) if args.kaggle_model_attach_probe_report else {"kind": "kaggle_model_attach_probe_json", "present": False, "ok": False},
    }
    write_json(summary_path, report)
    write_json(support_path, build_support_bundle(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense Qwen GPU+TPU+CPU frontier artifact.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-32b-bridge-report", default=DEFAULT_32B_BRIDGE_REPORT)
    parser.add_argument("--gpu-cpu-dense-fallback-report", default=DEFAULT_GPU_CPU_DENSE_FALLBACK_REPORT)
    parser.add_argument("--tpu-dense-loader-report", default=DEFAULT_TPU_DENSE_LOADER_REPORT)
    parser.add_argument("--kaggle-model-attach-probe-report", default=DEFAULT_KAGGLE_MODEL_ATTACH_PROBE_REPORT)
    parser.add_argument("--dense-adapter-report", default=DEFAULT_DENSE_ADAPTER_REPORT)
    parser.add_argument("--kaggle-input-root", default="/kaggle/input")
    parser.add_argument("--fetch-hf-metadata", action="store_true")
    parser.add_argument("--hf-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--adapter-sequence-length", type=int, default=4)
    parser.add_argument("--run-jax-adapter", action="store_true")
    parser.add_argument("--require-tpu-adapter", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.adapter_sequence_length < 1 or args.adapter_sequence_length > 2048:
        raise SystemExit("--adapter-sequence-length must be between 1 and 2048")
    if args.require_tpu_adapter and not args.run_jax_adapter:
        raise SystemExit("--require-tpu-adapter requires --run-jax-adapter")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'three_accelerator_dense_qwen_frontier.json'}")
        print(f"Ready: {report.get('three_accelerator_dense_qwen_frontier_ready')}")
        print(f"Largest dense attempted: {report.get('largest_dense_model_attempted')}")
        print(f"Largest dense 1-token decoded: {report.get('largest_dense_model_1token_decoded')}")
        if report.get("blocker_codes"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blocker_codes") or []))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
