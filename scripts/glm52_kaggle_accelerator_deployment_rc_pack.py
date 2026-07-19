#!/usr/bin/env python3
"""Build GLM 5.2 Kaggle accelerator deployment RC evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_accelerator_deployment_rc_v1"
SUPPORT_SCHEMA = "glm52_kaggle_accelerator_deployment_rc_support_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-accelerator-deployment-rc"
MODEL_ID = "zai-org/GLM-5.2"
REQUIRED_PROVIDERS = ["kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"]
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"input_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def load_jsons(paths: list[str] | str | None) -> list[dict[str, Any]]:
    if not paths:
        return []
    if isinstance(paths, str):
        paths = [paths]
    return [report for report in (load_json(path) for path in paths) if report]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def summarize_source(report: dict[str, Any]) -> dict[str, Any]:
    model = _dict(report.get("model"))
    candidate = _dict(report.get("recommended_deployment_candidate"))
    stage_plan = _dict(report.get("stage_adapter_plan"))
    attach_plan = _dict(report.get("kaggle_attach_plan"))
    return {
        "schema": "glm52_source_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "resolver_ready": report.get("glm52_source_resolver_ready") is True,
        "model_id": str(model.get("model_id") or ""),
        "compatible_with_glm52": str(model.get("model_id") or "") == MODEL_ID,
        "architecture_class": str(model.get("architecture_class") or ""),
        "model_type": str(model.get("model_type") or ""),
        "num_hidden_layers": _int(model.get("num_hidden_layers")),
        "official_weight_key_count": _int(model.get("official_weight_key_count")),
        "official_weight_total_size_gb": model.get("official_weight_total_size_gb", 0),
        "candidate_count": _int(report.get("candidate_count")),
        "ready_candidate_count": _int(report.get("ready_candidate_count")),
        "recommended_candidate_id": str(candidate.get("candidate_id") or ""),
        "recommended_repo": str(candidate.get("repo") or ""),
        "recommended_format": str(candidate.get("format") or ""),
        "recommended_quantization": str(candidate.get("quantization") or ""),
        "recommended_known_total_size_gb": candidate.get("known_total_size_gb", 0),
        "recommended_blockers": [str(item) for item in _list(candidate.get("blockers"))],
        "stage_adapter_plan_ready": stage_plan.get("metadata_only") is True and _int(stage_plan.get("assigned_key_count_total")) > 0,
        "stage_runtime_adapter_verified": stage_plan.get("stage_runtime_adapter_verified") is True,
        "same_request_route_verified": stage_plan.get("same_request_route_verified") is True,
        "kaggle_models_source_verified": attach_plan.get("kaggle_models_source_verified") is True,
        "hf_source_verified": attach_plan.get("hf_source_verified") is True,
        "full_runtime_download_supported": attach_plan.get("full_runtime_download_supported") is True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_tpu(report: dict[str, Any]) -> dict[str, Any]:
    observations = [item for item in _list(report.get("observations")) if isinstance(item, dict)]
    last = observations[-1] if observations else {}
    status = str(report.get("last_status") or last.get("status") or "")
    ready = (
        report.get("tpu_runtime_ready") is True
        or report.get("web_tpu_execution_channel_ready") is True
        or report.get("llm_inference_ready") is True
        or report.get("tpu_stage_runtime_ready") is True
        or report.get("stage_runtime_report_verified") is True
    )
    queued = "QUEUED" in status.upper()
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if queued and not ready:
        blockers.append("kaggle_tpu_scheduler_queued")
    return {
        "schema": "glm52_kaggle_tpu_request_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "ref": str(report.get("ref") or report.get("notebook_ref") or ""),
        "last_status": status,
        "observation_count": len(observations),
        "tpu_runtime_ready": ready,
        "tpu_stage_runtime_ready": report.get("tpu_stage_runtime_ready") is True,
        "stage_runtime_report_verified": report.get("stage_runtime_report_verified") is True,
        "queued": queued,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_awq_stage_header(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_awq_stage_header_summary_v1",
            "present": False,
            "stage_header_ready": False,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "blockers": ["glm52_awq_stage_header_report_missing"],
            "public_artifact_safe": True,
        }
    return {
        "schema": "glm52_awq_stage_header_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "stage_header_ready": report.get("glm52_awq_stage_header_ready") is True,
        "model_repo": str(report.get("model_repo") or ""),
        "base_model_id": str(report.get("base_model_id") or ""),
        "quantization": str(report.get("quantization") or ""),
        "stage_id": _int(report.get("stage_id")),
        "stage_count": _int(report.get("stage_count")),
        "stage_layer_range": _list(report.get("stage_layer_range")),
        "assigned_weight_key_count": _int(report.get("assigned_weight_key_count")),
        "assigned_weight_file_count": _int(report.get("assigned_weight_file_count")),
        "header_file_count": _int(report.get("header_file_count")),
        "present_stage_key_count": _int(report.get("present_stage_key_count")),
        "missing_stage_key_count": _int(report.get("missing_stage_key_count")),
        "dtype_counts": _dict(report.get("dtype_counts")),
        "stage_family_hits": _dict(report.get("stage_family_hits")),
        "total_selected_tensor_storage_gb": report.get("total_selected_tensor_storage_gb", 0),
        "weight_tensor_values_loaded": report.get("weight_tensor_values_loaded") is True,
        "weight_tensor_values_public": report.get("weight_tensor_values_public") is True,
        "safetensors_header_payload_public": report.get("safetensors_header_payload_public") is True,
        "stage_runtime_adapter_verified": report.get("stage_runtime_adapter_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_awq_stage_value_probes(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {
            "schema": "glm52_awq_stage_value_probe_summary_v1",
            "present": False,
            "stage_value_probe_ready": False,
            "stage_value_probe_count": 0,
            "stage_value_probe_ready_count": 0,
            "weight_tensor_values_loaded": False,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    probe_summaries: list[dict[str, Any]] = []
    blockers: list[str] = []
    ready_count = 0
    provider_coverage: set[str] = set()
    public_safe = True
    for report in reports:
        selected = _dict(report.get("selected_tensor"))
        ready = bool(
            report.get("glm52_awq_stage_value_probe_ready") is True
            and report.get("weight_tensor_values_loaded") is True
            and _int(report.get("weight_value_byte_count")) > 0
            and _hash_ok(report.get("weight_value_sha256"))
        )
        if ready:
            ready_count += 1
            blockers.append("glm52_awq_stage_value_probe_is_not_runtime_success")
        else:
            blockers.append("glm52_awq_stage_value_probe_not_ready")
        blockers.extend(str(item) for item in _list(report.get("blockers")) if item)
        stage_count = _int(report.get("stage_count"))
        stage_id = _int(report.get("stage_id"), -1)
        provider = REQUIRED_PROVIDERS[stage_id] if stage_count == len(REQUIRED_PROVIDERS) and 0 <= stage_id < len(REQUIRED_PROVIDERS) else ""
        if provider and ready:
            provider_coverage.add(provider)
        public_safe = public_safe and report.get("public_artifact_safe") is True
        probe_summaries.append(
            {
                "source_schema": str(report.get("schema") or ""),
                "source_ok": report.get("ok") is True,
                "stage_value_probe_ready": ready,
                "provider": provider,
                "stage_id": stage_id,
                "stage_count": stage_count,
                "stage_layer_range": _list(report.get("stage_layer_range")),
                "assigned_weight_key_count": _int(report.get("assigned_weight_key_count")),
                "assigned_weight_file_count": _int(report.get("assigned_weight_file_count")),
                "header_file_count": _int(report.get("header_file_count")),
                "selected_tensor_dtype": str(selected.get("dtype") or ""),
                "selected_tensor_rank": _int(selected.get("rank"), -1),
                "selected_tensor_nbytes": _int(selected.get("tensor_nbytes")),
                "weight_value_byte_count": _int(report.get("weight_value_byte_count")),
                "weight_value_hash_present": _hash_ok(report.get("weight_value_sha256")),
                "weight_tensor_values_loaded": report.get("weight_tensor_values_loaded") is True,
                "weight_tensor_values_public": report.get("weight_tensor_values_public") is True,
                "safetensors_header_payload_public": report.get("safetensors_header_payload_public") is True,
                "stage_runtime_adapter_verified": report.get("stage_runtime_adapter_verified") is True,
                "same_request_route_verified": report.get("same_request_route_verified") is True,
                "same_request_decode_verified": report.get("same_request_decode_verified") is True,
                "stage_smoke_only": report.get("stage_smoke_only") is True,
                "public_artifact_safe": report.get("public_artifact_safe") is True,
            }
        )
    provider_aligned = set(REQUIRED_PROVIDERS).issubset(provider_coverage)
    if reports and not provider_aligned:
        blockers.append("glm52_awq_stage_value_probe_provider_coverage_incomplete")
    return {
        "schema": "glm52_awq_stage_value_probe_summary_v1",
        "present": True,
        "stage_value_probe_ready": ready_count == len(reports),
        "stage_value_probe_count": len(reports),
        "stage_value_probe_ready_count": ready_count,
        "provider_aligned_stage_value_probe_ready": provider_aligned,
        "provider_coverage": sorted(provider_coverage),
        "probes": probe_summaries,
        "weight_tensor_values_loaded": bool(reports) and ready_count == len(reports),
        "weight_tensor_values_public": any(item.get("weight_tensor_values_public") for item in probe_summaries),
        "safetensors_header_payload_public": any(item.get("safetensors_header_payload_public") for item in probe_summaries),
        "stage_runtime_adapter_verified": any(item.get("stage_runtime_adapter_verified") for item in probe_summaries),
        "same_request_route_verified": any(item.get("same_request_route_verified") for item in probe_summaries),
        "same_request_decode_verified": any(item.get("same_request_decode_verified") for item in probe_summaries),
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": public_safe,
    }


def summarize_tpu_stage_smoke(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_awq_tpu_stage_smoke_summary_v1",
            "present": False,
            "stage_runtime_adapter_smoke_ready": False,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    source_schema = str(report.get("schema") or "")
    status = str(report.get("last_status") or "")
    is_watch_report = source_schema == "glm52_kaggle_tpu_awq_stage_smoke_watch_v1" or (
        status and not report.get("model_repo") and not report.get("base_model_id")
    )
    if is_watch_report:
        queued = "QUEUED" in status.upper()
        stage_check = _dict(report.get("stage_smoke_check"))
        stage_output = _dict(report.get("stage_smoke_output"))
        stage_summary = _dict(report.get("stage_smoke_summary"))
        smoke_ready = bool(
            report.get("notebook_output_verified") is True
            and report.get("stage_runtime_adapter_smoke_ready") is True
            and stage_check.get("ok") is True
            and stage_output.get("present") is True
        )
        blockers = []
        if not smoke_ready:
            blockers.append("glm52_awq_tpu_stage_smoke_scheduler_queued" if queued else "glm52_awq_tpu_stage_smoke_output_missing")
        if smoke_ready:
            blockers.append("glm52_awq_tpu_stage_runtime_adapter_not_verified")
        return {
            "schema": "glm52_awq_tpu_stage_smoke_summary_v1",
            "present": True,
            "source_schema": source_schema,
            "source_ok": report.get("ok") is True,
            "queued_watch": True,
            "ref": str(report.get("ref") or ""),
            "last_status": status,
            "queued": queued,
            "stage_runtime_adapter_smoke_ready": smoke_ready,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "same_request_decode_verified": False,
            "tpu_runtime_ready": report.get("tpu_runtime_ready") is True,
            "jax_tpu_device_count": _int(stage_summary.get("jax_tpu_device_count")),
            "jax_shape_smoke_ready": smoke_ready,
            "stage_header_ready": smoke_ready,
            "model_repo": str(stage_summary.get("model_repo") or ""),
            "base_model_id": str(stage_summary.get("base_model_id") or ""),
            "quantization": str(stage_summary.get("quantization") or ""),
            "stage_id": _int(stage_summary.get("stage_id")),
            "stage_count": _int(stage_summary.get("stage_count")),
            "stage_layer_range": _list(stage_summary.get("stage_layer_range")),
            "assigned_weight_key_count": _int(stage_summary.get("assigned_weight_key_count")),
            "assigned_weight_file_count": 0,
            "header_file_count": 0,
            "present_stage_key_count": _int(stage_summary.get("present_stage_key_count")),
            "missing_stage_key_count": _int(stage_summary.get("missing_stage_key_count")),
            "dtype_counts": {},
            "stage_family_hits": {},
            "total_selected_tensor_storage_gb": 0,
            "weight_tensor_values_loaded": False,
            "weight_tensor_values_public": stage_summary.get("weight_tensor_values_public") is True,
            "safetensors_header_payload_public": False,
            "blockers": blockers,
            "public_artifact_safe": report.get("public_artifact_safe") is True,
        }
    assigned = _int(report.get("assigned_weight_key_count") or report.get("assigned_stage_key_count"))
    present_keys = _int(report.get("present_stage_key_count") or report.get("present_weight_key_count"))
    missing = _int(report.get("missing_stage_key_count") or report.get("missing_weight_key_count"))
    tpu_count = _int(report.get("jax_tpu_device_count") or report.get("tpu_device_count"))
    stage_header_ready = bool(
        report.get("glm52_awq_stage_header_ready") is True
        or report.get("stage_header_ready") is True
        or (assigned > 0 and present_keys == assigned and missing == 0)
    )
    jax_shape_ready = bool(
        report.get("jax_shape_smoke_ready") is True
        or report.get("jax_bf16_matmul_smoke_ready") is True
        or report.get("jax_tpu_stage_smoke_ready") is True
    )
    tpu_ready = bool(report.get("tpu_runtime_ready") is True or tpu_count > 0)
    smoke_ready = bool(
        report.get("ok") is True
        and tpu_ready
        and stage_header_ready
        and jax_shape_ready
        and str(report.get("base_model_id") or "") == MODEL_ID
    )
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if not tpu_ready:
        blockers.append("glm52_awq_tpu_runtime_not_ready")
    if not stage_header_ready:
        blockers.append("glm52_awq_tpu_stage_header_not_ready")
    if not jax_shape_ready:
        blockers.append("glm52_awq_tpu_jax_shape_smoke_not_ready")
    if smoke_ready and report.get("stage_runtime_adapter_verified") is not True:
        blockers.append("glm52_awq_tpu_stage_runtime_adapter_not_verified")
    return {
        "schema": "glm52_awq_tpu_stage_smoke_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "stage_runtime_adapter_smoke_ready": smoke_ready,
        "stage_runtime_adapter_verified": report.get("stage_runtime_adapter_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "tpu_runtime_ready": tpu_ready,
        "jax_tpu_device_count": tpu_count,
        "jax_shape_smoke_ready": jax_shape_ready,
        "stage_header_ready": stage_header_ready,
        "model_repo": str(report.get("model_repo") or ""),
        "base_model_id": str(report.get("base_model_id") or ""),
        "quantization": str(report.get("quantization") or ""),
        "stage_id": _int(report.get("stage_id")),
        "stage_count": _int(report.get("stage_count")),
        "stage_layer_range": _list(report.get("stage_layer_range")),
        "assigned_weight_key_count": assigned,
        "assigned_weight_file_count": _int(report.get("assigned_weight_file_count")),
        "header_file_count": _int(report.get("header_file_count")),
        "present_stage_key_count": present_keys,
        "missing_stage_key_count": missing,
        "dtype_counts": _dict(report.get("dtype_counts")),
        "stage_family_hits": _dict(report.get("stage_family_hits")),
        "total_selected_tensor_storage_gb": report.get("total_selected_tensor_storage_gb", 0),
        "weight_tensor_values_loaded": report.get("weight_tensor_values_loaded") is True,
        "weight_tensor_values_public": report.get("weight_tensor_values_public") is True,
        "safetensors_header_payload_public": report.get("safetensors_header_payload_public") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_kaggle_source_search(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_kaggle_public_source_search_summary_v1",
            "present": False,
            "search_ready": False,
            "kaggle_attach_source_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    attach_verified = report.get("kaggle_attach_source_verified") is True
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if not attach_verified:
        blockers.append("glm52_kaggle_attach_source_not_found")
    return {
        "schema": "glm52_kaggle_public_source_search_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "search_ready": report.get("glm52_kaggle_public_source_search_ready") is True,
        "kaggle_models_glm52_source_verified": report.get("kaggle_models_glm52_source_verified") is True,
        "kaggle_datasets_glm52_source_verified": report.get("kaggle_datasets_glm52_source_verified") is True,
        "kaggle_attach_source_verified": attach_verified,
        "query_count": _int(report.get("query_count")),
        "model_result_count": _int(report.get("model_result_count")),
        "dataset_result_count": _int(report.get("dataset_result_count")),
        "compatible_model_source_count": _int(report.get("compatible_model_source_count")),
        "compatible_dataset_source_count": _int(report.get("compatible_dataset_source_count")),
        "recommended_kaggle_kernel_model_sources": [
            str(item) for item in _list(report.get("recommended_kaggle_kernel_model_sources"))[:12]
        ],
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_stage_runtime_plan(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_stage_runtime_plan_summary_v1",
            "present": False,
            "plan_ready": False,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "provider_specs": [],
            "blockers": [],
            "public_artifact_safe": True,
        }
    specs = [item for item in _list(report.get("stage_specs")) if isinstance(item, dict)]
    providers = [str(spec.get("provider") or "") for spec in specs]
    blockers = [str(item) for item in _list(report.get("blockers"))]
    for provider in REQUIRED_PROVIDERS:
        if provider not in providers:
            blockers.append(f"stage_runtime_plan_provider_missing:{provider}")
    return {
        "schema": "glm52_stage_runtime_plan_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "plan_ready": report.get("glm52_stage_runtime_plan_ready") is True,
        "stage_runtime_adapter_verified": report.get("stage_runtime_adapter_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
        "provider_specs": [
            {
                "provider": str(spec.get("provider") or ""),
                "stage_id": _int(spec.get("stage_id")),
                "stage_layer_range": _list(spec.get("stage_layer_range")),
                "expected_stage_report_schema": str(spec.get("expected_stage_report_schema") or ""),
                "stage_runtime_adapter_verified": spec.get("stage_runtime_adapter_verified") is True,
                "same_request_route_verified": spec.get("same_request_route_verified") is True,
            }
            for spec in specs
        ],
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_stage_worker_package(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_stage_worker_package_summary_v1",
            "present": False,
            "package_ready": False,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "live_run_performed": False,
            "provider_packages": [],
            "blockers": [],
            "public_artifact_safe": True,
        }
    packages = [item for item in _list(report.get("packages")) if isinstance(item, dict)]
    providers = [str(pkg.get("provider") or "") for pkg in packages]
    blockers = [str(item) for item in _list(report.get("blockers"))]
    for provider in REQUIRED_PROVIDERS:
        if provider not in providers:
            blockers.append(f"stage_worker_package_provider_missing:{provider}")
    return {
        "schema": "glm52_stage_worker_package_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "package_ready": report.get("glm52_stage_worker_package_ready") is True,
        "stage_runtime_adapter_verified": report.get("stage_runtime_adapter_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
        "live_run_performed": report.get("live_run_performed") is True,
        "stage_runtime_package_kind": str(report.get("stage_runtime_package_kind") or ""),
        "full_prefix_probe_mode": str(report.get("full_prefix_probe_mode") or ""),
        "full_prefix_probe_full_stage_requested": report.get("full_prefix_probe_full_stage_requested") is True,
        "full_prefix_runtime_bundle_required": report.get("full_prefix_runtime_bundle_required") is True,
        "provider_packages": [
            {
                "provider": str(pkg.get("provider") or ""),
                "stage_id": _int(pkg.get("stage_id")),
                "kaggle_owner": str(pkg.get("kaggle_owner") or ""),
                "stage_layer_range": _list(pkg.get("stage_layer_range")),
                "full_prefix_probe_mode": str(pkg.get("full_prefix_probe_mode") or ""),
                "full_prefix_probe_layer_range": _list(pkg.get("full_prefix_probe_layer_range")),
                "full_prefix_probe_covers_full_stage": pkg.get("full_prefix_probe_covers_full_stage") is True,
                "expected_stage_report_schema": str(pkg.get("expected_stage_report_schema") or ""),
                "stage_runtime_package_kind": str(pkg.get("stage_runtime_package_kind") or ""),
                "full_prefix_runtime_bundle_present": pkg.get("full_prefix_runtime_bundle_present") is True,
                "embedded_runtime_bundle_present": pkg.get("embedded_runtime_bundle_present") is True,
                "embedded_runtime_bundle_file_count": _int(pkg.get("embedded_runtime_bundle_file_count")),
                "private_kernel": pkg.get("private_kernel") is True,
                "pushed_to_kaggle": pkg.get("pushed_to_kaggle") is True,
                "live_run_performed": pkg.get("live_run_performed") is True,
            }
            for pkg in packages
        ],
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_stage_worker_push_probe(
    report: dict[str, Any],
    *,
    stage_worker_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_stage_worker_push_probe_summary_v1",
            "present": False,
            "push_probe_ready": False,
            "live_run_performed": False,
            "stage_runtime_reports_collected": 0,
            "stage_runtime_reports_verified": 0,
            "required_stage_runtime_reports_verified": False,
            "required_provider_stage_runtime_reports_verified": False,
            "all_planned_stage_runtime_reports_verified": False,
            "planned_stage_count": 0,
            "missing_planned_stage_count": 0,
            "missing_planned_stage_coverage": [],
            "verified_provider_coverage": [],
            "verified_stage_count": 0,
            "verified_stage_coverage": [],
            "provider_pushes": [],
            "blockers": [],
            "public_artifact_safe": True,
        }
    pushes = [item for item in _list(report.get("pushes")) if isinstance(item, dict)]
    providers = [str(push.get("provider") or "") for push in pushes]
    blockers = [str(item) for item in _list(report.get("blockers"))]
    for provider in REQUIRED_PROVIDERS:
        if provider not in providers:
            blockers.append(f"stage_worker_push_provider_missing:{provider}")
    verified_providers = {
        str(push.get("provider") or "")
        for push in pushes
        if isinstance(push, dict) and push.get("stage_runtime_verified") is True
    }
    verified_stages = [
        {
            "provider": str(push.get("provider") or ""),
            "stage_id": _int(push.get("stage_id")),
            "stage_report_path": str(push.get("stage_report_path") or ""),
            "terminal_status": str(push.get("terminal_status") or ""),
            "cleanup_performed": push.get("cleanup_performed") is True,
        }
        for push in pushes
        if isinstance(push, dict) and push.get("stage_runtime_verified") is True
    ]
    verified_stage_keys = {
        (str(item.get("provider") or ""), _int(item.get("stage_id"), -1))
        for item in verified_stages
        if str(item.get("provider") or "")
    }
    package_entries = [
        item
        for item in _list(_dict(stage_worker_package).get("provider_packages"))
        if isinstance(item, dict)
    ]
    planned_stage_keys = {
        (str(item.get("provider") or ""), _int(item.get("stage_id"), -1))
        for item in package_entries
        if str(item.get("provider") or "")
    }
    if not planned_stage_keys:
        planned_stage_keys = {
            (str(push.get("provider") or ""), _int(push.get("stage_id"), -1))
            for push in pushes
            if isinstance(push, dict) and str(push.get("provider") or "")
        }
    missing_planned_stage_keys = sorted(planned_stage_keys - verified_stage_keys)
    required_provider_coverage_verified = set(REQUIRED_PROVIDERS).issubset(verified_providers)
    planned_stage_count = len(planned_stage_keys)
    all_planned_stage_runtime_reports_verified = bool(planned_stage_keys) and not missing_planned_stage_keys
    required_verified = required_provider_coverage_verified and all_planned_stage_runtime_reports_verified
    return {
        "schema": "glm52_stage_worker_push_probe_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "mode": str(report.get("mode") or ""),
        "push_probe_ready": report.get("glm52_stage_worker_push_probe_ready") is True,
        "live_run_performed": report.get("live_run_performed") is True,
        "stage_runtime_reports_collected": _int(report.get("stage_runtime_reports_collected")),
        "stage_runtime_reports_verified": _int(report.get("stage_runtime_reports_verified")),
        "required_stage_runtime_reports_verified": required_verified,
        "required_provider_stage_runtime_reports_verified": required_provider_coverage_verified,
        "all_planned_stage_runtime_reports_verified": all_planned_stage_runtime_reports_verified,
        "planned_stage_count": planned_stage_count,
        "missing_planned_stage_count": len(missing_planned_stage_keys),
        "missing_planned_stage_coverage": [
            {"provider": provider, "stage_id": stage_id}
            for provider, stage_id in missing_planned_stage_keys
        ],
        "verified_provider_coverage": sorted(verified_providers),
        "verified_stage_count": len(verified_stages),
        "verified_stage_coverage": sorted(verified_stages, key=lambda item: (item["provider"], item["stage_id"])),
        "stage_runtime_adapter_verified": report.get("stage_runtime_adapter_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
        "provider_pushes": [
            {
                "provider": str(push.get("provider") or ""),
                "stage_id": _int(push.get("stage_id")),
                "pushed": push.get("pushed") is True,
                "terminal_status": str(push.get("terminal_status") or ""),
                "output_collected": push.get("output_collected") is True,
                "stage_report_present": push.get("stage_report_present") is True,
                "stage_runtime_verified": push.get("stage_runtime_verified") is True,
                "stage_report_check_ok": _dict(push.get("stage_report_check")).get("ok") is True,
                "stage_report_path": str(push.get("stage_report_path") or ""),
                "cleanup_performed": push.get("cleanup_performed") is True,
            }
            for push in pushes
        ],
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_gpu_token_quota(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_gpu_token_quota_summary_v1",
            "present": False,
            "auth_ok_count": 0,
            "account_count": 0,
            "gpu_submission_accepted_count": 0,
            "gpu_session_limit_rejected_count": 0,
            "weekly_gpu_quota_exhausted_count": 0,
            "gpu_slot_available": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    source_summary = _dict(report.get("summary"))
    accounts = [item for item in _list(report.get("accounts")) if isinstance(item, dict)]
    accepted_count = _int(source_summary.get("gpu_submission_accepted_count"))
    session_limit_count = _int(source_summary.get("gpu_session_limit_rejected_count"))
    auth_failed_count = _int(source_summary.get("auth_failed_count"))
    exhausted_count = _int(source_summary.get("weekly_gpu_quota_exhausted_count"))
    exhausted_by_api_count = _int(source_summary.get("weekly_gpu_quota_exhausted_by_api_count"))
    reserved_exceeds_count = _int(source_summary.get("gpu_reserved_exceeds_remaining_by_api_count"))
    account_count = _int(source_summary.get("account_count"), len(accounts))
    blockers: list[str] = []
    if accepted_count <= 0:
        blockers.append("kaggle_gpu_submission_not_accepted")
    if session_limit_count > 0:
        blockers.append("kaggle_gpu_batch_session_limit_reached")
    if account_count > 0 and session_limit_count >= account_count:
        blockers.append("kaggle_gpu_all_accounts_session_limited")
    if auth_failed_count > 0:
        blockers.append("kaggle_gpu_token_auth_failed")
    if exhausted_count > 0:
        blockers.append("kaggle_weekly_gpu_quota_exhausted")
    if exhausted_by_api_count > 0:
        blockers.append("kaggle_weekly_gpu_quota_exhausted_by_api")
    if reserved_exceeds_count > 0:
        blockers.append("kaggle_gpu_reserved_time_exceeds_remaining_quota")
    return {
        "schema": "glm52_gpu_token_quota_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "requested_accelerator": str(report.get("requested_accelerator") or ""),
        "account_count": account_count,
        "auth_ok_count": _int(source_summary.get("auth_ok_count")),
        "auth_failed_count": auth_failed_count,
        "gpu_submission_accepted_count": accepted_count,
        "gpu_session_limit_rejected_count": session_limit_count,
        "weekly_gpu_quota_exhausted_count": exhausted_count,
        "weekly_gpu_quota_exhausted_by_api_count": exhausted_by_api_count,
        "gpu_reserved_exceeds_remaining_by_api_count": reserved_exceeds_count,
        "gpu_slot_available": accepted_count > 0,
        "quota_classes": _dict(source_summary.get("quota_classes")),
        "accounts": [
            {
                "label": str(account.get("label") or ""),
                "owner": str(account.get("owner") or ""),
                "auth_ok": account.get("auth_ok") is True,
                "push_accepted": account.get("push_accepted") is True,
                "quota_class": str(account.get("quota_class") or ""),
                "weekly_gpu_quota_exhausted": account.get("weekly_gpu_quota_exhausted") is True,
                "weekly_gpu_quota_exhausted_by_api": account.get("weekly_gpu_quota_exhausted_by_api") is True,
                "gpu_reserved_exceeds_remaining_by_api": account.get("gpu_reserved_exceeds_remaining_by_api") is True,
                "accelerator_quota": {
                    "quota_refresh_time": str(_dict(account.get("accelerator_quota")).get("quota_refresh_time") or ""),
                    "gpu_quota": _dict(_dict(account.get("accelerator_quota")).get("gpu_quota")),
                    "tpu_quota": _dict(_dict(account.get("accelerator_quota")).get("tpu_quota")),
                },
                "weekly_gpu_quota_available_inferred": account.get("weekly_gpu_quota_available_inferred") is True,
            }
            for account in accounts
        ],
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "token_values_public": report.get("token_values_public") is True,
    }


def summarize_decode_adapter_gap(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_decode_adapter_gap_summary_v1",
            "present": False,
            "decode_adapter_ready": False,
            "same_request_decode_ready": False,
            "stage_runtime_provider_coverage": [],
            "stage_decode_provider_coverage": [],
            "missing_capabilities": [],
            "blockers": [],
            "public_artifact_safe": True,
        }
    capabilities = [item for item in _list(report.get("required_capabilities")) if isinstance(item, dict)]
    missing = [
        str(item.get("capability") or "")
        for item in capabilities
        if item.get("required") is True and item.get("verified") is not True
    ]
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("decode_adapter_ready") is not True:
        blockers.append("glm52_full_decode_adapter_not_ready")
    return {
        "schema": "glm52_decode_adapter_gap_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "decode_adapter_ready": report.get("decode_adapter_ready") is True,
        "same_request_decode_ready": report.get("same_request_decode_ready") is True,
        "model_id": str(_dict(report.get("model")).get("model_id") or ""),
        "model_type": str(_dict(report.get("model")).get("model_type") or ""),
        "num_hidden_layers": _int(_dict(report.get("model")).get("num_hidden_layers")),
        "n_routed_experts": _int(_dict(report.get("model")).get("n_routed_experts")),
        "num_experts_per_tok": _int(_dict(report.get("model")).get("num_experts_per_tok")),
        "weight_key_count": _int(_dict(report.get("model")).get("weight_key_count")),
        "required_capability_count": len(capabilities),
        "missing_capabilities": sorted(item for item in missing if item),
        "stage_runtime_provider_coverage": [str(item) for item in _list(report.get("stage_runtime_provider_coverage"))],
        "stage_decode_provider_coverage": [str(item) for item in _list(report.get("stage_decode_provider_coverage"))],
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_transformers_decode_preflight(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_transformers_decode_preflight_summary_v1",
            "present": False,
            "adapter_foundation_ready": False,
            "decode_adapter_ready": False,
            "pack_quantized_runtime_ready": False,
            "stage_weight_mapping_ready": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    model = _dict(report.get("model"))
    mapping = _dict(report.get("stage_weight_mapping"))
    pack_runtime = _dict(report.get("pack_quantized_runtime"))
    transformer = _dict(report.get("transformers_runtime"))
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("adapter_foundation_ready") is True and report.get("decode_adapter_ready") is not True:
        blockers.append("glm52_full_decode_adapter_not_ready")
    return {
        "schema": "glm52_transformers_decode_preflight_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "adapter_foundation_ready": report.get("adapter_foundation_ready") is True,
        "decode_adapter_ready": report.get("decode_adapter_ready") is True,
        "model_id": str(model.get("model_id") or ""),
        "model_type": str(model.get("model_type") or ""),
        "normalization_action": str(model.get("normalization_action") or ""),
        "quantization_format": str(model.get("quantization_format") or ""),
        "quantization_weight_bits": [int(item) for item in _list(model.get("quantization_weight_bits")) if str(item).isdigit()],
        "transformers_version": str(transformer.get("transformers_version") or ""),
        "tiny_forward_ready": transformer.get("tiny_forward_ready") is True,
        "awq_config_normalized_ready": transformer.get("awq_config_normalized_ready") is True,
        "pack_quantized_runtime_ready": pack_runtime.get("ready") is True,
        "stage_weight_mapping_ready": mapping.get("stage_weight_mapping_ready") is True,
        "selected_layer_count": _int(mapping.get("selected_layer_count")),
        "dense_layer_count": _int(mapping.get("dense_layer_count")),
        "sparse_layer_count": _int(mapping.get("sparse_layer_count")),
        "full_indexer_layer_count": _int(mapping.get("full_indexer_layer_count")),
        "shared_indexer_layer_count": _int(mapping.get("shared_indexer_layer_count")),
        "required_key_count": _int(mapping.get("required_key_count")),
        "pack_required_key_count": _int(mapping.get("pack_required_key_count")),
        "missing_required_key_count": _int(mapping.get("missing_required_key_count")),
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_pack_quantized_dequant(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_pack_quantized_dequant_summary_v1",
            "present": False,
            "pack_quantized_dequant_verified": False,
            "pack_quantized_linear_slice_verified": False,
            "stage_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("pack_quantized_dequant_verified") is True:
        blockers.append("glm52_pack_quantized_dequant_slice_is_not_full_layer")
    if report.get("pack_quantized_linear_slice_verified") is True:
        blockers.append("glm52_pack_quantized_linear_slice_is_not_stage_decode")
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    return {
        "schema": "glm52_pack_quantized_dequant_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "quantization_format": str(report.get("quantization_format") or ""),
        "layer_id": _int(report.get("layer_id")),
        "expert_id": _int(report.get("expert_id")),
        "projection": str(report.get("projection") or ""),
        "row_count": _int(report.get("row_count")),
        "group_count": _int(report.get("group_count")),
        "pack_quantized_group_loaded": report.get("pack_quantized_group_loaded") is True,
        "pack_quantized_dequant_verified": report.get("pack_quantized_dequant_verified") is True,
        "pack_quantized_linear_slice_verified": report.get("pack_quantized_linear_slice_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "q_unpacked_hash_present": _hash_ok(report.get("q_unpacked_hash")),
        "zero_point_unpacked_hash_present": _hash_ok(report.get("zero_point_unpacked_hash")),
        "dequant_slice_shape": _list(report.get("dequant_slice_shape")),
        "dequant_slice_hash_present": _hash_ok(report.get("dequant_slice_hash")),
        "linear_slice_shape": _list(report.get("linear_slice_shape")),
        "linear_slice_hash_present": _hash_ok(report.get("linear_slice_hash")),
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_attention_projection(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_attention_projection_summary_v1",
            "present": False,
            "attention_projection_verified": False,
            "stage_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("attention_projection_verified") is True:
        blockers.extend(
            [
                "glm52_attention_projection_is_not_rope_attention",
                "glm52_attention_projection_is_not_o_proj",
                "glm52_attention_projection_is_not_stage_decode",
                "glm52_attention_projection_missing_attention_scores",
                "glm52_attention_projection_missing_kv_cache_update",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    return {
        "schema": "glm52_attention_projection_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "hidden_size": _int(report.get("hidden_size")),
        "num_attention_heads": _int(report.get("num_attention_heads")),
        "q_lora_rank": _int(report.get("q_lora_rank")),
        "kv_lora_rank": _int(report.get("kv_lora_rank")),
        "qk_head_dim": _int(report.get("qk_head_dim")),
        "qk_nope_head_dim": _int(report.get("qk_nope_head_dim")),
        "qk_rope_head_dim": _int(report.get("qk_rope_head_dim")),
        "v_head_dim": _int(report.get("v_head_dim")),
        "input_layernorm_verified": report.get("input_layernorm_verified") is True,
        "q_lora_projection_verified": report.get("q_lora_projection_verified") is True,
        "kv_lora_projection_verified": report.get("kv_lora_projection_verified") is True,
        "attention_projection_verified": report.get("attention_projection_verified") is True,
        "input_norm_hash_present": _hash_ok(report.get("input_norm_hash")),
        "q_a_output_shape": _list(report.get("q_a_output_shape")),
        "q_b_output_shape": _list(report.get("q_b_output_shape")),
        "query_shape": _list(report.get("query_shape")),
        "q_nope_shape": _list(report.get("q_nope_shape")),
        "q_pe_shape": _list(report.get("q_pe_shape")),
        "kv_a_output_shape": _list(report.get("kv_a_output_shape")),
        "kv_b_output_shape": _list(report.get("kv_b_output_shape")),
        "k_nope_shape": _list(report.get("k_nope_shape")),
        "value_shape": _list(report.get("value_shape")),
        "q_a_output_hash_present": _hash_ok(report.get("q_a_output_hash")),
        "q_b_output_hash_present": _hash_ok(report.get("q_b_output_hash")),
        "kv_a_output_hash_present": _hash_ok(report.get("kv_a_output_hash")),
        "kv_b_output_hash_present": _hash_ok(report.get("kv_b_output_hash")),
        "k_nope_hash_present": _hash_ok(report.get("k_nope_hash")),
        "value_hash_present": _hash_ok(report.get("value_hash")),
        "rope_applied": report.get("rope_applied") is True,
        "attention_scores_verified": report.get("attention_scores_verified") is True,
        "o_proj_verified": report.get("o_proj_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_attention_single_token(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_attention_single_token_summary_v1",
            "present": False,
            "single_token_attention_verified": False,
            "stage_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("single_token_attention_verified") is True:
        blockers.extend(
            [
                "glm52_attention_single_token_is_not_multi_token_prefill",
                "glm52_attention_single_token_is_not_dsa_indexer",
                "glm52_attention_single_token_is_not_kv_cache_decode",
                "glm52_attention_single_token_is_not_transformer_block",
                "glm52_attention_single_token_is_not_stage_decode",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    return {
        "schema": "glm52_attention_single_token_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "hidden_size": _int(report.get("hidden_size")),
        "num_attention_heads": _int(report.get("num_attention_heads")),
        "qk_head_dim": _int(report.get("qk_head_dim")),
        "v_head_dim": _int(report.get("v_head_dim")),
        "position_id": _int(report.get("position_id")),
        "rope_applied": report.get("rope_applied") is True,
        "attention_scores_verified": report.get("attention_scores_verified") is True,
        "attention_weights_verified": report.get("attention_weights_verified") is True,
        "o_proj_verified": report.get("o_proj_verified") is True,
        "single_token_attention_verified": report.get("single_token_attention_verified") is True,
        "kv_cache_updated": report.get("kv_cache_updated") is True,
        "dsa_indexer_verified": report.get("dsa_indexer_verified") is True,
        "query_states_shape": _list(report.get("query_states_shape")),
        "key_states_shape": _list(report.get("key_states_shape")),
        "value_states_shape": _list(report.get("value_states_shape")),
        "attention_scores_shape": _list(report.get("attention_scores_shape")),
        "attention_weights_shape": _list(report.get("attention_weights_shape")),
        "head_output_shape": _list(report.get("head_output_shape")),
        "attention_flattened_shape": _list(report.get("attention_flattened_shape")),
        "o_proj_weight_shape": _list(report.get("o_proj_weight_shape")),
        "o_proj_output_shape": _list(report.get("o_proj_output_shape")),
        "query_states_hash_present": _hash_ok(report.get("query_states_hash")),
        "key_states_hash_present": _hash_ok(report.get("key_states_hash")),
        "value_states_hash_present": _hash_ok(report.get("value_states_hash")),
        "attention_scores_hash_present": _hash_ok(report.get("attention_scores_hash")),
        "attention_weights_hash_present": _hash_ok(report.get("attention_weights_hash")),
        "head_output_hash_present": _hash_ok(report.get("head_output_hash")),
        "o_proj_output_hash_present": _hash_ok(report.get("o_proj_output_hash")),
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_kv_cache_decode(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_kv_cache_decode_summary_v1",
            "present": False,
            "kv_cache_prefill_verified": False,
            "kv_cache_update_verified": False,
            "kv_cache_decode_attention_verified": False,
            "o_proj_verified": False,
            "stage_decode_verified": False,
            "generated_token_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("kv_cache_decode_attention_verified") is True:
        blockers.extend(
            [
                "glm52_kv_cache_decode_is_not_dsa_masked_attention",
                "glm52_kv_cache_decode_is_not_transformer_block",
                "glm52_kv_cache_decode_is_not_stage_decode",
                "glm52_kv_cache_decode_missing_mlp_residual",
                "glm52_kv_cache_decode_missing_lm_head",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    return {
        "schema": "glm52_kv_cache_decode_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "hidden_size": _int(report.get("hidden_size")),
        "num_attention_heads": _int(report.get("num_attention_heads")),
        "qk_head_dim": _int(report.get("qk_head_dim")),
        "v_head_dim": _int(report.get("v_head_dim")),
        "prefill_length": _int(report.get("prefill_length")),
        "decode_length": _int(report.get("decode_length")),
        "updated_cache_length": _int(report.get("updated_cache_length")),
        "prefill_key_cache_shape": _list(report.get("prefill_key_cache_shape")),
        "prefill_value_cache_shape": _list(report.get("prefill_value_cache_shape")),
        "updated_key_cache_shape": _list(report.get("updated_key_cache_shape")),
        "updated_value_cache_shape": _list(report.get("updated_value_cache_shape")),
        "decode_query_shape": _list(report.get("decode_query_shape")),
        "attention_scores_shape": _list(report.get("attention_scores_shape")),
        "attention_weights_shape": _list(report.get("attention_weights_shape")),
        "head_output_shape": _list(report.get("head_output_shape")),
        "attention_flattened_shape": _list(report.get("attention_flattened_shape")),
        "o_proj_weight_shape": _list(report.get("o_proj_weight_shape")),
        "o_proj_output_shape": _list(report.get("o_proj_output_shape")),
        "prefill_key_cache_hash_present": _hash_ok(report.get("prefill_key_cache_hash")),
        "prefill_value_cache_hash_present": _hash_ok(report.get("prefill_value_cache_hash")),
        "updated_key_cache_hash_present": _hash_ok(report.get("updated_key_cache_hash")),
        "updated_value_cache_hash_present": _hash_ok(report.get("updated_value_cache_hash")),
        "decode_query_hash_present": _hash_ok(report.get("decode_query_hash")),
        "attention_scores_hash_present": _hash_ok(report.get("attention_scores_hash")),
        "attention_weights_hash_present": _hash_ok(report.get("attention_weights_hash")),
        "head_output_hash_present": _hash_ok(report.get("head_output_hash")),
        "o_proj_output_hash_present": _hash_ok(report.get("o_proj_output_hash")),
        "kv_cache_prefill_verified": report.get("kv_cache_prefill_verified") is True,
        "kv_cache_update_verified": report.get("kv_cache_update_verified") is True,
        "kv_cache_decode_attention_verified": report.get("kv_cache_decode_attention_verified") is True,
        "o_proj_verified": report.get("o_proj_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_layer_decode(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_layer_decode_summary_v1",
            "present": False,
            "layer_decode_verified": False,
            "stage_decode_verified": False,
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("layer_decode_verified") is True:
        blockers.extend(
            [
                "glm52_layer_decode_is_single_layer_only",
                "glm52_layer_decode_uses_basic_attention_not_dsa_masked_attention",
                "glm52_layer_decode_missing_lm_head",
                "glm52_layer_decode_is_not_stage_decode",
                "glm52_layer_decode_is_not_same_request",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    if report.get("same_request_decode_verified") is not True:
        blockers.append("glm52_same_request_decode_not_verified")
    return {
        "schema": "glm52_layer_decode_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "hidden_size": _int(report.get("hidden_size")),
        "num_attention_heads": _int(report.get("num_attention_heads")),
        "qk_head_dim": _int(report.get("qk_head_dim")),
        "v_head_dim": _int(report.get("v_head_dim")),
        "num_experts_per_tok": _int(report.get("num_experts_per_tok")),
        "prefill_length": _int(report.get("prefill_length")),
        "updated_cache_length": _int(report.get("updated_cache_length")),
        "attention_output_shape": _list(report.get("attention_output_shape")),
        "attention_residual_shape": _list(report.get("attention_residual_shape")),
        "post_attention_norm_shape": _list(report.get("post_attention_norm_shape")),
        "routed_output_shape": _list(report.get("routed_output_shape")),
        "shared_output_shape": _list(report.get("shared_output_shape")),
        "full_moe_output_shape": _list(report.get("full_moe_output_shape")),
        "layer_output_shape": _list(report.get("layer_output_shape")),
        "attention_output_hash_present": _hash_ok(report.get("attention_output_hash")),
        "attention_residual_hash_present": _hash_ok(report.get("attention_residual_hash")),
        "post_attention_norm_hash_present": _hash_ok(report.get("post_attention_norm_hash")),
        "full_moe_output_hash_present": _hash_ok(report.get("full_moe_output_hash")),
        "layer_output_hash_present": _hash_ok(report.get("layer_output_hash")),
        "executed_expert_count": _int(report.get("executed_expert_count")),
        "kv_cache_prefill_verified": report.get("kv_cache_prefill_verified") is True,
        "kv_cache_update_verified": report.get("kv_cache_update_verified") is True,
        "attention_decode_verified": report.get("attention_decode_verified") is True,
        "attention_residual_verified": report.get("attention_residual_verified") is True,
        "post_attention_norm_verified": report.get("post_attention_norm_verified") is True,
        "router_topk_verified": report.get("router_topk_verified") is True,
        "routed_expert_gather_verified": report.get("routed_expert_gather_verified") is True,
        "shared_experts_mlp_verified": report.get("shared_experts_mlp_verified") is True,
        "full_moe_mlp_verified": report.get("full_moe_mlp_verified") is True,
        "layer_decode_verified": report.get("layer_decode_verified") is True,
        "dsa_masked_attention_integrated": report.get("dsa_masked_attention_integrated") is True,
        "multi_layer_stage_runtime_verified": report.get("multi_layer_stage_runtime_verified") is True,
        "lm_head_verified": report.get("lm_head_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_lm_head_token(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_lm_head_token_summary_v1",
            "present": False,
            "lm_head_logits_token_selection_verified": False,
            "generated_token_verified": False,
            "stage_decode_verified": False,
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("lm_head_logits_token_selection_verified") is True:
        blockers.extend(
            [
                "glm52_lm_head_token_selection_uses_probe_hidden_not_full_model_hidden",
                "glm52_lm_head_token_selection_is_not_stage_decode",
                "glm52_lm_head_token_selection_is_not_same_request",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    if report.get("same_request_decode_verified") is not True:
        blockers.append("glm52_same_request_decode_not_verified")
    return {
        "schema": "glm52_lm_head_token_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "hidden_size": _int(report.get("hidden_size")),
        "vocab_size": _int(report.get("vocab_size")),
        "tie_word_embeddings": report.get("tie_word_embeddings") is True,
        "hidden_source": str(report.get("hidden_source") or ""),
        "norm_weight_shape": _list(report.get("norm_weight_shape")),
        "hidden_shape": _list(report.get("hidden_shape")),
        "normalized_hidden_shape": _list(report.get("normalized_hidden_shape")),
        "lm_head_shape": _list(report.get("lm_head_shape")),
        "lm_head_dtype": str(report.get("lm_head_dtype") or ""),
        "lm_head_nbytes": _int(report.get("lm_head_nbytes")),
        "lm_head_file_count": _int(report.get("lm_head_file_count")),
        "lm_head_rows_scanned": _int(report.get("lm_head_rows_scanned")),
        "lm_head_block_count": _int(report.get("lm_head_block_count")),
        "lm_head_row_block_size": _int(report.get("lm_head_row_block_size")),
        "top_k": _int(report.get("top_k")),
        "top_k_count": _int(report.get("top_k_count")),
        "hidden_hash_present": _hash_ok(report.get("hidden_hash")),
        "normalized_hidden_hash_present": _hash_ok(report.get("normalized_hidden_hash")),
        "selected_token_id_hash_present": _hash_ok(report.get("selected_token_id_hash")),
        "selected_logit_hash_present": _hash_ok(report.get("selected_logit_hash")),
        "top_token_ids_hash_present": _hash_ok(report.get("top_token_ids_hash")),
        "top_logits_hash_present": _hash_ok(report.get("top_logits_hash")),
        "final_norm_verified": report.get("final_norm_verified") is True,
        "lm_head_streamed_full_vocab": report.get("lm_head_streamed_full_vocab") is True,
        "lm_head_logits_token_selection_verified": report.get("lm_head_logits_token_selection_verified") is True,
        "selected_token_hash_verified": report.get("selected_token_hash_verified") is True,
        "full_model_hidden_verified": report.get("full_model_hidden_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_dsa_masked_layer_decode(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_dsa_masked_layer_decode_summary_v1",
            "present": False,
            "dsa_masked_attention_integrated": False,
            "layer_decode_verified": False,
            "stage_decode_verified": False,
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("dsa_masked_attention_integrated") is True:
        blockers.extend(
            [
                "glm52_dsa_masked_layer_decode_is_single_layer_only",
                "glm52_dsa_masked_layer_decode_uses_small_sequence_topk_cap",
                "glm52_dsa_masked_layer_decode_missing_lm_head",
                "glm52_dsa_masked_layer_decode_is_not_stage_decode",
                "glm52_dsa_masked_layer_decode_is_not_same_request",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    if report.get("same_request_decode_verified") is not True:
        blockers.append("glm52_same_request_decode_not_verified")
    return {
        "schema": "glm52_dsa_masked_layer_decode_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "hidden_size": _int(report.get("hidden_size")),
        "num_attention_heads": _int(report.get("num_attention_heads")),
        "qk_head_dim": _int(report.get("qk_head_dim")),
        "v_head_dim": _int(report.get("v_head_dim")),
        "num_experts_per_tok": _int(report.get("num_experts_per_tok")),
        "prefill_length": _int(report.get("prefill_length")),
        "updated_cache_length": _int(report.get("updated_cache_length")),
        "dsa_indexer_type": str(report.get("dsa_indexer_type") or ""),
        "dsa_index_n_heads": _int(report.get("dsa_index_n_heads")),
        "dsa_index_head_dim": _int(report.get("dsa_index_head_dim")),
        "dsa_index_topk_config": _int(report.get("dsa_index_topk_config")),
        "dsa_mask_topk_count": _int(report.get("dsa_mask_topk_count")),
        "dsa_mask_pruned_position_count": _int(report.get("dsa_mask_pruned_position_count")),
        "dsa_index_score_shape": _list(report.get("dsa_index_score_shape")),
        "dsa_attention_mask_shape": _list(report.get("dsa_attention_mask_shape")),
        "attention_scores_shape": _list(report.get("attention_scores_shape")),
        "attention_output_shape": _list(report.get("attention_output_shape")),
        "attention_residual_shape": _list(report.get("attention_residual_shape")),
        "post_attention_norm_shape": _list(report.get("post_attention_norm_shape")),
        "full_moe_output_shape": _list(report.get("full_moe_output_shape")),
        "layer_output_shape": _list(report.get("layer_output_shape")),
        "dsa_index_score_hash_present": _hash_ok(report.get("dsa_index_score_hash")),
        "dsa_topk_indices_hash_present": _hash_ok(report.get("dsa_topk_indices_hash")),
        "dsa_attention_mask_hash_present": _hash_ok(report.get("dsa_attention_mask_hash")),
        "attention_scores_hash_present": _hash_ok(report.get("attention_scores_hash")),
        "attention_output_hash_present": _hash_ok(report.get("attention_output_hash")),
        "layer_output_hash_present": _hash_ok(report.get("layer_output_hash")),
        "executed_expert_count": _int(report.get("executed_expert_count")),
        "dsa_indexer_verified": report.get("dsa_indexer_verified") is True,
        "dsa_mask_verified": report.get("dsa_mask_verified") is True,
        "dsa_mask_pruned_positions_verified": report.get("dsa_mask_pruned_positions_verified") is True,
        "kv_cache_prefill_verified": report.get("kv_cache_prefill_verified") is True,
        "kv_cache_update_verified": report.get("kv_cache_update_verified") is True,
        "attention_decode_verified": report.get("attention_decode_verified") is True,
        "dsa_masked_attention_integrated": report.get("dsa_masked_attention_integrated") is True,
        "attention_residual_verified": report.get("attention_residual_verified") is True,
        "post_attention_norm_verified": report.get("post_attention_norm_verified") is True,
        "full_moe_mlp_verified": report.get("full_moe_mlp_verified") is True,
        "layer_decode_verified": report.get("layer_decode_verified") is True,
        "full_dsa_topk_scale_verified": report.get("full_dsa_topk_scale_verified") is True,
        "lm_head_verified": report.get("lm_head_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_stage_hidden_lm_head(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_stage_hidden_lm_head_summary_v1",
            "present": False,
            "stage_hidden_lm_head_token_selection_verified": False,
            "generated_token_verified": False,
            "stage_decode_verified": False,
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("stage_hidden_lm_head_token_selection_verified") is True:
        blockers.extend(
            [
                "glm52_stage_hidden_lm_head_is_single_layer_only",
                "glm52_stage_hidden_lm_head_uses_small_sequence_topk_cap",
                "glm52_stage_hidden_lm_head_is_not_full_model_hidden",
                "glm52_stage_hidden_lm_head_is_not_stage_decode",
                "glm52_stage_hidden_lm_head_is_not_same_request",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    if report.get("same_request_decode_verified") is not True:
        blockers.append("glm52_same_request_decode_not_verified")
    return {
        "schema": "glm52_stage_hidden_lm_head_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "hidden_size": _int(report.get("hidden_size")),
        "vocab_size": _int(report.get("vocab_size")),
        "stage_hidden_source": str(report.get("stage_hidden_source") or ""),
        "stage_layer_id": _int(report.get("stage_layer_id")),
        "stage_prefill_length": _int(report.get("stage_prefill_length")),
        "stage_updated_cache_length": _int(report.get("stage_updated_cache_length")),
        "stage_dsa_indexer_type": str(report.get("stage_dsa_indexer_type") or ""),
        "stage_dsa_mask_topk_count": _int(report.get("stage_dsa_mask_topk_count")),
        "stage_dsa_mask_pruned_position_count": _int(report.get("stage_dsa_mask_pruned_position_count")),
        "stage_hidden_shape": _list(report.get("stage_hidden_shape")),
        "normalized_stage_hidden_shape": _list(report.get("normalized_stage_hidden_shape")),
        "lm_head_shape": _list(report.get("lm_head_shape")),
        "lm_head_dtype": str(report.get("lm_head_dtype") or ""),
        "lm_head_rows_scanned": _int(report.get("lm_head_rows_scanned")),
        "lm_head_block_count": _int(report.get("lm_head_block_count")),
        "top_k": _int(report.get("top_k")),
        "top_k_count": _int(report.get("top_k_count")),
        "stage_hidden_hash_present": _hash_ok(report.get("stage_hidden_hash")),
        "normalized_stage_hidden_hash_present": _hash_ok(report.get("normalized_stage_hidden_hash")),
        "selected_token_id_hash_present": _hash_ok(report.get("selected_token_id_hash")),
        "selected_logit_hash_present": _hash_ok(report.get("selected_logit_hash")),
        "top_token_ids_hash_present": _hash_ok(report.get("top_token_ids_hash")),
        "top_logits_hash_present": _hash_ok(report.get("top_logits_hash")),
        "stage_dsa_masked_attention_integrated": report.get("stage_dsa_masked_attention_integrated") is True,
        "stage_layer_decode_verified": report.get("stage_layer_decode_verified") is True,
        "stage_hidden_to_lm_head_verified": report.get("stage_hidden_to_lm_head_verified") is True,
        "lm_head_streamed_full_vocab": report.get("lm_head_streamed_full_vocab") is True,
        "stage_hidden_lm_head_token_selection_verified": report.get("stage_hidden_lm_head_token_selection_verified") is True,
        "partial_layer_token_hash_verified": report.get("partial_layer_token_hash_verified") is True,
        "full_model_hidden_verified": report.get("full_model_hidden_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_multi_layer_stage_decode(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_multi_layer_stage_decode_summary_v1",
            "present": False,
            "multi_layer_stage_hidden_verified": False,
            "stage_hidden_lm_head_token_selection_verified": False,
            "generated_token_verified": False,
            "stage_decode_verified": False,
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("multi_layer_stage_hidden_verified") is True:
        blockers.extend(
            [
                "glm52_multi_layer_stage_decode_uses_decode_token_chain_only",
                "glm52_multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs",
                "glm52_multi_layer_stage_decode_is_not_full_model_hidden",
                "glm52_multi_layer_stage_decode_is_not_kaggle_runtime",
                "glm52_multi_layer_stage_decode_is_not_same_request",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    if report.get("same_request_decode_verified") is not True:
        blockers.append("glm52_same_request_decode_not_verified")
    return {
        "schema": "glm52_multi_layer_stage_decode_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "hidden_size": _int(report.get("hidden_size")),
        "vocab_size": _int(report.get("vocab_size")),
        "stage_hidden_source": str(report.get("stage_hidden_source") or ""),
        "stage_layer_range": _list(report.get("stage_layer_range")),
        "stage_layer_count": _int(report.get("stage_layer_count")),
        "executed_layer_count": _int(report.get("executed_layer_count")),
        "stage_prefill_length": _int(report.get("stage_prefill_length")),
        "stage_updated_cache_length": _int(report.get("stage_updated_cache_length")),
        "decode_token_chain_only": report.get("decode_token_chain_only") is True,
        "prefill_hidden_carrier_full_layer_outputs_verified": report.get("prefill_hidden_carrier_full_layer_outputs_verified") is True,
        "layer_summary_count": len([item for item in _list(report.get("layer_summaries")) if isinstance(item, dict)]),
        "all_layers_dsa_masked_attention_integrated": report.get("all_layers_dsa_masked_attention_integrated") is True,
        "all_layers_moe_mlp_verified": report.get("all_layers_moe_mlp_verified") is True,
        "all_layer_outputs_chained": report.get("all_layer_outputs_chained") is True,
        "stage_hidden_shape": _list(report.get("stage_hidden_shape")),
        "normalized_stage_hidden_shape": _list(report.get("normalized_stage_hidden_shape")),
        "lm_head_shape": _list(report.get("lm_head_shape")),
        "lm_head_dtype": str(report.get("lm_head_dtype") or ""),
        "lm_head_rows_scanned": _int(report.get("lm_head_rows_scanned")),
        "lm_head_block_count": _int(report.get("lm_head_block_count")),
        "top_k": _int(report.get("top_k")),
        "top_k_count": _int(report.get("top_k_count")),
        "initial_decode_hidden_hash_present": _hash_ok(report.get("initial_decode_hidden_hash")),
        "stage_hidden_hash_present": _hash_ok(report.get("stage_hidden_hash")),
        "normalized_stage_hidden_hash_present": _hash_ok(report.get("normalized_stage_hidden_hash")),
        "selected_token_id_hash_present": _hash_ok(report.get("selected_token_id_hash")),
        "selected_logit_hash_present": _hash_ok(report.get("selected_logit_hash")),
        "top_token_ids_hash_present": _hash_ok(report.get("top_token_ids_hash")),
        "top_logits_hash_present": _hash_ok(report.get("top_logits_hash")),
        "multi_layer_stage_hidden_verified": report.get("multi_layer_stage_hidden_verified") is True,
        "multi_layer_decode_token_chain_verified": report.get("multi_layer_decode_token_chain_verified") is True,
        "stage_hidden_to_lm_head_verified": report.get("stage_hidden_to_lm_head_verified") is True,
        "lm_head_streamed_full_vocab": report.get("lm_head_streamed_full_vocab") is True,
        "stage_hidden_lm_head_token_selection_verified": report.get("stage_hidden_lm_head_token_selection_verified") is True,
        "partial_multi_layer_token_hash_verified": report.get("partial_multi_layer_token_hash_verified") is True,
        "full_prefill_stage_hidden_verified": report.get("full_prefill_stage_hidden_verified") is True,
        "full_model_hidden_verified": report.get("full_model_hidden_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "live_kaggle_runtime_verified": report.get("live_kaggle_runtime_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_full_prefix_stage_decode(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_full_prefix_stage_decode_summary_v1",
            "present": False,
            "full_prefix_stage_hidden_verified": False,
            "stage_hidden_lm_head_token_selection_verified": False,
            "generated_token_verified": False,
            "stage_decode_verified": False,
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("full_prefix_stage_hidden_verified") is True:
        blockers.extend(
            [
                "glm52_full_prefix_stage_decode_uses_small_sequence_probe",
                "glm52_full_prefix_stage_decode_is_not_kaggle_runtime",
                "glm52_full_prefix_stage_decode_is_not_same_request",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    if report.get("same_request_decode_verified") is not True:
        blockers.append("glm52_same_request_decode_not_verified")
    layers = [item for item in _list(report.get("layer_summaries")) if isinstance(item, dict)]
    return {
        "schema": "glm52_full_prefix_stage_decode_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "hidden_size": _int(report.get("hidden_size")),
        "vocab_size": _int(report.get("vocab_size")),
        "stage_hidden_source": str(report.get("stage_hidden_source") or ""),
        "stage_layer_range": _list(report.get("stage_layer_range")),
        "stage_layer_count": _int(report.get("stage_layer_count")),
        "executed_layer_count": _int(report.get("executed_layer_count")),
        "stage_prefill_length": _int(report.get("stage_prefill_length")),
        "stage_sequence_length": _int(report.get("stage_sequence_length")),
        "full_prefix_token_carrier_verified": report.get("full_prefix_token_carrier_verified") is True,
        "small_sequence_probe": report.get("small_sequence_probe") is True,
        "layer_summary_count": len(layers),
        "all_layers_full_prefix_verified": report.get("all_layers_full_prefix_verified") is True,
        "all_layer_outputs_chained": report.get("all_layer_outputs_chained") is True,
        "stage_hidden_sequence_shape": _list(report.get("stage_hidden_sequence_shape")),
        "stage_hidden_shape": _list(report.get("stage_hidden_shape")),
        "normalized_stage_hidden_shape": _list(report.get("normalized_stage_hidden_shape")),
        "lm_head_shape": _list(report.get("lm_head_shape")),
        "lm_head_dtype": str(report.get("lm_head_dtype") or ""),
        "lm_head_rows_scanned": _int(report.get("lm_head_rows_scanned")),
        "lm_head_block_count": _int(report.get("lm_head_block_count")),
        "top_k": _int(report.get("top_k")),
        "top_k_count": _int(report.get("top_k_count")),
        "stage_hidden_sequence_hash_present": _hash_ok(report.get("stage_hidden_sequence_hash")),
        "stage_hidden_hash_present": _hash_ok(report.get("stage_hidden_hash")),
        "normalized_stage_hidden_hash_present": _hash_ok(report.get("normalized_stage_hidden_hash")),
        "selected_token_id_hash_present": _hash_ok(report.get("selected_token_id_hash")),
        "selected_logit_hash_present": _hash_ok(report.get("selected_logit_hash")),
        "top_token_ids_hash_present": _hash_ok(report.get("top_token_ids_hash")),
        "top_logits_hash_present": _hash_ok(report.get("top_logits_hash")),
        "full_prefix_stage_hidden_verified": report.get("full_prefix_stage_hidden_verified") is True,
        "multi_layer_stage_hidden_verified": report.get("multi_layer_stage_hidden_verified") is True,
        "stage_hidden_to_lm_head_verified": report.get("stage_hidden_to_lm_head_verified") is True,
        "lm_head_streamed_full_vocab": report.get("lm_head_streamed_full_vocab") is True,
        "stage_hidden_lm_head_token_selection_verified": report.get("stage_hidden_lm_head_token_selection_verified") is True,
        "partial_full_prefix_token_hash_verified": report.get("partial_full_prefix_token_hash_verified") is True,
        "full_model_hidden_verified": report.get("full_model_hidden_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "live_kaggle_runtime_verified": report.get("live_kaggle_runtime_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_dsa_indexer(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_dsa_indexer_summary_v1",
            "present": False,
            "dsa_indexer_verified": False,
            "stage_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("dsa_indexer_verified") is True:
        blockers.extend(
            [
                "glm52_dsa_indexer_small_sequence_is_not_full_prefill",
                "glm52_dsa_indexer_is_not_kv_cache_decode",
                "glm52_dsa_indexer_is_not_attention_output",
                "glm52_dsa_indexer_is_not_stage_decode",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    return {
        "schema": "glm52_dsa_indexer_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "layer_indexer_type": str(report.get("layer_indexer_type") or ""),
        "sequence_length": _int(report.get("sequence_length")),
        "hidden_size": _int(report.get("hidden_size")),
        "q_lora_rank": _int(report.get("q_lora_rank")),
        "index_n_heads": _int(report.get("index_n_heads")),
        "index_head_dim": _int(report.get("index_head_dim")),
        "qk_rope_head_dim": _int(report.get("qk_rope_head_dim")),
        "index_topk_config": _int(report.get("index_topk_config")),
        "effective_topk": _int(report.get("effective_topk")),
        "hidden_norm_shape": _list(report.get("hidden_norm_shape")),
        "q_resid_shape": _list(report.get("q_resid_shape")),
        "indexer_query_shape": _list(report.get("indexer_query_shape")),
        "indexer_key_shape": _list(report.get("indexer_key_shape")),
        "head_weights_shape": _list(report.get("head_weights_shape")),
        "index_score_shape": _list(report.get("index_score_shape")),
        "topk_indices_shape": _list(report.get("topk_indices_shape")),
        "hidden_norm_hash_present": _hash_ok(report.get("hidden_norm_hash")),
        "q_resid_hash_present": _hash_ok(report.get("q_resid_hash")),
        "indexer_query_hash_present": _hash_ok(report.get("indexer_query_hash")),
        "indexer_key_hash_present": _hash_ok(report.get("indexer_key_hash")),
        "head_weights_hash_present": _hash_ok(report.get("head_weights_hash")),
        "index_score_hash_present": _hash_ok(report.get("index_score_hash")),
        "topk_indices_hash_present": _hash_ok(report.get("topk_indices_hash")),
        "dsa_indexer_verified": report.get("dsa_indexer_verified") is True,
        "dsa_topk_verified": report.get("dsa_topk_verified") is True,
        "indexer_cache_updated": report.get("indexer_cache_updated") is True,
        "attention_output_verified": report.get("attention_output_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_pack_quantized_expert_mlp(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_pack_quantized_expert_mlp_summary_v1",
            "present": False,
            "pack_quantized_expert_mlp_verified": False,
            "single_expert_mlp_verified": False,
            "stage_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    projections = [item for item in _list(report.get("projection_summaries")) if isinstance(item, dict)]
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("pack_quantized_expert_mlp_verified") is True:
        blockers.extend(
            [
                "glm52_pack_quantized_expert_mlp_is_single_expert_only",
                "glm52_pack_quantized_expert_mlp_is_not_attention",
                "glm52_pack_quantized_expert_mlp_is_not_topk_router",
                "glm52_pack_quantized_expert_mlp_is_not_stage_decode",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    return {
        "schema": "glm52_pack_quantized_expert_mlp_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "expert_id": _int(report.get("expert_id")),
        "hidden_size": _int(report.get("hidden_size")),
        "pack_quantized_expert_mlp_verified": report.get("pack_quantized_expert_mlp_verified") is True,
        "single_expert_mlp_verified": report.get("single_expert_mlp_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "final_output_shape": _list(report.get("final_output_shape")),
        "final_output_hash_present": _hash_ok(report.get("final_output_hash")),
        "projection_summaries": [
            {
                "projection": str(item.get("projection") or ""),
                "weight_shape": _list(item.get("weight_shape")),
                "output_shape": _list(item.get("output_shape")),
                "output_hash_present": _hash_ok(item.get("output_hash")),
                "pack_quantized_group_loaded": item.get("pack_quantized_group_loaded") is True,
            }
            for item in projections
        ],
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_pack_quantized_router_gather(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_pack_quantized_router_gather_summary_v1",
            "present": False,
            "router_topk_verified": False,
            "routed_expert_subset_verified": False,
            "stage_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    executed = [item for item in _list(report.get("executed_experts")) if isinstance(item, dict)]
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("routed_expert_subset_verified") is True:
        blockers.extend(
            [
                "glm52_pack_quantized_router_gather_is_subset_only",
                "glm52_pack_quantized_router_gather_missing_shared_experts",
                "glm52_pack_quantized_router_gather_is_not_attention",
                "glm52_pack_quantized_router_gather_is_not_stage_decode",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    return {
        "schema": "glm52_pack_quantized_router_gather_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "hidden_size": _int(report.get("hidden_size")),
        "n_routed_experts": _int(report.get("n_routed_experts")),
        "num_experts_per_tok": _int(report.get("num_experts_per_tok")),
        "router_topk_count": _int(report.get("router_topk_count")),
        "router_topk_verified": report.get("router_topk_verified") is True,
        "router_topk_indices_hash_present": _hash_ok(report.get("router_topk_indices_hash")),
        "router_topk_weights_hash_present": _hash_ok(report.get("router_topk_weights_hash")),
        "executed_expert_count": _int(report.get("executed_expert_count")),
        "requested_executed_expert_count": _int(report.get("requested_executed_expert_count")),
        "executed_experts": [
            {
                "topk_position": _int(item.get("topk_position")),
                "expert_id": _int(item.get("expert_id"), -1),
                "expert_weight_hash_present": _hash_ok(item.get("expert_weight_hash")),
                "expert_output_shape": _list(item.get("expert_output_shape")),
                "expert_output_hash_present": _hash_ok(item.get("expert_output_hash")),
            }
            for item in executed
        ],
        "routed_expert_subset_verified": report.get("routed_expert_subset_verified") is True,
        "routed_subset_output_shape": _list(report.get("routed_subset_output_shape")),
        "routed_subset_output_hash_present": _hash_ok(report.get("routed_subset_output_hash")),
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_pack_quantized_moe_mlp(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_pack_quantized_moe_mlp_summary_v1",
            "present": False,
            "pack_quantized_moe_mlp_verified": False,
            "full_moe_mlp_verified": False,
            "stage_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    executed = [item for item in _list(report.get("executed_experts")) if isinstance(item, dict)]
    shared = [item for item in _list(report.get("shared_projection_summaries")) if isinstance(item, dict)]
    blockers = [str(item) for item in _list(report.get("blockers"))]
    if report.get("full_moe_mlp_verified") is True:
        blockers.extend(
            [
                "glm52_pack_quantized_moe_mlp_is_not_attention",
                "glm52_pack_quantized_moe_mlp_is_not_transformer_block",
                "glm52_pack_quantized_moe_mlp_is_not_stage_decode",
                "glm52_pack_quantized_moe_mlp_missing_kv_cache",
                "glm52_pack_quantized_moe_mlp_missing_lm_head",
            ]
        )
    if report.get("stage_decode_verified") is not True:
        blockers.append("glm52_stage_decode_not_verified")
    return {
        "schema": "glm52_pack_quantized_moe_mlp_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "model_repo": str(report.get("model_repo") or ""),
        "model_type": str(report.get("model_type") or ""),
        "layer_id": _int(report.get("layer_id")),
        "hidden_size": _int(report.get("hidden_size")),
        "n_routed_experts": _int(report.get("n_routed_experts")),
        "num_experts_per_tok": _int(report.get("num_experts_per_tok")),
        "n_shared_experts": _int(report.get("n_shared_experts")),
        "moe_intermediate_size": _int(report.get("moe_intermediate_size")),
        "router_topk_count": _int(report.get("router_topk_count")),
        "router_topk_verified": report.get("router_topk_verified") is True,
        "router_topk_indices_hash_present": _hash_ok(report.get("router_topk_indices_hash")),
        "router_topk_weights_hash_present": _hash_ok(report.get("router_topk_weights_hash")),
        "executed_expert_count": _int(report.get("executed_expert_count")),
        "requested_executed_expert_count": _int(report.get("requested_executed_expert_count")),
        "executed_experts": [
            {
                "topk_position": _int(item.get("topk_position")),
                "expert_id": _int(item.get("expert_id"), -1),
                "expert_weight_hash_present": _hash_ok(item.get("expert_weight_hash")),
                "expert_output_shape": _list(item.get("expert_output_shape")),
                "expert_output_hash_present": _hash_ok(item.get("expert_output_hash")),
            }
            for item in executed
        ],
        "routed_expert_gather_verified": report.get("routed_expert_gather_verified") is True,
        "routed_output_shape": _list(report.get("routed_output_shape")),
        "routed_output_hash_present": _hash_ok(report.get("routed_output_hash")),
        "shared_experts_mlp_verified": report.get("shared_experts_mlp_verified") is True,
        "shared_projection_summaries": [
            {
                "projection": str(item.get("projection") or ""),
                "weight_dtype": str(item.get("weight_dtype") or ""),
                "weight_shape": _list(item.get("weight_shape")),
                "output_shape": _list(item.get("output_shape")),
                "output_hash_present": _hash_ok(item.get("output_hash")),
            }
            for item in shared
        ],
        "shared_output_shape": _list(report.get("shared_output_shape")),
        "shared_output_hash_present": _hash_ok(report.get("shared_output_hash")),
        "pack_quantized_moe_mlp_verified": report.get("pack_quantized_moe_mlp_verified") is True,
        "full_moe_mlp_verified": report.get("full_moe_mlp_verified") is True,
        "full_moe_output_shape": _list(report.get("full_moe_output_shape")),
        "full_moe_output_hash_present": _hash_ok(report.get("full_moe_output_hash")),
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_same_request(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "schema": "glm52_same_request_summary_v1",
            "present": False,
            "same_request_decode_verified": False,
            "generated_token_count": 0,
            "generated_token_hash_present": False,
            "accepted_providers": [],
            "model_id": "",
            "live_run_performed": False,
            "coordinator_request_verified": False,
            "stage_provider_coverage_verified": False,
            "cleanup_verified": False,
            "stage_report_count": 0,
            "blockers": ["glm52_same_request_report_missing"],
            "public_artifact_safe": True,
        }
    success = _dict(report.get("success"))
    same = _dict(report.get("same_request"))
    safety = _dict(report.get("safety"))
    cleanup = _dict(report.get("cleanup") or report.get("cleanup_status"))
    stages = [item for item in _list(report.get("stage_reports") or report.get("stages")) if isinstance(item, dict)]
    model_id = str(report.get("model_id") or _dict(report.get("model")).get("model_id") or success.get("model_id") or same.get("model_id") or "")
    accepted = [
        str(item)
        for item in (
            _list(success.get("accepted_providers"))
            or _list(report.get("accepted_providers"))
            or _list(report.get("accepted_stage_providers"))
            or _list(report.get("accepted_stage_backends"))
        )
    ]
    generated = _int(success.get("generated_token_count") or report.get("generated_token_count"))
    generated_hash = str(success.get("generated_token_hash") or report.get("generated_token_hash") or "")
    raw_verified = bool(
        report.get("glm52_kaggle_same_request_verified") is True
        or success.get("same_request_decode_verified") is True
        or report.get("same_request_decode_verified") is True
    )
    live_run = report.get("live_run_performed") is True
    coordinator_hash = str(same.get("coordinator_request_id_hash") or report.get("coordinator_request_id_hash") or "")
    coordinator_verified = same.get("coordinator_request_verified") is True or _hash_ok(coordinator_hash)
    stage_providers: set[str] = set()
    stage_request_hashes: set[str] = set()
    stage_blockers: list[str] = []
    for stage in stages:
        provider = str(stage.get("provider") or stage.get("backend") or stage.get("stage_provider") or "")
        stage_hash = str(stage.get("coordinator_request_id_hash") or stage.get("request_id_hash") or "")
        output_hash = str(stage.get("stage_output_hash") or stage.get("output_hash") or stage.get("activation_handoff_hash") or "")
        stage_model = str(stage.get("model_id") or _dict(stage.get("model")).get("model_id") or "")
        ready = bool(
            provider in REQUIRED_PROVIDERS
            and stage.get("stage_execution_verified") is True
            and stage.get("live_run_performed") is True
            and stage_model == MODEL_ID
            and _hash_ok(stage_hash)
            and _hash_ok(output_hash)
            and stage.get("stage_smoke_only") is not True
            and str(stage.get("schema") or "") != "glm52_awq_tpu_stage_smoke_v1"
            and stage.get("public_artifact_safe") is True
        )
        if ready:
            stage_providers.add(provider)
            stage_request_hashes.add(stage_hash)
        else:
            stage_blockers.append(f"same_request_stage_not_verified:{provider or 'missing'}")
    stage_coverage = set(REQUIRED_PROVIDERS).issubset(stage_providers)
    cleanup_verified = bool(
        (
            cleanup.get("temporary_kaggle_kernels_deleted") is True
            or cleanup.get("temporary_resources_deleted") is True
        )
        and (
            cleanup.get("temporary_private_packages_removed") is True
            or cleanup.get("private_packages_removed") is True
        )
        and cleanup.get("live_resources_left_running") is False
        and cleanup.get("public_artifact_safe") is not False
    )
    request_hashes = set(stage_request_hashes)
    if _hash_ok(coordinator_hash):
        request_hashes.add(coordinator_hash)
    single_request = len(request_hashes) == 1
    blockers = [str(item) for item in (_list(report.get("blockers")) + _list(success.get("blockers")) + _list(same.get("blockers")))]
    if model_id and model_id != MODEL_ID:
        blockers.append("same_request_model_id_not_glm52")
    if not model_id:
        blockers.append("same_request_model_id_missing")
    if generated < 1:
        blockers.append("same_request_generated_token_missing")
    if not _hash_ok(generated_hash):
        blockers.append("same_request_generated_token_hash_missing")
    if not live_run:
        blockers.append("same_request_live_run_not_performed")
    if not coordinator_verified:
        blockers.append("same_request_coordinator_request_missing")
    if not single_request:
        blockers.append("same_request_hash_not_unique")
    if not stage_coverage:
        blockers.extend(stage_blockers)
        blockers.append("same_request_stage_provider_coverage_missing")
    if not cleanup_verified:
        blockers.append("same_request_cleanup_not_verified")
    for provider in REQUIRED_PROVIDERS:
        if provider not in accepted:
            blockers.append(f"same_request_provider_missing:{provider}")
    verified = bool(
        raw_verified
        and model_id == MODEL_ID
        and generated >= 1
        and _hash_ok(generated_hash)
        and live_run
        and coordinator_verified
        and single_request
        and stage_coverage
        and cleanup_verified
        and set(REQUIRED_PROVIDERS).issubset(set(accepted))
    )
    return {
        "schema": "glm52_same_request_summary_v1",
        "present": True,
        "source_schema": str(report.get("schema") or ""),
        "same_request_decode_verified": verified,
        "generated_token_count": generated,
        "generated_token_hash_present": _hash_ok(generated_hash),
        "accepted_providers": accepted,
        "required_providers": REQUIRED_PROVIDERS,
        "model_id": model_id,
        "live_run_performed": live_run,
        "coordinator_request_verified": coordinator_verified,
        "stage_provider_coverage_verified": stage_coverage,
        "cleanup_verified": cleanup_verified,
        "stage_report_count": len(stages),
        "public_artifact_safe": report.get("public_artifact_safe") is True or safety.get("public_artifact_safe") is True,
        "blockers": sorted(set(blockers)),
    }


def determine_failure_stage(
    source: dict[str, Any],
    tpu: dict[str, Any],
    awq_header: dict[str, Any],
    tpu_stage_smoke: dict[str, Any],
    stage_worker_push_probe: dict[str, Any],
    transformers_decode_preflight: dict[str, Any],
    attention_projection: dict[str, Any],
    attention_single_token: dict[str, Any],
    kv_cache_decode: dict[str, Any],
    layer_decode: dict[str, Any],
    lm_head_token: dict[str, Any],
    dsa_masked_layer_decode: dict[str, Any],
    stage_hidden_lm_head: dict[str, Any],
    multi_layer_stage_decode: dict[str, Any],
    full_prefix_stage_decode: dict[str, Any],
    dsa_indexer: dict[str, Any],
    pack_quantized_dequant: dict[str, Any],
    pack_quantized_expert_mlp: dict[str, Any],
    pack_quantized_router_gather: dict[str, Any],
    pack_quantized_moe_mlp: dict[str, Any],
    decode_adapter_gap: dict[str, Any],
    same_request: dict[str, Any],
) -> str:
    if source.get("resolver_ready") is not True:
        return "glm52_source_not_ready"
    if awq_header.get("present") and awq_header.get("stage_header_ready") is not True:
        return "glm52_awq_stage_header_not_ready"
    if tpu_stage_smoke.get("present") and tpu_stage_smoke.get("stage_runtime_adapter_smoke_ready") is not True:
        return "glm52_awq_tpu_stage_smoke_not_ready"
    stage_runtime_reports_verified = stage_worker_push_probe.get("required_stage_runtime_reports_verified") is True
    if (
        stage_runtime_reports_verified is not True
        and (
            source.get("stage_runtime_adapter_verified") is not True
            or awq_header.get("stage_runtime_adapter_verified") is not True
        )
    ):
        return "glm52_stage_runtime_adapter_not_verified"
    if tpu.get("tpu_runtime_ready") is not True:
        return "kaggle_tpu_runtime_not_ready"
    if (
        transformers_decode_preflight.get("present")
        and transformers_decode_preflight.get("adapter_foundation_ready") is not True
    ):
        return "glm52_transformers_decode_adapter_foundation_not_ready"
    if (
        attention_projection.get("present")
        and attention_projection.get("attention_projection_verified") is not True
    ):
        return "glm52_attention_projection_not_ready"
    if (
        attention_single_token.get("present")
        and attention_single_token.get("single_token_attention_verified") is not True
    ):
        return "glm52_attention_single_token_not_ready"
    if (
        kv_cache_decode.get("present")
        and kv_cache_decode.get("kv_cache_decode_attention_verified") is not True
    ):
        return "glm52_kv_cache_decode_not_ready"
    if (
        layer_decode.get("present")
        and layer_decode.get("layer_decode_verified") is not True
    ):
        return "glm52_layer_decode_not_ready"
    if (
        lm_head_token.get("present")
        and lm_head_token.get("lm_head_logits_token_selection_verified") is not True
    ):
        return "glm52_lm_head_token_selection_not_ready"
    if (
        dsa_masked_layer_decode.get("present")
        and dsa_masked_layer_decode.get("dsa_masked_attention_integrated") is not True
    ):
        return "glm52_dsa_masked_layer_decode_not_ready"
    if (
        stage_hidden_lm_head.get("present")
        and stage_hidden_lm_head.get("stage_hidden_lm_head_token_selection_verified") is not True
    ):
        return "glm52_stage_hidden_lm_head_not_ready"
    if (
        multi_layer_stage_decode.get("present")
        and multi_layer_stage_decode.get("multi_layer_stage_hidden_verified") is not True
    ):
        return "glm52_multi_layer_stage_decode_not_ready"
    if (
        full_prefix_stage_decode.get("present")
        and full_prefix_stage_decode.get("full_prefix_stage_hidden_verified") is not True
    ):
        return "glm52_full_prefix_stage_decode_not_ready"
    if dsa_indexer.get("present") and dsa_indexer.get("dsa_indexer_verified") is not True:
        return "glm52_dsa_indexer_not_ready"
    if pack_quantized_dequant.get("present") and pack_quantized_dequant.get("pack_quantized_dequant_verified") is not True:
        return "glm52_pack_quantized_dequant_not_ready"
    if (
        pack_quantized_expert_mlp.get("present")
        and pack_quantized_expert_mlp.get("pack_quantized_expert_mlp_verified") is not True
    ):
        return "glm52_pack_quantized_expert_mlp_not_ready"
    if (
        pack_quantized_router_gather.get("present")
        and pack_quantized_router_gather.get("routed_expert_subset_verified") is not True
    ):
        return "glm52_pack_quantized_router_gather_not_ready"
    if (
        pack_quantized_moe_mlp.get("present")
        and pack_quantized_moe_mlp.get("full_moe_mlp_verified") is not True
    ):
        return "glm52_pack_quantized_moe_mlp_not_ready"
    if decode_adapter_gap.get("present") and decode_adapter_gap.get("decode_adapter_ready") is not True:
        return "glm52_full_decode_adapter_not_ready"
    if (
        transformers_decode_preflight.get("present")
        and transformers_decode_preflight.get("decode_adapter_ready") is not True
    ):
        return "glm52_full_decode_adapter_not_ready"
    if same_request.get("same_request_decode_verified") is not True:
        return "glm52_same_request_decode_not_verified"
    return "none"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_report = load_json(args.source_report)
    tpu_report = load_json(args.tpu_watch_report)
    awq_header_report = load_json(args.awq_stage_header_report)
    awq_stage_value_reports = load_jsons(args.awq_stage_value_report)
    tpu_stage_smoke_report = load_json(args.tpu_stage_smoke_report)
    kaggle_source_search_report = load_json(args.kaggle_source_search_report)
    stage_runtime_plan_report = load_json(args.stage_runtime_plan_report)
    stage_worker_package_report = load_json(args.stage_worker_package_report)
    stage_worker_push_probe_report = load_json(args.stage_worker_push_probe_report)
    gpu_token_quota_report = load_json(args.gpu_token_quota_report)
    transformers_decode_preflight_report = load_json(args.transformers_decode_preflight_report)
    attention_projection_report = load_json(args.attention_projection_report)
    attention_single_token_report = load_json(args.attention_single_token_report)
    kv_cache_decode_report = load_json(args.kv_cache_decode_report)
    layer_decode_report = load_json(args.layer_decode_report)
    lm_head_token_report = load_json(args.lm_head_token_report)
    dsa_masked_layer_decode_report = load_json(args.dsa_masked_layer_decode_report)
    stage_hidden_lm_head_report = load_json(args.stage_hidden_lm_head_report)
    multi_layer_stage_decode_report = load_json(args.multi_layer_stage_decode_report)
    full_prefix_stage_decode_report = load_json(args.full_prefix_stage_decode_report)
    dsa_indexer_report = load_json(args.dsa_indexer_report)
    pack_quantized_dequant_report = load_json(args.pack_quantized_dequant_report)
    pack_quantized_expert_mlp_report = load_json(args.pack_quantized_expert_mlp_report)
    pack_quantized_router_gather_report = load_json(args.pack_quantized_router_gather_report)
    pack_quantized_moe_mlp_report = load_json(args.pack_quantized_moe_mlp_report)
    decode_adapter_gap_report = load_json(args.decode_adapter_gap_report)
    same_report = load_json(args.same_request_report)

    source = summarize_source(source_report)
    tpu = summarize_tpu(tpu_report)
    awq_header = summarize_awq_stage_header(awq_header_report)
    awq_stage_value = summarize_awq_stage_value_probes(awq_stage_value_reports)
    tpu_stage_smoke = summarize_tpu_stage_smoke(tpu_stage_smoke_report)
    kaggle_source_search = summarize_kaggle_source_search(kaggle_source_search_report)
    stage_runtime_plan = summarize_stage_runtime_plan(stage_runtime_plan_report)
    stage_worker_package = summarize_stage_worker_package(stage_worker_package_report)
    stage_worker_push_probe = summarize_stage_worker_push_probe(
        stage_worker_push_probe_report,
        stage_worker_package=stage_worker_package,
    )
    gpu_token_quota = summarize_gpu_token_quota(gpu_token_quota_report)
    transformers_decode_preflight = summarize_transformers_decode_preflight(transformers_decode_preflight_report)
    attention_projection = summarize_attention_projection(attention_projection_report)
    attention_single_token = summarize_attention_single_token(attention_single_token_report)
    kv_cache_decode = summarize_kv_cache_decode(kv_cache_decode_report)
    layer_decode = summarize_layer_decode(layer_decode_report)
    lm_head_token = summarize_lm_head_token(lm_head_token_report)
    dsa_masked_layer_decode = summarize_dsa_masked_layer_decode(dsa_masked_layer_decode_report)
    stage_hidden_lm_head = summarize_stage_hidden_lm_head(stage_hidden_lm_head_report)
    multi_layer_stage_decode = summarize_multi_layer_stage_decode(multi_layer_stage_decode_report)
    full_prefix_stage_decode = summarize_full_prefix_stage_decode(full_prefix_stage_decode_report)
    dsa_indexer = summarize_dsa_indexer(dsa_indexer_report)
    pack_quantized_dequant = summarize_pack_quantized_dequant(pack_quantized_dequant_report)
    pack_quantized_expert_mlp = summarize_pack_quantized_expert_mlp(pack_quantized_expert_mlp_report)
    pack_quantized_router_gather = summarize_pack_quantized_router_gather(pack_quantized_router_gather_report)
    pack_quantized_moe_mlp = summarize_pack_quantized_moe_mlp(pack_quantized_moe_mlp_report)
    decode_adapter_gap = summarize_decode_adapter_gap(decode_adapter_gap_report)
    same = summarize_same_request(same_report)
    live_same_request_success = bool(
        same.get("same_request_decode_verified") is True
        and same.get("generated_token_count", 0) >= 1
        and same.get("generated_token_hash_present") is True
        and same.get("live_run_performed") is True
        and same.get("coordinator_request_verified") is True
        and same.get("stage_provider_coverage_verified") is True
        and same.get("cleanup_verified") is True
        and same.get("model_id") == MODEL_ID
        and set(REQUIRED_PROVIDERS).issubset(set(_list(same.get("accepted_providers"))))
    )
    success = bool(
        source.get("resolver_ready") is True
        and source.get("compatible_with_glm52") is True
        and tpu.get("tpu_runtime_ready") is True
        and live_same_request_success
    )

    blockers = set()
    for item in (source, tpu, awq_header, awq_stage_value, tpu_stage_smoke, kaggle_source_search, stage_runtime_plan, stage_worker_package, stage_worker_push_probe, gpu_token_quota, transformers_decode_preflight, attention_projection, attention_single_token, kv_cache_decode, layer_decode, lm_head_token, dsa_masked_layer_decode, stage_hidden_lm_head, multi_layer_stage_decode, full_prefix_stage_decode, dsa_indexer, pack_quantized_dequant, pack_quantized_expert_mlp, pack_quantized_router_gather, pack_quantized_moe_mlp, decode_adapter_gap, same):
        blockers.update(str(blocker) for blocker in _list(item.get("blockers")) if blocker)
    if not success:
        if awq_header.get("present") and awq_header.get("stage_header_ready") is not True:
            blockers.add("glm52_awq_stage_header_not_ready")
        if tpu_stage_smoke.get("present") and tpu_stage_smoke.get("stage_runtime_adapter_smoke_ready") is not True:
            blockers.add("glm52_awq_tpu_stage_smoke_not_ready")
        if source.get("stage_runtime_adapter_verified") is not True:
            blockers.add("glm52_stage_runtime_adapter_not_verified")
        if awq_header.get("stage_runtime_adapter_verified") is not True:
            blockers.add("glm52_awq_stage_runtime_adapter_not_verified")
        if tpu.get("tpu_runtime_ready") is not True:
            blockers.add("kaggle_tpu_runtime_not_ready")
        if (
            transformers_decode_preflight.get("present")
            and transformers_decode_preflight.get("decode_adapter_ready") is not True
        ):
            blockers.add("glm52_full_decode_adapter_not_ready")
        if (
            attention_projection.get("present")
            and attention_projection.get("attention_projection_verified") is not True
        ):
            blockers.add("glm52_attention_projection_not_ready")
        if (
            attention_single_token.get("present")
            and attention_single_token.get("single_token_attention_verified") is not True
        ):
            blockers.add("glm52_attention_single_token_not_ready")
        if (
            kv_cache_decode.get("present")
            and kv_cache_decode.get("kv_cache_decode_attention_verified") is not True
        ):
            blockers.add("glm52_kv_cache_decode_not_ready")
        if (
            layer_decode.get("present")
            and layer_decode.get("layer_decode_verified") is not True
        ):
            blockers.add("glm52_layer_decode_not_ready")
        if (
            lm_head_token.get("present")
            and lm_head_token.get("lm_head_logits_token_selection_verified") is not True
        ):
            blockers.add("glm52_lm_head_token_selection_not_ready")
        if (
            dsa_masked_layer_decode.get("present")
            and dsa_masked_layer_decode.get("dsa_masked_attention_integrated") is not True
        ):
            blockers.add("glm52_dsa_masked_layer_decode_not_ready")
        if (
            stage_hidden_lm_head.get("present")
            and stage_hidden_lm_head.get("stage_hidden_lm_head_token_selection_verified") is not True
        ):
            blockers.add("glm52_stage_hidden_lm_head_not_ready")
        if (
            multi_layer_stage_decode.get("present")
            and multi_layer_stage_decode.get("multi_layer_stage_hidden_verified") is not True
        ):
            blockers.add("glm52_multi_layer_stage_decode_not_ready")
        if (
            full_prefix_stage_decode.get("present")
            and full_prefix_stage_decode.get("full_prefix_stage_hidden_verified") is not True
        ):
            blockers.add("glm52_full_prefix_stage_decode_not_ready")
        if dsa_indexer.get("present") and dsa_indexer.get("dsa_indexer_verified") is not True:
            blockers.add("glm52_dsa_indexer_not_ready")
        if (
            pack_quantized_dequant.get("present")
            and pack_quantized_dequant.get("pack_quantized_dequant_verified") is not True
        ):
            blockers.add("glm52_pack_quantized_dequant_not_ready")
        if (
            pack_quantized_expert_mlp.get("present")
            and pack_quantized_expert_mlp.get("pack_quantized_expert_mlp_verified") is not True
        ):
            blockers.add("glm52_pack_quantized_expert_mlp_not_ready")
        if (
            pack_quantized_router_gather.get("present")
            and pack_quantized_router_gather.get("routed_expert_subset_verified") is not True
        ):
            blockers.add("glm52_pack_quantized_router_gather_not_ready")
        if (
            pack_quantized_moe_mlp.get("present")
            and pack_quantized_moe_mlp.get("full_moe_mlp_verified") is not True
        ):
            blockers.add("glm52_pack_quantized_moe_mlp_not_ready")
        if decode_adapter_gap.get("present") and decode_adapter_gap.get("decode_adapter_ready") is not True:
            blockers.add("glm52_full_decode_adapter_not_ready")
        if same.get("same_request_decode_verified") is not True:
            blockers.add("glm52_same_request_decode_not_verified")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "glm52_kaggle_accelerator_deployment_rc_ready": True,
        "goal_achieved": success,
        "model": {
            "model_id": MODEL_ID,
            "compatible_model_required": True,
            "fallback_model_allowed_for_success": False,
        },
        "source": source,
        "tpu_request": tpu,
        "awq_stage_header": awq_header,
        "awq_stage_value_probe": awq_stage_value,
        "tpu_stage_smoke": tpu_stage_smoke,
        "kaggle_source_search": kaggle_source_search,
        "stage_runtime_plan": stage_runtime_plan,
        "stage_worker_package": stage_worker_package,
        "stage_worker_push_probe": stage_worker_push_probe,
        "gpu_token_quota": gpu_token_quota,
        "transformers_decode_preflight": transformers_decode_preflight,
        "attention_projection": attention_projection,
        "attention_single_token": attention_single_token,
        "kv_cache_decode": kv_cache_decode,
        "layer_decode": layer_decode,
        "lm_head_token": lm_head_token,
        "dsa_masked_layer_decode": dsa_masked_layer_decode,
        "stage_hidden_lm_head": stage_hidden_lm_head,
        "multi_layer_stage_decode": multi_layer_stage_decode,
        "full_prefix_stage_decode": full_prefix_stage_decode,
        "dsa_indexer": dsa_indexer,
        "pack_quantized_dequant": pack_quantized_dequant,
        "pack_quantized_expert_mlp": pack_quantized_expert_mlp,
        "pack_quantized_router_gather": pack_quantized_router_gather,
        "pack_quantized_moe_mlp": pack_quantized_moe_mlp,
        "decode_adapter_gap": decode_adapter_gap,
        "same_request": same,
        "success": {
            "same_request_decode_verified": success,
            "glm52_kaggle_cpu_gpu_tpu_same_request_verified": success,
            "generated_token_count": same.get("generated_token_count", 0) if success else 0,
            "accepted_providers": same.get("accepted_providers", []) if success else [],
            "required_providers": REQUIRED_PROVIDERS,
        },
        "failure_stage": determine_failure_stage(source, tpu, awq_header, tpu_stage_smoke, stage_worker_push_probe, transformers_decode_preflight, attention_projection, attention_single_token, kv_cache_decode, layer_decode, lm_head_token, dsa_masked_layer_decode, stage_hidden_lm_head, multi_layer_stage_decode, full_prefix_stage_decode, dsa_indexer, pack_quantized_dequant, pack_quantized_expert_mlp, pack_quantized_router_gather, pack_quantized_moe_mlp, decode_adapter_gap, same),
        "blockers": [] if success else sorted(blockers),
        "diagnosis_codes": [
            "glm52_source_ready" if source.get("resolver_ready") else "glm52_source_not_ready",
            "kaggle_tpu_runtime_ready" if tpu.get("tpu_runtime_ready") else "kaggle_tpu_runtime_not_ready",
            "glm52_same_request_verified" if success else "glm52_same_request_not_verified",
        ],
        "completion_boundary": {
            "queue_evidence_is_not_success": True,
            "metadata_only_source_is_not_success": True,
            "kaggle_source_search_is_not_success": True,
            "stage_header_evidence_is_not_success": True,
            "stage_value_probe_evidence_is_not_success": True,
            "stage_smoke_evidence_is_not_success": True,
            "transformers_decode_preflight_is_not_success": True,
            "attention_projection_is_not_success": True,
            "attention_single_token_is_not_success": True,
            "kv_cache_decode_is_not_success": True,
            "layer_decode_is_not_success": True,
            "lm_head_token_selection_is_not_success": True,
            "dsa_masked_layer_decode_is_not_success": True,
            "stage_hidden_lm_head_is_not_success": True,
            "multi_layer_stage_decode_is_not_success": True,
            "full_prefix_stage_decode_is_not_success": True,
            "dsa_indexer_is_not_success": True,
            "pack_quantized_dequant_slice_is_not_success": True,
            "pack_quantized_expert_mlp_is_not_success": True,
            "pack_quantized_router_gather_subset_is_not_success": True,
            "pack_quantized_moe_mlp_is_not_success": True,
            "decode_adapter_gap_evidence_is_not_success": True,
            "single_backend_inference_is_not_success": True,
            "fallback_model_is_not_success": True,
            "requires_real_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
        "next_resume_commands": [
            "python scripts/glm52_mcp_tpu_stage_runtime_watch.py --ref tpuowner/ct-glm52-tpu-value-op-r1 --output-dir dist/glm52-mcp-tpu-stage-runtime-watch-<date>-rN --token-section tpuowner --status-polls 10 --status-poll-interval-seconds 60 --json",
            "python scripts/glm52_mcp_tpu_stage_runtime_watch_check.py --report dist/glm52-mcp-tpu-stage-runtime-watch-<date>-rN/glm52_mcp_tpu_stage_runtime_watch.json --json",
            "python scripts/glm52_mcp_tpu_stage_runtime_watch_check.py --report dist/glm52-mcp-tpu-stage-runtime-watch-<date>-rN/glm52_mcp_tpu_stage_runtime_watch.json --require-ready --json",
            "python scripts/glm52_kaggle_stage_worker_push_probe.py --mode live --output-dir dist/glm52-kaggle-stage-worker-push-probe-<date>-rN --stage-worker-package-report dist/glm52-kaggle-stage-worker-package-20260705-r15-writable-embedded-bundle-bound-request/glm52_kaggle_stage_worker_package.json --token-section tpuowner --retain-nonterminal-tpu --wait-seconds 1800 --poll-interval-seconds 60 --json",
            "python scripts/glm52_kaggle_stage_worker_push_probe_check.py --report dist/glm52-kaggle-stage-worker-push-probe-<date>-rN/glm52_kaggle_stage_worker_push_probe.json --json",
            "python scripts/glm52_kaggle_stage_worker_push_probe_check.py --report dist/glm52-kaggle-stage-worker-push-probe-<date>-rN/glm52_kaggle_stage_worker_push_probe.json --require-live --json",
            "kaggle kernels status tpuowner/ct-glm52-tpu-value-op-r1",
            "kaggle kernels status tpuowner/ct-mcp-tpu-probe-0704-r2",
            "kaggle kernels status tpuowner/ct-glm52-awq-tpu-stage-smoke-0704-r1",
            "kaggle kernels output tpuowner/ct-glm52-awq-tpu-stage-smoke-0704-r1 -p dist/glm52-kaggle-tpu-awq-stage-smoke-<date>-rN/notebook-output --force --file-pattern glm52_awq_tpu_stage_smoke.json",
            "python scripts/glm52_awq_tpu_stage_smoke_check.py --report dist/glm52-kaggle-tpu-awq-stage-smoke-<date>-rN/notebook-output/glm52_awq_tpu_stage_smoke.json --require-ready --json",
            "python scripts/glm52_model_source_resolver.py --output-dir dist/glm52-model-source-resolver-<date>-rN",
            "python scripts/glm52_kaggle_public_source_search.py --output-dir dist/glm52-kaggle-public-source-search-<date>-rN --token-section tpuowner",
            "python scripts/glm52_awq_stage_header_probe.py --output-dir dist/glm52-awq-stage-header-probe-<date>-rN --model-repo cyankiwi/GLM-5.2-AWQ-INT4 --stage-id 4 --stage-count 12",
            "python scripts/glm52_awq_stage_value_probe.py --output-dir dist/glm52-awq-stage-value-probe-<date>-rN --model-repo cyankiwi/GLM-5.2-AWQ-INT4 --stage-id 4 --stage-count 12",
            "python scripts/glm52_awq_stage_value_probe_check.py --report dist/glm52-awq-stage-value-probe-<date>-rN/glm52_awq_stage_value_probe.json --require-ready --json",
            "python scripts/glm52_kaggle_stage_runtime_plan.py --output-dir dist/glm52-kaggle-stage-runtime-plan-<date>-rN --source-report dist/glm52-model-source-resolver-<date>-rN/glm52_model_source_resolver.json --awq-stage-header-report dist/glm52-awq-stage-header-probe-<date>-rN/glm52_awq_stage_header_probe.json",
            "python scripts/glm52_kaggle_stage_worker_package.py --output-dir dist/glm52-kaggle-stage-worker-package-<date>-rN --stage-runtime-plan-report dist/glm52-kaggle-stage-runtime-plan-<date>-rN/glm52_kaggle_stage_runtime_plan.json --kaggle-owner tpuowner --runtime-kind full_prefix_stage_decode --coordinator-request-id-hash sha256:8385016dbeb99152007a34bce07e028a1ac9a564a28b5b294ca54164b49afeee",
            "python scripts/glm52_kaggle_stage_worker_push_probe.py --mode preflight --output-dir dist/glm52-kaggle-stage-worker-push-probe-<date>-rN --stage-worker-package-report dist/glm52-kaggle-stage-worker-package-<date>-rN/glm52_kaggle_stage_worker_package.json",
            "python scripts/glm52_kaggle_stage_runtime_check.py --report <kaggle-cuda-stage.json> --require-verified --json",
            "python scripts/glm52_kaggle_stage_runtime_check.py --report <kaggle-jax-tpu-stage.json> --require-verified --json",
            "python scripts/glm52_kaggle_stage_runtime_check.py --report <kaggle-cpu-stage.json> --require-verified --json",
            "python scripts/glm52_kaggle_same_request_probe.py --mode assemble --output-dir dist/glm52-kaggle-same-request-<date>-rN --stage-report <kaggle-cuda-stage.json> --stage-report <kaggle-jax-tpu-stage.json> --stage-report <kaggle-cpu-stage.json> --coordinator-report <coordinator.json> --cleanup-report <cleanup.json>",
            "python scripts/glm52_kaggle_same_request_check.py --report dist/glm52-kaggle-same-request-<date>-rN/glm52_kaggle_same_request_probe.json --require-verified --json",
            "python scripts/glm52_decode_adapter_gap_probe.py --output-dir dist/glm52-decode-adapter-gap-probe-<date>-rN --stage-report <kaggle-cuda-stage.json> --stage-report <kaggle-jax-tpu-stage.json> --stage-report <kaggle-cpu-stage.json> --same-request-report dist/glm52-kaggle-same-request-<date>-rN/glm52_kaggle_same_request_probe.json --json",
            "python scripts/glm52_decode_adapter_gap_check.py --report dist/glm52-decode-adapter-gap-probe-<date>-rN/glm52_decode_adapter_gap_probe.json --json",
            "python scripts/glm52_transformers_decode_adapter_preflight.py --output-dir dist/glm52-transformers-decode-adapter-preflight-<date>-rN --stage-id -1 --stage-count 3 --json",
            "python scripts/glm52_transformers_decode_adapter_preflight_check.py --report dist/glm52-transformers-decode-adapter-preflight-<date>-rN/glm52_transformers_decode_adapter_preflight.json --require-foundation --json",
            "python scripts/glm52_attention_projection_probe.py --output-dir dist/glm52-attention-projection-probe-<date>-rN --layer-id 3 --json",
            "python scripts/glm52_attention_projection_check.py --report dist/glm52-attention-projection-probe-<date>-rN/glm52_attention_projection_probe.json --require-verified --json",
            "python scripts/glm52_attention_single_token_probe.py --output-dir dist/glm52-attention-single-token-probe-<date>-rN --layer-id 3 --position-id 7 --json",
            "python scripts/glm52_attention_single_token_check.py --report dist/glm52-attention-single-token-probe-<date>-rN/glm52_attention_single_token_probe.json --require-verified --json",
            "python scripts/glm52_kv_cache_decode_probe.py --output-dir dist/glm52-kv-cache-decode-probe-<date>-rN --layer-id 3 --prefill-length 4 --json",
            "python scripts/glm52_kv_cache_decode_check.py --report dist/glm52-kv-cache-decode-probe-<date>-rN/glm52_kv_cache_decode_probe.json --require-verified --json",
            "python scripts/glm52_layer_decode_probe.py --output-dir dist/glm52-layer-decode-probe-<date>-rN --layer-id 3 --prefill-length 4 --executed-expert-count 8 --json",
            "python scripts/glm52_layer_decode_check.py --report dist/glm52-layer-decode-probe-<date>-rN/glm52_layer_decode_probe.json --require-verified --json",
            "python scripts/glm52_lm_head_token_probe.py --output-dir dist/glm52-lm-head-token-probe-<date>-rN --top-k 5 --row-block-size 2048 --json",
            "python scripts/glm52_lm_head_token_check.py --report dist/glm52-lm-head-token-probe-<date>-rN/glm52_lm_head_token_probe.json --require-verified --json",
            "python scripts/glm52_dsa_masked_layer_decode_probe.py --output-dir dist/glm52-dsa-masked-layer-decode-probe-<date>-rN --layer-id 6 --prefill-length 8 --dsa-mask-topk 4 --executed-expert-count 8 --json",
            "python scripts/glm52_dsa_masked_layer_decode_check.py --report dist/glm52-dsa-masked-layer-decode-probe-<date>-rN/glm52_dsa_masked_layer_decode_probe.json --require-verified --json",
            "python scripts/glm52_stage_hidden_lm_head_probe.py --output-dir dist/glm52-stage-hidden-lm-head-probe-<date>-rN --layer-id 6 --prefill-length 8 --dsa-mask-topk 4 --executed-expert-count 8 --top-k 5 --row-block-size 2048 --json",
            "python scripts/glm52_stage_hidden_lm_head_check.py --report dist/glm52-stage-hidden-lm-head-probe-<date>-rN/glm52_stage_hidden_lm_head_probe.json --require-verified --json",
            "python scripts/glm52_multi_layer_stage_decode_probe.py --output-dir dist/glm52-multi-layer-stage-decode-probe-<date>-rN --layer-start 6 --layer-end 8 --prefill-length 8 --dsa-mask-topk 4 --executed-expert-count 8 --top-k 5 --row-block-size 2048 --json",
            "python scripts/glm52_multi_layer_stage_decode_check.py --report dist/glm52-multi-layer-stage-decode-probe-<date>-rN/glm52_multi_layer_stage_decode_probe.json --require-verified --json",
            "python scripts/glm52_full_prefix_stage_decode_probe.py --output-dir dist/glm52-full-prefix-stage-decode-probe-<date>-rN --layer-start 6 --layer-end 8 --prefill-length 2 --dsa-mask-topk 2 --executed-expert-count 8 --top-k 5 --row-block-size 2048 --json",
            "python scripts/glm52_full_prefix_stage_decode_check.py --report dist/glm52-full-prefix-stage-decode-probe-<date>-rN/glm52_full_prefix_stage_decode_probe.json --require-verified --json",
            "python scripts/glm52_dsa_indexer_probe.py --output-dir dist/glm52-dsa-indexer-probe-<date>-rN --layer-id 2 --sequence-length 8 --json",
            "python scripts/glm52_dsa_indexer_check.py --report dist/glm52-dsa-indexer-probe-<date>-rN/glm52_dsa_indexer_probe.json --require-verified --json",
            "python scripts/glm52_pack_quantized_dequant_probe.py --output-dir dist/glm52-pack-quantized-dequant-probe-<date>-rN --layer-id 3 --expert-id 0 --projection gate_proj --row-count 4 --group-count 2 --json",
            "python scripts/glm52_pack_quantized_dequant_check.py --report dist/glm52-pack-quantized-dequant-probe-<date>-rN/glm52_pack_quantized_dequant_probe.json --require-verified --json",
            "python scripts/glm52_pack_quantized_expert_mlp_probe.py --output-dir dist/glm52-pack-quantized-expert-mlp-probe-<date>-rN --layer-id 3 --expert-id 0 --json",
            "python scripts/glm52_pack_quantized_expert_mlp_check.py --report dist/glm52-pack-quantized-expert-mlp-probe-<date>-rN/glm52_pack_quantized_expert_mlp_probe.json --require-verified --json",
            "python scripts/glm52_pack_quantized_router_gather_probe.py --output-dir dist/glm52-pack-quantized-router-gather-probe-<date>-rN --layer-id 3 --executed-expert-count 8 --json",
            "python scripts/glm52_pack_quantized_router_gather_check.py --report dist/glm52-pack-quantized-router-gather-probe-<date>-rN/glm52_pack_quantized_router_gather_probe.json --require-verified --json",
            "python scripts/glm52_pack_quantized_moe_mlp_probe.py --output-dir dist/glm52-pack-quantized-moe-mlp-probe-<date>-rN --layer-id 3 --executed-expert-count 8 --json",
            "python scripts/glm52_pack_quantized_moe_mlp_check.py --report dist/glm52-pack-quantized-moe-mlp-probe-<date>-rN/glm52_pack_quantized_moe_mlp_probe.json --require-verified --json",
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks

    support_path = output_dir / "glm52_kaggle_accelerator_deployment_rc_support.json"
    summary_path = output_dir / "glm52_kaggle_accelerator_deployment_rc.json"
    support = {
        "schema": SUPPORT_SCHEMA,
        "generated_at": utc_now(),
        "source_report": artifact_entry(Path(args.source_report), output_dir, kind="source_report", ok=bool(source_report)),
        "tpu_watch_report": artifact_entry(Path(args.tpu_watch_report), output_dir, kind="tpu_watch_report", ok=bool(tpu_report)),
        "awq_stage_header_report": artifact_entry(Path(args.awq_stage_header_report), output_dir, kind="awq_stage_header_report", ok=bool(awq_header_report)) if args.awq_stage_header_report else {"kind": "awq_stage_header_report", "present": False},
        "awq_stage_value_reports": [
            artifact_entry(Path(path), output_dir, kind="awq_stage_value_report", ok=bool(load_json(path)))
            for path in (args.awq_stage_value_report or [])
        ],
        "tpu_stage_smoke_report": artifact_entry(Path(args.tpu_stage_smoke_report), output_dir, kind="tpu_stage_smoke_report", ok=bool(tpu_stage_smoke_report)) if args.tpu_stage_smoke_report else {"kind": "tpu_stage_smoke_report", "present": False},
        "kaggle_source_search_report": artifact_entry(Path(args.kaggle_source_search_report), output_dir, kind="kaggle_source_search_report", ok=bool(kaggle_source_search_report)) if args.kaggle_source_search_report else {"kind": "kaggle_source_search_report", "present": False},
        "stage_runtime_plan_report": artifact_entry(Path(args.stage_runtime_plan_report), output_dir, kind="stage_runtime_plan_report", ok=bool(stage_runtime_plan_report)) if args.stage_runtime_plan_report else {"kind": "stage_runtime_plan_report", "present": False},
        "stage_worker_package_report": artifact_entry(Path(args.stage_worker_package_report), output_dir, kind="stage_worker_package_report", ok=bool(stage_worker_package_report)) if args.stage_worker_package_report else {"kind": "stage_worker_package_report", "present": False},
        "stage_worker_push_probe_report": artifact_entry(Path(args.stage_worker_push_probe_report), output_dir, kind="stage_worker_push_probe_report", ok=bool(stage_worker_push_probe_report)) if args.stage_worker_push_probe_report else {"kind": "stage_worker_push_probe_report", "present": False},
        "gpu_token_quota_report": artifact_entry(Path(args.gpu_token_quota_report), output_dir, kind="gpu_token_quota_report", ok=bool(gpu_token_quota_report)) if args.gpu_token_quota_report else {"kind": "gpu_token_quota_report", "present": False},
        "transformers_decode_preflight_report": artifact_entry(Path(args.transformers_decode_preflight_report), output_dir, kind="transformers_decode_preflight_report", ok=bool(transformers_decode_preflight_report)) if args.transformers_decode_preflight_report else {"kind": "transformers_decode_preflight_report", "present": False},
        "attention_projection_report": artifact_entry(Path(args.attention_projection_report), output_dir, kind="attention_projection_report", ok=bool(attention_projection_report)) if args.attention_projection_report else {"kind": "attention_projection_report", "present": False},
        "attention_single_token_report": artifact_entry(Path(args.attention_single_token_report), output_dir, kind="attention_single_token_report", ok=bool(attention_single_token_report)) if args.attention_single_token_report else {"kind": "attention_single_token_report", "present": False},
        "kv_cache_decode_report": artifact_entry(Path(args.kv_cache_decode_report), output_dir, kind="kv_cache_decode_report", ok=bool(kv_cache_decode_report)) if args.kv_cache_decode_report else {"kind": "kv_cache_decode_report", "present": False},
        "layer_decode_report": artifact_entry(Path(args.layer_decode_report), output_dir, kind="layer_decode_report", ok=bool(layer_decode_report)) if args.layer_decode_report else {"kind": "layer_decode_report", "present": False},
        "lm_head_token_report": artifact_entry(Path(args.lm_head_token_report), output_dir, kind="lm_head_token_report", ok=bool(lm_head_token_report)) if args.lm_head_token_report else {"kind": "lm_head_token_report", "present": False},
        "dsa_masked_layer_decode_report": artifact_entry(Path(args.dsa_masked_layer_decode_report), output_dir, kind="dsa_masked_layer_decode_report", ok=bool(dsa_masked_layer_decode_report)) if args.dsa_masked_layer_decode_report else {"kind": "dsa_masked_layer_decode_report", "present": False},
        "stage_hidden_lm_head_report": artifact_entry(Path(args.stage_hidden_lm_head_report), output_dir, kind="stage_hidden_lm_head_report", ok=bool(stage_hidden_lm_head_report)) if args.stage_hidden_lm_head_report else {"kind": "stage_hidden_lm_head_report", "present": False},
        "multi_layer_stage_decode_report": artifact_entry(Path(args.multi_layer_stage_decode_report), output_dir, kind="multi_layer_stage_decode_report", ok=bool(multi_layer_stage_decode_report)) if args.multi_layer_stage_decode_report else {"kind": "multi_layer_stage_decode_report", "present": False},
        "full_prefix_stage_decode_report": artifact_entry(Path(args.full_prefix_stage_decode_report), output_dir, kind="full_prefix_stage_decode_report", ok=bool(full_prefix_stage_decode_report)) if args.full_prefix_stage_decode_report else {"kind": "full_prefix_stage_decode_report", "present": False},
        "dsa_indexer_report": artifact_entry(Path(args.dsa_indexer_report), output_dir, kind="dsa_indexer_report", ok=bool(dsa_indexer_report)) if args.dsa_indexer_report else {"kind": "dsa_indexer_report", "present": False},
        "pack_quantized_dequant_report": artifact_entry(Path(args.pack_quantized_dequant_report), output_dir, kind="pack_quantized_dequant_report", ok=bool(pack_quantized_dequant_report)) if args.pack_quantized_dequant_report else {"kind": "pack_quantized_dequant_report", "present": False},
        "pack_quantized_expert_mlp_report": artifact_entry(Path(args.pack_quantized_expert_mlp_report), output_dir, kind="pack_quantized_expert_mlp_report", ok=bool(pack_quantized_expert_mlp_report)) if args.pack_quantized_expert_mlp_report else {"kind": "pack_quantized_expert_mlp_report", "present": False},
        "pack_quantized_router_gather_report": artifact_entry(Path(args.pack_quantized_router_gather_report), output_dir, kind="pack_quantized_router_gather_report", ok=bool(pack_quantized_router_gather_report)) if args.pack_quantized_router_gather_report else {"kind": "pack_quantized_router_gather_report", "present": False},
        "pack_quantized_moe_mlp_report": artifact_entry(Path(args.pack_quantized_moe_mlp_report), output_dir, kind="pack_quantized_moe_mlp_report", ok=bool(pack_quantized_moe_mlp_report)) if args.pack_quantized_moe_mlp_report else {"kind": "pack_quantized_moe_mlp_report", "present": False},
        "decode_adapter_gap_report": artifact_entry(Path(args.decode_adapter_gap_report), output_dir, kind="decode_adapter_gap_report", ok=bool(decode_adapter_gap_report)) if args.decode_adapter_gap_report else {"kind": "decode_adapter_gap_report", "present": False},
        "same_request_report": artifact_entry(Path(args.same_request_report), output_dir, kind="same_request_report", ok=bool(same_report)) if args.same_request_report else {"kind": "same_request_report", "present": False},
        "public_artifact_safe": True,
    }
    write_json(support_path, support)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
        "support_json": artifact_entry(support_path, output_dir, kind="support_json", schema=SUPPORT_SCHEMA, ok=True),
    }
    write_json(summary_path, report)
    report["artifacts"]["summary_json"] = artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok")))
    write_json(summary_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--tpu-watch-report", required=True)
    parser.add_argument("--awq-stage-header-report", default="")
    parser.add_argument("--awq-stage-value-report", action="append", default=[])
    parser.add_argument("--tpu-stage-smoke-report", default="")
    parser.add_argument("--kaggle-source-search-report", default="")
    parser.add_argument("--stage-runtime-plan-report", default="")
    parser.add_argument("--stage-worker-package-report", default="")
    parser.add_argument("--stage-worker-push-probe-report", default="")
    parser.add_argument("--gpu-token-quota-report", default="")
    parser.add_argument("--transformers-decode-preflight-report", default="")
    parser.add_argument("--attention-projection-report", default="")
    parser.add_argument("--attention-single-token-report", default="")
    parser.add_argument("--kv-cache-decode-report", default="")
    parser.add_argument("--layer-decode-report", default="")
    parser.add_argument("--lm-head-token-report", default="")
    parser.add_argument("--dsa-masked-layer-decode-report", default="")
    parser.add_argument("--stage-hidden-lm-head-report", default="")
    parser.add_argument("--multi-layer-stage-decode-report", default="")
    parser.add_argument("--full-prefix-stage-decode-report", default="")
    parser.add_argument("--dsa-indexer-report", default="")
    parser.add_argument("--pack-quantized-dequant-report", default="")
    parser.add_argument("--pack-quantized-expert-mlp-report", default="")
    parser.add_argument("--pack-quantized-router-gather-report", default="")
    parser.add_argument("--pack-quantized-moe-mlp-report", default="")
    parser.add_argument("--decode-adapter-gap-report", default="")
    parser.add_argument("--same-request-report", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'glm52_kaggle_accelerator_deployment_rc.json'}")
        print(f"Goal achieved: {report.get('goal_achieved')}")
        print(f"Failure stage: {report.get('failure_stage')}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
