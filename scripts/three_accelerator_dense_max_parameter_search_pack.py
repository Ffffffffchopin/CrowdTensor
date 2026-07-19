#!/usr/bin/env python3
"""Build the dense GPU+TPU+CPU max-parameter search artifact."""

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

from scripts import three_accelerator_dense_qwen_frontier_pack as frontier_pack  # noqa: E402


SCHEMA = "three_accelerator_dense_max_parameter_search_v1"
SUPPORT_BUNDLE_SCHEMA = "three_accelerator_dense_max_parameter_search_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/three-accelerator-dense-max-parameter-search"
DEFAULT_FRONTIER_REPORT = (
    "dist/three-accelerator-dense-qwen-frontier-20260626-r8-live-72b-stage-plan-retained-32b/"
    "three_accelerator_dense_qwen_frontier.json"
)
DEFAULT_32B_BRIDGE_REPORT = (
    "dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r6-existing-session-4token/"
    "gpu_tpu_cpu_same_request_runtime_bridge_probe.json"
)
DEFAULT_72B_BRIDGE_REPORT = ""
DEFAULT_72B_ATTACH_STAGE_PLAN_REPORT = (
    "dist/kaggle-model-attach-probe-20260626-r7-72b-cpu-stage-plan/"
    "kaggle_model_attach_probe.json"
)
DEFAULT_72B_TPU_STAGE_LOAD_REPORT = (
    "dist/kaggle-tpu-72b-stage-owned-loader-probe-web-live-20260626-r3-stage32-40-one-layer-bridge-executor/"
    "kaggle_tpu_32b_stage_owned_loader_probe.json"
)
DEFAULT_WEB_TPU_CHANNEL_REPORT = ""
DEFAULT_WEB_TPU_ACTIVE_EVENT_REPORT = ""
DEFAULT_WEB_TPU_START_WAIT_REPORT = ""
DEFAULT_COLAB_TPU_REACQUIRE_REPORT = ""
DEFAULT_COLAB_TPU_RUNTIME_STABILITY_REPORT = ""
SENSITIVE_FRAGMENTS = frontier_pack.SENSITIVE_FRAGMENTS + (
    "JUPYTER_TOKEN",
    "XSRF-TOKEN",
    "_xsrf",
    "kaggle_session",
    "jupyterServerHttpUrl",
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
    return frontier_pack.parameter_value(value)


def max_parameter(values: list[str]) -> str:
    clean = [item for item in values if str(item or "").strip()]
    return max(clean, key=parameter_value, default="")


def infer_parameter_class(model_id: str, fallback: str = "") -> str:
    text = model_id.lower()
    for value in ("235b", "100b", "72b", "32b", "14b", "7b"):
        if value in text:
            return value
    return fallback


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
        "public_artifact_safe": bool(
            report.get("public_artifact_safe") is True
            or _dict(report.get("safety")).get("public_artifact_safe") is True
        ),
    }


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str, ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file(), "schema": schema}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def summarize_frontier(path: Path) -> dict[str, Any]:
    report = load_json(path)
    return {
        "schema": "dense_max_search_frontier_import_v1",
        "source": source_summary(path, report, kind="dense_qwen_frontier"),
        "frontier_ready": report.get("three_accelerator_dense_qwen_frontier_ready") is True,
        "max_attempted": str(report.get("largest_dense_model_attempted") or ""),
        "max_attached": str(report.get("largest_dense_model_attached") or ""),
        "max_stage_preflighted": str(report.get("largest_dense_model_stage_preflighted") or ""),
        "max_loaded": str(report.get("largest_dense_model_loaded") or ""),
        "max_same_request_decode": str(report.get("largest_dense_model_1token_decoded") or ""),
        "same_request_dense_frontier_success": report.get("same_request_dense_frontier_success") is True,
        "same_request_dense_32b_success": report.get("same_request_dense_32b_success") is True,
        "generated_token_count": _int(report.get("generated_token_count")),
        "accepted_stage_backends": sorted(
            str(item) for item in _list(_dict(report.get("baseline_32b_three_accelerator")).get("accepted_stage_backends"))
        ),
        "blockers": [str(item) for item in _list(report.get("blocker_codes")) if item],
        "failure_stage": str(report.get("frontier_failure_stage") or ""),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_32b_bridge(path: Path) -> dict[str, Any]:
    report = load_json(path)
    backends = sorted(str(item) for item in _list(report.get("accepted_stage_backends")) if item)
    safety = _dict(report.get("safety"))
    ready = bool(
        report.get("schema") == "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1"
        and report.get("ok") is True
        and report.get("same_request_runtime_bridge_verified") is True
        and report.get("gpu_tpu_cpu_32b_same_request_verified") is True
        and report.get("same_request_32b_model_verified") is True
        and {"cuda", "jax_tpu", "cpu"}.issubset(set(backends))
        and _int(report.get("generated_token_count")) >= 1
    )
    return {
        "schema": "dense_max_search_32b_same_request_import_v1",
        "source": source_summary(path, report, kind="gpu_tpu_cpu_32b_same_request_bridge"),
        "parameter_class": "32b",
        "model_id": str(report.get("target_model_id") or "Qwen/Qwen2.5-32B-Instruct"),
        "quantization": "none",
        "same_request_decode_verified": ready,
        "generated_token_count": _int(report.get("generated_token_count")),
        "accepted_stage_backends": backends,
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True,
        "tpu_runtime_adapter_ready": report.get("tpu_32b_runtime_adapter_ready") is True,
        "public_artifact_safe": bool(report.get("public_artifact_safe") is True or safety.get("public_artifact_safe") is True),
    }


def summarize_72b_bridge(path: Path) -> dict[str, Any]:
    report = load_json(path)
    if report.get("schema") == "kaggle_32b_full_heterogeneous_probe_v1":
        backends = {"cuda": False, "jax_tpu": False, "cpu": False}
        for item in _list(report.get("stage_summaries")):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("resource_kind") or "")
            if kind == "gpu":
                backends["cuda"] = True
            elif kind == "web_tpu":
                backends["jax_tpu"] = True
            elif kind == "cpu":
                backends["cpu"] = True
        accepted_backends = sorted(name for name, present in backends.items() if present)
        full_ready = bool(
            report.get("ok") is True
            and report.get("gpu_tpu_cpu_72b_same_request_verified") is True
            and report.get("same_request_72b_full_model_verified") is True
            and report.get("full_72b_weight_loading_public_claim") is True
            and report.get("full_72b_layer_coverage_verified") is True
            and report.get("gpu_tpu_cpu_72b_full_topology_verified") is True
            and report.get("stage_owned_full_precision_runtime_verified") is True
            and {"cuda", "jax_tpu", "cpu"}.issubset(set(accepted_backends))
            and _int(report.get("generated_token_count")) >= 1
        )
        model = _dict(report.get("model"))
        return {
            "schema": "dense_max_search_72b_same_request_import_v1",
            "source": source_summary(path, report, kind="gpu_tpu_cpu_72b_full_heterogeneous"),
            "imported": path.is_file(),
            "parameter_class": "72b" if model.get("parameter_count_b") == 72 else "",
            "model_id": str(model.get("repo") or ""),
            "quantization": "none",
            "same_request_stage_decode_verified": full_ready,
            "same_request_full_model_decode_verified": full_ready,
            "generated_token_count": _int(report.get("generated_token_count")),
            "accepted_stage_backends": accepted_backends,
            "full_72b_layer_coverage_verified": report.get("full_72b_layer_coverage_verified") is True,
            "gpu_tpu_cpu_72b_full_topology_verified": report.get("gpu_tpu_cpu_72b_full_topology_verified") is True,
            "stage_count": _int(model.get("stage_count")),
            "expected_layer_count": _int(model.get("expected_layer_count")),
            "tpu_stage_ready": any(item.get("resource_kind") == "web_tpu" and item.get("stage_weight_load_ready") is True for item in _list(report.get("stage_summaries")) if isinstance(item, dict)),
            "tpu_stage_layer_range": next((item.get("stage_layer_range") or [] for item in _list(report.get("stage_summaries")) if isinstance(item, dict) and item.get("resource_kind") == "web_tpu"), []),
            "tpu_executed_layer_count": 0,
            "tpu_loaded_execution_tensor_gb": 0,
            "full_72b_tpu_stage_loading_public_claim": full_ready,
            "full_72b_weight_loading_public_claim": report.get("full_72b_weight_loading_public_claim") is True,
            "blockers": [str(item) for item in _list(report.get("blockers")) if item],
            "public_artifact_safe": report.get("public_artifact_safe") is True or _dict(report.get("safety")).get("public_artifact_safe") is True,
        }
    backends = sorted(str(item) for item in _list(report.get("accepted_stage_backends")) if item)
    safety = _dict(report.get("safety"))
    stage_ready = bool(
        report.get("schema") == "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1"
        and report.get("ok") is True
        and report.get("same_request_runtime_bridge_verified") is True
        and report.get("gpu_tpu_cpu_72b_same_request_stage_verified") is True
        and report.get("same_request_72b_stage_verified") is True
        and {"cuda", "jax_tpu", "cpu"}.issubset(set(backends))
        and _int(report.get("generated_token_count")) >= 1
    )
    full_ready = bool(
        stage_ready
        and report.get("gpu_tpu_cpu_72b_same_request_verified") is True
        and report.get("same_request_72b_full_model_verified") is True
        and report.get("full_72b_weight_loading_public_claim") is True
    )
    runtime = _dict(report.get("runtime_device_summary"))
    return {
        "schema": "dense_max_search_72b_same_request_import_v1",
        "source": source_summary(path, report, kind="gpu_tpu_cpu_72b_same_request_bridge"),
        "imported": path.is_file(),
        "parameter_class": "72b" if infer_parameter_class(str(report.get("target_model_id") or ""), "") == "72b" else "",
        "model_id": str(report.get("target_model_id") or ""),
        "quantization": "none",
        "same_request_stage_decode_verified": stage_ready,
        "same_request_full_model_decode_verified": full_ready,
        "generated_token_count": _int(report.get("generated_token_count")),
        "accepted_stage_backends": backends,
        "tpu_stage_ready": report.get("tpu_target_runtime_adapter_ready") is True,
        "tpu_stage_layer_range": _dict(report.get("stage_reports")).get("jax_tpu_stage", {}).get("stage_layer_range", []),
        "tpu_executed_layer_count": _int(runtime.get("tpu_executed_layer_count")),
        "tpu_loaded_execution_tensor_gb": runtime.get("tpu_loaded_execution_tensor_gb", 0),
        "full_72b_tpu_stage_loading_public_claim": report.get("full_72b_tpu_stage_loading_public_claim") is True,
        "full_72b_weight_loading_public_claim": report.get("full_72b_weight_loading_public_claim") is True,
        "blockers": [str(item) for item in _list(report.get("blockers")) if item],
        "public_artifact_safe": bool(report.get("public_artifact_safe") is True or safety.get("public_artifact_safe") is True),
    }


def summarize_attach_stage_plan(path: Path) -> dict[str, Any]:
    report = load_json(path)
    runtime = _dict(report.get("runtime_report"))
    stage_plan = _dict(runtime.get("stage_plan"))
    cleanup = _dict(report.get("cleanup_status"))
    stage_backends = [str(item) for item in _list(stage_plan.get("stage_backends")) if item]
    ready = bool(
        report.get("schema") == "kaggle_model_attach_probe_v1"
        and report.get("ok") is True
        and report.get("kaggle_model_attach_probe_ready") is True
        and report.get("kaggle_model_attach_used") is True
        and report.get("parameter_class") == "72b"
        and runtime.get("path_present") is True
        and runtime.get("weight_index_present") is True
        and _int(runtime.get("safetensors_file_count")) >= 1
        and runtime.get("stage_owned_preflight_verified") is True
        and stage_plan.get("schema") == "kaggle_model_attach_stage_plan_v1"
        and {"cuda", "jax_tpu", "cpu"}.issubset(set(stage_backends))
        and cleanup.get("temporary_kaggle_kernel_deleted") is True
        and cleanup.get("temporary_private_package_removed") is True
    )
    return {
        "schema": "dense_max_search_72b_attach_stage_plan_import_v1",
        "source": source_summary(path, report, kind="kaggle_72b_dense_attach_stage_plan"),
        "parameter_class": str(report.get("parameter_class") or ""),
        "model_id": str(report.get("hf_repo") or "Qwen/Qwen2.5-72B-Instruct"),
        "model_source": str(report.get("model_source") or ""),
        "attached_runtime_path": str(report.get("expected_attached_path") or ""),
        "dense_full_precision": runtime.get("quantization_config_present") is not True,
        "attach_verified": ready,
        "stage_owned_preflight_verified": ready,
        "safetensors_file_count": _int(runtime.get("safetensors_file_count")),
        "weight_index_key_count": _int(runtime.get("weight_index_key_count")),
        "stage_plan_stage_count": _int(stage_plan.get("stage_count")),
        "stage_plan_backends": stage_backends,
        "stage_plan_assigned_key_count_total": _int(stage_plan.get("assigned_key_count_total")),
        "stage_plan_present_key_count_total": _int(stage_plan.get("present_key_count_total")),
        "stage_plan_total_logical_tensor_gb": stage_plan.get("total_planned_logical_tensor_gb", 0),
        "stage_plan_max_stage_logical_tensor_gb": stage_plan.get("max_stage_planned_logical_tensor_gb", 0),
        "cleanup": {
            "temporary_kaggle_kernel_deleted": cleanup.get("temporary_kaggle_kernel_deleted") is True,
            "temporary_private_package_removed": cleanup.get("temporary_private_package_removed") is True,
            "live_resources_left_running": cleanup.get("live_resources_left_running") is True,
        },
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_tpu_stage_attempt(path: Path) -> dict[str, Any]:
    report = load_json(path)
    runtime = _dict(report.get("runtime_report"))
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    is_colab_wrapper = report.get("schema") == "colab_tpu_qwen_stage_loader_probe_v1"
    model_id = str(report.get("model_repo") or runtime.get("model_repo") or "")
    parameter_class = infer_parameter_class(model_id, "72b")
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    if not blockers:
        blockers = [str(item) for item in _list(runtime.get("blockers")) if item]
    executed_layer_count = _int(report.get("executed_layer_count"))
    loaded_key_count = _int(report.get("loaded_execution_tensor_key_count"))
    if executed_layer_count == 0:
        executed_layer_count = _int(runtime.get("executed_layer_count"))
    if loaded_key_count == 0:
        loaded_key_count = _int(runtime.get("loaded_execution_tensor_key_count"))
    header_ready = report.get("stage_owned_header_verified") is True or runtime.get("stage_owned_header_verified") is True
    partial_tpu_ready = report.get("partial_tensor_to_tpu_verified") is True or runtime.get("partial_tensor_to_tpu_verified") is True
    full_stage_ready = (
        report.get("full_stage_owned_tpu_loader_ready") is True
        or runtime.get("full_stage_owned_tpu_loader_ready") is True
        or report.get("colab_qwen_stage_loader_ready") is True
    )
    stage_output_hash = str(report.get("stage_output_hash") or runtime.get("stage_output_hash") or "")
    tpu_device_count = _int(report.get("tpu_device_count"))
    if tpu_device_count == 0:
        tpu_device_count = _int(runtime.get("tpu_device_count"))
    layer_forward_ready = executed_layer_count >= 1 and bool(str(report.get("stage_output_hash") or ""))
    if not layer_forward_ready:
        layer_forward_ready = executed_layer_count >= 1 and bool(stage_output_hash)
    full_ready = bool(
        report.get("ok") is True
        and parameter_class == "72b"
        and header_ready
        and loaded_key_count >= 1
        and layer_forward_ready
        and full_stage_ready
        and tpu_device_count >= 1
    )
    cleanup_ok = bool(lifecycle.get("kernels_deleted") is True and lifecycle.get("private_packages_removed") is True)
    if is_colab_wrapper:
        cleanup_ok = True
    failure_stage = ""
    if not full_ready:
        if "web_tpu_jupyter_execute_timeout" in blockers:
            failure_stage = "tpu_web_jupyter_execute_timeout_before_72b_header_load"
        elif is_colab_wrapper and report.get("ok") is not True:
            failure_stage = "colab_tpu_72b_stage_loader_not_verified"
        elif not header_ready:
            failure_stage = "tpu_72b_stage_header_not_verified"
        elif not partial_tpu_ready:
            failure_stage = "tpu_72b_stage_tensor_load_not_verified"
        elif executed_layer_count < 1:
            failure_stage = "tpu_72b_layer_forward_not_verified"
        else:
            failure_stage = "tpu_72b_stage_load_attempt_failed"
    return {
        "schema": "dense_max_search_72b_tpu_stage_load_attempt_import_v1",
        "source": source_summary(
            path,
            report,
            kind="colab_tpu_72b_stage_load_attempt" if is_colab_wrapper else "kaggle_web_tpu_72b_stage_load_attempt",
        ),
        "provider": "colab_cli" if is_colab_wrapper else "kaggle_web",
        "parameter_class": parameter_class,
        "model_id": model_id,
        "stage_layer_range": report.get("stage_layer_range") or runtime.get("stage_layer_range") or [],
        "stage_owned_header_verified": header_ready,
        "partial_tensor_to_tpu_verified": partial_tpu_ready,
        "full_stage_owned_tpu_loader_ready": full_stage_ready,
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True or runtime.get("stage_local_kv_cache_verified") is True,
        "executed_layer_count": executed_layer_count,
        "loaded_execution_tensor_key_count": loaded_key_count,
        "loaded_execution_tensor_gb": report.get("loaded_execution_tensor_gb", runtime.get("loaded_execution_tensor_gb", 0)),
        "tpu_device_count": tpu_device_count,
        "tpu_device_kind": str(report.get("tpu_device_kind") or runtime.get("tpu_device_kind") or ""),
        "stage_output_hash_present": bool(stage_output_hash),
        "tpu_72b_stage_load_and_forward_verified": full_ready,
        "blocked_reason": str(report.get("blocked_reason") or runtime.get("blocked_reason") or ""),
        "blockers": blockers,
        "failure_stage": failure_stage,
        "cleanup": {
            "temporary_kaggle_kernels_deleted": cleanup_ok,
            "temporary_private_packages_removed": cleanup_ok,
            "private_kernel_push_count": _int(lifecycle.get("private_kernel_push_count")),
            "web_runtime_execution_count": _int(lifecycle.get("web_runtime_execution_count")),
            "live_resources_left_running": False,
        },
        "public_artifact_safe": bool(
            report.get("public_artifact_safe") is True
            and (_dict(report.get("safety")).get("public_artifact_safe") is True or runtime.get("public_artifact_safe") is True or is_colab_wrapper)
        ),
    }


def summarize_web_tpu_channel(path: Path) -> dict[str, Any]:
    report = load_json(path)
    return {
        "schema": "dense_max_search_web_tpu_channel_import_v1",
        "source": source_summary(path, report, kind="kaggle_web_tpu_execution_channel_probe"),
        "imported": path.is_file(),
        "web_tpu_execution_channel_ready": report.get("web_tpu_execution_channel_ready") is True,
        "small_jax_cell_ready": report.get("small_jax_cell_ready") is True,
        "tiny_qwen_like_cell_ready": report.get("tiny_qwen_like_cell_ready") is True,
        "tpu_runtime_attached": report.get("tpu_runtime_attached") is True,
        "tpu_device_count": _int(report.get("tpu_device_count")),
        "failure_stage": str(report.get("failure_stage") or ""),
        "blocked_reason": str(report.get("blocked_reason") or ""),
        "blockers": [str(item) for item in _list(report.get("blocker_codes")) if item],
        "cleanup": {
            "temporary_kaggle_kernels_deleted": _dict(report.get("cleanup_status")).get("temporary_kaggle_kernels_deleted") is True,
            "temporary_private_packages_removed": _dict(report.get("cleanup_status")).get("temporary_private_packages_removed") is True,
            "live_resources_left_running": _dict(report.get("cleanup_status")).get("live_resources_left_running") is True,
            "web_runtime_execution_count": _int(_dict(report.get("cleanup_status")).get("web_runtime_execution_count")),
        },
        "public_artifact_safe": bool(report.get("public_artifact_safe") is True and _dict(report.get("safety")).get("public_artifact_safe") is True),
    }


def summarize_web_tpu_active_event(path: Path) -> dict[str, Any]:
    report = load_json(path)
    cleanup = _dict(report.get("cleanup_status"))
    active_event_ready = bool(
        report.get("schema") == "kaggle_web_tpu_active_event_probe_v1"
        and report.get("active_event_runtime_ready") is True
        and report.get("active_event_running") is True
        and report.get("jupyter_frame_visible") is True
        and report.get("jupyter_session_or_kernel_visible") is True
    )
    failure_stage = ""
    if path.is_file() and not active_event_ready:
        if report.get("active_event_queued") is True:
            failure_stage = "active_event_queued"
        elif report.get("active_event_running") is not True:
            failure_stage = "active_event_not_running"
        elif report.get("jupyter_frame_visible") is not True:
            failure_stage = "active_event_jupyter_frame_not_visible"
        else:
            failure_stage = "active_event_runtime_not_ready"
    return {
        "schema": "dense_max_search_web_tpu_active_event_import_v1",
        "source": source_summary(path, report, kind="kaggle_web_tpu_active_event_probe"),
        "imported": path.is_file(),
        "active_event_runtime_ready": active_event_ready,
        "active_event_count": _int(report.get("active_event_count")),
        "tpu_v5e_active_event_visible": report.get("tpu_v5e_active_event_visible") is True,
        "active_event_queued": report.get("active_event_queued") is True,
        "active_event_running": report.get("active_event_running") is True,
        "active_event_dialog_opened": report.get("active_event_dialog_opened") is True,
        "active_event_opened": report.get("active_event_opened") is True,
        "jupyter_frame_visible": report.get("jupyter_frame_visible") is True,
        "jupyter_session_or_kernel_visible": report.get("jupyter_session_or_kernel_visible") is True,
        "jupyter_session_count": _int(report.get("jupyter_session_count")),
        "jupyter_kernel_count": _int(report.get("jupyter_kernel_count")),
        "failure_stage": failure_stage,
        "blocked_reason": str(report.get("blocked_reason") or ""),
        "blockers": [str(item) for item in _list(report.get("blocker_codes")) if item],
        "cleanup": {
            "temporary_kaggle_kernels_deleted": cleanup.get("temporary_kaggle_kernels_deleted") is True,
            "temporary_private_packages_removed": cleanup.get("temporary_private_packages_removed") is True,
            "live_resources_left_running": cleanup.get("live_resources_left_running") is True,
        },
        "public_artifact_safe": bool(report.get("public_artifact_safe") is True and _dict(report.get("safety")).get("public_artifact_safe") is True),
    }


def summarize_web_tpu_start_wait(path: Path) -> dict[str, Any]:
    report = load_json(path)
    cleanup = _dict(report.get("cleanup_status"))
    return {
        "schema": "dense_max_search_web_tpu_start_wait_import_v1",
        "source": source_summary(path, report, kind="kaggle_web_tpu_start_wait_probe"),
        "imported": path.is_file(),
        "start_clicked": report.get("start_clicked") is True,
        "web_tpu_ui_runtime_ready": report.get("web_tpu_ui_runtime_ready") is True,
        "bounded_wait_seconds": report.get("bounded_wait_seconds", 0),
        "failure_stage": "" if report.get("web_tpu_ui_runtime_ready") is True else "web_tpu_start_wait_runtime_not_ready",
        "blocked_reason": str(report.get("blocked_reason") or ""),
        "blockers": [str(item) for item in _list(report.get("blocker_codes")) if item],
        "cleanup": {
            "temporary_kaggle_kernels_deleted": cleanup.get("temporary_kaggle_kernels_deleted") is True,
            "temporary_private_packages_removed": cleanup.get("temporary_private_packages_removed") is True,
            "live_resources_left_running": cleanup.get("live_resources_left_running") is True,
        },
        "public_artifact_safe": bool(report.get("public_artifact_safe") is True and _dict(report.get("safety")).get("public_artifact_safe") is True),
    }


def summarize_colab_tpu_reacquire(path: Path) -> dict[str, Any]:
    report = load_json(path)
    attempts = [
        {
            "attempt_index": _int(item.get("attempt_index")),
            "accelerator_requested": str(item.get("accelerator_requested") or ""),
            "ok": item.get("ok") is True,
            "http_status": item.get("http_status"),
            "diagnosis_codes": [str(value) for value in _list(item.get("diagnosis_codes")) if value],
            "blockers": [str(value) for value in _list(item.get("blockers")) if value],
            "public_artifact_safe": item.get("public_artifact_safe") is True,
        }
        for item in _list(report.get("attempts"))
        if isinstance(item, dict)
    ]
    return {
        "schema": "dense_max_search_colab_tpu_reacquire_import_v1",
        "source": source_summary(path, report, kind="colab_tpu_reacquire_retry_probe"),
        "imported": path.is_file(),
        "colab_tpu_reacquire_ready": report.get("colab_tpu_reacquire_ready") is True,
        "attempts_requested": _int(report.get("attempts_requested")),
        "attempts_completed": _int(report.get("attempts_completed")),
        "accelerators_attempted": [str(item) for item in _list(report.get("accelerators_attempted")) if item],
        "successful_attempt_index": _int(report.get("successful_attempt_index")),
        "accelerator": str(report.get("accelerator") or ""),
        "endpoint_hash_present": bool(report.get("endpoint_hash")),
        "runtime_proxy_host_hash_present": bool(report.get("runtime_proxy_host_hash")),
        "attempts": attempts,
        "blockers": [str(item) for item in _list(report.get("blockers")) if item],
        "public_artifact_safe": bool(
            report.get("public_artifact_safe") is True
            and report.get("oauth_token_public") is False
            and report.get("runtime_proxy_token_public") is False
            and report.get("runtime_proxy_url_public") is False
            and report.get("endpoint_public") is False
            and report.get("credentials_public") is False
            and report.get("private_runtime_state_public") is False
        ),
    }


def summarize_colab_tpu_runtime_stability(path: Path) -> dict[str, Any]:
    report = load_json(path)
    kernel_error = str(report.get("kernel_error") or "")
    kernel_error_type = kernel_error.split(":", 1)[0] if kernel_error else ""
    return {
        "schema": "dense_max_search_colab_tpu_runtime_stability_import_v1",
        "source": source_summary(path, report, kind="colab_tpu_runtime_stability_probe"),
        "imported": path.is_file(),
        "colab_tpu_runtime_stably_acquired": report.get("colab_tpu_runtime_stably_acquired") is True,
        "runtime_proxy_connected": report.get("runtime_proxy_connected") is True,
        "rounds_requested": _int(report.get("rounds_requested")),
        "rounds_completed": _int(report.get("rounds_completed")),
        "rounds_ready": _int(report.get("rounds_ready")),
        "observed_device_count_max": _int(report.get("observed_device_count_max")),
        "accelerator": str(report.get("accelerator") or ""),
        "variant": str(report.get("variant") or ""),
        "endpoint_hash_present": bool(report.get("endpoint_hash")),
        "runtime_proxy_host_hash_present": bool(report.get("runtime_proxy_host_hash")),
        "kernel_id_hash_present": bool(report.get("kernel_id_hash")),
        "session_id_hash_present": bool(report.get("session_id_hash")),
        "kernel_error_type": kernel_error_type,
        "kernel_error_digest": ("sha256:" + hashlib.sha256(kernel_error.encode("utf-8")).hexdigest()) if kernel_error else "",
        "kernel_error_public": False,
        "blockers": [] if report.get("colab_tpu_runtime_stably_acquired") is True else ["colab_tpu_runtime_stability_not_ready"],
        "public_artifact_safe": bool(
            report.get("public_artifact_safe") is True
            and report.get("runtime_proxy_token_public") is False
            and report.get("runtime_proxy_url_public") is False
            and report.get("endpoint_public") is False
        ),
    }


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "ok": report.get("ok") is True,
        "generated_at": report.get("generated_at"),
        "report_schema": report.get("schema"),
        "canonical_summary": {
            "max_successful_same_request_decode_parameter_class": report.get(
                "max_successful_same_request_decode_parameter_class"
            ),
            "max_attempted_parameter_class": report.get("max_attempted_parameter_class"),
            "max_attached_parameter_class": report.get("max_attached_parameter_class"),
            "max_stage_loaded_parameter_class": report.get("max_stage_loaded_parameter_class"),
            "max_tpu_executed_parameter_class": report.get("max_tpu_executed_parameter_class"),
            "generated_token_count": report.get("generated_token_count"),
            "accepted_stage_backends": report.get("accepted_stage_backends") or [],
            "blocker_codes": report.get("blocker_codes") or [],
        },
        "public_artifact_safe": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frontier = summarize_frontier(Path(args.frontier_report))
    bridge_32b = summarize_32b_bridge(Path(args.bridge_32b_report))
    bridge_72b = summarize_72b_bridge(Path(args.bridge_72b_report)) if str(args.bridge_72b_report or "").strip() else {
        "schema": "dense_max_search_72b_same_request_import_v1",
        "imported": False,
        "parameter_class": "",
        "same_request_stage_decode_verified": False,
        "same_request_full_model_decode_verified": False,
        "generated_token_count": 0,
        "accepted_stage_backends": [],
        "blockers": [],
        "public_artifact_safe": True,
    }
    attach_72b = summarize_attach_stage_plan(Path(args.attach_72b_stage_plan_report))
    tpu_72b = summarize_tpu_stage_attempt(Path(args.tpu_72b_stage_load_report))
    web_tpu_channel = summarize_web_tpu_channel(Path(args.web_tpu_channel_report)) if str(args.web_tpu_channel_report or "").strip() else {
        "schema": "dense_max_search_web_tpu_channel_import_v1",
        "imported": False,
        "web_tpu_execution_channel_ready": False,
        "blockers": [],
        "cleanup": {
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "web_runtime_execution_count": 0,
        },
        "public_artifact_safe": True,
    }
    web_tpu_active_event = summarize_web_tpu_active_event(Path(args.web_tpu_active_event_report)) if str(args.web_tpu_active_event_report or "").strip() else {
        "schema": "dense_max_search_web_tpu_active_event_import_v1",
        "imported": False,
        "active_event_runtime_ready": False,
        "active_event_count": 0,
        "blockers": [],
        "cleanup": {
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
        },
        "public_artifact_safe": True,
    }
    web_tpu_start_wait = summarize_web_tpu_start_wait(Path(args.web_tpu_start_wait_report)) if str(args.web_tpu_start_wait_report or "").strip() else {
        "schema": "dense_max_search_web_tpu_start_wait_import_v1",
        "imported": False,
        "start_clicked": False,
        "web_tpu_ui_runtime_ready": False,
        "blockers": [],
        "cleanup": {
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
        },
        "public_artifact_safe": True,
    }
    colab_tpu_reacquire = summarize_colab_tpu_reacquire(Path(args.colab_tpu_reacquire_report)) if str(args.colab_tpu_reacquire_report or "").strip() else {
        "schema": "dense_max_search_colab_tpu_reacquire_import_v1",
        "imported": False,
        "colab_tpu_reacquire_ready": False,
        "attempts_requested": 0,
        "attempts_completed": 0,
        "accelerators_attempted": [],
        "attempts": [],
        "blockers": [],
        "public_artifact_safe": True,
    }
    colab_tpu_runtime_stability = summarize_colab_tpu_runtime_stability(Path(args.colab_tpu_runtime_stability_report)) if str(args.colab_tpu_runtime_stability_report or "").strip() else {
        "schema": "dense_max_search_colab_tpu_runtime_stability_import_v1",
        "imported": False,
        "colab_tpu_runtime_stably_acquired": False,
        "runtime_proxy_connected": False,
        "rounds_requested": 0,
        "rounds_completed": 0,
        "rounds_ready": 0,
        "observed_device_count_max": 0,
        "blockers": [],
        "public_artifact_safe": True,
    }

    max_success_decode = (
        "72b" if bridge_72b.get("same_request_full_model_decode_verified") is True
        else "32b" if bridge_32b.get("same_request_decode_verified") is True
        else ""
    )
    max_attempted = max_parameter([frontier.get("max_attempted", ""), attach_72b.get("parameter_class", ""), tpu_72b.get("parameter_class", "")])
    max_attached = max_parameter([
        frontier.get("max_attached", ""),
        attach_72b.get("parameter_class", "") if attach_72b.get("attach_verified") is True else "",
    ])
    max_stage_preflighted = max_parameter([
        frontier.get("max_stage_preflighted", ""),
        attach_72b.get("parameter_class", "") if attach_72b.get("stage_owned_preflight_verified") is True else "",
    ])
    tpu_72b_verified = tpu_72b.get("tpu_72b_stage_load_and_forward_verified") is True
    max_stage_loaded = max_parameter([
        frontier.get("max_loaded", ""),
        "72b" if tpu_72b_verified else "",
    ])
    max_tpu_executed = max_parameter([
        "32b" if frontier.get("same_request_dense_32b_success") is True else "",
        "72b" if tpu_72b_verified else "",
    ])

    blockers = set()
    if max_success_decode != max_attempted:
        blockers.add("larger_than_32b_same_request_decode_not_verified")
    if bridge_72b.get("same_request_stage_decode_verified") is True and bridge_72b.get("same_request_full_model_decode_verified") is not True:
        blockers.add("dense_72b_same_request_stage_verified_but_full_model_decode_not_verified")
    if max_attached != max_attempted:
        blockers.add("largest_attempted_dense_model_not_attached")
    if max_stage_preflighted != max_attempted:
        blockers.add("largest_attempted_dense_model_not_stage_preflighted")
    if not tpu_72b_verified:
        blockers.add("dense_72b_tpu_stage_load_and_forward_not_verified")
        if tpu_72b.get("failure_stage"):
            blockers.add(str(tpu_72b["failure_stage"]))
    if web_tpu_channel.get("imported") is True and web_tpu_channel.get("web_tpu_execution_channel_ready") is not True:
        blockers.add("web_tpu_execution_channel_not_ready")
        if web_tpu_channel.get("failure_stage"):
            blockers.add(f"web_tpu_channel_{web_tpu_channel['failure_stage']}")
    active_event_overridden_by_channel = bool(
        web_tpu_channel.get("imported") is True
        and web_tpu_channel.get("web_tpu_execution_channel_ready") is True
    )
    if (
        web_tpu_active_event.get("imported") is True
        and web_tpu_active_event.get("active_event_runtime_ready") is not True
        and not active_event_overridden_by_channel
    ):
        blockers.add("web_tpu_active_event_not_ready")
        if web_tpu_active_event.get("failure_stage"):
            blockers.add(f"web_tpu_active_event_{web_tpu_active_event['failure_stage']}")
    if (
        web_tpu_start_wait.get("imported") is True
        and web_tpu_start_wait.get("web_tpu_ui_runtime_ready") is not True
        and not active_event_overridden_by_channel
    ):
        blockers.add("web_tpu_start_wait_runtime_not_ready")
    colab_reacquire_overridden_by_web_channel = bool(
        colab_tpu_reacquire.get("imported") is True
        and colab_tpu_reacquire.get("colab_tpu_reacquire_ready") is not True
        and
        web_tpu_channel.get("imported") is True
        and web_tpu_channel.get("web_tpu_execution_channel_ready") is True
    )
    if (
        colab_tpu_reacquire.get("imported") is True
        and colab_tpu_reacquire.get("colab_tpu_reacquire_ready") is not True
        and not colab_reacquire_overridden_by_web_channel
    ):
        blockers.add("colab_tpu_reacquire_not_ready")
    colab_runtime_overridden = bool(
        colab_tpu_runtime_stability.get("imported") is True
        and colab_tpu_runtime_stability.get("colab_tpu_runtime_stably_acquired") is not True
        and (
            colab_reacquire_overridden_by_web_channel
            or colab_tpu_reacquire.get("colab_tpu_reacquire_ready") is True
            or web_tpu_channel.get("web_tpu_execution_channel_ready") is True
        )
    )
    if (
        colab_tpu_runtime_stability.get("imported") is True
        and colab_tpu_runtime_stability.get("colab_tpu_runtime_stably_acquired") is not True
        and not colab_runtime_overridden
    ):
        blockers.add("colab_tpu_runtime_stability_not_ready")
    blockers.update(str(item) for item in _list(frontier.get("blockers")) if item)
    blockers.update(str(item) for item in _list(tpu_72b.get("blockers")) if item)
    blockers.update(str(item) for item in _list(web_tpu_channel.get("blockers")) if item)
    blockers.update(str(item) for item in _list(web_tpu_active_event.get("blockers")) if item)
    blockers.update(str(item) for item in _list(web_tpu_start_wait.get("blockers")) if item)
    blockers.update(str(item) for item in _list(colab_tpu_reacquire.get("blockers")) if item)
    if not colab_runtime_overridden:
        blockers.update(str(item) for item in _list(colab_tpu_runtime_stability.get("blockers")) if item)
    blockers.update(str(item) for item in _list(bridge_72b.get("blockers")) if item)

    failure_stage = ""
    if max_success_decode != max_attempted:
        failure_stage = str(tpu_72b.get("failure_stage") or frontier.get("failure_stage") or "larger_dense_same_request_decode_not_verified")
        if tpu_72b_verified:
            failure_stage = "dense_72b_same_request_decode_not_verified_after_tpu_stage_forward"
        if bridge_72b.get("same_request_stage_decode_verified") is True and bridge_72b.get("same_request_full_model_decode_verified") is not True:
            failure_stage = "dense_72b_stage_same_request_verified_but_full_model_decode_not_verified"
        if web_tpu_channel.get("imported") is True and web_tpu_channel.get("web_tpu_execution_channel_ready") is not True:
            failure_stage = f"web_tpu_channel_{web_tpu_channel.get('failure_stage') or 'not_ready'}"
        if (
            web_tpu_active_event.get("imported") is True
            and web_tpu_active_event.get("active_event_runtime_ready") is not True
            and not active_event_overridden_by_channel
        ):
            failure_stage = f"web_tpu_active_event_{web_tpu_active_event.get('failure_stage') or 'not_ready'}"
        if (
            web_tpu_start_wait.get("imported") is True
            and web_tpu_start_wait.get("web_tpu_ui_runtime_ready") is not True
            and not active_event_overridden_by_channel
        ):
            failure_stage = str(web_tpu_start_wait.get("failure_stage") or "web_tpu_start_wait_runtime_not_ready")
        if (
            colab_tpu_reacquire.get("imported") is True
            and colab_tpu_reacquire.get("colab_tpu_reacquire_ready") is not True
            and not colab_reacquire_overridden_by_web_channel
        ):
            failure_stage = "colab_tpu_reacquire_not_ready"
        if (
            colab_tpu_runtime_stability.get("imported") is True
            and colab_tpu_runtime_stability.get("colab_tpu_runtime_stably_acquired") is not True
            and not colab_runtime_overridden
            and colab_tpu_reacquire.get("imported") is not True
        ):
            failure_stage = "colab_tpu_runtime_stability_not_ready"

    cleanup_status = {
        "temporary_kaggle_kernels_deleted": bool(
            attach_72b.get("cleanup", {}).get("temporary_kaggle_kernel_deleted") is True
            and tpu_72b.get("cleanup", {}).get("temporary_kaggle_kernels_deleted") is True
        ),
        "temporary_private_packages_removed": bool(
            attach_72b.get("cleanup", {}).get("temporary_private_package_removed") is True
            and tpu_72b.get("cleanup", {}).get("temporary_private_packages_removed") is True
        ),
        "live_resources_left_running": bool(
            attach_72b.get("cleanup", {}).get("live_resources_left_running") is True
            or tpu_72b.get("cleanup", {}).get("live_resources_left_running") is True
        ),
        "web_runtime_execution_count": tpu_72b.get("cleanup", {}).get("web_runtime_execution_count", 0),
        "web_tpu_channel_runtime_execution_count": web_tpu_channel.get("cleanup", {}).get("web_runtime_execution_count", 0),
        "note": "This pack did not create live resources; imported attempts carry cleanup evidence.",
    }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "three_accelerator_dense_max_parameter_search_ready": True,
        "output_dir": str(output_dir),
        "goal_scope": {
            "single_kaggle_account_only": True,
            "multi_account_limit_bypass_allowed": False,
            "accelerators_required_for_frontier_success": ["cuda", "jax_tpu", "cpu"],
            "dense_full_precision_main_path": True,
            "quantized_main_path_success_allowed": False,
            "kaggle_models_attach_preferred_for_large_weights": True,
            "bounded_live_attempts_required": True,
        },
        "max_successful_same_request_decode_parameter_class": max_success_decode,
        "max_attempted_parameter_class": max_attempted,
        "max_attached_parameter_class": max_attached,
        "max_stage_preflighted_parameter_class": max_stage_preflighted,
        "max_stage_loaded_parameter_class": max_stage_loaded,
        "max_tpu_executed_parameter_class": max_tpu_executed,
        "generated_token_count": _int(bridge_72b.get("generated_token_count")) if max_success_decode == "72b" else _int(bridge_32b.get("generated_token_count")),
        "accepted_stage_backends": bridge_72b.get("accepted_stage_backends") if max_success_decode == "72b" else bridge_32b.get("accepted_stage_backends") or [],
        "failure_stage": failure_stage,
        "blocker_codes": sorted(blockers),
        "frontier_import": frontier,
        "same_request_32b_import": bridge_32b,
        "same_request_72b_import": bridge_72b,
        "dense_72b_attach_stage_plan_import": attach_72b,
        "web_tpu_execution_channel_import": web_tpu_channel,
        "web_tpu_active_event_import": web_tpu_active_event,
        "web_tpu_start_wait_import": web_tpu_start_wait,
        "web_tpu_active_event_overridden_by_execution_channel": active_event_overridden_by_channel,
        "web_tpu_start_wait_overridden_by_execution_channel": active_event_overridden_by_channel,
        "colab_tpu_reacquire_import": colab_tpu_reacquire,
        "colab_tpu_reacquire_overridden_by_web_tpu_channel": colab_reacquire_overridden_by_web_channel,
        "colab_tpu_runtime_stability_import": colab_tpu_runtime_stability,
        "colab_tpu_runtime_stability_overridden": colab_runtime_overridden,
        "dense_72b_tpu_stage_load_attempt": tpu_72b,
        "attempt_ladder": [
            {
                "parameter_class": "32b",
                "model_id": bridge_32b.get("model_id"),
                "dense_full_precision": True,
                "same_request_decode_verified": bridge_32b.get("same_request_decode_verified") is True,
                "generated_token_count": bridge_32b.get("generated_token_count"),
                "accepted_stage_backends": bridge_32b.get("accepted_stage_backends") or [],
                "blocked_reason": "",
            },
            {
                "parameter_class": "72b",
                "model_id": attach_72b.get("model_id") or tpu_72b.get("model_id"),
                "dense_full_precision": True,
                "attach_verified": attach_72b.get("attach_verified") is True,
                "stage_owned_preflight_verified": attach_72b.get("stage_owned_preflight_verified") is True,
                "web_tpu_execution_channel_ready": web_tpu_channel.get("web_tpu_execution_channel_ready") is True,
                "web_tpu_active_event_runtime_ready": web_tpu_active_event.get("active_event_runtime_ready") is True,
                "web_tpu_start_wait_runtime_ready": web_tpu_start_wait.get("web_tpu_ui_runtime_ready") is True,
                "colab_tpu_reacquire_ready": colab_tpu_reacquire.get("colab_tpu_reacquire_ready") is True,
                "colab_tpu_reacquire_overridden_by_web_tpu_channel": colab_reacquire_overridden_by_web_channel,
                "colab_tpu_runtime_stably_acquired": colab_tpu_runtime_stability.get("colab_tpu_runtime_stably_acquired") is True,
                "colab_tpu_runtime_stability_overridden": colab_runtime_overridden,
                "tpu_stage_load_and_forward_verified": tpu_72b_verified,
                "same_request_stage_decode_verified": bridge_72b.get("same_request_stage_decode_verified") is True,
                "same_request_decode_verified": bridge_72b.get("same_request_full_model_decode_verified") is True,
                "blocked_reason": failure_stage,
            },
        ],
        "cleanup_status": cleanup_status,
        "diagnosis_codes": [
            "dense_max_parameter_search_ready",
            f"max_successful_same_request_decode_{max_success_decode or 'none'}",
            f"max_attempted_{max_attempted or 'none'}",
            "dense_72b_tpu_stage_load_and_forward_verified" if tpu_72b_verified else "dense_72b_tpu_stage_load_and_forward_not_verified",
            "dense_72b_same_request_stage_decode_verified" if bridge_72b.get("same_request_stage_decode_verified") else "dense_72b_same_request_stage_decode_not_verified",
            "dense_72b_full_model_same_request_decode_verified" if bridge_72b.get("same_request_full_model_decode_verified") else "dense_72b_full_model_same_request_decode_not_verified",
            "web_tpu_execution_channel_ready" if web_tpu_channel.get("web_tpu_execution_channel_ready") else "web_tpu_execution_channel_not_ready",
            "web_tpu_active_event_runtime_ready" if web_tpu_active_event.get("active_event_runtime_ready") else "web_tpu_active_event_runtime_not_ready",
            "web_tpu_start_wait_runtime_ready" if web_tpu_start_wait.get("web_tpu_ui_runtime_ready") else "web_tpu_start_wait_runtime_not_ready",
            "colab_tpu_reacquire_ready" if colab_tpu_reacquire.get("colab_tpu_reacquire_ready") else "colab_tpu_reacquire_not_ready",
            "colab_tpu_runtime_stability_ready" if colab_tpu_runtime_stability.get("colab_tpu_runtime_stably_acquired") else "colab_tpu_runtime_stability_not_ready",
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
            "This artifact distinguishes attach, stage preflight, stage load, TPU execution, and same-request decode.",
            "72B attach or stage-owned preflight is not counted as 72B inference.",
            "A 72B same-request stage bridge is not counted as full 72B same-request decode unless all-layer/full-weight evidence is present.",
            (
                "The current largest successful dense GPU+TPU+CPU same-request decode is 72B."
                if max_success_decode == "72b"
                else "The current largest successful dense GPU+TPU+CPU same-request decode remains 32B."
            ),
            (
                "The 72B TPU stage load and forward is verified."
                if tpu_72b_verified
                else "The 72B live TPU stage attempt did not produce public-safe proof of 72B tensor load or layer forward."
            ),
            "The current Web TPU execution-channel probe is recorded separately; if it is not ready, a new 72B live-load attempt is not meaningful.",
            "The current Web TPU Active Events state is recorded separately; queued events are scheduling evidence only.",
            "The Colab TPU reacquire retry import records only allocation availability, not model inference.",
            "The Colab TPU runtime stability import records whether an existing session proxy is still usable, not model inference.",
            "Larger dense models should only be attempted after 72B live stage loading and same-request decode are verified.",
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["three_accelerator_dense_max_parameter_search_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blocker_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks

    summary_path = output_dir / "three_accelerator_dense_max_parameter_search.json"
    support_path = output_dir / "support_bundle.json"
    write_json(summary_path, report)
    support = build_support_bundle(report)
    write_json(support_path, support)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
        "support_bundle_json": artifact_entry(support_path, output_dir, kind="support_bundle_json", schema=SUPPORT_BUNDLE_SCHEMA, ok=bool(support.get("ok"))),
        "frontier_json": artifact_entry(Path(args.frontier_report), output_dir, kind="frontier_json", schema="three_accelerator_dense_qwen_frontier_v1", ok=bool(frontier.get("frontier_ready"))),
        "same_request_32b_json": artifact_entry(Path(args.bridge_32b_report), output_dir, kind="same_request_32b_json", schema="gpu_tpu_cpu_same_request_runtime_bridge_probe_v1", ok=bool(bridge_32b.get("same_request_decode_verified"))),
        "same_request_72b_json": artifact_entry(Path(args.bridge_72b_report), output_dir, kind="same_request_72b_json", schema="gpu_tpu_cpu_same_request_runtime_bridge_probe_v1", ok=bool(bridge_72b.get("same_request_full_model_decode_verified"))) if str(args.bridge_72b_report or "").strip() else {"kind": "same_request_72b_json", "present": False, "ok": False, "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1"},
        "attach_72b_stage_plan_json": artifact_entry(Path(args.attach_72b_stage_plan_report), output_dir, kind="attach_72b_stage_plan_json", schema="kaggle_model_attach_probe_v1", ok=bool(attach_72b.get("stage_owned_preflight_verified"))),
        "tpu_72b_stage_load_attempt_json": artifact_entry(Path(args.tpu_72b_stage_load_report), output_dir, kind="tpu_72b_stage_load_attempt_json", schema="kaggle_tpu_32b_stage_owned_loader_probe_v1", ok=bool(tpu_72b.get("tpu_72b_stage_load_and_forward_verified"))),
        "web_tpu_channel_json": artifact_entry(Path(args.web_tpu_channel_report), output_dir, kind="web_tpu_channel_json", schema="kaggle_web_tpu_execution_channel_probe_v1", ok=bool(web_tpu_channel.get("web_tpu_execution_channel_ready"))) if str(args.web_tpu_channel_report or "").strip() else {"kind": "web_tpu_channel_json", "present": False, "ok": False, "schema": "kaggle_web_tpu_execution_channel_probe_v1"},
        "web_tpu_active_event_json": artifact_entry(Path(args.web_tpu_active_event_report), output_dir, kind="web_tpu_active_event_json", schema="kaggle_web_tpu_active_event_probe_v1", ok=bool(web_tpu_active_event.get("active_event_runtime_ready"))) if str(args.web_tpu_active_event_report or "").strip() else {"kind": "web_tpu_active_event_json", "present": False, "ok": False, "schema": "kaggle_web_tpu_active_event_probe_v1"},
        "web_tpu_start_wait_json": artifact_entry(Path(args.web_tpu_start_wait_report), output_dir, kind="web_tpu_start_wait_json", schema="kaggle_web_tpu_start_wait_probe_v1", ok=bool(web_tpu_start_wait.get("web_tpu_ui_runtime_ready"))) if str(args.web_tpu_start_wait_report or "").strip() else {"kind": "web_tpu_start_wait_json", "present": False, "ok": False, "schema": "kaggle_web_tpu_start_wait_probe_v1"},
        "colab_tpu_reacquire_json": artifact_entry(Path(args.colab_tpu_reacquire_report), output_dir, kind="colab_tpu_reacquire_json", schema="colab_tpu_reacquire_retry_probe_v1", ok=bool(colab_tpu_reacquire.get("colab_tpu_reacquire_ready"))) if str(args.colab_tpu_reacquire_report or "").strip() else {"kind": "colab_tpu_reacquire_json", "present": False, "ok": False, "schema": "colab_tpu_reacquire_retry_probe_v1"},
        "colab_tpu_runtime_stability_json": artifact_entry(Path(args.colab_tpu_runtime_stability_report), output_dir, kind="colab_tpu_runtime_stability_json", schema="colab_tpu_runtime_stability_probe_v1", ok=bool(colab_tpu_runtime_stability.get("colab_tpu_runtime_stably_acquired"))) if str(args.colab_tpu_runtime_stability_report or "").strip() else {"kind": "colab_tpu_runtime_stability_json", "present": False, "ok": False, "schema": "colab_tpu_runtime_stability_probe_v1"},
    }
    write_json(summary_path, report)
    write_json(support_path, build_support_bundle(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense GPU+TPU+CPU max-parameter search artifact.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frontier-report", default=DEFAULT_FRONTIER_REPORT)
    parser.add_argument("--bridge-32b-report", default=DEFAULT_32B_BRIDGE_REPORT)
    parser.add_argument("--bridge-72b-report", default=DEFAULT_72B_BRIDGE_REPORT)
    parser.add_argument("--attach-72b-stage-plan-report", default=DEFAULT_72B_ATTACH_STAGE_PLAN_REPORT)
    parser.add_argument("--tpu-72b-stage-load-report", default=DEFAULT_72B_TPU_STAGE_LOAD_REPORT)
    parser.add_argument("--web-tpu-channel-report", default=DEFAULT_WEB_TPU_CHANNEL_REPORT)
    parser.add_argument("--web-tpu-active-event-report", default=DEFAULT_WEB_TPU_ACTIVE_EVENT_REPORT)
    parser.add_argument("--web-tpu-start-wait-report", default=DEFAULT_WEB_TPU_START_WAIT_REPORT)
    parser.add_argument("--colab-tpu-reacquire-report", default=DEFAULT_COLAB_TPU_REACQUIRE_REPORT)
    parser.add_argument("--colab-tpu-runtime-stability-report", default=DEFAULT_COLAB_TPU_RUNTIME_STABILITY_REPORT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'three_accelerator_dense_max_parameter_search.json'}")
        print(f"Ready: {report.get('three_accelerator_dense_max_parameter_search_ready')}")
        print(f"Max successful same-request decode: {report.get('max_successful_same_request_decode_parameter_class')}")
        print(f"Max attempted: {report.get('max_attempted_parameter_class')}")
        if report.get("blocker_codes"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blocker_codes") or []))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
