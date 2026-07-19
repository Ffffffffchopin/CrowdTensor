#!/usr/bin/env python3
"""Build DeepSeek-V4-Flash Quantized Swarm RC evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "deepseek_v4_flash_quantized_swarm_rc_v1"
SUPPORT_BUNDLE_SCHEMA = "deepseek_v4_flash_quantized_swarm_rc_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-quantized-swarm-rc"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "runtime_proxy_token",
    "oauth_token",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_short_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:16]


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


def summarize_source(report: dict[str, Any]) -> dict[str, Any]:
    recommended = _dict(report.get("recommended_live_probe_candidate"))
    return {
        "schema": "deepseek_v4_flash_quantized_source_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "resolver_ready": report.get("deepseek_v4_flash_quantized_source_resolver_ready") is True,
        "model": _dict(report.get("model")),
        "candidate_count": _int(report.get("candidate_count")),
        "ready_candidate_count": _int(report.get("ready_candidate_count")),
        "recommended_candidate_id": str(recommended.get("candidate_id") or ""),
        "recommended_repo": str(recommended.get("repo") or ""),
        "recommended_quant": str(recommended.get("quant") or ""),
        "recommended_total_size_gb": _float(recommended.get("total_size_gb")),
        "recommended_split_file_count": _int(recommended.get("split_file_count")),
        "recommended_runtime_backend": str(recommended.get("runtime_backend") or ""),
        "recommended_runtime_fork": str(recommended.get("runtime_fork") or ""),
        "recommended_files": _list(recommended.get("files")),
        "recommended_blockers": [str(item) for item in _list(recommended.get("blockers"))],
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_single_kernel_probe(report: dict[str, Any]) -> dict[str, Any]:
    model = _dict(report.get("model"))
    runtime = _dict(report.get("runtime"))
    probe = _dict(report.get("probe_summary"))
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    return {
        "schema": "deepseek_v4_flash_single_kernel_probe_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "fresh_kaggle_run_performed": report.get("fresh_kaggle_run_performed") is True,
        "one_token_generation_verified": report.get("one_token_generation_verified") is True
        or report.get("probe_success") is True
        or probe.get("one_token_generation_verified") is True,
        "model_repo": str(model.get("repo") or ""),
        "quant": str(model.get("quant") or ""),
        "parameter_count_b": _float(model.get("parameter_count_b") or 284.0),
        "active_parameter_count_b": _float(model.get("active_parameter_count_b") or 13.0),
        "runtime_backend": str(runtime.get("backend") or runtime.get("runtime_backend") or ""),
        "cross_kernel_sharded": runtime.get("cross_kernel_sharded") is True,
        "gpu_count": _int(probe.get("gpu_count")),
        "gpu_names": [str(item) for item in _list(probe.get("gpu_names"))],
        "downloaded_file_count": _int(probe.get("downloaded_file_count")),
        "downloaded_mb": _int(probe.get("downloaded_mb")),
        "generated_token_count": _int(probe.get("generated_token_count")),
        "blocked_reason": str(report.get("blocked_reason") or ""),
        "diagnosis_codes": [str(item) for item in _list(probe.get("diagnosis_codes")) + _list(report.get("diagnosis_codes"))],
        "blockers": [str(item) for item in _list(probe.get("blockers")) + _list(report.get("blockers"))],
        "kernel_deleted": lifecycle.get("kernel_deleted") is True,
        "private_package_removed": lifecycle.get("private_package_removed") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True or _dict(report.get("safety")).get("public_artifact_safe") is True,
    }


def summarize_same_request(report: dict[str, Any]) -> dict[str, Any]:
    providers = set(str(item) for item in _list(report.get("accepted_providers")))
    provider_counts = _dict(report.get("provider_stage_counts"))
    colab_rpc = _dict(report.get("colab_rpc"))
    colab_fallback = _dict(colab_rpc.get("colab_fallback"))
    authuser_fallback = _dict(colab_rpc.get("authuser_fallback"))
    generated = _int(report.get("generated_token_count") or _dict(report.get("coordinator")).get("generated_token_count"))
    verified = bool(
        report.get("deepseek_v4_flash_quantized_same_request_verified") is True
        or report.get("same_request_decode_verified") is True
        or report.get("kaggle_colab_gpu_cpu_same_request_verified") is True
    )
    required_providers = {"kaggle_cuda", "colab_cuda", "cpu"}
    ready = bool(verified and generated >= 1 and required_providers.issubset(providers))
    return {
        "schema": "deepseek_v4_flash_same_request_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "same_request_decode_verified": ready,
        "generated_token_count": generated,
        "accepted_providers": sorted(providers),
        "provider_stage_counts": {
            "kaggle_cuda": _int(provider_counts.get("kaggle_cuda")),
            "colab_cuda": _int(provider_counts.get("colab_cuda")),
            "cpu": _int(provider_counts.get("cpu")),
        },
        "stage_task_counts": _dict(report.get("stage_task_counts")),
        "mode": str(report.get("mode") or ""),
        "live_run_performed": report.get("live_run_performed") is True,
        "failure_stage": str(report.get("failure_stage") or ""),
        "colab_fallback": {
            "present": bool(colab_fallback),
            "attempt_count": _int(colab_fallback.get("attempt_count")),
            "attempted_targets": [
                {
                    "accelerator": str(_dict(item).get("accelerator") or ""),
                    "authuser": str(_dict(item).get("authuser") or ""),
                }
                for item in _list(colab_fallback.get("attempted_targets"))
                if isinstance(item, dict)
            ],
            "selected_accelerator": str(colab_fallback.get("selected_accelerator") or ""),
            "selected_authuser": str(colab_fallback.get("selected_authuser") or ""),
            "attempts": [
                {
                    "accelerator": str(_dict(item).get("accelerator") or ""),
                    "authuser": str(_dict(item).get("authuser") or ""),
                    "ok": _dict(item).get("ok") is True,
                    "blockers": [str(blocker) for blocker in _list(_dict(item).get("blockers"))],
                    "manager_blocker": str(_dict(_dict(item).get("manager")).get("blocker") or ""),
                    "manager_attempt_count": _int(_dict(_dict(item).get("manager")).get("attempt_count")),
                }
                for item in _list(colab_fallback.get("attempts"))
                if isinstance(item, dict)
            ],
            "public_artifact_safe": bool(colab_fallback) and colab_fallback.get("public_artifact_safe") is True,
        },
        "colab_authuser_fallback": {
            "present": bool(authuser_fallback),
            "attempt_count": _int(authuser_fallback.get("attempt_count")),
            "attempted_authusers": [str(item) for item in _list(authuser_fallback.get("attempted_authusers"))],
            "selected_authuser": str(authuser_fallback.get("selected_authuser") or ""),
            "attempts": [
                {
                    "authuser": str(_dict(item).get("authuser") or ""),
                    "ok": _dict(item).get("ok") is True,
                    "blockers": [str(blocker) for blocker in _list(_dict(item).get("blockers"))],
                    "manager_blocker": str(_dict(_dict(item).get("manager")).get("blocker") or ""),
                    "manager_attempt_count": _int(_dict(_dict(item).get("manager")).get("attempt_count")),
                }
                for item in _list(authuser_fallback.get("attempts"))
                if isinstance(item, dict)
            ],
            "public_artifact_safe": bool(authuser_fallback) and authuser_fallback.get("public_artifact_safe") is True,
        },
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True or _dict(report.get("safety")).get("public_artifact_safe") is True,
    }


def summarize_kaggle_gpu_preflight(report: dict[str, Any]) -> dict[str, Any]:
    cleanup = _dict(report.get("cleanup"))
    ready = bool(
        report.get("ok") is True
        and report.get("simultaneous_t4x2_verified") is True
        and _int(report.get("accepted_submission_count")) >= 1
    )
    return {
        "schema": "deepseek_v4_flash_kaggle_gpu_preflight_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "kaggle_cuda_ready": ready,
        "requested_kernel_count": _int(report.get("requested_kernel_count")),
        "accepted_submission_count": _int(report.get("accepted_submission_count")),
        "simultaneous_t4x2_verified": report.get("simultaneous_t4x2_verified") is True,
        "max_observed_running_count": _int(report.get("max_observed_running_count")),
        "accelerator": str(report.get("accelerator") or ""),
        "owner_hash": sha256_short_text(str(report.get("owner") or "")) if report.get("owner") else "",
        "owner_public": False,
        "cleanup_attempted": cleanup.get("attempted") is True,
        "deleted_kernel_count": len(_list(cleanup.get("deleted_refs"))),
        "failed_delete_count": len(_list(cleanup.get("failed_delete_refs"))),
        "private_kernel_payloads_removed": report.get("private_kernel_payloads_removed") is True,
        "evidence_ready": report.get("evidence_ready") is True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_colab_cuda_preflight(report: dict[str, Any]) -> dict[str, Any]:
    devices = [item for item in _list(report.get("devices")) if isinstance(item, dict)]
    ready = bool(
        report.get("ok") is True
        and report.get("colab_cuda_runtime_ready") is True
        and _int(report.get("cuda_device_count")) >= 1
        and report.get("cuda_matmul_ready") is True
    )
    return {
        "schema": "deepseek_v4_flash_colab_cuda_preflight_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "colab_cuda_ready": ready,
        "cuda_available": report.get("cuda_available") is True,
        "cuda_device_count": _int(report.get("cuda_device_count")),
        "cuda_matmul_ready": report.get("cuda_matmul_ready") is True,
        "torch_version": str(report.get("torch_version") or ""),
        "cuda_version": str(report.get("cuda_version") or ""),
        "device_memory_total_mb": [int(_dict(item).get("total_memory_mb") or 0) for item in devices],
        "runtime_proxy_connected": report.get("runtime_proxy_connected") is True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_llama_v4_build_preflight(report: dict[str, Any]) -> dict[str, Any]:
    steps = _dict(report.get("steps"))
    manager = _dict(report.get("manager") or report.get("session_manager_public"))
    worker = _dict(report.get("worker_summary"))
    worker_steps = _dict(_dict(report.get("worker_report")).get("steps"))
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    is_kaggle_report = report.get("schema") == "deepseek_v4_flash_kaggle_llama_v4_build_preflight_v1"
    ready = bool(
        report.get("ok") is True
        or report.get("llama_v4_runtime_build_ready") is True
        or worker.get("worker_ok") is True
    )
    return {
        "schema": "deepseek_v4_flash_llama_v4_build_preflight_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "provider": "kaggle_cuda" if is_kaggle_report else "colab_cuda",
        "llama_v4_runtime_build_ready": ready,
        "repo": str(report.get("repo") or _dict(report.get("runtime")).get("repo_url") or worker.get("repo_url") or ""),
        "branch": str(report.get("branch") or _dict(report.get("runtime")).get("branch") or worker.get("branch") or ""),
        "commit_hash_public": str(report.get("commit_hash_public") or worker.get("commit_hash_public") or ""),
        "cuda_architectures": str(report.get("cuda_architectures") or _dict(report.get("runtime")).get("cuda_architectures") or worker.get("cuda_architectures") or ""),
        "patch_rpc_op_count_guard": report.get("patch_rpc_op_count_guard") is True or _dict(report.get("runtime")).get("patch_rpc_op_count_guard") is True or worker.get("patch_rpc_op_count_guard") is True,
        "patch_rpc_op_count_guard_ok": worker.get("patch_rpc_op_count_guard_ok") is True or _dict(steps.get("patch_rpc_op_count_guard")).get("ok") is True,
        "llama_cli_present": report.get("llama_cli_present") is True or worker.get("llama_cli_present") is True,
        "rpc_server_present": report.get("rpc_server_present") is True or worker.get("rpc_server_present") is True,
        "llama_cli_supports_rpc": report.get("llama_cli_supports_rpc") is True or worker.get("llama_cli_supports_rpc") is True,
        "llama_cli_supports_tensor_split": report.get("llama_cli_supports_tensor_split") is True or worker.get("llama_cli_supports_tensor_split") is True,
        "cmake_configure_ok": _dict(steps.get("cmake_configure")).get("ok") is True or _dict(worker_steps.get("cmake_configure")).get("ok") is True or worker.get("cmake_configure_ok") is True,
        "cmake_build_ok": _dict(steps.get("cmake_build")).get("ok") is True or _dict(worker_steps.get("cmake_build")).get("ok") is True or worker.get("cmake_build_ok") is True,
        "session_manager_ok": manager.get("ok") is True,
        "session_manager_blocker": str(manager.get("blocker") or ""),
        "fresh_kaggle_run_performed": report.get("fresh_kaggle_run_performed") is True,
        "kaggle_kernel_deleted": lifecycle.get("kernel_deleted") is True,
        "private_package_removed": lifecycle.get("private_package_removed") is True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_rpc_hello_diagnostic(report: dict[str, Any]) -> dict[str, Any]:
    worker = _dict(report.get("worker_summary"))
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    return {
        "schema": "deepseek_v4_flash_rpc_hello_diagnostic_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "rpc_hello_diagnostic_ready": report.get("rpc_hello_diagnostic_ready") is True
        or worker.get("rpc_hello_diagnostic_ready") is True,
        "fresh_kaggle_run_performed": report.get("fresh_kaggle_run_performed") is True,
        "server_count": _int(worker.get("server_count")),
        "server_names": [str(item) for item in _list(worker.get("server_names"))],
        "all_servers_alive": worker.get("all_servers_alive") is True,
        "all_rpc_hello_ok": worker.get("all_rpc_hello_ok") is True,
        "requested_accelerator": str(lifecycle.get("requested_accelerator") or ""),
        "kernel_deleted": lifecycle.get("kernel_deleted") is True,
        "private_package_removed": lifecycle.get("private_package_removed") is True,
        "blockers": [str(item) for item in _list(report.get("blockers")) + _list(worker.get("blockers"))],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes")) + _list(worker.get("diagnosis_codes"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True or _dict(report.get("safety")).get("public_artifact_safe") is True,
    }


def summarize_colab_cuda_reacquire_retry(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "deepseek_v4_flash_colab_cuda_reacquire_retry_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "colab_cuda_reacquire_ready": report.get("colab_cuda_reacquire_ready") is True,
        "attempts_completed": _int(report.get("attempts_completed")),
        "attempts_requested": _int(report.get("attempts_requested")),
        "successful_attempt_index": _int(report.get("successful_attempt_index")),
        "accelerators_attempted": [str(item) for item in _list(report.get("accelerators_attempted"))],
        "authusers_attempted": [str(item) for item in _list(report.get("authusers_attempted"))],
        "accelerator": str(report.get("accelerator") or ""),
        "authuser": str(report.get("authuser") or ""),
        "successful_report_path": str(report.get("successful_report_path") or ""),
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "credentials_public": report.get("credentials_public") is True,
        "private_runtime_state_public": report.get("private_runtime_state_public") is True,
    }


def summarize_colab_retry_same_request_auto(report: dict[str, Any]) -> dict[str, Any]:
    retry = _dict(report.get("retry_summary"))
    same = _dict(report.get("same_request_summary"))
    providers = [str(item) for item in _list(report.get("accepted_providers"))]
    return {
        "schema": "deepseek_v4_flash_colab_retry_same_request_auto_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "retry_ready": report.get("retry_ready") is True,
        "same_request_started": report.get("same_request_started") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "generated_token_count": _int(report.get("generated_token_count")),
        "accepted_providers": providers,
        "failure_stage": str(report.get("failure_stage") or ""),
        "retry_summary": {
            "schema": str(retry.get("schema") or ""),
            "ok": retry.get("ok") is True,
            "colab_cuda_reacquire_ready": retry.get("colab_cuda_reacquire_ready") is True,
            "attempts_completed": _int(retry.get("attempts_completed")),
            "accelerator": str(retry.get("accelerator") or ""),
            "authuser": str(retry.get("authuser") or ""),
            "blockers": [str(item) for item in _list(retry.get("blockers"))],
            "public_artifact_safe": retry.get("public_artifact_safe") is True,
        },
        "same_request_summary": {
            "schema": str(same.get("schema") or ""),
            "ok": same.get("ok") is True,
            "same_request_decode_verified": same.get("same_request_decode_verified") is True,
            "generated_token_count": _int(same.get("generated_token_count")),
            "failure_stage": str(same.get("failure_stage") or ""),
            "blockers": [str(item) for item in _list(same.get("blockers"))],
            "public_artifact_safe": same.get("public_artifact_safe") is True,
        },
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "credentials_public": report.get("credentials_public") is True,
        "private_runtime_state_public": report.get("private_runtime_state_public") is True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source_resolver_report) if args.source_resolver_report else Path()
    single_path = Path(args.single_kernel_probe_report) if args.single_kernel_probe_report else Path()
    same_path = Path(args.same_request_report) if args.same_request_report else Path()
    kaggle_gpu_path = Path(args.kaggle_gpu_preflight_report) if args.kaggle_gpu_preflight_report else Path()
    colab_cuda_path = Path(args.colab_cuda_preflight_report) if args.colab_cuda_preflight_report else Path()
    llama_v4_build_path = Path(args.llama_v4_build_preflight_report) if args.llama_v4_build_preflight_report else Path()
    rpc_hello_path = Path(args.rpc_hello_diagnostic_report) if args.rpc_hello_diagnostic_report else Path()
    colab_accelerator_probe_paths = [Path(item) for item in args.colab_accelerator_probe_report if item]
    colab_cuda_reacquire_retry_path = Path(args.colab_cuda_reacquire_retry_report) if args.colab_cuda_reacquire_retry_report else Path()
    colab_retry_same_request_auto_path = Path(args.colab_retry_same_request_auto_report) if args.colab_retry_same_request_auto_report else Path()
    source = summarize_source(load_json(source_path))
    single = summarize_single_kernel_probe(load_json(single_path))
    same = summarize_same_request(load_json(same_path))
    kaggle_gpu = summarize_kaggle_gpu_preflight(load_json(kaggle_gpu_path))
    colab_cuda = summarize_colab_cuda_preflight(load_json(colab_cuda_path))
    llama_v4_build = summarize_llama_v4_build_preflight(load_json(llama_v4_build_path))
    rpc_hello = summarize_rpc_hello_diagnostic(load_json(rpc_hello_path))
    colab_cuda_reacquire_retry = summarize_colab_cuda_reacquire_retry(load_json(colab_cuda_reacquire_retry_path))
    colab_retry_same_request_auto = summarize_colab_retry_same_request_auto(load_json(colab_retry_same_request_auto_path))
    same_success = same.get("same_request_decode_verified") is True
    blockers = sorted({
        *[str(item) for item in _list(source.get("blockers"))],
        *[str(item) for item in _list(source.get("recommended_blockers"))],
        *[str(item) for item in _list(single.get("blockers"))],
        *[str(item) for item in _list(same.get("blockers"))],
        *[str(item) for item in _list(kaggle_gpu.get("blockers"))],
        *[str(item) for item in _list(colab_cuda.get("blockers"))],
        *[str(item) for item in _list(llama_v4_build.get("blockers"))],
        *[str(item) for item in _list(rpc_hello.get("blockers"))],
        *[str(item) for item in _list(colab_cuda_reacquire_retry.get("blockers"))],
        *[str(item) for item in _list(colab_retry_same_request_auto.get("blockers"))],
    })
    if not source.get("resolver_ready"):
        blockers.append("deepseek_v4_flash_quantized_source_not_ready")
    if kaggle_gpu.get("present") and not kaggle_gpu.get("kaggle_cuda_ready"):
        blockers.append("kaggle_cuda_preflight_not_ready")
    if colab_cuda.get("present") and not colab_cuda.get("colab_cuda_ready"):
        blockers.append("colab_cuda_preflight_not_ready")
    if llama_v4_build.get("present") and not llama_v4_build.get("llama_v4_runtime_build_ready"):
        blockers.append("deepseek_v4_flash_llama_v4_runtime_build_not_ready")
    if llama_v4_build.get("present") and llama_v4_build.get("llama_v4_runtime_build_ready"):
        if llama_v4_build.get("provider") == "kaggle_cuda" and not llama_v4_build.get("kaggle_kernel_deleted"):
            blockers.append("deepseek_v4_flash_llama_v4_runtime_build_cleanup_missing")
    if rpc_hello.get("present") and not rpc_hello.get("rpc_hello_diagnostic_ready"):
        blockers.append("deepseek_v4_flash_rpc_hello_diagnostic_not_ready")
    if rpc_hello.get("present") and rpc_hello.get("rpc_hello_diagnostic_ready"):
        if not rpc_hello.get("kernel_deleted") or not rpc_hello.get("private_package_removed"):
            blockers.append("deepseek_v4_flash_rpc_hello_diagnostic_cleanup_missing")
    if colab_cuda_reacquire_retry.get("present") and not colab_cuda_reacquire_retry.get("colab_cuda_reacquire_ready"):
        blockers.append("colab_cuda_reacquire_not_ready")
    if colab_retry_same_request_auto.get("present") and not colab_retry_same_request_auto.get("retry_ready"):
        blockers.append("colab_cuda_reacquire_not_ready")
    if not same_success:
        blockers.append("deepseek_v4_flash_quantized_same_request_decode_not_verified")
    if source.get("recommended_runtime_backend") == "llama_cpp_v4_fork":
        blockers.append("deepseek_v4_flash_requires_v4_aware_llama_cpp_fork")
    blockers = sorted(set(blockers))
    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "deepseek_v4_flash_quantized_swarm_rc_ready": True,
        "objective": "Kaggle/Colab GPU + CPU worker same-request decode for quantized DeepSeek-V4-Flash",
        "model": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "architecture_class": "moe",
            "total_params_b": 284.0,
            "active_params_b": 13.0,
            "quantized": True,
            "selected_quant": source.get("recommended_quant"),
            "selected_repo": source.get("recommended_repo"),
            "selected_size_gb": source.get("recommended_total_size_gb"),
        },
        "success": {
            "same_request_decode_verified": same_success,
            "generated_token_count": _int(same.get("generated_token_count")),
            "required_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
            "accepted_providers": _list(same.get("accepted_providers")),
        },
        "source_resolver": source,
        "single_kernel_probe": single,
        "kaggle_gpu_preflight": kaggle_gpu,
        "colab_cuda_preflight": colab_cuda,
        "llama_v4_build_preflight": llama_v4_build,
        "rpc_hello_diagnostic": rpc_hello,
        "colab_cuda_reacquire_retry": colab_cuda_reacquire_retry,
        "colab_retry_same_request_auto": colab_retry_same_request_auto,
        "same_request": same,
        "blockers": blockers,
        "failure_stage": "" if same_success else failure_stage(
            source,
            single,
            same,
            kaggle_gpu,
            colab_cuda,
            llama_v4_build,
            colab_cuda_reacquire_retry,
            colab_retry_same_request_auto,
        ),
        "diagnosis_codes": [
            "deepseek_v4_flash_quantized_swarm_rc_ready",
            "deepseek_v4_flash_quantized_same_request_decode_verified" if same_success else "deepseek_v4_flash_quantized_same_request_decode_not_verified",
            "deepseek_v4_flash_quantized_sources_ready" if source.get("resolver_ready") else "deepseek_v4_flash_quantized_sources_not_ready",
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
        "artifacts": {},
    }
    leaks = public_redaction_errors(result)
    if leaks:
        result["ok"] = False
        result["deepseek_v4_flash_quantized_swarm_rc_ready"] = False
        result["public_artifact_safe"] = False
        result["safety"]["public_artifact_safe"] = False
        result["redaction_errors"] = leaks
        result["blockers"].append("public_redaction_scan_failed")
    support = {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "source_resolver_report": str(source_path) if source_path else "",
        "single_kernel_probe_report": str(single_path) if single_path else "",
        "same_request_report": str(same_path) if same_path else "",
        "kaggle_gpu_preflight_report": str(kaggle_gpu_path) if kaggle_gpu_path else "",
        "colab_cuda_preflight_report": str(colab_cuda_path) if colab_cuda_path else "",
        "llama_v4_build_preflight_report": str(llama_v4_build_path) if llama_v4_build_path else "",
        "rpc_hello_diagnostic_report": str(rpc_hello_path) if rpc_hello_path else "",
        "colab_accelerator_probe_reports": [str(path) for path in colab_accelerator_probe_paths],
        "colab_cuda_reacquire_retry_report": str(colab_cuda_reacquire_retry_path) if colab_cuda_reacquire_retry_path else "",
        "colab_retry_same_request_auto_report": str(colab_retry_same_request_auto_path) if colab_retry_same_request_auto_path else "",
        "public_artifact_safe": True,
    }
    support_path = output_dir / "deepseek_v4_flash_quantized_swarm_rc_support.json"
    summary_path = output_dir / "deepseek_v4_flash_quantized_swarm_rc.json"
    write_json(support_path, support)
    result["artifacts"] = {
        "summary_json": {"kind": "summary_json", "path": summary_path.name, "present": True, "schema": SCHEMA, "ok": bool(result.get("ok"))},
        "support_bundle_json": artifact_entry(support_path, output_dir, kind="support_bundle", schema=SUPPORT_BUNDLE_SCHEMA, ok=True),
    }
    write_json(summary_path, result)
    result["artifacts"]["summary_json"] = artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(result.get("ok")))
    write_json(summary_path, result)
    return result


def failure_stage(
    source: dict[str, Any],
    single: dict[str, Any],
    same: dict[str, Any],
    kaggle_gpu: dict[str, Any],
    colab_cuda: dict[str, Any],
    llama_v4_build: dict[str, Any],
    colab_cuda_reacquire_retry: dict[str, Any],
    colab_retry_same_request_auto: dict[str, Any],
) -> str:
    if not source.get("resolver_ready"):
        return "source_resolver_not_ready"
    if (
        colab_retry_same_request_auto.get("present")
        and not colab_retry_same_request_auto.get("same_request_decode_verified")
        and colab_retry_same_request_auto.get("failure_stage")
    ):
        return str(colab_retry_same_request_auto.get("failure_stage"))
    if colab_cuda_reacquire_retry.get("present") and not colab_cuda_reacquire_retry.get("colab_cuda_reacquire_ready"):
        return "colab_cuda_reacquire_not_ready"
    if same.get("present") and same.get("live_run_performed") and same.get("failure_stage"):
        return str(same.get("failure_stage"))
    if kaggle_gpu.get("present") and not kaggle_gpu.get("kaggle_cuda_ready"):
        if "kaggle_gpu_quota_or_session_limit" in set(kaggle_gpu.get("blockers") or []):
            return "kaggle_cuda_quota_or_session_limit"
        return "kaggle_cuda_preflight_not_ready"
    if colab_cuda.get("present") and not colab_cuda.get("colab_cuda_ready"):
        return "colab_cuda_runtime_not_ready"
    if llama_v4_build.get("present") and not llama_v4_build.get("llama_v4_runtime_build_ready"):
        if llama_v4_build.get("session_manager_blocker"):
            return "llama_v4_runtime_build_colab_execute_failed"
        return "llama_v4_runtime_build_preflight_not_ready"
    if llama_v4_build.get("present") and llama_v4_build.get("llama_v4_runtime_build_ready"):
        if llama_v4_build.get("provider") == "kaggle_cuda" and not llama_v4_build.get("kaggle_kernel_deleted"):
            return "llama_v4_runtime_build_cleanup_missing"
    if not single.get("present") and not same.get("present"):
        return "same_request_live_probe_not_started"
    if same.get("present") and not same.get("live_run_performed"):
        return str(same.get("failure_stage") or "same_request_live_probe_not_started")
    if single.get("present") and not single.get("one_token_generation_verified"):
        return str(single.get("blocked_reason") or "single_kernel_quantized_probe_not_verified")
    return "same_request_decode_not_verified"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-resolver-report", required=True)
    parser.add_argument("--single-kernel-probe-report", default="")
    parser.add_argument("--same-request-report", default="")
    parser.add_argument("--kaggle-gpu-preflight-report", default="")
    parser.add_argument("--colab-cuda-preflight-report", default="")
    parser.add_argument("--llama-v4-build-preflight-report", default="")
    parser.add_argument("--rpc-hello-diagnostic-report", default="")
    parser.add_argument("--colab-accelerator-probe-report", action="append", default=[])
    parser.add_argument("--colab-cuda-reacquire-retry-report", default="")
    parser.add_argument("--colab-retry-same-request-auto-report", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'deepseek_v4_flash_quantized_swarm_rc.json'}")
        print(f"Same-request success: {report['success']['same_request_decode_verified']}")
        if report.get("failure_stage"):
            print(f"Failure stage: {report['failure_stage']}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
