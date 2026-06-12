"""Inference RC helpers for the CrowdTensor core technology layer.

The RC layer builds on :mod:`crowdtensor.large_model_shard` without changing
the Alpha schemas.  It adds runtime probing, device profiling, a planner v2
view, runner/supervisor evidence, benchmark v2, correctness summaries, serving
hook artifacts, and future-runtime adapter descriptors.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from crowdtensor import large_model_shard as alpha


RC_SCHEMA = "core_technology_inference_rc_v1"
RC_SUPPORT_BUNDLE_SCHEMA = "core_technology_inference_rc_support_bundle_v1"
RUNTIME_ADAPTER_INTERFACE_SCHEMA = "large_model_runtime_adapter_interface_v1"
RUNTIME_ADAPTER_PROBE_SCHEMA = "large_model_runtime_adapter_probe_v2"
DEVICE_PROFILE_SCHEMA = "large_model_device_profile_v2"
PARTITION_MANIFEST_SCHEMA = "large_model_partition_manifest_v2"
RUNNER_RESULT_SCHEMA = "large_model_runner_result_v1"
BENCHMARK_SCHEMA = "large_model_benchmark_v2"
CORRECTNESS_SCHEMA = "large_model_correctness_summary_v1"
SERVING_HOOKS_SCHEMA = "large_model_serving_hooks_v1"
STREAM_EVENT_SCHEMA = "large_model_stream_event_v1"
BOUNDED_BATCH_SCHEMA = "large_model_batch_request_v1"
ROUTE_HEALTH_SCHEMA = "large_model_route_health_v1"

SUPPORTED_RUNTIME = "llama_cpp_rpc"
UNSUPPORTED_RUNTIMES = ("vllm", "sglang", "tensorrt_llm", "petals_like")
DEFAULT_RC_MAX_NEW_TOKENS = 8
MAX_REAL_RUN_TOKENS = 8
MAX_REAL_RUN_TIMEOUT_SECONDS = 20 * 60


def stable_hash_payload(value: Any) -> str:
    return alpha.stable_hash_payload(value)


def stable_hash_text(value: str) -> str:
    return alpha.stable_hash_text(value)


def short_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    return alpha.read_json_object(path)


def read_json_value(value: str, *, expect_list: bool = False) -> Any:
    text = str(value or "").strip()
    if not text:
        return [] if expect_list else {}
    path = Path(text)
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        loaded = json.loads(text)
    if expect_list and not isinstance(loaded, list):
        raise ValueError("expected a JSON list")
    if not expect_list and not isinstance(loaded, dict):
        raise ValueError("expected a JSON object")
    return loaded


def command_result_digest(value: str) -> dict[str, Any]:
    data = value.encode("utf-8", errors="replace")
    return {
        "char_count": len(value),
        "byte_count": len(data),
        "digest": short_digest(data),
        "public_text": False,
    }


def clamp_real_run_tokens(max_new_tokens: int) -> int:
    return max(1, min(MAX_REAL_RUN_TOKENS, int(max_new_tokens or DEFAULT_RC_MAX_NEW_TOKENS)))


def clamp_real_timeout(timeout_seconds: float) -> float:
    try:
        parsed = float(timeout_seconds)
    except (TypeError, ValueError):
        parsed = 120.0
    return max(1.0, min(float(MAX_REAL_RUN_TIMEOUT_SECONDS), parsed))


def executable_probe(binary: str, *, timeout_seconds: float = 3.0) -> dict[str, Any]:
    resolved = shutil.which(str(binary or ""))
    probe = {
        "binary": str(binary or ""),
        "path": resolved or "",
        "available": bool(resolved),
        "version_command": [str(binary or ""), "--version"],
        "version_available": False,
        "version_digest": "",
        "version_public_text": False,
        "diagnosis_codes": [],
    }
    if not resolved:
        probe["diagnosis_codes"].append("large_model_runtime_binary_missing")
        return probe
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        probe["diagnosis_codes"].append("large_model_runtime_version_probe_failed")
        probe["version_error_type"] = type(exc).__name__
        return probe
    combined = "\n".join([completed.stdout or "", completed.stderr or ""]).strip()
    probe["version_returncode"] = completed.returncode
    probe["version_available"] = completed.returncode == 0 and bool(combined)
    probe["version_digest"] = stable_hash_text(combined) if combined else ""
    if probe["version_available"]:
        probe["diagnosis_codes"].append("large_model_runtime_version_probe_ready")
    else:
        probe["diagnosis_codes"].append("large_model_runtime_version_unknown")
    return probe


def endpoint_socket_probe(endpoint: str, *, timeout_seconds: float = 0.25) -> dict[str, Any]:
    endpoint = str(endpoint or "")
    host = alpha.endpoint_host(endpoint, default="")
    port = alpha.endpoint_port(endpoint, default=0)
    started = time.monotonic()
    result = {
        "endpoint": endpoint,
        "host": host,
        "port": port,
        "controlled_network_endpoint": alpha.is_controlled_rpc_endpoint(endpoint),
        "reachable": False,
        "latency_ms": None,
        "diagnosis_codes": [],
    }
    if not result["controlled_network_endpoint"]:
        result["diagnosis_codes"].append("large_model_rpc_endpoint_not_controlled")
    if not host or port <= 0:
        result["diagnosis_codes"].append("large_model_rpc_endpoint_invalid")
        return result
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            pass
    except OSError:
        result["diagnosis_codes"].append("large_model_rpc_endpoint_unreachable")
        return result
    result["reachable"] = True
    result["latency_ms"] = round((time.monotonic() - started) * 1000.0, 3)
    result["diagnosis_codes"].append("large_model_rpc_endpoint_reachable")
    return result


def model_file_probe(model_manifest: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(str(model_manifest.get("model_path") or alpha.DEFAULT_MODEL_PATH)).expanduser()
    result = {
        "model_id": model_manifest.get("model_id"),
        "model_path": str(model_path),
        "model_path_exists": model_path.is_file(),
        "model_path_public": False,
        "suffix": model_path.suffix.lower(),
        "size_mb": 0,
        "metadata_hash": "",
        "gguf_like": model_path.suffix.lower() == ".gguf",
        "diagnosis_codes": [],
    }
    if not model_path.is_file():
        result["diagnosis_codes"].append("large_model_model_file_missing")
        return result
    try:
        stat = model_path.stat()
    except OSError:
        result["diagnosis_codes"].append("large_model_model_file_stat_failed")
        return result
    result["size_mb"] = int(stat.st_size // (1024 * 1024))
    result["metadata_hash"] = stable_hash_payload({
        "path": str(model_path),
        "size": stat.st_size,
        "mtime_ns": getattr(stat, "st_mtime_ns", 0),
        "suffix": model_path.suffix.lower(),
    })
    result["diagnosis_codes"].append("large_model_model_file_probe_ready")
    if result["gguf_like"]:
        result["diagnosis_codes"].append("large_model_gguf_model_file_ready")
    else:
        result["diagnosis_codes"].append("large_model_model_file_not_gguf")
    return result


def build_runtime_adapter_interface(runtime_backend: str = SUPPORTED_RUNTIME) -> dict[str, Any]:
    descriptors: list[dict[str, Any]] = [
        {
            "adapter_kind": SUPPORTED_RUNTIME,
            "status": "supported",
            "schema": alpha.RUNTIME_ADAPTER_SCHEMA,
            "controlled_network_only": True,
            "not_public_rpc_safe": True,
            "capabilities": [
                "gguf_model_path",
                "rpc_worker_endpoints",
                "tensor_split",
                "short_generation",
                "fixture_runner",
                "real_runner_when_runtime_available",
            ],
        }
    ]
    for name in UNSUPPORTED_RUNTIMES:
        descriptors.append({
            "adapter_kind": name,
            "status": "unsupported",
            "diagnosis_codes": ["unsupported_runtime_backend"],
            "operator_action": "Use llama_cpp_rpc for the RC, or implement this adapter behind the same interface.",
        })
    selected = next((item for item in descriptors if item["adapter_kind"] == runtime_backend), None)
    if selected is None:
        selected = {
            "adapter_kind": runtime_backend,
            "status": "unsupported",
            "diagnosis_codes": ["unsupported_runtime_backend"],
        }
    return {
        "schema": RUNTIME_ADAPTER_INTERFACE_SCHEMA,
        "selected_runtime_backend": runtime_backend,
        "selected_supported": selected.get("status") == "supported",
        "descriptors": descriptors,
        "selected_descriptor": selected,
        "diagnosis_codes": ["large_model_runtime_adapter_interface_ready"]
        + ([] if selected.get("status") == "supported" else ["unsupported_runtime_backend"]),
    }


def build_runtime_probe(
    *,
    adapter: dict[str, Any],
    model_manifest: dict[str, Any],
    llama_cli: str,
    llama_rpc_server: str,
    endpoint_timeout_seconds: float = 0.25,
) -> dict[str, Any]:
    client_probe = executable_probe(llama_cli)
    server_probe = executable_probe(llama_rpc_server)
    endpoint_probes = [
        endpoint_socket_probe(str(item.get("rpc_endpoint") or ""), timeout_seconds=endpoint_timeout_seconds)
        for item in adapter.get("worker_endpoints") or []
        if isinstance(item, dict)
    ]
    model_probe = model_file_probe(model_manifest)
    command_validation = {
        "client_has_rpc_flag": "--rpc" in [str(item) for item in adapter.get("client_command") or []],
        "client_has_tensor_split": "--tensor-split" in [str(item) for item in adapter.get("client_command") or []],
        "worker_command_count": len(adapter.get("worker_commands") or []),
        "worker_commands_public_safe": True,
    }
    codes = [
        "large_model_runtime_adapter_probe_ready",
        "large_model_runtime_command_template_ready",
    ]
    for child in [client_probe, server_probe, model_probe, *endpoint_probes]:
        codes.extend(child.get("diagnosis_codes") or [])
    if client_probe.get("available") and server_probe.get("available"):
        codes.append("large_model_runtime_binaries_available")
    else:
        codes.append("large_model_runtime_binaries_missing")
    if all(item.get("reachable") for item in endpoint_probes) and endpoint_probes:
        codes.append("large_model_rpc_endpoints_reachable")
    else:
        codes.append("large_model_rpc_endpoints_not_reachable")
    if model_probe.get("model_path_exists"):
        codes.append("large_model_local_model_probe_ready")
    else:
        codes.append("large_model_local_model_missing")
    seen: set[str] = set()
    return {
        "schema": RUNTIME_ADAPTER_PROBE_SCHEMA,
        "adapter_kind": adapter.get("adapter_kind"),
        "runtime_backend": adapter.get("runtime_backend"),
        "client_binary": client_probe,
        "rpc_server_binary": server_probe,
        "model_file": model_probe,
        "rpc_endpoint_health": endpoint_probes,
        "command_validation": command_validation,
        "controlled_lan_vpn_only": True,
        "not_public_rpc_safe": True,
        "real_runtime_ready": bool(
            client_probe.get("available")
            and server_probe.get("available")
            and model_probe.get("model_path_exists")
            and endpoint_probes
            and all(item.get("reachable") for item in endpoint_probes)
        ),
        "diagnosis_codes": [code for code in codes if not (code in seen or seen.add(code))],
    }


def linux_mem_total_mb() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return 0
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(int(parts[1]) / 1024)
                except ValueError:
                    return 0
    return 0


def gpu_probe() -> list[dict[str, Any]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    devices: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[2]))
            free = int(float(parts[3]))
        except ValueError:
            continue
        devices.append({
            "gpu_index": index,
            "name": parts[1],
            "backend": "cuda",
            "vram_total_mb": total,
            "vram_free_mb": free,
            "usable_memory_mb": max(0, int(free * 0.85)),
        })
    return devices


def build_device_profile_v2(
    *,
    raw_devices: list[dict[str, Any]] | dict[str, Any] | None = None,
    rpc_endpoints: list[str] | None = None,
) -> dict[str, Any]:
    if raw_devices is not None:
        normalized = alpha.normalize_device_profiles(raw_devices)
        devices = []
        for index, item in enumerate(normalized):
            devices.append({
                "schema": DEVICE_PROFILE_SCHEMA,
                "device_id": item.get("device_id"),
                "source": "json-import",
                "role": item.get("role"),
                "backend": item.get("backend"),
                "rpc_endpoint": item.get("rpc_endpoint"),
                "controlled_network_endpoint": item.get("controlled_network_endpoint"),
                "ram_total_mb": item.get("ram_total_mb"),
                "vram_total_mb": item.get("vram_total_mb"),
                "vram_free_mb": item.get("vram_free_mb", item.get("usable_memory_mb")),
                "estimated_usable_memory_mb": item.get("usable_memory_mb"),
                "latency_ms": item.get("latency_ms"),
                "bandwidth_mbps": item.get("bandwidth_mbps"),
                "backend_capabilities": [item.get("backend")] if item.get("backend") else [],
                "diagnosis_codes": item.get("diagnosis_codes") or ["large_model_device_profile_import_ready"],
                "profile_index": index,
            })
        profile = {
            "schema": DEVICE_PROFILE_SCHEMA,
            "source": "json-import",
            "host": platform.node(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count() or 0,
            "ram_total_mb": linux_mem_total_mb(),
            "gpu_devices": [],
            "devices": devices,
            "diagnosis_codes": ["large_model_device_profile_import_ready", "large_model_device_profile_v2_ready"],
        }
        return profile

    gpu_devices = gpu_probe()
    ram_mb = linux_mem_total_mb()
    endpoints = rpc_endpoints or ["http://127.0.0.1:50052"]
    devices: list[dict[str, Any]] = []
    if gpu_devices:
        for index, gpu in enumerate(gpu_devices):
            endpoint = endpoints[min(index, len(endpoints) - 1)]
            devices.append({
                "schema": DEVICE_PROFILE_SCHEMA,
                "device_id": f"{platform.node() or 'localhost'}-cuda-{gpu['gpu_index']}",
                "source": "local-probe",
                "role": "remote-rpc-worker",
                "backend": "cuda",
                "rpc_endpoint": endpoint,
                "controlled_network_endpoint": alpha.is_controlled_rpc_endpoint(endpoint),
                "ram_total_mb": ram_mb,
                "vram_total_mb": gpu.get("vram_total_mb"),
                "vram_free_mb": gpu.get("vram_free_mb"),
                "estimated_usable_memory_mb": gpu.get("usable_memory_mb"),
                "latency_ms": 0.5 if alpha.is_controlled_rpc_endpoint(endpoint) else None,
                "bandwidth_mbps": 1000 if alpha.is_controlled_rpc_endpoint(endpoint) else None,
                "backend_capabilities": ["cuda", "llama_cpp_rpc"],
                "diagnosis_codes": ["large_model_gpu_device_profile_ready"],
            })
    else:
        endpoint = endpoints[0]
        devices.append({
            "schema": DEVICE_PROFILE_SCHEMA,
            "device_id": f"{platform.node() or 'localhost'}-cpu",
            "source": "local-probe",
            "role": "local-fallback-or-fixture",
            "backend": "cpu",
            "rpc_endpoint": endpoint,
            "controlled_network_endpoint": alpha.is_controlled_rpc_endpoint(endpoint),
            "ram_total_mb": ram_mb,
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "estimated_usable_memory_mb": max(1024, int(ram_mb * 0.5)) if ram_mb else 0,
            "latency_ms": 0.1,
            "bandwidth_mbps": 10000,
            "backend_capabilities": ["cpu", "fixture"],
            "diagnosis_codes": ["large_model_cpu_device_profile_ready", "large_model_gpu_not_detected"],
        })
    return {
        "schema": DEVICE_PROFILE_SCHEMA,
        "source": "local-probe",
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
        "ram_total_mb": ram_mb,
        "gpu_devices": gpu_devices,
        "devices": devices,
        "diagnosis_codes": ["large_model_device_profile_v2_ready"]
        + (["large_model_gpu_probe_ready"] if gpu_devices else ["large_model_cpu_fallback_profile_ready"]),
    }


def device_profile_to_alpha_devices(profile: dict[str, Any]) -> list[dict[str, Any]]:
    devices = []
    for index, device in enumerate(profile.get("devices") or []):
        if not isinstance(device, dict):
            continue
        devices.append({
            "device_id": device.get("device_id") or f"device-{index}",
            "role": device.get("role") or "remote-rpc-worker",
            "backend": device.get("backend") or "cpu",
            "rpc_endpoint": device.get("rpc_endpoint") or f"http://127.0.0.1:{50052 + index}",
            "vram_total_mb": device.get("vram_total_mb") or 0,
            "ram_total_mb": device.get("ram_total_mb") or 0,
            "usable_memory_mb": device.get("estimated_usable_memory_mb") or device.get("usable_memory_mb") or 0,
            "latency_ms": device.get("latency_ms") or 0.0,
            "bandwidth_mbps": device.get("bandwidth_mbps") or 0.0,
        })
    return alpha.normalize_device_profiles(devices)


def build_partition_manifest_v2(
    *,
    model_manifest: dict[str, Any],
    devices: list[dict[str, Any]],
    reserved_kv_cache_mb: int = alpha.DEFAULT_KV_CACHE_MB,
) -> dict[str, Any]:
    base = alpha.plan_partitions(
        model_manifest=model_manifest,
        devices=devices,
        reserved_kv_cache_mb=reserved_kv_cache_mb,
    )
    tensor_split = alpha.tensor_split_for_devices(devices)
    layer_count = max(1, int(model_manifest.get("layer_count") or alpha.DEFAULT_LAYER_COUNT))
    model_size = max(1, int(model_manifest.get("model_size_mb") or alpha.DEFAULT_MODEL_SIZE_MB))
    prefill_mb = int(reserved_kv_cache_mb + model_size * 0.08)
    decode_mb = int(reserved_kv_cache_mb + max(1, layer_count) * 4)
    single_required = (base.get("memory_budget") or {}).get("single_device_required_memory_mb", 0)
    single_max = (base.get("memory_budget") or {}).get("single_device_max_usable_memory_mb", 0)
    blocker_details = []
    for code in base.get("blockers") or []:
        blocker_details.append({
            "code": code,
            "summary": "Partition planner could not produce a runnable controlled placement.",
            "operator_action": "Adjust model size, device memory, endpoint control boundary, or worker count.",
        })
    manifest = {
        "schema": PARTITION_MANIFEST_SCHEMA,
        "compat_schema": alpha.PARTITION_MANIFEST_SCHEMA,
        "strategy": "layer-range-tensor-split-v2",
        "base_partition_manifest_v1": base,
        "runnable": bool(base.get("runnable")),
        "model": base.get("model"),
        "assignments": base.get("assignments"),
        "tensor_split_plan": {
            "schema": "llama_cpp_tensor_split_plan_v1",
            "tensor_split": tensor_split,
            "tensor_split_arg": ",".join(str(item) for item in tensor_split),
            "worker_count": len(devices),
        },
        "kv_cache_reservation": {
            "reserved_kv_cache_mb_per_device": reserved_kv_cache_mb,
            "prefix_cache_metadata_public": True,
            "kv_cache_public": False,
        },
        "prefill_decode_memory_estimate": {
            "prefill_workspace_mb": prefill_mb,
            "decode_workspace_mb": decode_mb,
            "estimate_kind": "planner-estimate",
        },
        "fallback_feasibility": {
            "single_device_required_memory_mb": single_required,
            "single_device_max_usable_memory_mb": single_max,
            "single_device_fallback_feasible": bool(single_max and single_max >= single_required),
        },
        "multi_worker_feasibility": {
            "worker_count": len(devices),
            "multi_worker_feasible": bool(base.get("runnable") and len(devices) >= 1),
            "controlled_network_only": bool((base.get("network_estimate") or {}).get("controlled_network_only")),
        },
        "blocker_details": blocker_details,
        "diagnosis_codes": list(base.get("diagnosis_codes") or []) + [
            "large_model_partition_manifest_v2_ready",
            "large_model_tensor_split_plan_ready",
            "large_model_prefill_decode_memory_estimate_ready",
        ],
    }
    manifest["partition_hash"] = stable_hash_payload({
        "schema": manifest["schema"],
        "base": base.get("partition_hash"),
        "tensor_split": tensor_split,
        "kv": manifest["kv_cache_reservation"],
    })
    return manifest


def import_real_run_report(path: str | Path) -> dict[str, Any]:
    payload = read_json_object(path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    required = ["ttft_ms", "tokens_per_second", "wall_time_seconds", "generated_token_count", "output_digest"]
    missing = [name for name in required if metrics.get(name) is None]
    if missing:
        raise ValueError(f"real run report missing required fields: {', '.join(missing)}")
    return {
        "schema": RUNNER_RESULT_SCHEMA,
        "runner_mode": "real-import",
        "ok": True,
        "real_runtime_verified": True,
        "source": str(path),
        "generated_token_count": int(metrics.get("generated_token_count") or 0),
        "max_new_tokens": int(metrics.get("max_new_tokens") or metrics.get("generated_token_count") or 0),
        "output_digest": str(metrics.get("output_digest")),
        "output_text_public": False,
        "generated_token_ids_public": False,
        "metrics": {
            "ttft_ms": float(metrics.get("ttft_ms")),
            "tokens_per_second": float(metrics.get("tokens_per_second")),
            "wall_time_seconds": float(metrics.get("wall_time_seconds")),
            "p50_latency_ms": metrics.get("p50_latency_ms"),
            "p95_latency_ms": metrics.get("p95_latency_ms"),
            "memory_peak_mb": metrics.get("memory_peak_mb"),
            "network_bytes_per_token": metrics.get("network_bytes_per_token"),
            "cache_hits": metrics.get("cache_hits", 0),
            "cache_misses": metrics.get("cache_misses", 0),
        },
        "process_cleanup": {"required": False, "completed": True},
        "diagnosis_codes": ["large_model_real_run_imported", "large_model_runner_real_runtime_verified"],
    }


def build_fixture_runner_result(
    *,
    max_new_tokens: int,
    model_manifest: dict[str, Any],
    partition_manifest: dict[str, Any],
) -> dict[str, Any]:
    token_count = clamp_real_run_tokens(max_new_tokens)
    output_digest = stable_hash_payload({
        "fixture": "large-model-runner",
        "model": model_manifest.get("artifact_hash"),
        "partition": partition_manifest.get("partition_hash"),
        "tokens": token_count,
    })
    return {
        "schema": RUNNER_RESULT_SCHEMA,
        "runner_mode": "fixture",
        "ok": True,
        "real_runtime_verified": False,
        "generated_token_count": token_count,
        "max_new_tokens": token_count,
        "output_digest": output_digest,
        "output_text_public": False,
        "generated_token_ids_public": False,
        "metrics": {
            "ttft_ms": 2500.0,
            "tokens_per_second": 4.0,
            "wall_time_seconds": round(2.5 + token_count / 4.0, 3),
            "p50_latency_ms": 220.0,
            "p95_latency_ms": 450.0,
            "memory_peak_mb": max([int(item.get("estimated_total_memory_mb") or 0) for item in partition_manifest.get("assignments") or []] or [0]),
            "network_bytes_per_token": 262144,
            "cache_hits": 0,
            "cache_misses": token_count,
        },
        "process_cleanup": {"required": False, "completed": True},
        "diagnosis_codes": ["large_model_runner_fixture_ready", "large_model_real_runtime_not_verified"],
    }


def build_plan_runner_result(
    *,
    max_new_tokens: int,
    runtime_probe: dict[str, Any],
    partition_manifest: dict[str, Any],
) -> dict[str, Any]:
    blockers = []
    if not runtime_probe.get("real_runtime_ready"):
        blockers.extend([
            code for code in runtime_probe.get("diagnosis_codes") or []
            if code.endswith("_missing") or code.endswith("_unreachable") or code.endswith("_not_reachable")
        ])
    if not partition_manifest.get("runnable"):
        blockers.append("large_model_partition_not_runnable")
    return {
        "schema": RUNNER_RESULT_SCHEMA,
        "runner_mode": "plan",
        "ok": True,
        "real_runtime_verified": False,
        "generated_token_count": 0,
        "max_new_tokens": clamp_real_run_tokens(max_new_tokens),
        "output_digest": "",
        "output_text_public": False,
        "generated_token_ids_public": False,
        "metrics": {
            "ttft_ms": None,
            "tokens_per_second": None,
            "wall_time_seconds": 0.0,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "memory_peak_mb": None,
            "network_bytes_per_token": None,
            "cache_hits": 0,
            "cache_misses": 0,
        },
        "blockers": blockers,
        "process_cleanup": {"required": False, "completed": True},
        "diagnosis_codes": ["large_model_runner_plan_ready", "large_model_real_runtime_not_verified"] + blockers,
    }


def run_real_supervisor(
    *,
    adapter: dict[str, Any],
    model_manifest: dict[str, Any],
    runtime_probe: dict[str, Any],
    max_new_tokens: int,
    timeout_seconds: float,
    start_workers: bool,
) -> dict[str, Any]:
    max_new_tokens = clamp_real_run_tokens(max_new_tokens)
    timeout_seconds = clamp_real_timeout(timeout_seconds)
    blockers: list[str] = []
    if not runtime_probe.get("client_binary", {}).get("available"):
        blockers.append("large_model_llama_cli_missing")
    if start_workers and not runtime_probe.get("rpc_server_binary", {}).get("available"):
        blockers.append("large_model_llama_rpc_server_missing")
    if not model_manifest.get("model_path_exists"):
        blockers.append("large_model_model_file_missing")
    if not start_workers and not runtime_probe.get("real_runtime_ready"):
        blockers.append("large_model_rpc_endpoints_not_reachable")
    if blockers:
        return {
            "schema": RUNNER_RESULT_SCHEMA,
            "runner_mode": "real",
            "ok": False,
            "real_runtime_verified": False,
            "generated_token_count": 0,
            "max_new_tokens": max_new_tokens,
            "output_digest": "",
            "output_text_public": False,
            "generated_token_ids_public": False,
            "metrics": {"wall_time_seconds": 0.0},
            "blockers": blockers,
            "process_cleanup": {"required": False, "completed": True},
            "diagnosis_codes": ["large_model_runner_real_blocked", *blockers],
        }

    worker_processes: list[subprocess.Popen[str]] = []
    started = time.monotonic()
    cleanup_completed = True
    try:
        if start_workers:
            for item in adapter.get("worker_commands") or []:
                if not isinstance(item, dict):
                    continue
                worker_processes.append(subprocess.Popen(
                    [str(part) for part in item.get("command") or []],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ))
            time.sleep(1.0)
        with tempfile.TemporaryDirectory(prefix="crowdtensor_large_model_prompt_") as tmp:
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text("CrowdTensor validates a controlled large model inference route.\n", encoding="utf-8")
            command = [str(part) for part in adapter.get("client_command") or []]
            command = [str(prompt_file) if part == alpha.DEFAULT_PROMPT_PLACEHOLDER else part for part in command]
            if "-n" in command:
                command[command.index("-n") + 1] = str(max_new_tokens)
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        wall = round(time.monotonic() - started, 3)
    except subprocess.TimeoutExpired:
        return {
            "schema": RUNNER_RESULT_SCHEMA,
            "runner_mode": "real",
            "ok": False,
            "real_runtime_verified": False,
            "generated_token_count": 0,
            "max_new_tokens": max_new_tokens,
            "output_digest": "",
            "output_text_public": False,
            "generated_token_ids_public": False,
            "metrics": {"wall_time_seconds": round(time.monotonic() - started, 3)},
            "blockers": ["large_model_runner_timeout"],
            "process_cleanup": {"required": bool(worker_processes), "completed": cleanup_completed},
            "diagnosis_codes": ["large_model_runner_real_failed", "large_model_runner_timeout"],
        }
    finally:
        for process in worker_processes:
            if process.poll() is None:
                process.terminate()
        for process in worker_processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cleanup_completed = False
                process.kill()

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    generated_estimate = min(max_new_tokens, max(0, len(stdout.split())))
    ok = completed.returncode == 0 and generated_estimate > 0
    tokens_per_second = round(generated_estimate / wall, 4) if wall > 0 and generated_estimate else 0.0
    return {
        "schema": RUNNER_RESULT_SCHEMA,
        "runner_mode": "real",
        "ok": bool(ok),
        "real_runtime_verified": bool(ok),
        "returncode": completed.returncode,
        "generated_token_count": generated_estimate,
        "token_count_source": "stdout_whitespace_estimate",
        "max_new_tokens": max_new_tokens,
        "output_digest": command_result_digest(stdout)["digest"],
        "stderr_digest": command_result_digest(stderr)["digest"] if stderr else "",
        "stdout_public": False,
        "stderr_public": False,
        "output_text_public": False,
        "generated_token_ids_public": False,
        "metrics": {
            "ttft_ms": None,
            "tokens_per_second": tokens_per_second,
            "wall_time_seconds": wall,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "memory_peak_mb": None,
            "network_bytes_per_token": None,
            "cache_hits": 0,
            "cache_misses": generated_estimate,
        },
        "process_cleanup": {"required": bool(worker_processes), "completed": cleanup_completed},
        "diagnosis_codes": ["large_model_runner_real_runtime_verified"] if ok else ["large_model_runner_real_failed"],
    }


def build_runner_result(
    *,
    mode: str,
    adapter: dict[str, Any],
    model_manifest: dict[str, Any],
    runtime_probe: dict[str, Any],
    partition_manifest: dict[str, Any],
    max_new_tokens: int,
    timeout_seconds: float,
    start_workers: bool = False,
    real_run_report: str = "",
) -> dict[str, Any]:
    if real_run_report:
        return import_real_run_report(real_run_report)
    if mode == "fixture":
        return build_fixture_runner_result(
            max_new_tokens=max_new_tokens,
            model_manifest=model_manifest,
            partition_manifest=partition_manifest,
        )
    if mode == "real":
        return run_real_supervisor(
            adapter=adapter,
            model_manifest=model_manifest,
            runtime_probe=runtime_probe,
            max_new_tokens=max_new_tokens,
            timeout_seconds=timeout_seconds,
            start_workers=start_workers,
        )
    return build_plan_runner_result(
        max_new_tokens=max_new_tokens,
        runtime_probe=runtime_probe,
        partition_manifest=partition_manifest,
    )


def build_benchmark_v2(
    *,
    runner_result: dict[str, Any],
    partition_manifest: dict[str, Any],
    device_profile: dict[str, Any],
    imported_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    real_verified = bool(runner_result.get("real_runtime_verified"))
    metrics = dict(runner_result.get("metrics") or {})
    imported_metrics_used = False
    if imported_benchmark:
        metrics.update(imported_benchmark.get("metrics") or {})
        imported_metrics_used = True
    memory_values = [
        int(device.get("estimated_usable_memory_mb") or 0)
        for device in device_profile.get("devices") or []
        if isinstance(device, dict)
    ]
    benchmark = {
        "schema": BENCHMARK_SCHEMA,
        "measurement_kind": "real-runtime" if real_verified else f"{runner_result.get('runner_mode')}-evidence",
        "real_runtime_verified": real_verified,
        "fixture": not real_verified,
        "ttft_ms": metrics.get("ttft_ms"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "p50_latency_ms": metrics.get("p50_latency_ms"),
        "p95_latency_ms": metrics.get("p95_latency_ms"),
        "wall_time_seconds": metrics.get("wall_time_seconds"),
        "memory_summary": {
            "host_ram_total_mb": device_profile.get("ram_total_mb"),
            "worker_usable_memory_mb": memory_values,
            "max_worker_usable_memory_mb": max(memory_values or [0]),
            "memory_peak_mb": metrics.get("memory_peak_mb"),
        },
        "network_summary": {
            "network_bytes_per_token": metrics.get("network_bytes_per_token"),
            "tensor_split": (partition_manifest.get("tensor_split_plan") or {}).get("tensor_split"),
        },
        "cache_summary": {
            "cache_hits": metrics.get("cache_hits", 0),
            "cache_misses": metrics.get("cache_misses", 0),
            "kv_cache_public": False,
        },
        "comparison": {
            "compares_single_device_fallback": True,
            "compares_sharded_adapter_path": True,
            "single_device_fallback_feasible": (partition_manifest.get("fallback_feasibility") or {}).get("single_device_fallback_feasible"),
            "sharded_adapter_feasible": (partition_manifest.get("multi_worker_feasibility") or {}).get("multi_worker_feasible"),
        },
        "failure_diagnosis": {
            "runner_ok": bool(runner_result.get("ok")),
            "blockers": runner_result.get("blockers") or partition_manifest.get("blocker_details") or [],
            "imported_benchmark_without_runner_verification": bool(imported_benchmark and not real_verified),
        },
        "imported_benchmark_used": imported_metrics_used,
        "runner_real_runtime_verified": bool(runner_result.get("real_runtime_verified")),
        "diagnosis_codes": [
            "large_model_benchmark_v2_ready",
            "large_model_single_device_comparison_ready",
            "large_model_sharded_adapter_comparison_ready",
            "large_model_benchmark_import_ready" if imported_metrics_used else "large_model_benchmark_fixture_or_runner_metrics",
        ] + (["large_model_real_runtime_verified"] if real_verified else ["large_model_real_runtime_not_verified"]),
    }
    benchmark["benchmark_hash"] = stable_hash_payload(benchmark)
    return benchmark


def build_correctness_summary(
    *,
    runner_result: dict[str, Any],
    model_manifest: dict[str, Any],
    adapter: dict[str, Any],
    partition_manifest: dict[str, Any],
    baseline_digest: str = "",
) -> dict[str, Any]:
    summary = {
        "schema": CORRECTNESS_SCHEMA,
        "generated_token_count": int(runner_result.get("generated_token_count") or 0),
        "max_new_tokens": int(runner_result.get("max_new_tokens") or 0),
        "output_digest": runner_result.get("output_digest") or "",
        "output_text_public": False,
        "generated_token_ids_public": False,
        "baseline_comparison": {
            "available": bool(baseline_digest),
            "baseline_digest": baseline_digest,
            "match": bool(baseline_digest and baseline_digest == runner_result.get("output_digest")),
        },
        "hash_consistency": {
            "model_artifact_hash": model_manifest.get("artifact_hash"),
            "adapter_hash": adapter.get("adapter_hash"),
            "partition_hash": partition_manifest.get("partition_hash"),
            "runner_result_hash": stable_hash_payload({
                "mode": runner_result.get("runner_mode"),
                "output": runner_result.get("output_digest"),
                "tokens": runner_result.get("generated_token_count"),
            }),
        },
        "diagnosis_codes": ["large_model_correctness_summary_ready"],
    }
    if runner_result.get("real_runtime_verified"):
        summary["diagnosis_codes"].append("large_model_correctness_real_runtime_summary_ready")
    else:
        summary["diagnosis_codes"].append("large_model_correctness_fixture_or_plan_summary_ready")
    return summary


def build_serving_hooks(
    *,
    runner_result: dict[str, Any],
    partition_manifest: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    token_count = max(0, int(runner_result.get("generated_token_count") or 0))
    stream_events = []
    for index in range(token_count):
        stream_events.append({
            "schema": STREAM_EVENT_SCHEMA,
            "event_index": index,
            "event_type": "token" if index + 1 < token_count else "complete",
            "token_count": index + 1,
            "max_new_tokens": max_new_tokens,
            "output_digest": runner_result.get("output_digest"),
            "generated_text_public": False,
            "generated_token_ids_public": False,
        })
    batch_request = {
        "schema": BOUNDED_BATCH_SCHEMA,
        "request_count": 1,
        "max_batch_size": 1,
        "max_new_tokens": max_new_tokens,
        "cancel_requested": False,
        "timeout_seconds": clamp_real_timeout(120),
        "raw_prompt_public": False,
    }
    route_health = {
        "schema": ROUTE_HEALTH_SCHEMA,
        "partition_hash": partition_manifest.get("partition_hash"),
        "worker_count": (partition_manifest.get("tensor_split_plan") or {}).get("worker_count"),
        "healthy": bool(partition_manifest.get("runnable") and runner_result.get("ok")),
        "controlled_network_only": True,
    }
    hooks = {
        "schema": SERVING_HOOKS_SCHEMA,
        "streaming_event_schema": STREAM_EVENT_SCHEMA,
        "streaming_event_emitter_ready": True,
        "sample_stream_events": stream_events,
        "bounded_batch_request_schema": BOUNDED_BATCH_SCHEMA,
        "bounded_batch_request": batch_request,
        "cancellation_field": "cancel_requested",
        "timeout_field": "timeout_seconds",
        "kv_prefix_cache_metadata": {
            "cache_handle_schema": "large_model_cache_handle_v1",
            "prefix_cache_metadata_public": True,
            "kv_cache_public": False,
        },
        "health_aware_route_metadata_schema": ROUTE_HEALTH_SCHEMA,
        "route_health": route_health,
        "diagnosis_codes": [
            "large_model_serving_hooks_v1_ready",
            "large_model_streaming_event_emitter_ready",
            "large_model_bounded_batch_request_ready",
            "large_model_cancellation_timeout_fields_ready",
            "large_model_route_health_metadata_ready",
        ],
    }
    return hooks


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    return alpha.artifact_entry(path, output_dir, kind=kind, schema=schema, ok=ok)


def artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    count = len(artifacts)
    present = sum(1 for item in artifacts.values() if isinstance(item, dict) and item.get("present"))
    return {
        "schema": "core_technology_inference_rc_artifact_summary_v1",
        "artifact_count": count,
        "present_artifact_count": present,
        "public_artifact_safe": True,
        "support_bundle": artifacts.get("support_bundle_json", {}).get("path") if artifacts else "",
        "inspect_first": artifacts.get("summary_markdown", {}).get("path") if artifacts else "",
    }


def public_redaction_errors(value: Any) -> list[str]:
    return alpha.public_redaction_errors(value)


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RC_SUPPORT_BUNDLE_SCHEMA,
        "ok": bool(report.get("ok")),
        "real_runtime_verified": bool(report.get("real_runtime_verified")),
        "real_7b_runtime_verified": bool(report.get("real_7b_runtime_verified")),
        "mode": report.get("mode"),
        "diagnosis_codes": report.get("diagnosis_codes") if isinstance(report.get("diagnosis_codes"), list) else [],
        "blockers": report.get("blockers") if isinstance(report.get("blockers"), list) else [],
        "artifact_summary": report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {},
        "boundary": report.get("boundary") if isinstance(report.get("boundary"), dict) else {},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Core Technology Inference RC",
        "",
        f"- Schema: `{report.get('schema')}`",
        f"- OK: `{bool(report.get('ok'))}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Real runtime verified: `{bool(report.get('real_runtime_verified'))}`",
        f"- Real 7B runtime verified: `{bool(report.get('real_7b_runtime_verified'))}`",
        f"- Runtime backend: `{((report.get('runtime_adapter') or {}).get('adapter_kind') if isinstance(report.get('runtime_adapter'), dict) else '')}`",
        f"- Runner: `{((report.get('runner_result') or {}).get('runner_mode') if isinstance(report.get('runner_result'), dict) else '')}`",
        f"- Benchmark: `{((report.get('benchmark') or {}).get('measurement_kind') if isinstance(report.get('benchmark'), dict) else '')}`",
        "",
        "## Boundary",
        "",
        "- Controlled LAN/VPN/local-process inference RC only.",
        "- Not production Petals/Hivemind parity, not public RPC security, not P2P/NAT traversal, not training or fine-tuning, and not a GPU marketplace.",
        "- Public artifacts redact raw prompts, generated text, generated token ids, activations, KV cache, credentials, leases, idempotency material, private env files, and registries.",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- None")
    lines.extend(["", "## Diagnosis", ""])
    for code in report.get("diagnosis_codes") or []:
        lines.append(f"- `{code}`")
    lines.append("")
    return "\n".join(lines)


def build_rc_report(
    *,
    output_dir: Path,
    mode: str,
    alpha_report: dict[str, Any],
    adapter_interface: dict[str, Any],
    runtime_probe: dict[str, Any],
    device_profile: dict[str, Any],
    runtime_adapter: dict[str, Any],
    partition_manifest: dict[str, Any],
    runner_result: dict[str, Any],
    benchmark: dict[str, Any],
    correctness: dict[str, Any],
    serving_hooks: dict[str, Any],
) -> dict[str, Any]:
    real_verified = bool(runner_result.get("real_runtime_verified"))
    blockers: list[str] = []
    for source in [runtime_probe, partition_manifest, runner_result, benchmark]:
        for item in source.get("blockers") or []:
            blockers.append(str(item))
        for item in source.get("blocker_details") or []:
            if isinstance(item, dict) and item.get("code"):
                blockers.append(str(item.get("code")))
            elif item:
                blockers.append(str(item))
    for code in runtime_probe.get("diagnosis_codes") or []:
        if code in {
            "large_model_runtime_binaries_missing",
            "large_model_rpc_endpoints_not_reachable",
            "large_model_local_model_missing",
            "large_model_rpc_endpoint_not_controlled",
        }:
            blockers.append(code)
    if not partition_manifest.get("runnable"):
        blockers.append("large_model_partition_not_runnable")
    if mode == "real" and not real_verified:
        blockers.append("large_model_real_runtime_not_verified")
    codes = [
        "core_technology_inference_rc_ready",
        "large_model_runtime_adapter_interface_ready",
        "large_model_runtime_adapter_probe_ready",
        "large_model_device_profile_v2_ready",
        "large_model_partition_manifest_v2_ready",
        "large_model_runner_supervisor_ready",
        "large_model_benchmark_v2_ready",
        "large_model_correctness_summary_ready",
        "large_model_serving_hooks_v1_ready",
        "large_model_public_artifact_redaction_ready",
    ]
    for source in [adapter_interface, runtime_probe, device_profile, partition_manifest, runner_result, benchmark, correctness, serving_hooks]:
        codes.extend(source.get("diagnosis_codes") or [])
    if real_verified:
        codes.append("core_technology_real_7b_runtime_verified")
    else:
        codes.extend(["core_technology_real_7b_runtime_not_verified", "core_technology_fixture_or_plan_ready"])
    seen: set[str] = set()
    diagnosis_codes = [code for code in codes if not (code in seen or seen.add(code))]
    ok = bool(
        alpha_report.get("ok")
        and adapter_interface.get("selected_supported")
        and partition_manifest.get("schema") == PARTITION_MANIFEST_SCHEMA
        and runner_result.get("schema") == RUNNER_RESULT_SCHEMA
        and benchmark.get("schema") == BENCHMARK_SCHEMA
        and correctness.get("schema") == CORRECTNESS_SCHEMA
        and serving_hooks.get("schema") == SERVING_HOOKS_SCHEMA
        and (mode != "real" or real_verified)
    )
    report = {
        "schema": RC_SCHEMA,
        "ok": ok,
        "mode": mode,
        "output_dir": str(output_dir),
        "real_runtime_verified": real_verified,
        "real_7b_runtime_verified": real_verified,
        "alpha_report": alpha_report,
        "adapter_interface": adapter_interface,
        "runtime_adapter_probe": runtime_probe,
        "device_profile": device_profile,
        "runtime_adapter": runtime_adapter,
        "partition_manifest": partition_manifest,
        "runner_result": runner_result,
        "benchmark": benchmark,
        "correctness_summary": correctness,
        "serving_readiness_hooks": serving_hooks,
        "blockers": [item for index, item in enumerate(blockers) if item and item not in blockers[:index]],
        "diagnosis_codes": diagnosis_codes,
        "boundary": {
            "inference_core_technology_only": True,
            "controlled_lan_vpn_only": True,
            "not_public_rpc_safe": True,
            "not_production_petals_hivemind": True,
            "not_p2p_nat_traversal": True,
            "not_training_or_finetuning": True,
            "not_gpu_marketplace": True,
            "not_13b_70b_claim": True,
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
    }
    errors = public_redaction_errors(report)
    if errors:
        report["ok"] = False
        report["safety"]["public_artifact_safe"] = False
        report["redaction_errors"] = errors
        report["diagnosis_codes"].append("large_model_public_artifact_redaction_failed")
    return report
