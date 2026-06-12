"""Core technology Alpha helpers for large-model sharded inference.

The code in this module intentionally stays independent of the CLI and of any
particular model runtime.  The first adapter contract targets llama.cpp RPC and
GGUF because it is a practical consumer-device runtime, but the partition,
workload, and benchmark shapes are meant to be reused by later vLLM, SGLang,
TensorRT-LLM, or Petals-like adapters.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ALPHA_SCHEMA = "large_model_shard_alpha_v1"
RUNTIME_ADAPTER_SCHEMA = "large_model_runtime_adapter_v1"
PARTITION_MANIFEST_SCHEMA = "large_model_partition_manifest_v1"
WORKLOAD_CONTRACT_SCHEMA = "large_model_sharded_generate_v1"
BENCHMARK_SCHEMA = "large_model_shard_benchmark_v1"
DEVICE_PROFILE_SCHEMA = "large_model_device_profile_v1"
MODEL_MANIFEST_SCHEMA = "large_model_manifest_v1"
SUPPORT_BUNDLE_SCHEMA = "large_model_shard_alpha_support_bundle_v1"

DEFAULT_MODEL_ID = "gguf-7b-alpha-fixture"
DEFAULT_MODEL_PATH = "models/gguf-7b-alpha.Q4_K_M.gguf"
DEFAULT_QUANTIZATION = "Q4_K_M"
DEFAULT_LAYER_COUNT = 32
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_MODEL_SIZE_MB = 7168
DEFAULT_KV_CACHE_MB = 512
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_PROMPT_HASH = "sha256:prompt-redacted"
DEFAULT_PROMPT_PLACEHOLDER = "PROMPT_FILE"
DEFAULT_LLAMA_CLI = "llama-cli"
DEFAULT_LLAMA_RPC_SERVER = "rpc-server"

PUBLIC_REDACTION_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "Bearer ",
    "lease_token",
    "idempotency_key",
    '"prompt_text":',
    '"raw_prompt":',
    '"generated_text":',
    '"output_text":',
    '"generated_token_ids":',
    '"token_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"kv_cache":',
    '"past_key_values":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)


def stable_hash_payload(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def non_negative_int(value: Any, *, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def non_negative_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, parsed)


def read_json_object(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object in {path}")
    return loaded


def is_controlled_rpc_endpoint(endpoint: str) -> bool:
    """Return true when an endpoint looks limited to local/private routing."""

    endpoint = str(endpoint or "").strip()
    if not endpoint:
        return False
    parsed = urlparse(endpoint if "://" in endpoint else f"tcp://{endpoint}")
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost"} or host.endswith(".local") or host.endswith(".lan"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


def rpc_host_port(endpoint: str) -> str:
    parsed = urlparse(endpoint if "://" in endpoint else f"tcp://{endpoint}")
    host = str(parsed.hostname or "")
    port = int(parsed.port or 50052)
    return f"{host}:{port}"


def endpoint_host(endpoint: str, *, default: str = "127.0.0.1") -> str:
    parsed = urlparse(endpoint if "://" in endpoint else f"tcp://{endpoint}")
    return str(parsed.hostname or default)


def endpoint_port(endpoint: str, *, default: int = 50052) -> int:
    parsed = urlparse(endpoint if "://" in endpoint else f"tcp://{endpoint}")
    return int(parsed.port or default)


def default_device_profiles() -> list[dict[str, Any]]:
    return [
        normalize_device_profile({
            "device_id": "consumer-node-a",
            "role": "remote-rpc-worker",
            "backend": "cuda",
            "rpc_endpoint": "http://127.0.0.1:50052",
            "vram_total_mb": 8192,
            "usable_memory_mb": 6144,
            "ram_total_mb": 32768,
            "latency_ms": 1.5,
            "bandwidth_mbps": 1000,
        }),
        normalize_device_profile({
            "device_id": "consumer-node-b",
            "role": "remote-rpc-worker",
            "backend": "cuda",
            "rpc_endpoint": "http://127.0.0.1:50053",
            "vram_total_mb": 8192,
            "usable_memory_mb": 6144,
            "ram_total_mb": 32768,
            "latency_ms": 1.8,
            "bandwidth_mbps": 1000,
        }),
    ]


def normalize_device_profile(profile: dict[str, Any], *, index: int = 0) -> dict[str, Any]:
    endpoint = str(profile.get("rpc_endpoint") or profile.get("endpoint") or f"http://127.0.0.1:{50052 + index}")
    backend = str(profile.get("backend") or "cpu").strip().lower()
    usable = non_negative_int(
        profile.get("usable_memory_mb")
        or profile.get("vram_free_mb")
        or profile.get("vram_total_mb")
        or profile.get("ram_total_mb"),
        default=0,
    )
    normalized = {
        "schema": DEVICE_PROFILE_SCHEMA,
        "device_id": str(profile.get("device_id") or profile.get("name") or f"device-{index}"),
        "role": str(profile.get("role") or "remote-rpc-worker"),
        "backend": backend,
        "rpc_endpoint": endpoint,
        "controlled_network_endpoint": is_controlled_rpc_endpoint(endpoint),
        "vram_total_mb": non_negative_int(profile.get("vram_total_mb"), default=0),
        "ram_total_mb": non_negative_int(profile.get("ram_total_mb"), default=0),
        "usable_memory_mb": usable,
        "latency_ms": non_negative_float(profile.get("latency_ms"), default=0.0),
        "bandwidth_mbps": non_negative_float(profile.get("bandwidth_mbps"), default=0.0),
        "device_name": str(profile.get("device_name") or profile.get("name") or ""),
    }
    normalized["diagnosis_codes"] = []
    if not normalized["controlled_network_endpoint"]:
        normalized["diagnosis_codes"].append("large_model_rpc_endpoint_not_controlled")
    if normalized["usable_memory_mb"] <= 0:
        normalized["diagnosis_codes"].append("large_model_device_memory_unknown")
    return normalized


def normalize_device_profiles(raw_devices: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    if raw_devices is None:
        return default_device_profiles()
    if isinstance(raw_devices, dict):
        value = raw_devices.get("devices")
        if isinstance(value, list):
            raw_devices = value
        else:
            raw_devices = [raw_devices]
    devices = [
        normalize_device_profile(device if isinstance(device, dict) else {}, index=index)
        for index, device in enumerate(raw_devices)
    ]
    return devices or default_device_profiles()


def build_model_manifest(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_path: str = DEFAULT_MODEL_PATH,
    quantization: str = DEFAULT_QUANTIZATION,
    layer_count: int = DEFAULT_LAYER_COUNT,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    model_size_mb: int = DEFAULT_MODEL_SIZE_MB,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    path = Path(str(model_path or DEFAULT_MODEL_PATH)).expanduser()
    file_size_mb = 0
    if path.is_file():
        try:
            file_size_mb = int(path.stat().st_size // (1024 * 1024))
        except OSError:
            file_size_mb = 0
    resolved_model_size = non_negative_int(
        metadata.get("model_size_mb") or metadata.get("estimated_model_size_mb") or model_size_mb,
        default=DEFAULT_MODEL_SIZE_MB,
    )
    manifest = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "model_id": str(metadata.get("model_id") or model_id or DEFAULT_MODEL_ID),
        "model_path": str(model_path or DEFAULT_MODEL_PATH),
        "model_path_exists": path.is_file(),
        "model_file_size_mb": file_size_mb,
        "model_size_mb": resolved_model_size,
        "quantization": str(metadata.get("quantization") or quantization or DEFAULT_QUANTIZATION),
        "context_length": non_negative_int(metadata.get("context_length") or context_length, default=DEFAULT_CONTEXT_LENGTH),
        "layer_count": non_negative_int(metadata.get("layer_count") or metadata.get("num_layers") or layer_count, default=DEFAULT_LAYER_COUNT),
        "architecture": str(metadata.get("architecture") or metadata.get("model_type") or "llama-like"),
        "artifact_hash": str(metadata.get("artifact_hash") or stable_hash_text(str(model_path or DEFAULT_MODEL_PATH))),
        "metadata_source": "provided-json" if metadata else "cli-defaults",
    }
    if manifest["layer_count"] <= 0:
        manifest["layer_count"] = DEFAULT_LAYER_COUNT
    if manifest["context_length"] <= 0:
        manifest["context_length"] = DEFAULT_CONTEXT_LENGTH
    return manifest


def tensor_split_for_devices(devices: list[dict[str, Any]]) -> list[float]:
    memory_values = [max(1, non_negative_int(device.get("usable_memory_mb"), default=1)) for device in devices]
    total = float(sum(memory_values)) or 1.0
    return [round(value / total, 6) for value in memory_values]


def build_llama_cpp_rpc_adapter(
    *,
    model_manifest: dict[str, Any],
    devices: list[dict[str, Any]],
    llama_cli: str = DEFAULT_LLAMA_CLI,
    rpc_server: str = DEFAULT_LLAMA_RPC_SERVER,
    prompt_placeholder: str = DEFAULT_PROMPT_PLACEHOLDER,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    real_runtime_verified: bool = False,
) -> dict[str, Any]:
    rpc_targets = [rpc_host_port(str(device.get("rpc_endpoint") or "")) for device in devices]
    tensor_split = tensor_split_for_devices(devices)
    worker_commands: list[dict[str, Any]] = []
    for device in devices:
        endpoint = str(device.get("rpc_endpoint") or "")
        command = [
            rpc_server,
            "-H",
            endpoint_host(endpoint),
            "-p",
            str(endpoint_port(endpoint)),
        ]
        if str(device.get("backend") or "").lower() in {"cuda", "metal", "vulkan", "cpu"}:
            command.extend(["--device", str(device.get("backend")).upper() + "0" if device.get("backend") != "cpu" else "CPU"])
        worker_commands.append({
            "device_id": device.get("device_id"),
            "rpc_endpoint": endpoint,
            "controlled_network_endpoint": bool(device.get("controlled_network_endpoint")),
            "command": command,
            "command_line": " ".join(command),
        })
    cli_command = [
        llama_cli,
        "-m",
        str(model_manifest.get("model_path") or DEFAULT_MODEL_PATH),
        "--prompt-file",
        prompt_placeholder,
        "-n",
        str(max(1, int(max_new_tokens or DEFAULT_MAX_NEW_TOKENS))),
        "-c",
        str(model_manifest.get("context_length") or DEFAULT_CONTEXT_LENGTH),
        "-ngl",
        "99",
        "--rpc",
        ",".join(rpc_targets),
        "--tensor-split",
        ",".join(str(value) for value in tensor_split),
    ]
    adapter = {
        "schema": RUNTIME_ADAPTER_SCHEMA,
        "adapter_kind": "llama_cpp_rpc",
        "runtime_backend": "llama.cpp RPC / GGUF",
        "model": model_manifest,
        "worker_endpoints": [
            {
                "device_id": device.get("device_id"),
                "rpc_endpoint": device.get("rpc_endpoint"),
                "controlled_network_endpoint": bool(device.get("controlled_network_endpoint")),
                "backend": device.get("backend"),
                "usable_memory_mb": device.get("usable_memory_mb"),
            }
            for device in devices
        ],
        "rpc_targets": rpc_targets,
        "tensor_split": tensor_split,
        "worker_commands": worker_commands,
        "client_command": cli_command,
        "client_command_line": " ".join(cli_command),
        "real_runtime_verified": bool(real_runtime_verified),
        "real_runtime_required_for_completion": False,
        "controlled_network_only": True,
        "not_public_rpc_safe": True,
        "not_petals_hivemind_production": True,
        "not_large_model_serving_production": True,
        "upstream_runtime_notes": [
            "llama.cpp RPC exposes ggml devices through rpc-server and is treated here as a controlled-network adapter only",
            "do not run this adapter on open or sensitive public networks",
        ],
        "metadata_fields": {
            "model_id": model_manifest.get("model_id"),
            "model_path": model_manifest.get("model_path"),
            "quantization": model_manifest.get("quantization"),
            "context_length": model_manifest.get("context_length"),
            "layer_count": model_manifest.get("layer_count"),
            "artifact_hash": model_manifest.get("artifact_hash"),
        },
    }
    adapter["adapter_hash"] = stable_hash_payload({
        "adapter_kind": adapter["adapter_kind"],
        "model": adapter["metadata_fields"],
        "rpc_targets": adapter["rpc_targets"],
        "tensor_split": adapter["tensor_split"],
    })
    return adapter


def plan_partitions(
    *,
    model_manifest: dict[str, Any],
    devices: list[dict[str, Any]],
    reserved_kv_cache_mb: int = DEFAULT_KV_CACHE_MB,
) -> dict[str, Any]:
    layer_count = max(1, int(model_manifest.get("layer_count") or DEFAULT_LAYER_COUNT))
    model_size_mb = max(1, int(model_manifest.get("model_size_mb") or DEFAULT_MODEL_SIZE_MB))
    per_layer_mb = max(1.0, model_size_mb / float(layer_count))
    remaining = layer_count
    next_layer = 0
    assignments: list[dict[str, Any]] = []
    runnable = True
    blockers: list[str] = []
    normalized_devices = [dict(device) for device in devices]
    if not normalized_devices:
        runnable = False
        blockers.append("large_model_no_devices")
    for index, device in enumerate(normalized_devices):
        if remaining <= 0:
            break
        usable = int(device.get("usable_memory_mb") or 0)
        allocatable = max(0, usable - int(reserved_kv_cache_mb))
        capacity_layers = int(math.floor(allocatable / per_layer_mb)) if per_layer_mb > 0 else 0
        remaining_devices = max(1, len(normalized_devices) - index)
        target_layers = int(math.ceil(remaining / float(remaining_devices)))
        assign_count = min(remaining, max(0, min(capacity_layers, target_layers)))
        if assign_count <= 0 and capacity_layers > 0 and remaining_devices == 1:
            assign_count = min(remaining, capacity_layers)
        layer_end = next_layer + assign_count
        memory_estimate = int(math.ceil(assign_count * per_layer_mb + (reserved_kv_cache_mb if assign_count else 0)))
        assignments.append({
            "device_id": device.get("device_id"),
            "backend": device.get("backend"),
            "rpc_endpoint": device.get("rpc_endpoint"),
            "stage_role": f"layer_range_{len(assignments)}",
            "layer_start": next_layer if assign_count else None,
            "layer_end": layer_end if assign_count else None,
            "layer_count": assign_count,
            "estimated_model_memory_mb": int(math.ceil(assign_count * per_layer_mb)),
            "reserved_kv_cache_mb": int(reserved_kv_cache_mb) if assign_count else 0,
            "estimated_total_memory_mb": memory_estimate,
            "usable_memory_mb": usable,
            "memory_budget_ok": bool(assign_count > 0 and memory_estimate <= usable),
            "latency_ms": device.get("latency_ms"),
            "bandwidth_mbps": device.get("bandwidth_mbps"),
            "controlled_network_endpoint": bool(device.get("controlled_network_endpoint")),
        })
        if assign_count <= 0:
            runnable = False
            blockers.append(f"large_model_device_capacity_missing:{device.get('device_id')}")
        next_layer = layer_end
        remaining -= assign_count
    if remaining > 0:
        runnable = False
        blockers.append("large_model_partition_insufficient_memory")
    uncontrolled = [str(device.get("device_id")) for device in normalized_devices if not device.get("controlled_network_endpoint")]
    if uncontrolled:
        runnable = False
        blockers.append("large_model_rpc_endpoint_not_controlled")
    assigned_layers = sum(int(item.get("layer_count") or 0) for item in assignments)
    manifest = {
        "schema": PARTITION_MANIFEST_SCHEMA,
        "strategy": "layer-range-greedy-v1",
        "runnable": bool(runnable and assigned_layers == layer_count),
        "model": {
            "model_id": model_manifest.get("model_id"),
            "artifact_hash": model_manifest.get("artifact_hash"),
            "quantization": model_manifest.get("quantization"),
            "layer_count": layer_count,
            "context_length": model_manifest.get("context_length"),
            "model_size_mb": model_size_mb,
        },
        "device_count": len(normalized_devices),
        "assigned_layer_count": assigned_layers,
        "unassigned_layer_count": max(0, layer_count - assigned_layers),
        "per_layer_memory_mb": round(per_layer_mb, 3),
        "reserved_kv_cache_mb_per_device": int(reserved_kv_cache_mb),
        "assignments": assignments,
        "memory_budget": {
            "total_usable_memory_mb": sum(int(device.get("usable_memory_mb") or 0) for device in normalized_devices),
            "estimated_required_memory_mb": sum(int(item.get("estimated_total_memory_mb") or 0) for item in assignments),
            "single_device_required_memory_mb": int(math.ceil(model_size_mb + reserved_kv_cache_mb)),
            "single_device_max_usable_memory_mb": max([int(device.get("usable_memory_mb") or 0) for device in normalized_devices] or [0]),
        },
        "network_estimate": {
            "max_latency_ms": max([float(device.get("latency_ms") or 0.0) for device in normalized_devices] or [0.0]),
            "min_bandwidth_mbps": min([float(device.get("bandwidth_mbps") or 0.0) for device in normalized_devices] or [0.0]),
            "controlled_network_only": not uncontrolled,
        },
        "blockers": blockers,
        "diagnosis_codes": [],
        "not_runtime_execution": True,
    }
    if manifest["runnable"]:
        manifest["diagnosis_codes"].extend([
            "large_model_partition_manifest_ready",
            "large_model_layer_range_placement_ready",
            "large_model_memory_budget_ready",
        ])
    else:
        manifest["diagnosis_codes"].extend(blockers or ["large_model_partition_not_ready"])
    manifest["partition_hash"] = stable_hash_payload({
        "strategy": manifest["strategy"],
        "model": manifest["model"],
        "assignments": assignments,
    })
    return manifest


def build_workload_contract(
    *,
    model_manifest: dict[str, Any],
    adapter: dict[str, Any],
    partition_manifest: dict[str, Any],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    contract = {
        "schema": WORKLOAD_CONTRACT_SCHEMA,
        "version": 1,
        "workload_type": "large_model_sharded_generate",
        "adapter_backend": adapter.get("adapter_kind"),
        "runtime_backend": adapter.get("runtime_backend"),
        "model_artifact": {
            "model_id": model_manifest.get("model_id"),
            "model_path_public": False,
            "artifact_hash": model_manifest.get("artifact_hash"),
            "quantization": model_manifest.get("quantization"),
            "context_length": model_manifest.get("context_length"),
            "layer_count": model_manifest.get("layer_count"),
        },
        "partition_manifest": {
            "schema": partition_manifest.get("schema"),
            "partition_hash": partition_manifest.get("partition_hash"),
            "strategy": partition_manifest.get("strategy"),
            "assigned_layer_count": partition_manifest.get("assigned_layer_count"),
            "device_count": partition_manifest.get("device_count"),
        },
        "steps": [
            {
                "name": "prefill",
                "stage": "prefill",
                "input": "prompt_hash_and_cache_policy",
                "output": "cache_handle_and_first_decode_state",
                "raw_prompt_public": False,
            },
            {
                "name": "decode",
                "stage": "decode",
                "input": "cache_handle_and_previous_token",
                "output": "safe_stream_event_or_result_hash",
                "generated_token_ids_public": False,
            },
            {
                "name": "finalize",
                "stage": "generate",
                "input": "decode_state",
                "output": "redacted_generation_summary",
                "raw_generated_text_public": False,
            },
        ],
        "cache": {
            "cache_handle_schema": "large_model_cache_handle_v1",
            "kv_cache_public": False,
            "prefix_cache_metadata_public": True,
            "cache_hit_miss_metrics_public": True,
            "cache_migration_supported": False,
        },
        "serving_readiness": {
            "streaming_event_schema": "large_model_stream_event_v1",
            "bounded_batch_request_schema": "large_model_batch_request_v1",
            "cancellation_field": "cancel_requested",
            "health_aware_route_metadata_schema": "large_model_route_health_v1",
            "max_new_tokens": max(1, int(max_new_tokens or DEFAULT_MAX_NEW_TOKENS)),
        },
        "redaction_policy": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
            "lease_material_public": False,
            "public_artifact_safe": True,
        },
        "boundaries": {
            "controlled_lan_vpn_only": True,
            "not_public_rpc_safe": True,
            "not_petals_hivemind_production": True,
            "not_p2p_nat_traversal": True,
            "not_gpu_marketplace": True,
        },
    }
    contract["contract_hash"] = stable_hash_payload(contract)
    return contract


def imported_real_benchmark(path: str | Path) -> dict[str, Any]:
    payload = read_json_object(path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    required = ["ttft_ms", "tokens_per_second", "memory_peak_mb", "network_bytes_per_token"]
    missing = [name for name in required if metrics.get(name) is None]
    if missing:
        raise ValueError(f"real benchmark report missing required metrics: {', '.join(missing)}")
    return {
        "source": str(path),
        "metrics": metrics,
        "real_runtime_verified": True,
        "fixture": False,
    }


def build_benchmark_report(
    *,
    model_manifest: dict[str, Any],
    adapter: dict[str, Any],
    partition_manifest: dict[str, Any],
    real_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    single_required = int((partition_manifest.get("memory_budget") or {}).get("single_device_required_memory_mb") or 0)
    single_max = int((partition_manifest.get("memory_budget") or {}).get("single_device_max_usable_memory_mb") or 0)
    sharded_required = int((partition_manifest.get("memory_budget") or {}).get("estimated_required_memory_mb") or 0)
    real_verified = bool(real_benchmark and real_benchmark.get("real_runtime_verified"))
    measurement_kind = "real-runtime" if real_verified else "fixture-planning-estimate"
    benchmark = {
        "schema": BENCHMARK_SCHEMA,
        "model_id": model_manifest.get("model_id"),
        "adapter_kind": adapter.get("adapter_kind"),
        "measurement_kind": measurement_kind,
        "real_runtime_verified": real_verified,
        "fixture": not real_verified,
        "single_device_fallback": {
            "path": "single-device-fallback",
            "real_measurement": False,
            "runnable_on_largest_device": bool(single_max >= single_required and single_required > 0),
            "required_memory_mb": single_required,
            "largest_device_usable_memory_mb": single_max,
            "fallback_mode": "cpu-or-offload" if single_max < single_required else "single-device-gpu-possible",
            "ttft_ms": 4200.0,
            "tokens_per_second": 2.1,
            "memory_peak_mb": max(single_required, single_max),
            "network_bytes_per_token": 0,
        },
        "sharded_adapter_path": {
            "path": "llama-cpp-rpc-sharded",
            "real_measurement": real_verified,
            "runnable_by_plan": bool(partition_manifest.get("runnable")),
            "required_memory_mb": sharded_required,
            "device_count": partition_manifest.get("device_count"),
            "ttft_ms": 2800.0,
            "tokens_per_second": 4.8,
            "p50_latency_ms": 210.0,
            "p95_latency_ms": 420.0,
            "memory_peak_mb": max([int(item.get("estimated_total_memory_mb") or 0) for item in partition_manifest.get("assignments") or []] or [0]),
            "network_bytes_per_token": 262144,
            "cache_hits": 0,
            "cache_misses": max(1, int(model_manifest.get("layer_count") or 1)),
        },
        "comparison": {
            "compares_single_device_fallback": True,
            "compares_sharded_adapter_path": True,
            "fixture_expected_sharded_tokens_per_second_gain": 2.286,
            "real_comparison_available": real_verified,
        },
        "correctness_summary": {
            "mode": "hash-or-imported-real-report",
            "baseline_reference": "single-device llama.cpp command when real runtime is available",
            "output_text_public": False,
            "generated_token_ids_public": False,
            "status": "real-runtime-imported" if real_verified else "fixture-contract-only",
        },
        "failure_diagnosis": {
            "diagnosis_codes": [],
            "blockers": [],
            "operator_action": "",
        },
    }
    if real_verified and real_benchmark:
        metrics = dict(real_benchmark.get("metrics") or {})
        benchmark["sharded_adapter_path"].update({
            "ttft_ms": metrics.get("ttft_ms"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "p50_latency_ms": metrics.get("p50_latency_ms"),
            "p95_latency_ms": metrics.get("p95_latency_ms"),
            "memory_peak_mb": metrics.get("memory_peak_mb"),
            "network_bytes_per_token": metrics.get("network_bytes_per_token"),
            "cache_hits": metrics.get("cache_hits", 0),
            "cache_misses": metrics.get("cache_misses", 0),
            "real_measurement": True,
        })
    codes: list[str] = [
        "large_model_benchmark_harness_ready",
        "large_model_single_device_comparison_ready",
        "large_model_sharded_adapter_comparison_ready",
    ]
    if real_verified:
        codes.append("large_model_real_runtime_verified")
    else:
        codes.append("large_model_real_runtime_not_verified")
        codes.append("large_model_fixture_benchmark_ready")
    if not partition_manifest.get("runnable"):
        codes.append("large_model_partition_not_runnable")
        benchmark["failure_diagnosis"]["blockers"] = list(partition_manifest.get("blockers") or [])
        benchmark["failure_diagnosis"]["operator_action"] = "Fix model/device memory, endpoint, or network blockers before a real run."
    else:
        codes.append("large_model_partition_benchmark_plan_ready")
        benchmark["failure_diagnosis"]["operator_action"] = "Run the generated llama.cpp RPC commands on LAN/VPN hosts, then import the real benchmark JSON."
    benchmark["failure_diagnosis"]["diagnosis_codes"] = codes
    benchmark["diagnosis_codes"] = codes
    benchmark["benchmark_hash"] = stable_hash_payload({
        "model_id": benchmark["model_id"],
        "adapter_kind": benchmark["adapter_kind"],
        "measurement_kind": benchmark["measurement_kind"],
        "single": benchmark["single_device_fallback"],
        "sharded": benchmark["sharded_adapter_path"],
    })
    return benchmark


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.relative_to(output_dir)
        value = str(relative)
    except ValueError:
        value = str(path)
    entry: dict[str, Any] = {
        "kind": kind,
        "path": value,
        "present": path.is_file(),
    }
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    count = len(artifacts)
    present = sum(1 for item in artifacts.values() if isinstance(item, dict) and item.get("present"))
    return {
        "schema": "large_model_shard_alpha_artifact_summary_v1",
        "artifact_count": count,
        "present_artifact_count": present,
        "public_artifact_safe": True,
        "support_bundle": artifacts.get("support_bundle_json", {}).get("path") if artifacts else "",
    }


def public_redaction_errors(value: Any) -> list[str]:
    text = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return [fragment for fragment in PUBLIC_REDACTION_FRAGMENTS if fragment in text]


def render_markdown(report: dict[str, Any]) -> str:
    alpha = report.get("alpha") if isinstance(report.get("alpha"), dict) else {}
    adapter = report.get("runtime_adapter") if isinstance(report.get("runtime_adapter"), dict) else {}
    partition = report.get("partition_manifest") if isinstance(report.get("partition_manifest"), dict) else {}
    benchmark = report.get("benchmark") if isinstance(report.get("benchmark"), dict) else {}
    contract = report.get("workload_contract") if isinstance(report.get("workload_contract"), dict) else {}
    lines = [
        "# CrowdTensor Large-Model Shard Alpha",
        "",
        f"- Schema: `{report.get('schema')}`",
        f"- OK: `{bool(report.get('ok'))}`",
        f"- Alpha ready: `{bool(alpha.get('ready'))}`",
        f"- Real runtime verified: `{bool(report.get('real_runtime_verified'))}`",
        f"- Evidence scope: `{report.get('evidence_scope')}`",
        f"- Adapter: `{adapter.get('adapter_kind')}`",
        f"- Model: `{((adapter.get('model') or {}).get('model_id') if isinstance(adapter.get('model'), dict) else '')}`",
        f"- Partition runnable: `{bool(partition.get('runnable'))}`",
        f"- Assigned layers: `{partition.get('assigned_layer_count')}/{(partition.get('model') or {}).get('layer_count') if isinstance(partition.get('model'), dict) else ''}`",
        f"- Workload contract: `{contract.get('schema')}`",
        f"- Benchmark measurement: `{benchmark.get('measurement_kind')}`",
        "",
        "## Runtime Commands",
        "",
    ]
    for item in adapter.get("worker_commands") or []:
        if not isinstance(item, dict):
            continue
        lines.append(f"- Worker `{item.get('device_id')}`: `{item.get('command_line')}`")
    if adapter.get("client_command_line"):
        lines.append(f"- Client: `{adapter.get('client_command_line')}`")
    lines.extend([
        "",
        "## Safety Boundary",
        "",
        "- This Alpha targets controlled LAN/VPN/local-process operation only.",
        "- It is not production Petals/Hivemind parity, not public RPC security, not NAT traversal, and not a GPU marketplace.",
        "- Public artifacts keep raw prompts, generated text, generated token ids, activations, KV cache, credentials, leases, and idempotency material out of reports.",
        "",
        "## Diagnosis",
        "",
    ])
    for code in report.get("diagnosis_codes") or []:
        lines.append(f"- `{code}`")
    lines.append("")
    return "\n".join(lines)


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "ok": bool(report.get("ok")),
        "real_runtime_verified": bool(report.get("real_runtime_verified")),
        "evidence_scope": report.get("evidence_scope"),
        "diagnosis_codes": report.get("diagnosis_codes") if isinstance(report.get("diagnosis_codes"), list) else [],
        "artifact_summary": report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {},
        "safety": report.get("safety") if isinstance(report.get("safety"), dict) else {},
        "boundary": report.get("boundary") if isinstance(report.get("boundary"), dict) else {},
    }


def build_alpha_report(
    *,
    output_dir: Path,
    model_manifest: dict[str, Any],
    devices: list[dict[str, Any]],
    adapter: dict[str, Any],
    partition_manifest: dict[str, Any],
    workload_contract: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    real_verified = bool(benchmark.get("real_runtime_verified"))
    codes = [
        "large_model_runtime_adapter_ready",
        "large_model_workload_contract_ready",
        "large_model_serving_hooks_ready",
        "large_model_public_artifact_redaction_ready",
        "large_model_core_alpha_boundary_ready",
    ]
    codes.extend(partition_manifest.get("diagnosis_codes") or [])
    codes.extend(benchmark.get("diagnosis_codes") or [])
    if real_verified:
        codes.append("large_model_7b_real_runtime_verified")
    else:
        codes.extend([
            "large_model_7b_plan_ready",
            "large_model_7b_real_runtime_deferred",
        ])
    seen: set[str] = set()
    diagnosis_codes = [code for code in codes if not (code in seen or seen.add(code))]
    alpha_ready = bool(
        adapter.get("schema") == RUNTIME_ADAPTER_SCHEMA
        and workload_contract.get("schema") == WORKLOAD_CONTRACT_SCHEMA
        and benchmark.get("schema") == BENCHMARK_SCHEMA
        and partition_manifest.get("runnable")
    )
    report: dict[str, Any] = {
        "schema": ALPHA_SCHEMA,
        "ok": alpha_ready,
        "alpha": {
            "ready": alpha_ready,
            "milestone": "core-technology-alpha-mvp",
            "core_technology_layer": True,
            "runtime_adapter_ready": adapter.get("schema") == RUNTIME_ADAPTER_SCHEMA,
            "partition_planner_ready": partition_manifest.get("schema") == PARTITION_MANIFEST_SCHEMA,
            "workload_contract_ready": workload_contract.get("schema") == WORKLOAD_CONTRACT_SCHEMA,
            "benchmark_harness_ready": benchmark.get("schema") == BENCHMARK_SCHEMA,
            "ci_safe_fixture_ready": not real_verified,
        },
        "output_dir": str(output_dir),
        "evidence_scope": "real-runtime" if real_verified else "fixture-contract-plan",
        "real_runtime_verified": real_verified,
        "model_manifest": model_manifest,
        "device_profiles": devices,
        "runtime_adapter": adapter,
        "partition_manifest": partition_manifest,
        "workload_contract": workload_contract,
        "benchmark": benchmark,
        "diagnosis_codes": diagnosis_codes,
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
            "real_runtime_claim_requires_real_benchmark": True,
        },
        "boundary": {
            "controlled_lan_vpn_only": True,
            "not_public_rpc_safe": True,
            "not_production_petals_hivemind": True,
            "not_p2p_nat_traversal": True,
            "not_gpu_marketplace": True,
            "not_training_or_finetuning": True,
            "not_full_vllm_sglang_tensorrt": True,
        },
        "next_commands": [
            {
                "label": "start llama.cpp RPC workers on controlled LAN/VPN hosts",
                "commands": [item.get("command_line") for item in adapter.get("worker_commands") or [] if isinstance(item, dict)],
            },
            {
                "label": "run llama.cpp client with RPC workers",
                "command_line": adapter.get("client_command_line"),
            },
            {
                "label": "validate Large-Model Shard Alpha report",
                "command_line": f"python scripts/large_model_shard_alpha_check.py --report {output_dir / 'large_model_shard_alpha.json'}",
            },
        ],
    }
    errors = public_redaction_errors(report)
    if errors:
        report["ok"] = False
        report["alpha"]["ready"] = False
        report["safety"]["public_artifact_safe"] = False
        report.setdefault("errors", []).extend([f"sensitive_public_fragment:{fragment}" for fragment in errors])
        report["diagnosis_codes"].append("large_model_public_artifact_redaction_failed")
    return report
