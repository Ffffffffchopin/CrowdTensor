"""Deterministic memory- and performance-aware heterogeneous stage placement."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Iterable

from .heterogeneous_training_manifest import (
    MANIFEST_SCHEMA,
    TPU_MANIFEST_SCHEMA,
    TrainingManifestError,
    stable_hash,
    validate_training_manifest,
)


CAPABILITY_SCHEMA = "crowdtensor_heterogeneous_miner_capability_v1"
TPU_CAPABILITY_SCHEMA = "crowdtensor_heterogeneous_miner_capability_v2"
PLACEMENT_SCHEMA = "crowdtensor_heterogeneous_training_placement_v1"
TPU_PLACEMENT_SCHEMA = "crowdtensor_heterogeneous_training_placement_v2"
RESOURCE_ESTIMATE_SCHEMA = "crowdtensor_heterogeneous_stage_resource_estimate_v1"
TPU_RESOURCE_ESTIMATE_SCHEMA = "crowdtensor_heterogeneous_stage_resource_estimate_v2"

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


class PlacementError(RuntimeError):
    """Raised when no capability-safe complete stage placement exists."""

    def __init__(self, code: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = str(code)
        self.diagnostics = dict(diagnostics or {})


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    return result if math.isfinite(result) else float(default)


def _positive_int(value: Any, *, minimum: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, result)


def discover_heterogeneous_training_capability(
    *,
    miner_id_hash: str = "",
    max_stage_count: int = 0,
    run_microbenchmark: bool = False,
    include_jax_tpu: bool = False,
) -> dict[str, Any]:
    """Discover public-safe CPU/GPU/optional JAX TPU capacity."""

    try:
        import psutil

        memory = psutil.virtual_memory()
        total_memory = int(memory.total)
        available_memory = int(memory.available)
        current_load = max(0.0, min(1.0, float(memory.percent) / 100.0))
        physical_cores = int(psutil.cpu_count(logical=False) or 0)
        logical_cores = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    except ImportError:
        page_size = int(os.sysconf("SC_PAGE_SIZE")) if hasattr(os, "sysconf") else 4096
        pages = int(os.sysconf("SC_PHYS_PAGES")) if hasattr(os, "sysconf") else 0
        total_memory = page_size * pages
        available_memory = total_memory
        current_load = 0.0
        logical_cores = int(os.cpu_count() or 1)
        physical_cores = logical_cores

    cpu_throughput = 0.0
    cpu_benchmark_ms = 0.0
    if run_microbenchmark:
        try:
            import torch

            torch.manual_seed(20260713)
            left = torch.ones((256, 256), dtype=torch.float32)
            right = torch.ones((256, 256), dtype=torch.float32)
            started = time.perf_counter()
            for _ in range(3):
                torch.mm(left, right)
            elapsed = max(1e-9, time.perf_counter() - started)
            cpu_benchmark_ms = elapsed * 1000.0 / 3.0
            cpu_throughput = 3.0 * 2.0 * 256**3 / elapsed
        except (ImportError, RuntimeError):
            pass

    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(int(torch.cuda.device_count())):
                properties = torch.cuda.get_device_properties(index)
                with torch.cuda.device(index):
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                major, minor = torch.cuda.get_device_capability(index)
                dtypes = ["float16", "float32"]
                if bool(torch.cuda.is_bf16_supported()):
                    dtypes.append("bfloat16")
                name = str(properties.name)
                gpus.append(
                    {
                        "device_id": f"cuda:{index}",
                        "device_index": index,
                        "device_name_hash": _sha256_text(name),
                        "total_memory_bytes": int(total_bytes),
                        "free_memory_bytes": int(free_bytes),
                        "compute_capability": f"{major}.{minor}",
                        "supported_dtypes": sorted(dtypes),
                        "throughput_units_per_second": 0.0,
                        "utilization_fraction": max(
                            0.0,
                            min(1.0, 1.0 - float(free_bytes) / max(1, float(total_bytes))),
                        ),
                        "raw_device_name_public": False,
                    }
                )
    except (ImportError, RuntimeError):
        pass

    tpu_groups = (
        _discover_jax_tpu_groups(run_microbenchmark=run_microbenchmark)
        if include_jax_tpu
        else []
    )

    identifier = str(miner_id_hash or "")
    if not identifier.startswith("sha256:"):
        identifier = _sha256_text(
            f"{platform.node()}:{os.getpid()}:{time.time_ns()}"
        )
    capacity = int(max_stage_count or max(1, len(gpus) or min(4, logical_cores)))
    report = {
        "schema": TPU_CAPABILITY_SCHEMA if include_jax_tpu else CAPABILITY_SCHEMA,
        "miner_id_hash": identifier,
        "cpu": {
            "device_id": "cpu",
            "physical_core_count": physical_cores,
            "logical_core_count": logical_cores,
            "total_memory_bytes": total_memory,
            "free_memory_bytes": available_memory,
            "supported_dtypes": ["bfloat16", "float32"],
            "throughput_units_per_second": cpu_throughput,
            "microbenchmark_latency_ms": cpu_benchmark_ms,
            "utilization_fraction": current_load,
        },
        "gpus": gpus,
        "network": {
            "measured_bandwidth_bytes_per_second": 0.0,
            "measured_round_trip_latency_ms": 0.0,
            "measurement_count": 0,
        },
        "stage_profiles": [],
        "current_load_fraction": current_load,
        "max_stage_count": capacity,
        "single_gpu_miner": len(gpus) == 1,
        "multi_gpu_miner": len(gpus) > 1,
        "cpu_stage_supported": True,
        "raw_device_names_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if include_jax_tpu:
        report["tpu_groups"] = tpu_groups
        report["jax_tpu_stage_supported"] = bool(tpu_groups)
    report["content_hash"] = stable_hash(report)
    return validate_miner_capability(report)


def _normalized_tpu_type(device_kinds: Iterable[str]) -> str:
    joined = " ".join(str(item).lower() for item in device_kinds)
    if "v5" in joined:
        return "TPU v5e"
    if "v4" in joined:
        return "TPU v4"
    if "v3" in joined:
        return "TPU v3"
    return "TPU"


def _discover_jax_tpu_groups(*, run_microbenchmark: bool) -> list[dict[str, Any]]:
    """Return one aggregate resource group for the addressable TPU slice."""

    try:
        import jax
    except (ImportError, RuntimeError):
        return []
    try:
        devices = [
            item
            for item in jax.devices()
            if str(getattr(item, "platform", "")).lower() == "tpu"
        ]
    except RuntimeError:
        return []
    if not devices:
        return []
    device_kinds = [str(getattr(item, "device_kind", "TPU")) for item in devices]
    per_device_limits: list[int] = []
    per_device_in_use: list[int] = []
    for device in devices:
        try:
            stats = dict(device.memory_stats() or {})
        except (AttributeError, RuntimeError, TypeError):
            stats = {}
        limit = _positive_int(
            stats.get("bytes_limit")
            or stats.get("bytes_reserved")
            or stats.get("peak_bytes_in_use")
        )
        in_use = _positive_int(stats.get("bytes_in_use"))
        if limit > 0:
            per_device_limits.append(limit)
            per_device_in_use.append(min(limit, in_use))
    memory_source = "jax_memory_stats"
    if len(per_device_limits) != len(devices):
        # Kaggle v5e runtimes do not consistently expose memory_stats(). The
        # documented 16 GiB/core capacity is retained with an explicit source.
        per_device_limits = [16 * 1024**3 for _ in devices]
        per_device_in_use = [0 for _ in devices]
        memory_source = "v5e_documented_capacity"
    compile_ms = 0.0
    steady_ms = 0.0
    throughput = 0.0
    if run_microbenchmark:
        try:
            import jax.numpy as jnp

            operation = jax.jit(lambda left, right: left @ right)
            left = jax.device_put(jnp.ones((512, 512), dtype=jnp.bfloat16), devices[0])
            right = jax.device_put(jnp.ones((512, 512), dtype=jnp.bfloat16), devices[0])
            started = time.perf_counter()
            operation(left, right).block_until_ready()
            compile_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            for _ in range(3):
                operation(left, right).block_until_ready()
            elapsed = max(1e-9, time.perf_counter() - started)
            steady_ms = elapsed * 1000.0 / 3.0
            throughput = 3.0 * 2.0 * 512**3 / elapsed
        except (ImportError, RuntimeError, TypeError):
            pass
    accelerator_type = _normalized_tpu_type(device_kinds)
    total_hbm = sum(per_device_limits)
    free_hbm = sum(
        max(0, limit - used)
        for limit, used in zip(per_device_limits, per_device_in_use)
    )
    return [
        {
            "device_id": "jax_tpu:0",
            "runtime_backend": "jax",
            "accelerator_type": accelerator_type,
            "device_kind_hash": stable_hash(sorted(device_kinds)),
            "device_count": len(devices),
            "mesh_axis_names": ["data"],
            "mesh_shape": [len(devices)],
            "total_hbm_bytes": total_hbm,
            "free_hbm_bytes": free_hbm,
            "per_device_hbm_bytes": min(per_device_limits),
            "hbm_measurement_source": memory_source,
            "supported_dtypes": ["bfloat16", "float32"],
            "throughput_units_per_second": throughput,
            "compile_microbenchmark_latency_ms": compile_ms,
            "steady_microbenchmark_latency_ms": steady_ms,
            "utilization_fraction": max(
                0.0, min(1.0, 1.0 - float(free_hbm) / max(1.0, float(total_hbm)))
            ),
            "all_devices_addressable": len(devices) > 0,
            "raw_device_names_public": False,
        }
    ]


def validate_miner_capability(value: Any) -> dict[str, Any]:
    schema = str(value.get("schema") or "") if isinstance(value, dict) else ""
    if schema not in {CAPABILITY_SCHEMA, TPU_CAPABILITY_SCHEMA}:
        raise ValueError("heterogeneous_miner_capability_schema_invalid")
    miner_id_hash = str(value.get("miner_id_hash") or "")
    if not miner_id_hash.startswith("sha256:"):
        raise ValueError("heterogeneous_miner_capability_identity_invalid")
    cpu = dict(value.get("cpu") or {})
    total_memory = _positive_int(cpu.get("total_memory_bytes"), minimum=1)
    free_memory = _positive_int(cpu.get("free_memory_bytes"), minimum=0)
    if free_memory > total_memory:
        raise ValueError("heterogeneous_miner_cpu_memory_invalid")
    cpu_dtypes = sorted({str(item).lower() for item in cpu.get("supported_dtypes") or []})
    if not cpu_dtypes or not set(cpu_dtypes).issubset(DTYPE_BYTES):
        raise ValueError("heterogeneous_miner_cpu_dtypes_invalid")
    canonical_cpu = {
        "device_id": "cpu",
        "physical_core_count": _positive_int(cpu.get("physical_core_count")),
        "logical_core_count": _positive_int(cpu.get("logical_core_count"), minimum=1),
        "total_memory_bytes": total_memory,
        "free_memory_bytes": free_memory,
        "supported_dtypes": cpu_dtypes,
        "throughput_units_per_second": max(
            0.0, _finite_float(cpu.get("throughput_units_per_second"))
        ),
        "microbenchmark_latency_ms": max(
            0.0, _finite_float(cpu.get("microbenchmark_latency_ms"))
        ),
        "utilization_fraction": max(
            0.0,
            min(1.0, _finite_float(cpu.get("utilization_fraction"))),
        ),
    }
    canonical_gpus = []
    seen_devices: set[str] = set()
    for raw in value.get("gpus") or []:
        if not isinstance(raw, dict):
            raise ValueError("heterogeneous_miner_gpu_invalid")
        device_id = str(raw.get("device_id") or "")
        if not device_id.startswith("cuda:") or device_id in seen_devices:
            raise ValueError("heterogeneous_miner_gpu_device_id_invalid")
        seen_devices.add(device_id)
        gpu_total = _positive_int(raw.get("total_memory_bytes"), minimum=1)
        gpu_free = _positive_int(raw.get("free_memory_bytes"))
        if gpu_free > gpu_total:
            raise ValueError("heterogeneous_miner_gpu_memory_invalid")
        dtypes = sorted({str(item).lower() for item in raw.get("supported_dtypes") or []})
        if not dtypes or not set(dtypes).issubset(DTYPE_BYTES):
            raise ValueError("heterogeneous_miner_gpu_dtypes_invalid")
        name_hash = str(raw.get("device_name_hash") or "")
        if not name_hash.startswith("sha256:"):
            raise ValueError("heterogeneous_miner_gpu_name_hash_invalid")
        canonical_gpus.append(
            {
                "device_id": device_id,
                "device_index": _positive_int(raw.get("device_index")),
                "device_name_hash": name_hash,
                "total_memory_bytes": gpu_total,
                "free_memory_bytes": gpu_free,
                "compute_capability": str(raw.get("compute_capability") or "unknown"),
                "supported_dtypes": dtypes,
                "throughput_units_per_second": max(
                    0.0, _finite_float(raw.get("throughput_units_per_second"))
                ),
                "utilization_fraction": max(
                    0.0,
                    min(1.0, _finite_float(raw.get("utilization_fraction"))),
                ),
                "raw_device_name_public": False,
            }
        )
    canonical_gpus.sort(key=lambda item: (item["device_index"], item["device_id"]))
    canonical_tpu_groups = []
    if schema == TPU_CAPABILITY_SCHEMA:
        seen_tpu_devices: set[str] = set()
        for raw in value.get("tpu_groups") or []:
            if not isinstance(raw, dict):
                raise ValueError("heterogeneous_miner_tpu_group_invalid")
            device_id = str(raw.get("device_id") or "")
            if not device_id.startswith("jax_tpu:") or device_id in seen_tpu_devices:
                raise ValueError("heterogeneous_miner_tpu_device_id_invalid")
            seen_tpu_devices.add(device_id)
            device_count = _positive_int(raw.get("device_count"), minimum=1)
            mesh_axis_names = [str(item) for item in raw.get("mesh_axis_names") or []]
            mesh_shape = [_positive_int(item, minimum=1) for item in raw.get("mesh_shape") or []]
            if (
                not mesh_axis_names
                or len(mesh_axis_names) != len(mesh_shape)
                or math.prod(mesh_shape) != device_count
                or len(set(mesh_axis_names)) != len(mesh_axis_names)
            ):
                raise ValueError("heterogeneous_miner_tpu_mesh_invalid")
            total_hbm = _positive_int(raw.get("total_hbm_bytes"), minimum=1)
            free_hbm = _positive_int(raw.get("free_hbm_bytes"))
            per_device_hbm = _positive_int(
                raw.get("per_device_hbm_bytes"), minimum=1
            )
            if free_hbm > total_hbm or per_device_hbm * device_count > total_hbm:
                raise ValueError("heterogeneous_miner_tpu_hbm_invalid")
            dtypes = sorted(
                {str(item).lower() for item in raw.get("supported_dtypes") or []}
            )
            if not dtypes or not set(dtypes).issubset(DTYPE_BYTES):
                raise ValueError("heterogeneous_miner_tpu_dtypes_invalid")
            device_kind_hash = str(raw.get("device_kind_hash") or "")
            if not device_kind_hash.startswith("sha256:"):
                raise ValueError("heterogeneous_miner_tpu_kind_hash_invalid")
            if str(raw.get("runtime_backend") or "") != "jax":
                raise ValueError("heterogeneous_miner_tpu_runtime_invalid")
            canonical_tpu_groups.append(
                {
                    "device_id": device_id,
                    "runtime_backend": "jax",
                    "accelerator_type": str(raw.get("accelerator_type") or "TPU"),
                    "device_kind_hash": device_kind_hash,
                    "device_count": device_count,
                    "mesh_axis_names": mesh_axis_names,
                    "mesh_shape": mesh_shape,
                    "total_hbm_bytes": total_hbm,
                    "free_hbm_bytes": free_hbm,
                    "per_device_hbm_bytes": per_device_hbm,
                    "hbm_measurement_source": str(
                        raw.get("hbm_measurement_source") or "unknown"
                    ),
                    "supported_dtypes": dtypes,
                    "throughput_units_per_second": max(
                        0.0,
                        _finite_float(raw.get("throughput_units_per_second")),
                    ),
                    "compile_microbenchmark_latency_ms": max(
                        0.0,
                        _finite_float(raw.get("compile_microbenchmark_latency_ms")),
                    ),
                    "steady_microbenchmark_latency_ms": max(
                        0.0,
                        _finite_float(raw.get("steady_microbenchmark_latency_ms")),
                    ),
                    "utilization_fraction": max(
                        0.0,
                        min(1.0, _finite_float(raw.get("utilization_fraction"))),
                    ),
                    "all_devices_addressable": bool(
                        raw.get("all_devices_addressable", False)
                    ),
                    "raw_device_names_public": False,
                }
            )
        canonical_tpu_groups.sort(key=lambda item: item["device_id"])
    elif value.get("tpu_groups"):
        raise ValueError("heterogeneous_miner_tpu_capability_schema_required")

    profiles = []
    for raw in value.get("stage_profiles") or []:
        if not isinstance(raw, dict):
            raise ValueError("heterogeneous_miner_stage_profile_invalid")
        samples = _positive_int(raw.get("sample_count"))
        forward_ms = max(0.0, _finite_float(raw.get("forward_latency_ms")))
        backward_ms = max(0.0, _finite_float(raw.get("backward_latency_ms")))
        if samples < 1 or forward_ms + backward_ms <= 0:
            raise ValueError("heterogeneous_miner_stage_profile_invalid")
        profile = {
                "stage_id": _positive_int(raw.get("stage_id")),
                "device_id": str(raw.get("device_id") or ""),
                "forward_latency_ms": forward_ms,
                "backward_latency_ms": backward_ms,
                "peak_memory_bytes": _positive_int(raw.get("peak_memory_bytes")),
                "sample_count": samples,
                "measured_at": max(0.0, _finite_float(raw.get("measured_at"))),
            }
        if schema == TPU_CAPABILITY_SCHEMA:
            profile.update(
                {
                    "compile_latency_ms": max(
                        0.0, _finite_float(raw.get("compile_latency_ms"))
                    ),
                    "steady_forward_latency_ms": max(
                        0.0,
                        _finite_float(
                            raw.get("steady_forward_latency_ms"),
                            default=forward_ms,
                        ),
                    ),
                    "steady_backward_latency_ms": max(
                        0.0,
                        _finite_float(
                            raw.get("steady_backward_latency_ms"),
                            default=backward_ms,
                        ),
                    ),
                }
            )
        profiles.append(profile)
    network = dict(value.get("network") or {})
    canonical = {
        "schema": schema,
        "miner_id_hash": miner_id_hash,
        "cpu": canonical_cpu,
        "gpus": canonical_gpus,
        "network": {
            "measured_bandwidth_bytes_per_second": max(
                0.0,
                _finite_float(network.get("measured_bandwidth_bytes_per_second")),
            ),
            "measured_round_trip_latency_ms": max(
                0.0, _finite_float(network.get("measured_round_trip_latency_ms"))
            ),
            "measurement_count": _positive_int(network.get("measurement_count")),
        },
        "stage_profiles": sorted(
            profiles, key=lambda item: (item["stage_id"], item["device_id"])
        ),
        "current_load_fraction": max(
            0.0,
            min(1.0, _finite_float(value.get("current_load_fraction"))),
        ),
        "max_stage_count": _positive_int(value.get("max_stage_count"), minimum=1),
        "single_gpu_miner": len(canonical_gpus) == 1,
        "multi_gpu_miner": len(canonical_gpus) > 1,
        "cpu_stage_supported": bool(value.get("cpu_stage_supported", True)),
        "raw_device_names_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if schema == TPU_CAPABILITY_SCHEMA:
        canonical["tpu_groups"] = canonical_tpu_groups
        canonical["jax_tpu_stage_supported"] = bool(canonical_tpu_groups)
    supplied_hash = str(value.get("content_hash") or "")
    canonical["content_hash"] = stable_hash(canonical)
    if supplied_hash and supplied_hash != canonical["content_hash"]:
        raise ValueError("heterogeneous_miner_capability_content_hash_mismatch")
    return canonical


def _lora_parameters_per_layer(manifest: dict[str, Any]) -> int:
    model = manifest["model"]
    rank = int(manifest["lora"]["rank"])
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    kv_width = hidden * kv_heads // heads
    dimensions = {
        "q_proj": (hidden, hidden),
        "k_proj": (hidden, kv_width),
        "v_proj": (hidden, kv_width),
        "o_proj": (hidden, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    count = 0
    for target in manifest["lora"]["target_modules"]:
        try:
            input_size, output_size = dimensions[str(target)]
        except KeyError as exc:
            raise TrainingManifestError(
                "heterogeneous_manifest_lora_target_unsupported"
            ) from exc
        count += rank * (input_size + output_size)
    return count


def estimate_stage_resources(
    manifest: dict[str, Any],
    stage: dict[str, Any] | int,
    *,
    device_type: str,
) -> dict[str, Any]:
    """Estimate stage peak memory with explicit auditable components."""

    canonical = validate_training_manifest(manifest)
    if isinstance(stage, int):
        try:
            spec = canonical["stages"][int(stage)]
        except IndexError as exc:
            raise ValueError("heterogeneous_stage_id_invalid") from exc
    else:
        spec = dict(stage)
    kind = str(device_type).lower()
    if kind not in {"cpu", "cuda", "jax_tpu"}:
        raise ValueError("heterogeneous_stage_device_type_invalid")
    if kind not in spec["allowed_device_types"]:
        raise ValueError("heterogeneous_stage_device_type_not_allowed")
    source_dtype = canonical["model"]["source_dtype"]
    dtype_field = {
        "cpu": "cpu_compute_dtype",
        "cuda": "cuda_compute_dtype",
        "jax_tpu": "jax_tpu_compute_dtype",
    }[kind]
    if dtype_field not in canonical["precision"]:
        raise ValueError("heterogeneous_stage_device_type_not_supported_by_manifest")
    compute_dtype = canonical["precision"][dtype_field]
    source_bytes = DTYPE_BYTES[source_dtype]
    compute_bytes = DTYPE_BYTES[compute_dtype]
    weights = math.ceil(
        int(spec["estimated_weight_bytes"]) * compute_bytes / source_bytes
    )
    trainable_parameters = _lora_parameters_per_layer(canonical) * int(
        spec["layer_count"]
    )
    optimizer_bytes_per_parameter = 8
    optimizer = trainable_parameters * optimizer_bytes_per_parameter
    trainable_parameter_bytes = trainable_parameters * DTYPE_BYTES[
        canonical["precision"]["optimizer_dtype"]
    ]
    trainable_gradient_bytes = trainable_parameter_bytes
    training = canonical["training"]
    activation_elements = (
        int(training["microbatch_size"])
        * int(training["sequence_length"])
        * int(canonical["model"]["hidden_size"])
    )
    activation_one = activation_elements * DTYPE_BYTES[
        canonical["precision"]["boundary_dtype"]
    ]
    cached_activation = activation_one * int(training["microbatches_per_step"])
    cached_gradient = cached_activation
    attention_workspace = (
        int(training["microbatch_size"])
        * int(canonical["model"]["num_attention_heads"])
        * int(training["sequence_length"]) ** 2
        * 4
    )
    workspace_fraction = {"cpu": 0.05, "cuda": 0.12, "jax_tpu": 0.16}[kind]
    workspace = max(
        64 * 1024 * 1024,
        math.ceil(weights * workspace_fraction),
        attention_workspace,
    )
    total = sum(
        (
            weights,
            trainable_parameter_bytes,
            trainable_gradient_bytes,
            optimizer,
            cached_activation,
            cached_gradient,
            workspace,
        )
    )
    result = {
        "schema": (
            TPU_RESOURCE_ESTIMATE_SCHEMA
            if kind == "jax_tpu"
            else RESOURCE_ESTIMATE_SCHEMA
        ),
        "manifest_hash": canonical["content_hash"],
        "stage_id": int(spec["stage_id"]),
        "device_type": kind,
        "source_dtype": source_dtype,
        "compute_dtype": compute_dtype,
        "source_weight_bytes": int(spec["estimated_weight_bytes"]),
        "resident_weight_bytes": weights,
        "lora_trainable_parameter_count": trainable_parameters,
        "lora_parameter_bytes": trainable_parameter_bytes,
        "lora_gradient_bytes": trainable_gradient_bytes,
        "optimizer_state_bytes": optimizer,
        "activation_bytes": cached_activation,
        "activation_gradient_bytes": cached_gradient,
        "workspace_bytes": workspace,
        "estimated_peak_bytes": total,
        "microbatch_size": int(training["microbatch_size"]),
        "microbatches_per_step": int(training["microbatches_per_step"]),
        "sequence_length": int(training["sequence_length"]),
        "public_artifact_safe": True,
    }
    if kind == "jax_tpu":
        # A v5e-8 Miner owns one eight-device group. The scheduler budgets
        # aggregate HBM while retaining an auditable even-shard estimate.
        result.update(
            {
                "jax_mesh_device_count": 8,
                "aggregate_hbm_required_bytes": total,
                "estimated_peak_bytes_per_device": math.ceil(total / 8),
                "parameter_sharding_required": True,
            }
        )
    result["content_hash"] = stable_hash(result)
    return result


@dataclass(frozen=True)
class _Device:
    miner_id_hash: str
    device_id: str
    device_type: str
    total_memory_bytes: int
    free_memory_bytes: int
    supported_dtypes: tuple[str, ...]
    throughput_units_per_second: float
    utilization_fraction: float
    max_stage_count: int
    miner_load_fraction: float
    network_bandwidth_bytes_per_second: float
    network_latency_ms: float
    profiles: tuple[dict[str, Any], ...]
    health_score: float = 1.0
    checkpoint_step: int = 0
    telemetry_at: float = 0.0
    consecutive_failures: int = 0
    compile_microbenchmark_latency_ms: float = 0.0
    steady_microbenchmark_latency_ms: float = 0.0
    group_device_count: int = 1
    mesh_axis_names: tuple[str, ...] = ()
    mesh_shape: tuple[int, ...] = ()
    accelerator_type: str = ""

    @property
    def key(self) -> str:
        return f"{self.miner_id_hash}/{self.device_id}"


def _runtime_telemetry(
    values: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, dict):
            continue
        miner_id_hash = str(raw.get("miner_id_hash") or "")
        device_id = str(raw.get("device_id") or "")
        if not miner_id_hash.startswith("sha256:") or not device_id:
            continue
        health_score = max(0.0, min(1.0, _finite_float(raw.get("health_score"), default=1.0)))
        result[f"{miner_id_hash}/{device_id}"] = {
            "health_score": health_score,
            "checkpoint_step": _positive_int(raw.get("checkpoint_step")),
            "telemetry_at": max(0.0, _finite_float(raw.get("reported_at"))),
            "consecutive_failures": _positive_int(raw.get("consecutive_failures")),
            "free_memory_bytes": _positive_int(raw.get("free_memory_bytes")),
            "throughput_units_per_second": max(
                0.0, _finite_float(raw.get("throughput_units_per_second"))
            ),
            "utilization_fraction": max(
                0.0, min(1.0, _finite_float(raw.get("utilization_fraction")))
            ),
            "network_bandwidth_bytes_per_second": max(
                0.0,
                _finite_float(raw.get("network_bandwidth_bytes_per_second")),
            ),
            "network_latency_ms": max(
                0.0, _finite_float(raw.get("network_latency_ms"))
            ),
        }
    return result


def _devices(
    capabilities: Iterable[dict[str, Any]],
    runtime_telemetry: Iterable[dict[str, Any]] = (),
) -> tuple[list[_Device], list[dict[str, Any]]]:
    devices: list[_Device] = []
    canonical_capabilities = []
    telemetry = _runtime_telemetry(runtime_telemetry)
    for raw in capabilities:
        capability = validate_miner_capability(raw)
        canonical_capabilities.append(capability)
        network = capability["network"]

        def device_values(
            device_id: str,
            *,
            free_memory_bytes: int,
            throughput_units_per_second: float,
            utilization_fraction: float,
        ) -> dict[str, Any]:
            observed = telemetry.get(
                f"{capability['miner_id_hash']}/{device_id}", {}
            )
            observed_free = int(observed.get("free_memory_bytes") or 0)
            observed_throughput = float(
                observed.get("throughput_units_per_second") or 0.0
            )
            observed_bandwidth = float(
                observed.get("network_bandwidth_bytes_per_second") or 0.0
            )
            observed_latency = float(observed.get("network_latency_ms") or 0.0)
            return {
                "miner_id_hash": capability["miner_id_hash"],
                "max_stage_count": int(capability["max_stage_count"]),
                "miner_load_fraction": max(
                    float(capability["current_load_fraction"]),
                    float(observed.get("utilization_fraction") or 0.0),
                ),
                "network_bandwidth_bytes_per_second": (
                    observed_bandwidth
                    or float(network["measured_bandwidth_bytes_per_second"])
                ),
                "network_latency_ms": (
                    observed_latency
                    or float(network["measured_round_trip_latency_ms"])
                ),
                "profiles": tuple(capability["stage_profiles"]),
                "free_memory_bytes": observed_free or int(free_memory_bytes),
                "throughput_units_per_second": (
                    observed_throughput or float(throughput_units_per_second)
                ),
                "utilization_fraction": max(
                    float(utilization_fraction),
                    float(observed.get("utilization_fraction") or 0.0),
                ),
                "health_score": float(observed.get("health_score", 1.0)),
                "checkpoint_step": int(observed.get("checkpoint_step") or 0),
                "telemetry_at": float(observed.get("telemetry_at") or 0.0),
                "consecutive_failures": int(
                    observed.get("consecutive_failures") or 0
                ),
            }

        if capability["cpu_stage_supported"]:
            cpu = capability["cpu"]
            devices.append(
                _Device(
                    device_id="cpu",
                    device_type="cpu",
                    total_memory_bytes=int(cpu["total_memory_bytes"]),
                    supported_dtypes=tuple(cpu["supported_dtypes"]),
                    **device_values(
                        "cpu",
                        free_memory_bytes=int(cpu["free_memory_bytes"]),
                        throughput_units_per_second=float(
                            cpu["throughput_units_per_second"]
                        ),
                        utilization_fraction=float(cpu["utilization_fraction"]),
                    ),
                )
            )
        for gpu in capability["gpus"]:
            devices.append(
                _Device(
                    device_id=str(gpu["device_id"]),
                    device_type="cuda",
                    total_memory_bytes=int(gpu["total_memory_bytes"]),
                    supported_dtypes=tuple(gpu["supported_dtypes"]),
                    **device_values(
                        str(gpu["device_id"]),
                        free_memory_bytes=int(gpu["free_memory_bytes"]),
                        throughput_units_per_second=float(
                            gpu["throughput_units_per_second"]
                        ),
                        utilization_fraction=float(gpu["utilization_fraction"]),
                    ),
                )
            )
        for tpu in capability.get("tpu_groups") or []:
            devices.append(
                _Device(
                    device_id=str(tpu["device_id"]),
                    device_type="jax_tpu",
                    total_memory_bytes=int(tpu["total_hbm_bytes"]),
                    supported_dtypes=tuple(tpu["supported_dtypes"]),
                    compile_microbenchmark_latency_ms=float(
                        tpu["compile_microbenchmark_latency_ms"]
                    ),
                    steady_microbenchmark_latency_ms=float(
                        tpu["steady_microbenchmark_latency_ms"]
                    ),
                    group_device_count=int(tpu["device_count"]),
                    mesh_axis_names=tuple(tpu["mesh_axis_names"]),
                    mesh_shape=tuple(int(item) for item in tpu["mesh_shape"]),
                    accelerator_type=str(tpu["accelerator_type"]),
                    **device_values(
                        str(tpu["device_id"]),
                        free_memory_bytes=int(tpu["free_hbm_bytes"]),
                        throughput_units_per_second=float(
                            tpu["throughput_units_per_second"]
                        ),
                        utilization_fraction=float(tpu["utilization_fraction"]),
                    ),
                )
            )
    devices.sort(key=lambda item: (item.miner_id_hash, item.device_type, item.device_id))
    canonical_capabilities.sort(key=lambda item: item["miner_id_hash"])
    return devices, canonical_capabilities


def _device_budget(device: _Device, scheduler: dict[str, Any]) -> tuple[int, int]:
    reserve = max(
        math.ceil(device.total_memory_bytes * float(scheduler["memory_reserve_fraction"])),
        int(
            scheduler[
                {
                    "cuda": "cuda_memory_reserve_bytes",
                    "cpu": "cpu_memory_reserve_bytes",
                    "jax_tpu": "tpu_memory_reserve_bytes",
                }[device.device_type]
            ]
        ),
    )
    return max(0, device.free_memory_bytes - reserve), reserve


def _profile_latency(device: _Device, stage_id: int) -> tuple[float, float, bool]:
    matches = [
        item
        for item in device.profiles
        if int(item["stage_id"]) == int(stage_id)
        and str(item["device_id"]) == device.device_id
    ]
    if matches:
        latest = max(matches, key=lambda item: (item["sample_count"], item["measured_at"]))
        steady = (
            float(latest.get("steady_forward_latency_ms") or latest["forward_latency_ms"])
            + float(latest.get("steady_backward_latency_ms") or latest["backward_latency_ms"])
        )
        return steady, float(latest.get("compile_latency_ms") or 0.0), True
    return 0.0, 0.0, False


def _compute_latency_ms(
    device: _Device, stage: dict[str, Any]
) -> tuple[float, float, bool]:
    measured, compile_ms, present = _profile_latency(
        device, int(stage["stage_id"])
    )
    if present:
        return measured, compile_ms, True
    throughput = float(device.throughput_units_per_second)
    if throughput <= 0:
        throughput = {
            "cuda": 1_500_000_000.0,
            "cpu": 80_000_000.0,
            "jax_tpu": 8_000_000_000.0,
        }[device.device_type]
    steady = float(stage["estimated_compute_units"]) / throughput * 1000.0
    compile_estimate = (
        float(device.compile_microbenchmark_latency_ms)
        if device.device_type == "jax_tpu"
        else 0.0
    )
    return steady, compile_estimate, False


def _transfer_latency_ms(
    previous: _Device | None,
    current: _Device,
    activation_bytes: int,
) -> float:
    if previous is None:
        return 0.0
    if previous.miner_id_hash == current.miner_id_hash:
        bandwidth = 12.0 * 1024**3 if previous.device_id != current.device_id else 100.0 * 1024**3
        return float(activation_bytes) / bandwidth * 1000.0
    bandwidth_values = [
        value
        for value in (
            previous.network_bandwidth_bytes_per_second,
            current.network_bandwidth_bytes_per_second,
        )
        if value > 0
    ]
    bandwidth = min(bandwidth_values) if bandwidth_values else 100.0 * 1024**2
    latency = max(previous.network_latency_ms, current.network_latency_ms, 10.0)
    return latency + float(activation_bytes) / bandwidth * 1000.0


def _mapping_signature(assignments: Iterable[dict[str, Any]]) -> list[tuple[int, str, str]]:
    return [
        (
            int(item["stage_id"]),
            str(item["miner_id_hash"]),
            str(item["device_id"]),
        )
        for item in sorted(assignments, key=lambda value: int(value["stage_id"]))
    ]


def _remaining_stages_feasible(
    remaining_stages: Iterable[dict[str, Any]],
    *,
    devices: list[_Device],
    stage_candidate_audit: dict[int, list[dict[str, Any]]],
    estimates: dict[tuple[int, str], dict[str, Any]],
    budgets: dict[str, tuple[int, int]],
    scheduler: dict[str, Any],
    device_used: dict[str, int],
    miner_stage_count: dict[str, int],
) -> bool:
    """Check that a partial beam state can still place every later stage."""

    stages = {int(item["stage_id"]): item for item in remaining_stages}
    if not stages:
        return True
    eligible_devices = {
        stage_id: [
            device
            for device in devices
            if any(
                row["eligible"]
                and row["miner_id_hash"] == device.miner_id_hash
                and row["device_id"] == device.device_id
                for row in stage_candidate_audit[stage_id]
            )
        ]
        for stage_id in stages
    }
    failed_states: set[
        tuple[
            tuple[int, ...],
            tuple[tuple[str, int], ...],
            tuple[tuple[str, int], ...],
        ]
    ] = set()

    def search(
        remaining_ids: tuple[int, ...],
        used: dict[str, int],
        counts: dict[str, int],
    ) -> bool:
        if not remaining_ids:
            return True
        state_key = (
            remaining_ids,
            tuple(sorted((key, int(value)) for key, value in used.items() if value)),
            tuple(
                sorted((key, int(value)) for key, value in counts.items() if value)
            ),
        )
        if state_key in failed_states:
            return False

        options_by_stage: list[tuple[int, int, list[_Device]]] = []
        for stage_id in remaining_ids:
            options = []
            for device in eligible_devices[stage_id]:
                estimate = estimates[(stage_id, device.device_type)]
                if (
                    int(used.get(device.key, 0))
                    + int(estimate["estimated_peak_bytes"])
                    > budgets[device.key][0]
                ):
                    continue
                miner_limit = min(
                    int(device.max_stage_count),
                    int(scheduler["max_stages_per_miner"]),
                )
                if int(counts.get(device.miner_id_hash, 0)) >= miner_limit:
                    continue
                options.append(device)
            if not options:
                failed_states.add(state_key)
                return False
            options_by_stage.append((len(options), stage_id, options))

        _, stage_id, options = min(
            options_by_stage,
            key=lambda item: (
                item[0],
                item[1],
                [(device.miner_id_hash, device.device_id) for device in item[2]],
            ),
        )
        next_remaining = tuple(item for item in remaining_ids if item != stage_id)
        for device in options:
            estimate_bytes = int(
                estimates[(stage_id, device.device_type)]["estimated_peak_bytes"]
            )
            next_used = {
                **used,
                device.key: int(used.get(device.key, 0)) + estimate_bytes,
            }
            next_counts = {
                **counts,
                device.miner_id_hash: int(counts.get(device.miner_id_hash, 0)) + 1,
            }
            if search(next_remaining, next_used, next_counts):
                return True
        failed_states.add(state_key)
        return False

    return search(
        tuple(sorted(stages)),
        dict(device_used),
        dict(miner_stage_count),
    )


def build_placement_plan(
    manifest: dict[str, Any],
    capabilities: Iterable[dict[str, Any]],
    *,
    previous_plan: dict[str, Any] | None = None,
    reason: str = "initial_placement",
    excluded_devices: Iterable[str] = (),
    runtime_telemetry: Iterable[dict[str, Any]] = (),
    current_checkpoint_step: int = 0,
) -> dict[str, Any]:
    """Build a complete, deterministic and auditable placement plan."""

    canonical = validate_training_manifest(manifest)
    telemetry_rows = [dict(item) for item in runtime_telemetry if isinstance(item, dict)]
    devices, canonical_capabilities = _devices(capabilities, telemetry_rows)
    excluded = {str(item) for item in excluded_devices}
    devices = [item for item in devices if item.key not in excluded]
    if not devices:
        raise PlacementError(
            "heterogeneous_placement_no_devices",
            diagnostics={"excluded_devices": sorted(excluded)},
        )
    if reason not in {
        "initial_placement",
        "miner_joined",
        "miner_left",
        "lease_expired",
        "device_oom",
        "straggler_detected",
        "owner_requested",
        "checkpoint_recovery",
        "health_degraded",
        "performance_rebalance",
        "coordinator_recovery",
    }:
        raise ValueError("heterogeneous_placement_reason_invalid")
    scheduler = canonical["scheduler"]
    required_dtype = {
        "cpu": canonical["precision"]["cpu_compute_dtype"],
        "cuda": canonical["precision"]["cuda_compute_dtype"],
    }
    if canonical["schema"] == TPU_MANIFEST_SCHEMA:
        required_dtype["jax_tpu"] = canonical["precision"][
            "jax_tpu_compute_dtype"
        ]
    estimates = {
        (int(stage["stage_id"]), kind): estimate_stage_resources(
            canonical, stage, device_type=kind
        )
        for stage in canonical["stages"]
        for kind in stage["allowed_device_types"]
    }
    previous_rows = [dict(item) for item in (previous_plan or {}).get("assignments") or []]
    reclaimable_previous_assignment_bytes: dict[str, int] = {}
    device_keys = {device.key for device in devices}
    for item in previous_rows:
        device_key = f"{item.get('miner_id_hash')}/{item.get('device_id')}"
        if device_key not in device_keys:
            continue
        resource = dict(item.get("resource_estimate") or {})
        estimated_peak = _positive_int(resource.get("estimated_peak_bytes"))
        if estimated_peak <= 0:
            estimated_peak = _positive_int(
                estimates.get(
                    (int(item.get("stage_id") or 0), str(item.get("device_type") or "")),
                    {},
                ).get("estimated_peak_bytes")
            )
        reclaimable_previous_assignment_bytes[device_key] = (
            reclaimable_previous_assignment_bytes.get(device_key, 0) + estimated_peak
        )
    budgets: dict[str, tuple[int, int]] = {}
    for device in devices:
        available, reserve = _device_budget(device, scheduler)
        maximum_after_reserve = max(0, int(device.total_memory_bytes) - reserve)
        reclaimable = int(
            reclaimable_previous_assignment_bytes.get(device.key, 0)
        )
        if device.free_memory_bytes <= 0 or device.health_score <= 0.0:
            reclaimable = 0
            reclaimable_previous_assignment_bytes.pop(device.key, None)
        budgets[device.key] = (
            min(maximum_after_reserve, available + reclaimable),
            reserve,
        )
    previous_assignments = {
        int(item["stage_id"]): (
            str(item["miner_id_hash"]),
            str(item["device_id"]),
        )
        for item in (previous_plan or {}).get("assignments") or []
    }
    stage_candidate_audit: dict[int, list[dict[str, Any]]] = {}
    forward_feasibility_rejection_count = 0
    for stage_index, stage in enumerate(canonical["stages"]):
        stage_id = int(stage["stage_id"])
        rows = []
        for device in devices:
            rejection = ""
            if device.health_score <= 0.0:
                rejection = "device_unhealthy"
            elif device.device_type not in stage["allowed_device_types"]:
                rejection = "device_type_not_allowed"
            elif required_dtype[device.device_type] not in device.supported_dtypes:
                rejection = "compute_dtype_not_supported"
            else:
                estimate = estimates[(stage_id, device.device_type)]
                if int(estimate["estimated_peak_bytes"]) > budgets[device.key][0]:
                    rejection = "insufficient_free_memory"
            rows.append(
                {
                    "miner_id_hash": device.miner_id_hash,
                    "device_id": device.device_id,
                    "device_type": device.device_type,
                    "available_after_reserve_bytes": budgets[device.key][0],
                    "reserve_bytes": budgets[device.key][1],
                    "reclaimable_previous_assignment_bytes": int(
                        reclaimable_previous_assignment_bytes.get(device.key, 0)
                    ),
                    "estimated_peak_bytes": int(
                        estimates.get((stage_id, device.device_type), {}).get(
                            "estimated_peak_bytes", 0
                        )
                    ),
                    "eligible": not rejection,
                    "rejection_reason": rejection,
                    "health_score": device.health_score,
                    "checkpoint_step": device.checkpoint_step,
                    "telemetry_at": device.telemetry_at,
                    "consecutive_failures": device.consecutive_failures,
                }
            )
        stage_candidate_audit[stage_id] = rows
        if not any(row["eligible"] for row in rows):
            raise PlacementError(
                "heterogeneous_placement_stage_has_no_eligible_device",
                diagnostics={"stage_id": stage_id, "candidates": rows},
            )

    # A beam keeps resource usage in state, avoiding the common greedy failure
    # where an early small stage consumes the only device that fits a later one.
    states: list[dict[str, Any]] = [
        {
            "score": 0.0,
            "assignments": [],
            "device_used": {},
            "miner_stage_count": {},
            "device_types": set(),
        }
    ]
    device_by_key = {item.key: item for item in devices}
    for stage_index, stage in enumerate(canonical["stages"]):
        stage_id = int(stage["stage_id"])
        next_states = []
        for state in states:
            previous_device = None
            if state["assignments"]:
                previous_device = device_by_key[state["assignments"][-1]["device_key"]]
            for device in devices:
                eligible = next(
                    row
                    for row in stage_candidate_audit[stage_id]
                    if row["miner_id_hash"] == device.miner_id_hash
                    and row["device_id"] == device.device_id
                )
                if not eligible["eligible"]:
                    continue
                estimate = estimates[(stage_id, device.device_type)]
                used = int(state["device_used"].get(device.key, 0))
                if used + int(estimate["estimated_peak_bytes"]) > budgets[device.key][0]:
                    continue
                miner_count = int(
                    state["miner_stage_count"].get(device.miner_id_hash, 0)
                )
                miner_limit = min(
                    int(device.max_stage_count), int(scheduler["max_stages_per_miner"])
                )
                if miner_count >= miner_limit:
                    continue
                compute_ms, compile_ms, measured = _compute_latency_ms(device, stage)
                transfer_ms = _transfer_latency_ms(
                    previous_device,
                    device,
                    int(estimate["activation_bytes"]),
                )
                load_penalty = (
                    device.miner_load_fraction + device.utilization_fraction
                ) * 100.0 * float(scheduler["load_cost_weight"])
                health_penalty = (
                    (1.0 - device.health_score) * 1000.0
                    + float(device.consecutive_failures) * 250.0
                )
                checkpoint_lag_steps = max(
                    0, int(current_checkpoint_step) - int(device.checkpoint_step)
                )
                checkpoint_freshness_penalty_ms = (
                    float(checkpoint_lag_steps) * 25.0
                    if previous_plan is not None
                    else 0.0
                )
                preference_penalty = (
                    0.0
                    if device.device_type == stage["preferred_device_type"]
                    else max(10.0, compute_ms * 0.25)
                )
                previous_device_key = previous_assignments.get(stage_id)
                migration_required = bool(
                    previous_device_key is not None
                    and previous_device_key
                    != (device.miner_id_hash, device.device_id)
                )
                migration_penalty_ms = 0.0
                if migration_required and reason not in {
                    "straggler_detected",
                    "owner_requested",
                    "health_degraded",
                    "performance_rebalance",
                    "coordinator_recovery",
                }:
                    bandwidth = max(
                        device.network_bandwidth_bytes_per_second,
                        100.0 * 1024**2,
                    )
                    migration_penalty_ms = max(
                        1000.0,
                        float(estimate["resident_weight_bytes"])
                        / bandwidth
                        * 1000.0
                        + max(device.network_latency_ms, 10.0),
                    )
                tpu_compile_cost_ms = 0.0
                if device.device_type == "jax_tpu" and (
                    previous_device_key
                    != (device.miner_id_hash, device.device_id)
                    or previous_plan is None
                ):
                    tpu_compile_cost_ms = float(compile_ms) * float(
                        scheduler.get("tpu_compile_cost_weight", 1.0)
                    )
                steady_compute_cost_ms = float(compute_ms) * float(
                    scheduler.get("tpu_steady_state_cost_weight", 1.0)
                    if device.device_type == "jax_tpu"
                    else 1.0
                )
                if scheduler["placement_policy"] == "memory-first":
                    utilization = (
                        used + int(estimate["estimated_peak_bytes"])
                    ) / max(1, budgets[device.key][0])
                    incremental = (
                        utilization * 1000.0
                        + steady_compute_cost_ms * 0.01
                        + tpu_compile_cost_ms * 0.01
                        + migration_penalty_ms
                        + health_penalty
                        + checkpoint_freshness_penalty_ms
                    )
                else:
                    incremental = (
                        steady_compute_cost_ms
                        + tpu_compile_cost_ms
                        + transfer_ms * float(scheduler["network_cost_weight"])
                        + load_penalty
                        + preference_penalty
                        + migration_penalty_ms
                        + health_penalty
                        + checkpoint_freshness_penalty_ms
                    )
                assignment = {
                    "stage_id": stage_id,
                    "miner_id_hash": device.miner_id_hash,
                    "device_id": device.device_id,
                    "device_type": device.device_type,
                    "device_key": device.key,
                    "resource_estimate": estimate,
                    "available_after_reserve_bytes": budgets[device.key][0],
                    "reclaimable_previous_assignment_bytes": int(
                        reclaimable_previous_assignment_bytes.get(device.key, 0)
                    ),
                    "device_used_after_assignment_bytes": used
                    + int(estimate["estimated_peak_bytes"]),
                    "compute_latency_ms": compute_ms,
                    "compute_latency_measured": measured,
                    "incoming_transfer_latency_ms": transfer_ms,
                    "load_penalty": load_penalty,
                    "health_penalty": health_penalty,
                    "health_score": device.health_score,
                    "checkpoint_step": device.checkpoint_step,
                    "checkpoint_lag_steps": checkpoint_lag_steps,
                    "checkpoint_freshness_penalty_ms": checkpoint_freshness_penalty_ms,
                    "telemetry_at": device.telemetry_at,
                    "consecutive_failures": device.consecutive_failures,
                    "preference_penalty": preference_penalty,
                    "migration_required": migration_required,
                    "migration_penalty_ms": migration_penalty_ms,
                    "incremental_score": incremental,
                    "selection_reason": (
                        "eligible_minimum_beam_score_with_measured_profile"
                        if measured
                        else "eligible_minimum_beam_score_with_capacity_estimate"
                    ),
                }
                if canonical["schema"] == TPU_MANIFEST_SCHEMA:
                    assignment.update(
                        {
                            "steady_compute_cost_ms": steady_compute_cost_ms,
                            "compile_latency_ms": compile_ms,
                            "tpu_compile_cost_ms": tpu_compile_cost_ms,
                        }
                    )
                next_device_used = {
                    **state["device_used"],
                    device.key: used + int(estimate["estimated_peak_bytes"]),
                }
                next_miner_stage_count = {
                    **state["miner_stage_count"],
                    device.miner_id_hash: miner_count + 1,
                }
                if not _remaining_stages_feasible(
                    canonical["stages"][stage_index + 1 :],
                    devices=devices,
                    stage_candidate_audit=stage_candidate_audit,
                    estimates=estimates,
                    budgets=budgets,
                    scheduler=scheduler,
                    device_used=next_device_used,
                    miner_stage_count=next_miner_stage_count,
                ):
                    forward_feasibility_rejection_count += 1
                    continue
                next_states.append(
                    {
                        "score": float(state["score"]) + incremental,
                        "assignments": [*state["assignments"], assignment],
                        "device_used": next_device_used,
                        "miner_stage_count": next_miner_stage_count,
                        "device_types": {
                            *state["device_types"],
                            device.device_type,
                        },
                    }
                )
        if not next_states:
            raise PlacementError(
                "heterogeneous_placement_capacity_exhausted",
                diagnostics={
                    "stage_id": stage_id,
                    "candidate_audit": stage_candidate_audit[stage_id],
                },
            )
        next_states.sort(
            key=lambda state: (
                round(float(state["score"]), 9),
                _mapping_signature(state["assignments"]),
            )
        )
        states = next_states[: int(scheduler["beam_width"])]

    required_devices = set(scheduler["required_device_types"])
    complete = [
        state for state in states if required_devices.issubset(state["device_types"])
    ]
    if not complete:
        raise PlacementError(
            "heterogeneous_placement_required_device_coverage_missing",
            diagnostics={
                "required_device_types": sorted(required_devices),
                "available_device_types": sorted({item.device_type for item in devices}),
            },
        )
    chosen = complete[0]
    assignments = []
    for assignment in chosen["assignments"]:
        public = dict(assignment)
        public.pop("device_key", None)
        assignments.append(public)
    previous_generation = int((previous_plan or {}).get("placement_generation") or 0)
    previous_signature = _mapping_signature((previous_plan or {}).get("assignments") or [])
    signature = _mapping_signature(assignments)
    changed = signature != previous_signature
    if previous_plan is None:
        generation = 1
    elif changed or reason != "initial_placement":
        generation = previous_generation + 1
    else:
        generation = max(1, previous_generation)
    device_summaries = []
    for device in devices:
        budget, reserve = budgets[device.key]
        summary = {
                "miner_id_hash": device.miner_id_hash,
                "device_id": device.device_id,
                "device_type": device.device_type,
                "total_memory_bytes": device.total_memory_bytes,
                "reported_free_memory_bytes": device.free_memory_bytes,
                "reserve_bytes": reserve,
                "reclaimable_previous_assignment_bytes": int(
                    reclaimable_previous_assignment_bytes.get(device.key, 0)
                ),
                "available_after_reserve_bytes": budget,
                "placed_peak_bytes": int(chosen["device_used"].get(device.key, 0)),
                "remaining_capacity_bytes": budget
                - int(chosen["device_used"].get(device.key, 0)),
                "throughput_units_per_second": device.throughput_units_per_second,
                "utilization_fraction": device.utilization_fraction,
                "health_score": device.health_score,
                "checkpoint_step": device.checkpoint_step,
                "telemetry_at": device.telemetry_at,
                "consecutive_failures": device.consecutive_failures,
            }
        if canonical["schema"] == TPU_MANIFEST_SCHEMA:
            summary.update(
                {
                    "group_device_count": device.group_device_count,
                    "mesh_axis_names": list(device.mesh_axis_names),
                    "mesh_shape": list(device.mesh_shape),
                    "accelerator_type": device.accelerator_type,
                }
            )
        device_summaries.append(summary)
    plan = {
        "schema": (
            TPU_PLACEMENT_SCHEMA
            if canonical["schema"] == TPU_MANIFEST_SCHEMA
            else PLACEMENT_SCHEMA
        ),
        "manifest_schema": canonical["schema"],
        "manifest_hash": canonical["content_hash"],
        "placement_generation": generation,
        "previous_placement_generation": previous_generation,
        "placement_changed": changed,
        "rebalance_reason": reason,
        "placement_policy": scheduler["placement_policy"],
        "stage_migration_cost_considered": True,
        "assignments": assignments,
        "stage_count": len(assignments),
        "complete_stage_coverage": len(assignments) == len(canonical["stages"]),
        "accepted_device_types": sorted(chosen["device_types"]),
        "required_device_types": sorted(required_devices),
        "required_device_coverage_complete": required_devices.issubset(
            chosen["device_types"]
        ),
        "single_gpu_miner_participating": any(
            capability["single_gpu_miner"]
            and any(
                row["miner_id_hash"] == capability["miner_id_hash"]
                and row["device_type"] == "cuda"
                for row in assignments
            )
            for capability in canonical_capabilities
        ),
        "multi_gpu_miner_participating": any(
            capability["multi_gpu_miner"]
            and any(
                row["miner_id_hash"] == capability["miner_id_hash"]
                and row["device_type"] == "cuda"
                for row in assignments
            )
            for capability in canonical_capabilities
        ),
        "cpu_miner_participating": any(
            row["device_type"] == "cpu" for row in assignments
        ),
        "total_score": float(chosen["score"]),
        "device_capacity": device_summaries,
        "candidate_audit": {
            str(stage_id): rows
            for stage_id, rows in sorted(stage_candidate_audit.items())
        },
        "excluded_devices": sorted(excluded),
        "capability_hashes": [
            capability["content_hash"] for capability in canonical_capabilities
        ],
        "generation_fencing_required": True,
        "runtime_telemetry_considered": bool(telemetry_rows),
        "forward_feasibility_checked": True,
        "forward_feasibility_rejection_count": int(
            forward_feasibility_rejection_count
        ),
        "resident_assignment_memory_reclaimed": bool(
            reclaimable_previous_assignment_bytes
        ),
        "health_score_considered": True,
        "checkpoint_freshness_considered": True,
        "current_checkpoint_step": int(current_checkpoint_step),
        "raw_device_names_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if canonical["schema"] == TPU_MANIFEST_SCHEMA:
        plan.update(
            {
                "tpu_miner_participating": any(
                    row["device_type"] == "jax_tpu" for row in assignments
                ),
                "tpu_compile_cost_considered": True,
            }
        )
    plan["content_hash"] = stable_hash(plan)
    return plan


def detect_stragglers(
    plan: dict[str, Any],
    stage_metrics: Iterable[dict[str, Any]],
    *,
    ratio: float,
    minimum_samples: int = 3,
) -> list[dict[str, Any]]:
    """Return stages whose measured latency persistently exceeds their peers."""

    if float(ratio) < 1.0 or int(minimum_samples) < 1:
        raise ValueError("heterogeneous_straggler_policy_invalid")
    rows = []
    for value in stage_metrics:
        if not isinstance(value, dict):
            continue
        samples = _positive_int(value.get("sample_count"))
        latency = _finite_float(value.get("total_latency_ms"))
        if samples >= int(minimum_samples) and latency > 0:
            rows.append(
                {
                    "stage_id": _positive_int(value.get("stage_id")),
                    "sample_count": samples,
                    "total_latency_ms": latency,
                }
            )
    if len(rows) < 2:
        return []
    ordered = sorted(item["total_latency_ms"] for item in rows)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    assignments = {
        int(item["stage_id"]): item for item in plan.get("assignments") or []
    }
    return [
        {
            **item,
            "median_stage_latency_ms": median,
            "straggler_ratio": item["total_latency_ms"] / median,
            "miner_id_hash": str(assignments.get(item["stage_id"], {}).get("miner_id_hash") or ""),
            "device_id": str(assignments.get(item["stage_id"], {}).get("device_id") or ""),
        }
        for item in sorted(rows, key=lambda value: value["stage_id"])
        if item["total_latency_ms"] > median * float(ratio)
    ]
