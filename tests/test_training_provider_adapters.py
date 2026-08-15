from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from crowdtensor.adapters.providers import (
    LegacyCapabilityProviderAdapter,
    legacy_capability_to_snapshots,
)
from crowdtensor.core import ResourceAvailability, stable_hash
from crowdtensor.core.plugins import ProviderAdapter
from crowdtensor.adapters.capabilities import CAPABILITY_SCHEMA


def _capability() -> dict:
    return {
        "schema": CAPABILITY_SCHEMA,
        "miner_id_hash": stable_hash("provider-machine"),
        "cpu": {
            "device_id": "cpu",
            "physical_core_count": 4,
            "logical_core_count": 8,
            "total_memory_bytes": 16 * 1024**3,
            "free_memory_bytes": 12 * 1024**3,
            "supported_dtypes": ["bfloat16", "float32"],
            "throughput_units_per_second": 2.0,
            "microbenchmark_latency_ms": 1.0,
            "utilization_fraction": 0.1,
        },
        "gpus": [
            {
                "device_id": f"cuda:{index}",
                "device_index": index,
                "device_name_hash": stable_hash(f"gpu-{index}"),
                "total_memory_bytes": 24 * 1024**3,
                "free_memory_bytes": 20 * 1024**3,
                "compute_capability": "8.9",
                "supported_dtypes": ["bfloat16", "float16", "float32"],
                "throughput_units_per_second": 4.0,
                "utilization_fraction": 0.0,
                "raw_device_name_public": False,
            }
            for index in range(2)
        ],
        "network": {
            "measured_bandwidth_bytes_per_second": 0.0,
            "measured_round_trip_latency_ms": 0.0,
            "measurement_count": 0,
        },
        "stage_profiles": [],
        "current_load_fraction": 0.1,
        "max_stage_count": 2,
        "single_gpu_miner": False,
        "multi_gpu_miner": True,
        "cpu_stage_supported": True,
        "raw_device_names_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def test_legacy_capability_maps_to_generic_provider_snapshots() -> None:
    snapshots = legacy_capability_to_snapshots(
        _capability(),
        provider_id="fixture",
        availability=ResourceAvailability.STABLE_WINDOW,
        stable_group_id="group-a",
    )
    assert [item.device_type for item in snapshots] == ["cpu", "cuda", "cuda"]
    assert all(item.stable_group_id == "group-a" for item in snapshots)
    assert all(item.source_hash.startswith("sha256:") for item in snapshots)
    assert "distributed_collective" in snapshots[1].capabilities


def test_legacy_provider_adapter_conforms_to_discovery_protocol() -> None:
    adapter = LegacyCapabilityProviderAdapter((_capability(),), provider_id="fixture")
    assert isinstance(adapter, ProviderAdapter)
    assert len(adapter.discover()) == 3


def test_provider_adapter_import_does_not_load_ml_frameworks() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import crowdtensor.adapters.providers; "
                "blocked={'torch','jax','transformers','accelerate','deepspeed'}; "
                "assert not blocked.intersection(sys.modules), sorted(blocked.intersection(sys.modules))"
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
