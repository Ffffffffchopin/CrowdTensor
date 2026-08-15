"""Adapters from existing resource discovery into v2 provider snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from crowdtensor.core.contracts import ContractError
from crowdtensor.core.execution import ProviderSnapshot, ResourceAvailability


def _resource_id(provider_id: str, machine_hash: str, suffix: str) -> str:
    digest = machine_hash.split(":", 1)[-1][:12]
    normalized = suffix.replace(":", ".").replace("_", ".").lower()
    return f"{provider_id}.{digest}.{normalized}"


def legacy_capability_to_snapshots(
    capability: Mapping[str, Any],
    *,
    provider_id: str,
    availability: ResourceAvailability | str = ResourceAvailability.INTERMITTENT,
    stable_group_id: str | None = None,
) -> tuple[ProviderSnapshot, ...]:
    """Map a validated heterogeneous v1/v2 capability without model coupling."""

    from .capabilities import validate_miner_capability

    try:
        canonical = validate_miner_capability(dict(capability))
    except (TypeError, ValueError) as exc:
        raise ContractError("legacy_provider_capability_invalid") from exc
    machine_hash = str(canonical["miner_id_hash"])
    source_hash = str(canonical["content_hash"])
    snapshots: list[ProviderSnapshot] = []
    cpu = canonical["cpu"]
    if canonical.get("cpu_stage_supported"):
        snapshots.append(
            ProviderSnapshot(
                provider_id=provider_id,
                resource_id=_resource_id(provider_id, machine_hash, "cpu"),
                machine_id_hash=machine_hash,
                device_type="cpu",
                device_count=max(1, int(cpu["logical_core_count"])),
                total_memory_bytes=int(cpu["total_memory_bytes"]),
                free_memory_bytes=int(cpu["free_memory_bytes"]),
                availability=availability,
                source_hash=source_hash,
                capabilities=("cpu", "elastic_delta", "peft_lora"),
                supported_dtypes=tuple(cpu["supported_dtypes"]),
                performance_score=float(cpu["throughput_units_per_second"]),
                stable_group_id=stable_group_id,
            )
        )
    for gpu in canonical.get("gpus") or []:
        snapshots.append(
            ProviderSnapshot(
                provider_id=provider_id,
                resource_id=_resource_id(provider_id, machine_hash, gpu["device_id"]),
                machine_id_hash=machine_hash,
                device_type="cuda",
                device_count=1,
                total_memory_bytes=int(gpu["total_memory_bytes"]),
                free_memory_bytes=int(gpu["free_memory_bytes"]),
                availability=availability,
                source_hash=source_hash,
                capabilities=(
                    "cuda",
                    "distributed_collective",
                    "elastic_delta",
                    "peft_lora",
                    "stable_sharded",
                ),
                supported_dtypes=tuple(gpu["supported_dtypes"]),
                performance_score=float(gpu["throughput_units_per_second"]),
                stable_group_id=stable_group_id,
            )
        )
    for group in canonical.get("tpu_groups") or []:
        snapshots.append(
            ProviderSnapshot(
                provider_id=provider_id,
                resource_id=_resource_id(provider_id, machine_hash, group["device_id"]),
                machine_id_hash=machine_hash,
                device_type="jax_tpu",
                device_count=int(group["device_count"]),
                total_memory_bytes=int(group["total_hbm_bytes"]),
                free_memory_bytes=int(group["free_hbm_bytes"]),
                availability=availability,
                source_hash=source_hash,
                capabilities=("jax", "jax_tpu", "stable_mesh"),
                supported_dtypes=tuple(group["supported_dtypes"]),
                performance_score=float(group["throughput_units_per_second"]),
                stable_group_id=stable_group_id,
            )
        )
    return tuple(sorted(snapshots, key=lambda item: item.resource_id))


@dataclass(frozen=True)
class LegacyCapabilityProviderAdapter:
    """Discovery-only adapter for already collected capability documents."""

    capabilities: tuple[Mapping[str, Any], ...]
    provider_id: str = "legacy"
    availability: ResourceAvailability = ResourceAvailability.INTERMITTENT
    stable_group_id: str | None = None

    def discover(self) -> tuple[ProviderSnapshot, ...]:
        snapshots: list[ProviderSnapshot] = []
        for capability in self.capabilities:
            snapshots.extend(
                legacy_capability_to_snapshots(
                    capability,
                    provider_id=self.provider_id,
                    availability=self.availability,
                    stable_group_id=self.stable_group_id,
                )
            )
        resource_ids = [item.resource_id for item in snapshots]
        if len(resource_ids) != len(set(resource_ids)):
            raise ContractError("provider_snapshot_resource_id_conflict")
        return tuple(sorted(snapshots, key=lambda item: item.resource_id))


@dataclass(frozen=True)
class LocalProviderAdapter:
    """Explicit local discovery; framework imports occur only in discover()."""

    provider_id: str = "local"
    availability: ResourceAvailability = ResourceAvailability.INTERMITTENT
    stable_group_id: str | None = None
    include_jax_tpu: bool = False
    run_microbenchmark: bool = False
    max_stage_count: int = 0

    def discover(self) -> tuple[ProviderSnapshot, ...]:
        from .capabilities import (
            discover_heterogeneous_training_capability,
        )

        capability = discover_heterogeneous_training_capability(
            include_jax_tpu=self.include_jax_tpu,
            run_microbenchmark=self.run_microbenchmark,
            max_stage_count=self.max_stage_count,
        )
        return legacy_capability_to_snapshots(
            capability,
            provider_id=self.provider_id,
            availability=self.availability,
            stable_group_id=self.stable_group_id,
        )
