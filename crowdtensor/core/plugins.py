"""Structural plugin contracts for framework and provider integrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import ContributionReceipt, TrainingMode, TrainingProject, WorkUnit
from .execution import (
    ProviderSnapshot,
    StableShardedLaunchSpec,
    TrainingExecutionPlan,
)


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str
    modes: frozenset[TrainingMode]
    checkpoint_formats: tuple[str, ...]
    supports_full_parameters: bool
    supports_peft: bool


@runtime_checkable
class ModelAdapterPlugin(Protocol):
    """Model-family semantics; existing model_adapter_v1 classes conform."""

    adapter_id: str

    def descriptor(self) -> dict[str, Any]: ...

    def supports(self, *, model_id: str, config: Mapping[str, Any]) -> bool: ...

    def validate_config(self, config: Mapping[str, Any]) -> dict[str, Any]: ...

    def partition(self, config: Mapping[str, Any], *, stage_count: int) -> Sequence[Any]: ...


@runtime_checkable
class TrainingBackend(Protocol):
    """Delegates numerical training to Accelerate, FSDP2, DeepSpeed, or peers."""

    backend_id: str

    def capabilities(self) -> BackendCapabilities: ...

    def validate_project(self, project: TrainingProject) -> tuple[str, ...]: ...

    def build_plan(
        self,
        project: TrainingProject,
        providers: Sequence[ProviderSnapshot],
        **options: Any,
    ) -> TrainingExecutionPlan: ...


@runtime_checkable
class ElasticDeltaBackend(TrainingBackend, Protocol):
    """Backend for independently retryable delta-producing Work Units."""

    def run_next_work_unit(self, worker_runtime: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class StableShardedBackend(TrainingBackend, Protocol):
    """Backend for one stable rank group and upstream collective runtime."""

    def build_launch_spec(
        self,
        plan: TrainingExecutionPlan,
        **options: Any,
    ) -> StableShardedLaunchSpec: ...

    def materialize_launch(
        self,
        spec: StableShardedLaunchSpec,
        *,
        workspace: Path,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ProviderAdapter(Protocol):
    """Discovers provider resources without importing model semantics."""

    provider_id: str

    def discover(self) -> tuple[ProviderSnapshot, ...]: ...


@runtime_checkable
class ManagedProviderAdapter(ProviderAdapter, Protocol):
    """Optional lifecycle extension for providers that allocate resources."""

    def acquire(self, plan: TrainingExecutionPlan) -> str: ...

    def release(self, allocation_id: str) -> None: ...


@runtime_checkable
class OptimizationPlugin(Protocol):
    """Optional policy or performance hook outside the correctness core."""

    plugin_id: str

    def validate_project(self, project: TrainingProject) -> tuple[str, ...]: ...

    def prepare_work(self, work: WorkUnit) -> WorkUnit: ...

    def validate_receipt(self, receipt: ContributionReceipt) -> tuple[str, ...]: ...


@runtime_checkable
class InferenceBackend(Protocol):
    """Secondary serving adapter for vLLM, SGLang, or another upstream engine."""

    backend_id: str

    def validate_model(self, model: Mapping[str, Any]) -> tuple[str, ...]: ...

    def serve(self, *, model: Mapping[str, Any], workspace: Path) -> str: ...
