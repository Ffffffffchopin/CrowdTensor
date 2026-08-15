"""Framework-neutral resource and execution-plan contracts.

The contracts in this module describe what a backend may run. They never
acquire hardware, import an ML framework, or contain rendezvous credentials.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping

from .contracts import ContractError, TrainingMode, stable_hash


PROVIDER_SNAPSHOT_SCHEMA = "crowdtensor_provider_snapshot_v2"
TRAINING_EXECUTION_PLAN_SCHEMA = "crowdtensor_training_execution_plan_v2"
STABLE_SHARDED_LAUNCH_SCHEMA = "crowdtensor_stable_sharded_launch_v2"

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
_PYTHON_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,254}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_DEVICE_TYPES = frozenset({"cpu", "cuda", "jax_tpu"})
_MIXED_PRECISION = frozenset({"no", "fp16", "bf16"})


class ResourceAvailability(str, Enum):
    """Whether a resource may participate in a coordinated rank window."""

    INTERMITTENT = "intermittent"
    STABLE_WINDOW = "stable_window"


def _require_identifier(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(result):
        raise ContractError(f"{field}_invalid")
    return result


def _require_hash(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not _HASH.fullmatch(result):
        raise ContractError(f"{field}_invalid")
    return result


def _require_non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field}_invalid")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field}_invalid") from exc
    if result < 0:
        raise ContractError(f"{field}_invalid")
    return result


def _require_positive_int(value: Any, field: str) -> int:
    result = _require_non_negative_int(value, field)
    if result < 1:
        raise ContractError(f"{field}_invalid")
    return result


def _require_identifier_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{field}_list_required")
    result = tuple(_require_identifier(item, field) for item in value)
    if len(set(result)) != len(result):
        raise ContractError(f"{field}_duplicate")
    return tuple(sorted(result))


def _verify_hashed_payload(
    value: Mapping[str, Any],
    *,
    schema: str,
    fields: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("execution_contract_object_required")
    payload = dict(value)
    if payload.get("schema") != schema:
        raise ContractError("execution_contract_schema_mismatch")
    allowed = fields | {"schema", "content_hash"}
    unknown = set(payload) - allowed
    missing = fields - set(payload)
    if unknown:
        raise ContractError(
            "execution_contract_unknown_fields:" + ",".join(sorted(unknown))
        )
    if missing:
        raise ContractError(
            "execution_contract_missing_fields:" + ",".join(sorted(missing))
        )
    supplied = _require_hash(payload.pop("content_hash", ""), "content_hash")
    if stable_hash(payload) != supplied:
        raise ContractError("execution_contract_content_hash_mismatch")
    return payload


@dataclass(frozen=True)
class ProviderSnapshot:
    """A public-safe point-in-time resource description."""

    provider_id: str
    resource_id: str
    machine_id_hash: str
    device_type: str
    device_count: int
    total_memory_bytes: int
    free_memory_bytes: int
    availability: ResourceAvailability
    source_hash: str
    capabilities: tuple[str, ...] = ()
    supported_dtypes: tuple[str, ...] = ()
    performance_score: float = 0.0
    stable_group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _require_identifier(self.provider_id, "provider_id")
        )
        object.__setattr__(
            self, "resource_id", _require_identifier(self.resource_id, "resource_id")
        )
        object.__setattr__(
            self,
            "machine_id_hash",
            _require_hash(self.machine_id_hash, "machine_id_hash"),
        )
        device_type = str(self.device_type or "").strip().lower()
        if device_type not in _DEVICE_TYPES:
            raise ContractError("provider_device_type_invalid")
        object.__setattr__(self, "device_type", device_type)
        object.__setattr__(
            self,
            "device_count",
            _require_positive_int(self.device_count, "provider_device_count"),
        )
        total = _require_positive_int(
            self.total_memory_bytes, "provider_total_memory_bytes"
        )
        free = _require_non_negative_int(
            self.free_memory_bytes, "provider_free_memory_bytes"
        )
        if free > total:
            raise ContractError("provider_free_memory_exceeds_total")
        object.__setattr__(self, "total_memory_bytes", total)
        object.__setattr__(self, "free_memory_bytes", free)
        try:
            availability = ResourceAvailability(self.availability)
        except ValueError as exc:
            raise ContractError("provider_availability_invalid") from exc
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "source_hash", _require_hash(self.source_hash, "source_hash"))
        object.__setattr__(
            self,
            "capabilities",
            _require_identifier_tuple(self.capabilities, "provider_capability"),
        )
        object.__setattr__(
            self,
            "supported_dtypes",
            _require_identifier_tuple(self.supported_dtypes, "provider_dtype"),
        )
        try:
            score = float(self.performance_score)
        except (TypeError, ValueError) as exc:
            raise ContractError("provider_performance_score_invalid") from exc
        if not math.isfinite(score) or score < 0.0:
            raise ContractError("provider_performance_score_invalid")
        object.__setattr__(self, "performance_score", score)
        if self.stable_group_id is not None:
            object.__setattr__(
                self,
                "stable_group_id",
                _require_identifier(self.stable_group_id, "stable_group_id"),
            )
        if availability is ResourceAvailability.STABLE_WINDOW and not self.stable_group_id:
            raise ContractError("stable_provider_group_required")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": PROVIDER_SNAPSHOT_SCHEMA,
            "provider_id": self.provider_id,
            "resource_id": self.resource_id,
            "machine_id_hash": self.machine_id_hash,
            "device_type": self.device_type,
            "device_count": self.device_count,
            "total_memory_bytes": self.total_memory_bytes,
            "free_memory_bytes": self.free_memory_bytes,
            "availability": self.availability.value,
            "source_hash": self.source_hash,
            "capabilities": list(self.capabilities),
            "supported_dtypes": list(self.supported_dtypes),
            "performance_score": self.performance_score,
            "stable_group_id": self.stable_group_id,
            "raw_device_names_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        payload["content_hash"] = stable_hash(payload)
        return payload

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProviderSnapshot:
        fields = {
            "provider_id",
            "resource_id",
            "machine_id_hash",
            "device_type",
            "device_count",
            "total_memory_bytes",
            "free_memory_bytes",
            "availability",
            "source_hash",
            "capabilities",
            "supported_dtypes",
            "performance_score",
            "stable_group_id",
            "raw_device_names_public",
            "private_paths_public",
            "public_artifact_safe",
        }
        payload = _verify_hashed_payload(value, schema=PROVIDER_SNAPSHOT_SCHEMA, fields=fields)
        if (
            payload["raw_device_names_public"] is not False
            or payload["private_paths_public"] is not False
            or payload["public_artifact_safe"] is not True
        ):
            raise ContractError("provider_snapshot_public_safety_invalid")
        return cls(
            provider_id=payload["provider_id"],
            resource_id=payload["resource_id"],
            machine_id_hash=payload["machine_id_hash"],
            device_type=payload["device_type"],
            device_count=payload["device_count"],
            total_memory_bytes=payload["total_memory_bytes"],
            free_memory_bytes=payload["free_memory_bytes"],
            availability=payload["availability"],
            source_hash=payload["source_hash"],
            capabilities=tuple(payload["capabilities"]),
            supported_dtypes=tuple(payload["supported_dtypes"]),
            performance_score=payload["performance_score"],
            stable_group_id=payload["stable_group_id"],
        )


@dataclass(frozen=True)
class TrainingExecutionPlan:
    """Backend selection and resource admission result for one project."""

    project_hash: str
    mode: TrainingMode
    backend_id: str
    selected_resources: tuple[ProviderSnapshot, ...]
    required_capabilities: tuple[str, ...]
    restart_semantics: str
    runtime_name: str
    runtime_available: bool
    runtime_version: str = ""
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_hash", _require_hash(self.project_hash, "project_hash"))
        try:
            mode = TrainingMode(self.mode)
        except ValueError as exc:
            raise ContractError("execution_plan_mode_invalid") from exc
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "backend_id", _require_identifier(self.backend_id, "backend_id")
        )
        resources = tuple(self.selected_resources)
        if any(not isinstance(item, ProviderSnapshot) for item in resources):
            raise ContractError("execution_plan_resource_invalid")
        resource_ids = [item.resource_id for item in resources]
        if len(resource_ids) != len(set(resource_ids)):
            raise ContractError("execution_plan_resource_duplicate")
        resources = tuple(sorted(resources, key=lambda item: (item.provider_id, item.resource_id)))
        object.__setattr__(self, "selected_resources", resources)
        object.__setattr__(
            self,
            "required_capabilities",
            _require_identifier_tuple(
                self.required_capabilities, "execution_required_capability"
            ),
        )
        required = set(self.required_capabilities)
        if any(not required.issubset(item.capabilities) for item in resources):
            raise ContractError("execution_plan_resource_capability_mismatch")
        expected_restart = {
            TrainingMode.ELASTIC_DELTA: "reissue_work_from_committed_checkpoint",
            TrainingMode.STABLE_SHARDED: "restart_rank_group_from_committed_checkpoint",
        }[mode]
        if self.restart_semantics != expected_restart:
            raise ContractError("execution_plan_restart_semantics_invalid")
        object.__setattr__(
            self, "runtime_name", _require_identifier(self.runtime_name, "runtime_name")
        )
        if not isinstance(self.runtime_available, bool):
            raise ContractError("execution_plan_runtime_available_boolean_required")
        version = str(self.runtime_version or "").strip()
        if self.runtime_available and not version:
            raise ContractError("execution_plan_runtime_version_required")
        object.__setattr__(self, "runtime_version", version)
        blockers = tuple(sorted({str(item).strip() for item in self.blockers if str(item).strip()}))
        if any(not _IDENTIFIER.fullmatch(item) for item in blockers):
            raise ContractError("execution_plan_blocker_invalid")
        if self.runtime_available and "runtime_unavailable" in blockers:
            raise ContractError("execution_plan_runtime_state_conflict")
        if not self.runtime_available and "runtime_unavailable" not in blockers:
            blockers = tuple(sorted((*blockers, "runtime_unavailable")))
        if not resources and "no_selected_resource" not in blockers:
            blockers = tuple(sorted((*blockers, "no_selected_resource")))
        if mode is TrainingMode.STABLE_SHARDED:
            if any(
                item.availability is not ResourceAvailability.STABLE_WINDOW
                for item in resources
            ):
                raise ContractError("stable_plan_intermittent_resource_forbidden")
            groups = {item.stable_group_id for item in resources}
            if resources and len(groups) != 1:
                raise ContractError("stable_plan_group_mismatch")
        object.__setattr__(self, "blockers", blockers)

    @property
    def execution_ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": TRAINING_EXECUTION_PLAN_SCHEMA,
            "project_hash": self.project_hash,
            "mode": self.mode.value,
            "backend_id": self.backend_id,
            "selected_resources": [item.to_dict() for item in self.selected_resources],
            "required_capabilities": list(self.required_capabilities),
            "restart_semantics": self.restart_semantics,
            "runtime_name": self.runtime_name,
            "runtime_available": self.runtime_available,
            "runtime_version": self.runtime_version,
            "blockers": list(self.blockers),
            "execution_ready": self.execution_ready,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        payload["content_hash"] = stable_hash(payload)
        return payload

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingExecutionPlan:
        fields = {
            "project_hash",
            "mode",
            "backend_id",
            "selected_resources",
            "required_capabilities",
            "restart_semantics",
            "runtime_name",
            "runtime_available",
            "runtime_version",
            "blockers",
            "execution_ready",
            "credential_values_public",
            "private_paths_public",
            "public_artifact_safe",
        }
        payload = _verify_hashed_payload(
            value, schema=TRAINING_EXECUTION_PLAN_SCHEMA, fields=fields
        )
        if (
            payload["credential_values_public"] is not False
            or payload["private_paths_public"] is not False
            or payload["public_artifact_safe"] is not True
        ):
            raise ContractError("execution_plan_public_safety_invalid")
        resources = payload["selected_resources"]
        if not isinstance(resources, list):
            raise ContractError("execution_plan_resources_list_required")
        plan = cls(
            project_hash=payload["project_hash"],
            mode=payload["mode"],
            backend_id=payload["backend_id"],
            selected_resources=tuple(ProviderSnapshot.from_dict(item) for item in resources),
            required_capabilities=tuple(payload["required_capabilities"]),
            restart_semantics=payload["restart_semantics"],
            runtime_name=payload["runtime_name"],
            runtime_available=payload["runtime_available"],
            runtime_version=payload["runtime_version"],
            blockers=tuple(payload["blockers"]),
        )
        if plan.execution_ready is not payload["execution_ready"]:
            raise ContractError("execution_plan_ready_state_mismatch")
        return plan


def _relative_control_path(value: Any, field: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{field}_must_be_relative")
    if path.parts[0] != ".crowdtensor":
        raise ContractError(f"{field}_outside_control_directory")
    return path.as_posix()


@dataclass(frozen=True)
class StableShardedLaunchSpec:
    """Secret-free Accelerate launch template for one stable rank group."""

    plan_hash: str
    project_hash: str
    backend_id: str
    distributed_type: str
    num_machines: int
    num_processes: int
    mixed_precision: str
    config_path: str
    checkpoint_path: str
    trainer_entrypoint: str
    transformer_layer_class: str | None
    command_template: tuple[str, ...]
    environment_names: tuple[str, ...]
    max_restarts: int
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_hash", _require_hash(self.plan_hash, "plan_hash"))
        object.__setattr__(self, "project_hash", _require_hash(self.project_hash, "project_hash"))
        object.__setattr__(
            self, "backend_id", _require_identifier(self.backend_id, "backend_id")
        )
        if self.distributed_type != "fsdp2":
            raise ContractError("stable_launch_distributed_type_invalid")
        machines = _require_positive_int(self.num_machines, "stable_launch_num_machines")
        processes = _require_positive_int(self.num_processes, "stable_launch_num_processes")
        if processes < machines:
            raise ContractError("stable_launch_process_count_invalid")
        object.__setattr__(self, "num_machines", machines)
        object.__setattr__(self, "num_processes", processes)
        precision = str(self.mixed_precision or "").lower()
        if precision not in _MIXED_PRECISION:
            raise ContractError("stable_launch_mixed_precision_invalid")
        object.__setattr__(self, "mixed_precision", precision)
        object.__setattr__(
            self, "config_path", _relative_control_path(self.config_path, "config_path")
        )
        object.__setattr__(
            self,
            "checkpoint_path",
            _relative_control_path(self.checkpoint_path, "checkpoint_path"),
        )
        entrypoint = str(self.trainer_entrypoint or "").strip().replace("\\", "/")
        if (
            not entrypoint
            or PurePosixPath(entrypoint).is_absolute()
            or ".." in PurePosixPath(entrypoint).parts
            or any(character.isspace() for character in entrypoint)
        ):
            raise ContractError("stable_launch_trainer_entrypoint_invalid")
        object.__setattr__(self, "trainer_entrypoint", entrypoint)
        if self.transformer_layer_class is not None:
            layer = str(self.transformer_layer_class).strip()
            if not _PYTHON_NAME.fullmatch(layer):
                raise ContractError("stable_launch_transformer_layer_class_invalid")
            object.__setattr__(self, "transformer_layer_class", layer)
        command = tuple(str(item) for item in self.command_template)
        if not command or any(not item or "\n" in item or "\x00" in item for item in command):
            raise ContractError("stable_launch_command_template_invalid")
        lowered = " ".join(command).lower()
        if any(marker in lowered for marker in ("bearer ", "api_token=", "password=", "secret=")):
            raise ContractError("stable_launch_command_contains_secret")
        object.__setattr__(self, "command_template", command)
        names = tuple(sorted(set(str(item) for item in self.environment_names)))
        if not names or any(not _ENVIRONMENT_NAME.fullmatch(item) for item in names):
            raise ContractError("stable_launch_environment_names_invalid")
        referenced_names = set(
            re.findall(r"\$\{([A-Z][A-Z0-9_]{0,127})\}", " ".join(command))
        )
        if referenced_names != set(names):
            raise ContractError("stable_launch_environment_binding_invalid")
        object.__setattr__(self, "environment_names", names)
        object.__setattr__(
            self,
            "max_restarts",
            _require_non_negative_int(self.max_restarts, "stable_launch_max_restarts"),
        )
        blockers = tuple(sorted({str(item).strip() for item in self.blockers if str(item).strip()}))
        if any(not _IDENTIFIER.fullmatch(item) for item in blockers):
            raise ContractError("stable_launch_blocker_invalid")
        object.__setattr__(self, "blockers", blockers)

    @property
    def execution_ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": STABLE_SHARDED_LAUNCH_SCHEMA,
            "plan_hash": self.plan_hash,
            "project_hash": self.project_hash,
            "backend_id": self.backend_id,
            "distributed_type": self.distributed_type,
            "num_machines": self.num_machines,
            "num_processes": self.num_processes,
            "mixed_precision": self.mixed_precision,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "trainer_entrypoint": self.trainer_entrypoint,
            "transformer_layer_class": self.transformer_layer_class,
            "command_template": list(self.command_template),
            "environment_names": list(self.environment_names),
            "restart_policy": "restart_rank_group_from_latest_committed_checkpoint",
            "max_restarts": self.max_restarts,
            "blockers": list(self.blockers),
            "execution_ready": self.execution_ready,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        payload["content_hash"] = stable_hash(payload)
        return payload

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StableShardedLaunchSpec:
        fields = {
            "plan_hash",
            "project_hash",
            "backend_id",
            "distributed_type",
            "num_machines",
            "num_processes",
            "mixed_precision",
            "config_path",
            "checkpoint_path",
            "trainer_entrypoint",
            "transformer_layer_class",
            "command_template",
            "environment_names",
            "restart_policy",
            "max_restarts",
            "blockers",
            "execution_ready",
            "credential_values_public",
            "private_paths_public",
            "public_artifact_safe",
        }
        payload = _verify_hashed_payload(
            value, schema=STABLE_SHARDED_LAUNCH_SCHEMA, fields=fields
        )
        if payload["restart_policy"] != "restart_rank_group_from_latest_committed_checkpoint":
            raise ContractError("stable_launch_restart_policy_invalid")
        if (
            payload["credential_values_public"] is not False
            or payload["private_paths_public"] is not False
            or payload["public_artifact_safe"] is not True
        ):
            raise ContractError("stable_launch_public_safety_invalid")
        spec = cls(
            plan_hash=payload["plan_hash"],
            project_hash=payload["project_hash"],
            backend_id=payload["backend_id"],
            distributed_type=payload["distributed_type"],
            num_machines=payload["num_machines"],
            num_processes=payload["num_processes"],
            mixed_precision=payload["mixed_precision"],
            config_path=payload["config_path"],
            checkpoint_path=payload["checkpoint_path"],
            trainer_entrypoint=payload["trainer_entrypoint"],
            transformer_layer_class=payload["transformer_layer_class"],
            command_template=tuple(payload["command_template"]),
            environment_names=tuple(payload["environment_names"]),
            max_restarts=payload["max_restarts"],
            blockers=tuple(payload["blockers"]),
        )
        if spec.execution_ready is not payload["execution_ready"]:
            raise ContractError("stable_launch_ready_state_mismatch")
        return spec
