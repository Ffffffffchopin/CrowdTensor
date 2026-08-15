"""Built-in and entry-point discovery for Training Architecture v2 backends."""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any, Callable

from crowdtensor.core.contracts import stable_hash
from crowdtensor.core.plugins import TrainingBackend
from crowdtensor.version import __version__


TRAINING_BACKEND_ENTRY_POINT_GROUP = "crowdtensor.training_backends.v2"
TRAINING_BACKEND_REGISTRY_SCHEMA = "crowdtensor_training_backend_registry_v2"


class TrainingBackendRegistryError(RuntimeError):
    """Fail-closed backend discovery error with no private module paths."""


def _builtin_factories() -> dict[str, Callable[[], TrainingBackend]]:
    from .accelerate import AccelerateFSDP2Backend
    from .elastic_peft import VolunteerPEFTBackend

    return {
        "accelerate_fsdp2": AccelerateFSDP2Backend,
        "volunteer_peft": VolunteerPEFTBackend,
    }


def _plugins_enabled() -> bool:
    return str(os.environ.get("CROWDTENSOR_DISABLE_TRAINING_BACKEND_PLUGINS") or "").lower() not in {
        "1",
        "true",
        "yes",
    }


def _entry_points() -> list[Any]:
    discovered = importlib.metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=TRAINING_BACKEND_ENTRY_POINT_GROUP))
    return list(discovered.get(TRAINING_BACKEND_ENTRY_POINT_GROUP, ()))


def _instantiate(value: Any) -> Any:
    if isinstance(value, type):
        return value()
    if isinstance(value, TrainingBackend):
        return value
    if callable(value):
        return value()
    return value


def _validate_backend(backend: Any, *, expected_id: str) -> TrainingBackend:
    if not isinstance(backend, TrainingBackend):
        raise TrainingBackendRegistryError("training_backend_protocol_invalid")
    backend_id = str(getattr(backend, "backend_id", "") or "")
    if backend_id != expected_id:
        raise TrainingBackendRegistryError("training_backend_id_mismatch")
    capabilities = backend.capabilities()
    if capabilities.backend_id != backend_id or not capabilities.modes:
        raise TrainingBackendRegistryError("training_backend_capabilities_invalid")
    return backend


def training_backend_records() -> dict[str, tuple[TrainingBackend, dict[str, Any]]]:
    records: dict[str, tuple[TrainingBackend, dict[str, Any]]] = {}
    for backend_id, factory in _builtin_factories().items():
        records[backend_id] = (
            _validate_backend(factory(), expected_id=backend_id),
            {
                "kind": "builtin",
                "distribution_name": "crowdtensord",
                "distribution_version": __version__,
                "module_path_public": False,
                "installed_location_public": False,
            },
        )
    if not _plugins_enabled():
        return records
    failures: list[str] = []
    for point in sorted(
        _entry_points(),
        key=lambda item: (str(getattr(item, "name", "")), str(getattr(item, "value", ""))),
    ):
        try:
            backend_id = str(point.name)
            if backend_id in records:
                raise TrainingBackendRegistryError("training_backend_id_conflict")
            backend = _validate_backend(_instantiate(point.load()), expected_id=backend_id)
            distribution = getattr(point, "dist", None)
            metadata = getattr(distribution, "metadata", {}) if distribution is not None else {}
            name = str(metadata.get("Name") or "") if hasattr(metadata, "get") else ""
            version = str(getattr(distribution, "version", "") or "")
            if not name or not version:
                raise TrainingBackendRegistryError(
                    "training_backend_distribution_metadata_missing"
                )
            records[backend_id] = (
                backend,
                {
                    "kind": "entry_point_plugin",
                    "entry_point_group": TRAINING_BACKEND_ENTRY_POINT_GROUP,
                    "entry_point_name": backend_id,
                    "distribution_name": name,
                    "distribution_version": version,
                    "module_path_public": False,
                    "installed_location_public": False,
                },
            )
        except Exception as exc:
            reason = str(exc).splitlines()[0]
            if not reason.startswith("training_backend_"):
                reason = "training_backend_plugin_load_failed:" + type(exc).__name__
            failures.append(reason[:180])
    if failures:
        raise TrainingBackendRegistryError(
            "training_backend_plugin_discovery_failed:" + sorted(set(failures))[0]
        )
    return records


def training_backends() -> tuple[TrainingBackend, ...]:
    records = training_backend_records()
    return tuple(records[key][0] for key in sorted(records))


def get_training_backend(backend_id: str) -> TrainingBackend:
    try:
        return training_backend_records()[str(backend_id)][0]
    except KeyError as exc:
        raise TrainingBackendRegistryError("training_backend_id_unsupported") from exc


def backend_registry_report() -> dict[str, Any]:
    records = training_backend_records()
    backends = []
    for backend_id in sorted(records):
        backend, registration = records[backend_id]
        capabilities = backend.capabilities()
        backends.append(
            {
                "backend_id": backend_id,
                "modes": sorted(item.value for item in capabilities.modes),
                "checkpoint_formats": list(capabilities.checkpoint_formats),
                "supports_full_parameters": capabilities.supports_full_parameters,
                "supports_peft": capabilities.supports_peft,
                "registration": registration,
            }
        )
    report = {
        "schema": TRAINING_BACKEND_REGISTRY_SCHEMA,
        "entry_point_group": TRAINING_BACKEND_ENTRY_POINT_GROUP,
        "plugin_discovery_enabled": _plugins_enabled(),
        "builtin_backend_count": sum(
            registration["kind"] == "builtin" for _backend, registration in records.values()
        ),
        "plugin_backend_count": sum(
            registration["kind"] == "entry_point_plugin"
            for _backend, registration in records.values()
        ),
        "backends": backends,
        "module_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report
