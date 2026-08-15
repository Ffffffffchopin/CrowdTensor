"""Numerical runtime bridges kept outside the framework-neutral core."""

from .registry import (
    TrainingBackendRegistryError,
    backend_registry_report,
    get_training_backend,
    training_backends,
)

__all__ = [
    "TrainingBackendRegistryError",
    "backend_registry_report",
    "get_training_backend",
    "training_backends",
]
