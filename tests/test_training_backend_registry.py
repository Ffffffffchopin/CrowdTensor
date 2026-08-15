from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import crowdtensor.backends.registry as registry
from crowdtensor.core import TrainingMode
from crowdtensor.core.plugins import BackendCapabilities


class ExampleBackend:
    backend_id = "example_backend"

    def capabilities(self):
        return BackendCapabilities(
            backend_id=self.backend_id,
            modes=frozenset({TrainingMode.ELASTIC_DELTA}),
            checkpoint_formats=("example",),
            supports_full_parameters=False,
            supports_peft=True,
        )

    def validate_project(self, project):
        return ()

    def build_plan(self, project, providers, **options):
        raise NotImplementedError


class FakeEntryPoint:
    def __init__(self, name, loaded, *, distribution="example-backend", version="1.0"):
        self.name = name
        self.value = "example:Backend"
        self._loaded = loaded
        self.dist = SimpleNamespace(metadata={"Name": distribution}, version=version)

    def load(self):
        return self._loaded


class FakeEntryPoints(list):
    def select(self, *, group):
        return self if group == registry.TRAINING_BACKEND_ENTRY_POINT_GROUP else []


def test_builtin_backend_registry_is_small_and_mode_explicit(monkeypatch) -> None:
    monkeypatch.setattr(registry.importlib.metadata, "entry_points", lambda: FakeEntryPoints())
    report = registry.backend_registry_report()
    assert [item["backend_id"] for item in report["backends"]] == [
        "accelerate_fsdp2",
        "volunteer_peft",
    ]
    assert report["builtin_backend_count"] == 2
    assert report["plugin_backend_count"] == 0


def test_external_backend_entry_point_is_discovered_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint("example_backend", ExampleBackend)]),
    )
    assert isinstance(registry.get_training_backend("example_backend"), ExampleBackend)
    with pytest.raises(registry.TrainingBackendRegistryError, match="discovery_failed"):
        monkeypatch.setattr(
            registry.importlib.metadata,
            "entry_points",
            lambda: FakeEntryPoints([FakeEntryPoint("volunteer_peft", ExampleBackend)]),
        )
        registry.training_backend_records()


def test_backend_registry_import_does_not_load_frameworks() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from crowdtensor.backends import backend_registry_report; "
                "backend_registry_report(); "
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
