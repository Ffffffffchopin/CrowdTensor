from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from crowdtensor.core.contracts import (
    ContributionReceipt,
    TrainingProject,
    WorkUnit,
)
from crowdtensor.core.plugins import (
    BackendCapabilities,
    InferenceBackend,
    ModelAdapterPlugin,
    OptimizationPlugin,
    ProviderAdapter,
    TrainingBackend,
)
from crowdtensor.model_adapter import QwenModelAdapter


def test_existing_model_adapter_conforms_to_v2_structural_boundary() -> None:
    assert isinstance(QwenModelAdapter(), ModelAdapterPlugin)


def test_core_import_does_not_load_ml_or_provider_frameworks() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import crowdtensor.core; "
                "blocked={'torch','jax','transformers','accelerate','deepspeed','boto3'}; "
                "loaded=blocked.intersection(sys.modules); "
                "assert not loaded, sorted(loaded)"
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


class _Backend:
    backend_id = "test_backend"

    def capabilities(self):
        return BackendCapabilities(
            backend_id=self.backend_id,
            modes=frozenset(),
            checkpoint_formats=("safetensors",),
            supports_full_parameters=False,
            supports_peft=True,
        )

    def validate_project(self, project: TrainingProject):
        return ()

    def build_plan(self, project, providers, **options):
        raise NotImplementedError

    def run_work_unit(self, work: WorkUnit, *, workspace: Path):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint, *, workspace: Path):
        raise NotImplementedError


def test_backend_protocol_is_framework_neutral() -> None:
    assert isinstance(_Backend(), TrainingBackend)


def test_protocols_are_runtime_checkable_without_importing_frameworks() -> None:
    class Provider:
        provider_id = "local"

        def discover(self):
            raise NotImplementedError

        def acquire(self, work):
            raise NotImplementedError

        def release(self, allocation_id):
            raise NotImplementedError

    class Optimizer:
        plugin_id = "none"

        def validate_project(self, project):
            return ()

        def prepare_work(self, work):
            return work

        def validate_receipt(self, receipt: ContributionReceipt):
            return ()

    class Inference:
        backend_id = "upstream"

        def validate_model(self, model):
            return ()

        def serve(self, *, model, workspace):
            return ""

    assert isinstance(Provider(), ProviderAdapter)
    assert isinstance(Optimizer(), OptimizationPlugin)
    assert isinstance(Inference(), InferenceBackend)
