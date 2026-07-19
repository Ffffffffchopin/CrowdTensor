from __future__ import annotations

import json

import pytest

from crowdtensor import hf_lora_training as hf
from crowdtensor.pipeline_lora_training import (
    CUDA_PIPELINE_SCHEMA,
    CUDAStageRuntime,
    _public_cuda_pipeline_report,
)
from crowdtensor.training_contract import TRAINING_SPEC_SCHEMA


def test_cuda_device_parser_requires_explicit_non_negative_index() -> None:
    assert hf._cuda_device_index("cuda") == 0
    assert hf._cuda_device_index("cuda:1") == 1
    with pytest.raises(ValueError):
        hf._cuda_device_index("cpu")
    with pytest.raises(ValueError):
        hf._cuda_device_index("cuda:-1")


def test_cuda_runtime_unavailable_is_classified_without_cpu_fallback(tmp_path, monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(hf, "_deps", lambda: (FakeTorch(), None, None, None, None, None))
    runtime = hf.CUDALoRATrainingRuntime("cuda:0")
    with pytest.raises(hf.CUDAUnavailableError, match="cuda_training_device_unavailable"):
        runtime.run(
            {"schema": TRAINING_SPEC_SCHEMA, "device": "cuda:0"},
            output_dir=tmp_path,
        )
    blocker = json.loads((tmp_path / "cuda_training_blocker.json").read_text(encoding="utf-8"))
    assert blocker["blocker"] == "cuda_training_device_unavailable"
    assert blocker["public_artifact_safe"] is True
    assert not (tmp_path / "training_result_public.json").exists()


def test_cuda_public_pipeline_report_removes_private_paths_and_tensor_values() -> None:
    public = _public_cuda_pipeline_report(
        {
            "schema": CUDA_PIPELINE_SCHEMA,
            "gpu_live_verified": True,
            "private_report_path": "/private/report.json",
            "final_checkpoint": {
                "schema": "crowdtensor_pipeline_global_checkpoint_v1",
                "manifest_path": "/private/global.json",
                "stages": [
                    {
                        "stage_id": 0,
                        "adapter_path": "/private/adapter.safetensors",
                        "optimizer_path": "/private/optimizer.pt",
                        "content_hash": "sha256:stage0",
                    }
                ],
            },
        }
    )
    encoded = json.dumps(public)
    assert "/private" not in encoded
    assert public["gpu_live_verified"] is True
    assert public["activation_values_public"] is False
    assert public["gradient_values_public"] is False


def test_cuda_stage_runtime_is_a_production_class_not_cpu_alias() -> None:
    assert CUDAStageRuntime.__name__ == "CUDAStageRuntime"
    assert CUDAStageRuntime.__module__ == "crowdtensor.pipeline_lora_training"
    assert CUDAStageRuntime is not hf.CUDATrainingRuntimeDryRun
