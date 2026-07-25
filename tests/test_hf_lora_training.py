from __future__ import annotations

import json
import tempfile
from pathlib import Path

import crowdtensor.hf_lora_training as hf_lora_training
from crowdtensor.hf_lora_training import (
    CPULoRATrainingRuntime,
    create_local_training_fixture,
    evaluate_adapter,
    training_spec_for_claim,
)
from crowdtensor.training_contract import public_training_spec, validate_adapter_delta


def test_real_cpu_transformers_peft_lora_runtime_and_export_load() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        fixture = create_local_training_fixture(root / "fixture", local_steps=4)
        spec = training_spec_for_claim(
            fixture,
            task_id="real-task",
            miner_id="real-miner",
            shard_index=0,
        )
        result = CPULoRATrainingRuntime().run(spec, output_dir=root / "worker")
        validation = validate_adapter_delta(
            result["adapter_delta"],
            expected={
                "job_id": spec["job_id"],
                "round_id": spec["round_id"],
                "model_manifest_hash": spec["model_manifest_hash"],
                "base_model_hash": spec["base_model_hash"],
                "base_adapter_hash": spec["base_adapter_hash"],
                "base_model_version": spec["base_model_version"],
                "adapter_version": spec["adapter_version"],
                "dataset_shard_index": spec["dataset_shard_index"],
                "dataset_shard_hash": spec["dataset_shard_hash"],
                "tensor_specs": fixture["lora"]["tensor_specs"],
            },
        )
        before = evaluate_adapter(
            base_model_path=fixture["model"]["base_model_path"],
            adapter_path=None,
            dataset_path=fixture["dataset"]["private_dataset_path"],
            sample_indexes=spec["sample_indexes"],
        )
        after = evaluate_adapter(
            base_model_path=fixture["model"]["base_model_path"],
            adapter_path=result["adapter_path"],
            dataset_path=fixture["dataset"]["private_dataset_path"],
            sample_indexes=spec["sample_indexes"],
        )
        public = public_training_spec(result)

    assert result["runtime"]["real_pytorch_autograd"] is True
    assert result["runtime"]["real_transformers"] is True
    assert result["runtime"]["real_peft_lora"] is True
    assert result["real_backward"] is True
    assert result["base_weights_frozen"] is True
    assert result["only_lora_trainable"] is True
    assert result["loss_reduced"] is True
    assert validation["accepted"] is True
    assert after["adapter_loaded"] is True
    assert after["mean_loss"] < before["mean_loss"]
    assert after["logits_hash"] != before["logits_hash"]
    serialized = json.dumps(public, sort_keys=True)
    assert "private_dataset.jsonl" not in serialized
    assert str(root) not in serialized


def test_runtime_disables_incompatible_optional_torchao_before_peft_load(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, adapter_path, *, is_trainable, local_files_only):
            calls.append("peft_load")
            assert base == "dense-base"
            assert adapter_path == "adapter"
            assert is_trainable is True
            assert local_files_only is True
            return type("Loaded", (), {})()

    def disable(base) -> bool:
        calls.append("torchao_check")
        assert base == "dense-base"
        return True

    monkeypatch.setattr(
        hf_lora_training,
        "disable_incompatible_optional_torchao_dispatch",
        disable,
    )
    model, disabled = hf_lora_training._load_peft_adapter(
        "dense-base", FakePeftModel, "adapter", is_trainable=True
    )

    assert calls == ["torchao_check", "peft_load"]
    assert disabled is True
    assert model._crowdtensor_outdated_optional_torchao_dispatch_disabled is True
