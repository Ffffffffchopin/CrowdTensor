import pytest
import torch
from crowdtensor.model_adapter import (
    MODEL_ADAPTER_DESCRIPTOR_SCHEMA,
    ModelAdapterError,
    adapter_registry_report,
    disable_incompatible_optional_torchao_dispatch,
    get_model_adapter,
    resolve_model_adapter,
)


QWEN_CONFIG = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
}

SMOL_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 30,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "vocab_size": 49152,
}

SMOL3_CONFIG = {
    "model_type": "smollm3",
    "architectures": ["SmolLM3ForCausalLM"],
    "num_hidden_layers": 36,
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "vocab_size": 128256,
}


def test_registry_is_versioned_and_preserves_builtin_families() -> None:
    report = adapter_registry_report()
    assert report["api_version"] == "model_adapter_v1.0"
    assert report["builtin_adapter_count"] == 3
    assert {"qwen2", "smollm2", "smollm3"}.issubset(
        report["supported_model_families"]
    )
    builtin_ids = {
        item["adapter_id"]
        for item in report["adapters"]
        if item["registration"]["kind"] == "builtin"
    }
    assert builtin_ids == {
        "qwen2_lora_v1",
        "smollm2_lora_v1",
        "smollm3_lora_v1",
    }
    assert all(item["schema"] == MODEL_ADAPTER_DESCRIPTOR_SCHEMA for item in report["adapters"])
    assert "full_parameter_training" in report["unsupported_capabilities"]


@pytest.mark.parametrize(
    ("adapter_id", "config"),
    [
        ("qwen2_lora_v1", QWEN_CONFIG),
        ("smollm2_lora_v1", SMOL_CONFIG),
        ("smollm3_lora_v1", SMOL3_CONFIG),
    ],
)
def test_adapters_partition_contiguously_and_estimate_resources(adapter_id: str, config: dict) -> None:
    adapter = get_model_adapter(adapter_id)
    stages = adapter.partition(config, stage_count=2)
    assert [item.stage_id for item in stages] == [0, 1]
    assert stages[0].owns_embedding is True
    assert stages[-1].owns_lm_head is True
    assert stages[0].layer_end == stages[1].layer_start
    assert stages[-1].layer_end == config["num_hidden_layers"]
    assert all(adapter.estimate_resources(config, item)["weight_bytes"] > 0 for item in stages)
    assert resolve_model_adapter(model_id=adapter.default_model_id, config=config).adapter_id == adapter_id


def test_qwen_production_manifest_is_now_adapter_backed_without_contract_regression() -> None:
    manifest = get_model_adapter("qwen2_lora_v1").production_manifest(
        target_steps=100,
        accelerators=["cpu", "cuda", "jax_tpu"],
    )
    assert manifest["model"]["model_type"] == "qwen2"
    assert manifest["scheduler"]["required_device_types"] == ["cpu", "cuda", "jax_tpu"]
    assert manifest["training"]["target_steps"] == 100


def test_adapters_fail_closed_for_unknown_models_and_unsupported_scheduler() -> None:
    with pytest.raises(ModelAdapterError, match="id_unsupported"):
        get_model_adapter("arbitrary")
    with pytest.raises(ModelAdapterError, match="unsupported_or_ambiguous"):
        resolve_model_adapter(model_id="unknown/model", config={"model_type": "unknown"})
    with pytest.raises(ModelAdapterError, match="full_production_scheduler_not_supported"):
        get_model_adapter("smollm2_lora_v1").production_manifest(
            target_steps=2, accelerators=["cuda"]
        )
    with pytest.raises(ModelAdapterError, match="full_production_scheduler_not_supported"):
        get_model_adapter("smollm3_lora_v1").production_manifest(
            target_steps=2, accelerators=["cuda"]
        )


def test_outdated_optional_torchao_is_disabled_for_dense_weights(monkeypatch) -> None:
    import peft.tuners.lora.torchao as peft_torchao

    original_find_spec = __import__("importlib").util.find_spec
    original_version = __import__("importlib").metadata.version
    original_available = peft_torchao.is_torchao_available
    monkeypatch.setattr(
        __import__("importlib").util,
        "find_spec",
        lambda name: object() if name == "torchao" else original_find_spec(name),
    )
    monkeypatch.setattr(
        __import__("importlib").metadata,
        "version",
        lambda name: "0.10.0" if name == "torchao" else original_version(name),
    )
    monkeypatch.setattr(peft_torchao, "is_torchao_available", original_available)

    assert disable_incompatible_optional_torchao_dispatch(torch.nn.Linear(4, 4)) is True
    assert peft_torchao.is_torchao_available() is False


def test_reload_adapter_applies_optional_dispatch_compatibility(monkeypatch, tmp_path) -> None:
    adapter = get_model_adapter("smollm2_lora_v1")
    base = torch.nn.Linear(4, 4)
    loaded = torch.nn.Linear(4, 4)
    calls: list[bool] = []

    monkeypatch.setattr(adapter, "load_model", lambda **_kwargs: base)
    monkeypatch.setattr(
        "crowdtensor.model_adapter.disable_incompatible_optional_torchao_dispatch",
        lambda model: calls.append(model is base) or True,
    )
    monkeypatch.setattr(
        "peft.PeftModel.from_pretrained",
        lambda model, _path, is_trainable: loaded,
    )
    result = adapter.reload_adapter(
        model_id=adapter.default_model_id,
        revision=adapter.default_revision,
        adapter_dir=tmp_path,
        device="cpu",
        dtype=torch.float32,
    )
    assert result is loaded
    assert calls == [True]
    assert result._crowdtensor_outdated_optional_torchao_dispatch_disabled is True
