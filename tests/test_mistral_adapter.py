import sys
from pathlib import Path

import pytest
import torch

from crowdtensor.model_adapter import ModelAdapterError, check_model_adapter_conformance
import crowdtensor.adapter_stage_training as stage_training


PLUGIN_SOURCE = Path("plugins/mistral_adapter/src").resolve()
if str(PLUGIN_SOURCE) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SOURCE))

from crowdtensor_mistral_adapter import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    MistralModelAdapter,
)


def test_mistral_plugin_contract_is_conformant_and_pinned() -> None:
    adapter = MistralModelAdapter()
    report = check_model_adapter_conformance(adapter)
    assert report["ok"] is True
    assert adapter.adapter_id == "mistral_lora_v1"
    assert adapter.default_model_id == MODEL_ID == "Locutusque/TinyMistral-248M-v2"
    assert adapter.default_revision == MODEL_REVISION
    assert adapter.default_model_license == "apache-2.0"
    assert adapter.architectures == ("MistralForCausalLM",)
    assert "sliding_window_attention" in adapter.extra_capabilities


def test_mistral_partition_and_resource_estimates_cover_all_layers() -> None:
    adapter = MistralModelAdapter()
    config = adapter.canonical_config()
    stages = adapter.partition(config, stage_count=2)
    assert [(item.layer_start, item.layer_end) for item in stages] == [(0, 6), (6, 12)]
    assert stages[0].owns_embedding is True
    assert stages[1].owns_lm_head is True
    assert all(adapter.estimate_resources(config, item)["weight_bytes"] > 0 for item in stages)
    assert config["sliding_window"] == 32


def test_mistral_rejects_invalid_gqa_window_and_production_overclaim() -> None:
    adapter = MistralModelAdapter()
    config = adapter.canonical_config()
    config["num_key_value_heads"] = 7
    with pytest.raises(ModelAdapterError, match="gqa_heads_invalid"):
        adapter.validate_config(config)
    with pytest.raises(ModelAdapterError, match="production_scheduler_not_supported"):
        adapter.production_manifest(target_steps=8, accelerators=["cpu", "cuda"])


def test_mistral_real_architecture_supports_both_lora_pipeline_stages(
    monkeypatch, tmp_path
) -> None:
    from transformers import MistralConfig, MistralForCausalLM

    def new_config() -> MistralConfig:
        return MistralConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            sliding_window=8,
            max_position_embeddings=32,
        )

    adapter = MistralModelAdapter()
    monkeypatch.setattr(stage_training, "get_model_adapter", lambda _adapter_id: adapter)
    monkeypatch.setattr(
        adapter,
        "load_model",
        lambda **kwargs: MistralForCausalLM(new_config()).to(kwargs["device"]),
    )
    for stage_id in (0, 1):
        model, causal, optimizer, start, end, details = (
            stage_training.configure_adapter_stage_model(
                adapter_id=adapter.adapter_id,
                model_id=adapter.default_model_id,
                model_revision=adapter.default_revision,
                model_config=new_config().to_dict(),
                stage_id=stage_id,
                split_index=1,
                device="cpu",
                cache_dir=str(tmp_path / f"cache-{stage_id}"),
                rank=2,
                alpha=4,
            )
        )
        before = stage_training.tensor_state_hash(
            stage_training.owned_lora_state(model, start=start, end=end)
        )
        if stage_id == 0:
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            hidden = causal.model(
                input_ids=input_ids, use_cache=False
            ).last_hidden_state
            loss = hidden.square().mean()
        else:
            hidden = torch.randn(1, 4, 32, requires_grad=True)
            final = causal.model(
                inputs_embeds=hidden, use_cache=False
            ).last_hidden_state
            loss = causal.lm_head(final).float().square().mean()
        loss.backward()
        optimizer.step()
        after = stage_training.tensor_state_hash(
            stage_training.owned_lora_state(model, start=start, end=end)
        )
        assert torch.isfinite(loss).item() is True
        assert before != after
        assert details["family"] == "mistral"
        assert details["architecture"] == "MistralForCausalLM"
        assert details["layer_start"] == stage_id
        assert details["layer_end"] == stage_id + 1
