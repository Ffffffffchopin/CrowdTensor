import sys
from pathlib import Path

import pytest

from crowdtensor.model_adapter import ModelAdapterError, check_model_adapter_conformance


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
