"""Official entry-point plugin for the pinned CrowdTensor Mistral Beta model."""

from __future__ import annotations

from typing import Any, Iterable

from crowdtensor.model_adapter import ModelAdapter, ModelAdapterError


PLUGIN_VERSION = "0.1.0b1"
MODEL_ID = "Locutusque/TinyMistral-248M-v2"
MODEL_REVISION = "0f57b17cb317bb322c7c1466b669c681f80c058f"


class MistralModelAdapter(ModelAdapter):
    adapter_id = "mistral_lora_v1"
    family = "mistral"
    model_types = ("mistral",)
    architectures = ("MistralForCausalLM",)
    default_model_id = MODEL_ID
    default_revision = MODEL_REVISION
    default_model_license = "apache-2.0"
    target_modules = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    default_config = {
        "model_type": "mistral",
        "architectures": ["MistralForCausalLM"],
        "num_hidden_layers": 12,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32005,
        "sliding_window": 32,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
    }
    recommended_stage_count = 2
    extra_capabilities = (
        "bounded_cpu_cuda_heterogeneous_live",
        "grouped_query_attention",
        "sliding_window_attention",
    )

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        value = super().validate_config(config)
        sliding_window = int(config.get("sliding_window") or 0)
        if sliding_window < 1:
            raise ModelAdapterError("mistral_adapter_sliding_window_invalid")
        if int(value["num_attention_heads"]) % int(value["num_key_value_heads"]):
            raise ModelAdapterError("mistral_adapter_gqa_heads_invalid")
        value["sliding_window"] = sliding_window
        value["max_position_embeddings"] = int(
            config.get("max_position_embeddings") or 0
        )
        if value["max_position_embeddings"] < sliding_window:
            raise ModelAdapterError("mistral_adapter_position_limit_invalid")
        return value

    def production_manifest(
        self, *, target_steps: int, accelerators: Iterable[str]
    ) -> dict[str, Any]:
        raise ModelAdapterError("mistral_full_production_scheduler_not_supported")


def adapter_factory() -> MistralModelAdapter:
    return MistralModelAdapter()


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "PLUGIN_VERSION",
    "MistralModelAdapter",
    "adapter_factory",
]
