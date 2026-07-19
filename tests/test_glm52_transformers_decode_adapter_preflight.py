from __future__ import annotations

from scripts import glm52_transformers_decode_adapter_preflight as preflight
from scripts import glm52_transformers_decode_adapter_preflight_check as check


def _config() -> dict:
    return {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 16,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "indexer_types": ["full", "full", "full", "shared"],
        "layer_types": ["deepseek_sparse_attention"] * 4,
        "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
            "ignore": ["model.layers.0.mlp.gate_proj"],
        },
    }


def _index() -> dict:
    weight_map = {}
    for layer in range(3):
        for key in preflight.layer_expected_keys(_config(), layer).values():
            for item in key:
                weight_map[item] = "model-00001-of-00001.safetensors"
    for key in preflight.layer_expected_keys(_config(), 3).values():
        for item in key:
            weight_map[item] = "model-00001-of-00001.safetensors"
    return {"metadata": {"total_size": 1234}, "weight_map": weight_map}


def test_normalizes_awq_layer_types_without_losing_glm_config() -> None:
    normalized, action = preflight.normalize_awq_config(_config())

    assert action == "removed_invalid_layer_types"
    assert "layer_types" not in normalized
    assert normalized["model_type"] == "glm_moe_dsa"


def test_stage_mapping_accepts_dense_and_sparse_pack_quantized_keys() -> None:
    args = preflight.parse_args(["--stage-id", "-1", "--stage-count", "1"])

    mapping = preflight.mapping_summary(preflight.normalize_awq_config(_config())[0], _index(), args)

    assert mapping["stage_weight_mapping_ready"] is True
    assert mapping["dense_layer_count"] == 3
    assert mapping["sparse_layer_count"] == 1
    assert mapping["missing_required_key_count"] == 0
    assert mapping["pack_required_key_count"] > 0


def test_checker_accepts_foundation_ready_but_decode_not_ready_report() -> None:
    report = {
        "schema": preflight.SCHEMA,
        "ok": True,
        "glm52_transformers_decode_adapter_preflight_ready": True,
        "decode_adapter_ready": False,
        "adapter_foundation_ready": True,
        "public_artifact_safe": True,
        "model": {
            "model_id": preflight.MODEL_ID,
            "config_ready": True,
            "index_ready": True,
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 4,
            "quantization_format": "pack-quantized",
            "quantization_weight_bits": [4],
        },
        "transformers_runtime": {
            "transformers_available": True,
            "glm_moe_dsa_config_class_available": True,
            "glm_moe_dsa_model_class_available": True,
            "awq_config_normalized_ready": True,
            "tiny_forward_ready": True,
            "tiny_forward_logits_shape": [1, 1, 32],
        },
        "pack_quantized_runtime": {
            "ready": False,
            "dependencies": [{"module": "compressed_tensors", "available": False}],
        },
        "stage_weight_mapping": {
            "stage_weight_mapping_ready": True,
            "required_key_count": 64,
            "pack_required_key_count": 16,
            "missing_required_key_count": 0,
            "sparse_layer_count": 1,
        },
        "blockers": ["glm52_pack_quantized_runtime_dependency_missing"],
        "completion_boundary": {
            "preflight_is_not_decode_success": True,
            "tiny_random_forward_is_not_glm52_inference": True,
            "weight_mapping_is_not_weight_loading": True,
            "requires_pack_quantized_dequant_runtime": True,
            "requires_stage_decode_verified": True,
            "requires_same_request_generated_token_hash": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }

    assert check.validate_report(report, require_foundation=True) == []
    assert "decode_adapter_not_ready" in check.validate_report(report, require_decode_ready=True)
