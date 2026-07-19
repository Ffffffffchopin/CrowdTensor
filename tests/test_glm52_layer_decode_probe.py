from __future__ import annotations

from scripts import glm52_layer_decode_check as check
from scripts import glm52_layer_decode_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _report(*, stage_decode: bool = False, same_request: bool = False) -> dict:
    hidden = 4
    heads = 2
    qk = 3
    value = 5
    prefill = 3
    updated = 4
    return {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "qk_head_dim": qk,
        "v_head_dim": value,
        "prefill_length": prefill,
        "updated_cache_length": updated,
        "num_experts_per_tok": 2,
        "router_topk_count": 2,
        "executed_expert_count": 2,
        "decode_input_shape": [hidden],
        "prefill_key_cache_shape": [prefill, heads, qk],
        "prefill_value_cache_shape": [prefill, heads, value],
        "updated_key_cache_shape": [updated, heads, qk],
        "updated_value_cache_shape": [updated, heads, value],
        "decode_query_shape": [heads, qk],
        "attention_scores_shape": [heads, updated],
        "attention_weights_shape": [heads, updated],
        "attention_head_output_shape": [heads, value],
        "attention_flattened_shape": [heads * value],
        "attention_output_shape": [hidden],
        "attention_residual_shape": [hidden],
        "post_attention_norm_shape": [hidden],
        "routed_output_shape": [hidden],
        "shared_output_shape": [hidden],
        "full_moe_output_shape": [hidden],
        "layer_output_shape": [hidden],
        "executed_experts": [
            {
                "expert_id": 7,
                "expert_weight_hash": _hash("a"),
                "expert_output_shape": [hidden],
                "expert_output_hash": _hash("b"),
            },
            {
                "expert_id": 11,
                "expert_weight_hash": _hash("c"),
                "expert_output_shape": [hidden],
                "expert_output_hash": _hash("d"),
            },
        ],
        "shared_projection_summaries": [
            {"projection": "gate_proj", "weight_shape": [2, hidden], "output_shape": [2], "output_hash": _hash("e")},
            {"projection": "up_proj", "weight_shape": [2, hidden], "output_shape": [2], "output_hash": _hash("f")},
            {"projection": "down_proj", "weight_shape": [hidden, 2], "output_shape": [hidden], "output_hash": _hash("1")},
        ],
        "decode_input_hash": _hash("2"),
        "prefill_key_cache_hash": _hash("3"),
        "updated_key_cache_hash": _hash("4"),
        "decode_query_hash": _hash("5"),
        "attention_scores_hash": _hash("6"),
        "attention_weights_hash": _hash("7"),
        "attention_output_hash": _hash("8"),
        "attention_residual_hash": _hash("9"),
        "post_attention_norm_hash": _hash("a"),
        "router_topk_indices_hash": _hash("b"),
        "router_topk_weights_hash": _hash("c"),
        "routed_output_hash": _hash("d"),
        "shared_output_hash": _hash("e"),
        "full_moe_output_hash": _hash("f"),
        "layer_output_hash": _hash("1"),
        "kv_cache_prefill_verified": True,
        "kv_cache_update_verified": True,
        "attention_decode_verified": True,
        "attention_residual_verified": True,
        "post_attention_norm_verified": True,
        "router_topk_verified": True,
        "routed_expert_gather_verified": True,
        "shared_experts_mlp_verified": True,
        "full_moe_mlp_verified": True,
        "layer_decode_verified": True,
        "dsa_masked_attention_integrated": False,
        "multi_layer_stage_runtime_verified": False,
        "lm_head_verified": False,
        "generated_token_verified": False,
        "stage_decode_verified": stage_decode,
        "same_request_decode_verified": same_request,
        "blockers": [
            "glm52_layer_decode_is_single_layer_only",
            "glm52_layer_decode_uses_basic_attention_not_dsa_masked_attention",
            "glm52_layer_decode_missing_lm_head",
            "glm52_layer_decode_is_not_stage_decode",
            "glm52_layer_decode_is_not_same_request",
        ],
        "completion_boundary": {
            "layer_decode_is_single_layer_only": True,
            "layer_decode_uses_basic_attention_not_dsa_masked_attention": True,
            "layer_decode_is_not_stage_decode": True,
            "layer_decode_is_not_same_request": True,
            "requires_multi_layer_stage_runtime": True,
            "requires_dsa_masked_attention_integration": True,
            "requires_lm_head_token_selection": True,
            "requires_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
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


def test_checker_accepts_layer_decode_without_stage_or_same_request_success() -> None:
    assert check.validate_report(_report(), require_verified=True) == []


def test_checker_rejects_layer_decode_stage_overclaim() -> None:
    assert "stage_decode_verified_overclaim" in check.validate_report(_report(stage_decode=True))


def test_checker_rejects_layer_decode_same_request_overclaim() -> None:
    assert "same_request_decode_verified_overclaim" in check.validate_report(_report(same_request=True))
