from __future__ import annotations

import torch

from scripts import glm52_attention_single_token_check as check
from scripts import glm52_attention_single_token_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def test_rope_helpers_match_split_half_formula() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cos = torch.ones(4)
    sin = torch.zeros(4)

    assert torch.allclose(probe.apply_rope_1d(x, cos, sin), x)
    assert torch.allclose(probe.rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_checker_accepts_single_token_attention_without_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 4,
        "num_attention_heads": 2,
        "qk_head_dim": 3,
        "v_head_dim": 5,
        "query_states_shape": [2, 3],
        "key_states_shape": [2, 3],
        "value_states_shape": [2, 5],
        "attention_scores_shape": [2, 1],
        "attention_weights_shape": [2, 1],
        "head_output_shape": [2, 5],
        "attention_flattened_shape": [10],
        "o_proj_weight_shape": [4, 10],
        "o_proj_output_shape": [4],
        "q_pe_rope_hash": _hash("a"),
        "k_pe_rope_hash": _hash("b"),
        "query_states_hash": _hash("c"),
        "key_states_hash": _hash("d"),
        "value_states_hash": _hash("e"),
        "attention_scores_hash": _hash("f"),
        "attention_weights_hash": _hash("1"),
        "head_output_hash": _hash("2"),
        "o_proj_output_hash": _hash("3"),
        "rope_applied": True,
        "attention_scores_verified": True,
        "attention_weights_verified": True,
        "o_proj_verified": True,
        "single_token_attention_verified": True,
        "kv_cache_updated": False,
        "dsa_indexer_verified": False,
        "stage_decode_verified": False,
        "completion_boundary": {
            "single_token_attention_is_not_multi_token_prefill": True,
            "single_token_attention_is_not_dsa_indexer": True,
            "single_token_attention_is_not_kv_cache_decode": True,
            "single_token_attention_is_not_transformer_block": True,
            "single_token_attention_is_not_stage_decode": True,
            "requires_stage_decode_verified": True,
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

    assert check.validate_report(report, require_verified=True) == []


def test_checker_rejects_single_token_attention_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 1,
        "num_attention_heads": 1,
        "qk_head_dim": 1,
        "v_head_dim": 1,
        "rope_applied": True,
        "attention_scores_verified": True,
        "attention_weights_verified": True,
        "o_proj_verified": True,
        "single_token_attention_verified": True,
        "kv_cache_updated": False,
        "dsa_indexer_verified": False,
        "stage_decode_verified": True,
        "completion_boundary": {},
        "safety": {},
    }

    assert "stage_decode_verified_overclaim" in check.validate_report(report)
