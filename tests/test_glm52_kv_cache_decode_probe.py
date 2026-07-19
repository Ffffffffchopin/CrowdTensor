from __future__ import annotations

import torch

from scripts import glm52_kv_cache_decode_check as check
from scripts import glm52_kv_cache_decode_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def test_pack_matrix_matches_linear_projection() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    hidden = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

    assert torch.allclose(probe.pack_matrix(weight, hidden), hidden @ weight.T)


def test_build_hidden_sequence_offsets_positions() -> None:
    hidden = probe.build_hidden_sequence(3, 4)

    assert hidden.shape == (3, 4)
    assert torch.allclose(hidden[1] - hidden[0], torch.full((4,), 0.001))


def test_checker_accepts_kv_cache_decode_without_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 4,
        "num_attention_heads": 2,
        "qk_head_dim": 3,
        "v_head_dim": 5,
        "prefill_length": 3,
        "updated_cache_length": 4,
        "prefill_key_cache_shape": [3, 2, 3],
        "prefill_value_cache_shape": [3, 2, 5],
        "updated_key_cache_shape": [4, 2, 3],
        "updated_value_cache_shape": [4, 2, 5],
        "decode_query_shape": [2, 3],
        "attention_scores_shape": [2, 4],
        "attention_weights_shape": [2, 4],
        "head_output_shape": [2, 5],
        "attention_flattened_shape": [10],
        "o_proj_weight_shape": [4, 10],
        "o_proj_output_shape": [4],
        "prefill_key_cache_hash": _hash("a"),
        "prefill_value_cache_hash": _hash("b"),
        "updated_key_cache_hash": _hash("c"),
        "updated_value_cache_hash": _hash("d"),
        "decode_query_hash": _hash("e"),
        "attention_scores_hash": _hash("f"),
        "attention_weights_hash": _hash("1"),
        "head_output_hash": _hash("2"),
        "o_proj_output_hash": _hash("3"),
        "kv_cache_prefill_verified": True,
        "kv_cache_update_verified": True,
        "kv_cache_decode_attention_verified": True,
        "o_proj_verified": True,
        "stage_decode_verified": False,
        "generated_token_verified": False,
        "completion_boundary": {
            "kv_cache_decode_is_not_dsa_masked_attention": True,
            "kv_cache_decode_is_not_transformer_block": True,
            "kv_cache_decode_is_not_stage_decode": True,
            "requires_mlp_residual_runtime": True,
            "requires_lm_head_token_selection": True,
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


def test_checker_rejects_kv_cache_decode_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 1,
        "num_attention_heads": 1,
        "qk_head_dim": 1,
        "v_head_dim": 1,
        "prefill_length": 1,
        "updated_cache_length": 2,
        "kv_cache_prefill_verified": True,
        "kv_cache_update_verified": True,
        "kv_cache_decode_attention_verified": True,
        "o_proj_verified": True,
        "stage_decode_verified": True,
        "generated_token_verified": False,
        "completion_boundary": {},
        "safety": {},
    }

    assert "stage_decode_verified_overclaim" in check.validate_report(report)
