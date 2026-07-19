from __future__ import annotations

import torch

from scripts import glm52_attention_projection_check as check
from scripts import glm52_attention_projection_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def test_rms_norm_matches_glm_formula() -> None:
    hidden = torch.tensor([3.0, 4.0], dtype=torch.float32)
    weight = torch.tensor([2.0, 0.5], dtype=torch.float32)

    output = probe.rms_norm(hidden, weight, 1e-5)

    variance = torch.tensor([(9.0 + 16.0) / 2.0])
    expected = weight * hidden * torch.rsqrt(variance + 1e-5)
    assert torch.allclose(output, expected)


def test_checker_accepts_projection_without_attention_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 4,
        "num_attention_heads": 2,
        "q_lora_rank": 3,
        "kv_lora_rank": 2,
        "qk_head_dim": 5,
        "qk_nope_head_dim": 3,
        "qk_rope_head_dim": 2,
        "v_head_dim": 4,
        "input_norm_shape": [4],
        "q_a_weight_shape": [3, 4],
        "q_a_output_shape": [3],
        "q_a_norm_shape": [3],
        "q_b_weight_shape": [10, 3],
        "q_b_output_shape": [10],
        "query_shape": [2, 5],
        "q_nope_shape": [2, 3],
        "q_pe_shape": [2, 2],
        "kv_a_weight_shape": [4, 4],
        "kv_a_output_shape": [4],
        "k_compressed_shape": [2],
        "k_pe_shape": [2],
        "k_compressed_norm_shape": [2],
        "kv_b_weight_shape": [14, 2],
        "kv_b_output_shape": [14],
        "k_nope_shape": [2, 3],
        "value_shape": [2, 4],
        "input_layernorm_verified": True,
        "q_lora_projection_verified": True,
        "kv_lora_projection_verified": True,
        "attention_projection_verified": True,
        "rope_applied": False,
        "attention_scores_verified": False,
        "o_proj_verified": False,
        "stage_decode_verified": False,
        "input_norm_hash": _hash("a"),
        "q_a_output_hash": _hash("b"),
        "q_a_norm_hash": _hash("c"),
        "q_b_output_hash": _hash("d"),
        "q_nope_hash": _hash("e"),
        "q_pe_hash": _hash("f"),
        "kv_a_output_hash": _hash("1"),
        "k_compressed_norm_hash": _hash("2"),
        "kv_b_output_hash": _hash("3"),
        "k_nope_hash": _hash("4"),
        "value_hash": _hash("5"),
        "completion_boundary": {
            "attention_projection_is_not_full_attention": True,
            "rope_not_applied": True,
            "attention_scores_not_computed": True,
            "o_proj_not_computed": True,
            "kv_cache_not_updated": True,
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


def test_checker_rejects_attention_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 1,
        "num_attention_heads": 1,
        "q_lora_rank": 1,
        "kv_lora_rank": 1,
        "qk_head_dim": 1,
        "qk_nope_head_dim": 1,
        "qk_rope_head_dim": 0,
        "v_head_dim": 1,
        "input_layernorm_verified": True,
        "q_lora_projection_verified": True,
        "kv_lora_projection_verified": True,
        "attention_projection_verified": True,
        "rope_applied": False,
        "attention_scores_verified": False,
        "o_proj_verified": False,
        "stage_decode_verified": True,
        "completion_boundary": {},
        "safety": {},
    }

    assert "stage_decode_verified_overclaim" in check.validate_report(report)
