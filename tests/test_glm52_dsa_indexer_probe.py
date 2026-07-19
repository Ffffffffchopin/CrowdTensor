from __future__ import annotations

import torch

from scripts import glm52_dsa_indexer_check as check
from scripts import glm52_dsa_indexer_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def test_layer_norm_rows_matches_torch_layer_norm() -> None:
    hidden = torch.tensor([[1.0, 2.0, 4.0], [2.0, 2.0, 3.0]], dtype=torch.float32)
    weight = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
    bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

    output = probe.layer_norm_rows(hidden, weight, bias, eps=1e-6)
    expected = torch.nn.functional.layer_norm(hidden, [3], weight=weight, bias=bias, eps=1e-6)

    assert torch.allclose(output, expected)


def test_apply_rope_sequence_broadcasts_positions_and_heads() -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    cos = torch.ones((2, 4), dtype=torch.float32)
    sin = torch.zeros((2, 4), dtype=torch.float32)

    assert torch.allclose(probe.apply_rope_sequence(x, cos, sin), x)


def test_checker_accepts_dsa_indexer_without_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "layer_indexer_type": "full",
        "hidden_size": 4,
        "sequence_length": 3,
        "q_lora_rank": 2,
        "index_n_heads": 2,
        "index_head_dim": 5,
        "effective_topk": 3,
        "hidden_norm_shape": [3, 4],
        "q_resid_shape": [3, 2],
        "indexer_query_shape": [3, 2, 5],
        "indexer_key_shape": [3, 5],
        "head_weights_shape": [3, 2],
        "index_score_shape": [3, 3],
        "topk_indices_shape": [3, 3],
        "wq_b_weight_shape": [10, 2],
        "wk_weight_shape": [5, 4],
        "weights_proj_shape": [2, 4],
        "hidden_norm_hash": _hash("a"),
        "q_resid_hash": _hash("b"),
        "indexer_query_hash": _hash("c"),
        "indexer_key_hash": _hash("d"),
        "head_weights_hash": _hash("e"),
        "index_score_hash": _hash("f"),
        "topk_indices_hash": _hash("1"),
        "dsa_indexer_verified": True,
        "dsa_topk_verified": True,
        "indexer_cache_updated": False,
        "attention_output_verified": False,
        "stage_decode_verified": False,
        "completion_boundary": {
            "dsa_indexer_small_sequence_is_not_full_prefill": True,
            "dsa_indexer_is_not_kv_cache_decode": True,
            "dsa_indexer_is_not_attention_output": True,
            "dsa_indexer_is_not_transformer_block": True,
            "dsa_indexer_is_not_stage_decode": True,
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


def test_checker_rejects_dsa_indexer_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "layer_indexer_type": "full",
        "hidden_size": 1,
        "sequence_length": 1,
        "q_lora_rank": 1,
        "index_n_heads": 1,
        "index_head_dim": 1,
        "effective_topk": 1,
        "dsa_indexer_verified": True,
        "dsa_topk_verified": True,
        "indexer_cache_updated": False,
        "attention_output_verified": False,
        "stage_decode_verified": True,
        "completion_boundary": {},
        "safety": {},
    }

    assert "stage_decode_verified_overclaim" in check.validate_report(report)
