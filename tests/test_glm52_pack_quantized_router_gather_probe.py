from __future__ import annotations

import torch

from scripts import glm52_pack_quantized_router_gather_check as check
from scripts import glm52_pack_quantized_router_gather_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def test_route_topk_matches_sigmoid_bias_and_scaling() -> None:
    config = {
        "hidden_size": 2,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
    }
    gate = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    bias = torch.tensor([0.0, 0.4, 0.0, 0.0])
    hidden = torch.tensor([1.0, 0.5])

    indices, weights = probe.route_topk(config, gate, bias, hidden)

    assert indices.tolist() == [1, 0]
    base = torch.sigmoid(torch.tensor([0.5, 1.0]))
    expected = base / base.sum() * 2.5
    assert torch.allclose(weights, expected)


def test_checker_accepts_router_gather_subset_without_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 4,
        "num_experts_per_tok": 2,
        "router_topk_count": 2,
        "executed_expert_count": 1,
        "router_topk_verified": True,
        "routed_expert_subset_verified": True,
        "stage_decode_verified": False,
        "router_topk_indices_hash": _hash("a"),
        "router_topk_weights_hash": _hash("b"),
        "routed_subset_output_shape": [4],
        "routed_subset_output_hash": _hash("c"),
        "executed_experts": [
            {
                "topk_position": 0,
                "expert_id": 1,
                "expert_weight_hash": _hash("d"),
                "expert_output_shape": [4],
                "expert_output_hash": _hash("e"),
            }
        ],
        "completion_boundary": {
            "routed_subset_is_not_full_moe_layer": True,
            "shared_experts_not_included": True,
            "attention_not_included": True,
            "stage_decode_not_included": True,
            "requires_transformer_block_runtime": True,
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


def test_checker_rejects_router_gather_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 1,
        "num_experts_per_tok": 1,
        "router_topk_count": 1,
        "executed_expert_count": 1,
        "router_topk_verified": True,
        "routed_expert_subset_verified": True,
        "stage_decode_verified": True,
        "router_topk_indices_hash": _hash("a"),
        "router_topk_weights_hash": _hash("b"),
        "routed_subset_output_shape": [1],
        "routed_subset_output_hash": _hash("c"),
        "executed_experts": [],
        "completion_boundary": {},
        "safety": {},
    }

    assert "stage_decode_overclaim" in check.validate_report(report)
