from __future__ import annotations

import torch

from scripts import glm52_pack_quantized_moe_mlp_check as check
from scripts import glm52_pack_quantized_moe_mlp_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def test_run_shared_experts_from_weights_matches_glm_mlp_formula() -> None:
    hidden = torch.tensor([1.0, 2.0], dtype=torch.float32)
    weights = {
        "gate_proj": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "up_proj": torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32),
        "down_proj": torch.tensor([[1.0, 1.0], [2.0, -1.0]], dtype=torch.float32),
    }

    output, summaries = probe.run_shared_experts_from_weights(hidden, weights)

    gated = torch.nn.functional.silu(torch.tensor([1.0, 2.0])) * torch.tensor([2.0, 6.0])
    expected = torch.matmul(weights["down_proj"], gated)
    assert torch.allclose(output, expected)
    assert [item["projection"] for item in summaries] == probe.PROJECTIONS
    assert summaries[-1]["output_shape"] == [2]


def test_checker_accepts_full_moe_mlp_without_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "hidden_size": 4,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "router_topk_count": 2,
        "executed_expert_count": 2,
        "router_topk_verified": True,
        "routed_expert_gather_verified": True,
        "shared_experts_mlp_verified": True,
        "pack_quantized_moe_mlp_verified": True,
        "full_moe_mlp_verified": True,
        "stage_decode_verified": False,
        "router_topk_indices_hash": _hash("a"),
        "router_topk_weights_hash": _hash("b"),
        "executed_experts": [
            {
                "topk_position": 0,
                "expert_id": 1,
                "expert_weight_hash": _hash("c"),
                "expert_output_shape": [4],
                "expert_output_hash": _hash("d"),
            },
            {
                "topk_position": 1,
                "expert_id": 2,
                "expert_weight_hash": _hash("e"),
                "expert_output_shape": [4],
                "expert_output_hash": _hash("f"),
            },
        ],
        "routed_output_shape": [4],
        "routed_output_hash": _hash("1"),
        "shared_projection_summaries": [
            {
                "projection": projection,
                "weight_dtype": "bfloat16",
                "weight_shape": [2, 4] if projection != "down_proj" else [4, 2],
                "output_shape": [2] if projection != "down_proj" else [4],
                "output_hash": _hash(str(index)),
            }
            for index, projection in enumerate(probe.PROJECTIONS)
        ],
        "shared_output_shape": [4],
        "shared_output_hash": _hash("2"),
        "full_moe_output_shape": [4],
        "full_moe_output_hash": _hash("3"),
        "completion_boundary": {
            "full_moe_mlp_is_not_attention": True,
            "full_moe_mlp_is_not_transformer_block": True,
            "full_moe_mlp_is_not_stage_decode": True,
            "requires_attention_runtime": True,
            "requires_residual_norm_runtime": True,
            "requires_stage_local_kv_cache": True,
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


def test_checker_rejects_moe_mlp_stage_decode_overclaim() -> None:
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
        "routed_expert_gather_verified": True,
        "shared_experts_mlp_verified": True,
        "pack_quantized_moe_mlp_verified": True,
        "full_moe_mlp_verified": True,
        "stage_decode_verified": True,
        "router_topk_indices_hash": _hash("a"),
        "router_topk_weights_hash": _hash("b"),
        "executed_experts": [],
        "routed_output_shape": [1],
        "routed_output_hash": _hash("1"),
        "shared_projection_summaries": [],
        "shared_output_shape": [1],
        "shared_output_hash": _hash("2"),
        "full_moe_output_shape": [1],
        "full_moe_output_hash": _hash("3"),
        "completion_boundary": {},
        "safety": {},
    }

    assert "stage_decode_overclaim" in check.validate_report(report)
