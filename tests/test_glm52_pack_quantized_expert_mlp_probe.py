from __future__ import annotations

import torch

from scripts import glm52_pack_quantized_expert_mlp_check as check
from scripts import glm52_pack_quantized_expert_mlp_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _projection_tensors(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    rows, cols = weight.shape
    packed = torch.zeros((rows, 1), dtype=torch.int32)
    shifted = (weight.to(torch.int32) + 8).clamp(0, 15)
    for row in range(rows):
        for col in range(cols):
            packed[row, 0] |= shifted[row, col] << (4 * col)
    zero_point = torch.zeros((1, 1), dtype=torch.int32)
    for row in range(rows):
        zero_point[0, 0] |= 8 << (4 * row)
    scale = torch.ones((rows, 1), dtype=torch.float32)
    return {
        "weight_packed": packed,
        "weight_scale": scale,
        "weight_zero_point": zero_point,
        "weight_shape": torch.tensor([rows, cols], dtype=torch.int64),
    }


def test_projection_linear_uses_full_dequantized_weight() -> None:
    tensors = _projection_tensors(torch.tensor([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=torch.int8))
    input_vec = torch.tensor([1.0, 0.5, -0.5, 2.0], dtype=torch.float32)

    output, weight_shape, output_shape = probe.projection_linear(tensors, input_vec)

    assert weight_shape == [2, 4]
    assert output_shape == [2]
    assert torch.allclose(output, torch.tensor([8.5, -8.5]))


def test_checker_accepts_single_expert_mlp_without_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "pack_quantized_expert_mlp_verified": True,
        "single_expert_mlp_verified": True,
        "stage_decode_verified": False,
        "hidden_size": 4,
        "projection_summaries": [
            {
                "projection": projection,
                "weight_shape": [2, 4] if projection != "down_proj" else [4, 2],
                "output_shape": [2] if projection != "down_proj" else [4],
                "output_hash": _hash(str(index)),
                "pack_quantized_group_loaded": True,
            }
            for index, projection in enumerate(probe.PROJECTIONS)
        ],
        "final_output_shape": [4],
        "final_output_hash": _hash("f"),
        "completion_boundary": {
            "single_expert_mlp_is_not_full_moe_layer": True,
            "single_expert_mlp_is_not_attention": True,
            "single_expert_mlp_is_not_topk_router": True,
            "single_expert_mlp_is_not_stage_decode": True,
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


def test_checker_rejects_expert_mlp_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "pack_quantized_expert_mlp_verified": True,
        "single_expert_mlp_verified": True,
        "stage_decode_verified": True,
        "hidden_size": 1,
        "projection_summaries": [],
        "final_output_shape": [1],
        "final_output_hash": _hash("f"),
        "completion_boundary": {},
        "safety": {},
    }

    assert "stage_decode_overclaim" in check.validate_report(report)
