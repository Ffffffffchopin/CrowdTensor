from __future__ import annotations

from scripts import glm52_pack_quantized_group_check as check
from scripts import glm52_pack_quantized_group_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _report() -> dict:
    return {
        "schema": probe.SCHEMA,
        "ok": True,
        "glm52_pack_quantized_group_probe_ready": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "quantization_format": "pack-quantized",
        "pack_quantized_group_loaded": True,
        "pack_quantized_group_dequantized": False,
        "stage_decode_verified": False,
        "loaded_fields": probe.PACK_FIELDS,
        "group_tensor_count": len(probe.PACK_FIELDS),
        "group_value_total_bytes": 128,
        "group_value_hash": _hash("a"),
        "tensor_summaries": [
            {
                "field": field,
                "value_loaded": True,
                "value_sha256": _hash(str(index)),
                "tensor_nbytes": 32,
            }
            for index, field in enumerate(probe.PACK_FIELDS)
        ],
        "completion_boundary": {
            "pack_group_load_is_not_dequant_success": True,
            "weight_value_hash_is_not_raw_value_publication": True,
            "requires_dequant_linear_runtime": True,
            "requires_stage_decode_verified": True,
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
        "public_artifact_safe": True,
    }


def test_target_keys_selects_pack_quantized_expert_group() -> None:
    keys = probe.target_keys(3, 0, "gate_proj")

    assert keys == [
        "model.layers.3.mlp.experts.0.gate_proj.weight_packed",
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.3.mlp.experts.0.gate_proj.weight_zero_point",
        "model.layers.3.mlp.experts.0.gate_proj.weight_shape",
    ]


def test_checker_accepts_public_safe_loaded_pack_group_without_decode_overclaim() -> None:
    assert check.validate_report(_report(), require_loaded=True) == []


def test_checker_rejects_decode_overclaim() -> None:
    report = _report()
    report["stage_decode_verified"] = True

    assert "stage_decode_overclaim" in check.validate_report(report)
