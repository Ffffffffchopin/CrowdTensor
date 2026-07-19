from __future__ import annotations

from scripts import glm52_multi_layer_stage_decode_check as check
from scripts import glm52_multi_layer_stage_decode_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _layer(layer_id: int) -> dict:
    return {
        "layer_id": layer_id,
        "layer_decode_token_verified": True,
        "dsa_indexer_type": "full" if layer_id == 6 else "shared",
        "dsa_indexer_source_layer_id": 6,
        "dsa_indexer_source_type": "full",
        "dsa_mask_topk_count": 4,
        "dsa_mask_pruned_position_count": 5,
        "attention_scores_shape": [2, 9],
        "attention_output_shape": [4],
        "attention_residual_shape": [4],
        "post_attention_norm_shape": [4],
        "full_moe_output_shape": [4],
        "layer_output_shape": [4],
        "router_topk_count": 8,
        "executed_expert_count": 8,
        "dsa_index_score_hash_present": True,
        "dsa_attention_mask_hash_present": True,
        "attention_output_hash_present": True,
        "full_moe_output_hash_present": True,
        "layer_output_hash": _hash(str(layer_id)),
    }


def _report(
    *,
    full_prefill: bool = False,
    generated_token: bool = False,
    stage_decode: bool = False,
    same_request: bool = False,
    live_kaggle: bool = False,
) -> dict:
    hidden = 4
    vocab = 8
    return {
        "schema": probe.SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_repo": probe.DEFAULT_MODEL_REPO,
        "model_type": "glm_moe_dsa",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_hidden_layers": 64,
        "stage_hidden_source": "dsa_masked_multi_layer_decode_token_chain",
        "stage_layer_range": [6, 8],
        "stage_layer_count": 2,
        "executed_layer_count": 2,
        "stage_prefill_length": 8,
        "stage_updated_cache_length": 9,
        "dsa_mask_topk_requested": 4,
        "decode_token_chain_only": True,
        "prefill_hidden_carrier_full_layer_outputs_verified": full_prefill,
        "initial_decode_hidden_shape": [hidden],
        "initial_decode_hidden_hash": _hash("a"),
        "layer_summaries": [_layer(6), _layer(7)],
        "all_layers_dsa_masked_attention_integrated": True,
        "all_layers_moe_mlp_verified": True,
        "all_layer_outputs_chained": True,
        "stage_hidden_shape": [hidden],
        "stage_hidden_hash": _hash("b"),
        "norm_weight_shape": [hidden],
        "normalized_stage_hidden_shape": [hidden],
        "normalized_stage_hidden_hash": _hash("c"),
        "lm_head_shape": [vocab, hidden],
        "lm_head_dtype": "BF16",
        "lm_head_nbytes": vocab * hidden * 2,
        "lm_head_file_count": 1,
        "lm_head_rows_scanned": vocab,
        "lm_head_block_count": 2,
        "lm_head_row_block_size": 4,
        "top_k": 3,
        "top_k_count": 3,
        "selected_token_id_hash": _hash("d"),
        "selected_logit_hash": _hash("e"),
        "top_token_ids_hash": _hash("f"),
        "top_logits_hash": _hash("0"),
        "multi_layer_stage_hidden_verified": True,
        "multi_layer_decode_token_chain_verified": True,
        "stage_hidden_to_lm_head_verified": True,
        "lm_head_streamed_full_vocab": True,
        "stage_hidden_lm_head_token_selection_verified": True,
        "partial_multi_layer_token_hash_verified": True,
        "full_prefill_stage_hidden_verified": full_prefill,
        "full_model_hidden_verified": False,
        "generated_token_verified": generated_token,
        "stage_decode_verified": stage_decode,
        "same_request_decode_verified": same_request,
        "live_kaggle_runtime_verified": live_kaggle,
        "blockers": [
            "glm52_multi_layer_stage_decode_uses_decode_token_chain_only",
            "glm52_multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs",
            "glm52_multi_layer_stage_decode_is_not_full_model_hidden",
            "glm52_multi_layer_stage_decode_is_not_kaggle_runtime",
            "glm52_multi_layer_stage_decode_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "multi_layer_stage_decode_uses_decode_token_chain_only": True,
            "multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs": True,
            "multi_layer_stage_decode_is_not_full_model_hidden": True,
            "multi_layer_stage_decode_is_not_kaggle_runtime": True,
            "multi_layer_stage_decode_is_not_same_request": True,
            "requires_full_prefill_layer_outputs": True,
            "requires_kaggle_stage_runtime": True,
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


def test_checker_accepts_multi_layer_stage_decode_without_success_overclaim() -> None:
    assert check.validate_report(_report(), require_verified=True) == []


def test_checker_rejects_full_prefill_overclaim() -> None:
    assert "full_prefill_stage_hidden_verified_overclaim" in check.validate_report(_report(full_prefill=True))


def test_checker_rejects_generated_token_overclaim() -> None:
    assert "generated_token_verified_overclaim" in check.validate_report(_report(generated_token=True))


def test_checker_rejects_stage_decode_overclaim() -> None:
    assert "stage_decode_verified_overclaim" in check.validate_report(_report(stage_decode=True))


def test_checker_rejects_same_request_overclaim() -> None:
    assert "same_request_decode_verified_overclaim" in check.validate_report(_report(same_request=True))


def test_checker_rejects_live_kaggle_runtime_overclaim() -> None:
    assert "live_kaggle_runtime_verified_overclaim" in check.validate_report(_report(live_kaggle=True))
