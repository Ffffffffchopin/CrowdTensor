from __future__ import annotations

from scripts import glm52_lm_head_token_check as check
from scripts import glm52_lm_head_token_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _report(
    *,
    full_model_hidden: bool = False,
    generated_token: bool = False,
    stage_decode: bool = False,
    same_request: bool = False,
) -> dict:
    hidden = 4
    vocab = 8
    return {
        "schema": probe.SCHEMA,
        "ok": True,
        "glm52_lm_head_token_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "tie_word_embeddings": False,
        "norm_weight_shape": [hidden],
        "hidden_source": "deterministic_probe_vector",
        "hidden_shape": [hidden],
        "normalized_hidden_shape": [hidden],
        "hidden_hash": _hash("a"),
        "normalized_hidden_hash": _hash("b"),
        "lm_head_shape": [vocab, hidden],
        "lm_head_dtype": "BF16",
        "lm_head_nbytes": vocab * hidden * 2,
        "lm_head_file_count": 1,
        "lm_head_rows_scanned": vocab,
        "lm_head_block_count": 2,
        "lm_head_row_block_size": 4,
        "top_k": 3,
        "top_k_count": 3,
        "selected_token_id_hash": _hash("c"),
        "selected_logit_hash": _hash("d"),
        "top_token_ids_hash": _hash("e"),
        "top_logits_hash": _hash("f"),
        "final_norm_verified": True,
        "lm_head_streamed_full_vocab": True,
        "lm_head_logits_token_selection_verified": True,
        "selected_token_hash_verified": True,
        "full_model_hidden_verified": full_model_hidden,
        "generated_token_verified": generated_token,
        "stage_decode_verified": stage_decode,
        "same_request_decode_verified": same_request,
        "blockers": [
            "glm52_lm_head_token_selection_uses_probe_hidden_not_full_model_hidden",
            "glm52_lm_head_token_selection_is_not_stage_decode",
            "glm52_lm_head_token_selection_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "lm_head_token_selection_uses_probe_hidden_not_full_model_hidden": True,
            "lm_head_token_selection_is_not_stage_decode": True,
            "lm_head_token_selection_is_not_same_request": True,
            "requires_full_model_or_stage_hidden": True,
            "requires_stage_decode_verified": True,
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


def test_checker_accepts_lm_head_token_selection_without_decode_success() -> None:
    assert check.validate_report(_report(), require_verified=True) == []


def test_checker_rejects_full_model_hidden_overclaim() -> None:
    assert "full_model_hidden_verified_overclaim" in check.validate_report(_report(full_model_hidden=True))


def test_checker_rejects_generated_token_overclaim() -> None:
    assert "generated_token_verified_overclaim" in check.validate_report(_report(generated_token=True))


def test_checker_rejects_stage_decode_overclaim() -> None:
    assert "stage_decode_verified_overclaim" in check.validate_report(_report(stage_decode=True))


def test_checker_rejects_same_request_overclaim() -> None:
    assert "same_request_decode_verified_overclaim" in check.validate_report(_report(same_request=True))
