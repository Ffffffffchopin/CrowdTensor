from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile

import torch

from scripts import glm52_full_prefix_stage_decode_check as check
from scripts import glm52_full_prefix_stage_decode_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _layer(layer_id: int) -> dict:
    return {
        "layer_id": layer_id,
        "layer_full_prefix_verified": True,
        "token_count": 3,
        "verified_token_count": 3,
        "dsa_indexer_types": ["full"] if layer_id == 6 else ["shared"],
        "dsa_indexer_source_layer_ids": [6],
        "final_token_dsa_mask_topk_count": 2,
        "final_token_dsa_mask_pruned_position_count": 1,
        "final_token_attention_scores_shape": [2, 3],
        "final_token_full_moe_output_shape": [4],
        "final_token_layer_output_shape": [4],
        "final_token_layer_output_hash": _hash(str(layer_id)),
    }


def _report(
    *,
    generated_token: bool = False,
    stage_decode: bool = False,
    same_request: bool = False,
    live_kaggle: bool = False,
) -> dict:
    hidden = 4
    vocab = 8
    seq_len = 3
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
        "stage_hidden_source": "dsa_masked_full_prefix_multi_layer_stage_hidden",
        "stage_layer_range": [6, 8],
        "stage_layer_count": 2,
        "executed_layer_count": 2,
        "stage_prefill_length": 2,
        "stage_sequence_length": seq_len,
        "stage_updated_cache_length": seq_len,
        "dsa_mask_topk_requested": 2,
        "full_prefix_token_carrier_verified": True,
        "small_sequence_probe": True,
        "initial_stage_hidden_sequence_shape": [seq_len, hidden],
        "initial_stage_hidden_sequence_hash": _hash("a"),
        "layer_summaries": [_layer(6), _layer(7)],
        "all_layers_full_prefix_verified": True,
        "all_layer_outputs_chained": True,
        "stage_hidden_sequence_shape": [seq_len, hidden],
        "stage_hidden_sequence_hash": _hash("b"),
        "stage_hidden_shape": [hidden],
        "stage_hidden_hash": _hash("c"),
        "norm_weight_shape": [hidden],
        "normalized_stage_hidden_shape": [hidden],
        "normalized_stage_hidden_hash": _hash("d"),
        "lm_head_shape": [vocab, hidden],
        "lm_head_dtype": "BF16",
        "lm_head_nbytes": vocab * hidden * 2,
        "lm_head_file_count": 1,
        "lm_head_rows_scanned": vocab,
        "lm_head_block_count": 2,
        "lm_head_row_block_size": 4,
        "top_k": 3,
        "top_k_count": 3,
        "selected_token_id_hash": _hash("e"),
        "selected_logit_hash": _hash("f"),
        "top_token_ids_hash": _hash("0"),
        "top_logits_hash": _hash("1"),
        "full_prefix_stage_hidden_verified": True,
        "multi_layer_stage_hidden_verified": True,
        "stage_hidden_to_lm_head_verified": True,
        "lm_head_streamed_full_vocab": True,
        "stage_hidden_lm_head_token_selection_verified": True,
        "partial_full_prefix_token_hash_verified": True,
        "full_model_hidden_verified": False,
        "generated_token_verified": generated_token,
        "stage_decode_verified": stage_decode,
        "same_request_decode_verified": same_request,
        "live_kaggle_runtime_verified": live_kaggle,
        "blockers": [
            "glm52_full_prefix_stage_decode_uses_small_sequence_probe",
            "glm52_full_prefix_stage_decode_is_not_kaggle_runtime",
            "glm52_full_prefix_stage_decode_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "full_prefix_stage_decode_uses_small_sequence_probe": True,
            "full_prefix_stage_decode_is_not_kaggle_runtime": True,
            "full_prefix_stage_decode_is_not_same_request": True,
            "requires_kaggle_stage_runtime": True,
            "requires_full_model_or_stage_partition": True,
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


def _handoff_report() -> dict:
    report = _report()
    for key in [
        "lm_head_shape",
        "lm_head_dtype",
        "lm_head_nbytes",
        "lm_head_file_count",
        "lm_head_rows_scanned",
        "lm_head_block_count",
        "lm_head_row_block_size",
        "top_k",
        "top_k_count",
        "selected_token_id_hash",
        "selected_logit_hash",
        "top_token_ids_hash",
        "top_logits_hash",
    ]:
        report.pop(key, None)
    report["lm_head_required"] = False
    report["lm_head_skipped_for_nonfinal_stage"] = True
    report["stage_handoff_only_verified"] = True
    report["stage_hidden_to_lm_head_verified"] = False
    report["lm_head_streamed_full_vocab"] = False
    report["stage_hidden_lm_head_token_selection_verified"] = False
    report["partial_full_prefix_token_hash_verified"] = False
    return report


def test_checker_accepts_full_prefix_stage_decode_without_success_overclaim() -> None:
    assert check.validate_report(_report(), require_verified=True) == []


def test_private_input_hidden_sequence_decodes_without_public_payload() -> None:
    hidden = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    raw_b64 = base64.b64encode(hidden.numpy().tobytes()).decode("ascii")
    args = probe.parse_args(
        [
            "--input-hidden-b64",
            raw_b64,
            "--input-hidden-shape-json",
            "[3,4]",
            "--input-hidden-dtype",
            "float16",
        ]
    )

    decoded, summary = probe.load_private_input_hidden_sequence(
        args,
        hidden_size=4,
        expected_sequence_length=3,
    )

    assert decoded is not None
    assert list(decoded.shape) == [3, 4]
    assert summary["initial_stage_hidden_sequence_source"] == "private_upstream_activation"
    assert summary["input_activation_consumed"] is True
    assert summary["input_activation_public"] is False
    assert summary["input_activation_hash"].startswith("sha256:")
    public_json = json.dumps(summary, sort_keys=True)
    assert raw_b64 not in public_json
    assert probe.public_redaction_errors(summary) == []


def test_private_input_hidden_sequence_rejects_shape_mismatch() -> None:
    hidden = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    raw_b64 = base64.b64encode(hidden.numpy().tobytes()).decode("ascii")
    args = probe.parse_args(
        [
            "--input-hidden-b64",
            raw_b64,
            "--input-hidden-shape-json",
            "[3,4]",
            "--input-hidden-dtype",
            "float16",
        ]
    )

    try:
        probe.load_private_input_hidden_sequence(args, hidden_size=5, expected_sequence_length=3)
    except RuntimeError as exc:
        assert str(exc) == "input_hidden_shape_mismatch"
    else:
        raise AssertionError("expected shape mismatch")


def test_private_output_activation_writes_payload_without_public_hidden() -> None:
    base = Path(tempfile.mkdtemp(prefix="ct_glm52_private_activation_"))
    output_path = base / "activation.json"
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    args = probe.parse_args(["--output-activation-path", str(output_path)])

    summary = probe.write_private_output_activation(args, hidden)
    private_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["output_activation_private_ready"] is True
    assert summary["output_activation_public"] is False
    assert summary["output_activation_shape"] == [3, 4]
    assert summary["output_activation_hash"].startswith("sha256:")
    assert private_payload["schema"] == "glm52_private_stage_activation_v1"
    assert private_payload["activation_hash"] == summary["output_activation_hash"]
    assert private_payload["hidden_shape"] == [3, 4]
    assert private_payload["hidden_dtype"] == "float16"
    assert private_payload["hidden_b64"]
    public_json = json.dumps(summary, sort_keys=True)
    assert private_payload["hidden_b64"] not in public_json
    assert probe.public_redaction_errors(summary) == []


def test_public_error_text_keeps_safe_errors_and_omits_sensitive_fragments() -> None:
    assert probe.public_error_text(RuntimeError("stage_layer_range_invalid")) == "stage_layer_range_invalid"
    assert probe.public_error_text(RuntimeError("token=secret")) == ""


def test_mlp_type_for_layer_defaults_to_first_dense_layers() -> None:
    assert probe._mlp_type_for_layer({"first_k_dense_replace": 3}, 0) == "dense"
    assert probe._mlp_type_for_layer({"first_k_dense_replace": 3}, 3) == "sparse"
    assert probe._mlp_type_for_layer({"mlp_layer_types": ["dense", "sparse"]}, 1) == "sparse"


def test_dense_token_ready_does_not_require_dsa_or_router_fields() -> None:
    layer = {
        "model_type": "glm_moe_dsa",
        "mlp_layer_type": "dense",
        "hidden_size": 4,
        "num_attention_heads": 2,
        "qk_head_dim": 3,
        "v_head_dim": 5,
        "updated_key_cache_shape": [1, 2, 3],
        "updated_value_cache_shape": [1, 2, 5],
        "attention_scores_shape": [2, 1],
        "attention_weights_shape": [2, 1],
        "attention_output_shape": [4],
        "attention_residual_shape": [4],
        "post_attention_norm_shape": [4],
        "dense_mlp_output_shape": [4],
        "dense_mlp_output_hash": _hash("a"),
        "layer_output_shape": [4],
        "layer_output_hash": _hash("b"),
    }

    assert probe._token_ready(layer, {"num_experts_per_tok": 8}, position=0) is True


def test_checker_accepts_handoff_only_full_prefix_stage_without_lm_head() -> None:
    assert check.validate_report(_handoff_report(), require_verified=True) == []


def test_checker_rejects_handoff_only_lm_head_overclaim() -> None:
    report = _handoff_report()
    report["lm_head_streamed_full_vocab"] = True

    assert "lm_head_streamed_full_vocab_handoff_overclaim" in check.validate_report(report)


def test_checker_rejects_generated_token_overclaim() -> None:
    assert "generated_token_verified_overclaim" in check.validate_report(_report(generated_token=True))


def test_checker_rejects_stage_decode_overclaim() -> None:
    assert "stage_decode_verified_overclaim" in check.validate_report(_report(stage_decode=True))


def test_checker_rejects_same_request_overclaim() -> None:
    assert "same_request_decode_verified_overclaim" in check.validate_report(_report(same_request=True))


def test_checker_rejects_live_kaggle_runtime_overclaim() -> None:
    assert "live_kaggle_runtime_verified_overclaim" in check.validate_report(_report(live_kaggle=True))
