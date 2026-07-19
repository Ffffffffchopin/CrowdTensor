from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

from scripts import deepseek_v4_flash_kaggle_web_tpu_stage_adapter_check as check
from scripts import deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe as probe


def _args(tmp_path: Path) -> argparse.Namespace:
    return probe.parse_args(["--output-dir", str(tmp_path)])


def _fixture_forward() -> dict:
    return {
        "input_shape": [1, 2, 2, 16],
        "output_shape": [1, 2, 2, 16],
        "dtype": "float32",
        "output_mean": 0.0,
        "output_std": 0.1,
        "output_finite": True,
        "stage_output_hash": "sha256:fixture",
        "stage_local_kv_cache_metadata": {
            "stage_local_only": True,
            "k_shape": [1, 1, 2, 8],
            "v_shape": [1, 1, 2, 8],
            "kv_payload_public": False,
            "past_key_values_public": False,
        },
        "components_exercised": {
            "manifold_hyper_connections": True,
            "mla_shared_kv_attention": True,
            "grouped_output_projection": True,
            "attention_sink": True,
            "topk_moe_router": True,
            "routed_experts": True,
            "shared_experts": True,
            "stage_local_kv_cache_shape": True,
        },
        "fixture_weights": True,
        "real_weight_tensor_values_loaded": False,
        "activation_payload_public": False,
        "kv_cache_public": False,
    }


def _real_tensor_load() -> dict:
    return {
        "loaded_tensor_count": 12,
        "requested_tensor_count": 12,
        "loaded_key_digest": "sha256:loaded",
        "dtype_counts": {"BF16": 3, "F32": 1, "F8_E4M3": 1, "F8_E8M0": 4, "I8": 3},
        "total_loaded_tensor_bytes": 19677696,
        "device_put_count": 12,
        "finite_tensor_count": 12,
        "tpu_device_kind": "TPU v5 lite",
        "real_fp8_block_dequant_smoke_ready": True,
        "real_fp8_block_dequant_smoke": {
            "ready": True,
            "smoke_kind": "deepseek_v4_fp8_e4m3_ue8m0_block_dequant_matmul",
            "weight_block_shape": [128, 128],
            "scale_shape": [8, 32],
            "block_size": [128, 128],
            "output_shape": [128],
            "output_hash": "sha256:fp8-out",
            "finite_on_device": True,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        },
        "real_i8_expert_dequant_smoke_ready": True,
        "real_i8_expert_dequant_smoke": {
            "ready": True,
            "smoke_kind": "deepseek_v4_i8_ue8m0_expert_w1_block_dequant_matmul",
            "expert_id": 0,
            "weight_block_shape": [128, 128],
            "scale_shape": [2048, 128],
            "scale_group_size": 16,
            "scale_group_count_used": 8,
            "output_shape": [128],
            "output_hash": "sha256:i8-expert-out",
            "finite_on_device": True,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        },
        "real_i8_expert_mlp_slice_smoke_ready": True,
        "real_i8_expert_mlp_slice_smoke": {
            "ready": True,
            "smoke_kind": "deepseek_v4_i8_ue8m0_expert_mlp_slice_forward",
            "expert_id": 0,
            "input_shape": [128],
            "intermediate_shape": [128],
            "output_shape": [128],
            "w1_block_shape": [128, 128],
            "w2_block_shape": [128, 128],
            "w3_block_shape": [128, 128],
            "w1_scale_group_size": 16,
            "w2_scale_group_size": 16,
            "w3_scale_group_size": 16,
            "w1_scale_group_count_used": 8,
            "w2_scale_group_count_used": 8,
            "w3_scale_group_count_used": 8,
            "output_hash": "sha256:i8-expert-mlp-out",
            "finite_on_device": True,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        },
        "real_fp4_topk_expert_mlp_forward_ready": True,
        "real_routed_expert_topk_count": 6,
        "real_routed_expert_loaded_tensor_count": 42,
        "real_routed_expert_total_loaded_tensor_bytes": 105383424,
        "real_fp4_topk_expert_mlp_forward": {
            "ready": True,
            "forward_kind": "deepseek_v4_jax_tpu_stage_selective_fp4_topk_routed_experts_plus_fp8_shared_expert",
            "topk": 6,
            "loaded_tensor_count": 42,
            "total_loaded_tensor_bytes": 105383424,
            "final_output_shape": [4096],
            "final_output_hash": "sha256:fp4-topk-final",
            "finite_output": True,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        },
        "real_router_smoke_ready": True,
        "real_router_smoke": {
            "ready": True,
            "router_kind": "deepseek_v4_moe_gate_topk",
            "input_shape": [4096],
            "gate_shape": [256, 4096],
            "top_k": 6,
            "topk_index_digest": "sha256:router-index",
            "topk_value_hash": "sha256:router-values",
            "finite_on_device": True,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        },
        "tensor_summaries": [
            {
                "key_digest": "sha256:key-a",
                "file_digest": "sha256:file-a",
                "dtype": "F32",
                "shape": [64],
                "byte_length": 256,
                "raw_payload_sha256": "sha256:payload-a",
                "device_put_ready": True,
                "finite_on_device": True,
                "weight_tensor_values_public": False,
            },
            {
                "key_digest": "sha256:key-b",
                "file_digest": "sha256:file-a",
                "dtype": "BF16",
                "shape": [4096],
                "byte_length": 8192,
                "raw_payload_sha256": "sha256:payload-b",
                "device_put_ready": True,
                "finite_on_device": True,
                "weight_tensor_values_public": False,
            },
        ],
        "tensor_payload_hashes_public": True,
        "weight_tensor_values_public": False,
        "real_weight_tensor_values_loaded": True,
        "stage_weight_values_loaded": False,
    }


def _cell(
    *,
    metadata: bool = True,
    tpu: bool = True,
    adapter: bool = False,
    fixture: bool = False,
    real_tensor_load: bool = False,
) -> dict:
    blockers = []
    diagnosis = []
    if metadata and tpu and not adapter and fixture:
        blockers = [
            "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented",
            "deepseek_v4_flash_quantized_fp8_nvfp4_tpu_loader_not_implemented",
        ]
        diagnosis = [
            "deepseek_v4_flash_stage_key_mapping_ready",
            "deepseek_v4_flash_jax_tpu_fixture_stage_forward_ready",
            "deepseek_v4_flash_web_tpu_fixture_stage_forward_ready_real_weight_gap",
        ]
    elif metadata and tpu and not adapter:
        blockers = [
            "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented",
            "deepseek_v4_flash_quantized_fp8_nvfp4_tpu_loader_not_implemented",
        ]
        diagnosis = [
            "deepseek_v4_flash_stage_key_mapping_ready",
            "deepseek_v4_flash_web_tpu_metadata_ready_adapter_gap",
        ]
    return {
        "schema": probe.CELL_SCHEMA,
        "cell_kind": "deepseek_stage_metadata",
        "ok": metadata and tpu,
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "metadata_ready": metadata,
        "stage_key_mapping_ready": metadata,
        "jax_imported": tpu,
        "tpu_runtime_ready": tpu,
        "tpu_device_count": 8 if tpu else 0,
        "tpu_device_kind": "TPU v5 lite" if tpu else "",
        "deepseek_v4_real_weight_tpu_tensor_load_ready": real_tensor_load,
        "deepseek_v4_real_weight_tpu_tensor_load": _real_tensor_load() if real_tensor_load else {},
        "deepseek_v4_jax_tpu_fixture_stage_forward_ready": fixture,
        "deepseek_v4_jax_tpu_fixture_stage_forward": _fixture_forward() if fixture else {},
        "deepseek_v4_jax_tpu_stage_forward_ready": adapter,
        "model_config": {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "n_shared_experts": 1,
            "q_lora_rank": 1024,
            "qk_rope_head_dim": 64,
            "moe_intermediate_size": 2048,
            "torch_dtype": "bfloat16",
            "quantization_config_present": True,
            "config_payload_public": False,
        },
        "weight_index": {
            "weight_key_count": 69187,
            "weight_file_count": 46,
            "metadata_total_size_bytes": 159609485896,
            "weight_map_payload_public": False,
        },
        "stage_mapping": {
            "layer_range": [16, 18],
            "selected_key_count": 3216,
            "selected_file_count": 3,
            "selected_key_digest": "sha256:keys",
            "selected_file_digest": "sha256:files",
            "family_hits": {
                "mla_attention": True,
                "moe_router": True,
                "shared_experts": True,
                "routed_experts": True,
                "hybrid_compression": True,
                "norms": True,
            },
            "stage_weight_values_loaded": False,
            "stage_weight_values_public": False,
        },
        "blockers": blockers,
        "diagnosis_codes": diagnosis,
        "jupyter_proxy_token_public": False,
        "public_artifact_safe": True,
    }


def test_rendered_deepseek_metadata_cell_compiles(tmp_path: Path) -> None:
    source = probe.render_deepseek_stage_metadata_cell(_args(tmp_path))
    path = tmp_path / "deepseek_stage_metadata_cell.py"
    path.write_text(source, encoding="utf-8")

    py_compile.compile(str(path), doraise=True)

    rendered = path.read_text(encoding="utf-8")
    assert "DeepSeek-V4-Flash" in rendered
    assert "run_deepseek_v4_tpu_fixture_stage_forward" in rendered
    assert "run_deepseek_v4_real_weight_tpu_tensor_load" in rendered
    assert "deepseek_v4_flash_real_weight_tpu_tensor_load_ready" in rendered
    assert "real_fp4_topk_expert_mlp_forward_ready" in rendered
    assert "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented" in rendered
    assert "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented" in rendered
    assert "jupyter_proxy_token_public" in rendered
    assert "public_artifact_safe" in rendered


def test_metadata_and_web_tpu_without_adapter_is_valid_blocker(tmp_path: Path) -> None:
    report = probe.build_report(_args(tmp_path), cell_report=_cell(), output_dir=tmp_path)

    assert report["ok"] is False
    assert report["metadata_ready"] is True
    assert report["kaggle_web_tpu_runtime_ready"] is True
    assert report["deepseek_v4_jax_tpu_stage_forward_ready"] is False
    assert report["failure_stage"] == "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented"
    assert "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented" in report["blockers"]
    assert check.validate_report(report) == []


def test_fixture_stage_forward_moves_blocker_to_real_weight_loader(tmp_path: Path) -> None:
    report = probe.build_report(_args(tmp_path), cell_report=_cell(fixture=True), output_dir=tmp_path)

    assert report["ok"] is False
    assert report["metadata_ready"] is True
    assert report["kaggle_web_tpu_runtime_ready"] is True
    assert report["deepseek_v4_jax_tpu_fixture_stage_forward_ready"] is True
    assert report["deepseek_v4_jax_tpu_stage_forward_ready"] is False
    assert report["failure_stage"] == "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented"
    assert "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented" in report["blockers"]
    assert "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented" not in report["blockers"]
    assert check.validate_report(report) == []


def test_real_weight_tpu_tensor_load_is_recorded_without_overclaim(tmp_path: Path) -> None:
    report = probe.build_report(_args(tmp_path), cell_report=_cell(fixture=True, real_tensor_load=True), output_dir=tmp_path)

    assert report["deepseek_v4_real_weight_tpu_tensor_load_ready"] is True
    assert report["stage_plan"]["real_weight_sample_tensors_loaded"] is True
    assert report["web_tpu_cell"]["deepseek_v4_real_weight_tpu_tensor_load"]["real_router_smoke_ready"] is True
    assert report["web_tpu_cell"]["deepseek_v4_real_weight_tpu_tensor_load"]["real_fp8_block_dequant_smoke_ready"] is True
    assert report["web_tpu_cell"]["deepseek_v4_real_weight_tpu_tensor_load"]["real_i8_expert_dequant_smoke_ready"] is True
    assert report["web_tpu_cell"]["deepseek_v4_real_weight_tpu_tensor_load"]["real_i8_expert_mlp_slice_smoke_ready"] is True
    assert report["web_tpu_cell"]["deepseek_v4_real_weight_tpu_tensor_load"]["real_fp4_topk_expert_mlp_forward_ready"] is True
    assert report["deepseek_v4_real_fp4_topk_expert_mlp_forward_ready"] is True
    assert report["stage_plan"]["real_fp4_topk_expert_mlp_forward_ready"] is True
    assert report["stage_plan"]["real_weight_values_loaded"] is False
    assert report["deepseek_v4_jax_tpu_stage_forward_ready"] is False
    assert report["failure_stage"] == "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented"
    assert check.validate_report(report) == []


def test_local_metadata_survives_web_tpu_timeout(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        cell_report={
            "schema": probe.CELL_SCHEMA,
            "cell_kind": "deepseek_stage_metadata",
            "ok": False,
            "blockers": ["web_tpu_jupyter_execute_timeout"],
            "diagnosis_codes": ["bridge_web_tpu_jupyter_execute_timeout"],
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
        },
        output_dir=tmp_path,
        local_metadata={
            "schema": "deepseek_v4_flash_stage_metadata_v1",
            "metadata_ready": True,
            "stage_key_mapping_ready": True,
            "model_config": _cell()["model_config"],
            "weight_index": _cell()["weight_index"],
            "stage_mapping": _cell()["stage_mapping"],
            "blockers": [],
            "public_artifact_safe": True,
        },
    )

    assert report["metadata_ready"] is True
    assert report["kaggle_web_tpu_runtime_ready"] is False
    assert report["failure_stage"] == "kaggle_web_tpu_runtime_not_ready"
    assert report["deepseek_metadata"]["source"] == "local_hf_api"
    assert check.validate_report(report) == []


def test_checker_rejects_success_without_stage_forward(tmp_path: Path) -> None:
    report = probe.build_report(_args(tmp_path), cell_report=_cell(), output_dir=tmp_path)
    report["ok"] = True
    report["deepseek_v4_flash_kaggle_web_tpu_stage_adapter_ready"] = True
    report["failure_stage"] = ""
    report["blockers"] = []

    errors = check.validate_report(report)

    assert "adapter_ready_without_stage_forward" in errors


def test_checker_accepts_full_adapter_ready_shape(tmp_path: Path) -> None:
    report = probe.build_report(_args(tmp_path), cell_report=_cell(adapter=True), output_dir=tmp_path)

    assert report["ok"] is True
    assert report["deepseek_v4_flash_kaggle_web_tpu_stage_adapter_ready"] is True
    assert report["failure_stage"] == ""
    assert check.validate_report(report) == []


def test_public_artifacts_are_redacted(tmp_path: Path) -> None:
    report = probe.build_report(_args(tmp_path), cell_report=_cell(), output_dir=tmp_path)

    assert probe.public_redaction_errors(report) == []
    scanned = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file())
    for fragment in [
        "KAGGLE_KEY",
        "HF_TOKEN",
        "Bearer ",
        "Cookie:",
        "jupyter-proxy",
        "token=",
        '"generated_token_ids":',
        '"activation":',
    ]:
        assert fragment not in scanned
