from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import deepseek_v4_flash_kaggle_tpu_swarm_rc_check as check
from scripts import deepseek_v4_flash_kaggle_tpu_swarm_rc_pack as pack


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_dsv4_kaggle_tpu_rc_"))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _source() -> dict:
    return {
        "schema": "deepseek_v4_flash_quantized_source_resolver_v1",
        "ok": True,
        "deepseek_v4_flash_quantized_source_resolver_ready": True,
        "model": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "architecture_class": "moe",
            "total_params_b": 284.0,
            "active_params_b": 13.0,
            "quantized_goal": True,
        },
        "candidate_count": 4,
        "ready_candidate_count": 4,
        "recommended_live_probe_candidate": {
            "candidate_id": "iq1-s-xl-gguf",
            "repo": "teamblobfish/DeepSeek-V4-Flash-GGUF",
            "quant": "IQ1_S-XL",
            "runtime_backend": "llama_cpp_v4_fork",
            "runtime_fork": "cchuter/llama.cpp@feat/v4-port-cuda",
            "total_size_gb": 61.540805,
            "split_file_count": 2,
            "files": [{"path": "a.gguf", "size_gb": 49.9}, {"path": "b.gguf", "size_gb": 11.6}],
            "blockers": ["deepseek_v4_flash_llama_cpp_runtime_wip"],
        },
        "blockers": ["deepseek_v4_flash_llama_cpp_runtime_wip"],
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True},
    }


def _adapter(*, forward: bool = False, fp4_topk: bool = False) -> dict:
    topk_forward = {
        "ready": True,
        "forward_kind": "deepseek_v4_jax_tpu_stage_selective_fp4_topk_routed_experts_plus_fp8_shared_expert",
        "topk": 6,
        "loaded_tensor_count": 42,
        "total_loaded_tensor_bytes": 105383424,
        "final_output_shape": [4096],
        "final_output_hash": "sha256:" + "b" * 64,
        "finite_output": True,
        "weight_tensor_values_public": False,
        "activation_payload_public": False,
    }
    return {
        "schema": "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe_v1",
        "ok": forward,
        "metadata_ready": True,
        "kaggle_web_tpu_runtime_ready": True,
        "deepseek_v4_jax_tpu_stage_forward_ready": forward,
        "deepseek_v4_jax_tpu_fixture_stage_forward_ready": True,
        "deepseek_v4_real_weight_tpu_tensor_load_ready": True,
        "deepseek_v4_real_fp4_topk_expert_mlp_forward_ready": fp4_topk,
        "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_ready": forward,
        "stage_plan": {
            "real_fp4_topk_expert_mlp_forward_ready": fp4_topk,
            "real_routed_expert_topk_count": 6 if fp4_topk else 0,
            "real_routed_expert_loaded_tensor_count": 42 if fp4_topk else 0,
            "real_routed_expert_total_loaded_tensor_bytes": 105383424 if fp4_topk else 0,
        },
        "deepseek_metadata": {
            "source": "local_hf_api",
            "metadata_ready": True,
            "stage_key_mapping_ready": True,
            "model_config": {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "num_hidden_layers": 43,
            },
            "weight_index": {"weight_key_count": 69187, "weight_file_count": 46},
            "stage_mapping": {
                "layer_range": [16, 18],
                "selected_key_count": 3145,
                "selected_file_count": 2,
                "family_hits": {
                    "mla_attention": True,
                    "moe_router": True,
                    "shared_experts": True,
                    "routed_experts": True,
                    "hybrid_compression": True,
                    "norms": True,
                },
            },
        },
        "web_tpu_cell": {
            "deepseek_v4_real_weight_tpu_tensor_load": {
                "loaded_tensor_count": 12,
                "total_loaded_tensor_bytes": 19677696,
                "dtype_counts": {"BF16": 3, "F32": 1, "F8_E4M3": 1, "F8_E8M0": 4, "I8": 3},
                "real_router_smoke_ready": True,
                "real_fp8_block_dequant_smoke_ready": True,
                "real_i8_expert_dequant_smoke_ready": True,
                "real_i8_expert_mlp_slice_smoke_ready": True,
                "real_fp4_topk_expert_mlp_forward": topk_forward if fp4_topk else {},
                "weight_tensor_values_public": False,
            }
        },
        "failure_stage": "" if forward else "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented",
        "blockers": [] if forward else ["deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented"],
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True},
    }


def _web_tpu_execution() -> dict:
    return {
        "schema": "kaggle_web_tpu_execution_channel_probe_v1",
        "ok": True,
        "web_tpu_execution_channel_ready": True,
        "small_jax_cell_ready": True,
        "tiny_qwen_like_cell_ready": True,
        "tpu_runtime_attached": True,
        "tpu_device_count": 8,
        "stage_local_kv_cache_verified": True,
        "failure_stage": "",
        "blocker_codes": [],
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True},
    }


def _web_tpu_queue_with_progress() -> dict:
    return {
        "schema": "kaggle_web_tpu_queue_monitor_probe_v1",
        "ok": True,
        "start_clicked": True,
        "web_tpu_runtime_ready": False,
        "queue_progress": {
            "queue_position_observed": True,
            "queue_position_changed": True,
            "queue_position_decreased": True,
            "unique_queue_positions": [22, 23],
        },
        "blocker_codes": ["kaggle_web_tpu_active_event_queued"],
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True},
    }


def _gpu_token_quota_probe() -> dict:
    return {
        "schema": "kaggle_gpu_token_weekly_quota_probe_v1",
        "requested_accelerator": "NvidiaTeslaT4",
        "summary": {
            "account_count": 3,
            "auth_ok_count": 3,
            "gpu_submission_accepted_count": 1,
            "weekly_gpu_quota_exhausted_count": 2,
        },
        "accounts": [
            {
                "label": "tpuowner",
                "owner": "tpuowner",
                "auth_ok": True,
                "push_accepted": False,
                "quota_class": "weekly_gpu_quota_exhausted",
                "weekly_gpu_quota_available_inferred": False,
                "weekly_gpu_quota_exhausted": True,
                "cleanup": {"attempted": False, "deleted": False, "failed": False},
            },
            {
                "label": "primary Kaggle account",
                "owner": "xuyuhaosuyi",
                "auth_ok": True,
                "push_accepted": False,
                "quota_class": "weekly_gpu_quota_exhausted",
                "weekly_gpu_quota_available_inferred": False,
                "weekly_gpu_quota_exhausted": True,
                "cleanup": {"attempted": False, "deleted": False, "failed": False},
            },
            {
                "label": "cpuowner",
                "owner": "cpuowner",
                "auth_ok": True,
                "push_accepted": True,
                "quota_class": "gpu_submission_accepted",
                "weekly_gpu_quota_available_inferred": True,
                "weekly_gpu_quota_exhausted": False,
                "cleanup": {"attempted": True, "deleted": True, "failed": False},
            },
        ],
        "private_kernel_payloads_removed": True,
        "public_artifact_safe": True,
        "token_file_public": False,
        "token_values_public": False,
    }


def _torch_smoke() -> dict:
    return {
        "schema": "deepseek_v4_flash_torch_stage_adapter_smoke_v1",
        "ok": True,
        "deepseek_v4_flash_torch_stage_adapter_smoke_ready": True,
        "real_deepseek_weights_loaded": False,
        "jax_tpu_translation_ready": False,
        "model": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "model_type": "deepseek_v4",
            "full_model_weight_values_loaded": False,
        },
        "reference_stage": {
            "ok": True,
            "transformers_reference_used": True,
            "tiny_fixture": True,
            "stage_owned_key_count": 27,
            "real_deepseek_v4_components_exercised": {
                "manifold_hyper_connections": True,
                "compressed_attention": True,
                "mla_shared_kv_attention": True,
                "grouped_output_projection": True,
                "moe_router": True,
                "routed_experts": True,
                "shared_experts": True,
                "stage_local_kv_cache_shape": True,
            },
        },
        "blockers": [],
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True},
    }


def _jax_smoke() -> dict:
    return {
        "schema": "deepseek_v4_flash_jax_stage_adapter_smoke_v1",
        "ok": False,
        "deepseek_v4_flash_jax_stage_adapter_smoke_ready": False,
        "jax_runtime_execution_requested": True,
        "jax_runtime_execution_ready": False,
        "tpu_runtime_required": False,
        "tpu_runtime_ready": False,
        "deepseek_v4_jax_stage_forward_ready": False,
        "deepseek_v4_jax_tpu_stage_forward_ready": False,
        "real_deepseek_weights_loaded": False,
        "model": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "model_type": "deepseek_v4",
            "fixture_config": True,
            "real_deepseek_weights_loaded": False,
            "full_model_weight_values_loaded": False,
        },
        "stage": {
            "stage_type": "decoder_layer_fixture_translation",
            "stage_owned_key_count": 27,
        },
        "numpy_reference": {
            "ok": True,
            "components_exercised": {
                "manifold_hyper_connections": True,
                "mla_shared_kv_attention": True,
                "grouped_output_projection": True,
                "attention_sink": True,
                "topk_moe_router": True,
                "routed_experts": True,
                "shared_experts": True,
                "hca_compressor_shape_metadata": True,
            },
        },
        "blockers": ["jax_missing"],
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True},
    }


def _safetensors_header(*, ready: bool = True) -> dict:
    return {
        "schema": "deepseek_v4_flash_safetensors_stage_header_probe_v1",
        "ok": True,
        "deepseek_v4_flash_safetensors_stage_header_probe_ready": True,
        "safetensors_header_ready": ready,
        "stage_header_shape_ready": ready,
        "model": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "expected_model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "architecture_class": "moe",
            "model_config": {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "num_hidden_layers": 43,
            },
        },
        "stage_mapping": {
            "layer_range": [16, 18],
            "selected_key_count": 3145,
            "selected_file_count": 2,
            "selected_key_digest": "sha256:abc",
            "selected_file_digest": "sha256:def",
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
        "headers": {
            "schema": "deepseek_v4_flash_safetensors_stage_header_summary_v1",
            "header_file_count": 2 if ready else 1,
            "selected_file_count": 2,
            "selected_key_count": 3145,
            "header_fetch_error_count": 0 if ready else 1,
            "missing_header_key_count": 0,
            "dtype_counts": {"BF16": 3145} if ready else {},
            "rank_counts": {"1": 100, "2": 3045} if ready else {},
            "total_selected_tensor_storage_bytes": 123456789 if ready else 0,
            "file_summaries": [],
            "safetensors_header_payload_public": False,
            "real_weight_tensor_values_loaded": False,
            "real_weight_tensor_values_public": False,
        },
        "failure_stage": "" if ready else "safetensors_range_header_fetch_failed",
        "blockers": [] if ready else ["safetensors_range_header_fetch_failed"],
        "public_artifact_safe": True,
        "safety": {
            "public_artifact_safe": True,
            "safetensors_header_payload_public": False,
            "weight_tensor_values_loaded": False,
            "weight_tensor_values_public": False,
        },
    }


def _same_request(*, providers: list[str] | None = None, generated: int = 1) -> dict:
    providers = providers or ["kaggle_cuda", "kaggle_web_tpu", "kaggle_cpu"]
    return {
        "schema": "deepseek_v4_flash_kaggle_tpu_same_request_probe_v1",
        "ok": True,
        "same_request_decode_verified": True,
        "deepseek_v4_flash_kaggle_tpu_same_request_verified": True,
        "generated_token_count": generated,
        "accepted_providers": providers,
        "provider_stage_counts": {provider: 1 for provider in providers},
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "blockers": [],
        "diagnosis_codes": ["deepseek_v4_flash_kaggle_tpu_same_request_decode_verified"],
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True},
    }


def _same_request_stage_slices() -> dict:
    return {
        "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
        "ok": True,
        "same_request_runtime_bridge_verified": True,
        "same_request_decode_verified": False,
        "gpu_tpu_cpu_deepseek_v4_same_request_verified": False,
        "generated_token_count": 1,
        "accepted_stage_backends": ["cpu", "cuda", "jax_tpu"],
        "accepted_providers": ["cpu", "cuda", "jax_tpu"],
        "provider_stage_counts": {"cpu": 1, "cuda": 1, "jax_tpu": 1},
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "deepseek_v4_same_request_stage_slice_verified": True,
        "deepseek_v4_gpu_stage_slice_verified": True,
        "deepseek_v4_cpu_stage_slice_verified": True,
        "deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified": True,
        "deepseek_v4_tpu_fp4_topk_expert_forward_verified": True,
        "deepseek_v4_gpu_fp4_topk_expert_forward_verified": True,
        "deepseek_v4_cpu_fp4_topk_expert_forward_verified": True,
        "deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified": True,
        "deepseek_v4_stage_layer_ranges": {"cuda": [16, 17], "jax_tpu": [17, 18], "cpu": [18, 19]},
        "deepseek_v4_distinct_backend_stage_layer_ranges_verified": True,
        "deepseek_v4_stage_layer_coverage_count": 3,
        "model_scope": "deepseek_v4_flash_same_request_gpu_tpu_cpu_fp4_topk_expert_forwards_not_full_decode",
        "blockers": ["deepseek_v4_full_same_request_decode_not_verified"],
        "diagnosis_codes": [
            "runtime_bridge_deepseek_v4_gpu_tpu_cpu_stage_slices_ready",
            "runtime_bridge_deepseek_v4_gpu_tpu_cpu_fp4_topk_forwards_ready",
        ],
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True},
    }


def _cpu_fp4_topk_expert_forward() -> dict:
    return {
        "schema": "deepseek_v4_flash_cpu_fp4_topk_expert_forward_probe_v1",
        "ok": True,
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "layer": 16,
        "diagnosis_codes": ["deepseek_v4_cpu_real_fp4_topk_expert_mlp_forward_ready"],
        "blockers": [],
        "fp4_topk_expert_forward": {
            "ready": True,
            "forward_kind": "deepseek_v4_stage_selective_fp4_topk_routed_experts_plus_fp8_shared_expert",
            "topk": 6,
            "loaded_tensor_count": 42,
            "total_loaded_tensor_bytes": 105383424,
            "final_output_shape": [4096],
            "final_output_hash": "sha256:" + "a" * 64,
            "finite_output": True,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        },
        "public_artifact_safe": True,
        "safety": {"public_artifact_safe": True, "weight_tensor_values_public": False},
    }


def test_current_blocker_rc_passes_without_overclaiming() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=False))
    web_tpu = _write(out / "web-tpu.json", _web_tpu_execution())
    torch_smoke = _write(out / "torch-smoke.json", _torch_smoke())
    jax_smoke = _write(out / "jax-smoke.json", _jax_smoke())
    header = _write(out / "header.json", _safetensors_header())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--web-tpu-execution-report",
            str(web_tpu),
            "--torch-stage-smoke-report",
            str(torch_smoke),
            "--jax-stage-smoke-report",
            str(jax_smoke),
            "--safetensors-stage-header-report",
            str(header),
        ])
    )

    assert report["ok"] is True
    assert report["success"]["same_request_decode_verified"] is False
    assert report["failure_stage"] == "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented"
    assert report["deepseek_tpu_adapter"]["real_i8_expert_dequant_smoke_ready"] is True
    assert report["deepseek_tpu_adapter"]["real_i8_expert_mlp_slice_smoke_ready"] is True
    assert report["deepseek_torch_stage_smoke"]["torch_stage_adapter_smoke_ready"] is True
    assert report["deepseek_jax_stage_smoke"]["numpy_reference_ready"] is True
    assert report["deepseek_safetensors_stage_header"]["safetensors_header_ready"] is True
    assert "deepseek_v4_flash_kaggle_tpu_same_request_decode_not_verified" in report["blockers"]
    assert check.validate_report(report) == []


def test_queue_summary_reads_nested_queue_progress() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    queue = _write(out / "queue.json", _web_tpu_queue_with_progress())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--web-tpu-queue-report",
            str(queue),
        ])
    )

    assert report["web_tpu_queue"]["queue_position_observed"] is True
    assert report["web_tpu_queue"]["queue_position_changed"] is True
    assert report["web_tpu_queue"]["queue_position_decreased"] is True
    assert report["web_tpu_queue"]["unique_queue_positions"] == [22, 23]
    assert check.validate_report(report) == []


def test_token_quota_probe_imports_cpuowner_gpu_availability() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    gpu_quota = _write(out / "gpu-quota.json", _gpu_token_quota_probe())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--kaggle-gpu-preflight-report",
            str(gpu_quota),
        ])
    )

    gpu_summary = report["kaggle_gpu_preflight"]
    assert gpu_summary["source_schema"] == "kaggle_gpu_token_weekly_quota_probe_v1"
    assert gpu_summary["source_ok"] is True
    assert gpu_summary["kaggle_cuda_ready"] is True
    assert gpu_summary["accepted_submission_count"] == 1
    assert gpu_summary["accelerator"] == "NvidiaTeslaT4"
    assert gpu_summary["token_gpu_accepted_accounts"] == ["cpuowner"]
    assert set(gpu_summary["token_gpu_exhausted_accounts"]) == {"tpuowner", "primary Kaggle account"}
    assert gpu_summary["cleanup_attempted"] is True
    assert gpu_summary["cleanup_failed"] is False
    assert report["success"]["same_request_decode_verified"] is False
    assert check.validate_report(report) == []


def test_success_requires_all_kaggle_gpu_webtpu_cpu_and_adapter_forward() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=True))
    same = _write(out / "same.json", _same_request())
    torch_smoke = _write(out / "torch-smoke.json", _torch_smoke())
    jax_smoke = _write(out / "jax-smoke.json", _jax_smoke())
    header = _write(out / "header.json", _safetensors_header())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--torch-stage-smoke-report",
            str(torch_smoke),
            "--jax-stage-smoke-report",
            str(jax_smoke),
            "--safetensors-stage-header-report",
            str(header),
            "--same-request-report",
            str(same),
        ])
    )

    assert report["success"]["same_request_decode_verified"] is True
    assert set(report["success"]["accepted_providers"]) == {"kaggle_cuda", "kaggle_web_tpu", "kaggle_cpu"}
    assert report["failure_stage"] == ""
    assert check.validate_report(report) == []


def test_three_provider_stage_slice_evidence_passes_without_full_decode_overclaim() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=True))
    same = _write(out / "same.json", _same_request_stage_slices())
    torch_smoke = _write(out / "torch-smoke.json", _torch_smoke())
    jax_smoke = _write(out / "jax-smoke.json", _jax_smoke())
    header = _write(out / "header.json", _safetensors_header())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--torch-stage-smoke-report",
            str(torch_smoke),
            "--jax-stage-smoke-report",
            str(jax_smoke),
            "--safetensors-stage-header-report",
            str(header),
            "--same-request-report",
            str(same),
        ])
    )

    assert report["success"]["same_request_decode_verified"] is False
    assert set(report["success"]["accepted_providers"]) == {"kaggle_cuda", "kaggle_web_tpu", "kaggle_cpu"}
    assert report["same_request"]["generated_token_count"] == 1
    assert report["same_request"]["deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified"] is True
    assert report["same_request"]["deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified"] is True
    assert report["same_request"]["deepseek_v4_tpu_fp4_topk_expert_forward_verified"] is True
    assert report["same_request"]["deepseek_v4_gpu_fp4_topk_expert_forward_verified"] is True
    assert report["same_request"]["deepseek_v4_cpu_fp4_topk_expert_forward_verified"] is True
    assert report["same_request"]["deepseek_v4_distinct_backend_stage_layer_ranges_verified"] is True
    assert report["same_request"]["deepseek_v4_stage_layer_coverage_count"] == 3
    assert report["same_request"]["deepseek_v4_stage_layer_ranges"] == {"cuda": [16, 17], "jax_tpu": [17, 18], "cpu": [18, 19]}
    assert report["same_request"]["model_scope"] == "deepseek_v4_flash_same_request_gpu_tpu_cpu_fp4_topk_expert_forwards_not_full_decode"
    assert "deepseek_v4_full_same_request_decode_not_verified" in report["same_request"]["blockers"]
    assert "deepseek_v4_flash_kaggle_tpu_same_request_decode_not_verified" in report["blockers"]
    assert check.validate_report(report) == []


def test_same_request_web_tpu_stage_proof_filters_stale_web_tpu_blockers() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=False, fp4_topk=True))
    same = _write(out / "same.json", _same_request_stage_slices())
    active_event = _write(
        out / "active.json",
        {
            "schema": "kaggle_web_tpu_active_event_probe_v1",
            "ok": True,
            "tpu_v5e_active_event_visible": False,
            "active_event_running": False,
            "blocked_reason": "kaggle_web_tpu_active_event_missing",
            "blocker_codes": ["kaggle_web_tpu_active_event_missing"],
            "public_artifact_safe": True,
        },
    )
    execution = _write(
        out / "exec.json",
        {
            "schema": "kaggle_web_tpu_execution_channel_probe_v1",
            "ok": False,
            "web_tpu_execution_channel_ready": False,
            "small_jax_cell_ready": True,
            "tiny_qwen_like_cell_ready": False,
            "tpu_runtime_attached": True,
            "tpu_device_count": 8,
            "failure_stage": "tiny_qwen_like_forward",
            "blocker_codes": ["tiny_qwen_like_cell_exception", "web_tpu_execution_channel_not_ready"],
            "public_artifact_safe": True,
        },
    )

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--web-tpu-active-event-report",
            str(active_event),
            "--web-tpu-execution-report",
            str(execution),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--same-request-report",
            str(same),
        ])
    )

    assert report["success"]["same_request_decode_verified"] is False
    assert "deepseek_v4_full_same_request_decode_not_verified" in report["blockers"]
    assert "kaggle_web_tpu_active_event_missing" not in report["blockers"]
    assert "tiny_qwen_like_cell_exception" not in report["blockers"]
    assert "web_tpu_execution_channel_not_ready" not in report["blockers"]
    assert "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented" not in report["blockers"]
    assert check.validate_report(report) == []


def test_queue_summary_uses_final_running_observation_to_filter_queue_blockers() -> None:
    summary = pack.summarize_web_tpu_queue(
        {
            "schema": "kaggle_web_tpu_queue_monitor_probe_v1",
            "ok": True,
            "queue_progress": {
                "queue_position_observed": True,
                "queue_position_changed": True,
                "queue_position_decreased": True,
                "unique_queue_positions": [18, 1],
            },
            "final_observation": {
                "active_event_running": True,
                "active_event_queued": False,
                "web_tpu_runtime_ready": False,
            },
            "blocker_codes": [
                "kaggle_web_tpu_active_event_queued",
                "kaggle_web_tpu_queue_prompt_visible",
                "kaggle_web_tpu_session_still_starting",
                "kaggle_web_tpu_jupyter_frame_not_visible",
            ],
            "public_artifact_safe": True,
        }
    )

    assert summary["active_event_running"] is True
    assert summary["queue_position_decreased"] is True
    assert "kaggle_web_tpu_active_event_queued" not in summary["blockers"]
    assert "kaggle_web_tpu_queue_prompt_visible" not in summary["blockers"]
    assert "kaggle_web_tpu_session_still_starting" not in summary["blockers"]
    assert "kaggle_web_tpu_jupyter_frame_not_visible" in summary["blockers"]


def test_cpu_fp4_topk_expert_forward_is_imported_without_success_overclaim() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=True))
    cpu_fp4 = _write(out / "cpu-fp4.json", _cpu_fp4_topk_expert_forward())
    torch_smoke = _write(out / "torch-smoke.json", _torch_smoke())
    jax_smoke = _write(out / "jax-smoke.json", _jax_smoke())
    header = _write(out / "header.json", _safetensors_header())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--torch-stage-smoke-report",
            str(torch_smoke),
            "--jax-stage-smoke-report",
            str(jax_smoke),
            "--safetensors-stage-header-report",
            str(header),
            "--cpu-fp4-topk-expert-report",
            str(cpu_fp4),
        ])
    )

    cpu_summary = report["deepseek_cpu_fp4_topk_expert_forward"]
    assert cpu_summary["stage_selective_fp4_topk_expert_forward_ready"] is True
    assert cpu_summary["topk"] == 6
    assert cpu_summary["final_output_shape"] == [4096]
    assert report["success"]["same_request_decode_verified"] is False
    assert check.validate_report(report) == []


def test_web_tpu_fp4_topk_adapter_is_imported_without_success_overclaim() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(fp4_topk=True))
    web_tpu = _write(out / "web-tpu.json", _web_tpu_execution())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--web-tpu-execution-report",
            str(web_tpu),
        ])
    )

    adapter_summary = report["deepseek_tpu_adapter"]
    assert adapter_summary["real_fp4_topk_expert_mlp_forward_ready"] is True
    assert adapter_summary["real_fp4_topk_count"] == 6
    assert adapter_summary["real_fp4_topk_loaded_tensor_count"] == 42
    assert adapter_summary["real_fp4_topk_total_loaded_tensor_bytes"] == 105383424
    assert adapter_summary["real_fp4_topk_final_output_shape"] == [4096]
    assert report["success"]["same_request_decode_verified"] is False
    assert "deepseek_v4_flash_kaggle_tpu_same_request_decode_not_verified" in report["blockers"]
    assert check.validate_report(report) == []


def test_gpu_cpu_only_same_request_does_not_count() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=True))
    same = _write(out / "same.json", _same_request(providers=["kaggle_cuda", "kaggle_cpu"]))
    torch_smoke = _write(out / "torch-smoke.json", _torch_smoke())
    jax_smoke = _write(out / "jax-smoke.json", _jax_smoke())
    header = _write(out / "header.json", _safetensors_header())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--torch-stage-smoke-report",
            str(torch_smoke),
            "--jax-stage-smoke-report",
            str(jax_smoke),
            "--safetensors-stage-header-report",
            str(header),
            "--same-request-report",
            str(same),
        ])
    )

    assert report["success"]["same_request_decode_verified"] is False
    assert "deepseek_v4_flash_kaggle_tpu_same_request_decode_not_verified" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_overclaimed_success_without_adapter_forward() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=False))
    same = _write(out / "same.json", _same_request())
    torch_smoke = _write(out / "torch-smoke.json", _torch_smoke())
    jax_smoke = _write(out / "jax-smoke.json", _jax_smoke())
    header = _write(out / "header.json", _safetensors_header())
    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--torch-stage-smoke-report",
            str(torch_smoke),
            "--jax-stage-smoke-report",
            str(jax_smoke),
            "--safetensors-stage-header-report",
            str(header),
            "--same-request-report",
            str(same),
        ])
    )
    report["success"]["same_request_decode_verified"] = True

    errors = check.validate_report(report)

    assert "success_without_deepseek_tpu_stage_forward" in errors


def test_checker_rejects_torch_smoke_jax_tpu_overclaim() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=False))
    torch_smoke_payload = _torch_smoke()
    torch_smoke_payload["jax_tpu_translation_ready"] = True
    torch_smoke = _write(out / "torch-smoke.json", torch_smoke_payload)
    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--torch-stage-smoke-report",
            str(torch_smoke),
        ])
    )

    errors = check.validate_report(report)

    assert "deepseek_torch_stage_smoke_jax_tpu_overclaim" in errors


def test_checker_rejects_jax_stage_smoke_tpu_overclaim() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=False))
    jax_payload = _jax_smoke()
    jax_payload["deepseek_v4_jax_tpu_stage_forward_ready"] = True
    jax_smoke = _write(out / "jax-smoke.json", jax_payload)
    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--jax-stage-smoke-report",
            str(jax_smoke),
        ])
    )

    errors = check.validate_report(report)

    assert "deepseek_jax_stage_smoke_tpu_forward_overclaim" in errors


def test_checker_rejects_success_without_ready_safetensors_header() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    adapter = _write(out / "adapter.json", _adapter(forward=True))
    same = _write(out / "same.json", _same_request())
    header = _write(out / "header.json", _safetensors_header(ready=False))
    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--deepseek-tpu-adapter-report",
            str(adapter),
            "--safetensors-stage-header-report",
            str(header),
            "--same-request-report",
            str(same),
        ])
    )
    report["success"]["same_request_decode_verified"] = True
    report["blockers"] = []

    errors = check.validate_report(report)

    assert "success_without_safetensors_stage_header_ready" in errors


def test_checker_rejects_safetensors_header_weight_overclaim() -> None:
    out = _tmp_dir()
    source = _write(out / "source.json", _source())
    header_payload = _safetensors_header()
    header_payload["headers"]["real_weight_tensor_values_loaded"] = True
    header = _write(out / "header.json", header_payload)
    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(out / "rc"),
            "--source-resolver-report",
            str(source),
            "--safetensors-stage-header-report",
            str(header),
        ])
    )

    errors = check.validate_report(report)

    assert "deepseek_safetensors_stage_header_weight_value_overclaim" in errors
