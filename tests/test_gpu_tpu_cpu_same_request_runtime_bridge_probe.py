from __future__ import annotations

import py_compile
import inspect
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_cpu_same_request_runtime_bridge_probe as probe


def test_unpack_fp4_e2m1_numpy_expands_two_values_per_byte() -> None:
    import numpy as np

    packed = np.asarray([[0x71, 0xEA]], dtype=np.uint8).view(np.int8)
    unpacked = probe.unpack_fp4_e2m1_numpy(packed)

    assert unpacked.shape == (1, 4)
    assert unpacked.tolist() == [[0.5, 6.0, -1.0, -4.0]]


def test_dequant_block_scaled_numpy_derives_block_shape_from_scale_grid() -> None:
    import numpy as np

    quantized = np.arange(16, dtype=np.float32).reshape(4, 4)
    scales = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dequantized, block_m, block_n = probe.dequant_block_scaled_numpy(quantized, scales)

    assert (block_m, block_n) == (2, 2)
    assert dequantized.shape == (4, 4)
    assert dequantized[0, 0] == 0.0
    assert dequantized[0, 2] == 4.0
    assert dequantized[2, 0] == 24.0
    assert dequantized[2, 2] == 40.0


def test_rendered_gpu_bridge_kernel_compiles(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        probe.render_gpu_kernel(
            coordinator_url="http://127.0.0.1:9256",
            token="test-token",
            target_generated_token_count=1,
        ),
        encoding="utf-8",
    )

    py_compile.compile(str(kernel_path), doraise=True)

    rendered = kernel_path.read_text(encoding="utf-8")
    assert "gpu_tpu_cpu_bridge_cuda_stage_v1" in rendered
    assert "activation_payload_public" in rendered
    assert "TARGET_GENERATED_TOKEN_COUNT = 1" in rendered
    assert "accepted_count >= max(1, int(TARGET_GENERATED_TOKEN_COUNT))" in rendered
    assert "cuda" in rendered


def test_rendered_gpu_deepseek_real_slice_kernel_compiles(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        probe.render_gpu_kernel(
            coordinator_url="http://127.0.0.1:9256",
            token="test-token",
            target_generated_token_count=1,
            deepseek_real_stage_slice=True,
            deepseek_stage_layer_start=16,
            deepseek_stage_layer_end=17,
        ),
        encoding="utf-8",
    )

    py_compile.compile(str(kernel_path), doraise=True)

    rendered = kernel_path.read_text(encoding="utf-8")
    assert "DEEPSEEK_REAL_STAGE_SLICE = True" in rendered
    assert "DEEPSEEK_STAGE_LAYER_START = 16" in rendered
    assert "DEEPSEEK_STAGE_LAYER_END = 17" in rendered
    assert '"stage_layer_range"' in rendered
    assert "run_deepseek_v4_real_weight_cuda_slice" in rendered
    assert "deepseek_v4_real_weight_cuda_tensor_load_ready" in rendered
    assert "real_i8_expert_mlp_slice_smoke_ready" in rendered
    assert "real_fp4_topk_expert_mlp_forward_ready" in rendered


def test_rendered_kaggle_cpu_deepseek_real_slice_kernel_compiles(tmp_path: Path) -> None:
    kernel_path = tmp_path / "cpu_kernel.py"
    kernel_path.write_text(
        probe.render_kaggle_cpu_kernel(
            coordinator_url="http://127.0.0.1:9256",
            token="test-token",
            deepseek_real_stage_slice=True,
            deepseek_stage_layer_start=16,
            deepseek_stage_layer_end=17,
        ),
        encoding="utf-8",
    )

    py_compile.compile(str(kernel_path), doraise=True)

    rendered = kernel_path.read_text(encoding="utf-8")
    assert "gpu_tpu_cpu_bridge_kaggle_cpu_stage_v1" in rendered
    assert '"provider": "kaggle_cpu"' in rendered
    assert '"kaggle_kernel": True' in rendered
    assert "DEEPSEEK_STAGE_LAYER_START = 16" in rendered
    assert "DEEPSEEK_STAGE_LAYER_END = 17" in rendered
    assert '"stage_layer_range"' in rendered
    assert "run_deepseek_v4_real_weight_cpu_slice" in rendered
    assert "real_fp4_topk_expert_mlp_forward_ready" in rendered


def test_rendered_kaggle_cpu_deepseek_fp4_helper_executes(tmp_path: Path) -> None:
    import numpy as np

    rendered = probe.render_kaggle_cpu_kernel(
        coordinator_url="http://127.0.0.1:9256",
        token="test-token",
        deepseek_real_stage_slice=True,
        deepseek_stage_layer_start=16,
        deepseek_stage_layer_end=17,
    )
    namespace = {"__name__": "rendered_cpu_kernel_test"}

    exec(compile(rendered, str(tmp_path / "cpu_kernel.py"), "exec"), namespace)
    unpacked = namespace["unpack_fp4_e2m1_numpy"](np.asarray([[0x71]], dtype=np.uint8).view(np.int8))

    assert unpacked.tolist() == [[0.5, 6.0]]


def test_bridge_report_does_not_claim_32b_success(tmp_path: Path) -> None:
    args = probe.parse_args(["--output-dir", str(tmp_path)])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={"ok": True, "cuda_device_count": 1},
        tpu_report={"ok": True, "tpu_device_count": 8},
        cpu_report={"ok": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["ok"] is True
    assert report["same_request_runtime_bridge_verified"] is True
    assert report["gpu_tpu_cpu_32b_same_request_verified"] is False
    assert report["same_request_32b_model_verified"] is False
    assert report["not_32b_weight_success"] is True
    assert probe.public_redaction_errors(report) == []


def test_parse_args_defaults_to_kaggle_web_tpu_provider(tmp_path: Path) -> None:
    args = probe.parse_args(["--output-dir", str(tmp_path)])

    assert args.tpu_provider == "kaggle_web"
    assert args.colab_session_name == "ct-colab-tpu-v5e1"
    assert args.colab_session_config.endswith(".config/colab-cli/sessions.json")


def test_colab_tpu_shape_bridge_report_is_same_request_not_32b(tmp_path: Path) -> None:
    args = probe.parse_args(["--output-dir", str(tmp_path), "--tpu-provider", "colab_cli"])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu", "tpu_provider": "colab_cli"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={"ok": True, "cuda_device_count": 1},
        tpu_report={"ok": True, "tpu_device_count": 1, "tpu_provider": "colab_cli"},
        cpu_report={"ok": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["same_request_runtime_bridge_verified"] is True
    assert report["runtime_device_summary"]["tpu_provider"] == "colab_cli"
    assert report["gpu_tpu_cpu_32b_same_request_verified"] is False
    assert report["not_32b_weight_success"] is True
    assert probe.public_redaction_errors(report) == []


def test_colab_tpu_provider_can_render_32b_loader_path(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--tpu-provider",
        "colab_cli",
        "--web-tpu-32b-execute",
    ])

    code = probe.render_web_tpu_32b_stage_code(args, coordinator_url="http://127.0.0.1:1", token="test-token")

    assert "BRIDGE_COORDINATOR_URL" in code
    assert "full_stage_owned_tpu_loader_ready" in code
    assert "bridge_jax_tpu_32b_stage_owned_loader_ready" in code


def test_web_tpu_deepseek_stage_code_renders_bridge_submit(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-deepseek-stage-execute",
    ])

    code = probe.render_web_tpu_deepseek_stage_code(args, coordinator_url="http://127.0.0.1:1", token="test-token")

    compile(code, "<deepseek-web-tpu-bridge-stage>", "exec")
    assert "deepseek-ai/DeepSeek-V4-Flash" in code
    assert "real_i8_expert_mlp_slice_smoke_ready" in code
    assert "real_fp4_topk_expert_mlp_forward_ready" in code
    assert "web-tpu-bridge-stage1-deepseek-v4-slice" in code
    assert "BRIDGE_COORDINATOR_URL" in code
    assert "deepseek_v4_stage_owned_slice_loaded" in code
    assert "deepseek_metadata" in code
    assert "kaggle_web_tpu_runtime_ready" in code
    assert "adapter_stage_key_mapping_ready" in code


def test_deepseek_backend_layer_ranges_can_be_configured_independently(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-deepseek-stage-execute",
        "--deepseek-gpu-stage-layer-start",
        "16",
        "--deepseek-gpu-stage-layer-end",
        "17",
        "--deepseek-tpu-stage-layer-start",
        "17",
        "--deepseek-tpu-stage-layer-end",
        "18",
        "--deepseek-cpu-stage-layer-start",
        "18",
        "--deepseek-cpu-stage-layer-end",
        "19",
    ])

    gpu_code = probe.render_gpu_kernel(
        coordinator_url="http://127.0.0.1:1",
        token="test-token",
        deepseek_real_stage_slice=True,
        deepseek_stage_layer_start=probe.resolved_deepseek_layer_range(args, "gpu")[0],
        deepseek_stage_layer_end=probe.resolved_deepseek_layer_range(args, "gpu")[1],
    )
    cpu_code = probe.render_kaggle_cpu_kernel(
        coordinator_url="http://127.0.0.1:1",
        token="test-token",
        deepseek_real_stage_slice=True,
        deepseek_stage_layer_start=probe.resolved_deepseek_layer_range(args, "cpu")[0],
        deepseek_stage_layer_end=probe.resolved_deepseek_layer_range(args, "cpu")[1],
    )
    tpu_code = probe.render_web_tpu_deepseek_stage_code(args, coordinator_url="http://127.0.0.1:1", token="test-token")

    assert probe.resolved_deepseek_layer_range(args, "gpu") == (16, 17)
    assert probe.resolved_deepseek_layer_range(args, "tpu") == (17, 18)
    assert probe.resolved_deepseek_layer_range(args, "cpu") == (18, 19)
    assert "DEEPSEEK_STAGE_LAYER_START = 16" in gpu_code
    assert "DEEPSEEK_STAGE_LAYER_START = 18" in cpu_code
    assert "LAYER_START = 17" in tpu_code
    assert "LAYER_END = 18" in tpu_code


def test_web_tpu_deepseek_stage_code_can_force_new_session(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-deepseek-stage-execute",
        "--web-tpu-force-new-session",
    ])

    probe.render_web_tpu_deepseek_stage_code(args, coordinator_url="http://127.0.0.1:1", token="test-token")

    assert args.web_tpu_force_new_session is True
    assert "web_tpu_force_new_session" in inspect.getsource(probe._execute_web_tpu_code_via_iframe_direct)


def test_bridge_report_records_deepseek_tpu_slice_without_full_decode_overclaim(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-deepseek-stage-execute",
    ])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={"ok": True, "cuda_device_count": 1},
        tpu_report={
            "ok": True,
            "tpu_device_count": 8,
            "stage_owned_model_loaded": True,
            "deepseek_v4_stage_owned_slice_loaded": True,
            "real_i8_expert_mlp_slice_smoke_ready": True,
            "real_weight_sample_loaded_tensor_count": 12,
        },
        cpu_report={"ok": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["same_request_runtime_bridge_verified"] is True
    assert report["same_request_target_parameter_class"] == "deepseek_v4_flash"
    assert report["deepseek_v4_same_request_stage_slice_verified"] is True
    assert report["gpu_tpu_cpu_deepseek_v4_same_request_verified"] is False
    assert report["model_scope"] == "deepseek_v4_flash_same_request_tpu_real_weight_slice_not_full_decode"
    assert "deepseek_v4_gpu_cpu_real_stage_not_verified" in report["blockers"]
    assert "deepseek_v4_full_same_request_decode_not_verified" in report["blockers"]
    assert probe.public_redaction_errors(report) == []


def test_bridge_report_records_deepseek_all_backend_slices_without_full_decode_overclaim(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-deepseek-stage-execute",
    ])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    real_slice = {
        "ok": True,
        "stage_owned_model_loaded": True,
        "deepseek_v4_stage_owned_slice_loaded": True,
        "real_i8_expert_mlp_slice_smoke_ready": True,
        "real_weight_sample_loaded_tensor_count": 12,
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={**real_slice, "cuda_device_count": 2},
        tpu_report={**real_slice, "tpu_device_count": 8},
        cpu_report=real_slice,
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["same_request_runtime_bridge_verified"] is True
    assert report["deepseek_v4_same_request_stage_slice_verified"] is True
    assert report["deepseek_v4_gpu_stage_slice_verified"] is True
    assert report["deepseek_v4_cpu_stage_slice_verified"] is True
    assert report["deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified"] is True
    assert report["gpu_tpu_cpu_deepseek_v4_same_request_verified"] is False
    assert report["model_scope"] == "deepseek_v4_flash_same_request_gpu_tpu_cpu_real_weight_slices_not_full_decode"
    assert "deepseek_v4_gpu_cpu_real_stage_not_verified" not in report["blockers"]
    assert "deepseek_v4_gpu_tpu_cpu_fp4_topk_forward_not_verified" in report["blockers"]
    assert "deepseek_v4_full_same_request_decode_not_verified" in report["blockers"]
    assert set(report["accepted_providers"]) == {"cuda", "jax_tpu", "cpu"}
    assert probe.public_redaction_errors(report) == []


def test_bridge_report_records_deepseek_all_backend_fp4_topk_forwards_without_full_decode_overclaim(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-deepseek-stage-execute",
        "--deepseek-gpu-stage-layer-start",
        "16",
        "--deepseek-gpu-stage-layer-end",
        "17",
        "--deepseek-tpu-stage-layer-start",
        "17",
        "--deepseek-tpu-stage-layer-end",
        "18",
        "--deepseek-cpu-stage-layer-start",
        "18",
        "--deepseek-cpu-stage-layer-end",
        "19",
    ])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    real_forward = {
        "ok": True,
        "stage_owned_model_loaded": True,
        "deepseek_v4_stage_owned_slice_loaded": True,
        "real_i8_expert_mlp_slice_smoke_ready": True,
        "real_fp4_topk_expert_mlp_forward_ready": True,
        "real_weight_sample_loaded_tensor_count": 13,
        "real_routed_expert_loaded_tensor_count": 42,
        "real_routed_expert_total_loaded_tensor_bytes": 105383424,
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={**real_forward, "cuda_device_count": 2, "stage_layer_range": [16, 17]},
        tpu_report={**real_forward, "tpu_device_count": 8, "stage_layer_range": [17, 18]},
        cpu_report={**real_forward, "stage_layer_range": [18, 19]},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified"] is True
    assert report["deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified"] is True
    assert report["deepseek_v4_distinct_backend_stage_layer_ranges_verified"] is True
    assert report["deepseek_v4_stage_layer_coverage_count"] == 3
    assert report["deepseek_v4_stage_layer_ranges"] == {"cuda": [16, 17], "jax_tpu": [17, 18], "cpu": [18, 19]}
    assert report["gpu_tpu_cpu_deepseek_v4_same_request_verified"] is False
    assert report["model_scope"] == "deepseek_v4_flash_same_request_gpu_tpu_cpu_fp4_topk_expert_forwards_not_full_decode"
    assert "deepseek_v4_gpu_tpu_cpu_fp4_topk_forward_not_verified" not in report["blockers"]
    assert "deepseek_v4_full_same_request_decode_not_verified" in report["blockers"]
    assert probe.public_redaction_errors(report) == []


def test_bridge_report_requires_kaggle_cpu_when_requested(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-deepseek-stage-execute",
        "--kaggle-cpu-stage",
    ])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    real_forward = {
        "ok": True,
        "stage_owned_model_loaded": True,
        "deepseek_v4_stage_owned_slice_loaded": True,
        "real_i8_expert_mlp_slice_smoke_ready": True,
        "real_fp4_topk_expert_mlp_forward_ready": True,
        "real_weight_sample_loaded_tensor_count": 13,
        "real_routed_expert_loaded_tensor_count": 42,
        "real_routed_expert_total_loaded_tensor_bytes": 105383424,
    }

    local_cpu_report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={**real_forward, "cuda_device_count": 2},
        tpu_report={**real_forward, "tpu_device_count": 8},
        cpu_report=real_forward,
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )
    assert local_cpu_report["ok"] is False
    assert "kaggle_cpu_stage_not_verified" in local_cpu_report["blockers"]

    kaggle_cpu_report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={**real_forward, "cuda_device_count": 2},
        tpu_report={**real_forward, "tpu_device_count": 8},
        cpu_report={**real_forward, "provider": "kaggle_cpu", "kaggle_kernel": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        cpu_steps=[
            {"name": "kaggle_cpu_kernel_push", "accepted": True, "ok": True},
            {"name": "kaggle_cpu_kernel_delete", "ok": True},
        ],
        errors=[],
    )
    assert kaggle_cpu_report["ok"] is True
    assert kaggle_cpu_report["runtime_device_summary"]["cpu_stage_provider"] == "kaggle_cpu"
    assert kaggle_cpu_report["runtime_device_summary"]["kaggle_cpu_stage_ready"] is True
    assert "kaggle_cpu_stage_not_verified" not in kaggle_cpu_report["blockers"]
    assert "kaggle_cpu_kernel_deleted" in kaggle_cpu_report["diagnosis_codes"]
    assert probe.public_redaction_errors(kaggle_cpu_report) == []


def test_deepseek_bridge_gpu_weekly_quota_failure_is_explicit(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-deepseek-stage-execute",
    ])
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status={
            "ready": False,
            "generated_token_count": 0,
            "activation_hashes": [],
            "stage_task_counts": {"stage0": 0, "stage1": 0, "stage2": 0},
            "completed_tasks": [],
        },
        gpu_report={},
        tpu_report={},
        cpu_report={"ok": False},
        gpu_steps=[
            {
                "name": "kaggle_kernel_push",
                "ok": True,
                "stdout_tail": "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.",
            }
        ],
        errors=[],
    )

    assert report["same_request_runtime_bridge_verified"] is False
    assert report["model_scope"] == "deepseek_v4_flash_tpu_real_weight_slice_bridge_requested_not_verified"
    assert "kaggle_gpu_weekly_quota_reached" in report["blockers"]
    assert "kaggle_gpu_weekly_quota_reached" in report["diagnosis_codes"]
    assert "configured for a real DeepSeek-V4-Flash weight slice" in report["limitations"][1]
    assert probe.public_redaction_errors(report) == []


def test_bridge_report_can_export_32b_live_proof_when_real_tpu_stage_verified(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-32b-execute",
        "--cuda-stage-32b-weight-evidence-ready",
    ])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={"ok": True, "cuda_device_count": 1},
        tpu_report={
            "ok": True,
            "tpu_device_count": 8,
            "stage_owned_model_loaded": True,
            "qwen32b_stage_owned_loader_ready": True,
            "full_stage_owned_tpu_loader_ready": True,
            "tpu_32b_runtime_adapter_ready": True,
            "stage_local_kv_cache_verified": True,
            "executed_layer_count": 21,
            "missing_stage_key_count": 0,
            "loaded_execution_tensor_key_count": 252,
            "loaded_execution_tensor_gb": 19.1,
            "stage_layer_range": [21, 42],
            "stage_output_hash": "sha256:tpuout",
        },
        cpu_report={"ok": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )
    live = probe.build_live_proof_from_bridge(report)

    assert report["ok"] is True
    assert report["same_request_runtime_bridge_verified"] is True
    assert report["gpu_tpu_cpu_32b_same_request_verified"] is True
    assert report["same_request_32b_model_verified"] is True
    assert report["not_32b_weight_success"] is False
    assert live["schema"] == probe.rc_pack.LIVE_PROOF_SCHEMA
    assert live["ok"] is True
    assert live["gpu_tpu_cpu_32b_same_request_verified"] is True
    assert {item["backend"] for item in live["accepted_stage_tasks"]} == {"cuda", "jax_tpu", "cpu"}
    assert probe.public_redaction_errors(report) == []


def test_bridge_default_tpu_tensor_key_follows_stage_start_for_72b(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--target-model-id",
        "Qwen/Qwen2.5-72B-Instruct",
        "--web-tpu-32b-execute",
        "--web-tpu-32b-stage-start",
        "32",
        "--web-tpu-32b-stage-end",
        "40",
        "--web-tpu-32b-execute-layer-count",
        "8",
    ])

    assert args.web_tpu_32b_tensor_key == "model.layers.32.input_layernorm.weight"


def test_bridge_report_records_72b_stage_without_full_model_overclaim(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--target-model-id",
        "Qwen/Qwen2.5-72B-Instruct",
        "--web-tpu-32b-execute",
        "--web-tpu-32b-stage-start",
        "32",
        "--web-tpu-32b-stage-end",
        "40",
        "--web-tpu-32b-execute-layer-count",
        "8",
    ])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={"ok": True, "cuda_device_count": 2},
        tpu_report={
            "ok": True,
            "tpu_device_count": 8,
            "stage_owned_model_loaded": True,
            "qwen32b_stage_owned_loader_ready": True,
            "full_stage_owned_tpu_loader_ready": True,
            "tpu_32b_runtime_adapter_ready": True,
            "stage_local_kv_cache_verified": True,
            "executed_layer_count": 8,
            "missing_stage_key_count": 0,
            "loaded_execution_tensor_key_count": 96,
            "loaded_execution_tensor_gb": 13.08,
            "stage_layer_range": [32, 40],
            "stage_output_hash": "sha256:tpuout72",
        },
        cpu_report={"ok": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["same_request_runtime_bridge_verified"] is True
    assert report["same_request_target_parameter_class"] == "72b"
    assert report["gpu_tpu_cpu_72b_same_request_stage_verified"] is True
    assert report["same_request_72b_stage_verified"] is True
    assert report["gpu_tpu_cpu_72b_same_request_verified"] is False
    assert report["same_request_72b_full_model_verified"] is False
    assert report["full_72b_tpu_stage_loading_public_claim"] is True
    assert report["full_72b_weight_loading_public_claim"] is False
    assert report["gpu_tpu_cpu_32b_same_request_verified"] is False
    assert report["same_request_32b_model_verified"] is False
    assert "qwen72b_full_model_same_request_decode_not_verified" in report["blockers"]
    assert probe.public_redaction_errors(report) == []


def test_bridge_report_requires_target_generated_token_count(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-32b-execute",
        "--cuda-stage-32b-weight-evidence-ready",
        "--target-generated-token-count",
        "4",
    ])
    one_token_status = {
        "ready": False,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    tpu_report = {
        "ok": True,
        "tpu_device_count": 8,
        "stage_owned_model_loaded": True,
        "qwen32b_stage_owned_loader_ready": True,
        "full_stage_owned_tpu_loader_ready": True,
        "tpu_32b_runtime_adapter_ready": True,
        "stage_local_kv_cache_verified": True,
        "executed_layer_count": 21,
        "missing_stage_key_count": 0,
    }

    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=one_token_status,
        gpu_report={"ok": True, "cuda_device_count": 1},
        tpu_report=tpu_report,
        cpu_report={"ok": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["same_request_runtime_bridge_verified"] is False
    assert report["gpu_tpu_cpu_32b_same_request_verified"] is False
    assert report["target_generated_token_count"] == 4
    assert "same_request_runtime_bridge_not_verified" in report["blockers"]


def test_bridge_report_accepts_four_completed_tokens(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--web-tpu-32b-execute",
        "--cuda-stage-32b-weight-evidence-ready",
        "--target-generated-token-count",
        "4",
    ])
    status = {
        "ready": True,
        "generated_token_count": 4,
        "activation_hashes": [f"sha256:{idx}" for idx in range(8)],
        "stage_task_counts": {"stage0": 4, "stage1": 4, "stage2": 4},
        "completed_tasks": [
            {"runtime_device": {"backend": backend}, "stage_id": stage}
            for _ in range(4)
            for stage, backend in [(0, "cuda"), (1, "jax_tpu"), (2, "cpu")]
        ],
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={"ok": True, "cuda_device_count": 1},
        tpu_report={
            "ok": True,
            "tpu_device_count": 8,
            "stage_owned_model_loaded": True,
            "qwen32b_stage_owned_loader_ready": True,
            "full_stage_owned_tpu_loader_ready": True,
            "tpu_32b_runtime_adapter_ready": True,
            "stage_local_kv_cache_verified": True,
            "executed_layer_count": 21,
            "missing_stage_key_count": 0,
            "loaded_execution_tensor_key_count": 252,
            "loaded_execution_tensor_gb": 19.1,
        },
        cpu_report={"ok": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["same_request_runtime_bridge_verified"] is True
    assert report["gpu_tpu_cpu_32b_same_request_verified"] is True
    assert report["generated_token_count"] == 4


def test_extract_web_tpu_report_from_stdout_uses_last_public_report() -> None:
    stdout = "\n".join([
        "progress",
        '{"report": {"ok": false, "blockers": ["old"]}}',
        "more progress",
        '{"schema": "x", "report": {"ok": true, "tpu_device_count": 8}}',
    ])

    report = probe.extract_web_tpu_report_from_stdout(stdout)

    assert report["ok"] is True
    assert report["tpu_device_count"] == 8
    assert report["jupyter_proxy_token_public"] is False


def test_classify_web_tpu_exception_maps_current_iframe_failures() -> None:
    blocker, diagnosis = probe.classify_web_tpu_exception(RuntimeError("jupyter_api_unavailable_404"))

    assert blocker == "web_tpu_jupyter_api_unavailable"
    assert diagnosis == "bridge_web_tpu_jupyter_api_unavailable"


def test_web_tpu_failure_is_public_safe() -> None:
    report = probe.web_tpu_failure(
        "web_tpu_jupyter_api_unavailable",
        "bridge_web_tpu_jupyter_api_unavailable",
        error_type="RuntimeError",
        error_text="token=secret",
    )

    assert report["ok"] is False
    assert report["jupyter_proxy_token_public"] is False
    assert "token=secret" not in str(report)
    assert probe.public_redaction_errors(report) == []


def test_bridge_32b_tpu_stage_without_cuda_evidence_stays_blocked(tmp_path: Path) -> None:
    args = probe.parse_args(["--output-dir", str(tmp_path), "--web-tpu-32b-execute"])
    status = {
        "ready": True,
        "generated_token_count": 1,
        "activation_hashes": ["sha256:a", "sha256:b"],
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "completed_tasks": [
            {"runtime_device": {"backend": "cuda"}, "stage_id": 0},
            {"runtime_device": {"backend": "jax_tpu"}, "stage_id": 1},
            {"runtime_device": {"backend": "cpu"}, "stage_id": 2},
        ],
    }
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status=status,
        gpu_report={"ok": True, "cuda_device_count": 1},
        tpu_report={
            "ok": True,
            "tpu_device_count": 8,
            "stage_owned_model_loaded": True,
            "qwen32b_stage_owned_loader_ready": True,
            "full_stage_owned_tpu_loader_ready": True,
            "tpu_32b_runtime_adapter_ready": True,
            "stage_local_kv_cache_verified": True,
            "executed_layer_count": 21,
            "missing_stage_key_count": 0,
        },
        cpu_report={"ok": True},
        gpu_steps=[{"name": "kaggle_kernel_delete", "ok": True}],
        errors=[],
    )

    assert report["same_request_runtime_bridge_verified"] is True
    assert report["gpu_tpu_cpu_32b_same_request_verified"] is False
    assert "cuda_stage_32b_weight_evidence_not_imported" in report["blockers"]


def test_bridge_state_stage0_can_be_claimed_and_counted() -> None:
    state = probe.BridgeState()

    claimed = state.claim(miner_id="stage0-miner", stage_id=0)

    assert claimed["ok"] is True
    assert claimed["done"] is False
    assert claimed["task"]["stage_id"] == 0

    submitted = state.submit(
        {
            "task_id": claimed["task"]["task_id"],
            "stage_id": 0,
            "activation": {"activation_hash": "sha256:stage0"},
            "activation_hash": "sha256:stage0",
            "output_hash": "sha256:out0",
            "runtime_device": {"backend": "cuda"},
            "kv_cache": {"ready": True},
        }
    )
    status = state.public_status()

    assert submitted["accepted"] is True
    assert status["stage_task_counts"]["stage0"] == 1
    assert status["pending_count"] == 1


def test_bridge_state_completes_four_token_stage_chain() -> None:
    state = probe.BridgeState(target_generated_token_count=4)

    for step in range(4):
        stage0 = state.claim(miner_id="stage0-miner", stage_id=0)["task"]
        assert stage0["generation_step"] == step
        assert state.submit(
            {
                "task_id": stage0["task_id"],
                "stage_id": 0,
                "generation_step": step,
                "activation": {"activation_hash": f"sha256:stage0-{step}"},
                "activation_hash": f"sha256:stage0-{step}",
                "output_hash": f"sha256:out0-{step}",
                "runtime_device": {"backend": "cuda"},
                "kv_cache": {"ready": True},
            }
        )["accepted"] is True

        stage1 = state.claim(miner_id="stage1-miner", stage_id=1)["task"]
        assert stage1["generation_step"] == step
        assert state.submit(
            {
                "task_id": stage1["task_id"],
                "stage_id": 1,
                "generation_step": step,
                "activation": {"activation_hash": f"sha256:stage1-{step}"},
                "activation_hash": f"sha256:stage1-{step}",
                "output_hash": f"sha256:out1-{step}",
                "runtime_device": {"backend": "jax_tpu"},
                "kv_cache": {"ready": True},
            }
        )["accepted"] is True

        stage2 = state.claim(miner_id="stage2-miner", stage_id=2)["task"]
        assert stage2["generation_step"] == step
        assert state.submit(
            {
                "task_id": stage2["task_id"],
                "stage_id": 2,
                "generation_step": step,
                "activation_hash": f"sha256:stage1-{step}",
                "output_hash": f"sha256:out2-{step}",
                "next_token_hash": f"sha256:token-{step}",
                "runtime_device": {"backend": "cpu"},
                "kv_cache": {"ready": True},
            }
        )["accepted"] is True

    status = state.public_status()

    assert status["ready"] is True
    assert status["generated_token_count"] == 4
    assert status["stage_task_counts"] == {"stage0": 4, "stage1": 4, "stage2": 4}
    assert [item["generation_step"] for item in status["completed_tasks"]] == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]


def test_wait_for_ready_returns_when_tpu_thread_exits_before_stage1() -> None:
    state = probe.BridgeState()
    claimed = state.claim(miner_id="stage0-miner", stage_id=0)
    assert state.submit(
        {
            "task_id": claimed["task"]["task_id"],
            "stage_id": 0,
            "activation": {"activation_hash": "sha256:stage0"},
            "activation_hash": "sha256:stage0",
            "output_hash": "sha256:out0",
            "runtime_device": {"backend": "cuda"},
            "kv_cache": {"ready": True},
        }
    )["accepted"] is True
    sleeping = threading.Thread(target=lambda: __import__("time").sleep(0.2))
    sleeping.start()
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join(timeout=1.0)

    status = probe.wait_for_ready(state, timeout_seconds=5.0, threads=[sleeping, finished, sleeping])

    sleeping.join(timeout=1.0)
    assert status["stage_task_counts"]["stage0"] == 1
    assert status["stage_task_counts"]["stage1"] == 0
    assert status["ready"] is False


def test_wait_for_ready_returns_after_stage0_when_tpu_thread_exits() -> None:
    state = probe.BridgeState()
    claimed = state.claim(miner_id="stage0-miner", stage_id=0)
    assert state.submit(
        {
            "task_id": claimed["task"]["task_id"],
            "stage_id": 0,
            "activation": {"activation_hash": "sha256:stage0"},
            "activation_hash": "sha256:stage0",
            "output_hash": "sha256:out0",
            "runtime_device": {"backend": "cuda"},
            "kv_cache": {"ready": True},
        }
    )["accepted"] is True
    sleeping_cpu = threading.Thread(target=lambda: __import__("time").sleep(1.0))
    sleeping_gpu = threading.Thread(target=lambda: __import__("time").sleep(1.0))
    failed_tpu = threading.Thread(target=lambda: None)
    sleeping_gpu.start()
    failed_tpu.start()
    sleeping_cpu.start()
    failed_tpu.join(timeout=1.0)

    status = probe.wait_for_ready(state, timeout_seconds=5.0, threads=[sleeping_gpu, failed_tpu, sleeping_cpu])

    sleeping_gpu.join(timeout=2.0)
    sleeping_cpu.join(timeout=2.0)
    assert status["stage_task_counts"]["stage0"] == 1
    assert status["stage_task_counts"]["stage1"] == 0
    assert status["pending_count"] == 1
    assert status["ready"] is False


def test_wait_for_ready_keeps_coordinator_for_late_stage0_after_tpu_exit() -> None:
    state = probe.BridgeState()

    def delayed_stage0() -> None:
        import time

        time.sleep(0.1)
        claimed = state.claim(miner_id="stage0-miner", stage_id=0)
        assert claimed.get("task")
        state.submit(
            {
                "task_id": claimed["task"]["task_id"],
                "stage_id": 0,
                "activation": {"activation_hash": "sha256:stage0"},
                "activation_hash": "sha256:stage0",
                "output_hash": "sha256:out0",
                "runtime_device": {"backend": "cuda"},
                "kv_cache": {"ready": True},
            }
        )

    gpu = threading.Thread(target=delayed_stage0)
    failed_tpu = threading.Thread(target=lambda: None)
    cpu = threading.Thread(target=lambda: __import__("time").sleep(0.2))
    gpu.start()
    failed_tpu.start()
    cpu.start()
    failed_tpu.join(timeout=1.0)

    status = probe.wait_for_ready(state, timeout_seconds=2.0, threads=[gpu, failed_tpu, cpu])

    gpu.join(timeout=1.0)
    cpu.join(timeout=1.0)
    assert status["stage_task_counts"]["stage0"] == 1
    assert status["stage_task_counts"]["stage1"] == 0
    assert status["pending_count"] == 1
    assert status["ready"] is False


def test_bridge_report_does_not_require_cleanup_when_gpu_kernel_not_created(tmp_path: Path) -> None:
    args = probe.parse_args(["--output-dir", str(tmp_path)])
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        coordinator_status={
            "ready": False,
            "generated_token_count": 0,
            "activation_hashes": [],
            "stage_task_counts": {"stage0": 0, "stage1": 0, "stage2": 0},
            "completed_tasks": [],
        },
        gpu_report={},
        tpu_report={},
        cpu_report={},
        gpu_steps=[
            {
                "name": "kaggle_kernel_push",
                "ok": True,
                "accepted": False,
                "stdout_tail": "Kernel push error: Maximum batch GPU session count of 2 reached.",
            }
        ],
        errors=[],
    )

    assert "kaggle_gpu_batch_session_limit_reached" in report["blockers"]
    assert "kaggle_gpu_kernel_cleanup_not_verified" not in report["blockers"]
    assert report["cleanup"]["kaggle_gpu_kernel_created"] is False
    assert report["cleanup"]["kaggle_gpu_kernel_deleted"] is True
    assert "kaggle_gpu_kernel_not_created" in report["diagnosis_codes"]


def test_web_tpu_iframe_execution_has_subprocess_hard_timeout(tmp_path: Path) -> None:
    args = probe.parse_args(["--output-dir", str(tmp_path), "--web-tpu-execute-timeout-seconds", "30"])

    def timeout_runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["child"], timeout=1)

    report = probe.execute_web_tpu_code_via_iframe(args, "print('unused')", runner=timeout_runner)

    assert report["ok"] is False
    assert "web_tpu_jupyter_execute_timeout" in report["blockers"]
    assert report["jupyter_proxy_token_public"] is False
    assert probe.public_redaction_errors(report) == []


def test_web_tpu_subprocess_timeout_uses_bounded_padding(tmp_path: Path) -> None:
    short_args = probe.parse_args(["--output-dir", str(tmp_path), "--web-tpu-execute-timeout-seconds", "30"])
    long_args = probe.parse_args(["--output-dir", str(tmp_path), "--web-tpu-execute-timeout-seconds", "900"])

    assert probe.web_tpu_subprocess_timeout_seconds(short_args) == 40.0
    assert probe.web_tpu_subprocess_timeout_seconds(long_args) == 960.0


def test_web_tpu_iframe_execution_parses_child_result(tmp_path: Path) -> None:
    args = probe.parse_args(["--output-dir", str(tmp_path), "--web-tpu-execute-timeout-seconds", "30"])
    child_report = {
        "ok": True,
        "schema": "crowdtensor_web_tpu_min_smoke_v1",
        "tpu_device_count": 8,
        "web_tpu_jupyter_steps": [{"name": "service_manager_ready", "ok": True}],
        "jupyter_proxy_token_public": False,
    }

    def ok_runner(*_args, **_kwargs):
        stdout = json_dumps({"ok": True, "result": child_report})
        return subprocess.CompletedProcess(["child"], 0, stdout=stdout, stderr="")

    report = probe.execute_web_tpu_code_via_iframe(args, "print('unused')", runner=ok_runner)

    assert report["ok"] is True
    assert report["tpu_device_count"] == 8
    assert report["web_tpu_jupyter_steps"][-1]["name"] == "web_tpu_execute_subprocess"


def test_public_executor_attempts_redacts_private_jupyter_material() -> None:
    attempts = [
        {
            "executor_name": "browser_iframe_service_manager_ws",
            "errors_public": [{"ename": "Timeout", "message_public": "jupyter_ws_execute_timeout"}],
            "steps": [
                {"name": "service_manager_ready", "ok": True, "baseUrl": "https://secret/jupyter-proxy?token=abc"},
                {"name": "service_manager_ws_execute", "ok": False, "timeout": True, "wsUrl": "wss://secret"},
            ],
            "parsed_report": {
                "ok": False,
                "blockers": ["web_tpu_jupyter_execute_timeout"],
            },
        }
    ]

    cleaned = probe.public_executor_attempts(attempts)

    assert cleaned == [
        {
            "executor_name": "browser_iframe_service_manager_ws",
            "ok": False,
            "parsed_report_present": True,
            "parsed_report_ok": False,
            "blockers": ["web_tpu_jupyter_execute_timeout"],
            "error_count": 1,
            "error_names": ["Timeout"],
            "steps": [
                {"name": "service_manager_ready", "ok": True},
                {"name": "service_manager_ws_execute", "ok": False, "timeout": True},
            ],
        }
    ]
    assert probe.public_redaction_errors({"attempts": cleaned}) == []


def test_service_manager_executor_has_bounded_kernel_start_and_shutdown() -> None:
    source = probe.browser_service_manager_executor_js()

    assert "jupyter_session_start_timeout" in source
    assert "forceNewSession" in source
    assert "if (!forceNewSession && existingModel && sm.sessions.connectTo)" in source
    assert "session_connectTo_existing" in source
    assert "sm.sessions.connectTo" in source
    assert "jupyter_kernel_info_timeout" in source
    assert "kernel_info_ready" in source
    assert "jupyter_session_shutdown_timeout" in source
    assert "ownsSession" in source
    assert "session_shutdown" in source


def test_service_manager_ws_executor_uses_server_settings_ws_url() -> None:
    source = probe.browser_service_manager_ws_executor_js()

    assert "service_manager_ws_execute" in source
    assert "forceNewSession" in source
    assert "if (!forceNewSession && existingModel && sm.sessions.connectTo)" in source
    assert "session_connectTo_existing" in source
    assert "serverSettings" in source
    assert "wsUrl" in source
    assert "/api/kernels/${kernelId}/channels" in source


def test_web_tpu_direct_proxy_executor_reuses_existing_kernel() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")

    assert "def execute_web_tpu_code_via_proxy_kernel" in source
    assert "jupyter_proxy_kernel_reuse" in source
    assert "browser_proxy_existing_kernel_ws" in source
    assert "execute_web_tpu_code_via_proxy_kernel(args, code)" in source


def test_web_tpu_iframe_execution_keeps_service_manager_fallback_available() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")

    assert "(\"browser_iframe_existing_kernel_ws\", browser_iframe_executor_js())" in source
    assert source.index("(\"browser_iframe_service_manager\", browser_service_manager_executor_js())") < source.index("(\"browser_iframe_existing_kernel_ws\", browser_iframe_executor_js())")
    assert source.index("execute_web_tpu_code_via_proxy_kernel(args, code)") > source.index("(\"browser_iframe_existing_kernel_ws\", browser_iframe_executor_js())")
    assert "if report.get(\"ok\") is True or executor_name == \"browser_iframe_service_manager\"" in source
    assert "parsed_report" in source


def test_gpu_cleanup_join_waits_for_kaggle_status_output_and_delete_timeouts() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")

    assert "float(args.kaggle_status_timeout_seconds)" in source
    assert "float(args.kaggle_output_timeout_seconds)" in source
    assert "float(args.kaggle_delete_timeout_seconds)" in source
    assert "min(300.0" not in source


def test_kaggle_cpu_stage_uses_cpu_timeout_and_failure_stops_coordinator() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")

    assert "task_timeout_seconds=float(args.cpu_task_timeout_seconds)" in source
    assert "if not coordinator_status.get(\"ready\"):" in source
    assert "server_stopped = True" in source


def test_classify_jupyter_execution_errors_kernel_not_ready() -> None:
    classified = probe.classify_jupyter_execution_errors(
        [{"ename": "Error", "message_public": "jupyter_kernel_info_timeout"}]
    )

    assert classified == (
        "web_tpu_jupyter_kernel_not_ready",
        "bridge_web_tpu_jupyter_kernel_not_ready",
    )


def json_dumps(value) -> str:
    return __import__("json").dumps(value, sort_keys=True) + "\n"
