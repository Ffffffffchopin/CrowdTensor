from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import three_accelerator_dense_max_parameter_search_check as check
from scripts import three_accelerator_dense_max_parameter_search_pack as pack


class ThreeAcceleratorDenseMaxParameterSearchTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_dense_max_search_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _frontier_report(self, base: Path) -> Path:
        return self._write_json(
            base / "frontier.json",
            {
                "schema": "three_accelerator_dense_qwen_frontier_v1",
                "ok": True,
                "three_accelerator_dense_qwen_frontier_ready": True,
                "largest_dense_model_attempted": "72b",
                "largest_dense_model_attached": "72b",
                "largest_dense_model_stage_preflighted": "72b",
                "largest_dense_model_loaded": "32b",
                "largest_dense_model_1token_decoded": "32b",
                "same_request_dense_frontier_success": False,
                "same_request_dense_32b_success": True,
                "generated_token_count": 4,
                "frontier_failure_stage": "larger_dense_live_stage_load_not_verified_after_stage_preflight",
                "blocker_codes": [
                    "larger_dense_live_stage_load_not_verified_after_stage_preflight",
                    "larger_than_32b_dense_decode_not_verified",
                ],
                "baseline_32b_three_accelerator": {
                    "accepted_stage_backends": ["cpu", "cuda", "jax_tpu"],
                },
                "public_artifact_safe": True,
                "safety": {"public_artifact_safe": True},
            },
        )

    def _bridge_32b_report(self, base: Path) -> Path:
        return self._write_json(
            base / "bridge_32b.json",
            {
                "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
                "ok": True,
                "same_request_runtime_bridge_verified": True,
                "gpu_tpu_cpu_32b_same_request_verified": True,
                "same_request_32b_model_verified": True,
                "target_model_id": "Qwen/Qwen2.5-32B-Instruct",
                "generated_token_count": 4,
                "accepted_stage_backends": ["cpu", "cuda", "jax_tpu"],
                "stage_local_kv_cache_verified": True,
                "tpu_32b_runtime_adapter_ready": True,
                "public_artifact_safe": True,
            },
        )

    def _attach_72b_stage_plan_report(self, base: Path) -> Path:
        stage_plan = {
            "schema": "kaggle_model_attach_stage_plan_v1",
            "stage_count": 10,
            "stage_backends": ["cuda", "cuda", "cuda", "cuda", "jax_tpu", "cpu", "cpu", "cpu", "cpu", "cpu"],
            "assigned_key_count_total": 963,
            "present_key_count_total": 963,
            "total_planned_logical_tensor_gb": 145.412407,
            "max_stage_planned_logical_tensor_gb": 16.534389,
        }
        return self._write_json(
            base / "attach_72b.json",
            {
                "schema": "kaggle_model_attach_probe_v1",
                "ok": True,
                "kaggle_model_attach_probe_ready": True,
                "kaggle_model_attach_used": True,
                "parameter_class": "72b",
                "hf_repo": "Qwen/Qwen2.5-72B-Instruct",
                "model_source": "qwen-lm/qwen2.5/Transformers/72b-instruct/1",
                "expected_attached_path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/72b-instruct/1",
                "runtime_report": {
                    "ok": True,
                    "path_present": True,
                    "weight_index_present": True,
                    "safetensors_file_count": 37,
                    "weight_index_key_count": 963,
                    "quantization_config_present": False,
                    "stage_owned_preflight_verified": True,
                    "stage_plan": stage_plan,
                },
                "cleanup_status": {
                    "temporary_kaggle_kernel_deleted": True,
                    "temporary_private_package_removed": True,
                    "live_resources_left_running": False,
                },
                "public_artifact_safe": True,
            },
        )

    def _tpu_72b_timeout_report(self, base: Path) -> Path:
        return self._write_json(
            base / "tpu_72b_timeout.json",
            {
                "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
                "ok": False,
                "model_repo": "Qwen/Qwen2.5-72B-Instruct",
                "stage_layer_range": [32, 40],
                "stage_owned_header_verified": False,
                "partial_tensor_to_tpu_verified": False,
                "full_stage_owned_tpu_loader_ready": False,
                "stage_local_kv_cache_verified": False,
                "executed_layer_count": 0,
                "loaded_execution_tensor_key_count": 0,
                "tpu_device_count": 0,
                "blocked_reason": "web_tpu_jupyter_execute_timeout",
                "blockers": ["web_tpu_jupyter_execute_timeout"],
                "runtime_report": {
                    "ok": False,
                    "blockers": ["web_tpu_jupyter_execute_timeout"],
                },
                "kaggle_lifecycle": {
                    "kernels_deleted": True,
                    "private_packages_removed": True,
                    "private_kernel_push_count": 0,
                    "web_runtime_execution_count": 0,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _tpu_72b_full_stage_report(self, base: Path) -> Path:
        return self._write_json(
            base / "tpu_72b_full_stage.json",
            {
                "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
                "ok": True,
                "model_repo": "Qwen/Qwen2.5-72B-Instruct",
                "stage_layer_range": [32, 40],
                "stage_owned_header_verified": True,
                "partial_tensor_to_tpu_verified": True,
                "full_stage_owned_tpu_loader_ready": True,
                "stage_local_kv_cache_verified": True,
                "executed_layer_count": 8,
                "loaded_execution_tensor_key_count": 96,
                "loaded_execution_tensor_gb": 13.078522,
                "tpu_device_count": 8,
                "tpu_device_kind": "TPU v5 lite",
                "stage_output_hash": "sha256:72btpuout",
                "blocked_reason": "",
                "blockers": [],
                "kaggle_lifecycle": {
                    "kernels_deleted": True,
                    "private_packages_removed": True,
                    "private_kernel_push_count": 0,
                    "web_runtime_execution_count": 1,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _colab_tpu_72b_stage_report(self, base: Path) -> Path:
        return self._write_json(
            base / "colab_tpu_72b_stage.json",
            {
                "schema": "colab_tpu_qwen_stage_loader_probe_v1",
                "ok": True,
                "colab_qwen_stage_loader_ready": True,
                "model_repo": "Qwen/Qwen2.5-72B-Instruct",
                "stage_layer_range": [32, 36],
                "executed_layer_count": 4,
                "loaded_execution_tensor_key_count": 48,
                "loaded_execution_tensor_gb": 6.539261,
                "missing_stage_key_count": 0,
                "tpu_device_count": 1,
                "stage_output_hash": "sha256:colab72bout",
                "runtime_proxy_token_public": False,
                "runtime_proxy_url_public": False,
                "endpoint_public": False,
                "public_artifact_safe": True,
                "runtime_report": {
                    "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
                    "ok": True,
                    "model_repo": "Qwen/Qwen2.5-72B-Instruct",
                    "stage_layer_range": [32, 36],
                    "stage_owned_header_verified": True,
                    "partial_tensor_to_tpu_verified": True,
                    "full_stage_owned_tpu_loader_ready": True,
                    "stage_local_kv_cache_verified": True,
                    "executed_layer_count": 4,
                    "loaded_execution_tensor_key_count": 48,
                    "loaded_execution_tensor_gb": 6.539261,
                    "tpu_device_count": 1,
                    "tpu_device_kind": "TPU v5 lite",
                    "stage_output_hash": "sha256:colab72bout",
                    "public_artifact_safe": True,
                },
            },
        )

    def _bridge_72b_stage_report(self, base: Path) -> Path:
        return self._write_json(
            base / "bridge_72b_stage.json",
            {
                "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
                "ok": True,
                "same_request_runtime_bridge_verified": True,
                "same_request_target_parameter_class": "72b",
                "same_request_72b_stage_verified": True,
                "gpu_tpu_cpu_72b_same_request_stage_verified": True,
                "same_request_72b_full_model_verified": False,
                "gpu_tpu_cpu_72b_same_request_verified": False,
                "target_model_id": "Qwen/Qwen2.5-72B-Instruct",
                "generated_token_count": 1,
                "accepted_stage_backends": ["cpu", "cuda", "jax_tpu"],
                "tpu_target_runtime_adapter_ready": True,
                "full_72b_tpu_stage_loading_public_claim": True,
                "full_72b_weight_loading_public_claim": False,
                "runtime_device_summary": {
                    "tpu_executed_layer_count": 8,
                    "tpu_loaded_execution_tensor_gb": 13.078522,
                },
                "stage_reports": {
                    "jax_tpu_stage": {
                        "stage_layer_range": [32, 40],
                    },
                },
                "blockers": ["qwen72b_full_model_same_request_decode_not_verified"],
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _bridge_72b_full_heterogeneous_report(self, base: Path) -> Path:
        return self._write_json(
            base / "bridge_72b_full_heterogeneous.json",
            {
                "schema": "kaggle_32b_full_heterogeneous_probe_v1",
                "ok": True,
                "model": {
                    "repo": "Qwen/Qwen2.5-72B-Instruct",
                    "parameter_count_b": 72,
                    "quantization": "none",
                    "stage_count": 10,
                    "expected_layer_count": 80,
                    "full_layer_coverage_verified": True,
                },
                "generated_token_count": 1,
                "gpu_tpu_cpu_72b_same_request_verified": True,
                "same_request_72b_full_model_verified": True,
                "full_72b_weight_loading_public_claim": True,
                "full_72b_layer_coverage_verified": True,
                "gpu_tpu_cpu_72b_full_topology_verified": True,
                "stage_owned_full_precision_runtime_verified": True,
                "stage_summaries": [
                    *[
                        {
                            "stage_id": i,
                            "resource_kind": "gpu",
                            "stage_weight_load_ready": True,
                            "runtime_buffers_ready": True,
                        }
                        for i in range(4)
                    ],
                    {
                        "stage_id": 4,
                        "resource_kind": "web_tpu",
                        "stage_weight_load_ready": True,
                        "runtime_buffers_ready": True,
                        "stage_layer_range": [32, 40],
                    },
                    *[
                        {
                            "stage_id": i,
                            "resource_kind": "cpu",
                            "stage_weight_load_ready": True,
                            "runtime_buffers_ready": True,
                        }
                        for i in range(5, 10)
                    ],
                ],
                "blockers": [],
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _web_tpu_channel_timeout_report(self, base: Path) -> Path:
        return self._write_json(
            base / "web_tpu_channel_timeout.json",
            {
                "schema": "kaggle_web_tpu_execution_channel_probe_v1",
                "ok": False,
                "web_tpu_execution_channel_ready": False,
                "small_jax_cell_ready": False,
                "tiny_qwen_like_cell_ready": False,
                "tpu_runtime_attached": False,
                "tpu_device_count": 0,
                "failure_stage": "jupyter_execute",
                "blocked_reason": "tiny_qwen_like_not_attempted_after_small_jax_failure",
                "blocker_codes": [
                    "web_tpu_execution_channel_not_ready",
                    "web_tpu_jupyter_execute_timeout",
                    "tiny_qwen_like_not_attempted_after_small_jax_failure",
                ],
                "cleanup_status": {
                    "temporary_kaggle_kernels_deleted": True,
                    "temporary_private_packages_removed": True,
                    "live_resources_left_running": False,
                    "web_runtime_execution_count": 0,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _web_tpu_channel_ready_report(self, base: Path) -> Path:
        return self._write_json(
            base / "web_tpu_channel_ready.json",
            {
                "schema": "kaggle_web_tpu_execution_channel_probe_v1",
                "ok": True,
                "web_tpu_execution_channel_ready": True,
                "small_jax_cell_ready": True,
                "tiny_qwen_like_cell_ready": True,
                "tpu_runtime_attached": True,
                "tpu_device_count": 8,
                "failure_stage": "",
                "blocked_reason": "",
                "blocker_codes": [],
                "cleanup_status": {
                    "temporary_kaggle_kernels_deleted": True,
                    "temporary_private_packages_removed": True,
                    "live_resources_left_running": False,
                    "web_runtime_execution_count": 2,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _web_tpu_active_event_running_no_frame_report(self, base: Path) -> Path:
        return self._write_json(
            base / "web_tpu_active_event_running_no_frame.json",
            {
                "schema": "kaggle_web_tpu_active_event_probe_v1",
                "ok": True,
                "active_event_probe_ready": True,
                "active_event_dialog_opened": True,
                "active_event_count": 1,
                "tpu_v5e_active_event_visible": True,
                "active_event_queued": False,
                "active_event_running": True,
                "active_event_runtime_ready": False,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "jupyter_session_count": 0,
                "jupyter_kernel_count": 0,
                "blocked_reason": "kaggle_web_tpu_jupyter_frame_not_visible",
                "blocker_codes": [
                    "kaggle_web_tpu_jupyter_frame_not_visible",
                    "kaggle_web_tpu_jupyter_session_not_visible",
                ],
                "cleanup_status": {
                    "temporary_kaggle_kernels_deleted": True,
                    "temporary_private_packages_removed": True,
                    "live_resources_left_running": False,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _web_tpu_active_event_queued_report(self, base: Path) -> Path:
        return self._write_json(
            base / "web_tpu_active_event_queued.json",
            {
                "schema": "kaggle_web_tpu_active_event_probe_v1",
                "ok": True,
                "active_event_probe_ready": True,
                "active_event_dialog_opened": True,
                "active_event_count": 1,
                "tpu_v5e_active_event_visible": True,
                "active_event_queued": True,
                "active_event_running": False,
                "active_event_runtime_ready": False,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "jupyter_session_count": 0,
                "jupyter_kernel_count": 0,
                "blocked_reason": "kaggle_web_tpu_active_event_queued",
                "blocker_codes": [
                    "kaggle_web_tpu_active_event_queued",
                    "kaggle_web_tpu_active_event_not_running",
                    "kaggle_web_tpu_jupyter_frame_not_visible",
                    "kaggle_web_tpu_jupyter_session_not_visible",
                ],
                "cleanup_status": {
                    "temporary_kaggle_kernels_deleted": True,
                    "temporary_private_packages_removed": True,
                    "live_resources_left_running": False,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _web_tpu_start_wait_not_ready_report(self, base: Path) -> Path:
        return self._write_json(
            base / "web_tpu_start_wait_not_ready.json",
            {
                "schema": "kaggle_web_tpu_start_wait_probe_v1",
                "ok": True,
                "start_clicked": True,
                "web_tpu_ui_runtime_ready": False,
                "bounded_wait_seconds": 300.0,
                "blocked_reason": "kaggle_web_tpu_jupyter_frame_not_visible",
                "blocker_codes": [
                    "kaggle_web_tpu_queue_visible",
                    "kaggle_web_tpu_session_still_starting",
                    "kaggle_web_tpu_jupyter_frame_not_visible",
                    "kaggle_web_tpu_jupyter_session_not_visible",
                ],
                "cleanup_status": {
                    "temporary_kaggle_kernels_deleted": True,
                    "temporary_private_packages_removed": True,
                    "live_resources_left_running": False,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _web_tpu_start_wait_start_click_not_accepted_report(self, base: Path) -> Path:
        return self._write_json(
            base / "web_tpu_start_wait_click_not_accepted.json",
            {
                "schema": "kaggle_web_tpu_start_wait_probe_v1",
                "ok": True,
                "start_clicked": False,
                "web_tpu_ui_runtime_ready": False,
                "bounded_wait_seconds": 30.0,
                "blocked_reason": "kaggle_web_tpu_jupyter_frame_not_visible",
                "blocker_codes": [
                    "kaggle_web_tpu_jupyter_frame_not_visible",
                    "kaggle_web_tpu_jupyter_session_not_visible",
                    "kaggle_web_tpu_session_still_starting",
                    "kaggle_web_tpu_start_session_not_clicked",
                    "kaggle_web_tpu_start_session_visible",
                ],
                "cleanup_status": {
                    "temporary_kaggle_kernels_deleted": True,
                    "temporary_private_packages_removed": True,
                    "live_resources_left_running": False,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _colab_tpu_reacquire_failed_report(self, base: Path) -> Path:
        return self._write_json(
            base / "colab_tpu_reacquire_failed.json",
            {
                "schema": "colab_tpu_reacquire_retry_probe_v1",
                "ok": False,
                "colab_tpu_reacquire_ready": False,
                "session_name": "ct-colab-tpu-v5e1",
                "accelerators_attempted": ["V5E1", "V6E1"],
                "attempts_requested": 2,
                "attempts_completed": 2,
                "successful_attempt_index": 0,
                "successful_report_path": "",
                "blockers": [
                    "colab_assignment_http_503",
                    "colab_assignment_http_400",
                    "colab_tpu_session_not_allocated",
                ],
                "attempts": [
                    {
                        "attempt_index": 1,
                        "accelerator_requested": "V5E1",
                        "ok": False,
                        "http_status": 503,
                        "blockers": ["colab_assignment_http_503"],
                        "public_artifact_safe": True,
                    },
                    {
                        "attempt_index": 2,
                        "accelerator_requested": "V6E1",
                        "ok": False,
                        "http_status": 400,
                        "blockers": ["colab_assignment_http_400"],
                        "public_artifact_safe": True,
                    },
                ],
                "endpoint_hash": "",
                "runtime_proxy_host_hash": "",
                "public_artifact_safe": True,
                "oauth_token_public": False,
                "runtime_proxy_token_public": False,
                "runtime_proxy_url_public": False,
                "endpoint_public": False,
                "credentials_public": False,
                "private_runtime_state_public": False,
            },
        )

    def _colab_tpu_runtime_stability_failed_report(self, base: Path) -> Path:
        return self._write_json(
            base / "colab_tpu_runtime_stability_failed.json",
            {
                "schema": "colab_tpu_runtime_stability_probe_v1",
                "ok": False,
                "colab_tpu_runtime_stably_acquired": False,
                "runtime_proxy_connected": False,
                "rounds_requested": 1,
                "rounds_completed": 0,
                "rounds_ready": 0,
                "observed_device_count_max": 0,
                "accelerator": "V5E1",
                "variant": "TPU",
                "endpoint_hash": "abc123",
                "runtime_proxy_host_hash": "def456",
                "kernel_id_hash": "ghi789",
                "session_id_hash": "jkl012",
                "kernel_error": "HTTPError: 404 Client Error",
                "runtime_proxy_token_public": False,
                "runtime_proxy_url_public": False,
                "endpoint_public": False,
                "public_artifact_safe": True,
            },
        )

    def _build_report(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_timeout_report(base)),
                ]
            )
        )

    def _build_report_with_colab_reacquire_failure(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--bridge-72b-report",
                    str(self._bridge_72b_stage_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._colab_tpu_72b_stage_report(base)),
                    "--colab-tpu-reacquire-report",
                    str(self._colab_tpu_reacquire_failed_report(base)),
                ]
            )
        )

    def _build_report_with_colab_reacquire_and_runtime_failure(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--bridge-72b-report",
                    str(self._bridge_72b_stage_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._colab_tpu_72b_stage_report(base)),
                    "--colab-tpu-reacquire-report",
                    str(self._colab_tpu_reacquire_failed_report(base)),
                    "--colab-tpu-runtime-stability-report",
                    str(self._colab_tpu_runtime_stability_failed_report(base)),
                ]
            )
        )

    def _build_report_with_active_event(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_timeout_report(base)),
                    "--web-tpu-active-event-report",
                    str(self._web_tpu_active_event_queued_report(base)),
                ]
            )
        )

    def _build_report_with_start_wait(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_timeout_report(base)),
                    "--web-tpu-start-wait-report",
                    str(self._web_tpu_start_wait_not_ready_report(base)),
                ]
            )
        )

    def _build_report_with_channel(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_timeout_report(base)),
                    "--web-tpu-channel-report",
                    str(self._web_tpu_channel_timeout_report(base)),
                ]
            )
        )

    def _build_report_with_ready_channel_and_nonready_active_event(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_timeout_report(base)),
                    "--web-tpu-channel-report",
                    str(self._web_tpu_channel_ready_report(base)),
                    "--web-tpu-active-event-report",
                    str(self._web_tpu_active_event_running_no_frame_report(base)),
                ]
            )
        )

    def _build_report_with_72b_stage_bridge(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--bridge-72b-report",
                    str(self._bridge_72b_stage_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_full_stage_report(base)),
                ]
            )
        )

    def _build_report_with_colab_72b_stage_loader(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--bridge-72b-report",
                    str(self._bridge_72b_stage_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._colab_tpu_72b_stage_report(base)),
                ]
            )
        )

    def _build_report_with_72b_full_heterogeneous_bridge(self, base: Path) -> dict:
        return pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--bridge-72b-report",
                    str(self._bridge_72b_full_heterogeneous_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_full_stage_report(base)),
                ]
            )
        )

    def test_current_search_records_72b_attempt_but_only_32b_decode(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)

        self.assertTrue(report["three_accelerator_dense_max_parameter_search_ready"])
        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "32b")
        self.assertEqual(report["max_attempted_parameter_class"], "72b")
        self.assertEqual(report["max_attached_parameter_class"], "72b")
        self.assertEqual(report["max_stage_preflighted_parameter_class"], "72b")
        self.assertEqual(report["max_stage_loaded_parameter_class"], "32b")
        self.assertEqual(report["max_tpu_executed_parameter_class"], "32b")
        self.assertEqual(report["generated_token_count"], 4)
        self.assertEqual(set(report["accepted_stage_backends"]), {"cpu", "cuda", "jax_tpu"})
        self.assertIn("web_tpu_jupyter_execute_timeout", report["blocker_codes"])
        self.assertIn("dense_72b_tpu_stage_load_and_forward_not_verified", report["blocker_codes"])
        self.assertEqual(
            report["failure_stage"],
            "tpu_web_jupyter_execute_timeout_before_72b_header_load",
        )
        self.assertEqual(check.validate_report(report), [])

    def test_current_search_imports_web_tpu_channel_blocker(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_channel(base)

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "32b")
        self.assertEqual(report["max_attempted_parameter_class"], "72b")
        self.assertEqual(report["max_stage_loaded_parameter_class"], "32b")
        self.assertEqual(report["max_tpu_executed_parameter_class"], "32b")
        self.assertFalse(report["web_tpu_execution_channel_import"]["web_tpu_execution_channel_ready"])
        self.assertEqual(report["failure_stage"], "web_tpu_channel_jupyter_execute")
        self.assertIn("web_tpu_execution_channel_not_ready", report["blocker_codes"])
        self.assertIn("web_tpu_channel_jupyter_execute", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_current_search_imports_web_tpu_active_event_queue_blocker(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_active_event(base)

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "32b")
        self.assertFalse(report["web_tpu_active_event_import"]["active_event_runtime_ready"])
        self.assertTrue(report["web_tpu_active_event_import"]["active_event_queued"])
        self.assertEqual(report["failure_stage"], "web_tpu_active_event_active_event_queued")
        self.assertIn("web_tpu_active_event_not_ready", report["blocker_codes"])
        self.assertIn("web_tpu_active_event_active_event_queued", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_current_search_imports_web_tpu_start_wait_blocker(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_start_wait(base)

        self.assertFalse(report["web_tpu_start_wait_import"]["web_tpu_ui_runtime_ready"])
        self.assertTrue(report["web_tpu_start_wait_import"]["start_clicked"])
        self.assertEqual(report["failure_stage"], "web_tpu_start_wait_runtime_not_ready")
        self.assertIn("web_tpu_start_wait_runtime_not_ready", report["blocker_codes"])
        self.assertIn("kaggle_web_tpu_session_still_starting", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_current_search_accepts_start_wait_click_not_accepted_blocker(self) -> None:
        base = self._tmp_dir()
        output_dir = base / "out"
        args = pack.parse_args(
            [
                "--output-dir",
                str(output_dir),
                "--frontier-report",
                str(self._frontier_report(base)),
                "--bridge-32b-report",
                str(self._bridge_32b_report(base)),
                "--attach-72b-stage-plan-report",
                str(self._attach_72b_stage_plan_report(base)),
                "--tpu-72b-stage-load-report",
                str(self._tpu_72b_timeout_report(base)),
                "--web-tpu-start-wait-report",
                str(self._web_tpu_start_wait_start_click_not_accepted_report(base)),
            ]
        )
        report = pack.build_report(args)

        self.assertFalse(report["web_tpu_start_wait_import"]["start_clicked"])
        self.assertIn("kaggle_web_tpu_start_session_not_clicked", report["blocker_codes"])
        self.assertIn("web_tpu_start_wait_runtime_not_ready", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_ready_web_tpu_channel_overrides_active_event_ui_frame_gap(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_ready_channel_and_nonready_active_event(base)

        self.assertTrue(report["web_tpu_execution_channel_import"]["web_tpu_execution_channel_ready"])
        self.assertFalse(report["web_tpu_active_event_import"]["active_event_runtime_ready"])
        self.assertTrue(report["web_tpu_active_event_overridden_by_execution_channel"])
        self.assertFalse(report.get("colab_tpu_reacquire_overridden_by_web_tpu_channel", False))
        self.assertNotIn("web_tpu_active_event_not_ready", report["blocker_codes"])
        self.assertNotIn("web_tpu_active_event_active_event_jupyter_frame_not_visible", report["blocker_codes"])
        self.assertEqual(
            report["failure_stage"],
            "tpu_web_jupyter_execute_timeout_before_72b_header_load",
        )
        self.assertEqual(check.validate_report(report), [])

    def test_ready_web_tpu_channel_overrides_stale_start_wait_failure(self) -> None:
        base = self._tmp_dir()
        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_timeout_report(base)),
                    "--web-tpu-channel-report",
                    str(self._web_tpu_channel_ready_report(base)),
                    "--web-tpu-start-wait-report",
                    str(self._web_tpu_start_wait_not_ready_report(base)),
                ]
            )
        )

        self.assertTrue(report["web_tpu_execution_channel_import"]["web_tpu_execution_channel_ready"])
        self.assertFalse(report["web_tpu_start_wait_import"]["web_tpu_ui_runtime_ready"])
        self.assertTrue(report["web_tpu_start_wait_overridden_by_execution_channel"])
        self.assertNotIn("web_tpu_start_wait_runtime_not_ready", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_ready_web_tpu_channel_overrides_colab_reacquire_fallback_failure(self) -> None:
        base = self._tmp_dir()
        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--bridge-72b-report",
                    str(self._bridge_72b_stage_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._colab_tpu_72b_stage_report(base)),
                    "--web-tpu-channel-report",
                    str(self._web_tpu_channel_ready_report(base)),
                    "--colab-tpu-reacquire-report",
                    str(self._colab_tpu_reacquire_failed_report(base)),
                ]
            )
        )

        self.assertTrue(report["web_tpu_execution_channel_import"]["web_tpu_execution_channel_ready"])
        self.assertFalse(report["colab_tpu_reacquire_import"]["colab_tpu_reacquire_ready"])
        self.assertTrue(report["colab_tpu_reacquire_overridden_by_web_tpu_channel"])
        self.assertNotIn("colab_tpu_reacquire_not_ready", report["blocker_codes"])
        self.assertEqual(
            report["failure_stage"],
            "dense_72b_stage_same_request_verified_but_full_model_decode_not_verified",
        )
        self.assertEqual(check.validate_report(report), [])

    def test_72b_stage_bridge_does_not_count_as_full_72b_decode(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_72b_stage_bridge(base)

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "32b")
        self.assertEqual(report["max_stage_loaded_parameter_class"], "72b")
        self.assertEqual(report["max_tpu_executed_parameter_class"], "72b")
        self.assertTrue(report["same_request_72b_import"]["same_request_stage_decode_verified"])
        self.assertFalse(report["same_request_72b_import"]["same_request_full_model_decode_verified"])
        self.assertTrue(report["attempt_ladder"][1]["same_request_stage_decode_verified"])
        self.assertFalse(report["attempt_ladder"][1]["same_request_decode_verified"])
        self.assertIn(
            "dense_72b_same_request_stage_verified_but_full_model_decode_not_verified",
            report["blocker_codes"],
        )
        self.assertEqual(
            report["failure_stage"],
            "dense_72b_stage_same_request_verified_but_full_model_decode_not_verified",
        )
        self.assertEqual(check.validate_report(report), [])

    def test_colab_72b_stage_loader_counts_as_stage_execution_not_full_decode(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_colab_72b_stage_loader(base)

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "32b")
        self.assertEqual(report["max_stage_loaded_parameter_class"], "72b")
        self.assertEqual(report["max_tpu_executed_parameter_class"], "72b")
        self.assertEqual(report["dense_72b_tpu_stage_load_attempt"]["provider"], "colab_cli")
        self.assertTrue(report["dense_72b_tpu_stage_load_attempt"]["tpu_72b_stage_load_and_forward_verified"])
        self.assertEqual(report["dense_72b_tpu_stage_load_attempt"]["tpu_device_count"], 1)
        self.assertTrue(report["same_request_72b_import"]["same_request_stage_decode_verified"])
        self.assertFalse(report["same_request_72b_import"]["same_request_full_model_decode_verified"])
        self.assertIn(
            "dense_72b_same_request_stage_verified_but_full_model_decode_not_verified",
            report["blocker_codes"],
        )
        self.assertEqual(check.validate_report(report), [])

    def test_colab_reacquire_failure_imports_current_blocker_without_raising_max_decode(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_colab_reacquire_failure(base)

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "32b")
        self.assertEqual(report["max_stage_loaded_parameter_class"], "72b")
        self.assertEqual(report["max_tpu_executed_parameter_class"], "72b")
        self.assertFalse(report["colab_tpu_reacquire_import"]["colab_tpu_reacquire_ready"])
        self.assertEqual(report["failure_stage"], "colab_tpu_reacquire_not_ready")
        self.assertIn("colab_tpu_reacquire_not_ready", report["blocker_codes"])
        self.assertIn("colab_assignment_http_503", report["blocker_codes"])
        self.assertIn("colab_assignment_http_400", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_colab_existing_runtime_failure_is_imported_without_private_proxy(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_colab_reacquire_and_runtime_failure(base)

        self.assertFalse(report["colab_tpu_runtime_stability_import"]["colab_tpu_runtime_stably_acquired"])
        self.assertFalse(report["colab_tpu_runtime_stability_import"]["runtime_proxy_connected"])
        self.assertTrue(report["colab_tpu_runtime_stability_import"]["endpoint_hash_present"])
        self.assertTrue(report["colab_tpu_runtime_stability_import"]["runtime_proxy_host_hash_present"])
        self.assertEqual(report["colab_tpu_runtime_stability_import"]["kernel_error_type"], "HTTPError")
        self.assertFalse(report["colab_tpu_runtime_stability_import"]["kernel_error_public"])
        self.assertIn("colab_tpu_runtime_stability_not_ready", report["blocker_codes"])
        self.assertEqual(report["failure_stage"], "colab_tpu_reacquire_not_ready")
        self.assertEqual(check.validate_report(report), [])

    def test_ready_web_tpu_channel_overrides_colab_runtime_stability_fallback_failure(self) -> None:
        base = self._tmp_dir()
        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--bridge-72b-report",
                    str(self._bridge_72b_stage_report(base)),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._colab_tpu_72b_stage_report(base)),
                    "--web-tpu-channel-report",
                    str(self._web_tpu_channel_ready_report(base)),
                    "--colab-tpu-runtime-stability-report",
                    str(self._colab_tpu_runtime_stability_failed_report(base)),
                ]
            )
        )

        self.assertTrue(report["web_tpu_execution_channel_import"]["web_tpu_execution_channel_ready"])
        self.assertTrue(report["colab_tpu_runtime_stability_overridden"])
        self.assertNotIn("colab_tpu_runtime_stability_not_ready", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_72b_full_heterogeneous_report_counts_as_72b_decode(self) -> None:
        base = self._tmp_dir()
        report = self._build_report_with_72b_full_heterogeneous_bridge(base)

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "72b")
        self.assertEqual(report["generated_token_count"], 1)
        self.assertEqual(set(report["accepted_stage_backends"]), {"cpu", "cuda", "jax_tpu"})
        self.assertTrue(report["same_request_72b_import"]["same_request_full_model_decode_verified"])
        self.assertTrue(report["same_request_72b_import"]["full_72b_layer_coverage_verified"])
        self.assertEqual(check.validate_report(report), [])

    def test_72b_full_heterogeneous_without_layer_coverage_does_not_count_as_decode(self) -> None:
        base = self._tmp_dir()
        bridge = self._bridge_72b_full_heterogeneous_report(base)
        loaded = json.loads(bridge.read_text(encoding="utf-8"))
        loaded["full_72b_layer_coverage_verified"] = False
        loaded["model"]["full_layer_coverage_verified"] = False
        bridge.write_text(json.dumps(loaded), encoding="utf-8")
        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "max-search"),
                    "--frontier-report",
                    str(self._frontier_report(base)),
                    "--bridge-32b-report",
                    str(self._bridge_32b_report(base)),
                    "--bridge-72b-report",
                    str(bridge),
                    "--attach-72b-stage-plan-report",
                    str(self._attach_72b_stage_plan_report(base)),
                    "--tpu-72b-stage-load-report",
                    str(self._tpu_72b_full_stage_report(base)),
                ]
            )
        )

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "32b")
        self.assertFalse(report["same_request_72b_import"]["same_request_full_model_decode_verified"])
        self.assertFalse(report["same_request_72b_import"]["full_72b_layer_coverage_verified"])
        self.assertEqual(check.validate_report(report), [])

    def test_checker_rejects_72b_decode_overclaim(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)
        report["max_successful_same_request_decode_parameter_class"] = "72b"

        errors = check.validate_report(report)

        self.assertIn("larger_decode_overclaim_without_same_request_ladder_proof", errors)
        self.assertIn("max_decode_72b_without_full_model_bridge_proof", errors)

    def test_checker_rejects_72b_tpu_executed_overclaim(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)
        report["max_tpu_executed_parameter_class"] = "72b"

        errors = check.validate_report(report)

        self.assertIn("max_tpu_executed_72b_without_tpu_stage_forward", errors)
        self.assertIn("max_tpu_executed_72b_without_loaded_keys", errors)
        self.assertIn("max_tpu_executed_72b_without_layer_forward", errors)

    def test_checker_rejects_72b_stage_loaded_overclaim(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)
        report["max_stage_loaded_parameter_class"] = "72b"

        errors = check.validate_report(report)

        self.assertIn("max_stage_loaded_72b_without_72b_tpu_load_forward", errors)

    def test_public_artifacts_are_redacted(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)

        self.assertEqual(pack.public_redaction_errors(report), [])
        scanned = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (base / "max-search").rglob("*")
            if path.is_file()
        )
        for fragment in [
            "KAGGLE_KEY",
            "HF_TOKEN",
            "Bearer ",
            "Cookie:",
            "kaggle-cookies",
            "JUPYTER_TOKEN",
            '"prompt":',
            '"generated_text":',
            '"generated_token_ids":',
            '"activation":',
            '"hidden_state":',
            '"logits":',
            '"kv_cache":',
            '"past_key_values":',
            '"lease_token":',
            '"idempotency_key":',
        ]:
            self.assertNotIn(fragment, scanned)


if __name__ == "__main__":
    unittest.main()
