from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_cpu_32b_heterogeneous_rc_check as check
from scripts import gpu_tpu_cpu_32b_heterogeneous_rc_pack as pack
from scripts import gpu_tpu_qwen_stage_adapter_plan as adapter_plan
from scripts import kaggle_tpu_qwen_stage_runtime_probe as tpu_runtime_probe
from crowdtensor import cli


class GpuTpuCpu32BHeterogeneousRcTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_gpu_tpu_cpu_32b_rc_test_"))

    def _base_args(self, output_dir: Path, *extra: str) -> list[str]:
        return [
            "--output-dir",
            str(output_dir),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            *extra,
        ]

    def test_blocker_report_is_valid_without_live_same_request_proof(self) -> None:
        output_dir = self._tmp_dir() / "blocker"
        report = pack.build_report(pack.parse_args(self._base_args(output_dir)))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["gpu_tpu_cpu_32b_heterogeneous_rc_ready"])
        self.assertFalse(report["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertFalse(report["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertFalse(report["live_tpu_stage_miner_integrated"])
        self.assertFalse(report["tpu_32b_runtime_adapter_ready"])
        self.assertFalse(report["stage_local_kv_cache_verified"])
        self.assertEqual(report["blocked_reason"], "same_request_live_proof_missing")
        self.assertIn("jax_tpu_llama_like_stage_runtime", report["blocker_report"]["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_fixture_success_models_true_32b_same_request_rc(self) -> None:
        output_dir = self._tmp_dir() / "success"
        report = pack.build_report(pack.parse_args(self._base_args(output_dir, "--live-proof-mode", "fixture-success")))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertTrue(report["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertTrue(report["live_tpu_stage_miner_integrated"])
        self.assertTrue(report["tpu_32b_runtime_adapter_ready"])
        self.assertTrue(report["stage_local_kv_cache_verified"])
        self.assertFalse(report["fallback_model_used"])
        self.assertEqual(report["blocked_reason"], "")
        self.assertEqual(check.validate_report(report), [])

    def test_fallback_live_proof_does_not_count_as_32b_success(self) -> None:
        output_dir = self._tmp_dir() / "fallback"
        report = pack.build_report(pack.parse_args(self._base_args(output_dir, "--live-proof-mode", "fixture-fallback")))

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertFalse(report["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertTrue(report["fallback_model_used"])
        self.assertTrue(report["live_tpu_stage_miner_integrated"])
        self.assertFalse(report["tpu_32b_runtime_adapter_ready"])
        self.assertIn("live_proof_not_32b_class", report["blocker_report"]["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_checker_rejects_overclaimed_success_without_live_tpu_stage(self) -> None:
        output_dir = self._tmp_dir() / "overclaim"
        report = pack.build_report(pack.parse_args(self._base_args(output_dir)))
        report["gpu_tpu_cpu_32b_bounded_rc_success"] = True
        report["gpu_tpu_cpu_32b_same_request_verified"] = True
        report["blocked_reason"] = ""

        errors = check.validate_report(report)

        self.assertIn("success_without_live_summary_success", errors)
        self.assertIn("success_without_live_tpu_stage", errors)
        self.assertIn("success_without_tpu_32b_adapter", errors)

    def test_public_artifacts_are_redacted(self) -> None:
        output_dir = self._tmp_dir() / "redaction"
        report = pack.build_report(pack.parse_args(self._base_args(output_dir)))

        self.assertEqual(pack.public_redaction_errors(report), [])
        scanned = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        for fragment in [
            "KAGGLE_KEY=",
            "KAGGLE_USERNAME=",
            "HF_TOKEN=",
            "Bearer ",
            "kaggle-cookies.json",
            "kaggle-web-storage-state.json",
            "operator.private.env",
            "miner.private.env",
            "kernel.py",
            '"prompt":',
            '"generated_text":',
            '"generated_token_ids":',
            '"activation":',
            '"activations":',
            '"hidden_state":',
            '"logits":',
            '"kv_cache":',
            '"past_key_values":',
            '"lease_token":',
            '"idempotency_key":',
        ]:
            self.assertNotIn(fragment, scanned)

    def test_check_script_builds_and_validates_report(self) -> None:
        output_dir = self._tmp_dir() / "check"
        result = check.build_check(check.parse_args([
            "--output-dir",
            str(output_dir),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["gpu_tpu_cpu_32b_heterogeneous_rc_ready"])
        self.assertFalse(result["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertEqual(result["blocked_reason"], "same_request_live_proof_missing")

    def test_tpu_allocation_attempt_is_carried_as_blocker_evidence(self) -> None:
        base = self._tmp_dir()
        tpu_attempt = {
            "schema": "kaggle_tpu_llm_probe_v1",
            "ok": False,
            "selected_accelerator": "tpuV5e8",
            "tpu_runtime_ready": False,
            "blocked_reason": "kaggle_tpu_kernel_queued_timeout",
            "diagnosis_codes": [
                "kaggle_tpu_accelerator_accepted",
                "kaggle_tpu_kernel_deleted",
                "kaggle_tpu_kernel_queued_timeout",
            ],
            "blockers": ["kaggle_tpu_kernel_queued_timeout", "kaggle_tpu_report_missing"],
            "kaggle_lifecycle": {
                "kernels_deleted": True,
                "private_packages_removed": True,
            },
        }
        tpu_path = base / "tpu-attempt.json"
        tpu_path.write_text(json.dumps(tpu_attempt, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-allocation-attempt-report",
            str(tpu_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["tpu_runtime_allocation_attempted"])
        self.assertTrue(report["tpu_runtime_allocation_blocked"])
        self.assertEqual(report["blocked_reason"], "kaggle_tpu_kernel_queued_timeout")
        self.assertIn("kaggle_tpu_kernel_queued_timeout", report["blocker_report"]["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_tpu_web_active_event_is_carried_as_blocker_evidence(self) -> None:
        base = self._tmp_dir()
        web_event = {
            "schema": "kaggle_tpu_web_active_event_extended_status_v1",
            "ok": True,
            "notebook_url_public": "https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit",
            "logged_in": True,
            "queue_seen": True,
            "queue_positions_public": ["#17"],
            "running": False,
            "bounded_wait_seconds": 1200,
            "poll_count": 60,
            "public_artifact_safe": True,
            "diagnosis_codes": ["kaggle_tpu_web_runtime_still_queued"],
            "blockers": ["kaggle_web_tpu_runtime_not_allocated_within_extended_bounded_wait"],
        }
        web_path = base / "web-event.json"
        web_path.write_text(json.dumps(web_event, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-web-active-event-report",
            str(web_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["tpu_runtime_allocation_attempted"])
        self.assertTrue(report["tpu_runtime_allocation_blocked"])
        self.assertFalse(report["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertFalse(report["live_tpu_stage_miner_integrated"])
        self.assertEqual(
            report["blocked_reason"],
            "kaggle_web_tpu_runtime_not_allocated_within_extended_bounded_wait",
        )
        self.assertIn(
            "kaggle_web_tpu_runtime_not_allocated_within_extended_bounded_wait",
            report["blocker_report"]["blockers"],
        )
        self.assertEqual(check.validate_report(report), [])

    def test_tpu_web_detached_status_is_valid_blocker_evidence(self) -> None:
        base = self._tmp_dir()
        web_event = {
            "schema": "kaggle_tpu_web_active_event_status_v1",
            "ok": True,
            "notebook_url_public": "https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit",
            "web_active_event_attempted": True,
            "running": False,
            "tpu_runtime_ready": False,
            "jupyter_api_status_ready": False,
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
            "cleanup": {"cookie_file_public": False, "storage_state_file_public": False},
            "diagnosis_codes": ["kaggle_tpu_web_runtime_not_currently_attached"],
            "blockers": [
                "kaggle_web_tpu_jupyter_proxy_not_visible",
                "kaggle_web_tpu_runtime_not_currently_ready",
            ],
        }
        web_path = base / "web-detached.json"
        web_path.write_text(json.dumps(web_event, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-web-active-event-report",
            str(web_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["tpu_runtime_allocation_attempted"])
        self.assertTrue(report["tpu_runtime_allocation_blocked"])
        self.assertFalse(report["tpu_runtime_allocation_ready"])
        self.assertEqual(report["blocked_reason"], "kaggle_web_tpu_jupyter_proxy_not_visible")
        self.assertIn("kaggle_web_tpu_jupyter_proxy_not_visible", report["blocker_report"]["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_tpu_web_starting_status_is_valid_blocker_evidence(self) -> None:
        base = self._tmp_dir()
        web_event = {
            "schema": "kaggle_tpu_web_active_event_status_v1",
            "ok": True,
            "notebook_url_public": "https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit",
            "web_active_event_attempted": True,
            "running": False,
            "tpu_runtime_ready": False,
            "jupyter_api_status_ready": False,
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
            "cleanup": {"cookie_file_public": False, "storage_state_file_public": False},
            "queue_seen": False,
            "diagnosis_codes": ["kaggle_tpu_web_session_still_starting"],
            "blockers": ["kaggle_web_tpu_session_still_starting"],
        }
        web_path = base / "web-starting.json"
        web_path.write_text(json.dumps(web_event, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-web-active-event-report",
            str(web_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["tpu_runtime_allocation_blocked"])
        self.assertFalse(report["tpu_runtime_allocation_ready"])
        self.assertEqual(report["blocked_reason"], "kaggle_web_tpu_session_still_starting")
        self.assertEqual(check.validate_report(report), [])

    def test_runtime_bridge_report_is_carried_without_32b_overclaim(self) -> None:
        base = self._tmp_dir()
        bridge = {
            "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
            "ok": True,
            "same_request_runtime_bridge_verified": True,
            "same_request_32b_model_verified": False,
            "gpu_tpu_cpu_32b_same_request_verified": False,
            "not_32b_weight_success": True,
            "generated_token_count": 1,
            "accepted_stage_backends": ["cpu", "cuda", "jax_tpu"],
            "activation_handoff_count": 2,
            "runtime_device_summary": {
                "cuda_stage_ready": True,
                "jax_tpu_stage_ready": True,
                "cpu_tail_ready": True,
                "tpu_device_count": 8,
                "cuda_device_count": 1,
            },
            "blocked_reason": "",
            "blockers": [],
            "safety": {"public_artifact_safe": True},
        }
        bridge_path = base / "bridge.json"
        bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--runtime-bridge-report",
            str(bridge_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertFalse(report["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertIn("runtime_bridge_only_32b_weight_success_missing", report["blocker_report"]["blockers"])
        self.assertTrue(report["runtime_bridge_summary"]["same_request_runtime_bridge_verified"])
        self.assertEqual(check.validate_report(report), [])

    def test_runtime_bridge_blocker_report_is_carried(self) -> None:
        base = self._tmp_dir()
        bridge = {
            "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
            "ok": False,
            "same_request_runtime_bridge_verified": False,
            "same_request_32b_model_verified": False,
            "gpu_tpu_cpu_32b_same_request_verified": False,
            "not_32b_weight_success": True,
            "generated_token_count": 0,
            "accepted_stage_backends": [],
            "activation_handoff_count": 0,
            "runtime_device_summary": {},
            "blocked_reason": "same_request_runtime_bridge_not_verified",
            "blockers": ["cuda_stage_not_ready", "same_request_runtime_bridge_not_verified"],
            "safety": {"public_artifact_safe": True},
        }
        bridge_path = base / "bridge-blocked.json"
        bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--runtime-bridge-report",
            str(bridge_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertIn("cuda_stage_not_ready", report["blocker_report"]["blockers"])
        self.assertFalse(report["runtime_bridge_summary"]["same_request_runtime_bridge_verified"])
        self.assertEqual(check.validate_report(report), [])

    def test_tpu_stage_adapter_plan_is_carried_without_overclaiming_runtime(self) -> None:
        base = self._tmp_dir()
        adapter_report = adapter_plan.build_report(adapter_plan.parse_args([
            "--output-dir",
            str(base / "adapter"),
            "--mode",
            "fixture",
        ]))
        adapter_path = base / "adapter" / "gpu_tpu_qwen_stage_adapter_plan.json"

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-stage-adapter-plan-report",
            str(adapter_path),
        ]))

        self.assertTrue(adapter_report["ok"], adapter_report)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["tpu_stage_adapter_plan_ready"])
        self.assertTrue(report["tpu_checkpoint_bridge_plan_ready"])
        self.assertTrue(report["tpu_stage_owned_loader_plan_ready"])
        self.assertFalse(report["tpu_32b_runtime_adapter_ready"])
        self.assertIn("jax_tpu_runtime_execution_not_performed", report["blocker_report"]["blockers"])
        self.assertNotIn("safetensors_or_maxtext_checkpoint_bridge", report["blocker_report"]["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_tpu_stage_runtime_probe_reduces_runtime_blocker_without_overclaiming_32b(self) -> None:
        base = self._tmp_dir()
        (base / "runtime").mkdir(parents=True, exist_ok=True)
        runtime_report = tpu_runtime_probe.build_report(
            tpu_runtime_probe.parse_args([
                "--output-dir",
                str(base / "runtime"),
                "--stage-profile",
                "qwen32b-one-layer",
            ]),
            output_dir=base / "runtime",
            accelerator_attempts=[
                {
                    "accelerator": "tpuV5e8",
                    "kernel_ref": "tester/cttpu-qwen-stage",
                    "steps": [
                        {"name": "kaggle_kernel_push", "ok": True, "accepted": True, "stdout_tail": "Kernel version 1 successfully pushed"},
                        {"name": "kaggle_kernel_delete", "ok": True},
                    ],
                }
            ],
            selected_report={
                "schema": tpu_runtime_probe.STAGE_SCHEMA,
                "requested_accelerator": "tpuV5e8",
                "ok": True,
                "tpu_runtime_ready": True,
                "qwen_like_stage_runtime_ready": True,
                "qwen32b_single_layer_runtime_ready": True,
                "stage_local_kv_cache_verified": True,
                "diagnosis_codes": ["kaggle_tpu_qwen_stage_runtime_ready"],
                "jax_tpu_stage": {
                    "shape_metadata": {
                        "input_shape": [1, 1, 5120],
                        "output_shape": [1, 1, 5120],
                        "dtype": "bfloat16",
                        "layout": "batch_seq_hidden",
                    },
                    "stage_input_hash": "sha256:input",
                    "stage_output_hash": "sha256:output",
                },
            },
        )
        runtime_path = base / "runtime" / "kaggle_tpu_qwen_stage_runtime_probe.json"
        runtime_path.write_text(json.dumps(runtime_report, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-stage-runtime-probe-report",
            str(runtime_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["tpu_qwen_like_stage_runtime_probe_ready"])
        self.assertTrue(report["tpu_qwen32b_single_layer_runtime_probe_ready"])
        self.assertFalse(report["tpu_32b_runtime_adapter_ready"])
        self.assertFalse(report["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertNotIn("jax_tpu_llama_like_stage_runtime", report["blocker_report"]["blockers"])
        self.assertIn("full_32b_tpu_stage_owned_runtime_not_verified", report["blocker_report"]["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_tpu_stage_loader_probe_partial_evidence_is_carried_without_overclaiming_32b(self) -> None:
        base = self._tmp_dir()
        loader_report = {
            "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
            "ok": True,
            "model_repo": "Qwen/Qwen2.5-32B-Instruct",
            "stage_layer_range": [21, 42],
            "stage_owned_header_verified": True,
            "partial_tensor_to_tpu_verified": True,
            "full_stage_owned_tpu_loader_ready": False,
            "tpu_32b_runtime_adapter_ready": False,
            "assigned_weight_key_count": 252,
            "assigned_weight_file_count": 6,
            "present_stage_key_count": 252,
            "missing_stage_key_count": 0,
            "selected_tensor_key_hash": "sha256:key",
            "selected_tensor_value_hash": "sha256:value",
            "selected_tensor_tpu_summary_hash": "sha256:summary",
            "selected_tensor_shape": [5120],
            "selected_tensor_dtype": "BF16",
            "selected_tensor_bytes": 10240,
            "tpu_device_count": 8,
            "blockers": ["full_stage_owned_tpu_loader_not_executed"],
            "diagnosis_codes": ["kaggle_web_tpu_32b_partial_tensor_to_tpu_verified"],
            "kaggle_lifecycle": {
                "web_runtime_execution_count": 1,
                "private_kernel_push_count": 0,
                "kernels_deleted": True,
                "private_packages_removed": True,
            },
            "safety": {"public_artifact_safe": True},
            "public_artifact_safe": True,
        }
        loader_path = base / "loader.json"
        loader_path.write_text(json.dumps(loader_report, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-stage-loader-probe-report",
            str(loader_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertFalse(report["tpu_32b_runtime_adapter_ready"])
        loader_summary = report["tpu_stage_loader_probe_summary"]
        self.assertTrue(loader_summary["stage_owned_header_verified"])
        self.assertTrue(loader_summary["partial_tensor_to_tpu_verified"])
        self.assertFalse(loader_summary["full_stage_owned_tpu_loader_ready"])
        self.assertIn("full_stage_owned_tpu_loader_not_executed", report["blocker_report"]["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_tpu_stage_loader_full_evidence_marks_adapter_ready_without_same_request_success(self) -> None:
        base = self._tmp_dir()
        loader_report = {
            "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
            "ok": True,
            "model_repo": "Qwen/Qwen2.5-32B-Instruct",
            "stage_layer_range": [21, 42],
            "stage_owned_header_verified": True,
            "partial_tensor_to_tpu_verified": True,
            "full_stage_owned_tpu_loader_ready": True,
            "tpu_32b_runtime_adapter_ready": True,
            "assigned_weight_key_count": 252,
            "assigned_weight_file_count": 6,
            "present_stage_key_count": 252,
            "missing_stage_key_count": 0,
            "selected_tensor_key_hash": "sha256:key",
            "selected_tensor_value_hash": "sha256:value",
            "selected_tensor_tpu_summary_hash": "sha256:summary",
            "selected_tensor_shape": [5120],
            "selected_tensor_dtype": "BF16",
            "selected_tensor_bytes": 10240,
            "tpu_device_count": 8,
            "blockers": [],
            "diagnosis_codes": ["kaggle_web_tpu_32b_full_stage_loader_ready"],
            "kaggle_lifecycle": {
                "web_runtime_execution_count": 1,
                "private_kernel_push_count": 0,
                "kernels_deleted": True,
                "private_packages_removed": True,
            },
            "safety": {"public_artifact_safe": True},
            "public_artifact_safe": True,
        }
        loader_path = base / "loader-full.json"
        loader_path.write_text(json.dumps(loader_report, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-stage-loader-probe-report",
            str(loader_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["tpu_32b_runtime_adapter_ready"])
        self.assertFalse(report["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertFalse(report["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertEqual(report["blocked_reason"], "same_request_live_proof_missing")
        self.assertEqual(check.validate_report(report), [])

    def test_checker_rejects_uncleaned_tpu_stage_runtime_probe(self) -> None:
        base = self._tmp_dir()
        runtime_report = {
            "schema": tpu_runtime_probe.SCHEMA,
            "ok": False,
            "selected_accelerator": "tpuV5e8",
            "stage_profile": "qwen32b-one-layer",
            "tpu_runtime_ready": False,
            "qwen_like_stage_runtime_ready": False,
            "qwen32b_single_layer_runtime_ready": False,
            "blocked_reason": "kaggle_tpu_kernel_queued_timeout",
            "blockers": ["kaggle_tpu_kernel_queued_timeout"],
            "kaggle_lifecycle": {
                "kernels_deleted": False,
                "private_packages_removed": False,
            },
            "safety": {"public_artifact_safe": True},
        }
        runtime_path = base / "runtime.json"
        runtime_path.write_text(json.dumps(runtime_report, indent=2, sort_keys=True), encoding="utf-8")
        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-stage-runtime-probe-report",
            str(runtime_path),
        ]))

        errors = check.validate_report(report)

        self.assertIn("tpu_stage_runtime_probe_kernel_cleanup_missing", errors)
        self.assertIn("tpu_stage_runtime_probe_private_cleanup_missing", errors)

    def test_checker_rejects_uncleaned_tpu_allocation_attempt(self) -> None:
        base = self._tmp_dir()
        tpu_attempt = {
            "schema": "kaggle_tpu_llm_probe_v1",
            "ok": False,
            "selected_accelerator": "tpuV5e8",
            "tpu_runtime_ready": False,
            "blocked_reason": "kaggle_tpu_kernel_queued_timeout",
            "diagnosis_codes": ["kaggle_tpu_kernel_queued_timeout"],
            "blockers": ["kaggle_tpu_kernel_queued_timeout"],
            "kaggle_lifecycle": {
                "kernels_deleted": False,
                "private_packages_removed": False,
            },
        }
        tpu_path = base / "tpu-attempt.json"
        tpu_path.write_text(json.dumps(tpu_attempt, indent=2, sort_keys=True), encoding="utf-8")
        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "none",
            "--tpu-allocation-attempt-report",
            str(tpu_path),
        ]))

        errors = check.validate_report(report)

        self.assertIn("tpu_allocation_attempt_kernel_cleanup_missing", errors)
        self.assertIn("tpu_allocation_attempt_private_cleanup_missing", errors)

    def test_external_live_success_import_validates(self) -> None:
        base = self._tmp_dir()
        live_report = pack.fixture_live_success_report(pack.parse_args(self._base_args(base / "ignored")))
        live_path = base / "live.json"
        live_path.write_text(json.dumps(live_report, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "external"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "external",
            "--live-same-request-report",
            str(live_path),
        ]))

        self.assertTrue(report["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertEqual(check.validate_report(report), [])

    def test_external_live_success_allows_matching_runtime_bridge_32b_fields(self) -> None:
        base = self._tmp_dir()
        live_report = pack.fixture_live_success_report(pack.parse_args(self._base_args(base / "ignored")))
        live_path = base / "live.json"
        live_path.write_text(json.dumps(live_report, indent=2, sort_keys=True), encoding="utf-8")
        bridge = {
            "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
            "ok": True,
            "same_request_runtime_bridge_verified": True,
            "same_request_32b_model_verified": True,
            "gpu_tpu_cpu_32b_same_request_verified": True,
            "not_32b_weight_success": False,
            "generated_token_count": 1,
            "accepted_stage_backends": ["cpu", "cuda", "jax_tpu"],
            "activation_handoff_count": 2,
            "runtime_device_summary": {
                "cuda_stage_ready": True,
                "jax_tpu_stage_ready": True,
                "cpu_tail_ready": True,
                "tpu_device_count": 8,
                "cuda_device_count": 1,
            },
            "blocked_reason": "",
            "blockers": [],
            "safety": {"public_artifact_safe": True},
        }
        bridge_path = base / "bridge.json"
        bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "external-bridge"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "external",
            "--live-same-request-report",
            str(live_path),
            "--runtime-bridge-report",
            str(bridge_path),
        ]))

        self.assertTrue(report["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertTrue(report["runtime_bridge_summary"]["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertNotIn("runtime_bridge_overclaims_32b_same_request", check.validate_report(report))
        self.assertEqual(check.validate_report(report), [])

    def test_external_live_success_filters_stale_adapter_plan_runtime_blocker(self) -> None:
        base = self._tmp_dir()
        live_report = pack.fixture_live_success_report(pack.parse_args(self._base_args(base / "ignored")))
        live_path = base / "live.json"
        live_path.write_text(json.dumps(live_report, indent=2, sort_keys=True), encoding="utf-8")
        adapter_report = adapter_plan.build_report(adapter_plan.parse_args([
            "--output-dir",
            str(base / "adapter"),
            "--mode",
            "fixture",
        ]))
        loader_report = {
            "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
            "ok": True,
            "model_repo": "Qwen/Qwen2.5-32B-Instruct",
            "stage_layer_range": [21, 42],
            "stage_owned_header_verified": True,
            "partial_tensor_to_tpu_verified": True,
            "full_stage_owned_tpu_loader_ready": True,
            "tpu_32b_runtime_adapter_ready": True,
            "assigned_weight_key_count": 252,
            "assigned_weight_file_count": 6,
            "present_stage_key_count": 252,
            "missing_stage_key_count": 0,
            "selected_tensor_key_hash": "sha256:key",
            "selected_tensor_value_hash": "sha256:value",
            "selected_tensor_tpu_summary_hash": "sha256:summary",
            "selected_tensor_shape": [5120],
            "selected_tensor_dtype": "BF16",
            "selected_tensor_bytes": 10240,
            "tpu_device_count": 8,
            "blockers": [],
            "diagnosis_codes": ["kaggle_web_tpu_32b_full_stage_loader_ready"],
            "kaggle_lifecycle": {
                "web_runtime_execution_count": 1,
                "private_kernel_push_count": 0,
                "kernels_deleted": True,
                "private_packages_removed": True,
            },
            "safety": {"public_artifact_safe": True},
            "public_artifact_safe": True,
        }
        loader_path = base / "loader-full.json"
        loader_path.write_text(json.dumps(loader_report, indent=2, sort_keys=True), encoding="utf-8")
        adapter_path = base / "adapter" / "gpu_tpu_qwen_stage_adapter_plan.json"
        self.assertIn("jax_tpu_runtime_execution_not_performed", adapter_report["blockers"])

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "external-clean"),
            "--execution-mode",
            "fixture",
            "--live-proof-mode",
            "external",
            "--live-same-request-report",
            str(live_path),
            "--tpu-stage-adapter-plan-report",
            str(adapter_path),
            "--tpu-stage-loader-probe-report",
            str(loader_path),
        ]))

        self.assertTrue(report["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertNotIn("jax_tpu_runtime_execution_not_performed", report["blocker_report"]["blockers"])
        self.assertNotIn("jax_tpu_runtime_execution_not_performed", report["stage_runtime_matrix"]["jax_tpu_stage"]["missing_items"])
        self.assertEqual(check.validate_report(report), [])

    def test_cli_wraps_heterogeneous_32b_rc(self) -> None:
        output_dir = self._tmp_dir() / "cli"
        summary = cli.build_gpu_tpu_cpu_32b_heterogeneous_rc(
            cli.parse_args([
                "heterogeneous-32b-rc",
                "--output-dir",
                str(output_dir),
                "--execution-mode",
                "fixture",
                "--live-proof-mode",
                "none",
            ])
        )

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "gpu_tpu_cpu_32b_heterogeneous_rc_cli_v1")
        self.assertTrue(summary["gpu_tpu_cpu_32b_heterogeneous_rc_ready"])
        self.assertFalse(summary["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertFalse(summary["gpu_tpu_cpu_32b_same_request_verified"])
        payload = json.loads((output_dir / "gpu_tpu_cpu_32b_heterogeneous_rc_cli_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["cli_schema"], "gpu_tpu_cpu_32b_heterogeneous_rc_cli_v1")

    def test_cli_wraps_tpu_stage_adapter_plan_report(self) -> None:
        base = self._tmp_dir()
        adapter_plan.build_report(adapter_plan.parse_args([
            "--output-dir",
            str(base / "adapter"),
            "--mode",
            "fixture",
        ]))
        output_dir = base / "cli-adapter"
        summary = cli.build_gpu_tpu_cpu_32b_heterogeneous_rc(
            cli.parse_args([
                "heterogeneous-32b-rc",
                "--output-dir",
                str(output_dir),
                "--execution-mode",
                "fixture",
                "--live-proof-mode",
                "none",
                "--tpu-stage-adapter-plan-report",
                str(base / "adapter" / "gpu_tpu_qwen_stage_adapter_plan.json"),
            ])
        )

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(summary["tpu_stage_adapter_plan_ready"])
        self.assertTrue((output_dir / "tpu_stage_adapter_plan_summary.json").is_file())

    def test_cli_wraps_tpu_web_active_event_report(self) -> None:
        base = self._tmp_dir()
        web_event = {
            "schema": "kaggle_tpu_web_active_event_status_v1",
            "ok": True,
            "notebook_url_public": "https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit",
            "logged_in": True,
            "queue_seen": True,
            "queue_positions_public": [],
            "running": False,
            "bounded_wait_seconds": 63,
            "public_artifact_safe": True,
            "diagnosis_codes": ["kaggle_tpu_web_runtime_still_queued"],
            "blockers": ["kaggle_web_tpu_runtime_not_allocated_within_short_status_check"],
        }
        web_path = base / "web-event.json"
        web_path.write_text(json.dumps(web_event, indent=2, sort_keys=True), encoding="utf-8")
        output_dir = base / "cli-web"

        summary = cli.build_gpu_tpu_cpu_32b_heterogeneous_rc(
            cli.parse_args([
                "heterogeneous-32b-rc",
                "--output-dir",
                str(output_dir),
                "--execution-mode",
                "fixture",
                "--live-proof-mode",
                "none",
                "--tpu-web-active-event-report",
                str(web_path),
            ])
        )

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(summary["tpu_runtime_allocation_attempted"])
        self.assertTrue(summary["tpu_runtime_allocation_blocked"])
        self.assertTrue((output_dir / "tpu_web_active_event_summary.json").is_file())

    def test_cli_fixture_success_can_model_future_live_success(self) -> None:
        output_dir = self._tmp_dir() / "cli-success"
        summary = cli.build_gpu_tpu_cpu_32b_heterogeneous_rc(
            cli.parse_args([
                "heterogeneous-32b-rc",
                "--output-dir",
                str(output_dir),
                "--execution-mode",
                "fixture",
                "--live-proof-mode",
                "fixture-success",
            ])
        )

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(summary["gpu_tpu_cpu_32b_bounded_rc_success"])
        self.assertTrue(summary["gpu_tpu_cpu_32b_same_request_verified"])
        self.assertFalse(summary["fallback_model_used"])


if __name__ == "__main__":
    unittest.main()
