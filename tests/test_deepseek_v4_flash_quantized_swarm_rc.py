from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import deepseek_v4_flash_quantized_swarm_rc_check as check
from scripts import deepseek_v4_flash_quantized_swarm_rc_pack as pack


class DeepSeekV4FlashQuantizedSwarmRCTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="ct_dsv4_swarm_rc_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _source_report(self) -> dict:
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
            "ready_candidate_count": 3,
            "recommended_live_probe_candidate": {
                "candidate_id": "iq1-s-xl-gguf",
                "repo": "teamblobfish/DeepSeek-V4-Flash-GGUF",
                "quant": "IQ1_S-XL",
                "runtime_backend": "llama_cpp_v4_fork",
                "runtime_fork": "cchuter/llama.cpp@feat/v4-port-cuda",
                "total_size_gb": 61.540805,
                "split_file_count": 2,
                "files": [
                    {"path": "IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00001-of-00002.gguf", "size_gb": 49.952658},
                    {"path": "IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00002-of-00002.gguf", "size_gb": 11.588147},
                ],
                "blockers": [
                    "stock_llama_cpp_cannot_load_deepseek_v4_flash",
                    "deepseek_v4_flash_llama_cpp_runtime_wip",
                    "t4_cuda_runtime_not_validated_for_deepseek_v4_flash",
                    "candidate_exceeds_single_t4x2_memory_budget",
                ],
            },
            "blockers": [
                "stock_llama_cpp_cannot_load_deepseek_v4_flash",
                "deepseek_v4_flash_llama_cpp_runtime_wip",
                "t4_cuda_runtime_not_validated_for_deepseek_v4_flash",
                "candidate_exceeds_single_t4x2_memory_budget",
            ],
            "public_artifact_safe": True,
            "safety": {"public_artifact_safe": True},
        }

    def _same_request_report(self, *, providers: list[str] | None = None, generated: int = 1) -> dict:
        providers = providers or ["kaggle_cuda", "colab_cuda", "cpu"]
        return {
            "schema": "deepseek_v4_flash_quantized_same_request_probe_v1",
            "ok": True,
            "deepseek_v4_flash_quantized_same_request_verified": True,
            "same_request_decode_verified": True,
            "generated_token_count": generated,
            "accepted_providers": providers,
            "provider_stage_counts": {provider: 1 for provider in providers},
            "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
            "blockers": [],
            "diagnosis_codes": ["deepseek_v4_flash_quantized_same_request_decode_verified"],
            "public_artifact_safe": True,
            "safety": {"public_artifact_safe": True},
        }

    def _same_request_preflight_report(self) -> dict:
        return {
            "schema": "deepseek_v4_flash_quantized_same_request_probe_v1",
            "ok": False,
            "deepseek_v4_flash_quantized_same_request_verified": False,
            "same_request_decode_verified": False,
            "generated_token_count": 0,
            "accepted_providers": [],
            "provider_stage_counts": {"kaggle_cuda": 0, "colab_cuda": 0, "cpu": 0},
            "stage_task_counts": {"kaggle_cuda_rpc": 0, "colab_cuda_rpc": 0, "cpu_rpc": 0},
            "mode": "preflight",
            "live_run_performed": False,
            "failure_stage": "same_request_live_probe_not_started",
            "blockers": [
                "deepseek_v4_flash_quantized_same_request_decode_not_verified",
                "deepseek_v4_flash_quantized_same_request_live_run_not_started",
            ],
            "diagnosis_codes": ["deepseek_v4_flash_quantized_same_request_probe_live_run_not_started"],
            "public_artifact_safe": True,
            "safety": {"public_artifact_safe": True},
        }

    def _same_request_live_colab_failure_report(self) -> dict:
        report = self._same_request_preflight_report()
        report["mode"] = "kaggle-auto"
        report["live_run_performed"] = True
        report["failure_stage"] = "colab_rpc_worker_background_poll_lost"
        report["blockers"] = [
            "deepseek_v4_flash_quantized_same_request_decode_not_verified",
            "colab_rpc_worker_background_poll_lost",
        ]
        report["diagnosis_codes"] = ["deepseek_v4_flash_quantized_same_request_probe_live_run_performed"]
        return report

    def _same_request_live_colab_authuser_fallback_failure_report(self) -> dict:
        report = self._same_request_live_colab_failure_report()
        report["blockers"].extend([
            "colab_rpc_worker_authuser_fallback_exhausted",
            "colab_rpc_worker_fallback_exhausted",
        ])
        report["colab_rpc"] = {
            "colab_fallback": {
                "attempt_count": 2,
                "attempted_targets": [
                    {"accelerator": "T4", "authuser": "0"},
                    {"accelerator": "L4", "authuser": "0"},
                ],
                "selected_accelerator": "",
                "selected_authuser": "",
                "attempts": [
                    {
                        "accelerator": "T4",
                        "authuser": "0",
                        "ok": False,
                        "blockers": ["colab_rpc_worker_background_launch_failed"],
                        "manager": {"blocker": "colab_cuda_reacquire_failed", "attempt_count": 1},
                    },
                    {
                        "accelerator": "L4",
                        "authuser": "0",
                        "ok": False,
                        "blockers": ["colab_rpc_worker_background_launch_failed"],
                        "manager": {"blocker": "colab_cuda_reacquire_failed", "attempt_count": 1},
                    },
                ],
                "public_artifact_safe": True,
            },
            "authuser_fallback": {
                "attempt_count": 2,
                "attempted_authusers": ["0", "1"],
                "selected_authuser": "",
                "attempts": [
                    {
                        "authuser": "0",
                        "ok": False,
                        "blockers": ["colab_rpc_worker_background_launch_failed"],
                        "manager": {"blocker": "colab_cuda_reacquire_failed", "attempt_count": 1},
                    },
                    {
                        "authuser": "1",
                        "ok": False,
                        "blockers": ["colab_rpc_worker_background_launch_failed"],
                        "manager": {"blocker": "colab_cuda_reacquire_failed", "attempt_count": 1},
                    },
                ],
                "public_artifact_safe": True,
            }
        }
        return report

    def _kaggle_quota_preflight(self) -> dict:
        return {
            "schema": "kaggle_gpu_concurrency_probe_v1",
            "ok": False,
            "evidence_ready": True,
            "owner": "private-owner",
            "accelerator": "NvidiaTeslaT4",
            "requested_kernel_count": 2,
            "accepted_submission_count": 0,
            "simultaneous_t4x2_verified": False,
            "max_observed_running_count": 0,
            "cleanup": {"attempted": False, "deleted_refs": [], "failed_delete_refs": []},
            "private_kernel_payloads_removed": True,
            "blockers": ["kaggle_gpu_kernel_push_rejected", "kaggle_gpu_quota_or_session_limit"],
            "diagnosis_codes": ["no_gpu_kernels_accepted"],
            "public_artifact_safe": True,
        }

    def _colab_cuda_preflight(self) -> dict:
        return {
            "schema": "colab_cuda_runtime_probe_v1",
            "ok": True,
            "colab_cuda_runtime_ready": True,
            "runtime_proxy_connected": True,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_matmul_ready": True,
            "torch_version": "2.11.0+cu128",
            "cuda_version": "12.8",
            "devices": [{"index": 0, "name_hash": "hash", "name_public": False, "total_memory_mb": 14912}],
            "public_artifact_safe": True,
            "blockers": [],
        }

    def _llama_v4_build_failure(self) -> dict:
        return {
            "schema": "deepseek_v4_flash_colab_llama_v4_rpc_build_preflight_wrapper_v1",
            "ok": False,
            "public_artifact_safe": True,
            "blockers": ["colab_llama_v4_build_marker_missing"],
            "manager": {
                "ok": False,
                "blocker": "colab_cuda_execute_failed",
                "attempts": [{"attempt": 1, "ok": False, "stale_detected": True}],
            },
            "output_type_count": 0,
        }

    def _kaggle_llama_v4_build_success(self) -> dict:
        return {
            "schema": "deepseek_v4_flash_kaggle_llama_v4_build_preflight_v1",
            "ok": True,
            "llama_v4_runtime_build_ready": True,
            "fresh_kaggle_run_performed": True,
            "runtime": {
                "runtime_backend": "llama_cpp_v4_fork",
                "repo_url": "https://github.com/cchuter/llama.cpp.git",
                "branch": "feat/v4-port-cuda",
                "cuda_architectures": "75",
                "patch_rpc_op_count_guard": True,
            },
            "worker_summary": {
                "worker_ok": True,
                "repo_url": "https://github.com/cchuter/llama.cpp.git",
                "branch": "feat/v4-port-cuda",
                "commit_hash_public": "781e978f3ee68144cb5922be9a5627610d091317",
                "cuda_architectures": "75",
                "patch_rpc_op_count_guard": True,
                "patch_rpc_op_count_guard_ok": True,
                "llama_cli_present": True,
                "rpc_server_present": True,
                "llama_cli_supports_rpc": True,
                "llama_cli_supports_tensor_split": True,
                "cmake_configure_ok": True,
                "cmake_build_ok": True,
                "blockers": [],
            },
            "kaggle_lifecycle": {
                "kernel_deleted": True,
                "private_package_removed": True,
            },
            "blockers": [],
            "diagnosis_codes": ["kaggle_llama_v4_runtime_build_ready"],
            "public_artifact_safe": True,
        }

    def _rpc_hello_diagnostic_success(self) -> dict:
        return {
            "schema": "deepseek_v4_flash_rpc_hello_diagnostic_probe_v1",
            "ok": True,
            "rpc_hello_diagnostic_ready": True,
            "fresh_kaggle_run_performed": True,
            "worker_summary": {
                "rpc_hello_diagnostic_ready": True,
                "server_count": 2,
                "server_names": ["kaggle-cuda0-rpc", "kaggle-cuda1-rpc"],
                "all_servers_alive": True,
                "all_rpc_hello_ok": True,
                "blockers": [],
                "diagnosis_codes": ["deepseek_v4_flash_rpc_hello_diagnostic_ready"],
            },
            "kaggle_lifecycle": {
                "requested_accelerator": "NvidiaTeslaT4",
                "kernel_deleted": True,
                "private_package_removed": True,
            },
            "blockers": [],
            "diagnosis_codes": ["deepseek_v4_flash_rpc_hello_diagnostic_ready"],
            "public_artifact_safe": True,
        }

    def _colab_cuda_reacquire_retry_failure(self) -> dict:
        return {
            "schema": "colab_cuda_reacquire_retry_probe_v1",
            "ok": False,
            "colab_cuda_reacquire_ready": False,
            "attempts_completed": 6,
            "attempts_requested": 6,
            "successful_attempt_index": 0,
            "accelerators_attempted": ["T4"],
            "authusers_attempted": ["0", "1"],
            "accelerator": "",
            "authuser": "",
            "successful_report_path": "",
            "blockers": [
                "colab_cuda_session_not_allocated",
                "colab_gpu_assignment_error_publichttperror",
                "colab_gpu_assignment_http_503",
            ],
            "public_artifact_safe": True,
            "credentials_public": False,
            "private_runtime_state_public": False,
        }

    def _colab_retry_same_request_auto_failure(self) -> dict:
        return {
            "schema": "deepseek_v4_flash_colab_retry_same_request_auto_v1",
            "ok": False,
            "deepseek_v4_flash_colab_retry_same_request_ready": False,
            "retry_ready": False,
            "same_request_started": False,
            "same_request_decode_verified": False,
            "generated_token_count": 0,
            "accepted_providers": [],
            "failure_stage": "colab_cuda_reacquire_not_ready",
            "retry_summary": {
                "schema": "colab_cuda_reacquire_retry_probe_v1",
                "ok": False,
                "colab_cuda_reacquire_ready": False,
                "attempts_completed": 6,
                "accelerator": "",
                "authuser": "",
                "blockers": ["colab_gpu_assignment_http_503"],
                "public_artifact_safe": True,
            },
            "same_request_summary": {
                "schema": "",
                "ok": False,
                "same_request_decode_verified": False,
                "generated_token_count": 0,
                "failure_stage": "",
                "blockers": [],
                "public_artifact_safe": False,
            },
            "blockers": [
                "colab_cuda_reacquire_not_ready",
                "colab_cuda_session_not_allocated",
                "colab_gpu_assignment_error_publichttperror",
                "colab_gpu_assignment_http_503",
            ],
            "public_artifact_safe": True,
            "credentials_public": False,
            "private_runtime_state_public": False,
        }

    def test_blocker_rc_passes_without_overclaiming_same_request_success(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
            ])
        )

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertEqual(report["failure_stage"], "same_request_live_probe_not_started")
        self.assertIn("deepseek_v4_flash_quantized_same_request_decode_not_verified", report["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_same_request_decode_requires_all_three_worker_families(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        same = self._write_json(out / "same.json", self._same_request_report())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--same-request-report",
                str(same),
            ])
        )

        self.assertTrue(report["success"]["same_request_decode_verified"], report)
        self.assertEqual(report["failure_stage"], "")
        self.assertEqual(set(report["success"]["accepted_providers"]), {"kaggle_cuda", "colab_cuda", "cpu"})
        self.assertEqual(check.validate_report(report), [])

    def test_missing_provider_is_not_counted_as_success(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        same = self._write_json(out / "same.json", self._same_request_report(providers=["kaggle_cuda", "cpu"]))
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--same-request-report",
                str(same),
            ])
        )

        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertIn("deepseek_v4_flash_quantized_same_request_decode_not_verified", report["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_kaggle_quota_preflight_becomes_current_failure_stage(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        kaggle = self._write_json(out / "kaggle.json", self._kaggle_quota_preflight())
        colab = self._write_json(out / "colab.json", self._colab_cuda_preflight())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--kaggle-gpu-preflight-report",
                str(kaggle),
                "--colab-cuda-preflight-report",
                str(colab),
            ])
        )

        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertEqual(report["failure_stage"], "kaggle_cuda_quota_or_session_limit")
        self.assertFalse(report["kaggle_gpu_preflight"]["kaggle_cuda_ready"])
        self.assertTrue(report["colab_cuda_preflight"]["colab_cuda_ready"])
        self.assertIn("kaggle_cuda_preflight_not_ready", report["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_llama_v4_build_failure_becomes_runtime_failure_stage(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        kaggle = self._write_json(out / "kaggle.json", {**self._kaggle_quota_preflight(), "ok": True, "accepted_submission_count": 2, "simultaneous_t4x2_verified": True, "blockers": []})
        colab = self._write_json(out / "colab.json", self._colab_cuda_preflight())
        build = self._write_json(out / "build.json", self._llama_v4_build_failure())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--kaggle-gpu-preflight-report",
                str(kaggle),
                "--colab-cuda-preflight-report",
                str(colab),
                "--llama-v4-build-preflight-report",
                str(build),
            ])
        )

        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertEqual(report["failure_stage"], "llama_v4_runtime_build_colab_execute_failed")
        self.assertFalse(report["llama_v4_build_preflight"]["llama_v4_runtime_build_ready"])
        self.assertIn("deepseek_v4_flash_llama_v4_runtime_build_not_ready", report["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_kaggle_llama_v4_build_success_unblocks_runtime_stage_but_not_decode(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        kaggle = self._write_json(out / "kaggle.json", {**self._kaggle_quota_preflight(), "ok": True, "accepted_submission_count": 2, "simultaneous_t4x2_verified": True, "blockers": []})
        colab = self._write_json(out / "colab.json", self._colab_cuda_preflight())
        build = self._write_json(out / "build.json", self._kaggle_llama_v4_build_success())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--kaggle-gpu-preflight-report",
                str(kaggle),
                "--colab-cuda-preflight-report",
                str(colab),
                "--llama-v4-build-preflight-report",
                str(build),
            ])
        )

        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertEqual(report["failure_stage"], "same_request_live_probe_not_started")
        self.assertTrue(report["llama_v4_build_preflight"]["llama_v4_runtime_build_ready"])
        self.assertNotIn("deepseek_v4_flash_llama_v4_runtime_build_not_ready", report["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_rpc_hello_diagnostic_is_recorded_without_counting_as_decode_success(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        diagnostic = self._write_json(out / "rpc-hello.json", self._rpc_hello_diagnostic_success())
        colab_retry = self._write_json(out / "colab-retry.json", self._colab_cuda_reacquire_retry_failure())
        auto = self._write_json(out / "auto.json", self._colab_retry_same_request_auto_failure())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--rpc-hello-diagnostic-report",
                str(diagnostic),
                "--colab-accelerator-probe-report",
                str(out / "colab-l4.json"),
                "--colab-cuda-reacquire-retry-report",
                str(colab_retry),
                "--colab-retry-same-request-auto-report",
                str(auto),
            ])
        )

        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertTrue(report["rpc_hello_diagnostic"]["rpc_hello_diagnostic_ready"])
        self.assertEqual(report["rpc_hello_diagnostic"]["server_count"], 2)
        self.assertEqual(report["failure_stage"], "colab_cuda_reacquire_not_ready")
        self.assertFalse(report["colab_cuda_reacquire_retry"]["colab_cuda_reacquire_ready"])
        self.assertEqual(report["colab_cuda_reacquire_retry"]["attempts_completed"], 6)
        self.assertFalse(report["colab_retry_same_request_auto"]["retry_ready"])
        self.assertFalse(report["colab_retry_same_request_auto"]["same_request_started"])
        self.assertIn("colab_gpu_assignment_http_503", report["blockers"])
        self.assertIn("colab_cuda_reacquire_not_ready", report["blockers"])
        self.assertNotIn("deepseek_v4_flash_rpc_hello_diagnostic_not_ready", report["blockers"])
        self.assertEqual(
            json.loads((out / "rc" / "deepseek_v4_flash_quantized_swarm_rc_support.json").read_text(encoding="utf-8"))["rpc_hello_diagnostic_report"],
            str(diagnostic),
        )
        self.assertEqual(
            json.loads((out / "rc" / "deepseek_v4_flash_quantized_swarm_rc_support.json").read_text(encoding="utf-8"))["colab_accelerator_probe_reports"],
            [str(out / "colab-l4.json")],
        )
        self.assertEqual(
            json.loads((out / "rc" / "deepseek_v4_flash_quantized_swarm_rc_support.json").read_text(encoding="utf-8"))["colab_cuda_reacquire_retry_report"],
            str(out / "colab-retry.json"),
        )
        self.assertEqual(
            json.loads((out / "rc" / "deepseek_v4_flash_quantized_swarm_rc_support.json").read_text(encoding="utf-8"))["colab_retry_same_request_auto_report"],
            str(out / "auto.json"),
        )
        self.assertEqual(check.validate_report(report), [])

    def test_same_request_preflight_keeps_live_not_started_failure_stage(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        same = self._write_json(out / "same-preflight.json", self._same_request_preflight_report())
        kaggle = self._write_json(out / "kaggle.json", {**self._kaggle_quota_preflight(), "ok": True, "accepted_submission_count": 2, "simultaneous_t4x2_verified": True, "blockers": []})
        colab = self._write_json(out / "colab.json", self._colab_cuda_preflight())
        build = self._write_json(out / "build.json", self._kaggle_llama_v4_build_success())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--same-request-report",
                str(same),
                "--kaggle-gpu-preflight-report",
                str(kaggle),
                "--colab-cuda-preflight-report",
                str(colab),
                "--llama-v4-build-preflight-report",
                str(build),
            ])
        )

        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertEqual(report["failure_stage"], "same_request_live_probe_not_started")
        self.assertFalse(report["same_request"]["live_run_performed"])
        self.assertEqual(check.validate_report(report), [])

    def test_live_same_request_failure_stage_takes_precedence_over_colab_preflight(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        same = self._write_json(out / "same-live.json", self._same_request_live_colab_failure_report())
        kaggle = self._write_json(out / "kaggle.json", {**self._kaggle_quota_preflight(), "ok": True, "accepted_submission_count": 2, "simultaneous_t4x2_verified": True, "blockers": []})
        colab_payload = {**self._colab_cuda_preflight(), "ok": False, "colab_cuda_runtime_ready": False, "blockers": ["colab_cuda_runtime_not_ready"]}
        colab = self._write_json(out / "colab.json", colab_payload)
        build = self._write_json(out / "build.json", self._kaggle_llama_v4_build_success())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--same-request-report",
                str(same),
                "--kaggle-gpu-preflight-report",
                str(kaggle),
                "--colab-cuda-preflight-report",
                str(colab),
                "--llama-v4-build-preflight-report",
                str(build),
            ])
        )

        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertTrue(report["same_request"]["live_run_performed"])
        self.assertEqual(report["failure_stage"], "colab_rpc_worker_background_poll_lost")
        self.assertEqual(check.validate_report(report), [])

    def test_same_request_summary_preserves_colab_authuser_fallback_evidence(self) -> None:
        out = self._tmp_dir()
        source = self._write_json(out / "source.json", self._source_report())
        same = self._write_json(out / "same-live.json", self._same_request_live_colab_authuser_fallback_failure_report())
        report = pack.build_report(
            pack.parse_args([
                "--output-dir",
                str(out / "rc"),
                "--source-resolver-report",
                str(source),
                "--same-request-report",
                str(same),
            ])
        )

        fallback = report["same_request"]["colab_authuser_fallback"]
        matrix = report["same_request"]["colab_fallback"]
        self.assertFalse(report["success"]["same_request_decode_verified"])
        self.assertTrue(fallback["present"])
        self.assertEqual(fallback["attempted_authusers"], ["0", "1"])
        self.assertEqual([item["manager_blocker"] for item in fallback["attempts"]], ["colab_cuda_reacquire_failed", "colab_cuda_reacquire_failed"])
        self.assertTrue(matrix["present"])
        self.assertEqual(matrix["attempted_targets"], [{"accelerator": "T4", "authuser": "0"}, {"accelerator": "L4", "authuser": "0"}])
        self.assertEqual([item["accelerator"] for item in matrix["attempts"]], ["T4", "L4"])
        self.assertIn("colab_rpc_worker_fallback_exhausted", report["blockers"])
        self.assertIn("colab_rpc_worker_authuser_fallback_exhausted", report["blockers"])
        self.assertEqual(check.validate_report(report), [])


if __name__ == "__main__":
    unittest.main()
