from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from scripts import kaggle_swarm_32b_quantized_feasibility_check as check
from scripts import kaggle_swarm_32b_quantized_feasibility_pack as pack


class KaggleSwarm32BQuantizedFeasibilityTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_kaggle_32b_feasibility_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _fixture_reports(self, base: Path, *, upper_bound_crossing: bool = False) -> dict[str, Path]:
        production_like = {
            "schema": "gpu_swarm_production_like_validation_v1",
            "ok": True,
            "gpu_swarm_production_like_validation_ready": True,
            "production_like_workload_ready": True,
            "largest_successful_model_tier": "14b",
            "largest_attempted_model_tier": "32b",
            "multi_token_decode_ready": True,
            "batch_or_multi_request_ready": True,
            "stage_requeue_or_failure_recovery_ready": True,
            "fresh_gpu_run_performed": False,
            "external_runtime_verified": False,
            "retained_evidence_imported": True,
            "production_like_workload": {
                "generated_token_count": 16,
                "request_count": 2,
                "stage_requeue_or_failure_recovery_ready": True,
                "latency_throughput_summary_ready": True,
                "stage_owned_weight_loading_ready": True,
            },
            "safety": {"public_artifact_safe": True},
        }
        core_status = {
            "schema": "core_technology_validation_status_v1",
            "ok": True,
            "core_validation_ready": True,
            "largest_successful_tier": "14b",
            "handoff_stage_selective_evidence": {
                "seven_b_multi_token_verified": True,
                "seven_b_generated_token_count": 2,
                "fourteen_b_dual_kaggle_verified": True,
                "fourteen_b_generated_token_count": 1,
                "n_stage_partition_plan_ready": True,
                "stage_selective_performance_report_ready": True,
            },
            "seven_b_eight_b_evidence": {
                "real_7b_runtime_verified": True,
                "generated_token_count": 2,
                "memory_peak_mb": 14608,
            },
        }
        large_kaggle = {
            "schema": "large_model_kaggle_validation_v1",
            "ok": True,
            "real_7b_runtime_verified": True,
            "core_validation_ready": True,
            "gpu_runtime_verified": True,
            "largest_successful_tier": "7b",
            "hardware": {
                "gpu_count": 2,
                "gpu_names": ["NVIDIA Tesla T4", "NVIDIA Tesla T4"],
                "gpu_memory_total_mb": [15360, 15360],
            },
            "diagnosis_codes": [
                "large_model_kaggle_gpu_hardware_verified",
                "large_model_kaggle_gpu_runtime_verified",
                "large_model_7b_runtime_verified",
            ],
        }
        fresh_probe = {
            "schema": "kaggle_32b_quantized_live_experiment_summary_v1",
            "ok": True,
            "conclusion": "fresh_kaggle_32b_q2k_download_and_runtime_prepare_verified_but_one_token_generation_blocked_by_kaggle_kill",
            "aggregate": {
                "fresh_kaggle_runs": 2,
                "gpu_hardware_verified": True,
                "q2k_all_splits_downloaded": True,
                "largest_downloaded_mb": 11740,
                "cuda_source_build_verified": True,
                "one_token_generation_verified": False,
                "blocked_at": "llama_run_start",
                "kaggle_terminal_status": "ERROR",
                "kaggle_log_signal": "Killed",
                "all_kernels_deleted": True,
                "all_private_packages_removed": True,
            },
            "experiments": [
                {
                    "backend": "source-cuda",
                    "fresh_kaggle_run_performed": True,
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                    "downloaded_file_count": 4,
                    "downloaded_mb": 11740,
                    "probe_stage": "llama_run_start",
                    "probe_success": False,
                    "one_token_generation_verified": False,
                    "cuda_build_ok": True,
                    "cuda_build_duration_seconds": 1777.453,
                    "kernel_deleted": True,
                    "private_package_removed": True,
                }
            ],
            "safety": {"public_artifact_safe": True},
        }
        stage_owned_probe = {
            "schema": "kaggle_32b_stage_owned_safetensors_probe_v1",
            "ok": True,
            "stage_owned_quantized_32b_loading_ready": True,
            "fresh_kaggle_run_performed": True,
            "gpu_hardware_verified": True,
            "coverage_ready": True,
            "stage_owned_download_scope_ready": True,
            "loads_only_stage_weight_keys_ready": True,
            "all_stage_reports_downloaded": True,
            "all_stage_owned_loading_ready": True,
            "model": {
                "repo": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                "quantization_format": "awq_safetensors",
            },
            "runtime": {
                "stage_count": 2,
                "stage_ids": [0, 1],
                "one_token_generation_verified": False,
                "stage_owned_loading_only": True,
            },
            "kaggle_lifecycle": {
                "kernels_deleted": True,
                "private_packages_removed": True,
            },
            "stage_summaries": [
                {
                    "stage_id": 0,
                    "stage_ok": True,
                    "gpu_verified": True,
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                    "stage_layer_range": [0, 32],
                    "assigned_weight_key_count": 833,
                    "assigned_weight_file_count": 3,
                    "downloaded_file_count": 3,
                    "loaded_weight_key_count": 833,
                    "loaded_tensor_gb": 9.000671,
                    "materialize_clone_requested": True,
                    "materialized_weight_key_count": 833,
                    "materialized_tensor_gb": 9.000671,
                    "retained_tensor_gb": 9.000671,
                    "loads_only_stage_weight_keys": True,
                    "cross_stage_weight_keys_loaded": False,
                    "stage_weight_downloads_only_stage_files": True,
                },
                {
                    "stage_id": 1,
                    "stage_ok": True,
                    "gpu_verified": True,
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                    "stage_layer_range": [32, 64],
                    "assigned_weight_key_count": 834,
                    "assigned_weight_file_count": 3,
                    "downloaded_file_count": 3,
                    "loaded_weight_key_count": 834,
                    "loaded_tensor_gb": 9.000681,
                    "materialize_clone_requested": True,
                    "materialized_weight_key_count": 834,
                    "materialized_tensor_gb": 9.000681,
                    "retained_tensor_gb": 9.000681,
                    "loads_only_stage_weight_keys": True,
                    "cross_stage_weight_keys_loaded": False,
                    "stage_weight_downloads_only_stage_files": True,
                },
            ],
            "safety": {"public_artifact_safe": True},
        }
        activation_decode_probe = {
            "schema": "kaggle_32b_stage_owned_activation_decode_probe_v1",
            "ok": True,
            "fresh_kaggle_run_performed": True,
            "execution_mode": "coordinator",
            "coordinator_direct_management_verified": True,
            "cross_kernel_activation_decode_verified": True,
            "one_token_generation_verified": True,
            "multi_token_decode_verified": True,
            "generated_token_count": 2,
            "max_new_tokens": 2,
            "stage_owned_awq_runtime_verified": True,
            "activation_handoff_verified": True,
            "model": {
                "repo": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                "quantization": "awq",
                "stage_count": 2,
                "split_index": 32,
            },
            "kaggle_lifecycle": {
                "kernels_deleted": True,
                "private_packages_removed": True,
                "private_activation_removed": True,
            },
            "comparison": {
                "two_kernel_generated_token_count": 2,
                "two_kernel_ready": True,
                "two_kernel_stability": "completed",
                "two_kernel_completed_task_count": 4,
                "two_kernel_stage_latency": {
                    "stage0": {"count": 2, "total_seconds": 11.09, "avg_seconds": 5.545, "max_seconds": 6.175},
                    "stage1": {"count": 2, "total_seconds": 10.881, "avg_seconds": 5.441, "max_seconds": 5.855},
                },
                "two_kernel_stage_memory": {
                    "stage0": {
                        "loaded_tensor_gb": 9.000671,
                        "loaded_weight_key_count": 833,
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 9275.845},
                        "cuda_memory_after_execution": {"cuda_available": True, "max_allocated_mb": 10904.348},
                    },
                    "stage1": {
                        "loaded_tensor_gb": 9.000681,
                        "loaded_weight_key_count": 834,
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 9275.854},
                        "cuda_memory_after_execution": {"cuda_available": True, "max_allocated_mb": 10904.357},
                    },
                },
                "single_kernel_attempted": True,
                "single_kernel_ok": True,
                "single_kernel_generated_token_count": 2,
                "single_kernel_wall_time_seconds": 139.362,
                "single_kernel_tokens_per_second": 0.014351,
                "single_kernel_stage_memory": {
                    "stage0": {
                        "loaded_tensor_gb": 9.000671,
                        "loaded_weight_key_count": 833,
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 9283.97},
                    },
                    "stage1": {
                        "loaded_tensor_gb": 9.000681,
                        "loaded_weight_key_count": 834,
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 9283.979},
                    },
                },
                "single_kernel_stability": "completed",
            },
            "single_kernel_baseline": {
                "ok": True,
                "generated_token_count": 2,
                "wall_time_seconds": 139.362,
                "tokens_per_second": 0.014351,
                "stage0": {"loaded_tensor_gb": 9.000671},
                "stage1": {"loaded_tensor_gb": 9.000681},
                "metrics": {"generated_token_count": 2, "wall_time_seconds": 139.362, "tokens_per_second": 0.014351},
            },
            "stage_summaries": [
                {
                    "mode": "stage0",
                    "stage_id": 0,
                    "ok": True,
                    "gpu_verified": True,
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                    "stage_layer_range": [0, 32],
                    "assigned_weight_key_count": 833,
                    "assigned_weight_file_count": 3,
                    "loaded_weight_key_count": 833,
                    "loaded_tensor_gb": 9.000671,
                    "awq_stage_model_prepared": True,
                    "activation_ready": True,
                    "activation_hash": "sha256:activation",
                    "diagnosis_codes": ["kaggle_32b_stage0_activation_ready"],
                },
                {
                    "mode": "stage1",
                    "stage_id": 1,
                    "ok": True,
                    "gpu_verified": True,
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                    "stage_layer_range": [32, 64],
                    "assigned_weight_key_count": 834,
                    "assigned_weight_file_count": 3,
                    "loaded_weight_key_count": 834,
                    "loaded_tensor_gb": 9.000681,
                    "awq_stage_model_prepared": True,
                    "activation_ready": True,
                    "stage1_decode_ready": True,
                    "generated_token_count": 1,
                    "activation_hash": "sha256:activation",
                    "output_hash": "sha256:output",
                    "diagnosis_codes": ["kaggle_32b_stage1_decode_ready"],
                },
            ],
            "safety": {
                "public_artifact_safe": True,
                "activation_public": False,
                "hidden_state_public": False,
                "generated_token_ids_public": False,
            },
        }
        if upper_bound_crossing:
            activation_decode_probe["upper_bound_crossing_verified"] = True
            activation_decode_probe["generated_token_count"] = 1
            activation_decode_probe["max_new_tokens"] = 1
            activation_decode_probe["model"]["stage_count"] = 4
            activation_decode_probe["comparison"]["upper_bound_crossing_verified"] = True
            activation_decode_probe["comparison"]["two_kernel_generated_token_count"] = 1
            activation_decode_probe["comparison"]["two_kernel_completed_task_count"] = 4
            activation_decode_probe["comparison"]["two_kernel_stage_latency"] = {
                f"stage{stage_id}": {"count": 1, "total_seconds": 2.0 + stage_id, "avg_seconds": 2.0 + stage_id, "max_seconds": 2.0 + stage_id}
                for stage_id in range(4)
            }
            activation_decode_probe["comparison"]["two_kernel_stage_memory"] = {
                f"stage{stage_id}": {
                    "loaded_tensor_gb": 4.5 + stage_id / 10,
                    "loaded_weight_key_count": 420 + stage_id,
                    "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 5200.0 + stage_id},
                }
                for stage_id in range(4)
            }
            activation_decode_probe["comparison"]["single_kernel_ok"] = False
            activation_decode_probe["comparison"]["single_kernel_generated_token_count"] = 0
            activation_decode_probe["comparison"]["single_kernel_wall_time_seconds"] = 0.0
            activation_decode_probe["comparison"]["single_kernel_tokens_per_second"] = 0.0
            activation_decode_probe["comparison"]["single_kernel_stability"] = "failed_or_killed"
            activation_decode_probe["comparison"]["single_kernel_blockers"] = [
                "single_kernel_t4x2_gpu_count_below_required_stage_count"
            ]
            activation_decode_probe["comparison"]["single_kernel_stage_memory"] = {}
            activation_decode_probe["single_kernel_baseline"] = {
                "ok": False,
                "generated_token_count": 0,
                "metrics": {"generated_token_count": 0},
                "blockers": ["single_kernel_t4x2_gpu_count_below_required_stage_count"],
                "diagnosis_codes": ["single_kernel_t4x2_exceeds_gpu_count"],
            }
            activation_decode_probe["stage_summaries"] = [
                {
                    "mode": "shard0" if stage_id < 2 else "shard1",
                    "stage_id": stage_id,
                    "ok": True,
                    "gpu_verified": True,
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                    "stage_layer_range": [stage_id * 16, (stage_id + 1) * 16],
                    "assigned_weight_key_count": 420 + stage_id,
                    "assigned_weight_file_count": 2,
                    "loaded_weight_key_count": 420 + stage_id,
                    "loaded_tensor_gb": 4.5 + stage_id / 10,
                    "awq_stage_model_prepared": True,
                    "activation_ready": stage_id < 3,
                    "stage1_decode_ready": stage_id == 3,
                    "generated_token_count": 1 if stage_id == 3 else 0,
                    "activation_hash": f"sha256:a{stage_id}" if stage_id < 3 else "",
                    "output_hash": "sha256:o0" if stage_id == 3 else "",
                    "diagnosis_codes": ["kaggle_32b_stage_owned_awq_runtime_ready"],
                }
                for stage_id in range(4)
            ]
        return {
            "production_like": self._write_json(base / "gpu_swarm_production_like_validation.json", production_like),
            "core_status": self._write_json(base / "core_status.json", core_status),
            "large_kaggle": self._write_json(base / "large_model_kaggle_validation.json", large_kaggle),
            "fresh_probe": self._write_json(base / "kaggle_32b_quantized_live_experiment_summary.json", fresh_probe),
            "stage_owned_probe": self._write_json(base / "kaggle_32b_stage_owned_safetensors_probe.json", stage_owned_probe),
            "activation_decode_probe": self._write_json(base / "kaggle_32b_stage_owned_activation_decode_probe.json", activation_decode_probe),
        }

    def _pack_args(self, reports: dict[str, Path], output_dir: Path, *extra: str) -> list[str]:
        return [
            "--output-dir",
            str(output_dir),
            "--production-like-report",
            str(reports["production_like"]),
            "--core-status-report",
            str(reports["core_status"]),
            "--large-model-kaggle-report",
            str(reports["large_kaggle"]),
            "--fresh-32b-live-probe-report",
            str(reports["fresh_probe"]),
            "--fresh-32b-stage-owned-loading-probe-report",
            str(reports["stage_owned_probe"]),
            "--fresh-32b-activation-decode-probe-report",
            str(reports["activation_decode_probe"]),
            *extra,
        ]

    def test_pack_builds_ready_infeasibility_report_and_check_validates(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        output_dir = base / "report"

        report = pack.build_report(pack.parse_args(self._pack_args(reports, output_dir)))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["kaggle_swarm_32b_quantized_feasibility_ready"])
        self.assertTrue(report["candidate_32b_model_selected"])
        self.assertTrue(report["quantized_runtime_plan_ready"])
        self.assertTrue(report["kaggle_multi_kernel_topology_ready"])
        self.assertTrue(report["stage_partition_plan_ready"])
        self.assertTrue(report["per_stage_memory_estimate_ready"])
        self.assertTrue(report["activation_transfer_estimate_ready"])
        self.assertTrue(report["kaggle_stage_package_plan_ready"])
        self.assertTrue(report["stage_owned_loading_feasible"])
        self.assertTrue(report["one_token_generation_feasible"])
        self.assertTrue(report["multi_token_generation_feasible"])
        self.assertTrue(report["coordinator_direct_management_feasible"])
        self.assertFalse(report["batch_or_sequential_request_feasible"])
        self.assertFalse(report["stage_requeue_feasible"])
        self.assertEqual(report["largest_feasible_model_tier"], "32b-quantized-2token-rc")
        self.assertEqual(report["largest_attempted_model_tier"], "32b-quantized")
        self.assertEqual(report["feasibility_verdict"], "feasible_32b_multitoken_coordinator_rc")
        self.assertEqual(report["blocked_reason"], "")
        self.assertTrue(report["fresh_kaggle_run_performed"])
        self.assertTrue(report["external_runtime_verified"])
        self.assertTrue(report["retained_evidence_imported"])
        self.assertTrue(report["fresh_32b_live_probe_summary"]["gpu_hardware_verified"])
        self.assertTrue(report["fresh_32b_live_probe_summary"]["q2k_all_splits_downloaded"])
        self.assertTrue(report["fresh_32b_live_probe_summary"]["cuda_source_build_verified"])
        self.assertFalse(report["fresh_32b_live_probe_summary"]["one_token_generation_verified"])
        self.assertTrue(report["fresh_32b_stage_owned_loading_probe_summary"]["stage_owned_quantized_32b_loading_ready"])
        self.assertTrue(report["fresh_32b_stage_owned_loading_probe_summary"]["loads_only_stage_weight_keys_ready"])
        self.assertTrue(report["fresh_32b_activation_decode_probe_summary"]["cross_kernel_activation_decode_verified"])
        self.assertTrue(report["fresh_32b_activation_decode_probe_summary"]["one_token_generation_verified"])
        self.assertTrue(report["fresh_32b_activation_decode_probe_summary"]["multi_token_decode_verified"])
        self.assertTrue(report["fresh_32b_activation_decode_probe_summary"]["coordinator_direct_management_verified"])
        self.assertTrue(report["fresh_32b_activation_decode_probe_summary"]["single_kernel_ok"])
        self.assertEqual(
            report["fresh_32b_activation_decode_probe_summary"]["two_kernel_stage_memory"]["stage0"]["loaded_tensor_gb"],
            9.000671,
        )
        self.assertEqual(
            report["fresh_32b_activation_decode_probe_summary"]["single_kernel_stage_memory"]["stage1"]["loaded_weight_key_count"],
            834,
        )
        self.assertTrue(report["evidence_validation"]["fresh_32b_stage_owned_clone_verified"])
        self.assertFalse(report["blocker_details"]["fresh_32b_generation"]["blocked"])
        self.assertEqual(check.validate_report(report), [])

    def test_pack_builds_upper_bound_crossing_report_and_check_validates(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base, upper_bound_crossing=True)
        output_dir = base / "upper-bound"

        report = pack.build_report(pack.parse_args(self._pack_args(reports, output_dir)))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["multi_token_generation_feasible"])
        self.assertTrue(report["coordinator_direct_management_feasible"])
        self.assertTrue(report["upper_bound_crossing_feasible"])
        self.assertEqual(report["largest_feasible_model_tier"], "32b-quantized-4stage-upper-bound-rc")
        self.assertEqual(report["feasibility_verdict"], "feasible_32b_upper_bound_crossing_rc")
        summary = report["fresh_32b_activation_decode_probe_summary"]
        self.assertTrue(summary["upper_bound_crossing_verified"])
        self.assertFalse(summary["single_kernel_ok"])
        self.assertEqual(summary["single_kernel_blockers"], ["single_kernel_t4x2_gpu_count_below_required_stage_count"])
        self.assertEqual(summary["two_kernel_stage_memory"]["stage3"]["loaded_weight_key_count"], 423)
        self.assertIn("fresh_32b_legacy_live_probe_one_token_blocked", report["diagnosis_codes"])
        self.assertNotIn("fresh_32b_one_token_generation_blocked", report["diagnosis_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_memory_runtime_and_blocker_details_are_explicit(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        report = pack.build_report(pack.parse_args(self._pack_args(reports, base / "blocker")))

        runtime = report["quantized_runtime_plan"]
        memory = report["stage_partition_plan"]["memory_estimate"]
        blockers = report["blocker_details"]
        self.assertEqual(runtime["selected_runtime_adapter"], "hf-awq-stage-selective-kaggle")
        self.assertEqual(runtime["runtime_adapter_blocker"], "")
        self.assertTrue(runtime["selected_runtime_adapter_integrated_for_32b_kaggle_swarm"])
        self.assertLess(memory["margin_mb_per_stage"], 0)
        self.assertGreater(memory["required_vram_mb_per_stage"], memory["available_vram_mb_per_gpu"])
        for name in [
            "runtime_adapter",
            "vram",
            "model_format",
            "kaggle_quota",
            "download_build_time",
            "activation_transfer",
            "stage_partitioning",
            "missing_live_hardware",
            "fresh_32b_generation",
        ]:
            self.assertIn(name, blockers)
        self.assertFalse(blockers["runtime_adapter"]["blocked"])
        self.assertFalse(blockers["vram"]["blocked"])
        self.assertFalse(blockers["model_format"]["blocked"])
        self.assertFalse(blockers["kaggle_quota"]["blocked"])
        self.assertFalse(blockers["download_build_time"]["blocked"])
        self.assertFalse(blockers["fresh_32b_generation"]["blocked"])

    def test_fixture_mode_is_ci_safe_and_public_artifacts_are_redacted(self) -> None:
        base = self._tmp_dir()
        output_dir = base / "fixture"
        report = pack.build_report(pack.parse_args([
            "--execution-mode",
            "fixture",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["execution_mode"], "fixture")
        self.assertFalse(report["fresh_kaggle_run_performed"])
        self.assertFalse(report["kaggle_multi_kernel_topology"]["private_package_payloads_written"])
        self.assertEqual(pack.public_redaction_errors(report), [])
        scanned = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        for fragment in [
            "KAGGLE_KEY=",
            "KAGGLE_USERNAME=",
            "CROWDTENSOR_MINER_TOKEN=",
            "CROWDTENSOR_OBSERVER_TOKEN=",
            "CROWDTENSOR_ADMIN_TOKEN=",
            "operator.private.env",
            "miner.private.env",
            "miner_registry.json",
            "kernel.py",
            "SOURCE_TARBALL_B64",
            "MINER_ENV_TEXT",
            "INLINE_KERNEL_PAYLOAD_B64",
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

    def test_check_script_builds_and_validates_fixture_report(self) -> None:
        base = self._tmp_dir()
        result = check.build_check(check.parse_args([
            "--execution-mode",
            "fixture",
            "--output-dir",
            str(base / "check"),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["kaggle_swarm_32b_quantized_feasibility_ready"])
        self.assertEqual(result["errors"], [])

    def test_cli_wrapper_generates_summary(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        output_dir = base / "cli"

        args = cli.parse_args([
            "gpu-swarm",
            "kaggle-32b-feasibility",
            "--output-dir",
            str(output_dir),
            "--production-like-report",
            str(reports["production_like"]),
            "--core-status-report",
            str(reports["core_status"]),
            "--large-model-kaggle-report",
            str(reports["large_kaggle"]),
            "--fresh-32b-live-probe-report",
            str(reports["fresh_probe"]),
            "--fresh-32b-stage-owned-loading-probe-report",
            str(reports["stage_owned_probe"]),
            "--fresh-32b-activation-decode-probe-report",
            str(reports["activation_decode_probe"]),
            "--json",
        ])
        summary = cli.build_kaggle_swarm_32b_quantized_feasibility(args)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], pack.SCHEMA)
        self.assertEqual(summary["cli_schema"], "kaggle_swarm_32b_quantized_feasibility_cli_v1")
        self.assertEqual(summary["largest_attempted_model_tier"], "32b-quantized")
        self.assertTrue((output_dir / "kaggle_swarm_32b_quantized_feasibility_cli_summary.json").is_file())

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_kaggle_swarm_32b_quantized_feasibility(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor Kaggle Swarm 32B Quantized Feasibility RC", output)
        self.assertIn("feasible_32b_multitoken_coordinator_rc", output)
        self.assertIn("adapter=hf-awq-stage-selective-kaggle", output)
        self.assertIn("margin_mb_per_stage=-331", output)

    def test_argument_validation_rejects_unbounded_attempts_and_missing_imports(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args(["--max-fresh-model-attempts", "3"])
        with self.assertRaises(SystemExit):
            pack.parse_args(["--max-attempt-timeout-minutes", "61"])
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "gpu-swarm",
                "kaggle-32b-feasibility",
                "--max-requeue-attempts",
                "2",
            ])
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "gpu-swarm",
                "kaggle-32b-feasibility",
                "--production-like-report",
                "/tmp/missing-kaggle-32b-production-like.json",
            ])


if __name__ == "__main__":
    unittest.main()
