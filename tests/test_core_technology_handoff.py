from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from crowdtensor import core_technology_handoff as handoff
from crowdtensor import large_model_inference_rc as inference_rc
from scripts import core_technology_handoff_check as check
from scripts import core_technology_handoff_pack as pack


class CoreTechnologyHandoffTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_core_handoff_test_"))

    def _write_json(self, output_dir: Path, name: str, payload: dict) -> Path:
        path = output_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _live_report(self, *, model_id: str, token_count: int, multi_token: bool) -> dict:
        return {
            "schema": "real_llm_internet_beta_v1",
            "ok": True,
            "mode": "kaggle-auto",
            "workload": {
                "hf_model_id": model_id,
                "max_new_tokens": token_count,
                "real_llm_backend": "hf_transformers_cuda",
                "real_llm_execution_mode": "stage_selective_hf",
                "real_llm_partition_mode": "stage_local",
                "stage_mode": "split",
                "workload_type": "real_llm_sharded_infer",
            },
            "runtime_classification": {
                "external_runtime_verified": True,
                "kaggle_auto": True,
                "kaggle_notebook_verified": True,
            },
            "payload_summaries": {
                "kaggle_package": {
                    "schema": "kaggle_real_llm_live_package_v1",
                    "ok": True,
                    "stages": [
                        {
                            "stage": "stage0",
                            "role": "normal",
                            "gpu_accelerator_enabled": True,
                            "cuda_preflight_present": True,
                            "hf_runtime_enabled": True,
                            "real_llm_stage_role_present": True,
                            "real_llm_backend": "hf_transformers_cuda",
                            "real_llm_execution_mode": "stage_selective_hf",
                            "real_llm_partition_mode": "stage_local",
                        },
                        {
                            "stage": "stage1",
                            "role": "normal",
                            "gpu_accelerator_enabled": True,
                            "cuda_preflight_present": True,
                            "hf_runtime_enabled": True,
                            "real_llm_stage_role_present": True,
                            "real_llm_backend": "hf_transformers_cuda",
                            "real_llm_execution_mode": "stage_selective_hf",
                            "real_llm_partition_mode": "stage_local",
                        },
                    ],
                },
                "external_alpha": {
                    "schema": "real_llm_internet_alpha_v1",
                    "ok": True,
                    "workload": {
                        "hf_model_id": model_id,
                        "max_new_tokens": token_count,
                        "real_llm_execution_mode": "stage_selective_hf",
                        "real_llm_partition_mode": "stage_local",
                        "stage_mode": "split",
                    },
                    "runtime_classification": {"external_runtime_verified": True},
                    "generation": {
                        "generated_text_hash": "sha256:" + "a" * 64,
                        "generated_text_redacted": True,
                        "generated_token_count": token_count,
                        "max_new_tokens": token_count,
                        "multi_token_generation_ready": multi_token,
                    },
                    "diagnosis_codes": [
                        "external_runtime_verified",
                        "generation_complete",
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                        "real_llm_sharded_ready",
                    ],
                },
            },
            "diagnosis_codes": [
                "external_runtime_verified",
                "generation_complete",
                "kaggle_auto_ready",
                "kaggle_kernels_deleted",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "real_llm_internet_beta_ready",
            ] + (["multi_token_generation_ready"] if multi_token else []),
        }

    def _stage_selective_plan_report(self) -> dict:
        def model_plan(model_id: str, parameter_count: int) -> dict:
            stage_plans = [
                {
                    "stage_id": index,
                    "stage_count": 4,
                    "stage_layer_range": [index * 4, (index + 1) * 4],
                    "loads_only_stage_weight_keys": True,
                    "estimated_stage_weight_bytes_fp32": parameter_count,
                }
                for index in range(4)
            ]
            return {
                "schema": "large_model_stage_selective_model_plan_v1",
                "model_id": model_id,
                "parameter_count_estimate": parameter_count,
                "target_stage_count": 4,
                "n_stage_plan_ready": True,
                "two_stage_plan_ready": True,
                "dual_kaggle_kernel_fit_estimate": True,
                "two_stage_practical_fit_with_overhead_guard": False,
                "n_stage_max_stage_weight_gb_fp16_estimate": 8.0,
                "two_stage_max_stage_weight_gb_fp16_estimate": 14.0,
                "n_stage_partition_plan": {
                    "schema": "real_llm_n_stage_partition_plan_v1",
                    "ready": True,
                    "stage_count": 4,
                    "stage_ranges_valid": True,
                    "stage_plans": stage_plans,
                },
            }
        return {
            "schema": "large_model_stage_selective_plan_v1",
            "ok": True,
            "target_stage_count": 4,
            "kaggle_gpu_memory_gb": 15.0,
            "model_plans": [
                model_plan("Qwen/Qwen2.5-7B-Instruct", 7_615_000_000),
                model_plan("Qwen/Qwen2.5-14B-Instruct", 14_700_000_000),
            ],
            "limitations": ["planning only"],
            "safety": {"public_artifact_safe": True},
            "diagnosis_codes": [
                "large_model_7b_partition_plan_ready",
                "large_model_14b_partition_plan_ready",
                "large_model_n_stage_partition_plan_ready",
            ],
        }

    def _performance_report(self) -> dict:
        return {
            "schema": "real_llm_sharded_evidence_v1",
            "ok": True,
            "performance": {
                "schema": "real_llm_sharded_performance_summary_v1",
                "public_artifact_safe": True,
                "memory": {
                    "full_model_parameter_count": 14_700_000_000,
                    "stage0_parameter_count": 7_385_014_272,
                    "stage1_parameter_count": 7_385_019_392,
                    "stage0_loaded_tensor_bytes": 14_770_028_544,
                    "stage1_loaded_tensor_bytes": 14_770_038_784,
                    "stage0_weight_download_scope": "stage_owned_weight_files",
                    "stage1_weight_download_scope": "stage_owned_weight_files",
                    "stage0_weight_download_file_count": 4,
                    "stage1_weight_download_file_count": 5,
                    "stage_weight_downloads_only_stage_files": True,
                    "stage_gpu_memory_reduced": True,
                },
                "latency": {
                    "effective_elapsed_seconds": 264.3,
                    "total_stage_elapsed_ms": 264300.0,
                    "stage0_elapsed_ms": 109700.0,
                    "stage1_elapsed_ms": 154600.0,
                },
                "throughput": {
                    "generated_token_count": 1,
                    "max_new_tokens": 1,
                    "completed_generation_steps": 1,
                    "tokens_per_second_effective": 0.0037,
                },
                "failure_recovery": {
                    "requeue_observed": False,
                    "stage0_attempt": 1,
                    "stage1_attempt": 1,
                },
            },
            "safety": {
                "generated_text_redacted": True,
                "generated_token_ids_redacted": True,
                "raw_activation_redacted": True,
            },
        }

    def test_pack_report_is_ci_safe_and_check_validates_it(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["--output-dir", str(output_dir), "--mode", "fixture"]))

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], handoff.HANDOFF_SCHEMA)
        self.assertFalse(report["real_runtime_verified"])
        self.assertFalse(report["real_7b_runtime_verified"])
        self.assertEqual(report["inference_rc_report"]["schema"], inference_rc.RC_SCHEMA)
        self.assertIn("core_technology_handoff_rc_ready", report["diagnosis_codes"])
        self.assertIn("external_real_runtime_resources_required", report["blockers"])
        self.assertTrue(report["safety"]["public_artifact_safe"])
        self.assertEqual(handoff.public_redaction_errors(report), [])
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], report["artifact_summary"]["artifact_count"])
        check.validate_report(report)

        for name in [
            "core_technology_handoff_rc.json",
            "core_technology_handoff_rc.md",
            "deployment_runbook.json",
            "next_layer_contract.json",
            "adapter_conformance.json",
            "test_gate_summary.json",
            "support_bundle.json",
            "inference-rc/core_technology_inference_rc.json",
        ]:
            self.assertTrue((output_dir / name).is_file(), name)

    def test_handoff_contracts_cover_deployment_adapters_and_next_layers(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["--output-dir", str(output_dir)]))
        deployment = report["deployment_runbook"]
        next_layer = report["next_layer_integration_contract"]
        adapter = report["adapter_conformance"]
        tests = report["test_gate_summary"]

        self.assertEqual(deployment["schema"], handoff.DEPLOYMENT_RUNBOOK_SCHEMA)
        self.assertTrue(deployment["local_fixture"]["ci_safe"])
        self.assertEqual(deployment["local_real_runtime"]["max_new_tokens_max"], 8)
        self.assertIn("process_leak_check", deployment["cleanup"])
        self.assertTrue(deployment["lan_vpn_two_worker_runtime"]["controlled_network_only"])

        self.assertEqual(next_layer["schema"], handoff.NEXT_LAYER_CONTRACT_SCHEMA)
        self.assertTrue(next_layer["ready"])
        self.assertIn("crowdtensor large-model-shard-rc", next_layer["control_layer"]["stable_entrypoints"])
        self.assertEqual(next_layer["sample_control_request"]["raw_prompt_public"], False)
        self.assertIn("runner_result.real_runtime_verified", next_layer["permissions_trust_billing_layer"]["core_signals"])
        self.assertIn("output_digest", next_layer["correctness_contract"])

        self.assertEqual(adapter["schema"], handoff.ADAPTER_CONFORMANCE_SCHEMA)
        self.assertTrue(adapter["ready"])
        self.assertEqual(set(adapter["future_runtime_backends"]), set(inference_rc.UNSUPPORTED_RUNTIMES))
        self.assertTrue(all(item["conformant"] for item in adapter["descriptor_checks"]))

        self.assertEqual(tests["schema"], handoff.TEST_GATE_SCHEMA)
        self.assertIn("deployment/runbook artifact generation", tests["coverage"])
        self.assertIn("backward compatibility", tests["coverage"])

    def test_real_run_import_marks_handoff_real_verified(self) -> None:
        output_dir = self._tmp_dir()
        real_run_path = output_dir / "real_run.json"
        real_run_path.write_text(
            json.dumps({
                "metrics": {
                    "ttft_ms": 111.0,
                    "tokens_per_second": 9.25,
                    "wall_time_seconds": 1.1,
                    "generated_token_count": 3,
                    "max_new_tokens": 3,
                    "output_digest": "sha256:" + "2" * 64,
                }
            }),
            encoding="utf-8",
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--real-run-report",
            str(real_run_path),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["real_runtime_verified"])
        self.assertFalse(report["real_7b_runtime_verified"])
        self.assertEqual(report["runner_result"]["runner_mode"], "real-import")
        self.assertIn("core_technology_real_runtime_verified", report["diagnosis_codes"])
        self.assertIn("core_technology_real_7b_runtime_not_verified", report["blockers"])
        self.assertNotIn("external_real_runtime_resources_required", report["blockers"])
        check.validate_report(report)

    def test_imports_large_model_stage_selective_evidence_for_next_layers(self) -> None:
        output_dir = self._tmp_dir()
        seven_b = self._write_json(output_dir, "seven_b.json", self._live_report(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            token_count=2,
            multi_token=True,
        ))
        fourteen_b = self._write_json(output_dir, "fourteen_b.json", self._live_report(
            model_id="Qwen/Qwen2.5-14B-Instruct",
            token_count=1,
            multi_token=False,
        ))
        plan = self._write_json(output_dir, "plan.json", self._stage_selective_plan_report())
        performance = self._write_json(output_dir, "performance.json", self._performance_report())

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--seven-b-live-report",
            str(seven_b),
            "--fourteen-b-live-report",
            str(fourteen_b),
            "--stage-selective-plan-report",
            str(plan),
            "--stage-selective-performance-report",
            str(performance),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["core_technology_large_model_alpha_ready"])
        evidence = report["large_model_stage_selective_evidence"]
        self.assertTrue(evidence["checks"]["seven_b_multi_token_verified"])
        self.assertTrue(evidence["checks"]["fourteen_b_dual_kaggle_verified"])
        self.assertTrue(evidence["checks"]["n_stage_partition_plan_ready"])
        self.assertTrue(evidence["checks"]["stage_selective_performance_report_ready"])
        self.assertEqual(evidence["not_completed"], [])
        self.assertEqual(evidence["seven_b_live"]["generated_token_count"], 2)
        self.assertEqual(evidence["fourteen_b_live"]["generated_token_count"], 1)
        self.assertEqual(evidence["n_stage_partition"]["target_stage_count"], 4)
        self.assertEqual(
            evidence["stage_selective_performance"]["memory"]["stage0_weight_download_scope"],
            "stage_owned_weight_files",
        )
        self.assertIn("core_technology_large_model_alpha_ready", report["diagnosis_codes"])
        self.assertIn("core_technology_7b_multi_token_verified", report["diagnosis_codes"])
        self.assertIn("core_technology_14b_dual_kaggle_verified", report["diagnosis_codes"])
        self.assertTrue(report["next_layer_integration_contract"]["large_model_stage_selective_contract"]["ready"])
        self.assertTrue(report["deployment_runbook"]["import_stage_selective_live_evidence"]["ready"])
        self.assertTrue((output_dir / "large_model_stage_selective_evidence.json").is_file())
        self.assertEqual(handoff.public_redaction_errors(report), [])
        check.validate_report(report)

    def test_cli_wrapper_generates_handoff_summary(self) -> None:
        output_dir = self._tmp_dir()
        args = cli.parse_args(["core-tech-handoff", "--output-dir", str(output_dir)])
        summary = cli.build_core_technology_handoff_rc(args)

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["cli_schema"], "core_technology_handoff_rc_cli_v1")
        self.assertEqual(summary["schema"], handoff.HANDOFF_SCHEMA)
        self.assertFalse(summary["real_runtime_verified"])
        self.assertTrue((output_dir / "core_technology_handoff_rc_cli_summary.json").is_file())
        self.assertEqual(summary["artifact_summary"]["present_artifact_count"], summary["artifact_summary"]["artifact_count"])

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_core_technology_handoff_rc(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor core technology Handoff RC", output)
        self.assertIn("real_7b=False", output)
        self.assertIn("next_layer", output)

    def test_cli_wrapper_imports_large_model_stage_selective_evidence(self) -> None:
        output_dir = self._tmp_dir()
        seven_b = self._write_json(output_dir, "seven_b.json", self._live_report(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            token_count=2,
            multi_token=True,
        ))
        fourteen_b = self._write_json(output_dir, "fourteen_b.json", self._live_report(
            model_id="Qwen/Qwen2.5-14B-Instruct",
            token_count=1,
            multi_token=False,
        ))
        plan = self._write_json(output_dir, "plan.json", self._stage_selective_plan_report())
        performance = self._write_json(output_dir, "performance.json", self._performance_report())

        args = cli.parse_args([
            "core-tech-handoff",
            "--output-dir",
            str(output_dir),
            "--seven-b-live-report",
            str(seven_b),
            "--fourteen-b-live-report",
            str(fourteen_b),
            "--stage-selective-plan-report",
            str(plan),
            "--stage-selective-performance-report",
            str(performance),
        ])
        summary = cli.build_core_technology_handoff_rc(args)

        self.assertTrue(summary["ok"])
        self.assertTrue(summary["core_technology_large_model_alpha_ready"])
        self.assertTrue(summary["large_model_stage_selective_evidence"]["checks"]["seven_b_multi_token_verified"])
        self.assertTrue(summary["artifacts"]["large_model_stage_selective_evidence_json"]["present"])
        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_core_technology_handoff_rc(summary)
        self.assertIn("stage_selective_large_model", rendered.getvalue())

    def test_cli_rejects_bad_handoff_args(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--layer-count", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--reserved-kv-cache-mb", "-1"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--mode", "real", "--max-new-tokens", "9"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--mode", "real", "--real-timeout-seconds", "1201"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--real-run-report", "/tmp/does-not-exist.json"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--seven-b-live-report", "/tmp/does-not-exist.json"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--stage-selective-performance-report", "/tmp/does-not-exist.json"])


if __name__ == "__main__":
    unittest.main()
