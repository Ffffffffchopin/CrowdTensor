from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import core_technology_validation_status_check as check
from scripts import core_technology_validation_status_pack as pack


class CoreTechnologyValidationStatusTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_core_status_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_default_retained_evidence_keeps_core_incomplete(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["--output-dir", str(output_dir)]))

        self.assertFalse(report["ok"])
        self.assertFalse(report["core_validation_ready"])
        self.assertTrue(report["small_tier_gpu_validated"])
        self.assertFalse(report["seven_b_eight_b_validated"])
        self.assertEqual(report["largest_successful_tier"], "small")
        self.assertIn("core_technology_validation_incomplete", report["diagnosis_codes"])
        self.assertIn("core_7b_8b_real_runtime_not_verified", report["blockers"])
        self.assertTrue(report["readiness_truth"]["small_tier_success_is_not_7b_8b_completion"])
        self.assertEqual(report["public_leak_paths"], [])
        self.assertIn("llama_like_local_evidence", report)
        self.assertFalse(report["llama_like_local_evidence"].get("large_model_validation"))
        check.validate_report(report)

        for name in [
            "core_technology_validation_status.json",
            "core_technology_validation_status.md",
            "support_bundle.json",
        ]:
            self.assertTrue((output_dir / name).is_file(), name)

    def test_synthetic_7b_success_can_mark_core_ready(self) -> None:
        output_dir = self._tmp_dir()
        small_report = self._write_json(
            output_dir / "small.json",
            {
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "beta": {
                    "model_id": "gpt2-xl",
                    "backend": "hf_transformers_cuda",
                    "partition_mode": "stage_local",
                    "stage_count": 2,
                },
                "payload_summaries": {
                    "real_llm_internet_beta": {
                        "generation": {
                            "generated_token_count": 1,
                            "generated_text_redacted": True,
                        }
                    }
                },
                "model_execution_support": {
                    "parameter_count_estimate": 1558000000,
                    "large_model_sharded_execution_ready": False,
                    "partial_weight_loading_plan_ready": True,
                    "partial_weight_runtime_execution_ready": False,
                    "true_partial_weight_loading_ready": False,
                    "large_model_blockers": ["real_llm_llama_like_runtime_execution_missing"],
                },
                "diagnosis_codes": [
                    "public_swarm_gpu_beta_kaggle_auto_ready",
                    "external_runtime_verified",
                    "kaggle_kernels_deleted",
                    "decoded_tokens_match",
                ],
            },
        )
        seven_report = self._write_json(
            output_dir / "seven.json",
            {
                "schema": "large_model_kaggle_validation_run_v1",
                "ok": True,
                "validation": {
                    "real_7b_runtime_verified": True,
                    "real_runtime_verified": True,
                    "gpu_runtime_verified": True,
                    "sharded_path_verified": True,
                    "multi_worker_sharded_path_verified": True,
                    "core_validation_ready": True,
                },
                "hardware": {
                    "kaggle_gpu_verified": True,
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                },
                "diagnosis_codes": ["large_model_kaggle_gpu_hardware_verified"],
            },
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir / "status"),
            "--small-gpu-report",
            str(small_report),
            "--seven-b-blocker-report",
            str(seven_report),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["core_validation_ready"])
        self.assertTrue(report["seven_b_eight_b_validated"])
        self.assertIn("core_technology_validation_ready", report["diagnosis_codes"])
        check.validate_report(report, require_core_ready=True)

    def test_partial_weight_plan_ready_does_not_mark_core_ready_without_7b_runtime(self) -> None:
        output_dir = self._tmp_dir()
        small_report = self._write_json(
            output_dir / "small-partial-plan.json",
            {
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "beta": {
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "backend": "hf_transformers_cuda",
                    "partition_mode": "stage_local",
                    "stage_count": 2,
                },
                "payload_summaries": {
                    "real_llm_internet_beta": {
                        "generation": {
                            "generated_token_count": 1,
                            "generated_text_redacted": True,
                        }
                    }
                },
                "model_execution_support": {
                    "parameter_count_estimate": 7615000000,
                    "large_model_sharded_execution_ready": False,
                    "partial_weight_loading_plan_ready": True,
                    "partial_weight_runtime_execution_ready": False,
                    "true_partial_weight_loading_ready": False,
                    "large_model_blockers": ["real_llm_llama_like_runtime_execution_missing"],
                },
                "diagnosis_codes": [
                    "public_swarm_gpu_beta_kaggle_auto_ready",
                    "external_runtime_verified",
                    "kaggle_kernels_deleted",
                    "decoded_tokens_match",
                    "real_llm_partial_weight_plan_ready",
                ],
            },
        )
        seven_report = self._write_json(
            output_dir / "seven-blocked.json",
            {
                "schema": "large_model_kaggle_validation_run_v1",
                "ok": False,
                "validation": {
                    "real_7b_runtime_verified": False,
                    "real_runtime_verified": False,
                    "gpu_runtime_verified": False,
                    "sharded_path_verified": False,
                    "multi_worker_sharded_path_verified": False,
                    "core_validation_ready": False,
                },
                "hardware": {
                    "kaggle_gpu_verified": True,
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                },
                "diagnosis_codes": ["large_model_kaggle_gpu_hardware_verified"],
            },
        )
        llama_report = self._write_json(
            output_dir / "llama-local.json",
            {
                "schema": "real_llm_sharded_evidence_v1",
                "ok": True,
                "artifact": {
                    "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                    "backend": "hf_transformers_cpu",
                    "partition_mode": "stage_local",
                },
                "generation": {"generated_token_count": 1},
                "diagnosis_codes": [
                    "real_llm_sharded_ready",
                    "stage_local_partition_ready",
                    "decoded_tokens_match",
                ],
            },
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir / "status"),
            "--small-gpu-report",
            str(small_report),
            "--seven-b-blocker-report",
            str(seven_report),
            "--llama-like-local-report",
            str(llama_report),
        ]))

        self.assertFalse(report["core_validation_ready"])
        self.assertFalse(report["seven_b_eight_b_validated"])
        self.assertTrue(report["llama_like_local_evidence"]["ready"])
        self.assertFalse(report["llama_like_local_evidence"]["large_model_validation"])
        self.assertTrue(report["readiness_truth"]["partial_weight_plan_is_not_runtime_execution"])
        self.assertIn("real_llm_partial_weight_runtime_execution_missing", report["blockers"])
        check.validate_report(report)

    def test_public_leak_detection_blocks_report(self) -> None:
        output_dir = self._tmp_dir()
        small_report = self._write_json(
            output_dir / "small-leak.json",
            {
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "generated_text": "not public",
                "diagnosis_codes": [
                    "public_swarm_gpu_beta_kaggle_auto_ready",
                    "external_runtime_verified",
                    "kaggle_kernels_deleted",
                ],
            },
        )
        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir / "status"),
            "--small-gpu-report",
            str(small_report),
            "--seven-b-blocker-report",
            str(output_dir / "missing-seven.json"),
        ]))

        self.assertFalse(report["ok"])
        self.assertIn("core_validation_status_public_leak_detected", report["diagnosis_codes"])
        self.assertTrue(report["public_leak_paths"])


if __name__ == "__main__":
    unittest.main()
