from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import kaggle_colab_gpu_cpu_max_parameter_search_check as check
from scripts import kaggle_colab_gpu_cpu_large_model_blocker_pack as blocker_pack
from scripts import kaggle_colab_gpu_cpu_max_parameter_search_pack as pack


class KaggleColabGpuCpuMaxParameterSearchTests(unittest.TestCase):
    def _tmp(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="ct_kaggle_colab_gpu_cpu_max_"))

    def _write(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _probe(self, *, parameter: int, ok: bool, stage_count: int, completed: int) -> dict:
        providers = ["kaggle_cuda", "colab_cuda", "cpu"] if ok or completed >= 3 else ["kaggle_cuda"]
        return {
            "schema": "kaggle_32b_full_heterogeneous_probe_v1",
            "ok": ok,
            "kaggle_colab_gpu_cpu_same_request_verified": ok,
            "same_request_72b_kaggle_colab_gpu_cpu_full_model_verified": ok and parameter == 72,
            "generated_token_count": 1 if ok else 0,
            "accepted_providers": providers,
            "provider_stage_counts": {
                "kaggle_cuda": 2 if completed >= 2 else 1,
                "colab_cuda": 1 if completed >= 3 else 0,
                "cpu": max(0, completed - 3),
                "web_tpu": 0,
            },
            "quantization": "none",
            "model": {
                "repo": f"Qwen/Qwen2.5-{parameter}B-Instruct",
                "parameter_count_b": float(parameter),
                "quantization": "none",
                "precision": "bf16_or_fp16_stage_runtime",
                "stage_count": stage_count,
                "stage_ranges": [[index, index + 1] for index in range(stage_count)],
                "expected_layer_count": 80 if parameter == 72 else (70 if parameter == 176 else 64),
                "full_layer_coverage_verified": True,
            },
            "coordinator": {"generated_token_count": 1 if ok else 0},
            "stage_task_counts": {
                f"stage{index}": 1 if index < completed else 0
                for index in range(stage_count)
            },
            "blockers": [] if ok else ["coordinator_stage_task_counts_incomplete", "cpu_stage_kernel_not_accepted"],
            "diagnosis_codes": ["kaggle_colab_gpu_cpu_topology_ready" if ok else "kaggle_colab_gpu_cpu_topology_not_ready"],
            "kaggle_lifecycle": {
                "requested_topology": "2KaggleGPU_stages_1ColabGPU_stages_5CPU_stages",
                "actual_gpu_push_count": 1,
                "actual_colab_gpu_runtime_count": 1 if completed >= 3 else 0,
                "actual_cpu_push_count": max(0, completed - 3),
                "kernels_deleted": True,
                "private_packages_removed": True,
            },
            "safety": {"public_artifact_safe": True},
        }

    def test_pack_and_check_keep_32b_max_when_72b_fails(self) -> None:
        base = self._tmp()
        baseline = self._write(base / "32b.json", self._probe(parameter=32, ok=True, stage_count=8, completed=8))
        attempt = self._write(base / "72b.json", self._probe(parameter=72, ok=False, stage_count=12, completed=8))

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "out"),
            "--baseline-32b-report",
            str(baseline),
            "--attempt-report",
            str(attempt),
        ]))

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "32b")
        self.assertEqual(report["max_attempted_parameter_class"], "72b")
        self.assertEqual(report["failure_stage"], "kaggle_kernel_acceptance")
        self.assertEqual(check.validate_report(report), [])

    def test_checker_rejects_larger_overclaim(self) -> None:
        base = self._tmp()
        baseline = self._write(base / "32b.json", self._probe(parameter=32, ok=True, stage_count=8, completed=8))
        attempt = self._write(base / "72b.json", self._probe(parameter=72, ok=False, stage_count=12, completed=8))
        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "out"),
            "--baseline-32b-report",
            str(baseline),
            "--attempt-report",
            str(attempt),
        ]))
        report["max_successful_same_request_decode_parameter_class"] = "72b"

        self.assertIn("max_successful_decode_mismatch", check.validate_report(report))

    def test_pack_accepts_recovered_72b_runtime_evidence(self) -> None:
        base = self._tmp()
        baseline = self._write(base / "32b.json", self._probe(parameter=32, ok=True, stage_count=8, completed=8))
        recovered = self._write(
            base / "72b_recovered.json",
            {
                "schema": "kaggle_colab_gpu_cpu_72b_recovered_runtime_evidence_v1",
                "ok": True,
                "public_artifact_safe": True,
                "kaggle_cleanup_verified": True,
                "model": {
                    "repo": "Qwen/Qwen2.5-72B-Instruct",
                    "parameter_count_b": 72,
                    "quantization": "none",
                    "stage_ranges": [[0, 2], [2, 8], [8, 14], [14, 20], [20, 35], [35, 50], [50, 65], [65, 80]],
                    "full_layer_coverage_verified": True,
                },
                "topology": {
                    "kaggle_gpu_stage_ids": [1, 2],
                    "colab_gpu_stage_ids": [3],
                    "kaggle_cpu_stage_ids": [0, 4, 5, 6, 7],
                    "accepted_provider_families": ["kaggle_cuda", "colab_cuda", "cpu"],
                },
                "coordinator": {
                    "ready": True,
                    "completed_task_count": 8,
                    "generated_token_count": 1,
                    "stage_seen": list(range(8)),
                    "stage_task_counts": {f"stage{index}": 1 for index in range(8)},
                    "kv_cache_ready": True,
                    "activation_hash_count": 7,
                    "generated_token_hash_count": 1,
                    "raw_prompt_public": False,
                    "generated_token_ids_public": False,
                    "activation_public": False,
                    "hidden_state_public": False,
                },
                "raw_main_report_written": False,
                "raw_main_report_blocker": "colab_nonfinal_worker_waited_for_global_ready_before_fix",
                "credentials_public": False,
                "private_runtime_state_public": False,
            },
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "out"),
            "--baseline-32b-report",
            str(baseline),
            "--attempt-report",
            str(recovered),
        ]))

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "72b")
        self.assertEqual(report["max_attempted_parameter_class"], "72b")
        self.assertEqual(check.validate_report(report), [])

    def test_pack_rejects_larger_partial_layer_success_claim(self) -> None:
        base = self._tmp()
        baseline = self._write(base / "32b.json", self._probe(parameter=32, ok=True, stage_count=8, completed=8))
        success_72b = self._write(base / "72b.json", self._probe(parameter=72, ok=True, stage_count=10, completed=10))
        attempt = self._probe(parameter=176, ok=True, stage_count=10, completed=10)
        attempt["model"]["repo"] = "bigscience/bloom"
        attempt["model"]["full_layer_coverage_verified"] = False
        attempt["blockers"] = ["larger_model_full_layer_coverage_not_verified"]
        attempt_path = self._write(base / "176b_partial.json", attempt)

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "out"),
            "--baseline-32b-report",
            str(baseline),
            "--attempt-report",
            str(success_72b),
            "--attempt-report",
            str(attempt_path),
        ]))

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "72b")
        self.assertEqual(report["max_successful_dense_full_precision_parameter_class"], "72b")
        self.assertEqual(report["max_successful_moe_total_parameter_class"], "")
        self.assertEqual(report["max_successful_moe_activated_parameter_class"], "")
        self.assertEqual(report["max_attempted_parameter_class"], "176b")
        self.assertEqual(check.validate_report(report), [])

    def test_pack_records_alternate_dense_and_moe_blockers_without_overclaim(self) -> None:
        base = self._tmp()
        baseline = self._write(base / "32b.json", self._probe(parameter=32, ok=True, stage_count=8, completed=8))
        success_72b = self._write(base / "72b.json", self._probe(parameter=72, ok=True, stage_count=10, completed=10))
        dense_405 = self._write(
            base / "405b_blocker.json",
            {
                "schema": "kaggle_colab_gpu_cpu_large_model_blocker_v1",
                "ok": False,
                "public_artifact_safe": True,
                "model": {
                    "repo": "meta-llama/Meta-Llama-3.1-405B",
                    "parameter_class": "405b",
                    "parameter_count_b": 405,
                    "architecture_class": "dense",
                    "quantization": "none",
                    "precision": "bf16_or_fp16_stage_runtime",
                    "stage_count": 32,
                    "stage_ranges": [[0, 1]],
                    "expected_layer_count": 126,
                    "full_layer_coverage_verified": False,
                },
                "accepted_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
                "provider_stage_counts": {"kaggle_cuda": 0, "colab_cuda": 0, "cpu": 0, "web_tpu": 0},
                "generated_token_count": 0,
                "coordinator": {"generated_token_count": 0},
                "stage_task_counts": {},
                "blockers": ["kaggle_model_license_agreement_required"],
                "failure_stage": "model_source_license_gated",
                "kaggle_lifecycle": {
                    "requested_topology": "source_discovery_only",
                    "actual_gpu_push_count": 0,
                    "actual_colab_gpu_runtime_count": 0,
                    "actual_cpu_push_count": 0,
                    "kernels_deleted": True,
                    "private_packages_removed": True,
                },
                "source_evidence": {
                    "model_source_refs": ["metaresearch/llama-3.1/Transformers/405b/1"],
                    "source_candidate": {"kaggle_kernel_model_source": "metaresearch/llama-3.1/Transformers/405b/1"},
                },
            },
        )
        moe_235 = self._write(
            base / "235b_moe_blocker.json",
            {
                "schema": "kaggle_colab_gpu_cpu_large_model_blocker_v1",
                "ok": False,
                "public_artifact_safe": True,
                "model": {
                    "repo": "Qwen/Qwen3-235B-A22B",
                    "parameter_class": "235b-a22b",
                    "parameter_count_b": 235,
                    "architecture_class": "moe",
                    "moe_total_parameter_count_b": 235,
                    "moe_active_parameter_count_b": 22,
                    "quantization": "none",
                    "precision": "bf16_or_fp16_stage_runtime",
                    "stage_count": 24,
                    "stage_ranges": [[0, 1]],
                    "expected_layer_count": 94,
                    "full_layer_coverage_verified": False,
                },
                "accepted_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
                "provider_stage_counts": {"kaggle_cuda": 0, "colab_cuda": 0, "cpu": 0, "web_tpu": 0},
                "generated_token_count": 0,
                "coordinator": {"generated_token_count": 0},
                "stage_task_counts": {},
                "blockers": ["qwen3_moe_or_hybrid_adapter_required"],
                "failure_stage": "moe_adapter_not_verified",
                "kaggle_lifecycle": {
                    "requested_topology": "source_discovery_stage_plan_only",
                    "actual_gpu_push_count": 0,
                    "actual_colab_gpu_runtime_count": 0,
                    "actual_cpu_push_count": 0,
                    "kernels_deleted": True,
                    "private_packages_removed": True,
                },
                "source_evidence": {
                    "model_source_refs": ["qwen-lm/qwen-3/Transformers/235b-a22b/1"],
                    "source_candidate": {"kaggle_kernel_model_source": "qwen-lm/qwen-3/Transformers/235b-a22b/1"},
                },
            },
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "out"),
            "--baseline-32b-report",
            str(baseline),
            "--attempt-report",
            str(success_72b),
            "--attempt-report",
            str(dense_405),
            "--attempt-report",
            str(moe_235),
        ]))

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "72b")
        self.assertEqual(report["max_successful_dense_full_precision_parameter_class"], "72b")
        self.assertEqual(report["max_attempted_parameter_class"], "405b")
        self.assertEqual(report["max_successful_moe_total_parameter_class"], "")
        self.assertIn("metaresearch/llama-3.1/Transformers/405b/1", report["model_source_refs"])
        self.assertIn("qwen-lm/qwen-3/Transformers/235b-a22b/1", report["model_source_refs"])
        self.assertEqual(check.validate_report(report), [])

    def test_pack_keeps_671b_moe_attempt_out_of_dense_success(self) -> None:
        base = self._tmp()
        baseline = self._write(base / "32b.json", self._probe(parameter=32, ok=True, stage_count=8, completed=8))
        success_72b = self._write(base / "72b.json", self._probe(parameter=72, ok=True, stage_count=10, completed=10))
        moe_671 = self._write(
            base / "671b_moe_blocker.json",
            {
                "schema": "kaggle_colab_gpu_cpu_large_model_blocker_v1",
                "ok": False,
                "public_artifact_safe": True,
                "model": {
                    "repo": "deepseek-ai/DeepSeek-V3",
                    "parameter_class": "671b-v3",
                    "parameter_count_b": 671,
                    "architecture_class": "moe",
                    "moe_total_parameter_count_b": 671,
                    "moe_active_parameter_count_b": 37,
                    "quantization": "none",
                    "precision": "bf16_or_fp16_stage_runtime",
                    "stage_count": 16,
                    "stage_ranges": [[0, 1]],
                    "expected_layer_count": 61,
                    "full_layer_coverage_verified": False,
                },
                "accepted_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
                "provider_stage_counts": {"kaggle_cuda": 0, "colab_cuda": 0, "cpu": 0, "web_tpu": 0},
                "generated_token_count": 0,
                "coordinator": {"generated_token_count": 0},
                "stage_task_counts": {},
                "blockers": ["candidate_not_full_precision_bf16", "deepseek_mla_moe_adapter_required"],
                "failure_stage": "deepseek_fp8_not_full_precision_and_adapter_missing",
                "kaggle_lifecycle": {
                    "requested_topology": "source_discovery_only",
                    "actual_gpu_push_count": 0,
                    "actual_colab_gpu_runtime_count": 0,
                    "actual_cpu_push_count": 0,
                    "kernels_deleted": True,
                    "private_packages_removed": True,
                },
                "source_evidence": {
                    "model_source_refs": ["deepseek-ai/deepseek-v3/Transformers/deepseek-v3/2"],
                    "source_candidate": {"kaggle_kernel_model_source": "deepseek-ai/deepseek-v3/Transformers/deepseek-v3/2"},
                },
            },
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "out"),
            "--baseline-32b-report",
            str(baseline),
            "--attempt-report",
            str(success_72b),
            "--attempt-report",
            str(moe_671),
        ]))

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "72b")
        self.assertEqual(report["max_successful_dense_full_precision_parameter_class"], "72b")
        self.assertEqual(report["max_attempted_parameter_class"], "671b")
        self.assertEqual(report["max_successful_moe_total_parameter_class"], "")
        self.assertEqual(report["conclusions"]["max_attempted_dense_full_precision_parameter_class"], "72b")
        self.assertEqual(check.validate_report(report), [])

    def test_large_model_blocker_preserves_stage_plan_without_decode_success(self) -> None:
        base = self._tmp()
        source = self._write(
            base / "source.json",
            {
                "schema": "kaggle_alternate_llm_source_resolver_v1",
                "ok": True,
                "model_source_refs": ["qwen-lm/qwen-3/Transformers/235b-a22b/1"],
                "candidates": [
                    {
                        "parameter_class": "235b-a22b",
                        "architecture_class": "moe",
                        "active_params_b": 22,
                        "license_agreement_required": False,
                    }
                ],
            },
        )
        attach = self._write(
            base / "attach.json",
            {
                "schema": "kaggle_model_attach_probe_v1",
                "ok": True,
                "kaggle_model_attach_probe_ready": True,
                "stage_owned_preflight_verified": True,
                "model_source": "qwen-lm/qwen-3/Transformers/235b-a22b/1",
                "resolved_attached_path": "/kaggle/input/models/qwen-lm/qwen-3/transformers/235b-a22b/1",
                "blocker_codes": [],
                "runtime_report": {
                    "model_type": "qwen3_moe",
                    "num_hidden_layers": 94,
                    "stage_plan": {
                        "stage_count": 24,
                        "assigned_key_count_total": 36945,
                        "present_key_count_total": 36945,
                        "assigned_file_count_total": 118,
                        "total_planned_logical_tensor_gb": 470.187269,
                        "max_stage_planned_logical_tensor_gb": 21.1467,
                        "stage_owned_preflight_verified": True,
                    },
                },
            },
        )

        report = blocker_pack.build_report(blocker_pack.argparse.Namespace(
            output_dir=str(base / "out"),
            model_repo="Qwen/Qwen3-235B-A22B",
            parameter_class="235b-a22b",
            parameter_count_b=235,
            active_parameter_count_b=22,
            architecture_class="moe",
            expected_layer_count=94,
            stage_count=24,
            stage_ranges_json=json.dumps([[0, 4], [4, 8]]),
            requested_topology="source_attach_stage_plan_only",
            source_resolver_report=str(source),
            attach_probe_report=[str(attach)],
            stage_loader_report=[],
            blocker=["qwen3_moe_or_hybrid_adapter_required"],
            failure_stage="qwen3_moe_same_request_runtime_adapter_not_verified",
            json=False,
        ))

        self.assertFalse(report["same_request_decode_verified"])
        self.assertTrue(report["stage_owned_preflight_verified"])
        self.assertIn("alternate_llm_stage_plan_evidence_present", report["diagnosis_codes"])
        attach_summary = report["source_evidence"]["attach_probe_reports"][0]
        self.assertTrue(attach_summary["stage_owned_preflight_verified"])
        self.assertEqual(attach_summary["stage_plan"]["total_planned_logical_tensor_gb"], 470.187269)
        self.assertIn("moe_same_request_runtime_adapter_not_verified", report["blockers"])

    def test_pack_accepts_176b_source_blocker_attempt(self) -> None:
        base = self._tmp()
        baseline = self._write(base / "32b.json", self._probe(parameter=32, ok=True, stage_count=8, completed=8))
        success_72b = self._write(base / "72b.json", self._probe(parameter=72, ok=True, stage_count=10, completed=10))
        blocker = self._write(
            base / "176b_blocker.json",
            {
                "schema": "kaggle_colab_gpu_cpu_large_model_blocker_v1",
                "ok": False,
                "public_artifact_safe": True,
                "model": {
                    "repo": "bigscience/bloom",
                    "parameter_count_b": 176,
                    "quantization": "none",
                    "precision": "bf16_or_fp16_stage_runtime",
                    "stage_count": 21,
                    "stage_ranges": [[0, 2], [2, 4]],
                    "expected_layer_count": 70,
                    "full_layer_coverage_verified": False,
                },
                "accepted_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
                "provider_stage_counts": {"kaggle_cuda": 0, "colab_cuda": 0, "cpu": 0, "web_tpu": 0},
                "generated_token_count": 0,
                "coordinator": {"generated_token_count": 0},
                "stage_task_counts": {},
                "blockers": ["kaggle_model_attach_176b_transformers_source_unavailable"],
                "failure_stage": "model_source_attach_unavailable",
                "kaggle_lifecycle": {
                    "requested_topology": "5GPU_stages_16CPU_stages_planned_after_attach",
                    "actual_gpu_push_count": 0,
                    "actual_colab_gpu_runtime_count": 0,
                    "actual_cpu_push_count": 0,
                    "kernels_deleted": True,
                    "private_packages_removed": True,
                },
                "source_evidence": {"kaggle_mcp_observation": {"bloom_176b_transformers_model_present": False}},
            },
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "out"),
            "--baseline-32b-report",
            str(baseline),
            "--attempt-report",
            str(success_72b),
            "--attempt-report",
            str(blocker),
        ]))

        self.assertEqual(report["max_successful_same_request_decode_parameter_class"], "72b")
        self.assertEqual(report["max_attempted_parameter_class"], "176b")
        self.assertEqual(report["failure_stage"], "model_source_attach_unavailable")
        self.assertIn("kaggle_model_attach_176b_transformers_source_unavailable", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])


if __name__ == "__main__":
    unittest.main()
