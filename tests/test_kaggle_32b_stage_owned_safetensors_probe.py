from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest import mock

from scripts import kaggle_32b_stage_owned_safetensors_probe as probe


class Kaggle32BStageOwnedSafetensorsProbeTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            model_repo="Qwen/Qwen2.5-32B-Instruct-AWQ",
            stage_count=2,
            accelerator="NvidiaTeslaT4",
        )

    def test_local_expected_plan_covers_awq_32b_weight_index(self) -> None:
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 4,
            "hidden_size": 16,
        }
        weight_index = {
            "metadata": {"total_size": 1234},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00003.safetensors",
                "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00003.safetensors",
                "model.layers.3.self_attn.q_proj.weight": "model-00003-of-00003.safetensors",
                "model.norm.weight": "model-00003-of-00003.safetensors",
                "lm_head.weight": "model-00003-of-00003.safetensors",
            },
        }

        with mock.patch.object(probe, "fetch_hf_json", side_effect=[config, weight_index]):
            plan = probe.local_expected_plan(self._args(), stage_ids=[0, 1])

        self.assertTrue(plan["available"])
        self.assertTrue(plan["requested_stages_cover_all_weight_keys"])
        self.assertEqual(plan["covered_weight_key_count"], 7)
        self.assertEqual(plan["all_weight_file_count"], 3)
        self.assertEqual(plan["stage_plans"][0]["assigned_weight_file_count"], 2)
        self.assertEqual(plan["stage_plans"][1]["assigned_weight_file_count"], 2)
        self.assertEqual(plan["stage_plans"][0]["shared_boundary_file_count"], 1)
        self.assertEqual(plan["stage_plans"][1]["shared_boundary_file_count"], 1)

    def test_build_report_requires_stage_owned_loading_cleanup_and_coverage(self) -> None:
        args = self._args()
        stage_reports = [
            {
                "schema": probe.STAGE_REPORT_SCHEMA,
                "ok": True,
                "stage_owned_quantized_32b_loading_ready": True,
                "stage_id": 0,
                "stage_count": 2,
                "stage_layer_range": [0, 2],
                "assigned_weight_key_count": 3,
                "assigned_weight_file_count": 2,
                "assigned_weight_files": ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors"],
                "downloads": [{"filename": "model-00001-of-00003.safetensors"}],
                "loaded_weight_key_count": 3,
                "loaded_tensor_gb": 8.0,
                "materialized_tensor_gb": 8.0,
                "materialized_weight_key_count": 3,
                "materialize_clone_requested": True,
                "retained_tensor_gb": 8.0,
                "loads_only_stage_weight_keys": True,
                "cross_stage_weight_keys_loaded": False,
                "downloads_all_model_weight_files": False,
                "stage_weight_downloads_only_stage_files": True,
                "shared_boundary_file_count": 1,
                "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"]},
                "diagnosis_codes": ["kaggle_32b_stage_owned_tensor_load_ready"],
                "blockers": [],
            },
            {
                "schema": probe.STAGE_REPORT_SCHEMA,
                "ok": True,
                "stage_owned_quantized_32b_loading_ready": True,
                "stage_id": 1,
                "stage_count": 2,
                "stage_layer_range": [2, 4],
                "assigned_weight_key_count": 4,
                "assigned_weight_file_count": 2,
                "assigned_weight_files": ["model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"],
                "downloads": [{"filename": "model-00003-of-00003.safetensors"}],
                "loaded_weight_key_count": 4,
                "loaded_tensor_gb": 8.2,
                "materialized_tensor_gb": 8.2,
                "materialized_weight_key_count": 4,
                "materialize_clone_requested": True,
                "retained_tensor_gb": 8.2,
                "loads_only_stage_weight_keys": True,
                "cross_stage_weight_keys_loaded": False,
                "downloads_all_model_weight_files": False,
                "stage_weight_downloads_only_stage_files": True,
                "shared_boundary_file_count": 1,
                "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"]},
                "diagnosis_codes": ["kaggle_32b_stage_owned_tensor_load_ready"],
                "blockers": [],
            },
        ]
        stage_runs = [
            {
                "stage_id": 0,
                "kernel_ref": "owner/stage0",
                "steps": [
                    {"name": "kaggle_kernel_push", "ok": True},
                    {"name": "kaggle_kernel_delete", "ok": True},
                ],
            },
            {
                "stage_id": 1,
                "kernel_ref": "owner/stage1",
                "steps": [
                    {"name": "kaggle_kernel_push", "ok": True},
                    {"name": "kaggle_kernel_delete", "ok": True},
                ],
            },
        ]
        expected_plan = {
            "available": True,
            "requested_stages_cover_all_weight_keys": True,
            "stage_plans": [],
            "public_safe": True,
        }

        with mock.patch.object(probe, "local_expected_plan", return_value=expected_plan):
            report = probe.build_report(
                args,
                output_dir=Path("/tmp/crowdtensor-test-output"),
                stage_ids=[0, 1],
                packages=[],
                stage_runs=stage_runs,
                stage_reports=stage_reports,
            )

        self.assertTrue(report["ok"])
        self.assertTrue(report["stage_owned_quantized_32b_loading_ready"])
        self.assertTrue(report["all_stage_owned_loading_ready"])
        self.assertTrue(report["stage_owned_download_scope_ready"])
        self.assertTrue(report["loads_only_stage_weight_keys_ready"])
        self.assertTrue(report["coverage_ready"])
        self.assertIn("kaggle_32b_stage_owned_tensor_load_ready", report["diagnosis_codes"])
        self.assertIn("kaggle_kernels_deleted", report["diagnosis_codes"])
        self.assertEqual(report["blockers"], [])

    def test_build_report_rejects_cross_stage_key_load(self) -> None:
        args = self._args()
        stage_report = {
            "schema": probe.STAGE_REPORT_SCHEMA,
            "ok": False,
            "stage_owned_quantized_32b_loading_ready": False,
            "stage_id": 0,
            "stage_weight_downloads_only_stage_files": True,
            "loads_only_stage_weight_keys": False,
            "cross_stage_weight_keys_loaded": True,
            "hardware": {"kaggle_gpu_verified": True},
        }
        with mock.patch.object(probe, "local_expected_plan", return_value={"requested_stages_cover_all_weight_keys": True}):
            report = probe.build_report(
                args,
                output_dir=Path("/tmp/crowdtensor-test-output"),
                stage_ids=[0, 1],
                packages=[],
                stage_runs=[],
                stage_reports=[stage_report],
            )

        self.assertFalse(report["ok"])
        self.assertFalse(report["loads_only_stage_weight_keys_ready"])
        self.assertIn("stage_owned_tensor_key_load_not_ready", report["blockers"])

    def test_render_kernel_uses_python_bool_literals(self) -> None:
        args = argparse.Namespace(
            model_repo="Qwen/Qwen2.5-32B-Instruct-AWQ",
            stage_count=2,
            retain_tensors=True,
            materialize_clone=True,
            retain_limit_gb=24,
        )

        rendered = probe.render_kernel(args, stage_id=0)

        self.assertIn("RETAIN_TENSORS = True", rendered)
        self.assertIn("MATERIALIZE_CLONE = True", rendered)
        self.assertNotIn("RETAIN_TENSORS = true", rendered)
        self.assertNotIn("MATERIALIZE_CLONE = true", rendered)


if __name__ == "__main__":
    unittest.main()
