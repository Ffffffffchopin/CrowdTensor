from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import kaggle_dense_model_source_resolver as resolver
from scripts import three_accelerator_dense_qwen_frontier_check as check
from scripts import three_accelerator_dense_qwen_frontier_pack as pack


class ThreeAcceleratorDenseQwenFrontierTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_dense_frontier_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _bridge_report(self, base: Path) -> Path:
        return self._write_json(
            base / "bridge.json",
            {
                "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
                "ok": True,
                "same_request_runtime_bridge_verified": True,
                "gpu_tpu_cpu_32b_same_request_verified": True,
                "same_request_32b_model_verified": True,
                "target_model_id": "Qwen/Qwen2.5-32B-Instruct",
                "generated_token_count": 4,
                "accepted_stage_backends": ["cuda", "jax_tpu", "cpu"],
                "stage_local_kv_cache_verified": True,
                "public_artifact_safe": True,
            },
        )

    def _gpu_cpu_fallback_report(self, base: Path) -> Path:
        return self._write_json(
            base / "gpu_cpu_fallback.json",
            {
                "schema": "kaggle_32b_full_heterogeneous_probe_v1",
                "ok": True,
                "quantization": "none",
                "full_precision_32b": True,
                "four_t4_five_cpu_topology_verified": True,
                "stage_owned_full_precision_runtime_verified": True,
                "multi_token_generation_verified": True,
                "stage_local_kv_cache_verified": True,
                "generated_token_count": 2,
                "model": {
                    "repo": "Qwen/Qwen2.5-32B-Instruct",
                    "parameter_count_b": 32,
                    "stage_count": 9,
                    "quantization": "none",
                },
                "stage_summaries": [
                    {"stage_id": 0, "resource_kind": "gpu", "ok": True},
                    {"stage_id": 1, "resource_kind": "gpu", "ok": True},
                    {"stage_id": 4, "resource_kind": "cpu", "ok": True},
                ],
                "stage_task_counts": {"stage0": 2, "stage1": 2, "stage4": 2},
                "kaggle_lifecycle": {
                    "kernels_deleted": True,
                    "private_packages_removed": True,
                },
                "safety": {"public_artifact_safe": True},
            },
        )

    def _tpu_loader_report(self, base: Path) -> Path:
        return self._write_json(
            base / "tpu_loader.json",
            {
                "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
                "ok": True,
                "model_repo": "Qwen/Qwen2.5-32B-Instruct",
                "full_stage_owned_tpu_loader_ready": True,
                "tpu_32b_runtime_adapter_ready": True,
                "stage_owned_header_verified": True,
                "partial_tensor_to_tpu_verified": True,
                "stage_local_kv_cache_verified": True,
                "stage_layer_range": [21, 42],
                "executed_layer_count": 21,
                "full_stage_layer_count": 21,
                "loaded_execution_tensor_key_count": 252,
                "loaded_execution_tensor_gb": 19.072947,
                "tpu_device_count": 8,
                "tpu_device_kind": "TPU v5 lite",
                "kaggle_lifecycle": {
                    "kernels_deleted": True,
                    "private_packages_removed": True,
                },
                "safety": {"public_artifact_safe": True},
                "public_artifact_safe": True,
            },
        )

    def _attach_probe_report(self, base: Path) -> Path:
        return self._write_json(
            base / "attach_probe.json",
            {
                "schema": "kaggle_model_attach_probe_v1",
                "ok": True,
                "kaggle_model_attach_probe_ready": True,
                "kaggle_model_attach_used": True,
                "parameter_class": "7b",
                "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
                "model_source": "qwen-lm/qwen2.5/Transformers/7b-instruct/1",
                "expected_attached_path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
                "runtime_report": {
                    "ok": True,
                    "path_present": True,
                    "config_json_present": True,
                    "weight_index_present": True,
                    "tokenizer_json_present": True,
                    "safetensors_file_count": 4,
                    "weight_index_key_count": 339,
                    "weight_index_file_count": 4,
                    "model_type": "qwen2",
                    "torch_dtype": "bfloat16",
                    "quantization_config_present": False,
                },
                "cleanup_status": {
                    "temporary_kaggle_kernel_deleted": True,
                    "temporary_private_package_removed": True,
                    "live_resources_left_running": False,
                },
                "blocker_codes": [],
                "public_artifact_safe": True,
            },
        )

    def _attach_probe_72b_report(self, base: Path) -> Path:
        payload = json.loads(self._attach_probe_report(base).read_text(encoding="utf-8"))
        payload["parameter_class"] = "72b"
        payload["hf_repo"] = "Qwen/Qwen2.5-72B-Instruct"
        payload["model_source"] = "qwen-lm/qwen2.5/Transformers/72b-instruct/1"
        payload["expected_attached_path"] = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/72b-instruct/1"
        payload["runtime_report"]["expected_attached_path"] = payload["expected_attached_path"]
        payload["runtime_report"]["safetensors_file_count"] = 37
        payload["runtime_report"]["weight_index_key_count"] = 963
        payload["runtime_report"]["weight_index_file_count"] = 37
        payload["runtime_report"]["hidden_size"] = 8192
        payload["runtime_report"]["num_hidden_layers"] = 80
        return self._write_json(base / "attach_probe_72b.json", payload)

    def _attach_probe_72b_stage_plan_report(self, base: Path) -> Path:
        payload = json.loads(self._attach_probe_72b_report(base).read_text(encoding="utf-8"))
        payload["stage_plan_requested"] = True
        payload["stage_owned_preflight_verified"] = True
        stage_plans = []
        for stage_id in range(10):
            stage_plans.append(
                {
                    "stage_id": stage_id,
                    "backend": ["cuda", "cuda", "cuda", "cuda", "jax_tpu", "cpu", "cpu", "cpu", "cpu", "cpu"][stage_id],
                    "layer_range": [stage_id * 8, (stage_id + 1) * 8],
                    "assigned_key_count": 96,
                    "present_key_count": 96,
                    "missing_key_count": 0,
                    "assigned_file_count": 5,
                    "logical_tensor_gb": 14.5,
                    "stage_owned_header_verified": True,
                }
            )
        payload["runtime_report"]["stage_plan_enabled"] = True
        payload["runtime_report"]["stage_owned_preflight_verified"] = True
        payload["runtime_report"]["stage_plan"] = {
            "schema": "kaggle_model_attach_stage_plan_v1",
            "enabled": True,
            "stage_count": 10,
            "stage_backends": ["cuda", "cuda", "cuda", "cuda", "jax_tpu", "cpu", "cpu", "cpu", "cpu", "cpu"],
            "num_hidden_layers": 80,
            "hidden_size": 8192,
            "assigned_key_count_total": 960,
            "present_key_count_total": 960,
            "assigned_file_count_total": 37,
            "total_planned_logical_tensor_gb": 145.0,
            "max_stage_planned_logical_tensor_gb": 14.5,
            "stage_owned_preflight_verified": True,
            "stage_plans": stage_plans,
            "weight_tensor_values_public": False,
            "public_artifact_safe": True,
        }
        return self._write_json(base / "attach_probe_72b_stage_plan.json", payload)

    def _fake_hf(self, model_repo: str, filename: str, *, timeout_seconds: float = 90.0) -> dict:
        if filename == "config.json":
            size = 80 if "72B" in model_repo else 64 if "32B" in model_repo else 48 if "14B" in model_repo else 28
            hidden = 8192 if "72B" in model_repo else 5120 if ("32B" in model_repo or "14B" in model_repo) else 3584
            return {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": hidden,
                "intermediate_size": 176,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_hidden_layers": size,
                "vocab_size": 32000,
                "torch_dtype": "bfloat16",
            }
        return {
            "metadata": {"total_size": 1024},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            },
        }

    def _build_report(self, base: Path) -> dict:
        bridge = self._bridge_report(base)
        gpu_cpu_fallback = self._gpu_cpu_fallback_report(base)
        tpu_loader = self._tpu_loader_report(base)
        with mock.patch.object(resolver, "fetch_hf_json", side_effect=self._fake_hf):
            return pack.build_report(
                pack.parse_args([
                    "--output-dir",
                    str(base / "frontier"),
                    "--baseline-32b-bridge-report",
                    str(bridge),
                    "--gpu-cpu-dense-fallback-report",
                    str(gpu_cpu_fallback),
                    "--tpu-dense-loader-report",
                    str(tpu_loader),
                    "--fetch-hf-metadata",
                ])
            )

    def _adapter_report(self, base: Path) -> Path:
        return self._write_json(
            base / "adapter.json",
            {
                "schema": "qwen_dense_jax_tpu_stage_adapter_smoke_v1",
                "ok": True,
                "torch_reference_forward_ready": True,
                "jax_runtime_execution_ready": True,
                "tpu_runtime_ready": False,
                "tpu_jax_qwen_stage_runtime_ready": False,
                "stage_local_kv_cache_verified": True,
                "dense_full_precision_only": True,
                "quantized_weight_adapter_used": False,
                "shape_metadata": {
                    "input_shape": [1, 4, 64],
                    "output_shape": [1, 4, 64],
                    "activation_payload_public": False,
                },
                "qwen_components_exercised": {
                    "rms_norm": True,
                    "rope": True,
                    "grouped_query_attention": True,
                    "causal_attention": True,
                    "swiglu_mlp": True,
                    "stage_local_kv_cache": True,
                },
                "blockers": [],
                "public_artifact_safe": True,
            },
        )

    def test_frontier_records_dense_72b_attempt_but_only_32b_decode(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["three_accelerator_dense_qwen_frontier_ready"])
        self.assertEqual(report["largest_dense_model_attempted"], "72b")
        self.assertEqual(report["largest_dense_model_attach_candidate"], "72b")
        self.assertEqual(report["largest_dense_model_loaded"], "32b")
        self.assertEqual(report["largest_dense_model_1token_decoded"], "32b")
        self.assertTrue(report["all_three_accelerators_same_request_verified"])
        self.assertTrue(report["same_request_dense_32b_success"])
        self.assertFalse(report["same_request_dense_frontier_success"])
        self.assertTrue(report["baseline_32b_gpu_cpu_dense_fallback"]["gpu_cpu_dense_fallback_verified"])
        self.assertTrue(report["gpu_stage_runtime_ready"])
        self.assertTrue(report["cpu_stage_runtime_ready"])
        self.assertTrue(report["tpu_jax_qwen_stage_runtime_ready"])
        self.assertTrue(report["retained_tpu_dense_qwen_stage"]["tpu_jax_qwen_stage_runtime_ready"])
        self.assertIn("larger_than_32b_dense_decode_not_verified", report["blocker_codes"])
        self.assertIn("kaggle_model_attach_not_mounted_in_current_runtime", report["blocker_codes"])
        self.assertNotIn("tpu_dense_qwen_jax_stage_runtime_not_verified", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_frontier_can_import_cpu_jax_adapter_without_claiming_tpu(self) -> None:
        base = self._tmp_dir()
        bridge = self._bridge_report(base)
        gpu_cpu_fallback = self._gpu_cpu_fallback_report(base)
        tpu_loader = self._write_json(base / "missing_tpu_loader.json", {})
        adapter = self._adapter_report(base)
        with mock.patch.object(resolver, "fetch_hf_json", side_effect=self._fake_hf):
            report = pack.build_report(
                pack.parse_args([
                    "--output-dir",
                    str(base / "frontier-import"),
                    "--baseline-32b-bridge-report",
                    str(bridge),
                    "--gpu-cpu-dense-fallback-report",
                    str(gpu_cpu_fallback),
                    "--tpu-dense-loader-report",
                    str(tpu_loader),
                    "--dense-adapter-report",
                    str(adapter),
                    "--fetch-hf-metadata",
                ])
            )

        self.assertTrue(report["tpu_dense_qwen_adapter"]["jax_runtime_execution_ready"])
        self.assertFalse(report["tpu_jax_qwen_stage_runtime_ready"])
        self.assertIn("tpu_dense_qwen_jax_stage_runtime_not_verified", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_frontier_can_import_live_kaggle_attach_probe(self) -> None:
        base = self._tmp_dir()
        bridge = self._bridge_report(base)
        gpu_cpu_fallback = self._gpu_cpu_fallback_report(base)
        tpu_loader = self._tpu_loader_report(base)
        attach_probe = self._attach_probe_report(base)
        with mock.patch.object(resolver, "fetch_hf_json", side_effect=self._fake_hf):
            report = pack.build_report(
                pack.parse_args([
                    "--output-dir",
                    str(base / "frontier-attach"),
                    "--baseline-32b-bridge-report",
                    str(bridge),
                    "--gpu-cpu-dense-fallback-report",
                    str(gpu_cpu_fallback),
                    "--tpu-dense-loader-report",
                    str(tpu_loader),
                    "--kaggle-model-attach-probe-report",
                    str(attach_probe),
                    "--fetch-hf-metadata",
                ])
            )

        self.assertTrue(report["kaggle_model_attach_used"])
        self.assertTrue(report["kaggle_model_attach_probe"]["kaggle_model_attach_probe_ready"])
        self.assertEqual(report["largest_dense_model_attached"], "7b")
        self.assertTrue(report["same_request_dense_32b_success"])
        self.assertFalse(report["same_request_dense_frontier_success"])
        self.assertNotIn("kaggle_model_attach_not_mounted_in_current_runtime", report["blocker_codes"])
        self.assertIn("larger_than_32b_dense_decode_not_verified", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_frontier_records_72b_attached_without_loaded_or_decode_overclaim(self) -> None:
        base = self._tmp_dir()
        bridge = self._bridge_report(base)
        gpu_cpu_fallback = self._gpu_cpu_fallback_report(base)
        tpu_loader = self._tpu_loader_report(base)
        attach_probe = self._attach_probe_72b_report(base)
        with mock.patch.object(resolver, "fetch_hf_json", side_effect=self._fake_hf):
            report = pack.build_report(
                pack.parse_args([
                    "--output-dir",
                    str(base / "frontier-attach-72b"),
                    "--baseline-32b-bridge-report",
                    str(bridge),
                    "--gpu-cpu-dense-fallback-report",
                    str(gpu_cpu_fallback),
                    "--tpu-dense-loader-report",
                    str(tpu_loader),
                    "--kaggle-model-attach-probe-report",
                    str(attach_probe),
                    "--fetch-hf-metadata",
                ])
            )

        self.assertEqual(report["largest_dense_model_attached"], "72b")
        self.assertEqual(report["largest_dense_model_stage_preflighted"], "")
        self.assertEqual(report["largest_dense_model_loaded"], "32b")
        self.assertEqual(report["largest_dense_model_1token_decoded"], "32b")
        self.assertFalse(report["same_request_dense_frontier_success"])
        self.assertEqual(report["frontier_failure_stage"], "larger_dense_same_request_decode_not_verified_after_model_attach")
        self.assertIn("larger_dense_same_request_decode_not_verified_after_model_attach", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_frontier_records_72b_stage_plan_without_loaded_or_decode_overclaim(self) -> None:
        base = self._tmp_dir()
        bridge = self._bridge_report(base)
        gpu_cpu_fallback = self._gpu_cpu_fallback_report(base)
        tpu_loader = self._tpu_loader_report(base)
        attach_probe = self._attach_probe_72b_stage_plan_report(base)
        with mock.patch.object(resolver, "fetch_hf_json", side_effect=self._fake_hf):
            report = pack.build_report(
                pack.parse_args([
                    "--output-dir",
                    str(base / "frontier-stage-plan-72b"),
                    "--baseline-32b-bridge-report",
                    str(bridge),
                    "--gpu-cpu-dense-fallback-report",
                    str(gpu_cpu_fallback),
                    "--tpu-dense-loader-report",
                    str(tpu_loader),
                    "--kaggle-model-attach-probe-report",
                    str(attach_probe),
                    "--fetch-hf-metadata",
                ])
            )

        self.assertEqual(report["largest_dense_model_attached"], "72b")
        self.assertEqual(report["largest_dense_model_stage_preflighted"], "72b")
        self.assertEqual(report["largest_dense_model_loaded"], "32b")
        self.assertEqual(report["largest_dense_model_1token_decoded"], "32b")
        self.assertFalse(report["same_request_dense_frontier_success"])
        self.assertEqual(report["frontier_failure_stage"], "larger_dense_live_stage_load_not_verified_after_stage_preflight")
        self.assertIn("larger_dense_live_stage_load_not_verified_after_stage_preflight", report["blocker_codes"])
        self.assertNotIn("larger_dense_same_request_decode_not_verified_after_model_attach", report["blocker_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_checker_rejects_larger_decode_overclaim(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)
        report["largest_dense_model_1token_decoded"] = "72b"

        errors = check.validate_report(report)

        self.assertIn("larger_dense_decode_claim_without_ladder_proof", errors)

    def test_checker_rejects_larger_loaded_overclaim(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)
        report["largest_dense_model_loaded"] = "72b"

        errors = check.validate_report(report)

        self.assertIn("larger_dense_loaded_claim_without_live_or_attached_proof", errors)

    def test_public_artifacts_are_redacted(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)

        self.assertEqual(pack.public_redaction_errors(report), [])
        scanned = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (base / "frontier").rglob("*")
            if path.is_file()
        )
        for fragment in [
            "KAGGLE_KEY",
            "HF_TOKEN",
            "Bearer ",
            "Cookie:",
            "kaggle-cookies",
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
