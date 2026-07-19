from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import heterogeneous_capacity_frontier_check as check
from scripts import heterogeneous_capacity_frontier_pack as pack


class HeterogeneousCapacityFrontierTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_capacity_frontier_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _baseline_reports(self, base: Path) -> tuple[Path, Path]:
        bridge = {
            "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
            "ok": True,
            "same_request_runtime_bridge_verified": True,
            "gpu_tpu_cpu_32b_same_request_verified": True,
            "same_request_32b_model_verified": True,
            "target_model_id": "Qwen/Qwen2.5-32B-Instruct",
            "generated_token_count": 4,
            "target_generated_token_count": 4,
            "accepted_stage_backends": ["cuda", "jax_tpu", "cpu"],
            "stage_task_counts": {"stage0": 4, "stage1": 4, "stage2": 4},
            "public_artifact_safe": True,
        }
        serving = {
            "schema": "heterogeneous_32b_serving_v1",
            "ok": True,
            "live_external_runtime_verified": True,
            "target_model_id": "Qwen/Qwen2.5-32B-Instruct",
            "generated_token_count": 4,
            "target_generated_token_count": 4,
            "public_artifact_safe": True,
        }
        return (
            self._write_json(base / "bridge.json", bridge),
            self._write_json(base / "serving.json", serving),
        )

    def _stage_load_report(self, base: Path) -> Path:
        payload = {
            "schema": "kaggle_32b_stage_owned_safetensors_probe_v1",
            "ok": True,
            "stage_owned_quantized_32b_loading_ready": True,
            "coverage_ready": True,
            "all_stage_reports_downloaded": True,
            "all_stage_owned_loading_ready": True,
            "loads_only_stage_weight_keys_ready": True,
            "model": {"repo": "Qwen/Qwen2.5-72B-Instruct-AWQ"},
            "runtime": {"stage_count": 6},
            "expected_plan": {
                "model_repo": "Qwen/Qwen2.5-72B-Instruct-AWQ",
                "stage_count": 6,
                "covered_weight_key_count": 138,
                "weight_key_count": 138,
            },
            "kaggle_lifecycle": {
                "actual_push_count": 6,
                "kernels_deleted": True,
                "private_packages_removed": True,
            },
            "stage_summaries": [
                {"stage_id": index, "loaded_tensor_gb": 3.5, "loads_only_stage_weight_keys": True}
                for index in range(6)
            ],
            "safety": {"public_artifact_safe": True},
        }
        return self._write_json(base / "stage-load-72b.json", payload)

    def _partial_stage_load_report(self, base: Path) -> Path:
        payload = {
            "schema": "kaggle_32b_stage_owned_safetensors_stage_probe_v1",
            "ok": True,
            "model_repo": "cyankiwi/Solar-Open-100B-AWQ-4bit",
            "stage_owned_quantized_32b_loading_ready": True,
            "stage_id": 8,
            "stage_count": 10,
            "stage_layer_range": [40, 44],
            "assigned_weight_key_count": 4684,
            "loaded_weight_key_count": 4684,
            "loaded_tensor_gb": 4.49,
            "materialized_tensor_gb": 4.49,
            "loads_only_stage_weight_keys": True,
            "cross_stage_weight_keys_loaded": False,
            "stage_weight_downloads_only_stage_files": True,
            "temp_cleanup": {"ok": True, "path_public": False},
            "safety": {"public_artifact_safe": True},
        }
        return self._write_json(base / "partial-stage-load-100b.json", payload)

    def _hf_fixture(self, model_id: str, filename: str) -> dict:
        if filename == "config.json":
            if "235B" in model_id:
                return {
                    "model_type": "qwen3_moe",
                    "architectures": ["Qwen3MoeForCausalLM"],
                    "num_hidden_layers": 6,
                    "hidden_size": 1024,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "num_experts": 128,
                    "vocab_size": 151936,
                    "quantization_config": {"quant_method": "awq", "bits": 4},
                }
            return {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 6,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "vocab_size": 152064,
                "quantization_config": {"quant_method": "awq", "bits": 4},
            }
        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "lm_head.weight": "model-00003-of-00003.safetensors",
            "model.norm.weight": "model-00003-of-00003.safetensors",
        }
        for layer in range(6):
            suffixes = [
                "input_layernorm.weight",
                "self_attn.q_proj.qweight",
                "self_attn.q_proj.qzeros",
                "self_attn.q_proj.scales",
                "self_attn.k_proj.qweight",
                "self_attn.k_proj.qzeros",
                "self_attn.k_proj.scales",
                "self_attn.v_proj.qweight",
                "self_attn.v_proj.qzeros",
                "self_attn.v_proj.scales",
                "self_attn.o_proj.qweight",
                "self_attn.o_proj.qzeros",
                "self_attn.o_proj.scales",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.qweight",
                "mlp.gate_proj.qzeros",
                "mlp.gate_proj.scales",
                "mlp.up_proj.qweight",
                "mlp.up_proj.qzeros",
                "mlp.up_proj.scales",
                "mlp.down_proj.qweight",
                "mlp.down_proj.qzeros",
                "mlp.down_proj.scales",
            ]
            for suffix in suffixes:
                shard = 1 + min(2, layer // 2)
                weight_map[f"model.layers.{layer}.{suffix}"] = f"model-{shard:05d}-of-00003.safetensors"
        return {
            "metadata": {"total_size": 123_000_000_000 if "235B" in model_id else 41_000_000_000},
            "weight_map": weight_map,
        }

    def _header_for_stage(self, model_id: str, index: dict, filename: str) -> tuple[int, dict]:
        header: dict[str, dict] = {}
        for key, value in index["weight_map"].items():
            if value != filename:
                continue
            header[key] = {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}
        return 128, header

    def _build_report(self, base: Path) -> dict:
        bridge, serving = self._baseline_reports(base)
        stage_load = self._stage_load_report(base)
        partial_stage_load = self._partial_stage_load_report(base)

        def fetch(model_id: str, filename: str, *, timeout_seconds: float = 60.0) -> dict:
            return self._hf_fixture(model_id, filename)

        def header(model_id: str, filename: str, *, timeout_seconds: float, max_header_bytes: int) -> tuple[int, dict]:
            index = self._hf_fixture(model_id, "model.safetensors.index.json")
            return self._header_for_stage(model_id, index, filename)

        with mock.patch.object(pack, "fetch_hf_json", side_effect=fetch), mock.patch.object(pack, "load_safetensors_header", side_effect=header):
            return pack.build_report(
                pack.parse_args(
                    [
                        "--output-dir",
                        str(base / "frontier"),
                        "--baseline-32b-bridge-report",
                        str(bridge),
                        "--baseline-32b-serving-report",
                        str(serving),
                        "--larger-stage-owned-load-report",
                        str(stage_load),
                        "--partial-stage-owned-load-report",
                        str(partial_stage_load),
                        "--candidate",
                        "72b-awq|Qwen/Qwen2.5-72B-Instruct-AWQ|awq_safetensors|decode",
                        "--candidate",
                        "100b-compressed|cyankiwi/Solar-Open-100B-AWQ-4bit|compressed_tensors_4bit_safetensors|stage_load",
                        "--candidate",
                        "235b-awq|QuixiAI/Qwen3-235B-A22B-AWQ|awq_safetensors|stage_load",
                        "--stage-count",
                        "6",
                        "--stage-backends",
                        "cuda,cuda,jax_tpu,cpu,cpu,cpu",
                    ]
                )
            )

    def test_pack_builds_capacity_frontier_without_overclaiming_larger_decode(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["heterogeneous_capacity_frontier_ready"])
        self.assertTrue(report["baseline_32b"]["gpu_tpu_cpu_same_request_verified"])
        self.assertEqual(report["conclusions"]["max_gpu_tpu_cpu_same_request_parameter_class"], "32b")
        self.assertEqual(report["conclusions"]["max_1token_decode_parameter_class"], "32b")
        self.assertEqual(report["conclusions"]["max_stage_owned_load_parameter_class"], "72b-awq")
        self.assertEqual(report["conclusions"]["max_partial_stage_owned_load_parameter_class"], "100b-compressed")
        self.assertEqual(report["conclusions"]["max_stage_owned_load_preflight_parameter_class"], "235b-awq")
        self.assertFalse(report["conclusions"]["larger_than_32b_decode_verified"])
        self.assertTrue(report["conclusions"]["capacity_frontier_validation_complete"])
        self.assertIn("larger_than_32b_same_request_decode_not_verified", report["blockers"])
        self.assertEqual(check.validate_report(report), [])

    def test_candidates_record_stage_owned_header_preflight_and_block_decode(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)
        candidates = {item["parameter_class"]: item for item in report["candidates"]}

        self.assertTrue(candidates["72b-awq"]["stage_owned_load_preflight_verified"])
        self.assertTrue(candidates["72b-awq"]["stage_owned_load_verified"])
        self.assertFalse(candidates["72b-awq"]["one_token_decode_verified"])
        self.assertIn("fresh_larger_than_32b_decode_not_yet_verified", candidates["72b-awq"]["blockers"])
        self.assertIn("quantized_jax_tpu_runtime_adapter_missing", candidates["72b-awq"]["blockers"])
        self.assertTrue(candidates["235b-awq"]["stage_owned_load_preflight_verified"])
        self.assertIn("fresh_larger_than_32b_stage_owned_load_not_yet_verified", candidates["235b-awq"]["blockers"])
        self.assertTrue(candidates["100b-compressed"]["partial_stage_owned_load_verified"])

    def test_checker_rejects_larger_decode_overclaim_without_proof(self) -> None:
        base = self._tmp_dir()
        report = self._build_report(base)
        report["conclusions"]["max_gpu_tpu_cpu_same_request_parameter_class"] = "72b-awq"

        errors = check.validate_report(report)

        self.assertIn("max_same_request_decode_should_remain_32b_without_larger_live_proof", errors)

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
            "jupyter-proxy",
            "token=",
            "kernel.py",
            '"prompt":',
            '"generated_text":',
            '"generated_token_ids":',
            '"activation":',
            '"hidden_state":',
            '"kv_cache":',
            '"past_key_values":',
        ]:
            self.assertNotIn(fragment, scanned)

    def test_check_script_builds_and_validates_report(self) -> None:
        base = self._tmp_dir()
        bridge, serving = self._baseline_reports(base)
        stage_load = self._stage_load_report(base)
        partial_stage_load = self._partial_stage_load_report(base)

        def fetch(model_id: str, filename: str, *, timeout_seconds: float = 60.0) -> dict:
            return self._hf_fixture(model_id, filename)

        def header(model_id: str, filename: str, *, timeout_seconds: float, max_header_bytes: int) -> tuple[int, dict]:
            index = self._hf_fixture(model_id, "model.safetensors.index.json")
            return self._header_for_stage(model_id, index, filename)

        with mock.patch.object(pack, "fetch_hf_json", side_effect=fetch), mock.patch.object(pack, "load_safetensors_header", side_effect=header):
            result = check.build_check(
                check.parse_args(
                    [
                        "--output-dir",
                        str(base / "check"),
                        "--baseline-32b-bridge-report",
                        str(bridge),
                        "--baseline-32b-serving-report",
                        str(serving),
                        "--larger-stage-owned-load-report",
                        str(stage_load),
                        "--partial-stage-owned-load-report",
                        str(partial_stage_load),
                        "--candidate",
                        "72b-awq|Qwen/Qwen2.5-72B-Instruct-AWQ|awq_safetensors|decode",
                        "--candidate",
                        "100b-compressed|cyankiwi/Solar-Open-100B-AWQ-4bit|compressed_tensors_4bit_safetensors|stage_load",
                        "--candidate",
                        "235b-awq|QuixiAI/Qwen3-235B-A22B-AWQ|awq_safetensors|stage_load",
                        "--stage-count",
                        "6",
                        "--stage-backends",
                        "cuda,cuda,jax_tpu,cpu,cpu,cpu",
                    ]
                )
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["max_gpu_tpu_cpu_same_request_parameter_class"], "32b")
        self.assertEqual(result["max_stage_owned_load_parameter_class"], "72b-awq")
        self.assertEqual(result["max_partial_stage_owned_load_parameter_class"], "100b-compressed")
        self.assertEqual(result["max_stage_owned_load_preflight_parameter_class"], "235b-awq")


if __name__ == "__main__":
    unittest.main()
