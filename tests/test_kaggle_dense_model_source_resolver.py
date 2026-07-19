from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import kaggle_dense_model_source_resolver as resolver


class KaggleDenseModelSourceResolverTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_dense_source_test_"))

    def _fake_hf(self, model_repo: str, filename: str, *, timeout_seconds: float = 90.0) -> dict:
        if filename == "config.json":
            return {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 176,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_hidden_layers": 4,
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

    def test_default_dense_qwen_candidates_resolve_with_hf_metadata(self) -> None:
        output_dir = self._tmp_dir()

        with mock.patch.object(resolver, "fetch_hf_json", side_effect=self._fake_hf):
            report = resolver.build_report(
                resolver.parse_args([
                    "--output-dir",
                    str(output_dir),
                    "--fetch-hf-metadata",
                ])
            )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["kaggle_dense_model_source_resolver_ready"])
        self.assertEqual(report["largest_dense_attach_candidate"]["parameter_class"], "72b")
        self.assertIn("qwen-lm/qwen2.5/Transformers/72b-instruct/1", report["kernel_model_sources"])
        self.assertFalse(report["kaggle_model_attach_used"])
        self.assertIn("kaggle_model_attach_paths_not_present_in_current_runtime", report["blockers"])
        self.assertEqual(resolver.public_redaction_errors(report), [])

    def test_attached_path_is_detected_without_runtime_download(self) -> None:
        root = self._tmp_dir()
        model_path = root / "models" / "qwen-lm" / "qwen2.5" / "transformers" / "7b-instruct" / "1"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        (model_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}), encoding="utf-8")
        (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")

        report = resolver.build_report(
            resolver.parse_args([
                "--output-dir",
                str(self._tmp_dir()),
                "--kaggle-input-root",
                str(root),
                "--candidate",
                "7b|Qwen/Qwen2.5-7B-Instruct|qwen-lm|qwen2.5|Transformers|7b-instruct|1|Apache 2.0|15242807035",
            ])
        )

        candidate = report["candidates"][0]
        self.assertEqual(
            candidate["attached_runtime_path"],
            str(root / "models" / "qwen-lm" / "qwen2.5" / "transformers" / "7b-instruct" / "1"),
        )
        self.assertTrue(candidate["attach_path_present"])
        self.assertTrue(candidate["kaggle_model_attach_used_in_current_environment"])
        self.assertFalse(candidate["runtime_disk_download_required"])
        self.assertTrue(report["kaggle_model_attach_used"])

    def test_quantized_candidate_is_not_dense_main_path(self) -> None:
        report = resolver.build_report(
            resolver.parse_args([
                "--output-dir",
                str(self._tmp_dir()),
                "--candidate",
                "72b|Qwen/Qwen2.5-72B-Instruct-AWQ|qwen-lm|qwen2.5|Transformers|72b-instruct-awq|1|Apache 2.0|41607438677",
            ])
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["candidates"][0]["full_precision_dense_candidate"])
        self.assertIn("candidate_not_full_precision_dense_transformers", report["candidates"][0]["blockers"])


if __name__ == "__main__":
    unittest.main()
