from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import kaggle_alternate_llm_source_resolver as resolver


class KaggleAlternateLlmSourceResolverTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="ct_alt_llm_source_"))

    def _fake_hf(self, model_repo: str, filename: str, *, timeout_seconds: float = 90.0) -> dict:
        if filename == "config.json":
            if "DeepSeek" in model_repo or "deepseek" in model_repo:
                return {
                    "architectures": ["DeepseekV3ForCausalLM"],
                    "model_type": "deepseek_v3",
                    "hidden_size": 7168,
                    "num_hidden_layers": 61,
                    "num_attention_heads": 128,
                    "num_key_value_heads": 128,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "torch_dtype": "bfloat16",
                    "vocab_size": 129280,
                }
            if "Qwen3" in model_repo:
                return {
                    "architectures": ["Qwen3MoeForCausalLM"],
                    "model_type": "qwen3_moe",
                    "hidden_size": 4096,
                    "num_hidden_layers": 94,
                    "num_attention_heads": 64,
                    "num_key_value_heads": 4,
                    "num_experts": 128,
                    "num_experts_per_tok": 8,
                    "torch_dtype": "bfloat16",
                    "vocab_size": 151936,
                }
            return {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 16384,
                "num_hidden_layers": 126,
                "num_attention_heads": 128,
                "num_key_value_heads": 8,
                "torch_dtype": "bfloat16",
                "vocab_size": 128256,
            }
        return {
            "metadata": {"total_size": 123456},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            },
        }

    def test_default_candidates_classify_dense_and_moe_without_weight_downloads(self) -> None:
        with mock.patch.object(resolver, "fetch_hf_json", side_effect=self._fake_hf):
            report = resolver.build_report(
                resolver.parse_args([
                    "--output-dir",
                    str(self._tmp_dir()),
                    "--fetch-hf-metadata",
                ])
            )

        self.assertTrue(report["ok"], report)
        candidates = {item["parameter_class"]: item for item in report["candidates"]}
        self.assertTrue(candidates["405b"]["dense_full_precision_candidate"])
        self.assertEqual(candidates["405b"]["candidate_id"], "405b")
        self.assertEqual(candidates["405b"]["kaggle_ref"], "metaresearch/llama-3.1/Transformers/405b/1")
        self.assertEqual(candidates["405b"]["kaggle_owner"], "metaresearch")
        self.assertEqual(candidates["405b"]["kaggle_model"], "llama-3.1")
        self.assertEqual(candidates["405b"]["parameter_count_b"], 405.0)
        self.assertEqual(candidates["405b"]["active_parameter_count_b"], 405.0)
        self.assertEqual(candidates["405b"]["total_size_bytes"], 123456)
        self.assertEqual(candidates["405b"]["model_type"], "llama")
        self.assertEqual(candidates["405b"]["num_hidden_layers"], 126)
        self.assertTrue(candidates["405b"]["license_agreement_required"])
        self.assertIn("kaggle_model_license_agreement_required", candidates["405b"]["blockers"])
        self.assertTrue(candidates["235b-a22b"]["moe_candidate"])
        self.assertIn("qwen3_moe_or_hybrid_adapter_required", candidates["235b-a22b"]["blockers"])
        self.assertTrue(candidates["671b-v3"]["moe_candidate"])
        self.assertIn("deepseek_mla_moe_adapter_required", candidates["671b-v3"]["blockers"])
        self.assertFalse(candidates["235b-a22b"]["hf_metadata_fallback"]["weight_tensor_downloaded"])
        self.assertIn("deepseek-ai/deepseek-v3/Transformers/deepseek-v3/2", report["model_source_refs"])
        self.assertEqual(resolver.public_redaction_errors(report), [])


if __name__ == "__main__":
    unittest.main()
