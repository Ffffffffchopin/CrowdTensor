from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import deepseek_v4_flash_quantized_source_resolver as resolver


class DeepSeekV4FlashQuantizedSourceResolverTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="ct_dsv4_source_resolver_"))

    def _fake_model_api(self, repo: str, *, timeout_seconds: float = 90.0) -> dict:
        return {
            "id": repo,
            "private": False,
            "gated": False,
            "downloads": 123,
            "tags": ["gguf", "llama.cpp", "deepseek-v4", "license:mit"],
        }

    def _fake_tree_api(self, repo: str, *, timeout_seconds: float = 90.0) -> list[dict]:
        if repo == "teamblobfish/DeepSeek-V4-Flash-GGUF":
            return [
                {
                    "path": "IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00001-of-00002.gguf",
                    "type": "file",
                    "lfs": {"oid": "a", "size": 40_000_000_000},
                },
                {
                    "path": "IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00002-of-00002.gguf",
                    "type": "file",
                    "lfs": {"oid": "b", "size": 17_000_000_000},
                },
                {
                    "path": "IQ1_M/DeepSeek-V4-Flash-IQ1_M-00001-of-00002.gguf",
                    "type": "file",
                    "lfs": {"oid": "c", "size": 41_000_000_000},
                },
                {
                    "path": "IQ1_M/DeepSeek-V4-Flash-IQ1_M-00002-of-00002.gguf",
                    "type": "file",
                    "lfs": {"oid": "d", "size": 19_000_000_000},
                },
            ]
        if repo == "Preyazz/DeepSeek-V4-Flash-GGUF":
            return [
                {
                    "path": "DeepSeek-V4-Flash-Q2_K.gguf",
                    "type": "file",
                    "lfs": {"oid": "e", "size": 103_000_000_000},
                }
            ]
        return [
            {
                "path": "DeepSeek-V4-Flash-FP4-FP8-native.gguf",
                "type": "file",
                "lfs": {"oid": "f", "size": 156_000_000_000},
            }
        ]

    def _fake_readme(self, repo: str, *, timeout_seconds: float = 90.0) -> str:
        if repo == "teamblobfish/DeepSeek-V4-Flash-GGUF":
            return "THIS IS A WIP. These quants don't load on upstream llama.cpp. __CUDA_ARCH__ >= 890."
        return "Requires llama.cpp built from PR. The deepseek4 architecture is not yet in stable llama.cpp releases."

    def test_build_report_records_quantized_sources_without_weight_download(self) -> None:
        with (
            mock.patch.object(resolver, "hf_model_api", side_effect=self._fake_model_api),
            mock.patch.object(resolver, "hf_tree_api", side_effect=self._fake_tree_api),
            mock.patch.object(resolver, "hf_readme", side_effect=self._fake_readme),
        ):
            report = resolver.build_report(
                resolver.parse_args([
                    "--output-dir",
                    str(self._tmp_dir()),
                    "--runtime-download-budget-gb",
                    "80",
                ])
            )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["model"]["total_params_b"], 284.0)
        self.assertEqual(report["model"]["active_params_b"], 13.0)
        self.assertEqual(report["smallest_ready_candidate"]["candidate_id"], "iq1-s-xl-gguf")
        self.assertEqual(report["recommended_live_probe_candidate"]["candidate_id"], "iq1-s-xl-gguf")
        self.assertEqual(report["recommended_live_probe_candidate"]["total_size_gb"], 57.0)
        self.assertIn("stock_llama_cpp_cannot_load_deepseek_v4_flash", report["blockers"])
        self.assertIn("deepseek_v4_flash_llama_cpp_runtime_wip", report["blockers"])
        self.assertIn("t4_cuda_runtime_not_validated_for_deepseek_v4_flash", report["blockers"])
        q2 = next(item for item in report["candidates"] if item["candidate_id"] == "q2-k-single-gguf")
        self.assertIn("candidate_exceeds_runtime_download_budget", q2["blockers"])
        self.assertEqual(resolver.public_redaction_errors(report), [])


if __name__ == "__main__":
    unittest.main()
