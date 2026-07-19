from __future__ import annotations

import unittest

from scripts import kaggle_colab_gpu_cpu_heterogeneous_check as check


def ready_report() -> dict:
    return {
        "schema": "kaggle_32b_full_heterogeneous_probe_v1",
        "ok": True,
        "accepted_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
        "provider_stage_counts": {"kaggle_cuda": 1, "colab_cuda": 1, "cpu": 1, "web_tpu": 0},
        "generated_token_count": 1,
        "max_new_tokens": 1,
        "quantization": "none",
        "kaggle_colab_gpu_cpu_same_request_verified": True,
        "model": {"stage_count": 3, "parameter_count_b": 0.5},
        "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
        "coordinator": {
            "generated_token_count": 1,
            "generated_token_hashes": ["sha256:t"],
            "activation_hashes": ["sha256:a0", "sha256:a1"],
        },
        "kaggle_lifecycle": {
            "kernels_deleted": True,
            "private_packages_removed": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "raw_prompt_public": False,
            "generated_token_ids_public": False,
        },
    }


class KaggleColabGpuCpuHeterogeneousCheckTests(unittest.TestCase):
    def test_ready_report_passes(self) -> None:
        result = check.check_report(ready_report())

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_runtime_probe_is_not_same_request_success(self) -> None:
        result = check.check_report({
            "schema": "colab_cuda_runtime_probe_v1",
            "ok": True,
            "colab_cuda_runtime_ready": True,
        })

        self.assertFalse(result["ok"])
        self.assertIn("schema_mismatch", result["errors"])
        self.assertIn("required_providers_missing", result["errors"])

    def test_rejects_missing_colab_provider(self) -> None:
        report = ready_report()
        report["accepted_providers"] = ["kaggle_cuda", "cpu"]
        report["provider_stage_counts"]["colab_cuda"] = 0

        result = check.check_report(report)

        self.assertFalse(result["ok"])
        self.assertIn("required_providers_missing", result["errors"])
        self.assertIn("colab_cuda_stage_count_missing", result["errors"])

    def test_rejects_private_material(self) -> None:
        report = ready_report()
        report["runtime_proxy_token"] = "secret"

        result = check.check_report(report)

        self.assertFalse(result["ok"])
        self.assertIn("private_material_public", result["errors"])


if __name__ == "__main__":
    unittest.main()
