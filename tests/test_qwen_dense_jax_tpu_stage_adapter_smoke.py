from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import qwen_dense_jax_tpu_stage_adapter_smoke as smoke


class QwenDenseJaxTpuStageAdapterSmokeTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_qwen_dense_adapter_test_"))

    def test_fixture_torch_reference_exercises_qwen_components_without_overclaiming_tpu(self) -> None:
        report = smoke.build_report(
            smoke.parse_args([
                "--output-dir",
                str(self._tmp_dir()),
            ])
        )

        self.assertFalse(report["ok"])
        self.assertTrue(report["torch_reference_forward_ready"])
        self.assertFalse(report["jax_runtime_execution_ready"])
        self.assertFalse(report["tpu_jax_qwen_stage_runtime_ready"])
        self.assertIn("jax_execution_not_requested", report["blockers"])
        components = report["qwen_components_exercised"]
        for key in ["rms_norm", "rope", "grouped_query_attention", "causal_attention", "swiglu_mlp", "stage_local_kv_cache"]:
            self.assertTrue(components[key], key)
        self.assertTrue(report["stage_local_kv_cache_verified"])
        self.assertEqual(smoke.public_redaction_errors(report), [])

    def test_require_tpu_without_jax_is_invalid(self) -> None:
        with self.assertRaises(SystemExit):
            smoke.parse_args(["--require-tpu"])

    def test_run_jax_records_missing_jax_as_blocker_in_current_ci(self) -> None:
        report = smoke.build_report(
            smoke.parse_args([
                "--output-dir",
                str(self._tmp_dir()),
                "--run-jax",
                "--require-tpu",
            ])
        )

        self.assertFalse(report["ok"])
        self.assertTrue(report["torch_reference_forward_ready"])
        self.assertFalse(report["tpu_runtime_ready"])
        self.assertIn("jax_tpu_runtime_not_available", report["blockers"])
        self.assertTrue(set(report["blockers"]).intersection({"jax_missing", "jax_tpu_device_missing"}))


if __name__ == "__main__":
    unittest.main()
