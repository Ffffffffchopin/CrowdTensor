from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_qwen_stage_adapter_plan as plan


class GpuTpuQwenStageAdapterPlanTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_qwen_tpu_adapter_test_"))

    def test_fixture_plan_maps_stage_owned_keys_without_runtime_overclaim(self) -> None:
        output_dir = self._tmp_dir() / "fixture"
        report = plan.build_report(plan.parse_args([
            "--output-dir",
            str(output_dir),
            "--mode",
            "fixture",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["checkpoint_bridge_plan_ready"])
        self.assertTrue(report["stage_owned_tpu_loader_plan_ready"])
        self.assertFalse(report["tpu_32b_runtime_adapter_ready"])
        self.assertFalse(report["jax_tpu_runtime_execution_ready"])
        self.assertFalse(report["same_request_live_heterogeneous_verified"])
        self.assertIn("jax_tpu_runtime_execution_not_performed", report["blockers"])
        self.assertGreater(report["mapping"]["assigned_key_count"], 0)
        self.assertEqual(report["mapping"]["unsupported_key_count"], 0)
        self.assertTrue(report["mapping"]["all_assigned_keys_mapped"])
        self.assertIn("activation_metadata", report["shape_protocol"])
        self.assertNotIn("activation", report["shape_protocol"])

    def test_public_artifacts_are_redacted(self) -> None:
        output_dir = self._tmp_dir() / "redaction"
        report = plan.build_report(plan.parse_args([
            "--output-dir",
            str(output_dir),
            "--mode",
            "fixture",
        ]))

        self.assertEqual(plan.public_redaction_errors(report), [])
        scanned = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        for fragment in [
            "KAGGLE_KEY=",
            "KAGGLE_USERNAME=",
            "HF_TOKEN=",
            "Bearer ",
            "kaggle-cookies.json",
            "kaggle-web-storage-state.json",
            "operator.private.env",
            "miner.private.env",
            "kernel.py",
            '"prompt":',
            '"generated_text":',
            '"generated_token_ids":',
            '"activation":',
            '"activations":',
            '"hidden_state":',
            '"logits":',
            '"kv_cache":',
            '"past_key_values":',
            '"lease_token":',
            '"idempotency_key":',
        ]:
            self.assertNotIn(fragment, scanned)

    def test_invalid_layer_range_blocks_plan(self) -> None:
        report = plan.build_report(plan.parse_args([
            "--output-dir",
            str(self._tmp_dir() / "invalid"),
            "--mode",
            "fixture",
            "--tpu-layer-start",
            "30",
            "--tpu-layer-end",
            "30",
        ]))

        self.assertFalse(report["ok"])
        self.assertIn("invalid_tpu_layer_range", report["blockers"])

    def test_artifact_written_with_schema(self) -> None:
        output_dir = self._tmp_dir() / "artifact"
        report = plan.build_report(plan.parse_args([
            "--output-dir",
            str(output_dir),
            "--mode",
            "fixture",
        ]))
        payload = json.loads((output_dir / "gpu_tpu_qwen_stage_adapter_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], plan.SCHEMA)
        self.assertEqual(payload["artifacts"]["summary_json"]["schema"], plan.SCHEMA)
        self.assertTrue(report["artifacts"]["summary_json"]["present"])


if __name__ == "__main__":
    unittest.main()
