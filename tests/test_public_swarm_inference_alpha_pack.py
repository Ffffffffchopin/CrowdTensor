from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_inference_alpha_check as check
from scripts import public_swarm_inference_alpha_pack as pack


class PublicSwarmInferenceAlphaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_alpha_test_"))

    def test_live_kaggle_aggregates_live_and_local_requeue(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args([
            "--mode",
            "live-kaggle",
            "--output-dir",
            str(output_dir),
            "--public-host",
            "24.199.118.54",
            "--port",
            "9220",
            "--base-port",
            "9221",
            "--kaggle-owner",
            "xuyuhaosuyi",
        ]), runner=check.fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "public_swarm_inference_alpha_v1")
        self.assertEqual(report["mode"], "live-kaggle")
        for code in [
            "public_swarm_inference_alpha_ready",
            "public_swarm_session_ready",
            "public_swarm_live_kaggle_ready",
            "local_stage_requeue_ready",
            "stage_requeue_ready",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "token_rotation_required",
            "public_swarm_live_requeue_ready",
            "external_stage_requeue_ready",
        ]:
            self.assertIn(code, report["diagnosis_codes"])
        self.assertTrue(report["session"]["live_external_runtime_verified"])
        self.assertTrue(report["session"]["local_stage_requeue_verified"])
        self.assertTrue(report["session"]["live_stage_requeue_verified"])
        self.assertTrue(report["session"]["live_kaggle_kernels_deleted"])
        self.assertTrue(report["safety"]["live_requeue_verified"])
        self.assertTrue(report["artifacts"]["public_swarm_inference_alpha_json"]["present"])
        self.assertFalse(report["artifacts"]["local_requeue_json"]["present"])
        self.assertFalse(report["artifacts"]["live_swarm_beta_json"]["present"])
        self.assertTrue(report["artifact_cleanup"]["child_artifacts_pruned"])
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("stage0-secret", serialized)
        self.assertNotIn("hidden_state", serialized)
        self.assertNotIn("generated_text", serialized)

    def test_local_generated_requires_requeue(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args([
            "--mode",
            "local-generated",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "xuyuhaosuyi",
        ]), runner=check.fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_inference_alpha_ready", report["diagnosis_codes"])
        self.assertIn("local_stage_requeue_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_live_kaggle_ready", report["diagnosis_codes"])
        self.assertFalse(report["safety"]["live_kaggle_required"])

    def test_keep_child_artifacts_preserves_child_reports_for_debugging(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args([
            "--mode",
            "live-kaggle",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--keep-child-artifacts",
        ]), runner=check.fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["artifact_cleanup"]["child_artifacts_pruned"])
        self.assertTrue(report["artifacts"]["local_requeue_json"]["present"])
        self.assertTrue(report["artifacts"]["live_swarm_beta_json"]["present"])

    def test_skip_local_requeue_blocks_alpha_ready(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args([
            "--mode",
            "live-kaggle",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--skip-local-requeue",
        ]), runner=check.fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertIn("local_stage_requeue_skipped", report["diagnosis_codes"])
        self.assertIn("public_swarm_inference_alpha_blocked", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_inference_alpha_ready", report["diagnosis_codes"])

    def test_check_contract(self) -> None:
        output_dir = self._tmp_dir()
        report = check.build_check(check.parse_args(["--output-dir", str(output_dir)]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "public_swarm_inference_alpha_check_v1")
        self.assertEqual(report["missing_codes"], [])
        self.assertEqual(report["sensitive_leaks"], [])
        self.assertIn("public_swarm_inference_alpha_check_ready", report["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
