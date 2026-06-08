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
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generation_public"])
        self.assertFalse(report["output_request"]["generation_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "imported-or-built-in-validation-prompts")
        self.assertIsInstance(report["prompt_scope"]["prompt_count"], int)
        self.assertGreaterEqual(report["prompt_scope"]["prompt_count"], 0)
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(report["answer_scope"]["terminal_only"])
        self.assertEqual(report["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(report["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertFalse(report["answer_scope"]["raw_prompt_public"])
        self.assertFalse(report["answer_scope"]["raw_generation_public"])
        self.assertFalse(report["answer_scope"]["generation_ids_public"])
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generation_public"])
        self.assertFalse(report["shareable_summary"]["generation_ids_public"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        markdown = (output_dir / "public_swarm_inference_alpha.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- output request note:", markdown)
        self.assertIn("answer text", markdown)
        self.assertIn("prompt scope: `source=imported-or-built-in-validation-prompts", markdown)
        self.assertIn("prompt scope note: Public Swarm Alpha aggregates fixed or imported validation prompts", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not a local answer transcript", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generation_public=False generation_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("stage0-secret", serialized)
        self.assertNotIn("hidden_state", serialized)
        self.assertNotIn("generated_text", serialized)
        self.assertNotIn("generated_token_ids", serialized)

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
        self.assertFalse(report["output_request"]["include_output"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")

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
        self.assertEqual(report["scope_errors"], [])
        self.assertIn("public_swarm_inference_alpha_check_ready", report["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
