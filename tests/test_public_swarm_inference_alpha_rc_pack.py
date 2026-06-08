from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_inference_alpha_rc_pack as pack


class PublicSwarmInferenceAlphaRcPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_alpha_rc_test_"))

    def _assert_ready_guidance(self, report: dict, output_dir: Path) -> None:
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["user_status"]["next_step"], "review_artifacts")
        self.assertEqual(report["review_summary"]["schema"], "public_swarm_inference_alpha_rc_review_summary_v1")
        self.assertEqual(report["review_summary"]["state"], "ready")
        self.assertEqual(report["recommended_next_command"]["label"], "inspect Public Swarm Alpha RC evidence")
        self.assertIn("public_swarm_inference_alpha_rc.md", report["recommended_next_command"]["command_line"])
        self.assertGreaterEqual(len(report["next_commands"]), 3)
        self.assertFalse(report["not_completed"])
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], report["artifact_summary"]["artifact_count"])
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
        self.assertTrue((output_dir / "support_bundle.json").is_file())
        support = json.loads((output_dir / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["schema"], "public_swarm_inference_alpha_rc_support_bundle_v1")
        self.assertEqual(support["review_summary"]["state"], "ready")
        self.assertTrue(support["public_artifact_safe"])

    def _alpha_report(self, *, stage: str) -> dict:
        failure_mode = f"kill-{stage}-after-claim"
        stage_code = f"live_{stage}_requeue_ready"
        return {
            "schema": "public_swarm_inference_alpha_v1",
            "ok": True,
            "mode": "live-kaggle",
            "failure_mode": failure_mode,
            "diagnosis_codes": [
                "public_swarm_inference_alpha_ready",
                "public_swarm_session_ready",
                "public_swarm_live_requeue_ready",
                "public_swarm_live_kaggle_ready",
                "external_stage_requeue_ready",
                stage_code,
                "external_runtime_verified",
                "kaggle_kernels_deleted",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "token_rotation_required",
            ],
            "session": {
                "live_external_runtime_verified": True,
                "live_stage_requeue_verified": True,
                "live_kaggle_kernels_deleted": True,
                "live_summary": {
                    "live_requeue_summary": {
                        "enabled": True,
                        "failure_mode": failure_mode,
                        "target_stage": stage,
                        "claim_observed": True,
                        "victim_kernel_deleted": True,
                        "lease_expired": "<redacted>",
                        "rescued_result": True,
                        "victim_result_accepted": False,
                    },
                },
            },
            "safety": {
                "cpu_only": True,
                "read_only_workload": "real_llm_sharded_infer",
                "not_production": True,
                "not_p2p": True,
                "not_large_model_serving": True,
                "not_public_prompt_serving": True,
            },
            "artifact_cleanup": {"child_artifacts_pruned": True},
            "artifacts": {
                "local_requeue_json": {"present": False},
                "live_swarm_beta_json": {"present": False},
                "live_support_bundle_json": {"present": False},
            },
        }

    def _write_reports(self, root: Path) -> tuple[Path, Path, Path]:
        stage0_dir = root / "stage0"
        stage1_dir = root / "stage1"
        stage0_dir.mkdir(parents=True)
        stage1_dir.mkdir(parents=True)
        stage0 = stage0_dir / "public_swarm_inference_alpha.json"
        stage1 = stage1_dir / "public_swarm_inference_alpha.json"
        summary = root / "public-swarm-inference-alpha-live-requeue-summary.json"
        stage0.write_text(json.dumps(self._alpha_report(stage="stage0"), sort_keys=True), encoding="utf-8")
        stage1.write_text(json.dumps(self._alpha_report(stage="stage1"), sort_keys=True), encoding="utf-8")
        summary.write_text(json.dumps({
            "schema": "public_swarm_inference_alpha_live_requeue_summary_v1",
            "ok": True,
            "proofs": [
                {
                    "ok": True,
                    "target_stage": "stage0",
                    "claim_observed": True,
                    "victim_kernel_deleted": True,
                    "rescued_result": True,
                    "victim_result_accepted": False,
                },
                {
                    "ok": True,
                    "target_stage": "stage1",
                    "claim_observed": True,
                    "victim_kernel_deleted": True,
                    "rescued_result": True,
                    "victim_result_accepted": False,
                },
            ],
        }, sort_keys=True), encoding="utf-8")
        return stage0, stage1, summary

    def test_evidence_import_requires_both_stage_live_requeue_reports(self) -> None:
        root = self._tmp_dir()
        stage0, stage1, summary = self._write_reports(root)
        output_dir = self._tmp_dir() / "rc"
        report = pack.build_report(pack.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--stage0-report",
            str(stage0),
            "--stage1-report",
            str(stage1),
            "--summary-report",
            str(summary),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "public_swarm_inference_alpha_rc_v1")
        for code in [
            "public_swarm_inference_alpha_rc_ready",
            "public_swarm_alpha_rc_evidence_imported",
            "stage0_live_requeue_evidence_ready",
            "stage1_live_requeue_evidence_ready",
            "public_swarm_live_requeue_evidence_ready",
            "public_swarm_live_requeue_summary_ready",
            "public_swarm_alpha_private_artifacts_absent",
        ]:
            self.assertIn(code, report["diagnosis_codes"])
        self.assertTrue(report["imported_reports"]["stage0"]["ready"])
        self.assertTrue(report["imported_reports"]["stage1"]["ready"])
        self.assertTrue(report["artifacts"]["public_swarm_inference_alpha_rc_json"]["present"])
        self._assert_ready_guidance(report, output_dir)
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
        markdown = (output_dir / "public_swarm_inference_alpha_rc.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("## Artifact Summary", markdown)
        self.assertIn("## Not Completed", markdown)
        self.assertIn("Public Swarm Inference Alpha RC evidence is ready.", markdown)
        self.assertIn("- output request note:", markdown)
        self.assertIn("answer text", markdown)
        self.assertIn("prompt scope: `source=imported-or-built-in-validation-prompts", markdown)
        self.assertIn("prompt scope note: Public Swarm Alpha RC imports retained validation evidence", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not a local answer transcript", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generation_public=False generation_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        encoded = json.dumps(report, sort_keys=True)
        for fragment in ["generated_text", "generated_token_ids"]:
            self.assertNotIn(fragment, encoded)

    def test_evidence_import_blocks_on_private_artifacts(self) -> None:
        root = self._tmp_dir()
        stage0, stage1, summary = self._write_reports(root)
        (stage0.parent / "miner.private.env").write_text("secret", encoding="utf-8")
        output_dir = self._tmp_dir() / "rc"
        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--stage0-report",
            str(stage0),
            "--stage1-report",
            str(stage1),
            "--summary-report",
            str(summary),
        ]))

        self.assertFalse(report["ok"], report)
        self.assertIn("private_artifacts_present", report["imported_reports"]["stage0"]["failed_checks"])
        self.assertIn("public_swarm_inference_alpha_rc_blocked", report["diagnosis_codes"])
        self.assertTrue(report["imported_reports"]["stage0"]["private_artifacts_present"])
        self.assertEqual(report["imported_reports"]["stage0"]["private_artifact_count"], 1)
        self.assertNotIn("miner.private.env", json.dumps(report, sort_keys=True))
        self.assertEqual(report["user_status"]["state"], "evidence-import-blocked")
        self.assertTrue(report["not_completed"])
        self.assertIn("rerun Public Swarm Alpha RC evidence-import", report["recommended_next_command"]["label"])

    def test_local_smoke_uses_ci_safe_alpha_check(self) -> None:
        output_dir = self._tmp_dir() / "rc"
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_alpha_check.py", command[1])
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": "public_swarm_inference_alpha_check_v1",
                    "ok": True,
                    "diagnosis_codes": ["public_swarm_inference_alpha_check_ready"],
                }) + "\n",
                stderr="",
            )

        report = pack.build_report(pack.parse_args([
            "--mode",
            "local-smoke",
            "--output-dir",
            str(output_dir),
        ]), runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_alpha_rc_local_smoke_ready", report["diagnosis_codes"])
        self._assert_ready_guidance(report, output_dir)
        self.assertFalse(report["output_request"]["include_output"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
