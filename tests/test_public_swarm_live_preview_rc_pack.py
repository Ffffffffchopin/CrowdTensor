from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_live_preview_rc_check as check
from scripts import public_swarm_live_preview_rc_pack as pack


class PublicSwarmLivePreviewRCPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_live_preview_test_"))

    def test_check_builds_ready_local_smoke(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-smoke",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_live_preview_rc_check_v1")

    def test_local_smoke_report_records_output_scope(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args([
            "--mode",
            "local-smoke",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        report_path = output_dir / "live-preview" / "public_swarm_live_preview_rc.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_public_safe"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(report["answer_scope"]["terminal_only"])
        self.assertEqual(report["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(report["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertFalse(report["answer_scope"]["raw_prompt_public"])
        self.assertFalse(report["answer_scope"]["raw_generated_text_public"])
        self.assertFalse(report["answer_scope"]["generated_token_ids_public"])
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["user_status"]["recommended_label"], "inspect Live Preview RC evidence")
        self.assertEqual(report["review_summary"]["schema"], "public_swarm_live_preview_rc_review_summary_v1")
        self.assertEqual(report["review_summary"]["next_step"], "review_artifacts")
        self.assertEqual(report["recommended_next_command"]["label"], "inspect Live Preview RC evidence")
        self.assertTrue(any(item["label"] == "inspect support bundle" for item in report["next_commands"]))
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], report["artifact_summary"]["artifact_count"])

        markdown = (output_dir / "live-preview" / "public_swarm_live_preview_rc.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("- recommended: `inspect Live Preview RC evidence` reason=`review_artifacts`", markdown)
        self.assertIn("## Artifact Summary", markdown)
        self.assertIn("- present: `4` / `4`", markdown)
        self.assertIn("## Not Completed", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- output request note:", markdown)
        self.assertIn("answer text", markdown)
        self.assertIn(
            "- prompt scope: `source=prompt-text count=1 inline_prompt_text=True terminal_next_commands_local_private=True saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- prompt scope note: This Live Preview RC artifact records inherited prompt source/count and placeholder safety only; raw prompt text is excluded from public JSON, Markdown, and support bundles.",
            markdown,
        )
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not a local answer transcript", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        support_path = output_dir / "live-preview" / "support_bundle.json"
        support = json.loads(support_path.read_text(encoding="utf-8"))
        self.assertEqual(support["prompt_scope"], report["prompt_scope"])
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertEqual(support["user_status"], report["user_status"])
        self.assertEqual(support["review_summary"], report["review_summary"])
        self.assertEqual(support["artifact_summary"], report["artifact_summary"])
        self.assertEqual(support["recommended_next_command"]["label"], "inspect Live Preview RC evidence")

    def test_check_builds_ready_package(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_live_kaggle_contract(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "live-kaggle",
            "--failure-mode",
            "kill-stage0-after-claim",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_evidence_import(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_live_kaggle_requires_kaggle_owner(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "live-kaggle",
                "--output-dir",
                str(self._tmp_dir()),
                "--kaggle-owner",
                "",
            ])

    def test_report_redacts_sensitive_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": "local-smoke",
                "diagnosis_codes": ["public_swarm_live_preview_rc_ready"],
                "live_preview": {"ready": True},
                "steps": [],
                "safety": {},
                "limitations": [],
                "secret": "CROWDTENSOR_ADMIN_TOKEN=admin-secret",
                "artifacts": pack.base_artifacts(output_dir, ok=True),
            },
            output_dir=output_dir,
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=admin-secret", encoded)
        self.assertTrue(report["ok"])
        self.assertEqual(report["recommended_next_command"]["label"], "inspect Live Preview RC evidence")
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])


if __name__ == "__main__":
    unittest.main()
