from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_preview_v04_check as check
from scripts import public_swarm_preview_v04_pack as pack


class PublicSwarmPreviewV04PackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_preview_v04_test_"))

    def test_check_builds_ready_evidence_import_with_optional_model(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--require-optional-model-ready",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_preview_v04_check_v1")
        preview = json.loads((output_dir / "preview-v04" / "public_swarm_preview_v04.json").read_text(encoding="utf-8"))
        artifacts = preview["artifacts"]
        self.assertTrue(artifacts["product_swarm_mvp_source_json"]["present"])
        self.assertTrue(artifacts["optional_model_mvp_source_json"]["present"])
        self.assertFalse(preview["output_request"]["include_output"])
        self.assertFalse(preview["output_request"]["raw_prompt_public"])
        self.assertFalse(preview["output_request"]["raw_generated_text_public"])
        self.assertFalse(preview["output_request"]["generated_token_ids_public"])
        self.assertTrue(preview["output_request"]["public_artifact_safe"])
        self.assertEqual(preview["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(preview["prompt_scope"]["prompt_count"], 1)
        self.assertTrue(preview["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(preview["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(preview["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(preview["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertTrue(preview["prompt_scope"]["saved_artifacts_public_safe"])
        self.assertFalse(preview["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(preview["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(preview["prompt_scope"]["public_artifact_safe"])
        self.assertEqual(preview["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(preview["answer_scope"]["visible_in_terminal"])
        self.assertFalse(preview["answer_scope"]["terminal_only"])
        self.assertEqual(preview["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(preview["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertTrue(preview["answer_scope"]["public_artifact_safe"])
        self.assertTrue(preview["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(preview["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(preview["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(preview["shareable_summary"]["generated_token_ids_public"])
        self.assertEqual(preview["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(preview["shareable_summary"]["local_answer_terminal_only"])
        markdown = (output_dir / "preview-v04" / "public_swarm_preview_v04.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- output request note:", markdown)
        self.assertIn("local answer", markdown)
        self.assertIn("prompt scope: `source=prompt-text count=1 inline_prompt_text=True", markdown)
        self.assertIn("- prompt scope note:", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not a local answer transcript", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        support = json.loads((output_dir / "preview-v04" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")

    def test_check_builds_ready_package(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_missing_optional_model_blocks_when_required(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "evidence-import",
                "--require-optional-model-ready",
            ])

    def test_report_redacts_sensitive_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": pack.MODE_EVIDENCE_IMPORT,
                "preview": {"ready": True},
                "performance": {},
                "steps": [],
                "payload_summaries": {},
                "diagnosis_codes": ["public_swarm_preview_v04_ready"],
                "artifacts": pack.base_artifacts(output_dir, ok=True),
                "safety": {},
                "limitations": [],
                "secret": "CROWDTENSOR_ADMIN_TOKEN=admin-secret",
            },
            output_dir=output_dir,
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=admin-secret", encoded)
        self.assertTrue(report["ok"])
        self.assertFalse(report["output_request"]["include_output"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertNotIn("sensitive_output_detected", report["diagnosis_codes"])

    def test_package_runbook_uses_prompt_placeholder(self) -> None:
        output_dir = self._tmp_dir()
        prompt = "private preview prompt should not be saved"
        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--prompt-text",
            prompt,
        ])

        pack.build_report(args)

        runbook = (output_dir / "PUBLIC_SWARM_PREVIEW_V04.md").read_text(encoding="utf-8")
        report = json.loads((output_dir / "public_swarm_preview_v04.json").read_text(encoding="utf-8"))
        support = json.loads((output_dir / "support_bundle.json").read_text(encoding="utf-8"))
        encoded = json.dumps({"report": report, "support": support, "runbook": runbook}, sort_keys=True)
        self.assertNotIn(prompt, encoded)
        self.assertIn("--prompt-text '<your-private-prompt>'", runbook)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")

    def test_failed_step_redacts_prompt_text_from_tails(self) -> None:
        output_dir = self._tmp_dir()
        prompt = "private failure prompt"
        args = pack.parse_args([
            "local-smoke",
            "--output-dir",
            str(output_dir),
            "--prompt-text",
            prompt,
        ])

        def failing_runner(command: list[str], **_: object):
            del command
            return pack.subprocess.CompletedProcess(
                args=["cmd"],
                returncode=1,
                stdout=f"failed for {prompt}\n",
                stderr=f"stderr carried {prompt}\n",
            )

        report = pack.build_report(args, runner=failing_runner)
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(prompt, encoded)
        self.assertIn("<redacted>", encoded)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")


if __name__ == "__main__":
    unittest.main()
