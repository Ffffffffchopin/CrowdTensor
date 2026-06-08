from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_product_rc_check as check
from scripts import public_swarm_product_rc_pack as pack


class PublicSwarmProductRcPackTests(unittest.TestCase):
    def test_check_builds_ready_rc_from_synthetic_evidence(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_product_rc_check_test_"))
        result = check.run_check(check.parse_args([
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "4",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_product_rc_check_v1")
        report = json.loads((output_dir / "rc" / "public_swarm_product_rc.json").read_text(encoding="utf-8"))
        encoded = json.dumps(report, sort_keys=True)
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
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
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertFalse(report["shareable_summary"]["local_output_display_only"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        self.assertTrue(report["artifacts"]["public_swarm_product_rc_json"]["present"])
        self.assertTrue(report["artifacts"]["public_swarm_product_rc_markdown"]["present"])
        markdown = (output_dir / "rc" / "public_swarm_product_rc.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- include output: `False`", markdown)
        self.assertIn("- output request note:", markdown)
        self.assertIn("answer text", markdown)
        self.assertIn(
            "- prompt scope: `source=prompt-text count=1 inline_prompt_text=True terminal_next_commands_local_private=True saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not a local answer transcript", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        self.assertNotIn("private product rc check prompt", encoded)
        self.assertNotIn("private product rc check prompt", markdown)

    def test_pack_blocks_missing_gpu_evidence(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_product_rc_missing_gpu_test_"))

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "session_protocol_check.py" in " ".join(command):
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "schema": "session_protocol_check_v1"}) + "\n", stderr="")
            if "p2p_lite_discovery_check.py" in " ".join(command):
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "schema": "p2p_lite_discovery_check_v1"}) + "\n", stderr="")
            raise AssertionError(command)

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(output_dir / "missing.json"),
            "--prompt-text",
            "private missing gpu prompt",
            "--max-new-tokens",
            "4",
        ]), runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertFalse(report["ok"])
        self.assertIn("gpu_generation_evidence_import_blocked", report["diagnosis_codes"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertNotIn("private missing gpu prompt", encoded)

    def test_output_scope_errors_rejects_generated_text_public(self) -> None:
        report = {
            "output_request": pack.output_request_summary(),
            "prompt_scope": pack.prompt_scope_summary(),
            "answer_scope": pack.answer_scope_summary(),
            "shareable_summary": pack.shareable_summary(),
        }
        self.assertEqual(check.output_scope_errors(report), [])

        report["output_request"]["raw_generated_text_public"] = True
        report["shareable_summary"]["answer_scope_state"] = "terminal-visible"

        errors = check.output_scope_errors(report)
        self.assertIn("output_request_raw_generated_text_public_mismatch", errors)
        self.assertIn("shareable_answer_scope_state_mismatch", errors)


if __name__ == "__main__":
    unittest.main()
