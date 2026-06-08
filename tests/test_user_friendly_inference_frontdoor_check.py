from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "user_friendly_inference_frontdoor_check.py"
SPEC = importlib.util.spec_from_file_location("user_friendly_inference_frontdoor_check", SCRIPT_PATH)
frontdoor_check = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(frontdoor_check)


class UserFriendlyInferenceFrontdoorCheckTests(unittest.TestCase):
    def test_run_check_builds_redacted_infer_and_generate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = frontdoor_check.parse_args(["--output-dir", tmp])
            result = frontdoor_check.run_check(args)

            self.assertTrue(result["ok"], result)
            self.assertIn(frontdoor_check.CHECK_READY, result["diagnosis_codes"])
            self.assertFalse(result["safety"]["started_coordinator"])
            self.assertFalse(result["safety"]["submitted_live_task"])
            self.assertFalse(result["safety"]["fresh_kaggle_gpu_attempted"])
            self.assertFalse(result["safety"]["fresh_kaggle_gpu_verified"])
            self.assertTrue(result["safety"]["public_artifact_safe"])

            infer_verdict = result["checked_infer_verdict"]
            generate_verdict = result["checked_generate_verdict"]
            self.assertEqual(infer_verdict["schema"], "crowdtensor_inference_verdict_v1")
            self.assertEqual(infer_verdict["kind"], "Inference")
            self.assertEqual(infer_verdict["state"], "completed")
            self.assertEqual(infer_verdict["answer_scope_state"], "saved-terminal-redacted")
            self.assertFalse(infer_verdict["answer_visible_in_terminal"])
            self.assertEqual(infer_verdict["evidence_level"], "local-cpu-loopback")
            self.assertEqual(infer_verdict["gpu_state"], "local-cpu-only")
            self.assertFalse(infer_verdict["fresh_kaggle_gpu_verified"])
            self.assertTrue(infer_verdict["public_artifact_safe"])
            self.assertEqual(generate_verdict["kind"], "Generation")
            self.assertEqual(generate_verdict["answer_scope_state"], "saved-terminal-redacted")
            self.assertFalse(generate_verdict["answer_visible_in_terminal"])
            self.assertEqual(generate_verdict["evidence_level"], "existing-runtime-submit")
            self.assertEqual(generate_verdict["gpu_state"], "local-cpu-only")
            self.assertFalse(generate_verdict["fresh_kaggle_gpu_verified"])
            self.assertTrue(generate_verdict["public_artifact_safe"])

            self.assertTrue(result["infer"]["terminal_verdict"]["answer_visible_in_terminal"])
            self.assertTrue(result["generate"]["terminal_verdict"]["answer_visible_in_terminal"])
            self.assertEqual(result["infer"]["saved_answer_scope"], "saved-terminal-redacted")
            self.assertEqual(result["generate"]["saved_answer_scope"], "saved-terminal-redacted")

            for relative in [
                "infer/infer_summary.json",
                "infer/infer_summary.md",
                "generate/generate_summary.json",
                "generate/generate_summary.md",
                "user_friendly_inference_frontdoor_check.json",
            ]:
                self.assertTrue((Path(tmp) / relative).is_file(), relative)

            combined = "\n".join(
                (Path(tmp) / relative).read_text(encoding="utf-8")
                for relative in [
                    "infer/infer_summary.json",
                    "infer/infer_summary.md",
                    "generate/generate_summary.json",
                    "generate/generate_summary.md",
                ]
            )
            self.assertNotIn(frontdoor_check.PROMPT_TEXT, combined)
            self.assertNotIn(frontdoor_check.INFER_TEXT, combined)
            self.assertNotIn(frontdoor_check.GENERATE_TEXT, combined)
            self.assertNotIn(frontdoor_check.ADMIN_TOKEN, combined)
            self.assertNotIn('"generated_token_ids": [', combined)
            self.assertIn("- Verdict:", combined)
            self.assertIn("answer=saved-terminal-redacted", combined)
            self.assertIn("fresh_kaggle_gpu=False", combined)

    def test_validator_rejects_saved_answer_visibility_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "infer"
            report = frontdoor_check.build_fake_infer_report(output_dir, max_new_tokens=2)
            summary_path = output_dir / "infer_summary.json"
            markdown_path = output_dir / "infer_summary.md"
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))
            persisted["inference_verdict"]["answer_visible_in_terminal"] = True
            summary_path.write_text(json.dumps(persisted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            markdown_path.write_text(
                markdown_path.read_text(encoding="utf-8") + "\n" + frontdoor_check.INFER_TEXT + "\n",
                encoding="utf-8",
            )
            errors: list[str] = []

            frontdoor_check._validate_frontdoor_artifact(
                kind="Inference",
                report=report,
                summary_path=summary_path,
                markdown_path=markdown_path,
                raw_answer=frontdoor_check.INFER_TEXT,
                errors=errors,
            )

            self.assertIn("Inference_saved_verdict_answer_visibility_mismatch", errors)
            self.assertIn("Inference_artifact_leaked_frontdoor infer answer must rema", errors)
            self.assertIn("Inference_raw_answer_leaked", errors)


if __name__ == "__main__":
    unittest.main()
