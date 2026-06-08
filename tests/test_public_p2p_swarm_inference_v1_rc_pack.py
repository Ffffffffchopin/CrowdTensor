from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from scripts import public_p2p_swarm_inference_v1_rc_check as check
from scripts import public_p2p_swarm_inference_v1_rc_pack as pack


class PublicP2PSwarmInferenceV1RCPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_p2p_v1_rc_test_"))

    def test_check_builds_ready_evidence_import(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_p2p_swarm_inference_v1_rc_check_v1")
        report = json.loads((output_dir / "rc" / "public_p2p_swarm_inference_v1_rc.json").read_text(encoding="utf-8"))
        self.assertIn("public_p2p_swarm_inference_v1_rc_ready", report["diagnosis_codes"])
        self.assertIn("public_p2p_v1_rc_model_metadata_ready", report["diagnosis_codes"])
        self.assertTrue(report["rc"]["model_metadata_ready"])
        self.assertEqual(report["p2p"]["hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertTrue(report["p2p"]["local_model"]["compatible"])
        self.assertTrue(report["p2p"]["external_model"]["compatible"])
        self.assertTrue(report["p2p"]["signed_announcement_required"])
        self.assertIn("Hivemind/Petals production parity", report["not_completed"])
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
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        markdown = (output_dir / "rc" / "public_p2p_swarm_inference_v1_rc.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("prompt scope: `source=prompt-text count=1", markdown)
        self.assertIn("state=no-local-answer", markdown)
        self.assertIn("raw_generated_text_public=False", markdown)
        support = json.loads((output_dir / "rc" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["prompt_scope"], report["prompt_scope"])
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")

    def test_check_builds_ready_local_smoke_contract(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-smoke",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_cli_wrapper_redacts_peer_secret(self) -> None:
        output_dir = self._tmp_dir()
        args = cli.parse_args([
            "public-p2p-v1-rc",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--peer-secret",
            "peer-secret-value",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_p2p_swarm_inference_v1_rc_pack.py", command[1])
            self.assertIn("--peer-secret", command)
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": pack.SCHEMA,
                    "ok": True,
                    "diagnosis_codes": ["public_p2p_swarm_inference_v1_rc_ready"],
                }) + "\n",
                stderr="",
            )

        report = cli.build_public_p2p_swarm_inference_v1_rc(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_p2p_swarm_inference_v1_rc_cli_v1")
        self.assertNotIn("peer-secret-value", encoded)

    def test_package_mode_writes_signed_runbook(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args([
            pack.MODE_PACKAGE,
            "--output-dir",
            str(output_dir),
            "--json",
        ]))

        runbook = output_dir / "PUBLIC_P2P_SWARM_INFERENCE_V1_RC.md"
        self.assertTrue(runbook.is_file())
        text = runbook.read_text(encoding="utf-8")
        self.assertIn("--require-signed", text)
        self.assertIn("crowdtensor generate --p2p", text)
        self.assertFalse(report["ok"], report)
        self.assertIn("public_p2p_v1_rc_runbook_ready", report["diagnosis_codes"])

    def test_kaggle_auto_forwards_signed_peer_secret_to_v06(self) -> None:
        output_dir = self._tmp_dir()
        seen_commands: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            seen_commands.append(command)
            if "local-smoke" in command:
                return subprocess.CompletedProcess(command, 0, json.dumps(check.fake_signed_local_v06()) + "\n", "")
            if "kaggle-auto" in command:
                return subprocess.CompletedProcess(command, 0, json.dumps(check.fake_external_v06()) + "\n", "")
            return subprocess.CompletedProcess(command, 1, "", "unexpected command")

        report = pack.build_report(pack.parse_args([
            pack.MODE_KAGGLE_AUTO,
            "--output-dir",
            str(output_dir),
            "--peer-secret",
            "peer-secret-value",
            "--kaggle-owner",
            "owner",
            "--json",
        ]), runner=fake_runner)

        self.assertTrue(report["ok"], report)
        kaggle_command = next(command for command in seen_commands if "kaggle-auto" in command)
        self.assertIn("--peer-secret", kaggle_command)
        self.assertIn("peer-secret-value", kaggle_command)
        self.assertIn("--require-signed", kaggle_command)
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("peer-secret-value", encoded)

    def test_persist_report_refreshes_public_artifact_presence(self) -> None:
        output_dir = self._tmp_dir()
        signed_local = output_dir / "signed_local.json"
        signed_local.write_text(json.dumps(check.fake_signed_local_v06()), encoding="utf-8")
        report = pack.build_report(pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir),
            "--signed-local-report",
            str(signed_local),
            "--json",
        ]))

        self.assertTrue(report["artifacts"]["public_p2p_swarm_inference_v1_rc_json"]["present"])
        self.assertTrue(report["artifacts"]["public_p2p_swarm_inference_v1_rc_markdown"]["present"])
        self.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
        self.assertTrue(report["artifacts"]["signed_local_v06_json"]["present"])

    def test_missing_v06_prompt_scope_falls_back_to_imported_validation_prompts(self) -> None:
        output_dir = self._tmp_dir()
        signed_local = output_dir / "signed_local.json"
        payload = check.fake_signed_local_v06()
        payload.pop("prompt_scope", None)
        signed_local.write_text(json.dumps(payload), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir / "rc"),
            "--signed-local-report",
            str(signed_local),
            "--json",
        ]))
        encoded = json.dumps(report, sort_keys=True)

        self.assertEqual(report["prompt_scope"]["source"], "imported-or-built-in-validation-prompts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertFalse(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertNotIn("CrowdTensor public P2P v1 RC", encoded)

    def test_evidence_import_blocks_non_default_model_without_matching_v06_reports(self) -> None:
        output_dir = self._tmp_dir()
        signed_local = output_dir / "signed_local.json"
        external = output_dir / "external.json"
        kaggle = output_dir / "kaggle.json"
        signed_local.write_text(json.dumps(check.fake_signed_local_v06()), encoding="utf-8")
        external.write_text(json.dumps(check.fake_external_v06()), encoding="utf-8")
        kaggle.write_text(json.dumps(check.fake_kaggle_v06()), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir / "rc"),
            "--signed-local-report",
            str(signed_local),
            "--v06-external-report",
            str(external),
            "--v06-kaggle-report",
            str(kaggle),
            "--hf-model-id",
            "distilgpt2",
            "--json",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["rc"]["model_metadata_ready"])
        self.assertEqual(report["p2p"]["hf_model_id"], "distilgpt2")
        self.assertEqual(report["p2p"]["local_model"]["observed_hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertFalse(report["p2p"]["local_model"]["compatible"])
        self.assertFalse(report["p2p"]["external_model"]["compatible"])
        self.assertIn("public_p2p_v1_rc_local_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_p2p_v1_rc_external_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_p2p_v1_rc_kaggle_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_p2p_swarm_inference_v1_rc_blocked", report["diagnosis_codes"])

    def test_evidence_import_blocks_default_model_without_observed_model_metadata(self) -> None:
        output_dir = self._tmp_dir()
        signed_local = output_dir / "signed_local.json"
        external = output_dir / "external.json"
        kaggle = output_dir / "kaggle.json"
        signed_payload = check.fake_signed_local_v06()
        external_payload = check.fake_external_v06()
        kaggle_payload = check.fake_kaggle_v06()
        for payload in [signed_payload, external_payload, kaggle_payload]:
            payload.pop("hf_model_id", None)
            if isinstance(payload.get("p2p"), dict):
                payload["p2p"].pop("hf_model_id", None)
                payload["p2p"].pop("observed_hf_model_id", None)
                payload["p2p"].pop("model_id_match", None)
        signed_local.write_text(json.dumps(signed_payload), encoding="utf-8")
        external.write_text(json.dumps(external_payload), encoding="utf-8")
        kaggle.write_text(json.dumps(kaggle_payload), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir / "rc"),
            "--signed-local-report",
            str(signed_local),
            "--v06-external-report",
            str(external),
            "--v06-kaggle-report",
            str(kaggle),
            "--json",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["rc"]["model_metadata_ready"])
        self.assertFalse(report["p2p"]["local_model"]["model_id_present"])
        self.assertFalse(report["p2p"]["external_model"]["model_id_present"])
        self.assertFalse(report["p2p"]["kaggle_model"]["model_id_present"])
        self.assertFalse(report["p2p"]["local_model"]["compatible"])
        self.assertFalse(report["p2p"]["external_model"]["compatible"])
        self.assertFalse(report["p2p"]["kaggle_model"]["compatible"])
        self.assertIn("public_p2p_v1_rc_local_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_p2p_v1_rc_external_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_p2p_v1_rc_kaggle_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_p2p_swarm_inference_v1_rc_blocked", report["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
