from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import swarm_inference_beta_check as check
from scripts import swarm_inference_beta_pack as pack


class SwarmInferenceBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_swarm_beta_test_"))

    def test_prepare_creates_two_stage_join_packs_and_hashed_registry(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "prepare",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9200",
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
        ])

        report = pack.build_report(args)
        registry = json.loads((output_dir / "miner_registry.json").read_text(encoding="utf-8"))
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "swarm_inference_beta_v1")
        self.assertEqual(report["mode"], "prepare")
        self.assertIn("swarm_inference_beta_prepare_ready", report["diagnosis_codes"])
        self.assertIn("two_machine_runbook_ready", report["diagnosis_codes"])
        self.assertTrue((output_dir / "stage0" / "miner_join.sh").is_file())
        self.assertTrue((output_dir / "stage1" / "miner_join.sh").is_file())
        self.assertTrue((output_dir / "SWARM_INFERENCE_BETA.md").is_file())
        self.assertEqual({entry["miner_id"] for entry in registry["miners"]}, {"swarm-beta-stage0", "swarm-beta-stage1"})
        self.assertTrue(all(str(entry["token"]).startswith("sha256:") for entry in registry["miners"]))
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("CROWDTENSOR_MINER_TOKEN=", serialized)

    def test_verify_imports_external_beta_and_requires_split_ready_codes(self) -> None:
        output_dir = self._tmp_dir()
        prepare = pack.parse_args([
            "prepare",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
        ])
        pack.build_report(prepare)
        external = output_dir / "external" / "real_llm_internet_beta.json"
        check.write_external_beta(external)

        args = pack.parse_args([
            "verify",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
            "--real-internet-beta-report",
            str(external),
        ])
        report = pack.build_report(args, runner=check.fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertIn("swarm_inference_beta_ready", report["diagnosis_codes"])
        self.assertIn("real_llm_split_route_ready", report["diagnosis_codes"])
        self.assertIn("two_machine_swarm_inference_ready", report["diagnosis_codes"])
        self.assertIn("external_beta_evidence_imported", report["diagnosis_codes"])
        self.assertIn("decoded_tokens_match", report["diagnosis_codes"])
        self.assertTrue(report["remote_real_llm_sharded_beta_summary"]["stage_assignment"]["distinct_stage_miners"])
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("hidden_state", serialized)

    def test_collect_and_clean_are_safe_and_dry_run_by_default(self) -> None:
        output_dir = self._tmp_dir()
        pack.build_report(pack.parse_args([
            "prepare",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
        ]))
        collect = pack.build_report(pack.parse_args([
            "collect",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
        ]), runner=check.fake_runner)
        clean = pack.build_report(pack.parse_args(["clean", "--output-dir", str(output_dir)]))

        self.assertTrue(collect["ok"], collect)
        self.assertIn("swarm_inference_beta_collect_ready", collect["diagnosis_codes"])
        self.assertTrue(clean["ok"], clean)
        self.assertEqual(clean["cleanup_mode"], "dry_run")
        self.assertTrue((output_dir / "stage0" / "miner_join.sh").exists())
        self.assertIn("dry_run", {candidate["action"] for candidate in clean["candidates"]})

    def test_miner_command_requires_stage_env_and_redacts_token(self) -> None:
        output_dir = self._tmp_dir()
        pack.build_report(pack.parse_args([
            "prepare",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
        ]))

        report = pack.build_report(pack.parse_args([
            "miner",
            "--output-dir",
            str(output_dir),
            "--stage",
            "stage0",
        ]))
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "miner")
        self.assertIn("--real-llm-stage-role", report["command"])
        self.assertIn("stage0", report["command"])
        self.assertNotIn("CROWDTENSOR_MINER_TOKEN", serialized)

    def test_live_wraps_real_internet_beta_and_support_bundle(self) -> None:
        output_dir = self._tmp_dir()
        for relative in pack.LIVE_PRIVATE_RELATIVE_PATHS:
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("secret", encoding="utf-8")
        for relative in pack.LIVE_TRANSIENT_RELATIVE_DIRS:
            path = output_dir / relative
            path.mkdir(parents=True, exist_ok=True)
            if path.name == "package-live-rc":
                state_dir = path / "coordinator-state"
                state_dir.mkdir(parents=True, exist_ok=True)
                (state_dir / "tasks.jsonl").write_text('{"lease_token":"secret"}\n', encoding="utf-8")
            else:
                (path / "kernel.py").write_text("secret kernel", encoding="utf-8")
        args = pack.parse_args([
            "live",
            "--output-dir",
            str(output_dir),
            "--public-host",
            "24.199.118.54",
            "--port",
            "9210",
            "--base-port",
            "9211",
            "--kaggle-owner",
            "xuyuhaosuyi",
        ])

        report = pack.build_report(args, runner=check.fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "live")
        self.assertEqual(report["live_mode"], "kaggle-auto")
        for code in [
            "swarm_inference_beta_ready",
            "swarm_inference_beta_live_ready",
            "two_machine_swarm_inference_ready",
            "real_llm_internet_beta_ready",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "token_rotation_required",
            "swarm_inference_beta_support_bundle_ready",
        ]:
            self.assertIn(code, report["diagnosis_codes"])
        self.assertTrue(report["real_llm_internet_beta_summary"]["runtime_classification"]["external_runtime_verified"])
        self.assertTrue(report["real_llm_internet_beta_summary"]["kaggle_lifecycle"]["kernels_deleted"])
        self.assertTrue(report["artifacts"]["real_llm_internet_beta_json"]["present"])
        self.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
        self.assertIn("swarm_inference_beta_live_private_artifacts_cleaned", report["diagnosis_codes"])
        self.assertTrue(report["safety"]["local_private_artifacts_removed"])
        self.assertTrue(report["safety"]["raw_runtime_state_removed"])
        self.assertGreater(report["live_cleanup_summary"]["deleted_count"], 0)
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
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
        markdown = (output_dir / "swarm_inference_beta_live.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        child_report = json.loads((output_dir / "real-internet-beta" / "real_llm_internet_beta.json").read_text(encoding="utf-8"))
        self.assertTrue(child_report["artifacts"]["kept_json"]["present"])
        self.assertFalse(child_report["artifacts"]["deleted_runtime_state"]["present"])
        for relative in pack.LIVE_PRIVATE_RELATIVE_PATHS:
            self.assertFalse((output_dir / relative).exists(), relative)
        for relative in pack.LIVE_TRANSIENT_RELATIVE_DIRS:
            self.assertFalse((output_dir / relative).exists(), relative)
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("stage0-secret", serialized)
        self.assertNotIn("hidden_state", serialized)

    def test_live_can_keep_private_artifacts_for_debugging(self) -> None:
        output_dir = self._tmp_dir()
        private_file = output_dir / pack.LIVE_PRIVATE_RELATIVE_PATHS[0]
        private_file.parent.mkdir(parents=True, exist_ok=True)
        private_file.write_text("secret", encoding="utf-8")
        args = pack.parse_args([
            "live",
            "--output-dir",
            str(output_dir),
            "--public-host",
            "24.199.118.54",
            "--keep-live-private-artifacts",
        ])

        report = pack.build_report(args, runner=check.fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(private_file.exists())
        self.assertIn("swarm_inference_beta_live_private_artifacts_retained", report["diagnosis_codes"])
        self.assertFalse(report["safety"]["local_private_artifacts_removed"])
        self.assertFalse(report["safety"]["raw_runtime_state_removed"])

    def test_check_contract(self) -> None:
        report = check.build_check(check.parse_args([]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "swarm_inference_beta_check_v1")
        self.assertIn("swarm_inference_beta_check_ready", report["diagnosis_codes"])
        self.assertIn("swarm_inference_beta_live_ready", report["diagnosis_codes"])
        self.assertFalse(report["sensitive_leaks"])
        self.assertEqual(report["scope_errors"], [])


if __name__ == "__main__":
    unittest.main()
