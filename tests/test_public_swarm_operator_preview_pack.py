from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_operator_preview_check as check
from scripts import public_swarm_operator_preview_pack as pack


class PublicSwarmOperatorPreviewPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_operator_preview_test_"))

    def test_check_builds_ready_local_smoke(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-smoke",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_operator_preview_check_v1")
        report = json.loads((Path(result["output_dir"]) / "operator-preview" / "public_swarm_operator_preview.json").read_text(encoding="utf-8"))
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
        markdown = (Path(result["output_dir"]) / "operator-preview" / "public_swarm_operator_preview.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn(
            "- prompt scope: `source=prompt-text count=1 inline_prompt_text=True terminal_next_commands_local_private=True saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- prompt scope note: This Operator Preview artifact records inherited prompt source/count and placeholder safety only; raw prompt text is excluded from public JSON, Markdown, and support bundles.",
            markdown,
        )
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not a local answer transcript", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        support = json.loads((Path(result["output_dir"]) / "operator-preview" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["prompt_scope"], report["prompt_scope"])
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

    def test_check_builds_ready_live_kaggle(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "live-kaggle",
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

    def test_import_dual_live_evidence_requires_both_stage_reports(self) -> None:
        output_dir = self._tmp_dir()
        stage0 = output_dir / "stage0.json"
        stage1 = output_dir / "stage1.json"
        check.write_json(stage0, check.fake_live_payload(mode="live-kaggle", stage="stage0"))
        check.write_json(stage1, check.fake_live_payload(mode="live-kaggle", stage="stage1"))
        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "preview"),
            "--live-stage0-report",
            str(stage0),
            "--live-stage1-report",
            str(stage1),
            "--release-readiness-report",
            str(output_dir / "missing-release.json"),
            "--developer-preview-report",
            str(output_dir / "missing-developer.json"),
        ])

        summary, payload = pack.import_dual_live_evidence(args)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(payload["live_preview"]["stage0_live_requeue_ready"])
        self.assertTrue(payload["live_preview"]["stage1_live_requeue_ready"])

    def test_live_kaggle_falls_back_to_retained_evidence_when_external_runtime_blocked(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        developer_report = output_dir / "developer.json"
        release_report = output_dir / "release.json"
        stage0 = output_dir / "stage0.json"
        stage1 = output_dir / "stage1.json"
        check.fake_gpu_report(gpu_report)
        check.write_json(developer_report, check.fake_developer_payload("local"))
        check.write_json(release_report, check.fake_release_payload())
        check.write_json(stage0, check.fake_live_payload(mode="live-kaggle", stage="stage0"))
        check.write_json(stage1, check.fake_live_payload(mode="live-kaggle", stage="stage1"))

        def blocked_live_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "public_swarm_live_preview_rc_pack.py" in joined and "live-kaggle" in command:
                return check.completed({
                    "schema": pack.LIVE_PREVIEW_SCHEMA,
                    "ok": False,
                    "mode": "live-kaggle",
                    "diagnosis_codes": ["hf_dependencies_missing"],
                }, returncode=1)
            return check.fake_runner(command)

        args = pack.parse_args([
            "live-kaggle",
            "--output-dir",
            str(output_dir / "preview"),
            "--gpu-report",
            str(gpu_report),
            "--developer-preview-report",
            str(developer_report),
            "--release-readiness-report",
            str(release_report),
            "--live-stage0-report",
            str(stage0),
            "--live-stage1-report",
            str(stage1),
            "--kaggle-owner",
            "owner",
            "--allow-dirty-release",
            "--json",
        ])

        report = pack.build_report(args, runner=blocked_live_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["operator_preview"]["external_runtime_blocked"])
        self.assertIn("external_runtime_blocked", report["diagnosis_codes"])
        retained = report["payload_summaries"]["retained_live_evidence"]
        self.assertTrue(retained["ok"], retained)
        live_attempt = report["payload_summaries"]["live_attempt"]
        self.assertFalse(live_attempt["ok"], live_attempt)

    def test_local_smoke_degrades_to_cpu_fallback_when_hf_is_missing(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        release_report = output_dir / "release.json"
        check.fake_gpu_report(gpu_report)
        check.write_json(release_report, check.fake_release_payload())

        def degraded_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "public_swarm_developer_preview_pack.py" in joined:
                payload = check.fake_developer_payload("local")
                payload["ok"] = False
                payload["developer_preview"]["ready"] = False
                payload["developer_preview"]["product_beta_ready"] = False
                payload["developer_preview"]["support_bundle_ready"] = False
                payload["diagnosis_codes"] = [
                    code for code in payload["diagnosis_codes"] if code != "serve_join_generate_ready"
                ]
                payload["diagnosis_codes"].extend([
                    "developer_preview_blocked",
                    "hf_dependencies_missing",
                    "public_swarm_product_beta_blocked",
                ])
                return check.completed(payload, returncode=1)
            return check.fake_runner(command)

        args = pack.parse_args([
            "local-smoke",
            "--output-dir",
            str(output_dir / "preview"),
            "--gpu-report",
            str(gpu_report),
            "--release-readiness-report",
            str(release_report),
            "--allow-dirty-release",
            "--json",
        ])

        report = pack.build_report(args, runner=degraded_runner)

        self.assertTrue(report["ok"], report)
        preview = report["operator_preview"]
        self.assertFalse(preview["developer_preview_ready"])
        self.assertTrue(preview["developer_preview_degraded"])
        self.assertTrue(preview["cpu_fallback_ready"])
        self.assertTrue(preview["cpu_fallback_user_path_ready"])
        self.assertTrue(preview["user_path_ready"])
        self.assertFalse(preview["serve_join_generate_ready"])
        self.assertIn("operator_preview_cpu_fallback_user_path_ready", report["diagnosis_codes"])
        self.assertIn("developer_preview_degraded", report["diagnosis_codes"])
        self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])

    def test_evidence_import_synthesizes_developer_summary_from_retained_live(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        release_report = output_dir / "release.json"
        missing_developer = output_dir / "missing-developer.json"
        stage0 = output_dir / "stage0.json"
        stage1 = output_dir / "stage1.json"
        check.fake_gpu_report(gpu_report)
        check.write_json(release_report, check.fake_release_payload())
        check.write_json(stage0, check.fake_live_payload(mode="live-kaggle", stage="stage0"))
        check.write_json(stage1, check.fake_live_payload(mode="live-kaggle", stage="stage1"))

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "preview"),
            "--gpu-report",
            str(gpu_report),
            "--developer-preview-report",
            str(missing_developer),
            "--live-stage0-report",
            str(stage0),
            "--live-stage1-report",
            str(stage1),
            "--release-readiness-report",
            str(release_report),
            "--allow-dirty-release",
            "--json",
        ])

        report = pack.build_report(args)

        self.assertTrue(report["ok"], report)
        preview = report["operator_preview"]
        self.assertTrue(preview["developer_preview_ready"])
        self.assertTrue(preview["retained_evidence_ready"])
        self.assertTrue(preview["retained_evidence_user_path_ready"])
        self.assertTrue(preview["user_path_ready"])
        self.assertIn("operator_preview_retained_evidence_ready", report["diagnosis_codes"])
        self.assertIn("developer_preview_retained_evidence_ready", report["diagnosis_codes"])

    def test_report_redacts_sensitive_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": "local-smoke",
                "diagnosis_codes": ["public_swarm_operator_preview_ready"],
                "operator_preview": {"ready": True},
                "steps": [],
                "payload_summaries": {},
                "safety": {},
                "limitations": [],
                "secret": "CROWDTENSOR_ADMIN_TOKEN=admin-secret",
                "artifacts": pack.base_artifacts(output_dir, ok=True),
                "gpu_report_path": str(output_dir / "gpu.json"),
            },
            output_dir=output_dir,
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=admin-secret", encoded)
        self.assertTrue(report["ok"])
        self.assertFalse(report["output_request"]["include_output"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")


if __name__ == "__main__":
    unittest.main()
