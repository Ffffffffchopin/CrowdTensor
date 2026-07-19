from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from scripts import control_user_alpha_check as check
from scripts import control_user_alpha_pack as pack


class ControlUserAlphaTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_control_user_alpha_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _core_reports(self, base: Path, *, leak: bool = False) -> tuple[Path, Path]:
        handoff = {
            "schema": "core_technology_handoff_rc_v1",
            "ok": True,
            "core_technology_large_model_alpha_ready": True,
            "safety": {"public_artifact_safe": True},
            "next_layer_integration_contract": {
                "ready": True,
                "control_layer": {"route_health_schema": "large_model_route_health_v1"},
                "user_layer": {"answer_visibility": "public artifacts expose digests only"},
                "permissions_trust_billing_layer": {"core_signals": ["runtime_backend", "model_id"]},
            },
            "large_model_stage_selective_evidence": {
                "core_technology_large_model_alpha_ready": True,
                "evidence_scope": "live-kaggle-stage-selective",
                "checks": {
                    "seven_b_multi_token_verified": True,
                    "fourteen_b_dual_kaggle_verified": True,
                    "n_stage_partition_plan_ready": True,
                    "stage_selective_performance_report_ready": True,
                },
                "limitations": [
                    "This is not production P2P, not arbitrary public prompt serving, and not unbounded GPU pooling."
                ],
            },
        }
        if leak:
            handoff["generated_text"] = "do-not-save-this"
        status = {
            "schema": "core_technology_validation_status_v1",
            "ok": True,
            "core_validation_ready": True,
            "largest_successful_tier": "14b",
            "safety": {"public_artifact_safe": True},
            "handoff_stage_selective_evidence": {
                "schema": "core_technology_handoff_rc_v1",
                "ready": True,
                "evidence_scope": "live-kaggle-stage-selective-handoff",
                "seven_b_model_id": "Qwen/Qwen2.5-7B-Instruct",
                "seven_b_multi_token_verified": True,
                "seven_b_generated_token_count": 2,
                "fourteen_b_model_id": "Qwen/Qwen2.5-14B-Instruct",
                "fourteen_b_dual_kaggle_verified": True,
                "fourteen_b_generated_token_count": 1,
                "n_stage_partition_plan_ready": True,
                "target_stage_count": 4,
                "stage_weight_downloads_only_stage_files": True,
                "stage_selective_performance_report_ready": True,
                "tokens_per_second_effective": 0.003783,
                "latency_effective_elapsed_seconds": 264.345504,
                "limitations": [
                    "This is not production P2P, not arbitrary public prompt serving, and not unbounded GPU pooling."
                ],
            },
        }
        return (
            self._write_json(base / "core_handoff.json", handoff),
            self._write_json(base / "core_status.json", status),
        )

    def test_pack_consumes_core_evidence_into_control_and_user_ready_report(self) -> None:
        base = self._tmp_dir()
        handoff, status = self._core_reports(base)
        output_dir = base / "alpha"

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--core-handoff-report",
            str(handoff),
            "--core-status-report",
            str(status),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["core_handoff_imported"])
        self.assertTrue(report["core_validation_status_imported"])
        self.assertTrue(report["control_layer_ready"])
        self.assertTrue(report["user_layer_ready"])
        self.assertTrue(report["model_catalog_ready"])
        self.assertTrue(report["session_lifecycle_ready"])
        self.assertTrue(report["user_safe_inference_entrypoint_ready"])
        self.assertTrue(report["public_artifact_safe"])
        self.assertEqual(pack.public_redaction_errors(report), [])
        self.assertEqual(check.validate_report(report), [])

        model_ids = {item["model_id"] for item in report["model_catalog"]["models"]}
        self.assertEqual(
            {"Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"},
            model_ids,
        )
        events = [item["event"] for item in report["control_layer"]["session_lifecycle"]["events"]]
        self.assertEqual(["create", "list", "get", "cancel"], events)
        operations = [
            item["operation"]
            for item in report["control_layer"]["session_lifecycle"]["operations"]
        ]
        self.assertEqual(["create", "list", "get", "cancel"], operations)
        self.assertTrue(report["control_layer"]["scheduler"]["route"]["usable_now"])
        self.assertEqual(
            report["user_layer"]["answer_scope"]["scope_state"],
            "saved-terminal-redacted",
        )
        for name in ["control_user_alpha.json", "control_user_alpha.md", "support_bundle.json"]:
            self.assertTrue((output_dir / name).is_file(), name)

    def test_source_redaction_failure_blocks_public_artifact_ready(self) -> None:
        base = self._tmp_dir()
        handoff, status = self._core_reports(base, leak=True)

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(base / "leaky-alpha"),
            "--core-handoff-report",
            str(handoff),
            "--core-status-report",
            str(status),
        ]))

        self.assertFalse(report["ok"])
        self.assertFalse(report["public_artifact_safe"])
        self.assertIn("public_artifact_redaction_failed", report["diagnosis_codes"])
        self.assertTrue(report["safety"]["input_public_leak_paths"])
        errors = check.validate_report(report)
        self.assertIn("public_artifact_safe_missing", errors)

    def test_check_script_builds_and_validates_report(self) -> None:
        base = self._tmp_dir()
        handoff, status = self._core_reports(base)

        result = check.build_check(check.parse_args([
            "--output-dir",
            str(base / "check-alpha"),
            "--core-handoff-report",
            str(handoff),
            "--core-status-report",
            str(status),
        ]))

        self.assertTrue(result["ok"])
        self.assertTrue(result["control_layer_ready"])
        self.assertTrue(result["user_layer_ready"])
        self.assertEqual(result["errors"], [])

    def test_cli_wrapper_generates_one_command_smoke_summary(self) -> None:
        base = self._tmp_dir()
        handoff, status = self._core_reports(base)
        output_dir = base / "cli-alpha"

        args = cli.parse_args([
            "control-user-alpha",
            "--output-dir",
            str(output_dir),
            "--core-handoff-report",
            str(handoff),
            "--core-status-report",
            str(status),
            "--json",
        ])
        summary = cli.build_control_user_alpha(args)

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["cli_schema"], "control_user_alpha_cli_v1")
        self.assertEqual(summary["schema"], pack.SCHEMA)
        self.assertTrue(summary["control_layer_ready"])
        self.assertTrue(summary["user_layer_ready"])
        self.assertTrue(summary["public_artifact_safe"])
        self.assertTrue((output_dir / "control_user_alpha_cli_summary.json").is_file())

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_control_user_alpha(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor Control/User Alpha", output)
        self.assertIn("control=True", output)
        self.assertIn("user=True", output)

    def test_argument_validation_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args(["--max-new-tokens", "0"])
        with self.assertRaises(SystemExit):
            pack.parse_args(["--core-handoff-report", "/tmp/does-not-exist-control-user-alpha.json"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["control-user-alpha", "--max-new-tokens", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "control-user-alpha",
                "--core-handoff-report",
                "/tmp/does-not-exist-control-user-alpha.json",
            ])


if __name__ == "__main__":
    unittest.main()
