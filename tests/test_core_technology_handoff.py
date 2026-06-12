from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from crowdtensor import core_technology_handoff as handoff
from crowdtensor import large_model_inference_rc as inference_rc
from scripts import core_technology_handoff_check as check
from scripts import core_technology_handoff_pack as pack


class CoreTechnologyHandoffTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_core_handoff_test_"))

    def test_pack_report_is_ci_safe_and_check_validates_it(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["--output-dir", str(output_dir), "--mode", "fixture"]))

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], handoff.HANDOFF_SCHEMA)
        self.assertFalse(report["real_runtime_verified"])
        self.assertFalse(report["real_7b_runtime_verified"])
        self.assertEqual(report["inference_rc_report"]["schema"], inference_rc.RC_SCHEMA)
        self.assertIn("core_technology_handoff_rc_ready", report["diagnosis_codes"])
        self.assertIn("external_real_runtime_resources_required", report["blockers"])
        self.assertTrue(report["safety"]["public_artifact_safe"])
        self.assertEqual(handoff.public_redaction_errors(report), [])
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], report["artifact_summary"]["artifact_count"])
        check.validate_report(report)

        for name in [
            "core_technology_handoff_rc.json",
            "core_technology_handoff_rc.md",
            "deployment_runbook.json",
            "next_layer_contract.json",
            "adapter_conformance.json",
            "test_gate_summary.json",
            "support_bundle.json",
            "inference-rc/core_technology_inference_rc.json",
        ]:
            self.assertTrue((output_dir / name).is_file(), name)

    def test_handoff_contracts_cover_deployment_adapters_and_next_layers(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["--output-dir", str(output_dir)]))
        deployment = report["deployment_runbook"]
        next_layer = report["next_layer_integration_contract"]
        adapter = report["adapter_conformance"]
        tests = report["test_gate_summary"]

        self.assertEqual(deployment["schema"], handoff.DEPLOYMENT_RUNBOOK_SCHEMA)
        self.assertTrue(deployment["local_fixture"]["ci_safe"])
        self.assertEqual(deployment["local_real_runtime"]["max_new_tokens_max"], 8)
        self.assertIn("process_leak_check", deployment["cleanup"])
        self.assertTrue(deployment["lan_vpn_two_worker_runtime"]["controlled_network_only"])

        self.assertEqual(next_layer["schema"], handoff.NEXT_LAYER_CONTRACT_SCHEMA)
        self.assertTrue(next_layer["ready"])
        self.assertIn("crowdtensor large-model-shard-rc", next_layer["control_layer"]["stable_entrypoints"])
        self.assertEqual(next_layer["sample_control_request"]["raw_prompt_public"], False)
        self.assertIn("runner_result.real_runtime_verified", next_layer["permissions_trust_billing_layer"]["core_signals"])
        self.assertIn("output_digest", next_layer["correctness_contract"])

        self.assertEqual(adapter["schema"], handoff.ADAPTER_CONFORMANCE_SCHEMA)
        self.assertTrue(adapter["ready"])
        self.assertEqual(set(adapter["future_runtime_backends"]), set(inference_rc.UNSUPPORTED_RUNTIMES))
        self.assertTrue(all(item["conformant"] for item in adapter["descriptor_checks"]))

        self.assertEqual(tests["schema"], handoff.TEST_GATE_SCHEMA)
        self.assertIn("deployment/runbook artifact generation", tests["coverage"])
        self.assertIn("backward compatibility", tests["coverage"])

    def test_real_run_import_marks_handoff_real_verified(self) -> None:
        output_dir = self._tmp_dir()
        real_run_path = output_dir / "real_run.json"
        real_run_path.write_text(
            json.dumps({
                "metrics": {
                    "ttft_ms": 111.0,
                    "tokens_per_second": 9.25,
                    "wall_time_seconds": 1.1,
                    "generated_token_count": 3,
                    "max_new_tokens": 3,
                    "output_digest": "sha256:" + "2" * 64,
                }
            }),
            encoding="utf-8",
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--real-run-report",
            str(real_run_path),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["real_runtime_verified"])
        self.assertTrue(report["real_7b_runtime_verified"])
        self.assertEqual(report["runner_result"]["runner_mode"], "real-import")
        self.assertIn("core_technology_real_runtime_verified", report["diagnosis_codes"])
        self.assertNotIn("external_real_runtime_resources_required", report["blockers"])
        check.validate_report(report)

    def test_cli_wrapper_generates_handoff_summary(self) -> None:
        output_dir = self._tmp_dir()
        args = cli.parse_args(["core-tech-handoff", "--output-dir", str(output_dir)])
        summary = cli.build_core_technology_handoff_rc(args)

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["cli_schema"], "core_technology_handoff_rc_cli_v1")
        self.assertEqual(summary["schema"], handoff.HANDOFF_SCHEMA)
        self.assertFalse(summary["real_runtime_verified"])
        self.assertTrue((output_dir / "core_technology_handoff_rc_cli_summary.json").is_file())
        self.assertEqual(summary["artifact_summary"]["present_artifact_count"], summary["artifact_summary"]["artifact_count"])

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_core_technology_handoff_rc(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor core technology Handoff RC", output)
        self.assertIn("real_7b=False", output)
        self.assertIn("next_layer", output)

    def test_cli_rejects_bad_handoff_args(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--layer-count", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--reserved-kv-cache-mb", "-1"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--mode", "real", "--max-new-tokens", "9"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--mode", "real", "--real-timeout-seconds", "1201"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["core-tech-handoff", "--real-run-report", "/tmp/does-not-exist.json"])


if __name__ == "__main__":
    unittest.main()
