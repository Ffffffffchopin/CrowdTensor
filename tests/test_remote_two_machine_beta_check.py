from __future__ import annotations

import argparse
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_two_machine_beta_check.py"
SPEC = importlib.util.spec_from_file_location("remote_two_machine_beta_check", SCRIPT_PATH)
check = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check)


class RemoteTwoMachineBetaCheckTests(unittest.TestCase):
    def test_build_report_runs_both_workloads_and_requires_ready_codes(self) -> None:
        calls: list[tuple[str, int]] = []

        def fake_run_workload(args: argparse.Namespace, *, workload: str, port: int, output_dir: Path) -> dict:
            calls.append((workload, port))
            return {
                "name": f"remote_two_machine_{workload}",
                "ok": True,
                "workload": workload,
                "port": port,
                "schema": "remote_home_compute_demo_check_v1",
                "diagnosis_codes": [check.expected_ready_code(workload)],
            }

        args = argparse.Namespace(
            workload="all",
            base_port=9050,
            request_count=2,
            external_llm_request_count=2,
            scenario_id="route-baseline",
            startup_timeout=10.0,
            verify_timeout=40.0,
            command_timeout=120.0,
            timeout_seconds=240.0,
        )
        with patch.object(check, "run_workload", side_effect=fake_run_workload):
            report = check.build_report(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "remote_two_machine_beta_check_v1")
        self.assertEqual(report["step_count"], 2)
        self.assertEqual(calls, [("model-bundle", 9050), ("external-llm", 9051)])
        self.assertIn("remote_two_machine_beta_ready", report["diagnosis_codes"])
        self.assertIn("remote_two_machine_inference_ready", report["diagnosis_codes"])
        self.assertIn("remote_two_machine_external_llm_ready", report["diagnosis_codes"])
        self.assertTrue(report["safety"]["local_loopback_standin"])

    def test_validate_remote_check_rejects_sensitive_output(self) -> None:
        payload = {
            "schema": "remote_home_compute_demo_check_v1",
            "ok": True,
            "workload": "model-bundle",
            "diagnosis_codes": ["remote_two_machine_inference_ready"],
            "leak": "lease_token",
        }

        with self.assertRaises(SystemExit):
            check.validate_remote_check(payload, workload="model-bundle")

    def test_validate_remote_check_rejects_missing_ready_code(self) -> None:
        payload = {
            "schema": "remote_home_compute_demo_check_v1",
            "ok": True,
            "workload": "external-llm",
            "diagnosis_codes": ["remote_external_llm_ready"],
        }

        with self.assertRaises(SystemExit):
            check.validate_remote_check(payload, workload="external-llm")

    def test_main_outputs_schema(self) -> None:
        with (
            patch.object(check, "build_report", return_value={
                "ok": True,
                "schema": "remote_two_machine_beta_check_v1",
                "diagnosis_codes": ["remote_two_machine_beta_ready"],
            }),
            patch("builtins.print") as mocked_print,
            patch("sys.argv", ["remote_two_machine_beta_check.py"]),
        ):
            with self.assertRaises(SystemExit) as raised:
                check.main()

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_two_machine_beta_check_v1")


if __name__ == "__main__":
    unittest.main()
