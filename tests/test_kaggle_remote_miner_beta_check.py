from __future__ import annotations

import argparse
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "kaggle_remote_miner_beta_check.py"
SPEC = importlib.util.spec_from_file_location("kaggle_remote_miner_beta_check", SCRIPT_PATH)
check = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check)


class KaggleRemoteMinerBetaCheckTests(unittest.TestCase):
    def test_build_report_combines_prepare_and_loopback_steps(self) -> None:
        args = argparse.Namespace(
            coordinator_url="https://coord.example",
            miner_id="kaggle-cpu-1",
            port=9060,
            request_count=2,
            scenario_id="route-baseline",
            startup_timeout=10.0,
            verify_timeout=40.0,
            command_timeout=120.0,
            timeout_seconds=240.0,
        )

        with (
            patch.object(check, "run_prepare", return_value={"ok": True}),
            patch.object(check, "parse_private_env", side_effect=[
                {"CROWDTENSOR_OBSERVER_TOKEN": "observer-secret", "CROWDTENSOR_ADMIN_TOKEN": "admin-secret"},
                {"CROWDTENSOR_MINER_TOKEN": "miner-secret"},
            ]),
            patch.object(check, "validate_prepare", return_value={
                "name": "kaggle_prepare",
                "ok": True,
                "schema": "remote_home_compute_demo_v1",
                "diagnosis_codes": ["kaggle_remote_miner_prepare_ready"],
            }),
            patch.object(check, "run_loopback", return_value={"ok": True}),
            patch.object(check, "validate_loopback", return_value={
                "name": "kaggle_protocol_loopback",
                "ok": True,
                "schema": "remote_home_compute_demo_check_v1",
                "diagnosis_codes": ["remote_home_compute_ready"],
            }),
        ):
            report = check.build_report(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "kaggle_remote_miner_beta_check_v1")
        self.assertEqual(report["target"], "kaggle")
        self.assertEqual(report["step_count"], 2)
        self.assertIn("kaggle_remote_miner_beta_ready", report["diagnosis_codes"])
        self.assertIn("kaggle_remote_miner_prepare_ready", report["diagnosis_codes"])
        self.assertIn("remote_home_compute_ready", report["diagnosis_codes"])
        self.assertTrue(report["safety"]["kaggle_outbound_miner_only"])
        self.assertFalse(report["safety"]["gpu_tpu_workload_enabled"])

    def test_validate_prepare_rejects_token_leak(self) -> None:
        output_dir = Path("/tmp")
        payload = {
            "schema": "remote_home_compute_demo_v1",
            "ok": True,
            "target_environment": {
                "name": "kaggle",
                "kaggle_remote_miner_beta": True,
                "gpu_tpu_workload_enabled": False,
            },
            "diagnosis_codes": ["kaggle_remote_miner_prepare_ready"],
            "artifacts": {
                "kaggle_remote_miner_script": {"present": True},
                "kaggle_remote_miner_runbook": {"present": True},
                "miner_private_env": {"present": True},
                "operator_private_env": {"present": True},
            },
            "leak": "miner-secret",
        }

        with patch.object(Path, "read_text", side_effect=[
            "CROWDTENSOR_REMOTE_ENVIRONMENT='kaggle'\n",
            "Upload only `miner.private.env`\n",
        ]):
            with self.assertRaises(SystemExit):
                check.validate_prepare(
                    payload,
                    output_dir,
                    operator_env={"CROWDTENSOR_OBSERVER_TOKEN": "observer-secret"},
                    miner_env={"CROWDTENSOR_MINER_TOKEN": "miner-secret"},
                )

    def test_main_outputs_schema(self) -> None:
        with (
            patch.object(check, "build_report", return_value={
                "ok": True,
                "schema": "kaggle_remote_miner_beta_check_v1",
                "diagnosis_codes": ["kaggle_remote_miner_beta_ready"],
            }),
            patch("builtins.print") as mocked_print,
            patch("sys.argv", ["kaggle_remote_miner_beta_check.py"]),
        ):
            with self.assertRaises(SystemExit) as raised:
                check.main()

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "kaggle_remote_miner_beta_check_v1")


if __name__ == "__main__":
    unittest.main()
