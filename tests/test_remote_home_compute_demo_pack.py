from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import remote_home_compute_demo_pack as pack


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class RemoteHomeComputeDemoPackTests(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="crowdtensor_remote_home_pack_test_")

    def test_prepare_wraps_runbook_and_writes_summary(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_demo_runbook_pack.py", command[1])
            (output_dir / "remote_demo_runbook.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_demo_runbook.md").write_text("# Runbook\n", encoding="utf-8")
            (output_dir / "operator.private.env").write_text("export CROWDTENSOR_ADMIN_TOKEN='secret'\n", encoding="utf-8")
            (output_dir / "miner.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='secret'\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "remote_demo_runbook_v1",
                "demo": {
                    "coordinator_url": "https://coord.example",
                    "workload_type": "model_bundle_infer",
                    "route": "remote_python_model_bundle_infer",
                    "request_count": 4,
                    "scenario_schema": "model_bundle_inference_scenario_v1",
                    "scenario_id": "route-baseline",
                },
                "safety": {
                    "registry_hashed": True,
                    "public_artifact_redacted": True,
                },
            })

        args = pack.parse_args([
            "prepare",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
            "--replace",
        ])
        report = pack.build_prepare(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "remote_home_compute_demo_v1")
        self.assertEqual(report["mode"], "prepare")
        self.assertEqual(report["runbook_summary"]["schema"], "remote_demo_runbook_v1")
        self.assertIn("remote_home_compute_prepare_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["operator_private_env"]["present"])
        self.assertTrue((output_dir / "remote_home_compute_demo.json").is_file())
        self.assertTrue(any("--replace" in command for command in calls))

    def test_verify_wraps_acceptance_and_adds_ready_code(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_demo_acceptance_pack.py", command[1])
            (output_dir / "remote_demo_acceptance.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_demo_acceptance.md").write_text("# Acceptance\n", encoding="utf-8")
            (output_dir / "remote_compute_evidence.json").write_text("{}", encoding="utf-8")
            (output_dir / "support_bundle.json").write_text("{}", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "remote_demo_acceptance_v1",
                "diagnosis_codes": ["acceptance_ready"],
                "scenario": {
                    "scenario_schema": "model_bundle_inference_scenario_v1",
                    "scenario_id": "route-baseline",
                },
                "session_request": {
                    "created": True,
                    "task_id": "task-1",
                    "request_count": 4,
                },
                "evidence_summary": {
                    "schema": "remote_compute_evidence_v1",
                    "ok": True,
                },
                "observability_summary": {
                    "schema": "remote_demo_observability_v1",
                    "work_queue": {"accepted_results": 1},
                    "inference": {"requests_per_second": 12.0},
                },
            })

        args = pack.parse_args([
            "verify",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
        ])
        report = pack.build_verify(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "verify")
        self.assertEqual(report["acceptance_summary"]["schema"], "remote_demo_acceptance_v1")
        self.assertEqual(report["acceptance_summary"]["evidence_schema"], "remote_compute_evidence_v1")
        self.assertIn("acceptance_ready", report["diagnosis_codes"])
        self.assertIn("remote_home_compute_ready", report["diagnosis_codes"])
        self.assertTrue(any("--create-session" in command for command in calls))
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_verify_failure_preserves_diagnosis_and_redacts_stderr(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout=json.dumps({"ok": False, "diagnosis_codes": ["observer_auth_failed"]}) + "\n",
                stderr="observer-secret rejected",
            )

        args = pack.parse_args([
            "verify",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
        ])
        report = pack.build_verify(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertFalse(report["ok"])
        self.assertIn("observer_auth_failed", report["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)


if __name__ == "__main__":
    unittest.main()
