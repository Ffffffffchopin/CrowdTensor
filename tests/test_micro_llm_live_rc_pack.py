from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts import micro_llm_live_rc_pack as pack


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class FakeProcess:
    _next_pid = 1000

    def __init__(self, command: list[str], **_: object) -> None:
        self.command = command
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, _: int) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self.returncode is None:
            self.returncode = 0
        return ("accepted task\n", "")


class MicroLlmLiveRcPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_micro_live_rc_test_"))

    def test_local_generated_runs_prepare_verify_and_summarizes_stage_uploads(self) -> None:
        output_dir = self._tmp_dir()
        kaggle_dir = output_dir / "kaggle-real-runtime"
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            mode = command[2]
            if mode == "prepare":
                remote_dir = kaggle_dir / "remote-home-compute"
                remote_dir.mkdir(parents=True, exist_ok=True)
                (remote_dir / "operator.private.env").write_text(
                    "export CROWDTENSOR_OBSERVER_TOKEN='observer-secret'\nexport CROWDTENSOR_ADMIN_TOKEN='admin-secret'\n",
                    encoding="utf-8",
                )
                (remote_dir / "miner.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='stage0-secret'\n", encoding="utf-8")
                (remote_dir / "miner.stage1.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='stage1-secret'\n", encoding="utf-8")
                (kaggle_dir / "start_coordinator.sh").write_text("#!/usr/bin/env bash\nsleep 1\n", encoding="utf-8")
                for stage in ["stage0", "stage1"]:
                    upload = kaggle_dir / f"kaggle-upload-{stage}"
                    upload.mkdir(parents=True, exist_ok=True)
                    (upload / "miner.private.env").write_text("x\n", encoding="utf-8")
                    (upload / "kaggle_remote_miner.py").write_text(f"--micro-llm-stage-role {stage}\n", encoding="utf-8")
                    (upload / "KAGGLE_RUN.md").write_text("run\n", encoding="utf-8")
                return completed({
                    "schema": "kaggle_real_runtime_acceptance_v1",
                    "ok": True,
                    "mode": "prepare",
                    "diagnosis_codes": ["kaggle_artifacts_ready"],
                })
            if mode == "verify":
                remote_dir = kaggle_dir / "remote-home-compute"
                (remote_dir / "remote_micro_llm_sharded_beta.json").write_text(
                    json.dumps({
                        "schema": "remote_micro_llm_sharded_beta_v1",
                        "ok": True,
                        "diagnosis_codes": ["stage_assignment_valid"],
                    }),
                    encoding="utf-8",
                )
                (remote_dir / "support_bundle.json").write_text(json.dumps({"schema": "support_bundle_v1", "ok": True}), encoding="utf-8")
                return completed({
                    "schema": "kaggle_real_runtime_acceptance_v1",
                    "ok": True,
                    "mode": "verify",
                    "diagnosis_codes": [
                        "kaggle_artifacts_ready",
                        "coordinator_public_ready",
                        "kaggle_miner_seen",
                        "kaggle_result_accepted",
                        "kaggle_real_runtime_ready",
                        "kaggle_micro_llm_stage0_seen",
                        "kaggle_micro_llm_stage1_seen",
                        "kaggle_micro_llm_stage_assignment_valid",
                        "stage_assignment_valid",
                        "kaggle_micro_llm_sharded_result_accepted",
                        "kaggle_micro_llm_sharded_ready",
                    ],
                    "acceptance_summary": {"stage_assignment_valid": True},
                })
            raise AssertionError(command)

        args = pack.parse_args([
            "--mode",
            "local-generated",
            "--output-dir",
            str(output_dir),
            "--port",
            "9180",
            "--request-count",
            "2",
            "--decode-steps",
            "3",
        ])

        with mock.patch.object(pack, "wait_for_ready", return_value={"ok": True, "payload": {"ok": True}}):
            report = pack.build_report(args, runner=fake_runner, popen_factory=FakeProcess)

        serialized = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "micro_llm_live_rc_v1")
        self.assertEqual(report["mode"], "local-generated")
        self.assertIn("micro_llm_live_rc_ready", report["diagnosis_codes"])
        self.assertIn("local_generated_stage_upload_standins_ready", report["diagnosis_codes"])
        self.assertTrue(report["runtime_classification"]["local_generated_stage_upload_standins"])
        self.assertFalse(report["runtime_classification"]["external_runtime_verified"])
        self.assertEqual({item["stage_role"] for item in report["stage_packages"]}, {"stage0", "stage1"})
        self.assertEqual({item["name"] for item in report["process_summary"] if item["name"].endswith("_miner")}, {"stage0_miner", "stage1_miner"})
        self.assertNotIn("stage0-secret", serialized)
        self.assertNotIn("stage1-secret", serialized)
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(any(command[2] == "prepare" for command in calls))
        self.assertTrue(any(command[2] == "verify" for command in calls))

    def test_external_existing_marks_external_runtime_verified(self) -> None:
        output_dir = self._tmp_dir()
        kaggle_dir = output_dir / "kaggle-real-runtime"
        remote_dir = kaggle_dir / "remote-home-compute"
        remote_dir.mkdir(parents=True, exist_ok=True)
        (remote_dir / "operator.private.env").write_text(
            "export CROWDTENSOR_OBSERVER_TOKEN='observer-secret'\nexport CROWDTENSOR_ADMIN_TOKEN='admin-secret'\n",
            encoding="utf-8",
        )

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[2], "verify")
            self.assertIn("--coordinator-url", command)
            return completed({
                "schema": "kaggle_real_runtime_acceptance_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": [
                    "kaggle_micro_llm_stage0_seen",
                    "kaggle_micro_llm_stage1_seen",
                    "kaggle_micro_llm_stage_assignment_valid",
                    "stage_assignment_valid",
                    "kaggle_micro_llm_sharded_ready",
                ],
            })

        args = pack.parse_args([
            "--mode",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--kaggle-output-dir",
            str(kaggle_dir),
            "--coordinator-url",
            "http://24.199.118.54:9180",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])

        report = pack.build_report(args, runner=fake_runner, popen_factory=FakeProcess)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "external-existing")
        self.assertIn("external_runtime_verified", report["diagnosis_codes"])
        self.assertTrue(report["runtime_classification"]["external_runtime_verified"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)


if __name__ == "__main__":
    unittest.main()
