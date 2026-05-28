from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import real_llm_internet_alpha_pack as pack


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class RealLlmInternetAlphaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_real_internet_alpha_test_"))

    def _touch_live_artifacts(self, output_dir: Path) -> None:
        live_dir = output_dir / "live-rc"
        live_dir.mkdir(parents=True, exist_ok=True)
        (live_dir / "real_llm_live_rc.json").write_text("{}", encoding="utf-8")
        (live_dir / "real_llm_live_rc.md").write_text("# Live\n", encoding="utf-8")

    def _touch_package_artifacts(self, output_dir: Path) -> None:
        live_dir = output_dir / "package-live-rc"
        live_dir.mkdir(parents=True, exist_ok=True)
        (live_dir / "real_llm_live_rc.json").write_text("{}", encoding="utf-8")
        (live_dir / "real_llm_live_rc.md").write_text("# Package\n", encoding="utf-8")

    def _touch_requeue_artifacts(self, output_dir: Path, failure_mode: str) -> None:
        child = output_dir / f"requeue-{failure_mode}"
        evidence = child / "remote-loopback-real-llm-shard-infer" / "real_llm_sharded_evidence.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        (child / "remote_real_llm_sharded_beta.json").write_text("{}", encoding="utf-8")
        (child / "remote_real_llm_sharded_beta.md").write_text("# Requeue\n", encoding="utf-8")
        evidence.write_text("{}", encoding="utf-8")

    def test_local_generated_aggregates_live_rc_and_stage_requeue(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            script = Path(command[1]).name
            if script == "real_llm_live_rc_pack.py":
                self.assertEqual(command[command.index("--mode") + 1], "local-generated")
                self._touch_live_artifacts(output_dir)
                return completed({
                    "schema": "real_llm_live_rc_v1",
                    "ok": True,
                    "mode": "local-generated",
                    "diagnosis_codes": [
                        "real_llm_live_rc_ready",
                        "remote_real_llm_sharded_ready",
                        "real_llm_artifact_ready",
                        "stage_0_accepted",
                        "stage_1_accepted",
                        "activation_transport_ready",
                        "baseline_match",
                        "decoded_tokens_match",
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                    ],
                })
            if script == "remote_real_llm_sharded_beta_pack.py":
                failure_mode = command[command.index("--failure-mode") + 1]
                self.assertIn(failure_mode, {"kill-stage0-after-claim", "kill-stage1-after-claim"})
                self.assertIn("--require-distinct-stage-miners", command)
                self._touch_requeue_artifacts(output_dir, failure_mode)
                return completed({
                    "schema": "remote_real_llm_sharded_beta_v1",
                    "ok": True,
                    "mode": "remote-loopback",
                    "diagnosis_codes": [
                        "remote_real_llm_sharded_ready",
                        "stage_requeue_ready",
                        "stage_0_accepted",
                        "stage_1_accepted",
                    ],
                })
            raise AssertionError(command)

        args = pack.parse_args([
            "--mode",
            "local-generated",
            "--output-dir",
            str(output_dir),
            "--port",
            "9186",
            "--base-port",
            "9188",
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "real_llm_internet_alpha_v1")
        self.assertIn("real_llm_internet_alpha_ready", report["diagnosis_codes"])
        self.assertIn("real_llm_stage_requeue_ready", report["diagnosis_codes"])
        self.assertTrue(report["runtime_classification"]["stage_requeue_verified"])
        self.assertFalse(report["runtime_classification"]["external_runtime_verified"])
        self.assertTrue(report["artifacts"]["real_llm_live_rc_json"]["present"])
        self.assertTrue(report["artifacts"]["kill_stage0_after_claim_real_llm_sharded_evidence_json"]["present"])
        self.assertEqual(len(calls), 3)

    def test_package_maps_to_live_rc_kaggle_generated_without_ready_claim(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("real_llm_live_rc_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "kaggle-generated")
            self._touch_package_artifacts(output_dir)
            return completed({
                "schema": "real_llm_live_rc_v1",
                "ok": True,
                "mode": "kaggle-generated",
                "diagnosis_codes": [
                    "real_llm_live_rc_prepare_ready",
                    "kaggle_real_llm_stage_upload_packages_ready",
                    "real_llm_artifact_ready",
                ],
            })

        args = pack.parse_args(["--mode", "package", "--output-dir", str(output_dir)])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("real_llm_internet_alpha_package_ready", report["diagnosis_codes"])
        self.assertNotIn("real_llm_internet_alpha_ready", report["diagnosis_codes"])
        self.assertTrue(report["runtime_classification"]["package_only"])
        self.assertFalse(report["runtime_classification"]["external_runtime_verified"])

    def test_external_existing_marks_external_verified_and_redacts_tokens(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("real_llm_live_rc_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "external-existing")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self._touch_live_artifacts(output_dir)
            return completed({
                "schema": "real_llm_live_rc_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": [
                    "real_llm_live_rc_ready",
                    "external_runtime_verified",
                    "remote_real_llm_sharded_ready",
                ],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = pack.parse_args([
            "--mode",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://24.199.118.54:9186",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
        report = pack.build_report(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertIn("real_llm_internet_alpha_ready", report["diagnosis_codes"])
        self.assertTrue(report["runtime_classification"]["external_runtime_verified"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)


if __name__ == "__main__":
    unittest.main()
