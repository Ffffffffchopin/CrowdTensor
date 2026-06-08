from __future__ import annotations

import json
import py_compile
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import real_llm_live_rc_pack as pack


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def assert_ready_guidance(testcase: unittest.TestCase, report: dict, output_dir: Path) -> None:
    testcase.assertEqual(report["user_status"]["state"], "ready")
    testcase.assertEqual(report["review_summary"]["schema"], "real_llm_live_rc_review_summary_v1")
    testcase.assertEqual(report["review_summary"]["state"], "ready")
    testcase.assertEqual(report["recommended_next_command"]["label"], "inspect real LLM Live RC evidence")
    testcase.assertIn("real_llm_live_rc.md", report["recommended_next_command"]["command_line"])
    testcase.assertEqual(report["not_completed"], [])
    testcase.assertGreaterEqual(len(report["next_commands"]), 3)
    testcase.assertTrue(report["artifact_summary"]["public_artifact_safe"])
    testcase.assertEqual(
        report["artifact_summary"]["present_artifact_count"],
        report["artifact_summary"]["artifact_count"],
    )
    testcase.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
    support = json.loads((output_dir / "support_bundle.json").read_text(encoding="utf-8"))
    testcase.assertEqual(support["schema"], "real_llm_live_rc_support_bundle_v1")
    testcase.assertEqual(support["review_summary"]["state"], "ready")
    testcase.assertTrue(support["public_artifact_safe"])


class FakeProcess:
    _next_pid = 3000

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
        return ("accepted real llm task\n", "")


class RealLlmLiveRcPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_real_live_rc_test_"))

    def _artifact(self) -> dict:
        return {
            "schema": "real_llm_artifact_v1",
            "artifact_hash": "hash-real",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cpu",
            "split_index": 1,
            "loaded": True,
        }

    def test_local_generated_runs_remote_existing_verify_and_summarizes_stage_uploads(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_real_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "remote-existing")
            self.assertIn("--require-distinct-stage-miners", command)
            runtime_dir = output_dir / "remote-real-llm-runtime"
            (runtime_dir / "remote-existing-real-llm-shard-infer").mkdir(parents=True, exist_ok=True)
            (runtime_dir / "remote_real_llm_sharded_beta.json").write_text(
                json.dumps({
                    "schema": "remote_real_llm_sharded_beta_v1",
                    "ok": True,
                    "diagnosis_codes": ["remote_real_llm_sharded_ready"],
                }),
                encoding="utf-8",
            )
            (runtime_dir / "remote-existing-real-llm-shard-infer" / "real_llm_sharded_evidence.json").write_text(
                json.dumps({"schema": "real_llm_sharded_evidence_v1", "ok": True}),
                encoding="utf-8",
            )
            return completed({
                "schema": "remote_real_llm_sharded_beta_v1",
                "ok": True,
                "mode": "remote-existing",
                "diagnosis_codes": [
                    "remote_real_llm_sharded_ready",
                    "remote_real_llm_sharded_existing_ready",
                    "real_llm_sharded_ready",
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

        args = pack.parse_args([
            "--mode",
            "local-generated",
            "--output-dir",
            str(output_dir),
            "--port",
            "9184",
        ])

        with mock.patch.object(pack, "inspect_real_llm_artifact", return_value=self._artifact()):
            with mock.patch.object(pack, "wait_for_ready", return_value={"ok": True, "payload": {"ok": True}}):
                report = pack.build_report(args, runner=fake_runner, popen_factory=FakeProcess)

        serialized = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "real_llm_live_rc_v1")
        self.assertEqual(report["mode"], "local-generated")
        self.assertIn("real_llm_live_rc_ready", report["diagnosis_codes"])
        self.assertIn("local_generated_real_llm_stage_upload_standins_ready", report["diagnosis_codes"])
        self.assertTrue(report["runtime_classification"]["local_generated_stage_upload_standins"])
        self.assertFalse(report["runtime_classification"]["external_runtime_verified"])
        self.assertEqual({item["stage_role"] for item in report["stage_packages"]}, {"stage0", "stage1"})
        for package in report["stage_packages"]:
            self.assertTrue(package["launcher_has_stage_role"])
            self.assertTrue(package["hf_runtime_enabled"])
        self.assertEqual(
            {item["name"] for item in report["process_summary"] if item["name"].endswith("_miner")},
            {"stage0_miner", "stage1_miner"},
        )
        self.assertNotIn("CrowdTensor routes", serialized)
        self.assertNotIn("A miner returns", serialized)
        self.assertIn("## Review", (output_dir / "real_llm_live_rc.md").read_text(encoding="utf-8"))
        self.assertIn("## What To Do Next", (output_dir / "real_llm_live_rc.md").read_text(encoding="utf-8"))
        self.assertIn("## Output Scope", (output_dir / "real_llm_live_rc.md").read_text(encoding="utf-8"))
        self.assertIn("## Artifact Summary", (output_dir / "real_llm_live_rc.md").read_text(encoding="utf-8"))
        self.assertIn("## Not Completed", (output_dir / "real_llm_live_rc.md").read_text(encoding="utf-8"))
        assert_ready_guidance(self, report, output_dir)
        self.assertTrue(calls)

    def test_kaggle_generated_prepares_stage_packages_without_claiming_external_runtime(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "--mode",
            "kaggle-generated",
            "--output-dir",
            str(output_dir),
            "--public-host",
            "24.199.118.54",
        ])

        with mock.patch.object(pack, "inspect_real_llm_artifact", return_value=self._artifact()):
            report = pack.build_report(args, popen_factory=FakeProcess)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "kaggle-generated")
        self.assertIn("real_llm_live_rc_prepare_ready", report["diagnosis_codes"])
        self.assertIn("kaggle_real_llm_stage_upload_packages_ready", report["diagnosis_codes"])
        self.assertFalse(report["runtime_classification"]["external_runtime_verified"])
        self.assertNotIn("real_llm_live_rc_ready", report["diagnosis_codes"])
        self.assertEqual(report["user_status"]["state"], "package-ready")
        self.assertEqual(report["review_summary"]["state"], "package-ready")
        self.assertEqual(report["review_summary"]["attention"], "external verification pending")
        self.assertEqual(report["recommended_next_command"]["label"], "verify external real LLM Live RC runtime")
        self.assertIn("external verification still pending", report["not_completed"])
        self.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
        for stage in ["stage0", "stage1"]:
            script = output_dir / f"kaggle-upload-real-llm-{stage}" / "kaggle_remote_miner.py"
            py_compile.compile(str(script), doraise=True)
        for package in report["stage_packages"]:
            self.assertTrue(package["launcher_syntax_valid"])

    def test_external_existing_marks_external_runtime_verified_and_redacts_tokens(self) -> None:
        output_dir = self._tmp_dir()
        inspect_calls: list[dict] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("remote_real_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            return completed({
                "schema": "remote_real_llm_sharded_beta_v1",
                "ok": True,
                "mode": "remote-existing",
                "diagnosis_codes": [
                    "remote_real_llm_sharded_ready",
                    "remote_real_llm_sharded_existing_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "real_llm_artifact_ready",
                    "real_llm_sharded_ready",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                    "distinct_stage_miners",
                    "stage_assignment_valid",
                ],
            })

        def fake_inspect(**kwargs: object) -> dict:
            inspect_calls.append(dict(kwargs))
            return self._artifact()

        args = pack.parse_args([
            "--mode",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://24.199.118.54:9184",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--real-llm-backend",
            "hf_transformers_cuda",
            "--real-llm-partition-mode",
            "stage-local",
        ])
        with mock.patch.object(pack, "inspect_real_llm_artifact", side_effect=fake_inspect):
            report = pack.build_report(args, runner=fake_runner, popen_factory=FakeProcess)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "external-existing")
        self.assertEqual(report["artifact"]["partition_mode"], "stage_local")
        self.assertEqual(inspect_calls[0]["backend"], "hf_transformers_cuda")
        self.assertFalse(inspect_calls[0]["require_runtime"])
        self.assertIn("external_runtime_verified", report["diagnosis_codes"])
        self.assertIn("kaggle_real_llm_sharded_ready", report["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        assert_ready_guidance(self, report, output_dir)


if __name__ == "__main__":
    unittest.main()
