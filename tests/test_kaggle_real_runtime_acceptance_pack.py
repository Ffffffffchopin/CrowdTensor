from __future__ import annotations

import json
import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

from crowdtensor.micro_llm_artifact import build_default_micro_llm_artifact
from scripts import kaggle_real_runtime_acceptance_check as check
from scripts import kaggle_real_runtime_acceptance_pack as pack


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class KaggleRealRuntimeAcceptancePackTests(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="crowdtensor_kaggle_real_pack_test_")

    def test_prepare_generates_kaggle_only_upload_package_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        remote_dir = output_dir / "remote-home-compute"
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            remote_dir.mkdir(parents=True, exist_ok=True)
            (remote_dir / "operator.private.env").write_text(
                "export CROWDTENSOR_ADMIN_TOKEN='admin-secret'\nexport CROWDTENSOR_OBSERVER_TOKEN='observer-secret'\n",
                encoding="utf-8",
            )
            (remote_dir / "miner.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='miner-secret'\n", encoding="utf-8")
            (remote_dir / "miner_registry.json").write_text(
                json.dumps({"miners": [{"miner_id": "kaggle-cpu-1", "token": "sha256:abc"}]}),
                encoding="utf-8",
            )
            (remote_dir / "kaggle_remote_miner.py").write_text("print('kaggle')\n", encoding="utf-8")
            (remote_dir / "remote_home_compute_demo.json").write_text("{}", encoding="utf-8")
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "prepare",
                "diagnosis_codes": ["kaggle_remote_miner_prepare_ready"],
                "target_environment": {"name": "kaggle", "kaggle_remote_miner_beta": True},
            })

        args = pack.parse_args([
            "prepare",
            "--public-host",
            "24.199.118.54",
            "--port",
            "9180",
            "--miner-id",
            "kaggle-cpu-1",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "2",
            "--miner-token",
            "miner-secret",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--replace",
        ])

        report = pack.build_prepare(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "kaggle_real_runtime_acceptance_v1")
        self.assertEqual(report["mode"], "prepare")
        self.assertEqual(report["coordinator_url"], "http://24.199.118.54:9180")
        self.assertIn("kaggle_artifacts_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["kaggle_upload_miner_env"]["present"])
        self.assertTrue(report["artifacts"]["kaggle_upload_miner_script"]["present"])
        self.assertFalse((output_dir / "kaggle-upload" / "operator.private.env").exists())
        self.assertIn("--host 0.0.0.0", (output_dir / "start_coordinator.sh").read_text(encoding="utf-8"))
        self.assertIn("--port 9180", (output_dir / "start_coordinator.sh").read_text(encoding="utf-8"))
        self.assertIn("sha256:", (output_dir / "start_coordinator.sh").read_text(encoding="utf-8"))
        self.assertNotIn("miner-secret", serialized)
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(any("--target" in command and "kaggle" in command for command in calls))

    def test_prepare_micro_llm_split_generates_two_kaggle_upload_packages(self) -> None:
        output_dir = Path(self._tmp_dir())
        artifact_dir = output_dir / "micro-llm-artifact"
        build_default_micro_llm_artifact(artifact_dir)
        remote_dir = output_dir / "remote-home-compute"
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            self.assertIn("--workload", command)
            self.assertEqual(command[command.index("--workload") + 1], "micro-llm-sharded")
            self.assertIn("--stage-role", command)
            self.assertEqual(command[command.index("--stage-role") + 1], "stage0")
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], str(artifact_dir))
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            remote_dir.mkdir(parents=True, exist_ok=True)
            (remote_dir / "operator.private.env").write_text(
                "export CROWDTENSOR_ADMIN_TOKEN='admin-secret'\nexport CROWDTENSOR_OBSERVER_TOKEN='observer-secret'\n",
                encoding="utf-8",
            )
            (remote_dir / "miner.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='stage0-secret'\n", encoding="utf-8")
            (remote_dir / "miner_registry.json").write_text(
                json.dumps({"miners": [{"miner_id": "kaggle-cpu-1-stage0", "token": "sha256:abc"}]}),
                encoding="utf-8",
            )
            (remote_dir / "kaggle_remote_miner.py").write_text("print('stage0')\n", encoding="utf-8")
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "prepare",
                "workload_kind": "micro-llm-sharded",
                "diagnosis_codes": ["remote_micro_llm_sharded_prepare_ready", "kaggle_remote_miner_prepare_ready"],
                "demo": {"workload_kind": "micro-llm-sharded"},
            })

        args = pack.parse_args([
            "prepare",
            "--workload",
            "micro-llm-sharded",
            "--stage-mode",
            "split",
            "--public-host",
            "24.199.118.54",
            "--port",
            "9180",
            "--miner-id",
            "kaggle-cpu-1",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "2",
            "--decode-steps",
            "3",
            "--micro-llm-artifact",
            str(artifact_dir),
            "--prompt-texts",
            "arn,ten",
            "--miner-token",
            "stage0-secret",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--replace",
        ])

        report = pack.build_prepare(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["workload"]["kind"], "micro-llm-sharded")
        self.assertEqual(report["workload"]["stage_mode"], "split")
        self.assertIn("kaggle_artifacts_ready", report["diagnosis_codes"])
        self.assertTrue((output_dir / "kaggle-upload-stage0" / "miner.private.env").is_file())
        self.assertTrue((output_dir / "kaggle-upload-stage1" / "miner.private.env").is_file())
        self.assertFalse((output_dir / "kaggle-upload-stage0" / "operator.private.env").exists())
        self.assertFalse((output_dir / "kaggle-upload-stage1" / "operator.private.env").exists())
        stage0_script = (output_dir / "kaggle-upload-stage0" / "kaggle_remote_miner.py").read_text(encoding="utf-8")
        stage1_script = (output_dir / "kaggle-upload-stage1" / "kaggle_remote_miner.py").read_text(encoding="utf-8")
        self.assertIn("--micro-llm-stage-role", stage0_script)
        self.assertIn("stage0", stage0_script)
        self.assertIn("--max-request-attempts", stage0_script)
        self.assertIn("120", stage0_script)
        self.assertIn("--idle-sleep", stage0_script)
        self.assertIn("stage1", stage1_script)
        registry = json.loads((remote_dir / "miner_registry.json").read_text(encoding="utf-8"))
        self.assertEqual({item["miner_id"] for item in registry["miners"]}, {"kaggle-cpu-1-stage0", "kaggle-cpu-1-stage1"})
        launch = (output_dir / "start_coordinator.sh").read_text(encoding="utf-8")
        self.assertIn("micro_llm_sharded_infer", launch)
        self.assertIn(f"--micro-llm-artifact {artifact_dir}", launch)
        operator_commands = (output_dir / "operator_commands.md").read_text(encoding="utf-8")
        self.assertIn(f"--micro-llm-artifact {artifact_dir}", operator_commands)
        self.assertIn("--prompt-texts arn,ten", operator_commands)
        self.assertNotIn("stage0-secret", serialized)
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_verify_aggregates_ready_codes_from_doctor_verify_and_collect(self) -> None:
        output_dir = Path(self._tmp_dir())
        remote_dir = output_dir / "remote-home-compute"
        remote_dir.mkdir(parents=True, exist_ok=True)
        (remote_dir / "operator.private.env").write_text(
            "export CROWDTENSOR_ADMIN_TOKEN='admin-secret'\nexport CROWDTENSOR_OBSERVER_TOKEN='observer-secret'\n",
            encoding="utf-8",
        )
        (remote_dir / "miner.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='miner-secret'\n", encoding="utf-8")
        upload = output_dir / "kaggle-upload"
        upload.mkdir()
        (upload / "miner.private.env").write_text("x\n", encoding="utf-8")
        (upload / "kaggle_remote_miner.py").write_text("x\n", encoding="utf-8")
        (upload / "KAGGLE_RUN.md").write_text("x\n", encoding="utf-8")
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            mode = command[2]
            if mode == "doctor":
                return completed({
                    "schema": "remote_home_compute_doctor_v1",
                    "ok": True,
                    "diagnosis_codes": ["remote_home_compute_doctor_ready"],
                    "connectivity": {
                        "observations": {"health": {"ok": True}, "ready": {"ok": True}, "state": {"ok": True}},
                        "matched_capabilities": ["runtime:python-cli", "backend:cpu", "workload:model_bundle_infer"],
                        "missing_capabilities": [],
                    },
                })
            if mode == "verify":
                return completed({
                    "schema": "remote_home_compute_demo_v1",
                    "ok": True,
                    "mode": "verify",
                    "diagnosis_codes": ["remote_home_compute_ready"],
                    "acceptance_summary": {"task_id": "task-1", "evidence_ok": True},
                })
            if mode == "collect":
                return completed({
                    "schema": "remote_home_compute_collect_v1",
                    "ok": True,
                    "diagnosis_codes": ["remote_home_compute_collect_ready"],
                    "status_summary": {"ready": True, "task_id": "task-1"},
                })
            raise AssertionError(command)

        args = pack.parse_args([
            "verify",
            "--public-host",
            "24.199.118.54",
            "--port",
            "9180",
            "--miner-id",
            "kaggle-cpu-1",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "2",
        ])

        report = pack.build_verify(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertIn("coordinator_public_ready", report["diagnosis_codes"])
        self.assertIn("kaggle_miner_seen", report["diagnosis_codes"])
        self.assertIn("kaggle_result_accepted", report["diagnosis_codes"])
        self.assertIn("kaggle_real_runtime_ready", report["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("miner-secret", serialized)
        self.assertEqual([command[2] for command in calls], ["doctor", "verify", "collect"])

    def test_verify_micro_llm_split_forwards_stage_options_and_ready_codes(self) -> None:
        output_dir = Path(self._tmp_dir())
        remote_dir = output_dir / "remote-home-compute"
        remote_dir.mkdir(parents=True, exist_ok=True)
        (remote_dir / "operator.private.env").write_text(
            "export CROWDTENSOR_ADMIN_TOKEN='admin-secret'\nexport CROWDTENSOR_OBSERVER_TOKEN='observer-secret'\n",
            encoding="utf-8",
        )
        (remote_dir / "miner.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='stage0-secret'\n", encoding="utf-8")
        (remote_dir / "miner.stage1.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='stage1-secret'\n", encoding="utf-8")
        for stage in ["stage0", "stage1"]:
            upload = output_dir / f"kaggle-upload-{stage}"
            upload.mkdir()
            (upload / "miner.private.env").write_text("x\n", encoding="utf-8")
            (upload / "kaggle_remote_miner.py").write_text("x\n", encoding="utf-8")
            (upload / "KAGGLE_RUN.md").write_text("x\n", encoding="utf-8")
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--workload", command)
            self.assertEqual(command[command.index("--workload") + 1], "micro-llm-sharded")
            if command[2] in {"verify", "collect"}:
                self.assertIn("--decode-steps", command)
                self.assertEqual(command[command.index("--decode-steps") + 1], "3")
                self.assertIn("--stage-mode", command)
                self.assertIn("--require-distinct-stage-miners", command)
                self.assertIn("--micro-llm-artifact", command)
                self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
                self.assertIn("--prompt-texts", command)
                self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            if command[2] == "doctor":
                return completed({
                    "schema": "remote_home_compute_doctor_v1",
                    "ok": True,
                    "workload_kind": "micro-llm-sharded",
                    "diagnosis_codes": ["remote_home_compute_doctor_ready"],
                    "connectivity": {
                        "observations": {"health": {"ok": True}, "ready": {"ok": True}, "state": {"ok": True}},
                        "matched_capabilities": [
                            "runtime:python-cli",
                            "backend:cpu",
                            "workload:micro_llm_sharded_infer",
                            "stage_0_completed",
                            "stage_1_completed",
                        ],
                        "missing_capabilities": [],
                    },
                })
            if command[2] == "verify":
                beta_payload = {
                    "schema": "remote_micro_llm_sharded_beta_v1",
                    "ok": True,
                    "diagnosis_codes": ["remote_micro_llm_sharded_ready"],
                    "payload_summaries": {
                        "remote_existing_micro_llm_sharded_inference": {
                            "schema": "micro_llm_sharded_evidence_v1",
                            "ok": True,
                            "diagnosis_codes": ["stage_assignment_valid"],
                            "stage_assignment": {
                                "stage0_miner_id": "kaggle-cpu-1-stage0",
                                "stage1_miner_id": "kaggle-cpu-1-stage1",
                                "distinct_stage_miners": True,
                                "stage_assignment_valid": True,
                            },
                        }
                    },
                }
                (remote_dir / "remote_micro_llm_sharded_beta.json").write_text(json.dumps(beta_payload), encoding="utf-8")
                return completed({
                    "schema": "remote_home_compute_demo_v1",
                    "ok": True,
                    "mode": "verify",
                    "workload_kind": "micro-llm-sharded",
                    "diagnosis_codes": ["remote_micro_llm_sharded_ready"],
                    "acceptance_summary": {
                        "task_id": "task-1",
                        "evidence_ok": True,
                    },
                })
            if command[2] == "collect":
                return completed({
                    "schema": "remote_home_compute_collect_v1",
                    "ok": True,
                    "workload_kind": "micro-llm-sharded",
                    "diagnosis_codes": ["remote_home_compute_collect_ready", "stage_assignment_valid"],
                    "status_summary": {"ready": True, "task_id": "task-1"},
                    "acceptance_summary": {"stage_assignment_valid": True},
                })
            raise AssertionError(command)

        args = pack.parse_args([
            "verify",
            "--workload",
            "micro-llm-sharded",
            "--stage-mode",
            "split",
            "--public-host",
            "24.199.118.54",
            "--port",
            "9180",
            "--miner-id",
            "kaggle-cpu-1",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "2",
            "--decode-steps",
            "3",
            "--micro-llm-artifact",
            "dist/micro-llm-artifact",
            "--prompt-texts",
            "arn,ten",
        ])

        report = pack.build_verify(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertIn("kaggle_micro_llm_stage0_seen", report["diagnosis_codes"])
        self.assertIn("kaggle_micro_llm_stage1_seen", report["diagnosis_codes"])
        self.assertIn("kaggle_micro_llm_stage_assignment_valid", report["diagnosis_codes"])
        self.assertIn("kaggle_micro_llm_sharded_ready", report["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("stage0-secret", serialized)
        self.assertNotIn("stage1-secret", serialized)

    def test_check_validate_prepare_rejects_operator_env_in_upload(self) -> None:
        output_dir = Path(self._tmp_dir())
        (output_dir / "kaggle-upload").mkdir(parents=True)
        (output_dir / "kaggle-upload" / "operator.private.env").write_text("bad\n", encoding="utf-8")
        payload = {
            "schema": "kaggle_real_runtime_acceptance_v1",
            "ok": True,
            "mode": "prepare",
            "coordinator_url": "http://24.199.118.54:9180",
            "diagnosis_codes": ["kaggle_artifacts_ready"],
            "safety": {
                "temporary_http": True,
                "temporary_http_boundary_confirmed": True,
                "token_rotation_required": True,
                "public_http_not_production": True,
                "operator_env_excluded_from_kaggle": True,
                "cpu_only_workload": True,
                "not_production": True,
                "not_p2p": True,
            },
            "artifacts": {},
        }
        args = argparse.Namespace(public_host="24.199.118.54", port=9180)

        with self.assertRaises(SystemExit):
            check.validate_prepare(payload, output_dir, args)


if __name__ == "__main__":
    unittest.main()
