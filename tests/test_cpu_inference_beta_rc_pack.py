from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "cpu_inference_beta_rc_pack.py"
SPEC = importlib.util.spec_from_file_location("cpu_inference_beta_rc_pack", SCRIPT_PATH)
pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pack)


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class CpuInferenceBetaRCPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_cpu_infer_beta_rc_test_"))

    def test_build_report_aggregates_local_remote_kaggle_and_manifest(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            joined = " ".join(command)
            if "cpu-infer --mode local" in joined:
                local_dir = output_dir / "local"
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
                return completed({
                    "schema": "cpu_inference_beta_v1",
                    "ok": True,
                    "mode": "local",
                    "diagnosis_codes": ["cpu_inference_beta_ready"],
                })
            if "cpu-infer --mode remote-loopback" in joined:
                remote_dir = output_dir / "remote-loopback"
                remote_dir.mkdir(parents=True, exist_ok=True)
                (remote_dir / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
                return completed({
                    "schema": "cpu_inference_beta_v1",
                    "ok": True,
                    "mode": "remote-loopback",
                    "diagnosis_codes": ["remote_home_compute_ready", "remote_external_llm_ready"],
                })
            if "remote_two_machine_beta_check.py" in joined:
                return completed({
                    "schema": "remote_two_machine_beta_check_v1",
                    "ok": True,
                    "diagnosis_codes": ["remote_two_machine_beta_ready"],
                })
            if "remote-demo prepare" in joined and "--target kaggle" in joined:
                kaggle_dir = output_dir / "kaggle-remote-miner"
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                (kaggle_dir / "kaggle_remote_miner.py").write_text("print('ok')\n", encoding="utf-8")
                (kaggle_dir / "kaggle_remote_miner.md").write_text("Upload only `miner.private.env`\n", encoding="utf-8")
                (kaggle_dir / "miner_join.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
                (kaggle_dir / "MINER_JOIN.md").write_text("# Join\n", encoding="utf-8")
                (kaggle_dir / "miner.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='secret'\n", encoding="utf-8")
                (kaggle_dir / "remote_home_compute_demo.json").write_text("{}", encoding="utf-8")
                return completed({
                    "schema": "remote_home_compute_demo_v1",
                    "ok": True,
                    "target_environment": {
                        "name": "kaggle",
                        "kaggle_remote_miner_beta": True,
                        "gpu_tpu_workload_enabled": False,
                    },
                    "diagnosis_codes": ["kaggle_remote_miner_prepare_ready", "miner_join_pack_ready"],
                })
            if "kaggle_remote_miner_beta_check.py" in joined:
                return completed({
                    "schema": "kaggle_remote_miner_beta_check_v1",
                    "ok": True,
                    "diagnosis_codes": ["kaggle_remote_miner_beta_ready"],
                })
            if "demo_manifest_check.py" in joined:
                manifest_dir = output_dir / "demo-manifest"
                manifest_dir.mkdir(parents=True, exist_ok=True)
                (manifest_dir / "demo_manifest.json").write_text("{}", encoding="utf-8")
                (manifest_dir / "demo_manifest.md").write_text("# manifest\n", encoding="utf-8")
                return completed({
                    "schema": "demo_manifest_v1",
                    "ok": True,
                    "diagnosis_codes": ["demo_manifest_ready"],
                })
            raise AssertionError(command)

        args = pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--request-count",
            "1",
            "--external-llm-request-count",
            "1",
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "cpu_inference_beta_rc_v1")
        self.assertEqual(report["mode"], "beta-rc")
        self.assertIn("cpu_inference_beta_rc_ready", report["diagnosis_codes"])
        self.assertIn("local_cpu_inference_ready", report["diagnosis_codes"])
        self.assertIn("remote_loopback_ready", report["diagnosis_codes"])
        self.assertIn("two_machine_rehearsal_ready", report["diagnosis_codes"])
        self.assertIn("kaggle_remote_miner_artifacts_ready", report["diagnosis_codes"])
        self.assertIn("miner_join_pack_ready", report["diagnosis_codes"])
        self.assertIn("cpu_miner_beta_ready", report["diagnosis_codes"])
        self.assertEqual(report["miner_join_pack"]["schema"], "miner_join_pack_v1")
        self.assertTrue(report["miner_join_pack"]["ready"])
        self.assertTrue(report["artifacts"]["cpu_inference_beta_rc_json"]["present"])
        self.assertTrue(report["artifacts"]["kaggle_remote_miner_script"]["present"])
        self.assertTrue(report["artifacts"]["miner_join_script"]["present"])
        self.assertTrue(report["artifacts"]["miner_join_runbook"]["present"])
        self.assertTrue(any("remote_two_machine_beta_check.py" in " ".join(command) for command in calls))
        self.assertTrue(any("kaggle_remote_miner_beta_check.py" in " ".join(command) for command in calls))

    def test_build_report_imports_kaggle_real_runtime_evidence(self) -> None:
        output_dir = self._tmp_dir()
        real_report = output_dir / "kaggle_real_runtime_acceptance.json"
        real_report.write_text(json.dumps({
            "schema": "kaggle_real_runtime_acceptance_v1",
            "ok": True,
            "mode": "verify",
            "coordinator_url": "http://24.199.118.54:9180",
            "miner_id": "kaggle-cpu-1",
            "diagnosis_codes": ["kaggle_real_runtime_ready", "kaggle_result_accepted"],
            "acceptance_summary": {
                "task_id": "task-real",
                "scenario_id": "route-baseline",
                "request_count": 2,
            },
            "safety": {
                "token_rotation_required": True,
                "temporary_http": True,
                "operator_env_excluded_from_kaggle": True,
            },
        }), encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "remote-demo prepare" in joined and "--target kaggle" in joined:
                kaggle_dir = output_dir / "kaggle-remote-miner"
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                for name in ["kaggle_remote_miner.py", "kaggle_remote_miner.md", "miner_join.sh", "MINER_JOIN.md", "miner.private.env"]:
                    (kaggle_dir / name).write_text("x\n", encoding="utf-8")
                (kaggle_dir / "remote_home_compute_demo.json").write_text("{}", encoding="utf-8")
                return completed({"schema": "remote_home_compute_demo_v1", "ok": True, "diagnosis_codes": ["miner_join_pack_ready"]})
            if "demo_manifest_check.py" in joined:
                manifest_dir = output_dir / "demo-manifest"
                manifest_dir.mkdir(parents=True, exist_ok=True)
                (manifest_dir / "demo_manifest.json").write_text("{}", encoding="utf-8")
                (manifest_dir / "demo_manifest.md").write_text("# manifest\n", encoding="utf-8")
                return completed({"schema": "demo_manifest_v1", "ok": True, "diagnosis_codes": []})
            if "cpu-infer --mode local" in joined:
                (output_dir / "local").mkdir(parents=True, exist_ok=True)
                (output_dir / "local" / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
                return completed({"schema": "cpu_inference_beta_v1", "ok": True, "diagnosis_codes": []})
            if "cpu-infer --mode remote-loopback" in joined:
                (output_dir / "remote-loopback").mkdir(parents=True, exist_ok=True)
                (output_dir / "remote-loopback" / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
                return completed({"schema": "cpu_inference_beta_v1", "ok": True, "diagnosis_codes": []})
            return completed({"schema": "child_v1", "ok": True, "diagnosis_codes": []})

        args = pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--request-count",
            "1",
            "--external-llm-request-count",
            "1",
            "--kaggle-real-runtime-report",
            str(real_report),
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["real_runtime_evidence"]["ready"])
        self.assertEqual(report["real_runtime_evidence"]["task_id"], "task-real")
        self.assertTrue(report["real_runtime_evidence"]["token_rotation_required"])
        self.assertIn("real_runtime_evidence_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["kaggle_real_runtime_report"]["present"])

    def test_build_report_imports_real_llm_live_rc_evidence(self) -> None:
        output_dir = self._tmp_dir()
        real_report = output_dir / "real_llm_live_rc.json"
        real_report.write_text(json.dumps({
            "schema": "real_llm_live_rc_v1",
            "ok": True,
            "mode": "external-existing",
            "coordinator_url": "http://24.199.118.54:9184",
            "miner_id": "kaggle-real-llm",
            "diagnosis_codes": [
                "external_runtime_verified",
                "kaggle_real_llm_sharded_ready",
                "kaggle_real_llm_stage0_seen",
                "kaggle_real_llm_stage1_seen",
                "stage_assignment_valid",
            ],
            "payload_summaries": {
                "verify": {
                    "remote_existing_real_llm_sharded_inference": {
                        "stage_assignment": {
                            "stage0_miner_id": "kaggle-real-llm-stage0",
                            "stage1_miner_id": "kaggle-real-llm-stage1",
                            "stage_assignment_valid": True,
                            "distinct_stage_miners": True,
                        }
                    }
                }
            },
            "safety": {
                "token_rotation_required": True,
                "temporary_http": True,
            },
        }), encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "remote-demo prepare" in joined and "--target kaggle" in joined:
                kaggle_dir = output_dir / "kaggle-remote-miner"
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                for name in ["kaggle_remote_miner.py", "kaggle_remote_miner.md", "miner_join.sh", "MINER_JOIN.md", "miner.private.env"]:
                    (kaggle_dir / name).write_text("x\n", encoding="utf-8")
                (kaggle_dir / "remote_home_compute_demo.json").write_text("{}", encoding="utf-8")
                return completed({"schema": "remote_home_compute_demo_v1", "ok": True, "diagnosis_codes": ["miner_join_pack_ready"]})
            if "demo_manifest_check.py" in joined:
                manifest_dir = output_dir / "demo-manifest"
                manifest_dir.mkdir(parents=True, exist_ok=True)
                (manifest_dir / "demo_manifest.json").write_text("{}", encoding="utf-8")
                (manifest_dir / "demo_manifest.md").write_text("# manifest\n", encoding="utf-8")
                return completed({"schema": "demo_manifest_v1", "ok": True, "diagnosis_codes": []})
            if "cpu-infer --mode local" in joined:
                (output_dir / "local").mkdir(parents=True, exist_ok=True)
                (output_dir / "local" / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
                return completed({"schema": "cpu_inference_beta_v1", "ok": True, "diagnosis_codes": []})
            if "cpu-infer --mode remote-loopback" in joined:
                (output_dir / "remote-loopback").mkdir(parents=True, exist_ok=True)
                (output_dir / "remote-loopback" / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
                return completed({"schema": "cpu_inference_beta_v1", "ok": True, "diagnosis_codes": []})
            return completed({"schema": "child_v1", "ok": True, "diagnosis_codes": []})

        args = pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--request-count",
            "1",
            "--external-llm-request-count",
            "1",
            "--kaggle-real-runtime-report",
            str(real_report),
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["real_runtime_evidence"]["ready"])
        self.assertEqual(report["real_runtime_evidence"]["schema"], "real_llm_live_rc_v1")
        self.assertEqual(report["real_runtime_evidence"]["runtime_kind"], "real_llm_live_rc")
        self.assertEqual(report["real_runtime_evidence"]["stage0_miner_id"], "kaggle-real-llm-stage0")
        self.assertEqual(report["real_runtime_evidence"]["stage1_miner_id"], "kaggle-real-llm-stage1")
        self.assertTrue(report["real_runtime_evidence"]["operator_env_excluded_from_kaggle"])
        self.assertIn("real_runtime_evidence_ready", report["diagnosis_codes"])
        self.assertIn("kaggle_real_llm_sharded_ready", report["diagnosis_codes"])

    def test_failed_child_marks_beta_rc_blocked(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "cpu-infer" in command:
                return completed({"schema": "cpu_inference_beta_v1", "ok": False, "diagnosis_codes": ["child_failed"]}, returncode=1)
            return completed({"schema": "placeholder_v1", "ok": True, "diagnosis_codes": []})

        args = pack.parse_args(["--output-dir", str(output_dir), "--request-count", "1", "--external-llm-request-count", "1"])
        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"])
        self.assertIn("beta_rc_blocked", report["diagnosis_codes"])
        self.assertNotIn("cpu_inference_beta_rc_ready", report["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
