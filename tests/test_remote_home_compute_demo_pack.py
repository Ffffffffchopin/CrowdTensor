from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_external_llm_prepare_writes_runbook_and_redacts_runtime_secrets(self) -> None:
        output_dir = Path(self._tmp_dir())

        args = pack.parse_args([
            "prepare",
            "--workload",
            "external-llm",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "3",
            "--llm-runtime-url",
            "http://127.0.0.1:11434/v1/chat/completions",
            "--llm-runtime-api-key",
            "runtime-secret",
            "--replace",
        ])
        report = pack.build_prepare(args)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["demo"]["workload_kind"], "external-llm")
        self.assertEqual(report["demo"]["workload_type"], "external_llm_infer")
        self.assertEqual(report["runbook_summary"]["schema"], "remote_external_llm_runbook_v1")
        self.assertIn("remote_external_llm_prepare_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_external_llm_runbook_json"]["present"])
        self.assertTrue(report["artifacts"]["operator_private_env"]["present"])
        self.assertNotIn("runtime-secret", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)

    def test_external_llm_verify_collects_remote_evidence(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []
        args = pack.parse_args([
            "verify",
            "--workload",
            "external-llm",
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
            "--mock",
        ])

        def fake_session(parsed_args: object) -> dict:
            parsed_args.session_task_id = "task-ext-1"
            return {
                "ok": True,
                "attempts": 1,
                "elapsed_seconds": 0.1,
                "errors": [],
                "session_create": {
                    "ok": True,
                    "session": {
                        "created": True,
                        "schema": "inference_session_request_v1",
                        "task_id": "task-ext-1",
                        "request_count": 4,
                        "workload_type": "external_llm_infer",
                    },
                },
                "status": {
                    "observations": {
                        "health": {"ok": True},
                        "ready": {"ok": True},
                        "state": {"ok": True},
                        "admin_results": {"ok": True},
                    },
                    "summary": {
                        "ready": True,
                        "matched_capabilities": [
                            "runtime:python-cli",
                            "backend:cpu",
                            "workload:external_llm_infer",
                            "external_llm_runtime",
                            "accepted_result",
                            "validation:ok",
                            "request_count",
                            "completion_count",
                        ],
                        "missing_capabilities": [],
                        "task_id": "task-ext-1",
                        "expected_task_id": "task-ext-1",
                        "accepted_results": 1,
                        "profile": {"runtime": "python-cli", "backend": "cpu"},
                        "inference": {
                            "ok": True,
                            "request_count": 4,
                            "completion_count": 4,
                            "output_chars": 200,
                            "adapter_kind": "mock",
                            "requests_per_second": 10.0,
                        },
                    },
                },
            }

        def fake_run_json_command(command: list[str], **_: object) -> dict:
            calls.append(command)
            if "remote_external_llm_evidence_pack.py" in command[1]:
                (output_dir / "remote_external_llm_evidence.json").write_text("{}", encoding="utf-8")
                (output_dir / "remote_external_llm_evidence.md").write_text("# Evidence\n", encoding="utf-8")
                return {
                    "ok": True,
                    "payload": {
                        "ok": True,
                        "schema": "remote_external_llm_evidence_v1",
                        "mode": "collect",
                        "route_decision": {"name": "remote_python_external_llm_infer", "confidence": "ready", "usable_now": True},
                        "inference_summary": {
                            "request_count": 4,
                            "completion_count": 4,
                            "output_chars": 200,
                            "requests_per_second": 10.0,
                        },
                        "adapter": {"kind": "mock", "model_id": "mock-external-llm"},
                        "safety": {"read_only": True, "redaction_ok": True},
                        "observability_summary": {"schema": "remote_external_llm_observability_v1"},
                    },
                }
            if "support_bundle.py" in command[1]:
                (output_dir / "support_bundle.json").write_text("{}", encoding="utf-8")
                (output_dir / "support_bundle.md").write_text("# Support\n", encoding="utf-8")
                return {"ok": True, "payload": {"ok": True, "schema": "support_bundle_v1"}}
            raise AssertionError(command)

        with (
            patch.object(pack, "wait_for_external_llm_result", side_effect=fake_session),
            patch.object(pack, "run_json_command", side_effect=fake_run_json_command),
        ):
            report = pack.build_verify(args)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["acceptance_summary"]["schema"], "remote_external_llm_acceptance_v1")
        self.assertEqual(report["acceptance_summary"]["evidence_schema"], "remote_external_llm_evidence_v1")
        self.assertEqual(report["acceptance_summary"]["observability_schema"], "remote_external_llm_observability_v1")
        self.assertIn("remote_external_llm_ready", report["diagnosis_codes"])
        self.assertIn("remote_home_compute_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_external_llm_acceptance_json"]["present"])
        self.assertTrue(report["artifacts"]["remote_external_llm_evidence_json"]["present"])
        self.assertTrue(any("remote_external_llm_evidence_pack.py" in command[1] for command in calls))
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_doctor_reports_ready_from_files_and_connectivity(self) -> None:
        output_dir = Path(self._tmp_dir())
        (output_dir / "operator.private.env").write_text(
            "export CROWDTENSOR_OBSERVER_TOKEN='observer-secret'\n"
            "export CROWDTENSOR_ADMIN_TOKEN='admin-secret'\n",
            encoding="utf-8",
        )
        (output_dir / "miner.private.env").write_text(
            "export CROWDTENSOR_MINER_TOKEN='miner-secret'\n",
            encoding="utf-8",
        )
        (output_dir / "miner_registry.json").write_text(
            json.dumps({"miners": [{"miner_id": "remote-linux-1", "token": "sha256:abc"}]}),
            encoding="utf-8",
        )
        (output_dir / "remote_demo_runbook.json").write_text(
            json.dumps({
                "ok": True,
                "schema": "remote_demo_runbook_v1",
                "demo": {
                    "coordinator_url": "https://coord.example",
                    "workload_type": "model_bundle_infer",
                    "route": "remote_python_model_bundle_infer",
                },
                "safety": {"registry_hashed": True, "public_artifact_redacted": True},
            }),
            encoding="utf-8",
        )
        args = pack.parse_args([
            "doctor",
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

        with patch.object(pack, "collect_status_for_workload", return_value={
            "ready": {"task_lanes": [{"workload_type": "model_bundle_infer"}]},
            "observations": {
                "health": {"ok": True},
                "ready": {"ok": True},
                "state": {"ok": True},
                "admin_results": {"ok": True},
            },
            "summary": {"ready": False, "matched_capabilities": ["runtime:python-cli"], "missing_capabilities": []},
        }):
            report = pack.build_doctor(args)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "remote_home_compute_doctor_v1")
        self.assertIn("remote_home_compute_doctor_ready", report["diagnosis_codes"])
        self.assertTrue(report["environment"]["registry_hashed"])
        self.assertTrue((output_dir / "remote_home_compute_doctor.json").is_file())
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_collect_model_bundle_writes_summary_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = pack.parse_args([
            "collect",
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

        def fake_collect_status(parsed_args: object) -> dict:
            return {
                "summary": {
                    "ready": True,
                    "task_id": "task-1",
                    "accepted_results": 1,
                    "matched_capabilities": ["runtime:python-cli"],
                    "missing_capabilities": [],
                }
            }

        def fake_collect_artifacts(parsed_args: object, target: Path, *, secret_values: list[str]) -> dict:
            (target / "remote_compute_evidence.json").write_text("{}", encoding="utf-8")
            (target / "support_bundle.json").write_text("{}", encoding="utf-8")
            return {
                "evidence": {
                    "ok": True,
                    "path": str(target / "remote_compute_evidence.json"),
                    "markdown_path": str(target / "remote_compute_evidence.md"),
                    "summary": {
                        "schema": "remote_compute_evidence_v1",
                        "ok": True,
                        "mode": "collect",
                        "route": "remote_python_model_bundle_infer",
                        "requests_per_second": 12.0,
                    },
                },
                "support_bundle": {
                    "ok": True,
                    "path": str(target / "support_bundle.json"),
                    "markdown_path": str(target / "support_bundle.md"),
                    "summary": {"schema": "support_bundle_v1", "ok": True},
                },
            }

        with (
            patch.object(pack, "collect_status_for_workload", side_effect=fake_collect_status),
            patch.object(pack, "collect_model_bundle_artifacts", side_effect=fake_collect_artifacts),
        ):
            report = pack.build_collect(args)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "remote_home_compute_collect_v1")
        self.assertIn("remote_home_compute_collect_ready", report["diagnosis_codes"])
        self.assertEqual(report["evidence_summary"]["schema"], "remote_compute_evidence_v1")
        self.assertTrue(report["artifacts"]["remote_home_compute_collect_json"]["present"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_clean_is_dry_run_by_default_and_private_files_are_gated(self) -> None:
        output_dir = Path(self._tmp_dir())
        (output_dir / "remote_home_compute_demo.json").write_text("{}", encoding="utf-8")
        (output_dir / "operator.private.env").write_text("export CROWDTENSOR_ADMIN_TOKEN='secret'\n", encoding="utf-8")
        args = pack.parse_args(["clean", "--output-dir", str(output_dir)])

        report = pack.build_clean(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "remote_home_compute_cleanup_v1")
        self.assertEqual(report["mode"], "dry_run")
        self.assertTrue((output_dir / "remote_home_compute_demo.json").exists())
        paths = {candidate["path"]: candidate for candidate in report["candidates"]}
        self.assertEqual(paths["remote_home_compute_demo.json"]["action"], "dry_run")
        self.assertNotIn("operator.private.env", paths)

    def test_clean_apply_can_remove_private_files_when_explicit(self) -> None:
        output_dir = Path(self._tmp_dir())
        public_file = output_dir / "remote_home_compute_demo.json"
        private_file = output_dir / "operator.private.env"
        public_file.write_text("{}", encoding="utf-8")
        private_file.write_text("export CROWDTENSOR_ADMIN_TOKEN='secret'\n", encoding="utf-8")
        args = pack.parse_args(["clean", "--output-dir", str(output_dir), "--apply", "--include-private"])

        report = pack.build_clean(args)

        self.assertTrue(report["ok"], report)
        self.assertFalse(public_file.exists())
        self.assertFalse(private_file.exists())
        self.assertGreater(report["deleted_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
