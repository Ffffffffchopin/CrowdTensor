from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
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
            self.assertIn("--miner-token", command)
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
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
                "miner_join_pack": {
                    "schema": "miner_join_pack_v1",
                    "ready": True,
                    "recommended_command": "bash miner_join.sh",
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
        self.assertTrue(report["runbook_summary"]["miner_join_pack_ready"])
        self.assertEqual(report["two_machine_beta"]["name"], "Real two-machine CPU inference Beta")
        self.assertFalse(report["two_machine_beta"]["claim_boundary"]["model_sharding"])
        self.assertEqual(report["two_machine_beta"]["workload"]["route"], "remote_python_model_bundle_infer")
        self.assertIn("remote_home_compute_prepare_ready", report["diagnosis_codes"])
        self.assertIn("miner_join_pack_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["operator_private_env"]["present"])
        self.assertIn("miner_join_script", report["artifacts"])
        self.assertIn("miner_join_runbook", report["artifacts"])
        self.assertTrue((output_dir / "remote_home_compute_demo.json").is_file())
        self.assertTrue(any("--replace" in command for command in calls))

    def test_prepare_kaggle_target_writes_kaggle_artifacts(self) -> None:
        output_dir = Path(self._tmp_dir())

        args = pack.parse_args([
            "prepare",
            "--target",
            "kaggle",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "kaggle-cpu-1",
            "--output-dir",
            str(output_dir),
            "--replace",
        ])
        report = pack.build_prepare(args)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["target_environment"]["name"], "kaggle")
        self.assertTrue(report["target_environment"]["kaggle_remote_miner_beta"])
        self.assertFalse(report["target_environment"]["gpu_tpu_workload_enabled"])
        self.assertIn("kaggle_remote_miner_prepare_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["kaggle_remote_miner_script"]["present"])
        self.assertTrue(report["artifacts"]["kaggle_remote_miner_runbook"]["present"])
        script = (output_dir / "kaggle_remote_miner.py").read_text(encoding="utf-8")
        runbook = (output_dir / "kaggle_remote_miner.md").read_text(encoding="utf-8")
        self.assertIn("CROWDTENSOR_REMOTE_ENVIRONMENT", script)
        self.assertIn("kaggle", script)
        self.assertNotIn("operator.private.env", script)
        self.assertIn("Upload only `miner.private.env`", runbook)
        self.assertNotIn("CROWDTENSOR_MINER_TOKEN=", serialized)
        self.assertNotIn("CROWDTENSOR_OBSERVER_TOKEN=", serialized)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=", serialized)

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

    def test_micro_llm_sharded_prepare_writes_runbook(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = pack.parse_args([
            "prepare",
            "--workload",
            "micro-llm-sharded",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "2",
            "--decode-steps",
            "3",
            "--stage-role",
            "stage0",
            "--replace",
        ])

        report = pack.build_prepare(args)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["demo"]["workload_kind"], "micro-llm-sharded")
        self.assertEqual(report["demo"]["workload_type"], "micro_llm_sharded_infer")
        self.assertEqual(report["runbook_summary"]["schema"], "remote_micro_llm_sharded_runbook_v1")
        self.assertEqual(report["runbook_summary"]["decode_steps"], 3)
        runbook = json.loads((output_dir / "remote_micro_llm_sharded_runbook.json").read_text(encoding="utf-8"))
        self.assertEqual(runbook["miner_join_pack"]["stage_role"], "stage0")
        join_script = (output_dir / "miner_join.sh").read_text(encoding="utf-8")
        self.assertIn("--micro-llm-stage-role", join_script)
        self.assertIn("stage0", join_script)
        self.assertTrue(report["two_machine_beta"]["claim_boundary"]["pipeline_sharded_inference"])
        self.assertIn("remote_micro_llm_sharded_prepare_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_micro_llm_sharded_runbook_json"]["present"])
        self.assertNotIn("CROWDTENSOR_MINER_TOKEN=", serialized)

    def test_micro_llm_sharded_verify_collects_beta_evidence(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []
        args = pack.parse_args([
            "verify",
            "--workload",
            "micro-llm-sharded",
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
            "--request-count",
            "2",
            "--decode-steps",
            "3",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--micro-llm-artifact",
            "dist/micro-llm-artifact",
            "--prompt-texts",
            "arn,ten",
        ])
        (output_dir / "remote_micro_llm_sharded_runbook.json").write_text(json.dumps({
            "schema": "remote_micro_llm_sharded_runbook_v1",
            "ok": True,
            "demo": {
                "coordinator_url": "https://coord.example",
                "workload_type": "micro_llm_sharded_infer",
                "route": "remote_python_micro_llm_sharded_infer",
                "request_count": 2,
                "scenario_schema": "micro_llm_sharded_session_v1",
                "scenario_id": "route-baseline",
                "decode_steps": 3,
            },
            "safety": {"registry_hashed": True, "public_artifact_redacted": True},
            "miner_join_pack": {"schema": "miner_join_pack_v1", "ready": True},
        }), encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_micro_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--decode-steps", command)
            self.assertEqual(command[command.index("--decode-steps") + 1], "3")
            self.assertIn("--stage-mode", command)
            self.assertIn("--require-distinct-stage-miners", command)
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            (output_dir / "remote_micro_llm_sharded_beta.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_micro_llm_sharded_beta.md").write_text("# Beta\n", encoding="utf-8")
            return completed({
                "schema": "remote_micro_llm_sharded_beta_v1",
                "ok": True,
                "mode": "remote-existing",
                "decode_steps": 3,
                "diagnosis_codes": [
                    "remote_micro_llm_sharded_ready",
                    "baseline_match",
                    "decoded_tokens_match",
                ],
                "payload_summaries": {
                    "remote_existing_micro_llm_sharded_inference": {
                        "schema": "micro_llm_sharded_evidence_v1",
                        "ok": True,
                        "session": {
                            "schema": "micro_llm_sharded_session_v1",
                            "session_id": "session-1",
                            "stage_count": 2,
                            "stage_0_task_id": "task-0",
                            "stage_1_task_id": "task-1",
                            "request_count": 2,
                            "decode_steps": 3,
                        },
                        "stage_summary": {
                            "stage_0": {"activation_count": 6, "activation_bytes": 120},
                            "stage_1": {
                                "baseline_match": True,
                                "decoded_tokens_match": True,
                                "request_count": 2,
                                "decode_steps": 3,
                            },
                        },
                        "safety": {"redaction_ok": True, "read_only": True},
                    }
                },
            })

        with patch.object(pack, "run_json_command", return_value={"ok": True, "payload": {"schema": "support_bundle_v1", "ok": True}}):
            report = pack.build_verify(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["acceptance_summary"]["schema"], "remote_micro_llm_sharded_acceptance_v1")
        self.assertEqual(report["acceptance_summary"]["evidence_schema"], "remote_micro_llm_sharded_beta_v1")
        self.assertEqual(report["acceptance_summary"]["observability_schema"], "remote_micro_llm_sharded_observability_v1")
        self.assertTrue(report["acceptance_summary"]["baseline_match"])
        self.assertTrue(report["acceptance_summary"]["decoded_tokens_match"])
        self.assertIn("remote_micro_llm_sharded_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_micro_llm_sharded_acceptance_json"]["present"])
        self.assertTrue(report["artifacts"]["remote_micro_llm_sharded_beta_json"]["present"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_real_llm_sharded_prepare_writes_runbook_and_stage_launcher(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = pack.parse_args([
            "prepare",
            "--workload",
            "real-llm-sharded",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "real-stage0",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "1",
            "--stage-role",
            "stage0",
            "--target",
            "kaggle",
            "--replace",
        ])

        report = pack.build_prepare(args)
        serialized = json.dumps(report, sort_keys=True)
        runbook = json.loads((output_dir / "remote_real_llm_sharded_runbook.json").read_text(encoding="utf-8"))
        join_script = (output_dir / "miner_join.sh").read_text(encoding="utf-8")
        kaggle_script = (output_dir / "kaggle_remote_miner.py").read_text(encoding="utf-8")

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["demo"]["workload_kind"], "real-llm-sharded")
        self.assertEqual(report["demo"]["workload_type"], "real_llm_sharded_infer")
        self.assertEqual(report["runbook_summary"]["schema"], "remote_real_llm_sharded_runbook_v1")
        self.assertEqual(runbook["miner_join_pack"]["stage_role"], "stage0")
        self.assertEqual(runbook["miner_join_pack"]["hf_model_id"], "sshleifer/tiny-gpt2")
        self.assertIn("--real-llm-model-id", runbook["commands"]["start_coordinator"])
        self.assertIn("--enable-hf-tiny-gpt-runtime", join_script)
        self.assertIn("--real-llm-stage-role", join_script)
        self.assertIn("stage0", join_script)
        self.assertIn("--hf-model-id", kaggle_script)
        self.assertTrue(report["two_machine_beta"]["claim_boundary"]["tiny_real_llm_pipeline_sharding"])
        self.assertFalse(report["two_machine_beta"]["claim_boundary"]["large_model_sharding"])
        self.assertIn("remote_real_llm_sharded_prepare_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_real_llm_sharded_runbook_json"]["present"])
        self.assertNotIn("CROWDTENSOR_MINER_TOKEN=", serialized)
        self.assertNotIn("CrowdTensor routes home CPU", serialized)

    def test_real_llm_sharded_verify_collects_beta_evidence(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []
        args = pack.parse_args([
            "verify",
            "--workload",
            "real-llm-sharded",
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
            "--request-count",
            "1",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
            "--prompt-texts",
            "private prompt should stay out of public summary",
        ])
        (output_dir / "remote_real_llm_sharded_runbook.json").write_text(json.dumps({
            "schema": "remote_real_llm_sharded_runbook_v1",
            "ok": True,
            "demo": {
                "coordinator_url": "https://coord.example",
                "workload_type": "real_llm_sharded_infer",
                "route": "remote_python_real_llm_sharded_infer",
                "request_count": 1,
                "scenario_schema": "real_llm_sharded_session_v1",
                "scenario_id": "route-baseline",
                "hf_model_id": "sshleifer/tiny-gpt2",
            },
            "target_environment": {"name": "generic", "remote_environment": "generic"},
            "safety": {"registry_hashed": True, "public_artifact_redacted": True},
            "miner_join_pack": {"schema": "miner_join_pack_v1", "ready": True},
        }), encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_real_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--stage-mode", command)
            self.assertEqual(command[command.index("--stage-mode") + 1], "split")
            self.assertIn("--hf-model-id", command)
            self.assertEqual(command[command.index("--hf-model-id") + 1], "sshleifer/tiny-gpt2")
            self.assertIn("--prompt-texts", command)
            self.assertIn("--require-distinct-stage-miners", command)
            (output_dir / "remote_real_llm_sharded_beta.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_real_llm_sharded_beta.md").write_text("# Beta\n", encoding="utf-8")
            return completed({
                "schema": "remote_real_llm_sharded_beta_v1",
                "ok": True,
                "mode": "remote-existing",
                "diagnosis_codes": [
                    "remote_real_llm_sharded_ready",
                    "baseline_match",
                    "decoded_tokens_match",
                    "real_llm_artifact_ready",
                    "stage_assignment_valid",
                ],
                "payload_summaries": {
                    "remote_existing_real_llm_sharded_inference": {
                        "schema": "real_llm_sharded_evidence_v1",
                        "ok": True,
                        "session": {
                            "schema": "real_llm_sharded_session_v1",
                            "session_id": "session-1",
                            "stage_count": 2,
                            "stage_0_task_id": "task-0",
                            "stage_1_task_id": "task-1",
                            "request_count": 1,
                            "model_id": "sshleifer/tiny-gpt2",
                            "artifact_hash": "sha256:abc",
                        },
                        "artifact": {"model_id": "sshleifer/tiny-gpt2", "loaded": True},
                        "stage_summary": {
                            "stage_0": {"activation_count": 1, "activation_bytes": 512},
                            "stage_1": {
                                "baseline_match": True,
                                "decoded_tokens_match": True,
                                "request_count": 1,
                            },
                        },
                        "stage_assignment": {"distinct_stage_miners": True, "stage_assignment_valid": True},
                        "safety": {"redaction_ok": True, "read_only": True},
                    }
                },
            })

        with patch.object(pack, "run_json_command", return_value={"ok": True, "payload": {"schema": "support_bundle_v1", "ok": True}}):
            report = pack.build_verify(args, runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["acceptance_summary"]["schema"], "remote_real_llm_sharded_acceptance_v1")
        self.assertEqual(report["acceptance_summary"]["evidence_schema"], "remote_real_llm_sharded_beta_v1")
        self.assertEqual(report["acceptance_summary"]["observability_schema"], "remote_real_llm_sharded_observability_v1")
        self.assertTrue(report["acceptance_summary"]["baseline_match"])
        self.assertTrue(report["acceptance_summary"]["decoded_tokens_match"])
        self.assertIn("remote_real_llm_sharded_ready", report["diagnosis_codes"])
        self.assertIn("remote_two_machine_real_llm_sharded_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_real_llm_sharded_acceptance_json"]["present"])
        self.assertTrue(report["artifacts"]["remote_real_llm_sharded_beta_json"]["present"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("private prompt should stay out", serialized)
        self.assertTrue(calls)

    def test_real_llm_sharded_verify_preserves_hf_dependency_diagnosis(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = pack.parse_args([
            "verify",
            "--workload",
            "real-llm-sharded",
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
            "--request-count",
            "1",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
        ])
        (output_dir / "remote_real_llm_sharded_runbook.json").write_text(json.dumps({
            "schema": "remote_real_llm_sharded_runbook_v1",
            "ok": True,
            "demo": {
                "coordinator_url": "https://coord.example",
                "workload_type": "real_llm_sharded_infer",
                "route": "remote_python_real_llm_sharded_infer",
                "request_count": 1,
                "scenario_schema": "real_llm_sharded_session_v1",
                "scenario_id": "route-baseline",
            },
            "target_environment": {"name": "generic", "remote_environment": "generic"},
            "safety": {"registry_hashed": True, "public_artifact_redacted": True},
            "miner_join_pack": {"schema": "miner_join_pack_v1", "ready": True},
        }), encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            (output_dir / "remote_real_llm_sharded_beta.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_real_llm_sharded_beta.md").write_text("# Beta\n", encoding="utf-8")
            return completed({
                "schema": "remote_real_llm_sharded_beta_v1",
                "ok": False,
                "mode": "remote-existing",
                "diagnosis_codes": [
                    "hf_dependencies_missing",
                    "session_create_failed",
                    "remote_real_llm_sharded_failed",
                ],
                "payload_summaries": {
                    "remote_existing_real_llm_sharded_inference": {
                        "schema": None,
                        "ok": None,
                        "session": {},
                        "artifact": {},
                        "stage_summary": {},
                        "stage_assignment": {},
                        "safety": {},
                    }
                },
            })

        report = pack.build_verify(args, runner=fake_runner)

        self.assertFalse(report["ok"])
        self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])
        self.assertIn("session_create_failed", report["diagnosis_codes"])
        self.assertIn("hf_dependencies_missing", report["acceptance_summary"]["diagnosis_codes"])

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

    def test_collect_real_llm_sharded_keeps_remote_beta_wrapper_and_lower_evidence(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = pack.parse_args([
            "collect",
            "--workload",
            "real-llm-sharded",
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
            "--request-count",
            "1",
            "--stage-mode",
            "split",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
            "--task-id",
            "task-1",
        ])
        evidence = {
            "schema": "real_llm_sharded_evidence_v1",
            "ok": True,
            "diagnosis_codes": [
                "activation_transport_ready",
                "baseline_match",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "real_llm_artifact_ready",
                "real_llm_sharded_ready",
                "stage_0_accepted",
                "stage_1_accepted",
                "stage_assignment_valid",
            ],
            "session": {
                "schema": "real_llm_sharded_session_v1",
                "session_id": "session-1",
                "stage_count": 2,
                "stage_0_task_id": "task-0",
                "stage_1_task_id": "task-1",
                "request_count": 1,
                "model_id": "sshleifer/tiny-gpt2",
                "artifact_hash": "sha256:abc",
            },
            "artifact": {"model_id": "sshleifer/tiny-gpt2", "loaded": True},
            "stage_summary": {
                "stage_0": {"activation_count": 1, "activation_bytes": 512},
                "stage_1": {
                    "baseline_match": True,
                    "decoded_tokens_match": True,
                    "request_count": 1,
                },
            },
            "stage_assignment": {"distinct_stage_miners": True, "stage_assignment_valid": True},
            "safety": {"read_only": True, "redaction_ok": True},
        }
        fake_evidence_pack = types.SimpleNamespace(
            build_report=lambda **_: evidence,
            render_markdown=lambda payload: f"# {payload.get('schema')}\n",
        )

        def fake_collect_status(parsed_args: object) -> dict:
            return {
                "summary": {
                    "ready": True,
                    "task_id": "task-1",
                    "accepted_results": 2,
                    "matched_capabilities": ["runtime:python-cli"],
                    "missing_capabilities": [],
                    "inference": {
                        "session_id": "session-1",
                        "stage_0_task_id": "task-0",
                        "stage_1_task_id": "task-1",
                    },
                },
                "state": {
                    "tasks": [
                        {
                            "workload_type": "real_llm_sharded_infer",
                            "workload_metadata": {"session_id": "session-1", "stage_id": 0},
                        },
                        {
                            "workload_type": "real_llm_sharded_infer",
                            "workload_metadata": {"session_id": "session-1", "stage_id": 1},
                        },
                    ],
                    "model": {"global_step": 0},
                    "model_updates": 0,
                },
                "results": {"results": []},
            }

        with (
            patch.object(pack, "collect_status_for_workload", side_effect=fake_collect_status),
            patch.object(pack, "collect_sharded_status", side_effect=fake_collect_status),
            patch.object(pack, "run_json_command", return_value={"ok": True, "payload": {"schema": "support_bundle_v1", "ok": True}}),
            patch.dict(sys.modules, {"real_llm_sharded_inference_evidence_pack": fake_evidence_pack}),
        ):
            report = pack.build_collect(args)
        serialized = json.dumps(report, sort_keys=True)
        beta = json.loads((output_dir / "remote_real_llm_sharded_beta.json").read_text(encoding="utf-8"))
        lower = json.loads((output_dir / "real_llm_sharded_evidence.json").read_text(encoding="utf-8"))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["evidence_summary"]["schema"], "remote_real_llm_sharded_beta_v1")
        self.assertEqual(beta["schema"], "remote_real_llm_sharded_beta_v1")
        self.assertEqual(lower["schema"], "real_llm_sharded_evidence_v1")
        self.assertIn("remote_real_llm_sharded_existing_ready", beta["diagnosis_codes"])
        self.assertEqual(
            beta["payload_summaries"]["remote_existing_real_llm_sharded_inference"]["schema"],
            "real_llm_sharded_evidence_v1",
        )
        self.assertTrue(report["artifacts"]["remote_real_llm_sharded_beta_json"]["present"])
        self.assertTrue(report["artifacts"]["real_llm_sharded_evidence_json"]["present"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_collect_model_bundle_artifacts_pins_evidence_to_task_id(self) -> None:
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
            "--task-id",
            "task-created",
        ])
        commands: list[list[str]] = []

        def fake_run_json_command(command: list[str], *, timeout: float, secret_values: list[str] | None = None) -> dict:
            commands.append(command)
            if "remote_compute_evidence_pack.py" in command[1]:
                return {
                    "ok": True,
                    "payload": {
                        "schema": "remote_compute_evidence_v1",
                        "ok": True,
                        "route_decision": {"name": "remote_python_model_bundle_infer"},
                        "inference_summary": {"scenario_matches": True},
                        "safety": {},
                        "observability_summary": {},
                    },
                }
            return {"ok": True, "payload": {"schema": "support_bundle_v1", "ok": True}}

        with patch.object(pack, "run_json_command", side_effect=fake_run_json_command):
            pack.collect_model_bundle_artifacts(args, output_dir, secret_values=["observer-secret", "admin-secret"])

        evidence_command = commands[0]
        self.assertIn("--task-id", evidence_command)
        self.assertEqual(evidence_command[evidence_command.index("--task-id") + 1], "task-created")

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
