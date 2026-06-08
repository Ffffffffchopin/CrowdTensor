from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "cpu_inference_beta_pack.py"
SPEC = importlib.util.spec_from_file_location("cpu_inference_beta_pack", SCRIPT_PATH)
pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pack)


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class CpuInferenceBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_cpu_infer_beta_test_"))

    def _assert_ready_guidance(self, report: dict, output_dir: Path) -> None:
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["user_status"]["next_step"], "review_artifacts")
        self.assertEqual(report["review_summary"]["schema"], "cpu_inference_beta_review_summary_v1")
        self.assertEqual(report["review_summary"]["state"], "ready")
        self.assertEqual(report["recommended_next_command"]["label"], "inspect CPU Inference Beta evidence")
        self.assertIn("cpu_inference_beta.md", report["recommended_next_command"]["command_line"])
        self.assertFalse(report["not_completed"])
        self.assertGreaterEqual(len(report["next_commands"]), 3)
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], report["artifact_summary"]["artifact_count"])
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
        support = json.loads((output_dir / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["schema"], "cpu_inference_beta_support_bundle_v1")
        self.assertEqual(support["review_summary"]["state"], "ready")
        self.assertEqual(support["recommended_next_command"]["label"], "inspect CPU Inference Beta evidence")
        self.assertTrue(support["public_artifact_safe"])

    def test_local_mode_aggregates_home_and_llm_proofs(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if "home-infer" in command:
                home_dir = Path(command[command.index("--output-dir") + 1])
                home_dir.mkdir(parents=True, exist_ok=True)
                (home_dir / "home_inference_cli_summary.json").write_text("{}", encoding="utf-8")
                (home_dir / "home_compute_evidence.json").write_text("{}", encoding="utf-8")
                return completed({
                    "schema": "home_inference_cli_v1",
                    "ok": True,
                    "diagnosis_codes": ["home_compute_ready"],
                    "route": {"name": "local_cpu_model_bundle_infer", "workload": "model_bundle_infer"},
                    "inference": {"request_count": 2, "request_trace_count": 2, "read_only": True, "redaction_ok": True},
                })
            if "llm-infer" in command:
                llm_dir = Path(command[command.index("--output-dir") + 1])
                llm_dir.mkdir(parents=True, exist_ok=True)
                (llm_dir / "llm_inference_cli_summary.json").write_text("{}", encoding="utf-8")
                (llm_dir / "external_llm_evidence.json").write_text("{}", encoding="utf-8")
                return completed({
                    "schema": "llm_inference_cli_v1",
                    "ok": True,
                    "diagnosis_codes": ["external_llm_evidence_ready"],
                    "inference": {"request_count": 1, "completion_count": 1},
                })
            raise AssertionError(command)

        args = pack.parse_args([
            "--mode",
            "local",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "2",
            "--external-llm-request-count",
            "1",
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "cpu_inference_beta_v1")
        self.assertEqual(report["mode"], "local")
        self.assertIn("cpu_inference_beta_ready", report["diagnosis_codes"])
        self.assertIn("home_compute_ready", report["diagnosis_codes"])
        self.assertIn("external_llm_evidence_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["cpu_inference_beta_json"]["present"])
        self.assertTrue(report["artifacts"]["home_inference_cli_summary"]["present"])
        self._assert_ready_guidance(report, output_dir)
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_external_llm_output_public"])
        self.assertEqual(report["prompt_scope"]["source"], "built-in-fixed-scenarios")
        self.assertFalse(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["raw_external_llm_output_public"])
        markdown = (output_dir / "cpu_inference_beta.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("## Artifact Summary", markdown)
        self.assertIn("## Not Completed", markdown)
        self.assertTrue(any("home-infer" in command for command in calls))
        self.assertTrue(any("llm-infer" in command for command in calls))

    def test_remote_loopback_runs_selected_workload_check(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_check.py", command[1])
            self.assertIn("--workload", command)
            self.assertIn("external-llm", command)
            return completed({
                "schema": "remote_home_compute_demo_check_v1",
                "ok": True,
                "workload": "external-llm",
                "diagnosis_codes": ["remote_external_llm_ready", "remote_home_compute_ready"],
                "evidence_schema": "remote_external_llm_evidence_v1",
                "observability_schema": "remote_external_llm_observability_v1",
            })

        args = pack.parse_args([
            "--mode",
            "remote-loopback",
            "--workload",
            "external-llm",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "2",
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["workload"], "external-llm")
        self.assertIn("remote_external_llm_ready", report["diagnosis_codes"])
        self._assert_ready_guidance(report, output_dir)
        self.assertTrue(calls)

    def test_remote_existing_redacts_tokens_and_runtime_url(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "doctor" in command:
                return completed({"schema": "remote_home_compute_doctor_v1", "ok": True, "diagnosis_codes": ["doctor_ready"]})
            if "verify" in command:
                return completed({
                    "schema": "remote_home_compute_demo_v1",
                    "ok": True,
                    "diagnosis_codes": ["remote_external_llm_ready"],
                    "acceptance_summary": {"task_id": "task-1", "schema": "remote_external_llm_acceptance_v1"},
                    "step": {"stderr_tail": "observer-secret admin-secret runtime-secret http://127.0.0.1:11434"},
                })
            if "collect" in command:
                return completed({"schema": "remote_home_compute_collect_v1", "ok": True, "diagnosis_codes": ["collect_ready"]})
            raise AssertionError(command)

        args = pack.parse_args([
            "--mode",
            "remote-existing",
            "--workload",
            "external-llm",
            "--coordinator-url",
            "https://coord.example",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--llm-runtime-url",
            "http://127.0.0.1:11434",
            "--llm-runtime-api-key",
            "runtime-secret",
            "--output-dir",
            str(output_dir),
        ])
        report = pack.build_report(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertNotIn("observer-secret", encoded)
        self.assertNotIn("admin-secret", encoded)
        self.assertNotIn("runtime-secret", encoded)
        self.assertNotIn("http://127.0.0.1:11434", encoded)
        self._assert_ready_guidance(report, output_dir)
        command_blob = json.dumps(report["next_commands"], sort_keys=True)
        self.assertIn("OBSERVER_TOKEN", command_blob)
        self.assertIn("ADMIN_TOKEN", command_blob)
        self.assertIn("LLM_RUNTIME_URL", command_blob)
        self.assertIn("LLM_RUNTIME_API_KEY", command_blob)
        self.assertNotIn("observer-secret", command_blob)
        self.assertNotIn("runtime-secret", command_blob)

    def test_remote_existing_requires_tokens(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "--mode",
                "remote-existing",
                "--coordinator-url",
                "https://coord.example",
            ])


if __name__ == "__main__":
    unittest.main()
