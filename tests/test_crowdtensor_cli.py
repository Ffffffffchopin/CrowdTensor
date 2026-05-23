from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crowdtensor import cli


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class CrowdTensorCliTests(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="crowdtensor_cli_test_")

    def _cleanup_args(self, *extra: str) -> object:
        return cli.parse_args(["clean-artifacts", *extra])

    def test_local_proof_success_summarizes_steps_and_artifacts(self) -> None:
        calls: list[list[str]] = []
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if "doctor.py" in command[1]:
                return completed({"ok": True, "summary": {"errors": 0}})
            if "runtime_matrix.py" in command[1]:
                return completed({"ok": True, "diagnosis_summary": {"codes": ["cpu_baseline_ready"]}})
            if "home_compute_demo.py" in command[1]:
                return completed({"ok": True, "diagnosis_codes": ["home_compute_ready"]})
            if "demo_manifest_pack.py" in command[1]:
                (output_dir / "demo_manifest.json").write_text("{}", encoding="utf-8")
                (output_dir / "demo_manifest.md").write_text("# Demo\n", encoding="utf-8")
                return completed({"ok": True, "schema": "demo_manifest_v1"})
            raise AssertionError(command)

        args = cli.parse_args([
            "local-proof",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9000",
            "--request-count",
            "4",
        ])

        summary = cli.build_local_proof(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "local_proof_summary_v1")
        self.assertEqual([step["name"] for step in summary["steps"]], [
            "doctor",
            "runtime_matrix",
            "home_compute_demo",
            "demo_manifest",
        ])
        self.assertEqual(summary["diagnosis_codes"], ["cpu_baseline_ready", "home_compute_ready"])
        self.assertTrue(summary["artifacts"]["demo_manifest_json"]["present"])
        self.assertTrue((output_dir / "local_proof_summary.json").is_file())
        self.assertTrue(any("demo_manifest_pack.py" in command[1] for command in calls))

    def test_runtime_matrix_block_skips_demo_and_manifest(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "doctor.py" in command[1]:
                return completed({"ok": True})
            if "runtime_matrix.py" in command[1]:
                return completed({"ok": False, "diagnosis_summary": {"codes": ["runtime_matrix_blocked"]}}, returncode=1)
            raise AssertionError(f"unexpected command: {command}")

        args = cli.parse_args(["local-proof", "--output-dir", str(output_dir), "--base-port", "9001"])

        summary = cli.build_local_proof(args, runner=fake_runner)

        self.assertFalse(summary["ok"])
        self.assertIn("runtime_matrix_blocked", summary["errors"])
        skipped = [step for step in summary["steps"] if step.get("skipped")]
        self.assertEqual([step["name"] for step in skipped], ["home_compute_demo", "demo_manifest"])

    def test_summary_redacts_sensitive_payload_fragments(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "doctor.py" in command[1]:
                return completed({"ok": True, "lease_token": "secret-lease"})
            if "runtime_matrix.py" in command[1]:
                return completed({"ok": True})
            if "home_compute_demo.py" in command[1]:
                return completed({"ok": True, "inference_results": [{"x": 1}]})
            if "demo_manifest_pack.py" in command[1]:
                return completed({"ok": True, "schema": "demo_manifest_v1"})
            raise AssertionError(command)

        args = cli.parse_args(["local-proof", "--output-dir", str(output_dir), "--base-port", "9002"])

        summary = cli.build_local_proof(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertNotIn("secret-lease", serialized)
        self.assertNotIn("inference_results", serialized)
        self.assertNotIn("lease_token", serialized)

    def test_main_json_outputs_summary_and_exit_zero(self) -> None:
        summary = {"schema": "local_proof_summary_v1", "ok": True}
        with patch.object(cli, "build_local_proof", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["local-proof", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "local_proof_summary_v1")

    def test_cleanup_dry_run_keeps_candidates(self) -> None:
        root = Path(self._tmp_dir())
        tmp_root = Path(self._tmp_dir())
        cache = root / "crowdtensor" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "x.pyc").write_bytes(b"cache")
        proof = tmp_root / "crowdtensor_local_proof_old"
        proof.mkdir()
        (proof / "artifact.json").write_text("{}", encoding="utf-8")
        old_time = 1_700_000_000
        os.utime(proof, (old_time, old_time))

        args = self._cleanup_args("--json")
        report = cli.build_cleanup_report(args, root=root, tmp_root=tmp_root)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "cleanup_report_v1")
        self.assertEqual(report["mode"], "dry_run")
        self.assertTrue(cache.exists())
        self.assertTrue(proof.exists())
        actions = {candidate["action"] for candidate in report["candidates"]}
        self.assertIn("dry_run", actions)

    def test_cleanup_apply_deletes_cache_and_old_temp_dir(self) -> None:
        root = Path(self._tmp_dir())
        tmp_root = Path(self._tmp_dir())
        cache = root / "tests" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "test.pyc").write_bytes(b"cache")
        proof = tmp_root / "crowdtensor_local_proof_old"
        proof.mkdir()
        (proof / "artifact.json").write_text("{}", encoding="utf-8")
        old_time = 1_700_000_000
        os.utime(proof, (old_time, old_time))

        args = self._cleanup_args("--apply", "--older-than-hours", "0", "--json")
        report = cli.build_cleanup_report(args, root=root, tmp_root=tmp_root)

        self.assertTrue(report["ok"], report)
        self.assertFalse(cache.exists())
        self.assertFalse(proof.exists())
        self.assertGreater(report["deleted_bytes"], 0)
        self.assertEqual({candidate["action"] for candidate in report["candidates"]}, {"deleted"})

    def test_cleanup_reports_require_explicit_include_reports(self) -> None:
        root = Path(self._tmp_dir())
        tmp_root = Path(self._tmp_dir())
        report_path = tmp_root / "crowdtensor_acceptance.json"
        report_path.write_text("{}", encoding="utf-8")
        old_time = 1_700_000_000
        os.utime(report_path, (old_time, old_time))

        default_report = cli.build_cleanup_report(
            self._cleanup_args("--apply", "--older-than-hours", "0", "--json"),
            root=root,
            tmp_root=tmp_root,
        )
        self.assertTrue(report_path.exists())
        self.assertEqual(default_report["candidates"][0]["skip_reason"], "requires_include_reports")

        include_report = cli.build_cleanup_report(
            self._cleanup_args("--apply", "--include-reports", "--older-than-hours", "0", "--json"),
            root=root,
            tmp_root=tmp_root,
        )
        self.assertFalse(report_path.exists())
        self.assertEqual(include_report["candidates"][0]["action"], "deleted")

    def test_cleanup_skips_protected_paths_and_symlinks(self) -> None:
        root = Path(self._tmp_dir())
        tmp_root = Path(self._tmp_dir())
        protected = root / "state" / "__pycache__"
        protected.mkdir(parents=True)
        (protected / "state.pyc").write_bytes(b"cache")
        target = tmp_root / "target"
        target.mkdir()
        link = tmp_root / "crowdtensor_local_proof_link"
        link.symlink_to(target, target_is_directory=True)

        args = self._cleanup_args("--apply", "--older-than-hours", "0", "--json")
        report = cli.build_cleanup_report(args, root=root, tmp_root=tmp_root)

        self.assertTrue(protected.exists())
        self.assertTrue(link.exists())
        skipped = {candidate["skip_reason"] for candidate in report["candidates"]}
        self.assertIn("protected_repo_path", skipped)
        self.assertIn("symlink", skipped)

    def test_main_cleanup_json_outputs_report(self) -> None:
        report = {"schema": "cleanup_report_v1", "ok": True}
        with patch.object(cli, "build_cleanup_report", return_value=report), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["clean-artifacts", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "cleanup_report_v1")

    def test_home_infer_wraps_evidence_pack_and_writes_safe_summary(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("home_compute_evidence_pack.py", command[1])
            (output_dir / "home_compute_evidence.json").write_text("{}", encoding="utf-8")
            (output_dir / "home_compute_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "home_compute_evidence_v1",
                "diagnosis_codes": ["home_compute_ready"],
                "route_decision": {
                    "name": "local_cpu_model_bundle_infer",
                    "target": "cpu_baseline",
                    "workload": "model_bundle_infer",
                    "confidence": "ready",
                    "usable_now": True,
                },
                "inference_summary": {
                    "present": True,
                    "ok": True,
                    "workload_type": "model_bundle_infer",
                    "scenario_schema": "model_bundle_inference_scenario_v1",
                    "scenario_id": "route-baseline",
                    "scenario_description": "Fixed CPU read-only route prompts from the built-in bundle corpus.",
                    "scenario_request_count": 8,
                    "request_count": 4,
                    "request_trace_count": 4,
                    "requests_per_second": 123.4,
                    "read_only": True,
                    "redaction_ok": True,
                },
            })

        args = cli.parse_args([
            "home-infer",
            "--output-dir",
            str(output_dir),
            "--port",
            "9010",
            "--request-count",
            "4",
        ])

        summary = cli.build_home_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "home_inference_cli_v1")
        self.assertEqual(summary["evidence_schema"], "home_compute_evidence_v1")
        self.assertEqual(summary["route"]["name"], "local_cpu_model_bundle_infer")
        self.assertEqual(summary["diagnosis_codes"], ["home_compute_ready"])
        self.assertEqual(summary["scenario"]["scenario_id"], "route-baseline")
        self.assertEqual(summary["scenario"]["scenario_schema"], "model_bundle_inference_scenario_v1")
        self.assertEqual(summary["inference"]["request_trace_count"], 4)
        self.assertTrue(summary["artifacts"]["home_compute_evidence_json"]["present"])
        self.assertTrue((output_dir / "home_inference_cli_summary.json").is_file())
        self.assertTrue(any("--json-out" in command for command in calls))
        self.assertTrue(any("--scenario-id" in command and "route-baseline" in command for command in calls))

    def test_home_infer_forwards_runtime_report(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--runtime-report", command)
            self.assertIn("/tmp/runtime.json", command)
            return completed({"ok": True, "schema": "home_compute_evidence_v1"})

        args = cli.parse_args([
            "home-infer",
            "--output-dir",
            str(output_dir),
            "--runtime-report",
            "/tmp/runtime.json",
        ])

        summary = cli.build_home_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)

    def test_home_infer_failure_preserves_diagnosis_and_redacts_sensitive_payloads(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return completed({
                "ok": False,
                "schema": "home_compute_evidence_v1",
                "diagnosis_codes": ["trace_missing"],
                "inference_results": [{"raw": "payload"}],
                "lease_token": "secret-lease",
            }, returncode=1)

        args = cli.parse_args(["home-infer", "--output-dir", str(output_dir)])

        summary = cli.build_home_inference(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertFalse(summary["ok"])
        self.assertIn("trace_missing", summary["diagnosis_codes"])
        self.assertNotIn("secret-lease", serialized)
        self.assertNotIn("lease_token", serialized)
        self.assertNotIn("inference_results", serialized)

    def test_main_home_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "home_inference_cli_v1", "ok": True}
        with patch.object(cli, "build_home_inference", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["home-infer", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "home_inference_cli_v1")

    def test_remote_runbook_wraps_pack_and_writes_safe_summary(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_demo_runbook_pack.py", command[1])
            (output_dir / "remote_demo_runbook.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_demo_runbook.md").write_text("# Runbook\n", encoding="utf-8")
            (output_dir / "operator.private.env").write_text("CROWDTENSOR_ADMIN_TOKEN=secret\n", encoding="utf-8")
            (output_dir / "miner.private.env").write_text("CROWDTENSOR_MINER_TOKEN=secret\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "remote_demo_runbook_v1",
                "demo": {
                    "workload_type": "model_bundle_infer",
                    "scenario_schema": "model_bundle_inference_scenario_v1",
                    "scenario_id": "route-baseline",
                    "scenario_description": "Fixed CPU read-only route prompts from the built-in bundle corpus.",
                    "scenario_request_count": 8,
                },
            })

        args = cli.parse_args([
            "remote-runbook",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_runbook(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_runbook_cli_v1")
        self.assertEqual(summary["runbook_schema"], "remote_demo_runbook_v1")
        self.assertEqual(summary["workload_type"], "model_bundle_infer")
        self.assertEqual(summary["scenario"]["scenario_id"], "route-baseline")
        self.assertTrue(summary["artifacts"]["operator_private_env"]["present"])
        self.assertTrue((output_dir / "remote_runbook_cli_summary.json").is_file())
        self.assertTrue(any("--coordinator-url" in command for command in calls))
        self.assertTrue(any("--scenario-id" in command and "route-baseline" in command for command in calls))

    def test_remote_runbook_replace_forwards_to_pack(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--replace", command)
            return completed({"ok": True, "schema": "remote_demo_runbook_v1"})

        args = cli.parse_args([
            "remote-runbook",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
            "--replace",
        ])

        summary = cli.build_remote_runbook(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)

    def test_remote_acceptance_defaults_to_create_session_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--create-session", command)
            (output_dir / "remote_demo_acceptance.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_demo_acceptance.md").write_text("# Acceptance\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "remote_demo_acceptance_v1",
                "scenario": {"scenario_id": "route-baseline"},
                "diagnosis_codes": ["acceptance_ready"],
            })

        args = cli.parse_args([
            "remote-acceptance",
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

        summary = cli.build_remote_acceptance(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_acceptance_cli_v1")
        self.assertTrue(summary["create_session"])
        self.assertEqual(summary["scenario"]["scenario_id"], "route-baseline")
        self.assertEqual(summary["diagnosis_codes"], ["acceptance_ready"])
        self.assertTrue(any("--scenario-id" in command and "route-baseline" in command for command in calls))
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue((output_dir / "remote_acceptance_cli_summary.json").is_file())

    def test_remote_acceptance_no_create_session_uses_wait_only_mode(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertNotIn("--create-session", command)
            return completed({"ok": True, "schema": "remote_demo_acceptance_v1"})

        args = cli.parse_args([
            "remote-acceptance",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--no-create-session",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_acceptance(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertFalse(summary["create_session"])

    def test_remote_acceptance_failure_tail_redacts_token_values(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout=json.dumps({"ok": False, "diagnosis_codes": ["observer_auth_failed"]}) + "\n",
                stderr="token observer-secret rejected; admin-secret was not accepted",
            )

        args = cli.parse_args([
            "remote-acceptance",
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

        summary = cli.build_remote_acceptance(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertFalse(summary["ok"])
        self.assertIn("observer_auth_failed", summary["diagnosis_codes"])
        self.assertIn("<redacted>", serialized)
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_main_remote_runbook_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_runbook_cli_v1", "ok": True}
        with patch.object(cli, "build_remote_runbook", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["remote-runbook", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_runbook_cli_v1")

    def test_main_remote_acceptance_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_acceptance_cli_v1", "ok": True}
        with patch.object(cli, "build_remote_acceptance", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main([
                    "remote-acceptance",
                    "--coordinator-url",
                    "https://coord.example",
                    "--miner-id",
                    "remote-linux-1",
                    "--observer-token",
                    "observer-secret",
                    "--admin-token",
                    "admin-secret",
                    "--json",
                ])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_acceptance_cli_v1")


if __name__ == "__main__":
    unittest.main()
