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

    def test_product_serve_redacts_tokens_in_command_report(self) -> None:
        args = cli.parse_args([
            "serve",
            "--admin-token",
            "admin-secret",
            "--miner-token",
            "miner-secret",
            "--json",
        ])

        report = cli.build_product_serve(args)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertNotIn("admin-secret", encoded)
        self.assertNotIn("miner-secret", encoded)
        self.assertIn("<redacted>", report["command"])

    def test_product_generate_dry_run_uses_session_protocol(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--backend",
            "cuda",
            "--dry-run",
            "--json",
        ])

        report = cli.build_product_generate(args)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["session_request"]["schema"], "session_protocol_v1")
        self.assertNotIn("CrowdTensor prompt", encoded)
        self.assertIn("prompt_hash", encoded)

    def test_peer_check_wraps_discovery_check(self) -> None:
        args = cli.parse_args(["peer", "check", "--json"])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("p2p_lite_discovery_check.py", command[1])
            return completed({"schema": "p2p_lite_discovery_check_v1", "ok": True})

        report = cli.build_peer_cli(args, runner=fake_runner)

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], "p2p_lite_discovery_check_v1")

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

    def test_llm_infer_wraps_external_llm_evidence_and_redacts_runtime_secrets(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("external_llm_evidence_pack.py", command[1])
            (output_dir / "external_llm_evidence.json").write_text("{}", encoding="utf-8")
            (output_dir / "external_llm_evidence.md").write_text("# LLM Evidence\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "external_llm_evidence_v1",
                "diagnosis_codes": ["external_llm_evidence_ready"],
                "adapter": {
                    "kind": "http_openai_chat",
                    "model_id": "local-model",
                    "operator_owned_runtime": True,
                },
                "summary": {
                    "request_count": 3,
                    "completion_count": 3,
                    "output_chars": 128,
                    "requests_per_second": 12.5,
                },
            })

        args = cli.parse_args([
            "llm-infer",
            "--output-dir",
            str(output_dir),
            "--port",
            "9019",
            "--request-count",
            "3",
            "--llm-runtime-url",
            "http://127.0.0.1:11434/v1/chat/completions",
            "--llm-runtime-api-key",
            "secret-api-key",
            "--llm-runtime-model-id",
            "local-model",
        ])

        summary = cli.build_llm_inference(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "llm_inference_cli_v1")
        self.assertEqual(summary["evidence_schema"], "external_llm_evidence_v1")
        self.assertEqual(summary["adapter"]["kind"], "http_openai_chat")
        self.assertEqual(summary["inference"]["completion_count"], 3)
        self.assertEqual(summary["diagnosis_codes"], ["external_llm_evidence_ready"])
        self.assertTrue(summary["artifacts"]["external_llm_evidence_json"]["present"])
        self.assertTrue((output_dir / "llm_inference_cli_summary.json").is_file())
        self.assertTrue(any("--llm-runtime-url" in command for command in calls))
        self.assertNotIn("secret-api-key", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)

    def test_llm_infer_rejects_conflicting_runtime_modes(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "llm-infer",
                "--llm-runtime-cmd",
                "/bin/echo",
                "--llm-runtime-url",
                "http://127.0.0.1:11434/v1/chat/completions",
            ])

    def test_main_llm_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "llm_inference_cli_v1", "ok": True}
        with patch.object(cli, "build_llm_inference", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["llm-infer", "--mock", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "llm_inference_cli_v1")

    def test_cpu_infer_wraps_beta_pack_and_redacts_runtime_secrets(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("cpu_inference_beta_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertIn("remote-existing", command)
            self.assertIn("--workload", command)
            self.assertIn("external-llm", command)
            return completed({
                "schema": "cpu_inference_beta_v1",
                "ok": True,
                "mode": "remote-existing",
                "diagnosis_codes": ["cpu_inference_beta_ready"],
                "steps": [{"name": "remote_existing_external_llm_verify", "ok": True}],
                "step": {"stderr_tail": "observer-secret admin-secret runtime-secret http://127.0.0.1:11434"},
            })

        args = cli.parse_args([
            "cpu-infer",
            "--mode",
            "remote-existing",
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
            "--llm-runtime-url",
            "http://127.0.0.1:11434",
            "--llm-runtime-api-key",
            "runtime-secret",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_cpu_inference_beta(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "cpu_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "cpu_inference_beta_cli_v1")
        self.assertIn("cpu_inference_beta_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("runtime-secret", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)
        self.assertTrue(calls)

    def test_cpu_infer_remote_existing_requires_auth(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "cpu-infer",
                "--mode",
                "remote-existing",
                "--coordinator-url",
                "https://coord.example",
            ])

    def test_cpu_infer_beta_rc_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("cpu_inference_beta_rc_pack.py", command[1])
            self.assertIn("--kaggle-real-runtime-report", command)
            self.assertIn("/tmp/kaggle-real.json", command)
            return completed({
                "schema": "cpu_inference_beta_rc_v1",
                "ok": True,
                "mode": "beta-rc",
                "diagnosis_codes": ["cpu_inference_beta_rc_ready"],
            })

        args = cli.parse_args([
            "cpu-infer",
            "--mode",
            "beta-rc",
            "--output-dir",
            str(output_dir),
            "--kaggle-real-runtime-report",
            "/tmp/kaggle-real.json",
        ])

        summary = cli.build_cpu_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "cpu_inference_beta_rc_v1")
        self.assertEqual(summary["cli_schema"], "cpu_inference_beta_rc_cli_v1")
        self.assertIn("cpu_inference_beta_rc_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_cpu_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "cpu_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_cpu_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["cpu-infer", "--mode", "local", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "cpu_inference_beta_v1")

    def test_shard_infer_wraps_evidence_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("sharded_inference_evidence_pack.py", command[1])
            self.assertIn("--failure-mode", command)
            self.assertIn("kill-stage-after-claim", command)
            return completed({
                "schema": "sharded_inference_evidence_v1",
                "ok": True,
                "diagnosis_codes": [
                    "sharded_inference_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "activation_transport_ready",
                    "stage_requeue_ready",
                ],
                "session": {"session_id": "shard-session-test", "stage_count": 2},
                "stage_summary": {"stage_1": {"baseline_match": True}},
                "safety": {"read_only": True, "redaction_ok": True},
            })

        args = cli.parse_args([
            "shard-infer",
            "--output-dir",
            str(output_dir),
            "--failure-mode",
            "kill-stage-after-claim",
        ])

        summary = cli.build_sharded_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "sharded_inference_cli_v1")
        self.assertIn("sharded_inference_ready", summary["diagnosis_codes"])
        self.assertTrue(summary["artifacts"]["sharded_inference_cli_summary"]["present"])
        self.assertTrue(calls)

    def test_main_shard_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "sharded_inference_cli_v1", "ok": True}
        with patch.object(cli, "build_sharded_inference", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["shard-infer", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "sharded_inference_cli_v1")

    def test_micro_llm_shard_infer_wraps_evidence_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("micro_llm_sharded_inference_evidence_pack.py", command[1])
            self.assertIn("--decode-steps", command)
            self.assertIn("--stage-mode", command)
            self.assertIn("--micro-llm-artifact", command)
            self.assertIn("4", command)
            return completed({
                "schema": "micro_llm_sharded_evidence_v1",
                "ok": True,
                "diagnosis_codes": [
                    "micro_llm_sharded_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                ],
                "session": {"session_id": "micro-llm-session-test", "stage_count": 2, "decode_steps": 4},
                "stage_summary": {"stage_1": {"baseline_match": True, "decoded_tokens_match": True}},
                "safety": {"read_only": True, "redaction_ok": True},
            })

        args = cli.parse_args([
            "micro-llm-shard-infer",
            "--output-dir",
            str(output_dir),
            "--decode-steps",
            "4",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--micro-llm-artifact",
            str(output_dir / "artifact"),
        ])

        summary = cli.build_micro_llm_sharded_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "micro_llm_sharded_cli_v1")
        self.assertIn("micro_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertTrue(summary["artifacts"]["micro_llm_sharded_cli_summary"]["present"])
        self.assertTrue(calls)

    def test_micro_llm_artifact_cli_builds_artifact(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("micro_llm_artifact_pack.py", command[1])
            return completed({
                "schema": "micro_llm_artifact_v1",
                "ok": True,
                "artifact_id": "crowdtensor-micro-llm-alpha",
                "artifact_hash": "sha256:artifact",
                "artifact_version": 1,
                "manifest_path": str(output_dir / "manifest.json"),
            })

        args = cli.parse_args(["micro-llm-artifact", "--output-dir", str(output_dir)])
        summary = cli.build_micro_llm_artifact(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "micro_llm_artifact_cli_v1")
        self.assertEqual(summary["artifact_hash"], "sha256:artifact")
        self.assertTrue(summary["artifacts"]["micro_llm_artifact_cli_summary"]["present"])
        self.assertTrue(calls)

    def test_real_llm_shard_infer_wraps_evidence_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_sharded_inference_evidence_pack.py", command[1])
            self.assertIn("--hf-model-id", command)
            self.assertIn("--stage-mode", command)
            return completed({
                "schema": "real_llm_sharded_evidence_v1",
                "ok": True,
                "diagnosis_codes": [
                    "real_llm_sharded_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                    "real_llm_artifact_ready",
                ],
                "session": {"session_id": "real-llm-session-test", "stage_count": 2, "model_id": "sshleifer/tiny-gpt2"},
                "artifact": {"model_id": "sshleifer/tiny-gpt2", "artifact_hash": "sha256:real"},
                "stage_summary": {"stage_1": {"baseline_match": True, "decoded_tokens_match": True}},
                "safety": {"read_only": True, "redaction_ok": True},
            })

        args = cli.parse_args([
            "real-llm-shard-infer",
            "--output-dir",
            str(output_dir),
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
        ])

        summary = cli.build_real_llm_sharded_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_llm_sharded_cli_v1")
        self.assertIn("real_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertTrue(summary["artifacts"]["real_llm_sharded_cli_summary"]["present"])
        self.assertTrue(calls)

    def test_main_micro_llm_shard_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "micro_llm_sharded_cli_v1", "ok": True}
        with patch.object(cli, "build_micro_llm_sharded_inference", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["micro-llm-shard-infer", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "micro_llm_sharded_cli_v1")

    def test_shard_infer_beta_wraps_beta_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_sharded_inference_beta_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertIn("remote-loopback", command)
            self.assertIn("--failure-mode", command)
            return completed({
                "schema": "remote_sharded_inference_beta_v1",
                "ok": True,
                "mode": "remote-loopback",
                "diagnosis_codes": [
                    "remote_sharded_inference_ready",
                    "remote_sharded_loopback_ready",
                    "sharded_inference_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "activation_transport_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "shard-infer-beta",
            "--output-dir",
            str(output_dir),
            "--mode",
            "remote-loopback",
        ])
        summary = cli.build_remote_sharded_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_sharded_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "remote_sharded_inference_beta_cli_v1")
        self.assertIn("remote_sharded_inference_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_shard_infer_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_sharded_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_remote_sharded_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["shard-infer-beta", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_sharded_inference_beta_v1")

    def test_micro_llm_shard_infer_beta_wraps_beta_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_micro_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--decode-steps", command)
            self.assertIn("--mode", command)
            self.assertIn("--stage-mode", command)
            return completed({
                "schema": "remote_micro_llm_sharded_beta_v1",
                "ok": True,
                "mode": "remote-loopback",
                "diagnosis_codes": [
                    "remote_micro_llm_sharded_ready",
                    "remote_micro_llm_sharded_loopback_ready",
                    "micro_llm_sharded_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "micro-llm-shard-infer-beta",
            "--output-dir",
            str(output_dir),
            "--mode",
            "remote-loopback",
            "--decode-steps",
            "4",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
        ])
        summary = cli.build_remote_micro_llm_sharded_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_micro_llm_sharded_beta_v1")
        self.assertEqual(summary["cli_schema"], "remote_micro_llm_sharded_beta_cli_v1")
        self.assertIn("remote_micro_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_micro_llm_shard_infer_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_micro_llm_sharded_beta_v1", "ok": True}
        with patch.object(cli, "build_remote_micro_llm_sharded_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["micro-llm-shard-infer-beta", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_micro_llm_sharded_beta_v1")

    def test_real_llm_shard_infer_beta_wraps_beta_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_real_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--hf-model-id", command)
            self.assertIn("--mode", command)
            self.assertIn("--stage-mode", command)
            return completed({
                "schema": "remote_real_llm_sharded_beta_v1",
                "ok": True,
                "mode": "remote-loopback",
                "diagnosis_codes": [
                    "remote_real_llm_sharded_ready",
                    "remote_real_llm_sharded_loopback_ready",
                    "real_llm_sharded_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                    "real_llm_artifact_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "real-llm-shard-infer-beta",
            "--output-dir",
            str(output_dir),
            "--mode",
            "remote-loopback",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
        ])
        summary = cli.build_remote_real_llm_sharded_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_real_llm_sharded_beta_v1")
        self.assertEqual(summary["cli_schema"], "remote_real_llm_sharded_beta_cli_v1")
        self.assertIn("remote_real_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_real_llm_shard_infer_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_real_llm_sharded_beta_v1", "ok": True}
        with patch.object(cli, "build_remote_real_llm_sharded_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-llm-shard-infer-beta", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_real_llm_sharded_beta_v1")

    def test_swarm_infer_beta_wraps_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--real-internet-beta-report", command)
            return completed({
                "schema": "swarm_inference_beta_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["swarm_inference_beta_ready", "operator-secret", "admin-secret"],
                "step": {"stderr_tail": "operator-secret admin-secret"},
            })

        args = cli.parse_args([
            "swarm-infer-beta",
            "verify",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
            "--real-internet-beta-report",
            "/tmp/real_llm_internet_beta.json",
        ])
        summary = cli.build_swarm_inference_beta(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "swarm_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "swarm_inference_beta_cli_v1")
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_swarm_infer_beta_live_wraps_public_kaggle_auto_path(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "live")
            self.assertIn("--public-host", command)
            self.assertEqual(command[command.index("--public-host") + 1], "24.199.118.54")
            self.assertIn("--base-port", command)
            self.assertIn("--kaggle-owner", command)
            self.assertIn("--inline-kernel-payload", command)
            self.assertNotIn("--keep-live-private-artifacts", command)
            return completed({
                "schema": "swarm_inference_beta_v1",
                "ok": True,
                "mode": "live",
                "diagnosis_codes": ["swarm_inference_beta_live_ready"],
            })

        args = cli.parse_args([
            "swarm-infer-beta",
            "live",
            "--output-dir",
            str(output_dir),
            "--public-host",
            "24.199.118.54",
            "--port",
            "9210",
            "--base-port",
            "9211",
            "--kaggle-owner",
            "xuyuhaosuyi",
        ])
        summary = cli.build_swarm_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "swarm_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "swarm_inference_beta_cli_v1")
        self.assertTrue(calls)

        args_keep = cli.parse_args([
            "swarm-infer-beta",
            "live",
            "--output-dir",
            str(output_dir),
            "--keep-live-private-artifacts",
        ])

        def fake_runner_keep(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--keep-live-private-artifacts", command)
            return completed({"schema": "swarm_inference_beta_v1", "ok": True, "mode": "live"})

        cli.build_swarm_inference_beta(args_keep, runner=fake_runner_keep)

    def test_main_swarm_infer_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "swarm_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_swarm_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["swarm-infer-beta", "clean", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "swarm_inference_beta_v1")

    def test_swarm_session_wraps_public_alpha_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_alpha_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "live-kaggle")
            self.assertIn("--failure-mode", command)
            self.assertEqual(command[command.index("--failure-mode") + 1], "kill-stage0-after-claim")
            self.assertIn("--kaggle-owner", command)
            self.assertIn("--kaggle-push-timeout-seconds", command)
            self.assertEqual(command.count("--kaggle-push-timeout-seconds"), 1)
            self.assertNotIn("--keep-child-artifacts", command)
            return completed({
                "schema": "public_swarm_inference_alpha_v1",
                "ok": True,
                "mode": "live-kaggle",
                "diagnosis_codes": ["public_swarm_inference_alpha_ready"],
            })

        args = cli.parse_args([
            "swarm-session",
            "--mode",
            "live-kaggle",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "xuyuhaosuyi",
        ])
        summary = cli.build_public_swarm_inference_alpha(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_inference_alpha_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_alpha_cli_v1")
        self.assertTrue(calls)

    def test_main_swarm_session_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_inference_alpha_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_inference_alpha", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["swarm-session", "--mode", "local-generated", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_inference_alpha_v1")

    def test_public_swarm_alpha_rc_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_alpha_rc_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "evidence-import")
            self.assertIn("--stage0-report", command)
            self.assertIn("--stage1-report", command)
            self.assertIn("--summary-report", command)
            return completed({
                "schema": "public_swarm_inference_alpha_rc_v1",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_swarm_inference_alpha_rc_ready"],
            })

        args = cli.parse_args([
            "public-swarm-alpha-rc",
            "--output-dir",
            str(output_dir),
        ])
        summary = cli.build_public_swarm_inference_alpha_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_inference_alpha_rc_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_alpha_rc_cli_v1")
        self.assertTrue(calls)

    def test_main_public_swarm_alpha_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_inference_alpha_rc_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_inference_alpha_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-alpha-rc", "--mode", "local-smoke", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_inference_alpha_rc_v1")

    def test_public_swarm_beta_wraps_beta_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            return completed({
                "schema": "public_swarm_inference_beta_v1",
                "ok": True,
                "mode": "prepare",
                "diagnosis_codes": ["public_swarm_inference_beta_ready", "operator-secret", "admin-secret"],
            })

        args = cli.parse_args([
            "public-swarm-beta",
            "prepare",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_public_swarm_inference_beta(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_cli_v1")
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_public_swarm_beta_local_loopback_forwards_split_runtime_flags(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "local-loopback")
            self.assertIn("--base-port", command)
            self.assertIn("--hf-model-id", command)
            return completed({
                "schema": "public_swarm_inference_beta_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_inference_beta_ready", "local_loopback_ready"],
            })

        args = cli.parse_args([
            "public-swarm-beta",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9290",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
        ])
        summary = cli.build_public_swarm_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_cli_v1")

    def test_public_swarm_beta_evidence_import_forwards_retained_reports(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "evidence-import")
            self.assertIn("--alpha-rc-report", command)
            self.assertIn("--stage0-report", command)
            self.assertIn("--stage1-report", command)
            self.assertIn("--summary-report", command)
            return completed({
                "schema": "public_swarm_inference_beta_v1",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_swarm_inference_beta_ready", "external_live_evidence_imported"],
            })

        args = cli.parse_args([
            "public-swarm-beta",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--alpha-rc-report",
            "/tmp/alpha_rc.json",
            "--stage0-report",
            "/tmp/stage0.json",
            "--stage1-report",
            "/tmp/stage1.json",
            "--summary-report",
            "/tmp/summary.json",
        ])
        summary = cli.build_public_swarm_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("external_live_evidence_imported", summary["diagnosis_codes"])

    def test_public_swarm_beta_product_beta_forwards_product_flags(self) -> None:
        output_dir = Path(self._tmp_dir())
        gpu_report = output_dir / "gpu.json"
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "product-beta")
            self.assertIn("--gpu-report", command)
            self.assertEqual(command[command.index("--gpu-report") + 1], str(gpu_report))
            self.assertIn("--max-new-tokens", command)
            self.assertIn("--cpu-request-count", command)
            self.assertIn("--external-llm-request-count", command)
            return completed({
                "schema": "public_swarm_inference_beta_v1",
                "ok": True,
                "mode": "product-beta",
                "diagnosis_codes": [
                    "public_swarm_inference_beta_ready",
                    "public_swarm_product_beta_ready",
                    "session_protocol_ready",
                    "p2p_lite_discovery_ready",
                    "cpu_fallback_ready",
                ],
            })

        args = cli.parse_args([
            "public-swarm-beta",
            "product-beta",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(gpu_report),
            "--max-new-tokens",
            "4",
            "--cpu-request-count",
            "1",
            "--external-llm-request-count",
            "1",
        ])
        summary = cli.build_public_swarm_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["mode"], "product-beta")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_cli_v1")
        self.assertIn("public_swarm_product_beta_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_public_swarm_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-beta", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_inference_beta_v1")

    def test_public_swarm_beta_rc_wraps_rc_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_beta_rc_pack.py", command[1])
            self.assertEqual(command[2], "external-existing")
            self.assertIn("--coordinator-url", command)
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--max-new-tokens", command)
            return completed({
                "schema": "public_swarm_inference_beta_rc_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": [
                    "public_swarm_inference_beta_rc_ready",
                    "observer-secret",
                    "admin-secret",
                ],
            })

        args = cli.parse_args([
            "public-swarm-beta-rc",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
        ])
        summary = cli.build_public_swarm_inference_beta_rc(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_inference_beta_rc_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_rc_cli_v1")
        self.assertEqual(summary["mode"], "external-existing")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_main_public_swarm_beta_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_inference_beta_rc_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_inference_beta_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-beta-rc", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_inference_beta_rc_v1")

    def test_public_swarm_product_beta_wraps_product_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_product_beta_pack.py", command[1])
            self.assertEqual(command[2], "external-existing")
            self.assertIn("--coordinator-url", command)
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            return completed({
                "schema": "public_swarm_product_beta_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": [
                    "public_swarm_product_beta_ready",
                    "observer-secret",
                    "admin-secret",
                ],
            })

        args = cli.parse_args([
            "public-swarm-product-beta",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_public_swarm_product_beta(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_product_beta_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_product_beta_cli_v1")
        self.assertEqual(summary["mode"], "external-existing")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_main_public_swarm_product_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_product_beta_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_product_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-product-beta", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_product_beta_v1")

    def test_public_swarm_gpu_beta_wraps_gpu_pack(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_gpu_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "local-loopback")
            self.assertIn("--base-port", command)
            self.assertIn("--hf-model-id", command)
            return completed({
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_gpu_beta_ready", "hf_transformers_cuda_ready"],
            })

        args = cli.parse_args([
            "public-swarm-gpu-beta",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9390",
        ])
        summary = cli.build_public_swarm_gpu_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_gpu_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_gpu_inference_beta_cli_v1")

    def test_public_swarm_gpu_beta_evidence_import_forwards_gpu_report(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_gpu_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "evidence-import")
            self.assertIn("--gpu-report", command)
            self.assertEqual(command[command.index("--gpu-report") + 1], "/tmp/gpu.json")
            return completed({
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_swarm_gpu_beta_evidence_import_ready"],
            })

        args = cli.parse_args([
            "public-swarm-gpu-beta",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            "/tmp/gpu.json",
        ])
        summary = cli.build_public_swarm_gpu_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_gpu_inference_beta_cli_v1")

    def test_public_swarm_gpu_beta_kaggle_auto_forwards_kaggle_options(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_gpu_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "kaggle-auto")
            self.assertIn("--public-host", command)
            self.assertEqual(command[command.index("--public-host") + 1], "24.199.118.54")
            self.assertIn("--kaggle-owner", command)
            self.assertEqual(command[command.index("--kaggle-owner") + 1], "xuyuhaosuyi")
            self.assertIn("--kernel-slug-prefix", command)
            self.assertIn("--inline-kernel-payload", command)
            return completed({
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "mode": "kaggle-auto",
                "diagnosis_codes": ["public_swarm_gpu_beta_kaggle_auto_ready"],
            })

        args = cli.parse_args([
            "public-swarm-gpu-beta",
            "kaggle-auto",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--kernel-slug-prefix",
            "crowdtensor-public-swarm-gpu-beta-test",
        ])
        summary = cli.build_public_swarm_gpu_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_gpu_inference_beta_cli_v1")

    def test_gpu_generate_wraps_generation_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("gpu_sharded_generation_beta_pack.py", command[1])
            self.assertIn("local-loopback", command)
            self.assertIn("--max-new-tokens", command)
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "8")
            return completed({
                "schema": "gpu_sharded_generation_beta_v1",
                "ok": True,
                "diagnosis_codes": ["gpu_sharded_generation_ready"],
            })

        args = cli.parse_args([
            "gpu-generate",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
        ])

        summary = cli.build_gpu_sharded_generation_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "gpu_sharded_generation_beta_cli_v1")
        self.assertTrue(calls)

    def test_gpu_generate_evidence_import_forwards_report(self) -> None:
        output_dir = Path(self._tmp_dir())
        source = output_dir / "gpu.json"
        source.write_text("{}", encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("evidence-import", command)
            self.assertIn("--gpu-report", command)
            self.assertEqual(command[command.index("--gpu-report") + 1], str(source))
            return completed({
                "schema": "gpu_sharded_generation_beta_v1",
                "ok": True,
                "diagnosis_codes": ["gpu_sharded_generation_ready"],
            })

        args = cli.parse_args([
            "gpu-generate",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            "4",
        ])

        summary = cli.build_gpu_sharded_generation_beta(args, runner=fake_runner)

        self.assertEqual(summary["cli_schema"], "gpu_sharded_generation_beta_cli_v1")

    def test_main_public_swarm_gpu_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_gpu_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_gpu_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-gpu-beta", "local-smoke", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_gpu_inference_beta_v1")

    def test_main_gpu_generate_json_outputs_summary(self) -> None:
        summary = {"schema": "gpu_sharded_generation_beta_v1", "ok": True}
        with patch.object(cli, "build_gpu_sharded_generation_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["gpu-generate", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "gpu_sharded_generation_beta_v1")


    def test_micro_llm_live_rc_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("micro_llm_live_rc_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "local-generated")
            self.assertIn("--decode-steps", command)
            self.assertEqual(command[command.index("--decode-steps") + 1], "3")
            self.assertIn("--max-request-attempts", command)
            return completed({
                "schema": "micro_llm_live_rc_v1",
                "ok": True,
                "mode": "local-generated",
                "diagnosis_codes": [
                    "micro_llm_live_rc_ready",
                    "local_generated_stage_upload_standins_ready",
                    "kaggle_micro_llm_sharded_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "micro-llm-live-rc",
            "--output-dir",
            str(output_dir),
            "--port",
            "9180",
            "--request-count",
            "2",
            "--decode-steps",
            "3",
        ])
        summary = cli.build_micro_llm_live_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "micro_llm_live_rc_v1")
        self.assertEqual(summary["cli_schema"], "micro_llm_live_rc_cli_v1")
        self.assertIn("micro_llm_live_rc_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_micro_llm_live_rc_external_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("micro_llm_live_rc_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "external-existing")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--coordinator-url", command)
            return completed({
                "schema": "micro_llm_live_rc_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": ["micro_llm_live_rc_ready", "external_runtime_verified"],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = cli.parse_args([
            "micro-llm-live-rc",
            "--mode",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://24.199.118.54:9180",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_micro_llm_live_rc(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("external_runtime_verified", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_main_micro_llm_live_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "micro_llm_live_rc_v1", "ok": True}
        with patch.object(cli, "build_micro_llm_live_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["micro-llm-live-rc", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "micro_llm_live_rc_v1")

    def test_real_llm_live_rc_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_live_rc_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "local-generated")
            self.assertIn("--hf-model-id", command)
            self.assertIn("--max-request-attempts", command)
            return completed({
                "schema": "real_llm_live_rc_v1",
                "ok": True,
                "mode": "local-generated",
                "diagnosis_codes": [
                    "real_llm_live_rc_ready",
                    "local_generated_real_llm_stage_upload_standins_ready",
                    "remote_real_llm_sharded_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "real-llm-live-rc",
            "--output-dir",
            str(output_dir),
            "--port",
            "9184",
            "--request-count",
            "1",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
        ])
        summary = cli.build_real_llm_live_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_llm_live_rc_v1")
        self.assertEqual(summary["cli_schema"], "real_llm_live_rc_cli_v1")
        self.assertIn("real_llm_live_rc_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_real_llm_live_rc_external_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("real_llm_live_rc_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "external-existing")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--coordinator-url", command)
            return completed({
                "schema": "real_llm_live_rc_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": ["real_llm_live_rc_ready", "external_runtime_verified"],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = cli.parse_args([
            "real-llm-live-rc",
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
        ])
        summary = cli.build_real_llm_live_rc(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("external_runtime_verified", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_main_real_llm_live_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "real_llm_live_rc_v1", "ok": True}
        with patch.object(cli, "build_real_llm_live_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-llm-live-rc", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "real_llm_live_rc_v1")

    def test_real_llm_internet_alpha_wraps_alpha_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_internet_alpha_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "local-generated")
            self.assertIn("--base-port", command)
            self.assertIn("--hf-model-id", command)
            return completed({
                "schema": "real_llm_internet_alpha_v1",
                "ok": True,
                "mode": "local-generated",
                "diagnosis_codes": [
                    "real_llm_internet_alpha_ready",
                    "real_llm_stage_requeue_ready",
                    "real_llm_live_rc_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "real-llm-internet-alpha",
            "--output-dir",
            str(output_dir),
            "--port",
            "9186",
            "--base-port",
            "9188",
            "--request-count",
            "1",
        ])
        summary = cli.build_real_llm_internet_alpha(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_llm_internet_alpha_v1")
        self.assertEqual(summary["cli_schema"], "real_llm_internet_alpha_cli_v1")
        self.assertIn("real_llm_internet_alpha_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_real_llm_internet_alpha_external_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("real_llm_internet_alpha_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "external-existing")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--coordinator-url", command)
            return completed({
                "schema": "real_llm_internet_alpha_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": ["real_llm_internet_alpha_ready", "external_runtime_verified"],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = cli.parse_args([
            "real-llm-internet-alpha",
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
        summary = cli.build_real_llm_internet_alpha(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("external_runtime_verified", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_main_real_llm_internet_alpha_json_outputs_summary(self) -> None:
        summary = {"schema": "real_llm_internet_alpha_v1", "ok": True}
        with patch.object(cli, "build_real_llm_internet_alpha", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-llm-internet-alpha", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "real_llm_internet_alpha_v1")

    def test_real_llm_internet_beta_wraps_beta_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_internet_beta_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "kaggle-auto")
            self.assertIn("--kaggle-owner", command)
            self.assertIn("--kaggle-push-timeout-seconds", command)
            self.assertIn("--kaggle-status-timeout-seconds", command)
            return completed({
                "schema": "real_llm_internet_beta_v1",
                "ok": True,
                "mode": "kaggle-auto",
                "diagnosis_codes": [
                    "real_llm_internet_beta_ready",
                    "real_llm_internet_alpha_ready",
                    "external_runtime_verified",
                    "kaggle_kernels_deleted",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "real-llm-internet-beta",
            "--output-dir",
            str(output_dir),
            "--port",
            "9190",
            "--base-port",
            "9191",
            "--request-count",
            "2",
            "--kaggle-owner",
            "xuyuhaosuyi",
        ])
        summary = cli.build_real_llm_internet_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_llm_internet_beta_v1")
        self.assertEqual(summary["cli_schema"], "real_llm_internet_beta_cli_v1")
        self.assertIn("real_llm_internet_beta_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_real_llm_internet_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "real_llm_internet_beta_v1", "ok": True}
        with patch.object(cli, "build_real_llm_internet_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-llm-internet-beta", "--kaggle-owner", "xuyuhaosuyi", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "real_llm_internet_beta_v1")

    def test_release_ready_wraps_readiness_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("release_readiness_pack.py", command[1])
            return completed({
                "ok": True,
                "schema": "release_readiness_v1",
                "release_status": {
                    "ready": True,
                    "status": "ready",
                    "diagnosis_codes": ["release_ready"],
                },
            })

        args = cli.parse_args([
            "release-ready",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9024",
            "--request-count",
            "4",
            "--allow-dirty",
        ])

        report = cli.build_release_ready(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "release_readiness_v1")
        self.assertIn("release_ready", report["release_status"]["diagnosis_codes"])
        self.assertTrue(any("--allow-dirty" in command for command in calls))

    def test_main_release_ready_json_outputs_report(self) -> None:
        report = {"schema": "release_readiness_v1", "ok": True}
        with patch.object(cli, "build_release_ready", return_value=report), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["release-ready", "--allow-dirty", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "release_readiness_v1")

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

    def test_remote_demo_prepare_wraps_home_compute_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "prepare",
                "diagnosis_codes": ["remote_home_compute_prepare_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "prepare",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
            "--replace",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_demo_v1")
        self.assertEqual(summary["mode"], "prepare")
        self.assertTrue(any("--replace" in command for command in calls))

    def test_remote_demo_prepare_forwards_kaggle_target(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--target", command)
            self.assertIn("kaggle", command)
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "prepare",
                "target_environment": {"name": "kaggle", "kaggle_remote_miner_beta": True},
                "diagnosis_codes": ["kaggle_remote_miner_prepare_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "prepare",
            "--target",
            "kaggle",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "kaggle-cpu-1",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["target_environment"]["name"], "kaggle")
        self.assertIn("kaggle_remote_miner_prepare_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_remote_demo_verify_defaults_create_session_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--create-session", command)
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["remote_home_compute_ready"],
                "step": {
                    "stderr_tail": "observer-secret should be redacted",
                },
            })

        args = cli.parse_args([
            "remote-demo",
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

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_demo_v1")
        self.assertIn("remote_home_compute_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(any("--remote-timeout-seconds" in command for command in calls))

    def test_remote_demo_external_llm_forwards_workload_and_runtime_flags(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--workload", command)
            self.assertIn("external-llm", command)
            self.assertIn("--mock", command)
            self.assertIn("--llm-runtime-url", command)
            self.assertIn("--llm-runtime-api-key", command)
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["remote_external_llm_ready", "remote_home_compute_ready"],
                "demo": {"workload_kind": "external-llm", "workload_type": "external_llm_infer"},
            })

        args = cli.parse_args([
            "remote-demo",
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
            "--llm-runtime-url",
            "http://127.0.0.1:11434/v1/chat/completions",
            "--llm-runtime-api-key",
            "runtime-secret",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["demo"]["workload_type"], "external_llm_infer")
        self.assertIn("remote_external_llm_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("runtime-secret", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)

    def test_remote_demo_micro_llm_forwards_workload_and_decode_steps(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--workload", command)
            self.assertIn("micro-llm-sharded", command)
            self.assertIn("--decode-steps", command)
            self.assertEqual(command[command.index("--decode-steps") + 1], "3")
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["remote_micro_llm_sharded_ready", "remote_home_compute_ready"],
                "demo": {"workload_kind": "micro-llm-sharded", "workload_type": "micro_llm_sharded_infer"},
            })

        args = cli.parse_args([
            "remote-demo",
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

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["demo"]["workload_type"], "micro_llm_sharded_infer")
        self.assertIn("remote_micro_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_remote_demo_real_llm_forwards_hf_and_split_flags(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--workload", command)
            self.assertIn("real-llm-sharded", command)
            self.assertIn("--stage-mode", command)
            self.assertEqual(command[command.index("--stage-mode") + 1], "split")
            self.assertIn("--require-distinct-stage-miners", command)
            self.assertIn("--hf-model-id", command)
            self.assertEqual(command[command.index("--hf-model-id") + 1], "sshleifer/tiny-gpt2")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "real prompt")
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["remote_real_llm_sharded_ready", "remote_home_compute_ready"],
                "demo": {"workload_kind": "real-llm-sharded", "workload_type": "real_llm_sharded_infer"},
            })

        args = cli.parse_args([
            "remote-demo",
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
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
            "--prompt-texts",
            "real prompt",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["demo"]["workload_type"], "real_llm_sharded_infer")
        self.assertIn("remote_real_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_remote_demo_doctor_forwards_tokens_and_require_result(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "doctor")
            self.assertIn("--require-result", command)
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            return completed({
                "schema": "remote_home_compute_doctor_v1",
                "ok": True,
                "diagnosis_codes": ["remote_home_compute_doctor_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
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
            "--require-result",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_doctor_v1")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_remote_demo_collect_forwards_task_and_external_runtime_flags(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "collect")
            self.assertIn("--task-id", command)
            self.assertIn("task-1", command)
            self.assertIn("--mock", command)
            self.assertIn("--llm-runtime-url", command)
            return completed({
                "schema": "remote_home_compute_collect_v1",
                "ok": True,
                "diagnosis_codes": ["remote_home_compute_collect_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "collect",
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
            "--task-id",
            "task-1",
            "--mock",
            "--llm-runtime-url",
            "http://127.0.0.1:11434/v1/chat/completions",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_collect_v1")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)

    def test_remote_demo_clean_uses_cleanup_mode_without_workload_args(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "clean")
            self.assertIn("--apply", command)
            self.assertIn("--include-private", command)
            self.assertNotIn("--workload", command)
            return completed({
                "schema": "remote_home_compute_cleanup_v1",
                "ok": True,
                "diagnosis_codes": ["remote_home_compute_cleanup_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "clean",
            "--output-dir",
            str(output_dir),
            "--apply",
            "--include-private",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_cleanup_v1")
        self.assertTrue(calls)

    def test_remote_demo_kaggle_real_prepare_wraps_acceptance_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("kaggle_real_runtime_acceptance_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            self.assertIn("--public-host", command)
            self.assertIn("24.199.118.54", command)
            self.assertIn("--port", command)
            self.assertIn("9180", command)
            self.assertIn("--workload", command)
            self.assertEqual(command[command.index("--workload") + 1], "micro-llm-sharded")
            self.assertIn("--decode-steps", command)
            self.assertEqual(command[command.index("--decode-steps") + 1], "3")
            self.assertIn("--stage-mode", command)
            self.assertEqual(command[command.index("--stage-mode") + 1], "split")
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            self.assertIn("--require-distinct-stage-miners", command)
            self.assertIn("--replace", command)
            return completed({
                "schema": "kaggle_real_runtime_acceptance_v1",
                "ok": True,
                "mode": "prepare",
                "diagnosis_codes": ["kaggle_artifacts_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "kaggle-real",
            "--action",
            "prepare",
            "--public-host",
            "24.199.118.54",
            "--port",
            "9180",
            "--miner-id",
            "kaggle-cpu-1",
            "--workload",
            "micro-llm-sharded",
            "--decode-steps",
            "3",
            "--stage-mode",
            "split",
            "--micro-llm-artifact",
            "dist/micro-llm-artifact",
            "--prompt-texts",
            "arn,ten",
            "--output-dir",
            str(output_dir),
            "--replace",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "kaggle_real_runtime_acceptance_v1")
        self.assertIn("kaggle_artifacts_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_remote_demo_kaggle_real_verify_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("kaggle_real_runtime_acceptance_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--remote-timeout-seconds", command)
            self.assertIn("--workload", command)
            self.assertEqual(command[command.index("--workload") + 1], "micro-llm-sharded")
            self.assertIn("--stage-mode", command)
            self.assertIn("--require-distinct-stage-miners", command)
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            return completed({
                "schema": "kaggle_real_runtime_acceptance_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["kaggle_real_runtime_ready"],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = cli.parse_args([
            "remote-demo",
            "kaggle-real",
            "--action",
            "verify",
            "--public-host",
            "24.199.118.54",
            "--port",
            "9180",
            "--miner-id",
            "kaggle-cpu-1",
            "--workload",
            "micro-llm-sharded",
            "--stage-mode",
            "split",
            "--micro-llm-artifact",
            "dist/micro-llm-artifact",
            "--prompt-texts",
            "arn,ten",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("kaggle_real_runtime_ready", summary["diagnosis_codes"])
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

    def test_main_remote_demo_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_home_compute_demo_v1", "ok": True}
        with patch.object(cli, "build_remote_demo", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main([
                    "remote-demo",
                    "verify",
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
        self.assertEqual(payload["schema"], "remote_home_compute_demo_v1")


if __name__ == "__main__":
    unittest.main()
