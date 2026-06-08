from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_inference_beta_check as check
from scripts import public_swarm_inference_beta_pack as pack


class PublicSwarmInferenceBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_beta_test_"))

    def _alpha_stage_report(self, *, stage: str) -> dict:
        failure_mode = f"kill-{stage}-after-claim"
        stage_code = f"live_{stage}_requeue_ready"
        return {
            "schema": "public_swarm_inference_alpha_v1",
            "ok": True,
            "mode": "live-kaggle",
            "failure_mode": failure_mode,
            "diagnosis_codes": [
                "public_swarm_inference_alpha_ready",
                "public_swarm_session_ready",
                "public_swarm_live_requeue_ready",
                "public_swarm_live_kaggle_ready",
                "external_stage_requeue_ready",
                stage_code,
                "external_runtime_verified",
                "kaggle_kernels_deleted",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "token_rotation_required",
            ],
            "session": {
                "live_external_runtime_verified": True,
                "live_stage_requeue_verified": True,
                "live_kaggle_kernels_deleted": True,
                "live_summary": {
                    "live_requeue_summary": {
                        "enabled": True,
                        "failure_mode": failure_mode,
                        "target_stage": stage,
                        "claim_observed": True,
                        "victim_kernel_deleted": True,
                        "lease_expired": "<redacted>",
                        "rescued_result": True,
                        "victim_result_accepted": False,
                    },
                },
            },
            "safety": {
                "cpu_only": True,
                "read_only_workload": "real_llm_sharded_infer",
                "not_production": True,
                "not_p2p": True,
                "not_large_model_serving": True,
                "not_public_prompt_serving": True,
            },
            "artifact_cleanup": {"child_artifacts_pruned": True},
            "artifacts": {
                "local_requeue_json": {"present": False},
                "live_swarm_beta_json": {"present": False},
                "live_support_bundle_json": {"present": False},
            },
        }

    def _write_alpha_reports(self, root: Path) -> tuple[Path, Path, Path, Path]:
        stage0_dir = root / "stage0"
        stage1_dir = root / "stage1"
        stage0_dir.mkdir(parents=True)
        stage1_dir.mkdir(parents=True)
        stage0 = stage0_dir / "public_swarm_inference_alpha.json"
        stage1 = stage1_dir / "public_swarm_inference_alpha.json"
        summary = root / "public-swarm-inference-alpha-live-requeue-summary.json"
        stage0.write_text(json.dumps(self._alpha_stage_report(stage="stage0"), sort_keys=True), encoding="utf-8")
        stage1.write_text(json.dumps(self._alpha_stage_report(stage="stage1"), sort_keys=True), encoding="utf-8")
        summary.write_text(json.dumps({
            "schema": "public_swarm_inference_alpha_live_requeue_summary_v1",
            "ok": True,
            "proofs": [
                {
                    "ok": True,
                    "target_stage": "stage0",
                    "claim_observed": True,
                    "victim_kernel_deleted": True,
                    "rescued_result": True,
                    "victim_result_accepted": False,
                },
                {
                    "ok": True,
                    "target_stage": "stage1",
                    "claim_observed": True,
                    "victim_kernel_deleted": True,
                    "rescued_result": True,
                    "victim_result_accepted": False,
                },
            ],
        }, sort_keys=True), encoding="utf-8")
        alpha_rc = root / "public_swarm_inference_alpha_rc.json"
        alpha_rc.write_text(json.dumps({
            "schema": "public_swarm_inference_alpha_rc_v1",
            "ok": True,
            "mode": "evidence-import",
            "diagnosis_codes": [
                "public_swarm_inference_alpha_rc_ready",
                "public_swarm_alpha_rc_evidence_imported",
                "stage0_live_requeue_evidence_ready",
                "stage1_live_requeue_evidence_ready",
                "public_swarm_live_requeue_evidence_ready",
                "public_swarm_live_requeue_summary_ready",
                "public_swarm_alpha_private_artifacts_absent",
            ],
        }, sort_keys=True), encoding="utf-8")
        return alpha_rc, stage0, stage1, summary

    def test_evidence_import_promotes_alpha_rc_into_public_beta(self) -> None:
        root = self._tmp_dir()
        alpha_rc, stage0, stage1, summary = self._write_alpha_reports(root)
        output_dir = self._tmp_dir() / "beta"

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--alpha-rc-report",
            str(alpha_rc),
            "--stage0-report",
            str(stage0),
            "--stage1-report",
            str(stage1),
            "--summary-report",
            str(summary),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "public_swarm_inference_beta_v1")
        self.assertEqual(report["mode"], "evidence-import")
        for code in [
            "public_swarm_inference_beta_ready",
            "public_swarm_beta_evidence_import_ready",
            "external_live_evidence_imported",
            "stage0_live_requeue_evidence_ready",
            "stage1_live_requeue_evidence_ready",
            "two_stage_split_inference_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ]:
            self.assertIn(code, report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["public_swarm_inference_beta_json"]["present"])
        self.assertTrue((output_dir / "public_swarm_inference_beta.md").is_file())

    def test_evidence_import_blocks_when_alpha_rc_missing(self) -> None:
        root = self._tmp_dir()
        _, stage0, stage1, summary = self._write_alpha_reports(root)
        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(self._tmp_dir()),
            "--alpha-rc-report",
            str(root / "missing.json"),
            "--stage0-report",
            str(stage0),
            "--stage1-report",
            str(stage1),
            "--summary-report",
            str(summary),
        ]))

        self.assertFalse(report["ok"], report)
        self.assertIn("alpha_rc_report_missing", report["imported_reports"]["alpha_rc"]["failed_checks"])
        self.assertIn("public_swarm_inference_beta_blocked", report["diagnosis_codes"])

    def test_local_loopback_wraps_remote_real_llm_check(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_real_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "remote-loopback")
            self.assertIn("--stage-mode", command)
            self.assertEqual(command[command.index("--stage-mode") + 1], "split")
            self.assertIn("--require-distinct-stage-miners", command)
            child_dir = Path(command[command.index("--output-dir") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "remote_real_llm_sharded_beta.json").write_text("{}", encoding="utf-8")
            (child_dir / "remote_real_llm_sharded_beta.md").write_text("# remote\n", encoding="utf-8")
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": "remote_real_llm_sharded_beta_v1",
                    "ok": True,
                    "mode": "remote-loopback",
                    "diagnosis_codes": [
                        "remote_real_llm_sharded_ready",
                        "remote_real_llm_sharded_loopback_ready",
                        "real_llm_sharded_ready",
                        "real_llm_artifact_ready",
                        "activation_transport_ready",
                        "baseline_match",
                        "decoded_tokens_match",
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                    ],
                    "payload_summaries": {
                        "remote_loopback_real_llm_sharded_inference": {
                            "session": {"stage_count": 2, "request_count": 1, "model_id": "sshleifer/tiny-gpt2"},
                            "stage_assignment": {
                                "stage0_miner_id": "miner-stage0",
                                "stage1_miner_id": "miner-stage1",
                                "distinct_stage_miners": True,
                                "stage_assignment_valid": True,
                            },
                        },
                    },
                    "safety": {"cpu_only_default": True, "activation_payloads_redacted": True},
                }) + "\n",
                stderr="",
            )

        private_prompts = "private local prompt one,private local prompt two"
        report = pack.build_report(pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9290",
            "--prompt-texts",
            private_prompts,
        ]), runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_inference_beta_ready", report["diagnosis_codes"])
        self.assertIn("local_loopback_ready", report["diagnosis_codes"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_public_safe"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertNotIn("private local prompt one", encoded)
        self.assertNotIn("private local prompt two", encoded)
        self.assertTrue(calls)

    def test_product_beta_aggregates_product_rc_and_cpu_fallback(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        gpu_report.write_text("{}", encoding="utf-8")
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            joined = " ".join(command)
            if "public_swarm_product_rc_pack.py" in joined:
                self.assertIn("--gpu-report", command)
                self.assertEqual(command[command.index("--gpu-report") + 1], str(gpu_report))
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "public_swarm_product_rc.json").write_text("{}", encoding="utf-8")
                gpu_import = child_dir / "gpu-generation-import"
                gpu_import.mkdir(parents=True, exist_ok=True)
                (gpu_import / "gpu_sharded_generation_beta_evidence_import.json").write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=json.dumps({
                        "schema": "public_swarm_product_rc_v1",
                        "ok": True,
                        "diagnosis_codes": [
                            "public_swarm_product_rc_ready",
                            "coordinator_product_surface_ready",
                            "session_protocol_ready",
                            "p2p_lite_discovery_ready",
                            "gpu_generation_evidence_import_ready",
                        ],
                        "product_surface": {
                            "serve": {"ok": True},
                            "join_stage0": {"ok": True},
                            "join_stage1": {"ok": True},
                            "generate": {"ok": True},
                            "peer_check": {"ok": True},
                        },
                        "session_protocol": {"ok": True, "schema": "session_protocol_check_v1", "route_usable": True},
                        "p2p_lite": {"ok": True, "schema": "p2p_lite_discovery_check_v1", "cpu_route_ok": True, "cuda_route_ok": True},
                        "gpu_generation_import": {
                            "ok": True,
                            "schema": "gpu_sharded_generation_beta_v1",
                            "mode": "evidence-import",
                            "generated_text_hash": "sha256:generation",
                            "raw_generated_text_public": False,
                        },
                    }) + "\n",
                    stderr="",
                )
            if "cpu_inference_beta_pack.py" in joined:
                self.assertIn("--mode", command)
                self.assertEqual(command[command.index("--mode") + 1], "local")
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
                (child_dir / "cpu_inference_beta.md").write_text("# cpu\n", encoding="utf-8")
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=json.dumps({
                        "schema": "cpu_inference_beta_v1",
                        "ok": True,
                        "mode": "local",
                        "workload": "all",
                        "diagnosis_codes": ["cpu_inference_beta_ready", "local_cpu_inference_ready"],
                        "steps": [{"name": "home_infer", "ok": True}, {"name": "llm_infer_mock", "ok": True}],
                        "safety": {"cpu_only_default": True, "summary_excludes_raw_inference_payloads": True},
                    }) + "\n",
                    stderr="",
                )
            raise AssertionError(command)

        private_prompts = "private product prompt one,private product prompt two"
        report = pack.build_report(pack.parse_args([
            "product-beta",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(gpu_report),
            "--prompt-texts",
            private_prompts,
            "--max-new-tokens",
            "4",
            "--cpu-request-count",
            "1",
            "--external-llm-request-count",
            "1",
        ]), runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "product-beta")
        for code in [
            "public_swarm_inference_beta_ready",
            "public_swarm_product_beta_ready",
            "coordinator_product_surface_ready",
            "session_protocol_ready",
            "p2p_lite_discovery_ready",
            "gpu_generation_evidence_import_ready",
            "cpu_fallback_ready",
        ]:
            self.assertIn(code, report["diagnosis_codes"])
        self.assertTrue(report["beta"]["product_beta"])
        self.assertTrue(report["safety"]["p2p_lite_discovery_only"])
        self.assertTrue(report["artifacts"]["public_swarm_product_rc_json"]["present"])
        self.assertTrue(report["artifacts"]["cpu_inference_beta_json"]["present"])
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_public_safe"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(report["answer_scope"]["terminal_only"])
        self.assertEqual(report["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(report["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertFalse(report["answer_scope"]["raw_prompt_public"])
        self.assertFalse(report["answer_scope"]["raw_generated_text_public"])
        self.assertFalse(report["answer_scope"]["generated_token_ids_public"])
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        markdown = (output_dir / "public_swarm_inference_beta.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn(
            "- prompt scope: `source=prompt-texts count=2 inline_prompt_text=True terminal_next_commands_local_private=True saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        self.assertNotIn("private product prompt one", encoded)
        self.assertNotIn("private product prompt two", encoded)
        self.assertNotIn("private product prompt one", markdown)
        self.assertNotIn("private product prompt two", markdown)
        self.assertTrue(calls)

    def test_product_beta_blocks_when_cpu_fallback_missing(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        gpu_report.write_text("{}", encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "public_swarm_product_rc_pack.py" in joined:
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=json.dumps({
                        "schema": "public_swarm_product_rc_v1",
                        "ok": True,
                        "diagnosis_codes": [
                            "public_swarm_product_rc_ready",
                            "coordinator_product_surface_ready",
                            "session_protocol_ready",
                            "p2p_lite_discovery_ready",
                            "gpu_generation_evidence_import_ready",
                        ],
                    }) + "\n",
                    stderr="",
                )
            if "cpu_inference_beta_pack.py" in joined:
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=json.dumps({
                        "schema": "cpu_inference_beta_v1",
                        "ok": False,
                        "diagnosis_codes": ["cpu_inference_beta_failed"],
                    }) + "\n",
                    stderr="",
                )
            raise AssertionError(command)

        report = pack.build_report(pack.parse_args([
            "product-beta",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(gpu_report),
        ]), runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertIn("public_swarm_inference_beta_blocked", report["diagnosis_codes"])
        self.assertFalse(report["beta"]["cpu_fallback_ready"])

    def test_prepare_wraps_existing_swarm_beta_without_leaking_tokens(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": "swarm_inference_beta_v1",
                    "ok": True,
                    "mode": "prepare",
                    "diagnosis_codes": [
                        "swarm_inference_beta_prepare_ready",
                        "two_machine_runbook_ready",
                        "stage0_join_pack_ready",
                        "stage1_join_pack_ready",
                        "miner_registry_hashed",
                    ],
                    "stage_join_packs": [{"stage": "stage0"}, {"stage": "stage1"}],
                }) + "\n",
                stderr="",
            )

        report = pack.build_report(pack.parse_args([
            "prepare",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
        ]), runner=fake_runner)
        serialized = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_beta_operator_workflow_ready", report["diagnosis_codes"])
        self.assertIn("two_stage_join_pack_ready", report["diagnosis_codes"])
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(check.output_scope_errors(report), [])
        broken = dict(report)
        broken["output_request"] = {**report["output_request"], "include_output": True}
        self.assertIn("output_request_include_output_mismatch", check.output_scope_errors(broken))
        broken_prompt = dict(report)
        broken_prompt["prompt_scope"] = {**report["prompt_scope"], "raw_prompt_public": True}
        self.assertIn("prompt_scope_raw_prompt_public_mismatch", check.output_scope_errors(broken_prompt))
        self.assertTrue(calls)

    def test_product_beta_redacts_prompt_texts_from_failed_child_output(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        gpu_report.write_text("{}", encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "public_swarm_product_rc_pack.py" in joined:
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=1,
                    stdout="failed with private failure prompt one\n",
                    stderr="stderr private failure prompt two\n",
                )
            if "cpu_inference_beta_pack.py" in joined:
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=json.dumps({
                        "schema": "cpu_inference_beta_v1",
                        "ok": True,
                        "diagnosis_codes": ["cpu_inference_beta_ready", "local_cpu_inference_ready"],
                    }) + "\n",
                    stderr="",
                )
            raise AssertionError(command)

        report = pack.build_report(pack.parse_args([
            "product-beta",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(gpu_report),
            "--prompt-texts",
            "private failure prompt one,private failure prompt two",
        ]), runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertFalse(report["ok"])
        self.assertIn("public_swarm_inference_beta_blocked", report["diagnosis_codes"])
        self.assertIn("<redacted>", encoded)
        self.assertNotIn("private failure prompt one", encoded)
        self.assertNotIn("private failure prompt two", encoded)

    def test_local_loopback_forwards_prompt_texts_file(self) -> None:
        output_dir = self._tmp_dir()
        prompt_file = output_dir / "prompts.txt"
        prompt_file.write_text("first, comma prompt\nsecond prompt\n", encoding="utf-8")
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_real_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--prompt-texts-file", command)
            self.assertEqual(command[command.index("--prompt-texts-file") + 1], str(prompt_file))
            self.assertNotIn("--prompt-texts", command)
            self.assertNotIn("first, comma prompt", command)
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": pack.REMOTE_REAL_SCHEMA,
                    "ok": True,
                    "mode": "remote-loopback",
                    "diagnosis_codes": [
                        "remote_real_llm_sharded_ready",
                        "remote_real_llm_sharded_loopback_ready",
                        "real_llm_sharded_ready",
                        "real_llm_artifact_ready",
                        "activation_transport_ready",
                        "baseline_match",
                        "decoded_tokens_match",
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                    ],
                    "payload_summaries": {
                        "remote_loopback_real_llm_sharded_inference": {
                            "session": {"stage_count": 2, "request_count": 2, "model_id": "sshleifer/tiny-gpt2"},
                            "stage_assignment": {
                                "stage0_miner_id": "miner-stage0",
                                "stage1_miner_id": "miner-stage1",
                                "distinct_stage_miners": True,
                                "stage_assignment_valid": True,
                            },
                        },
                    },
                    "safety": {"cpu_only_default": True, "activation_payloads_redacted": True},
                }) + "\n",
                stderr="",
            )

        report = pack.build_report(pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--prompt-texts-file",
            str(prompt_file),
        ]), runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts-file")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertNotIn("first, comma prompt", encoded)
        self.assertTrue(calls)

    def test_verify_wraps_prompt_texts_file(self) -> None:
        output_dir = self._tmp_dir()
        prompt_file = output_dir / "prompts.txt"
        prompt_file.write_text("first, comma prompt\nsecond prompt\n", encoding="utf-8")
        args = pack.parse_args([
            "verify",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--prompt-texts-file",
            str(prompt_file),
        ])

        command = pack.swarm_common_command(args, output_dir=output_dir)

        self.assertIn("--prompt-texts-file", command)
        self.assertEqual(command[command.index("--prompt-texts-file") + 1], str(prompt_file))
        self.assertNotIn("--prompt-texts", command)
        self.assertNotIn("first, comma prompt", command)

    def test_prompt_texts_file_rejects_inline_batch(self) -> None:
        prompt_file = self._tmp_dir() / "prompts.txt"
        prompt_file.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            pack.parse_args([
                "local-loopback",
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])
        self.assertEqual(
            str(raised.exception),
            "public_swarm_inference_beta accepts either --prompt-texts or --prompt-texts-file, not both",
        )


if __name__ == "__main__":
    unittest.main()
