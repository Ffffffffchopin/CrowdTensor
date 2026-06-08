from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import usable_swarm_inference_check as check
from scripts import usable_swarm_inference_pack as pack


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class UsableSwarmInferencePackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_usable_swarm_test_"))

    def test_local_mode_requires_eight_token_p2p_real_generate_and_rescue(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("p2p_swarm_inference_v06_pack.py", command[1])
            self.assertEqual(command[2], "local-smoke")
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "8")
            return completed(check.fake_p2p_v06_payload())

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
        ]), runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["usable_swarm"]["ready"])
        p2p = report["readiness"]["p2p_product_path"]
        self.assertTrue(p2p["route_ready"])
        self.assertEqual(p2p["route_source"], "p2p-discovery")
        self.assertTrue(p2p["real_generate_ready"])
        self.assertTrue(p2p["kv_cache_ready"])
        self.assertTrue(p2p["real_stage_rescue_ready"])
        self.assertTrue(p2p["distinct_stage_miners"])
        self.assertEqual(p2p["generated_token_count"], 8)
        self.assertEqual(p2p["accepted_rows"], 16)
        self.assertEqual(p2p["usable_evidence_source"], "source_gate")
        self.assertTrue(p2p["model"]["compatible"])
        self.assertEqual(p2p["model"]["observed_hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertIn("usable_swarm_inference_ready", report["diagnosis_codes"])
        self.assertIn("usable_swarm_model_match_ready", report["diagnosis_codes"])
        self.assertIn("usable_real_llm_kv_cache_ready", report["diagnosis_codes"])
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
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
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        self.assertTrue((output_dir / "usable_swarm_inference.json").is_file())
        self.assertTrue((output_dir / "usable_swarm_inference.md").is_file())
        self.assertTrue((output_dir / "support_bundle.json").is_file())
        self.assertTrue((output_dir / "USABLE_SWARM_INFERENCE.md").is_file())
        self.assertTrue(calls)
        markdown = (output_dir / "usable_swarm_inference.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- output request note:", markdown)
        self.assertIn("local answer", markdown)
        self.assertIn(
            "- prompt scope: `source=prompt-text count=1 inline_prompt_text=True terminal_next_commands_local_private=True saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn("- prompt scope note:", markdown)
        self.assertIn("raw prompt text", markdown)
        self.assertIn("excluded from public JSON, Markdown", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not a local answer transcript", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        support = json.loads((output_dir / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["prompt_scope"], report["prompt_scope"])
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertNotIn('"generated_text":', encoded)
        self.assertNotIn('"generated_token_ids":', encoded)

    def test_check_validation_requires_top_level_p2p_route_source(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("p2p_swarm_inference_v06_pack.py", command[1])
            return completed(check.fake_p2p_v06_payload())

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
        ]), runner=fake_runner)
        report["readiness"]["p2p_product_path"]["route_source"] = "coordinator"

        errors = check.validate_report(
            report,
            mode=pack.MODE_LOCAL,
            expected_model_id=pack.DEFAULT_HF_MODEL_ID,
            required_tokens=8,
        )

        self.assertIn("usable_route_source_not_p2p_discovery", errors)
        self.assertNotIn("p2p_route_ready_missing", errors)

    def test_local_mode_forwards_and_preserves_bounded_batch_evidence(self) -> None:
        output_dir = self._tmp_dir()
        payload = check.fake_p2p_v06_payload()
        batch = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 2,
            "max_request_count": 4,
            "prompt_hashes": ["sha256:p1", "sha256:p2"],
            "prompt_char_counts": [12, 13],
            "result_count": 2,
            "results": [
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generated_token_count": 8,
                    "generated_text_hash": "sha256:g1",
                    "multi_token_generation_ready": True,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
                {
                    "request_id": "req-2",
                    "prompt_hash": "sha256:p2",
                    "generated_token_count": 8,
                    "generated_text_hash": "sha256:g2",
                    "multi_token_generation_ready": True,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
            ],
            "batch_generation_ready": True,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        payload["payload_summaries"]["local_p2p_discovery"]["real_generate_probe"]["batch"] = batch
        payload["diagnosis_codes"].extend(["p2p_real_generate_batch_ready", "public_swarm_generate_batch_ready"])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertNotIn("--prompt-text", command)
            return completed(payload)

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
            "--max-new-tokens",
            "8",
        ]), runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        p2p = report["readiness"]["p2p_product_path"]
        self.assertTrue(p2p["batch_ready"])
        self.assertTrue(p2p["batch"]["batch_generation_ready"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertIn("usable_real_llm_batch_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("first prompt", encoded)
        self.assertNotIn("second prompt", encoded)

    def test_local_mode_forwards_prompt_texts_file_to_p2p_path(self) -> None:
        output_dir = self._tmp_dir()
        prompt_file = output_dir / "prompts.txt"
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        payload = check.fake_p2p_v06_payload()
        batch = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 2,
            "max_request_count": 4,
            "prompt_hashes": ["sha256:p1", "sha256:p2"],
            "prompt_char_counts": [24, 13],
            "result_count": 2,
            "results": [
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generated_token_count": 8,
                    "generated_text_hash": "sha256:g1",
                    "multi_token_generation_ready": True,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
                {
                    "request_id": "req-2",
                    "prompt_hash": "sha256:p2",
                    "generated_token_count": 8,
                    "generated_text_hash": "sha256:g2",
                    "multi_token_generation_ready": True,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
            ],
            "batch_generation_ready": True,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        payload["payload_summaries"]["local_p2p_discovery"]["real_generate_probe"]["batch"] = batch
        payload["diagnosis_codes"].extend(["p2p_real_generate_batch_ready", "public_swarm_generate_batch_ready"])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--prompt-texts-file", command)
            self.assertEqual(command[command.index("--prompt-texts-file") + 1], str(prompt_file))
            self.assertNotIn("--prompt-texts", command)
            self.assertNotIn("--prompt-text", command)
            return completed(payload)

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir),
            "--prompt-texts-file",
            str(prompt_file),
            "--max-new-tokens",
            "8",
        ]), runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["readiness"]["p2p_product_path"]["batch_ready"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts-file")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        for prompt in prompts:
            self.assertNotIn(prompt, encoded)

    def test_local_mode_does_not_accept_batch_ready_code_without_batch_evidence(self) -> None:
        output_dir = self._tmp_dir()
        payload = check.fake_p2p_v06_payload()
        payload["payload_summaries"]["local_p2p_discovery"]["real_generate_probe"]["batch"] = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 1,
            "result_count": 1,
            "batch_generation_ready": False,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        payload["diagnosis_codes"].extend(["p2p_real_generate_batch_ready", "public_swarm_generate_batch_ready"])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--prompt-texts", command)
            return completed(payload)

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
            "--max-new-tokens",
            "8",
        ]), runner=fake_runner)

        self.assertTrue(report["ok"], report)
        p2p = report["readiness"]["p2p_product_path"]
        self.assertFalse(p2p["batch_ready"])
        self.assertFalse(p2p["batch"]["batch_generation_ready"])
        self.assertNotIn("usable_real_llm_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("p2p_real_generate_batch_ready", p2p["diagnosis_codes"])

    def test_local_mode_does_not_accept_batch_with_duplicate_request_identity(self) -> None:
        output_dir = self._tmp_dir()
        payload = check.fake_p2p_v06_payload()
        payload["payload_summaries"]["local_p2p_discovery"]["real_generate_probe"]["batch"] = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 2,
            "result_count": 2,
            "results": [
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:g1",
                    "multi_token_generation_ready": True,
                },
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:g1-dup",
                    "multi_token_generation_ready": True,
                },
            ],
            "batch_generation_ready": True,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        payload["diagnosis_codes"].extend(["p2p_real_generate_batch_ready", "public_swarm_generate_batch_ready"])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--prompt-texts", command)
            return completed(payload)

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
            "--max-new-tokens",
            "8",
        ]), runner=fake_runner)

        p2p = report["readiness"]["p2p_product_path"]
        self.assertTrue(report["ok"], report)
        self.assertFalse(p2p["batch_ready"])
        self.assertFalse(p2p["batch"]["batch_identity_ready"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_local_mode_forwards_and_preserves_stream_evidence(self) -> None:
        output_dir = self._tmp_dir()
        payload = check.fake_p2p_v06_payload()
        stream = {
            "enabled": True,
            "requested": True,
            "event_count": 8,
            "source": "admin-session-stream",
            "endpoint_ready": True,
            "stream_generation_ready": True,
            "progress": {
                "stream_progress_complete": True,
                "all_token_events_ready": True,
                "monotonic_progress": True,
                "observed_token_counts": list(range(1, 9)),
                "max_observed_token_count": 8,
                "max_new_tokens": 8,
                "source": "admin-session-stream",
            },
            "events": [],
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        payload["payload_summaries"]["local_p2p_discovery"]["real_generate_probe"]["stream"] = stream
        payload["diagnosis_codes"].extend([
            "p2p_real_generate_stream_ready",
            "public_swarm_generate_stream_ready",
            "public_swarm_generate_stream_endpoint_ready",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--stream-generation", command)
            return completed(payload)

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
            "--max-new-tokens",
            "8",
        ]), runner=fake_runner)

        self.assertTrue(report["ok"], report)
        p2p = report["readiness"]["p2p_product_path"]
        self.assertTrue(p2p["stream_ready"])
        self.assertTrue(p2p["stream"]["stream_generation_ready"])
        self.assertIn("usable_real_llm_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        self.assertTrue(calls)

    def test_evidence_import_tolerates_redacted_stream_progress_counts(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "source" / "p2p_v06.json"
        payload = check.fake_p2p_v06_payload(generated_tokens=8, accepted_rows=16)
        stream = {
            "enabled": False,
            "requested": False,
            "event_count": 0,
            "source": "disabled",
            "endpoint_ready": False,
            "stream_generation_ready": False,
            "progress": {
                "stream_progress_complete": False,
                "all_token_events_ready": "<redacted>",
                "monotonic_progress": False,
                "observed_token_counts": "<redacted>",
                "max_observed_token_count": "<redacted>",
                "max_new_tokens": 8,
                "source": "disabled",
            },
            "events": [],
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        payload["p2p"]["stream"] = stream
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "usable"),
            "--p2p-report",
            str(source),
            "--max-new-tokens",
            "8",
        ]))

        p2p = report["readiness"]["p2p_product_path"]
        self.assertTrue(report["ok"], report)
        self.assertFalse(p2p["stream"]["stream_generation_ready"])
        self.assertEqual(p2p["stream"]["event_count"], 0)
        self.assertEqual(p2p["stream"]["progress"]["source"], "disabled")

    def test_evidence_import_rejects_two_token_p2p_report_for_eight_token_goal(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "source" / "p2p_v06.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(check.fake_p2p_v06_payload(generated_tokens=2, accepted_rows=4)) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "usable"),
            "--p2p-report",
            str(source),
            "--max-new-tokens",
            "8",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertIn("usable_swarm_inference_blocked", report["diagnosis_codes"])
        self.assertIn("real small HF multi-token generation", report["not_completed"])
        self.assertFalse(report["readiness"]["p2p_product_path"]["generation_target_ready"])

    def test_evidence_import_rejects_missing_kv_cache_for_product_path(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "source" / "p2p_v06.json"
        payload = check.fake_p2p_v06_payload(generated_tokens=8, accepted_rows=16)
        payload["p2p"].pop("kv_cache", None)
        real_probe = payload["payload_summaries"]["local_p2p_discovery"]["real_generate_probe"]
        real_probe.pop("kv_cache", None)
        real_probe["diagnosis_codes"] = [
            code for code in real_probe.get("diagnosis_codes", [])
            if code != "p2p_real_generate_kv_cache_ready"
        ]
        payload["diagnosis_codes"] = [
            code for code in payload.get("diagnosis_codes", [])
            if code != "p2p_real_generate_kv_cache_ready"
        ]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "usable"),
            "--p2p-report",
            str(source),
            "--max-new-tokens",
            "8",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertIn("usable_swarm_inference_blocked", report["diagnosis_codes"])
        self.assertIn("real LLM persistent dual-stage KV cache reuse", report["not_completed"])
        self.assertFalse(report["readiness"]["p2p_product_path"]["kv_cache_ready"])

    def test_evidence_import_blocks_non_default_model_without_matching_report(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "source" / "p2p_v06.json"
        payload = check.fake_p2p_v06_payload(generated_tokens=8, accepted_rows=16)
        payload["hf_model_id"] = pack.DEFAULT_HF_MODEL_ID
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "usable"),
            "--p2p-report",
            str(source),
            "--hf-model-id",
            "distilgpt2",
            "--max-new-tokens",
            "8",
        ]))

        p2p = report["readiness"]["p2p_product_path"]
        self.assertFalse(report["ok"], report)
        self.assertFalse(p2p["model"]["compatible"])
        self.assertEqual(p2p["model"]["expected_hf_model_id"], "distilgpt2")
        self.assertEqual(p2p["model"]["observed_hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertIn("usable_swarm_model_mismatch", report["diagnosis_codes"])
        self.assertIn("imported P2P evidence model match", report["not_completed"])

    def test_evidence_import_blocks_default_model_without_observed_model_metadata(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "source" / "p2p_v06.json"
        payload = check.fake_p2p_v06_payload(generated_tokens=8, accepted_rows=16)
        payload.pop("hf_model_id", None)
        payload["p2p"].pop("hf_model_id", None)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "usable"),
            "--p2p-report",
            str(source),
            "--max-new-tokens",
            "8",
        ]))

        p2p = report["readiness"]["p2p_product_path"]
        self.assertFalse(report["ok"], report)
        self.assertFalse(p2p["model"]["model_id_present"])
        self.assertFalse(p2p["model"]["model_id_match"])
        self.assertFalse(p2p["model"]["compatible"])
        self.assertFalse(p2p["model"]["default_model_retained_evidence"])
        self.assertIn("usable_swarm_model_mismatch", report["diagnosis_codes"])
        self.assertIn("usable_swarm_inference_blocked", report["diagnosis_codes"])
        self.assertIn("imported P2P evidence model match", report["not_completed"])

    def test_package_mode_writes_user_runbook(self) -> None:
        output_dir = self._tmp_dir()

        private_prompt = "private package prompt should not be saved"
        report = pack.build_report(pack.parse_args([
            "package",
            "--output-dir",
            str(output_dir),
            "--prompt-text",
            private_prompt,
            "--max-new-tokens",
            "8",
        ]))

        self.assertTrue(report["ok"], report)
        runbook = output_dir / "USABLE_SWARM_INFERENCE.md"
        self.assertTrue(runbook.is_file())
        text = runbook.read_text(encoding="utf-8")
        self.assertIn("crowdtensor p2pd", text)
        self.assertIn("crowdtensor serve --p2p", text)
        self.assertIn("crowdtensor join --p2p", text)
        self.assertIn("crowdtensor generate --p2p", text)
        self.assertIn("--max-new-tokens 8", text)
        self.assertIn("Two-Machine/Public Rehearsal", text)
        self.assertIn("--i-understand-public-bind", text)
        self.assertIn("COORDINATOR_PUBLIC_HOST", text)
        self.assertIn("CROWDTENSOR_PROMPT_TEXT", text)
        self.assertNotIn(private_prompt, text)
        self.assertFalse(report["output_request"]["include_output"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")

    def test_check_script_validates_local_contract(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], check.SCHEMA)
        self.assertIn("usable_swarm_inference_check_ready", result["diagnosis_codes"])

    def test_check_script_validates_package_contract(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], check.SCHEMA)
        self.assertIn("usable_swarm_inference_check_ready", result["diagnosis_codes"])
        report = json.loads((output_dir / "usable" / "usable_swarm_inference.json").read_text(encoding="utf-8"))
        self.assertTrue(report["usable_swarm"]["package_only"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")

    def test_check_script_validates_non_default_model_local_contract(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args([
            "--mode",
            "local",
            "--hf-model-id",
            "distilgpt2",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["hf_model_id"], "distilgpt2")
        report = json.loads((output_dir / "usable" / "usable_swarm_inference.json").read_text(encoding="utf-8"))
        p2p = report["readiness"]["p2p_product_path"]
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["usable_swarm"]["hf_model_id"], "distilgpt2")
        self.assertEqual(p2p["model"]["expected_hf_model_id"], "distilgpt2")
        self.assertEqual(p2p["model"]["observed_hf_model_id"], "distilgpt2")
        self.assertTrue(p2p["model"]["compatible"])
        self.assertIn("usable_swarm_model_match_ready", report["diagnosis_codes"])
        self.assertNotIn("usable_swarm_model_mismatch", report["diagnosis_codes"])

    def test_check_script_real_local_uses_real_runner_and_forwards_options(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def real_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("p2p_swarm_inference_v06_pack.py", command[1])
            self.assertEqual(command[2], "local-smoke")
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "4")
            self.assertEqual(command[command.index("--p2p-port") + 1], "18788")
            self.assertEqual(command[command.index("--coordinator-port") + 1], "18789")
            self.assertIn("--stream-generation", command)
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertNotIn("--prompt-text", command)
            payload = check.fake_p2p_v06_payload(generated_tokens=4, accepted_rows=8)
            payload["payload_summaries"]["local_p2p_discovery"]["real_generate_probe"]["batch"] = {
                "enabled": True,
                "request_count": 2,
                "expected_request_count": 2,
                "observed_request_count": 2,
                "result_count": 2,
                "results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 4,
                        "max_new_tokens": 4,
                        "generated_text_hash": "sha256:g1",
                        "multi_token_generation_ready": True,
                    },
                    {
                        "request_id": "req-2",
                        "prompt_hash": "sha256:p2",
                        "generated_token_count": 4,
                        "max_new_tokens": 4,
                        "generated_text_hash": "sha256:g2",
                        "multi_token_generation_ready": True,
                    },
                ],
                "batch_generation_ready": True,
            }
            payload["payload_summaries"]["local_p2p_discovery"]["real_generate_probe"]["stream"] = {
                "enabled": True,
                "requested": True,
                "event_count": 8,
                "source": "admin-session-stream",
                "endpoint_ready": True,
                "stream_generation_ready": True,
                "progress": {
                    "stream_progress_complete": True,
                    "all_token_events_ready": True,
                    "monotonic_progress": True,
                    "expected_request_count": 2,
                    "per_request_progress": [
                        {
                            "request_key": "req-1",
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "event_count": 4,
                            "observed_token_counts": [1, 2, 3, 4],
                            "max_observed_token_count": 4,
                            "target_token_count": 4,
                            "monotonic_progress": True,
                            "stream_progress_complete": True,
                        },
                        {
                            "request_key": "req-2",
                            "request_id": "req-2",
                            "prompt_hash": "sha256:p2",
                            "event_count": 4,
                            "observed_token_counts": [1, 2, 3, 4],
                            "max_observed_token_count": 4,
                            "target_token_count": 4,
                            "monotonic_progress": True,
                            "stream_progress_complete": True,
                        },
                    ],
                    "per_request_progress_complete": True,
                    "per_request_monotonic_progress": True,
                    "observed_token_counts": [1, 2, 3, 4],
                    "max_observed_token_count": 4,
                    "max_new_tokens": 4,
                    "source": "admin-session-stream",
                },
                "events": [],
            }
            payload["diagnosis_codes"].extend([
                "p2p_real_generate_batch_ready",
                "p2p_real_generate_stream_ready",
                "public_swarm_generate_batch_ready",
                "public_swarm_generate_stream_ready",
                "public_swarm_generate_stream_endpoint_ready",
            ])
            return completed(payload)

        result = check.run_check(check.parse_args([
            "--mode",
            "local",
            "--real-local",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "4",
            "--p2p-port",
            "18788",
            "--coordinator-port",
            "18789",
            "--prompt-texts",
            "first prompt,second prompt",
            "--stream-generation",
        ]), runner=real_runner)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["real_local"])
        self.assertEqual(result["max_new_tokens"], 4)
        self.assertTrue(calls)
        report = json.loads((output_dir / "usable" / "usable_swarm_inference.json").read_text(encoding="utf-8"))
        p2p = report["readiness"]["p2p_product_path"]
        self.assertTrue(p2p["batch_ready"])
        self.assertTrue(p2p["stream_ready"])
        self.assertNotIn("first prompt", json.dumps(report, sort_keys=True))
        self.assertNotIn("second prompt", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
