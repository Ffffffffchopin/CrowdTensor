from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from crowdtensor import cli
from scripts import public_swarm_inference_v2_check as check
from scripts import public_swarm_inference_v2_pack as pack


def safe_batch(tokens: int = 16) -> dict[str, Any]:
    return {
        "enabled": True,
        "request_count": 2,
        "expected_request_count": 2,
        "observed_request_count": 2,
        "prompt_hashes": ["sha256:p1", "sha256:p2"],
        "prompt_char_counts": [12, 13],
        "result_count": 2,
        "results": [
            {
                "request_id": "req-1",
                "prompt_hash": "sha256:p1",
                "generated_token_count": tokens,
                "max_new_tokens": tokens,
                "generated_text_hash": "sha256:g1",
                "decoded_tokens_match": True,
                "multi_token_generation_ready": True,
            },
            {
                "request_id": "req-2",
                "prompt_hash": "sha256:p2",
                "generated_token_count": tokens,
                "max_new_tokens": tokens,
                "generated_text_hash": "sha256:g2",
                "decoded_tokens_match": True,
                "multi_token_generation_ready": True,
            },
        ],
        "batch_identity_ready": True,
        "batch_generation_ready": True,
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


class PublicSwarmInferenceV2PackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_v2_test_"))

    def _write_sources(self, output_dir: Path, *, external_tokens: int = 16) -> dict[str, Path]:
        source_dir = output_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "usable": source_dir / "usable.json",
            "preview": source_dir / "preview.json",
            "real_p2p": source_dir / "real_p2p.json",
            "gpu": source_dir / "gpu.json",
        }
        paths["usable"].write_text(json.dumps(check.fake_usable_report(16)) + "\n", encoding="utf-8")
        paths["preview"].write_text(json.dumps(check.fake_preview_report(16)) + "\n", encoding="utf-8")
        paths["real_p2p"].write_text(json.dumps(check.fake_real_p2p_report(external_tokens)) + "\n", encoding="utf-8")
        paths["gpu"].write_text(json.dumps(check.fake_gpu_report(16)) + "\n", encoding="utf-8")
        return paths

    def _batch_stream(self, tokens: int = 16) -> dict[str, object]:
        return {
            "enabled": True,
            "requested": True,
            "event_count": tokens * 2,
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
                        "event_count": tokens,
                        "observed_token_counts": list(range(1, tokens + 1)),
                        "max_observed_token_count": tokens,
                        "target_token_count": tokens,
                        "monotonic_progress": True,
                        "stream_progress_complete": True,
                    },
                    {
                        "request_key": "req-2",
                        "request_id": "req-2",
                        "prompt_hash": "sha256:p2",
                        "event_count": tokens,
                        "observed_token_counts": list(range(1, tokens + 1)),
                        "max_observed_token_count": tokens,
                        "target_token_count": tokens,
                        "monotonic_progress": True,
                        "stream_progress_complete": True,
                    },
                ],
                "per_request_progress_complete": True,
                "per_request_monotonic_progress": True,
                "observed_token_counts": list(range(1, tokens + 1)),
                "max_observed_token_count": tokens,
                "max_new_tokens": tokens,
                "source": "admin-session-stream",
            },
            "events": [],
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }

    def _local_runner(
        self,
        calls: list[list[str]],
        *,
        usable_payload: dict[str, object] | None = None,
        real_p2p_payload: dict[str, object] | None = None,
    ):
        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            tokens = int(command[command.index("--max-new-tokens") + 1]) if "--max-new-tokens" in command else 16
            joined = " ".join(command)
            if "usable_swarm_inference_pack.py" in joined:
                payload = usable_payload or check.fake_usable_report(tokens)
                if "--output-dir" in command:
                    child_dir = Path(command[command.index("--output-dir") + 1])
                    child_dir.mkdir(parents=True, exist_ok=True)
                    (child_dir / "usable_swarm_inference.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(payload) + "\n", stderr="")
            if "real_p2p_swarm_inference_core_rc_pack.py" in joined:
                payload = real_p2p_payload or check.fake_real_p2p_report(tokens)
                payload = dict(payload)
                payload["mode"] = "local-smoke"
                payload["external"] = {"external_runtime_verified": False}
                if "--output-dir" in command:
                    child_dir = Path(command[command.index("--output-dir") + 1])
                    child_dir.mkdir(parents=True, exist_ok=True)
                    (child_dir / "real_p2p_swarm_inference_core_rc.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(payload) + "\n", stderr="")
            raise AssertionError(command)

        return fake_runner

    def test_fresh_external_short_report_does_not_satisfy_16_token_v2_gate(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=2)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        external = report["readiness"]["external_validation"]
        self.assertFalse(report["ok"], report)
        self.assertFalse(external["ready"])
        self.assertFalse(external["generation_target_ready"])
        self.assertFalse(external["fresh_external_runtime_verified"])
        self.assertIn("external signed/real P2P validation at token target", report["not_completed"])
        self.assertIn("public_swarm_v2_external_fresh_run_action_required", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_v2_fresh_external_runtime_verified", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_inference_v2_ready", report["diagnosis_codes"])

    def test_fresh_external_16_token_report_satisfies_v2_gate(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        external = report["readiness"]["external_validation"]
        self.assertTrue(report["ok"], report)
        self.assertTrue(external["ready"])
        self.assertTrue(external["model"]["compatible"])
        self.assertTrue(external["model"]["model_id_present"])
        self.assertTrue(report["readiness"]["local_p2p_generate"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["p2p_route_hardening"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["p2p_route_hardening"]["model"]["model_id_present"])
        self.assertTrue(external["generation_target_ready"])
        self.assertEqual(external["accepted_rows"], 32)
        self.assertTrue(external["accepted_rows_ready"])
        self.assertTrue(external["fresh_external_runtime_verified"])
        self.assertIn("public_swarm_inference_v2_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_model_match_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_fresh_external_runtime_verified", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_external_stage_rows_ready", report["diagnosis_codes"])
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["user_status"]["next_step"], "review_artifacts")
        self.assertEqual(report["review_summary"]["state"], "ready")
        self.assertEqual(report["review_summary"]["next_step"], "review_artifacts")
        self.assertEqual(report["review_summary"]["recommended_label"], "review v2 evidence")
        self.assertIn("public_swarm_inference_v2.md", report["review_summary"]["inspect_first"])
        self.assertEqual(report["recommended_next_command"]["reason"], "v2_ready")
        self.assertIn("public_swarm_inference_v2.md", report["recommended_next_command"]["command_line"])
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
        markdown = (output_dir / "v2" / "public_swarm_inference_v2.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("## Review", markdown)
        self.assertIn("- state: `ready`", markdown)
        self.assertIn("- recommended: `review v2 evidence` reason=`v2_ready`", markdown)
        self.assertIn(
            "- prompt scope: `source=prompt-text count=1 inline_prompt_text=True terminal_next_commands_local_private=True saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        support = json.loads((output_dir / "v2" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["prompt_scope"], report["prompt_scope"])
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertEqual(support["review_summary"]["state"], "ready")
        self.assertEqual(support["user_status"]["state"], "ready")
        self.assertEqual(support["recommended_next_command"]["reason"], "v2_ready")

    def test_fresh_external_blocks_when_stage_rows_below_token_target(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        real_p2p_payload = json.loads(sources["real_p2p"].read_text(encoding="utf-8"))
        real_p2p_payload["stage_assignment"] = {
            "completed_rows": 30,
            "distinct_stage_miners": True,
            "max_generation_step": 15,
            "stage0_miner_id": "external-stage0",
            "stage1_miner_id": "external-stage1",
        }
        real_p2p_payload["ledger"] = {"accepted_rows": 30, "error": ""}
        sources["real_p2p"].write_text(json.dumps(real_p2p_payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        external = report["readiness"]["external_validation"]
        self.assertFalse(report["ok"], report)
        self.assertEqual(external["accepted_rows"], 30)
        self.assertFalse(external["accepted_rows_ready"])
        self.assertIn("external signed/real P2P accepted stage rows", report["not_completed"])
        self.assertEqual(report["user_status"]["state"], "blocked")
        self.assertEqual(report["user_status"]["next_step"], "fix_blockers")
        self.assertEqual(report["review_summary"]["state"], "blocked")
        self.assertEqual(report["review_summary"]["recommended_label"], "import fresh external evidence")
        self.assertEqual(report["recommended_next_command"]["reason"], "refresh_external_p2p_evidence")
        self.assertIn("dist/<fresh-real-p2p-run>/real_p2p_swarm_inference_core_rc.json", report["recommended_next_command"]["command_line"])
        self.assertNotIn("public_swarm_v2_external_stage_rows_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_v2_fresh_external_runtime_verified", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_inference_v2_ready", report["diagnosis_codes"])

    def test_default_model_still_requires_explicit_external_model_metadata(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        real_p2p_payload = json.loads(sources["real_p2p"].read_text(encoding="utf-8"))
        real_p2p_payload.pop("hf_model_id", None)
        sources["real_p2p"].write_text(json.dumps(real_p2p_payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2-missing-default-model"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        readiness = report["readiness"]
        self.assertFalse(report["ok"], report)
        self.assertFalse(readiness["external_validation"]["model"]["compatible"])
        self.assertFalse(readiness["external_validation"]["model"]["model_id_present"])
        self.assertFalse(readiness["p2p_route_hardening"]["model"]["compatible"])
        self.assertFalse(readiness["p2p_route_hardening"]["model"]["model_id_present"])
        self.assertIn("external signed/real P2P evidence model match", report["not_completed"])
        self.assertIn("signed or real P2P evidence model match", report["not_completed"])
        self.assertIn("public_swarm_v2_external_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_p2p_model_mismatch", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_v2_model_match_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_inference_v2_ready", report["diagnosis_codes"])

    def test_import_wrapper_expected_model_does_not_mask_missing_observed_model(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        real_p2p_payload = json.loads(sources["real_p2p"].read_text(encoding="utf-8"))
        real_p2p_payload.pop("hf_model_id", None)
        real_p2p_payload["mode"] = "evidence-import"
        real_p2p_payload["hf_model_id"] = pack.DEFAULT_HF_MODEL_ID
        real_p2p_payload["expected_hf_model_id"] = pack.DEFAULT_HF_MODEL_ID
        real_p2p_payload["imported"] = {
            "schema": pack.REAL_P2P_SCHEMA,
            "ok": True,
            "mode": "kaggle-auto",
            "hf_model_id": None,
        }
        sources["real_p2p"].write_text(json.dumps(real_p2p_payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2-wrapper-missing-observed-model"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        readiness = report["readiness"]
        self.assertFalse(report["ok"], report)
        self.assertFalse(readiness["external_validation"]["model"]["compatible"])
        self.assertEqual(readiness["external_validation"]["model"]["observed_hf_model_id"], "")
        self.assertFalse(readiness["p2p_route_hardening"]["model"]["compatible"])
        self.assertEqual(readiness["p2p_route_hardening"]["model"]["observed_hf_model_id"], "")
        self.assertIn("public_swarm_v2_external_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_p2p_model_mismatch", report["diagnosis_codes"])

    def test_local_mode_forwards_and_preserves_usable_batch_evidence(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        usable_payload = check.fake_usable_report(16)
        usable_payload["readiness"]["p2p_product_path"]["batch_ready"] = True
        usable_payload["readiness"]["p2p_product_path"]["batch"] = safe_batch(16)
        usable_payload["diagnosis_codes"].extend(["usable_real_llm_batch_ready", "public_swarm_generate_batch_ready"])
        calls: list[list[str]] = []

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir / "v2-local"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--prompt-texts",
            "first prompt,second prompt",
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]), runner=self._local_runner(calls, usable_payload=usable_payload))
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(len(calls), 2)
        usable_command = next(command for command in calls if "usable_swarm_inference_pack.py" in " ".join(command))
        real_p2p_command = next(command for command in calls if "real_p2p_swarm_inference_core_rc_pack.py" in " ".join(command))
        self.assertIn("--prompt-texts", usable_command)
        self.assertEqual(usable_command[usable_command.index("--prompt-texts") + 1], "first prompt,second prompt")
        self.assertNotIn("--prompt-text", usable_command)
        self.assertNotIn("--prompt-texts", real_p2p_command)
        local = report["readiness"]["local_p2p_generate"]
        self.assertTrue(local["batch_ready"])
        self.assertEqual(local["route_source"], "p2p-discovery")
        self.assertTrue(local["route_source_consistent"])
        self.assertTrue(local["batch"]["batch_generation_ready"])
        self.assertTrue(report["readiness"]["real_p2p_local_route_hardening"]["ready"])
        self.assertIn("real-p2p-local", report["artifacts"]["real_p2p"]["path"])
        self.assertIn("public_swarm_v2_real_p2p_local_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_batch_generation_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("first prompt", encoded)
        self.assertNotIn("second prompt", encoded)

    def test_local_mode_forwards_prompt_texts_file_to_usable_path(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        prompt_file = output_dir / "prompts.txt"
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        usable_payload = check.fake_usable_report(16)
        usable_payload["readiness"]["p2p_product_path"]["batch_ready"] = True
        usable_payload["readiness"]["p2p_product_path"]["batch"] = safe_batch(16)
        usable_payload["diagnosis_codes"].extend(["usable_real_llm_batch_ready", "public_swarm_generate_batch_ready"])
        calls: list[list[str]] = []

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir / "v2-local-file"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--prompt-texts-file",
            str(prompt_file),
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]), runner=self._local_runner(calls, usable_payload=usable_payload))

        self.assertTrue(report["ok"], report)
        usable_command = next(command for command in calls if "usable_swarm_inference_pack.py" in " ".join(command))
        self.assertIn("--prompt-texts-file", usable_command)
        self.assertEqual(usable_command[usable_command.index("--prompt-texts-file") + 1], str(prompt_file))
        self.assertNotIn("--prompt-texts", usable_command)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts-file")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(str(prompt_file), encoded)
        for prompt in prompts:
            self.assertNotIn(prompt, encoded)
        runbook = (output_dir / "v2-local-file" / "PUBLIC_SWARM_INFERENCE_V2.md").read_text(encoding="utf-8")
        self.assertIn('--prompt "<prompt>"', runbook)
        self.assertNotIn(str(prompt_file), runbook)
        for prompt in prompts:
            self.assertNotIn(prompt, runbook)

    def test_runbook_and_recommended_command_do_not_leak_single_prompt_text(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        private_prompt = "private customer prompt must stay local"

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2-private-prompt"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--prompt-text",
            private_prompt,
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        encoded = json.dumps(report, sort_keys=True)
        runbook = (output_dir / "v2-private-prompt" / "PUBLIC_SWARM_INFERENCE_V2.md").read_text(encoding="utf-8")
        markdown = (output_dir / "v2-private-prompt" / "public_swarm_inference_v2.md").read_text(encoding="utf-8")
        support = (output_dir / "v2-private-prompt" / "support_bundle.json").read_text(encoding="utf-8")
        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_inference_v2.md", report["recommended_next_command"]["command_line"])
        self.assertIn('--prompt "<prompt>"', runbook)
        self.assertNotIn(private_prompt, encoded)
        self.assertNotIn(private_prompt, runbook)
        self.assertNotIn(private_prompt, markdown)
        self.assertNotIn(private_prompt, support)

    def test_blocked_recommended_command_uses_prompt_placeholder(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        private_prompt = "blocked private prompt"
        usable_payload = check.fake_usable_report(16)
        usable_payload["readiness"]["p2p_product_path"]["kv_cache_ready"] = False
        sources["usable"].write_text(json.dumps(usable_payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2-private-prompt-blocked"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--prompt-text",
            private_prompt,
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        encoded = json.dumps(report, sort_keys=True)
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["recommended_next_command"]["label"], "rerun local v2 gate")
        self.assertIn("--prompt-text '<prompt>'", report["recommended_next_command"]["command_line"])
        self.assertNotIn(private_prompt, encoded)

    def test_local_mode_rejects_usable_batch_ready_code_without_batch_evidence(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        usable_payload = check.fake_usable_report(16)
        usable_payload["readiness"]["p2p_product_path"]["batch_ready"] = False
        usable_payload["readiness"]["p2p_product_path"]["batch"] = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 1,
            "result_count": 1,
            "batch_generation_ready": False,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        usable_payload["diagnosis_codes"].extend(["usable_real_llm_batch_ready", "public_swarm_generate_batch_ready"])
        calls: list[list[str]] = []

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir / "v2-local-batch-code-only"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--prompt-texts",
            "first prompt,second prompt",
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]), runner=self._local_runner(calls, usable_payload=usable_payload))

        local = report["readiness"]["local_p2p_generate"]
        self.assertFalse(report["ok"], report)
        self.assertFalse(local["batch_ready"])
        self.assertFalse(local["batch"]["batch_generation_ready"])
        self.assertNotIn("public_swarm_v2_batch_generation_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertIn("local batch generation", report["not_completed"])

    def test_local_mode_rejects_inconsistent_usable_route_source(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        usable_payload = check.fake_usable_report(16)
        p2p = usable_payload["readiness"]["p2p_product_path"]
        p2p["route_source"] = "coordinator"
        p2p["route"]["route_source"] = "p2p-discovery"

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir / "v2-local-bad-route-source"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]), runner=self._local_runner([], usable_payload=usable_payload))

        local = report["readiness"]["local_p2p_generate"]
        self.assertFalse(report["ok"], report)
        self.assertFalse(local["ready"])
        self.assertEqual(local["route_source"], "coordinator")
        self.assertFalse(local["route_source_consistent"])
        self.assertNotIn("public_swarm_v2_local_p2p_generate_ready", report["diagnosis_codes"])

    def test_local_mode_rejects_duplicate_usable_batch_identity(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        usable_payload = check.fake_usable_report(16)
        batch = safe_batch(16)
        batch["results"][1]["request_id"] = "req-1"
        batch["results"][1]["prompt_hash"] = "sha256:p1"
        usable_payload["readiness"]["p2p_product_path"]["batch_ready"] = True
        usable_payload["readiness"]["p2p_product_path"]["batch"] = batch
        usable_payload["diagnosis_codes"].extend(["usable_real_llm_batch_ready", "public_swarm_generate_batch_ready"])

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir / "v2-local-batch-duplicate"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--prompt-texts",
            "first prompt,second prompt",
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]), runner=self._local_runner([], usable_payload=usable_payload))

        local = report["readiness"]["local_p2p_generate"]
        self.assertFalse(report["ok"], report)
        self.assertFalse(local["batch_ready"])
        self.assertFalse(local["batch"]["batch_identity_ready"])
        self.assertNotIn("public_swarm_v2_batch_generation_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_local_mode_forwards_and_requires_stream_evidence(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        usable_payload = check.fake_usable_report(16)
        usable_payload["readiness"]["p2p_product_path"]["stream_ready"] = True
        usable_payload["readiness"]["p2p_product_path"]["stream"] = self._batch_stream(16)
        usable_payload["diagnosis_codes"].extend([
            "usable_real_llm_stream_ready",
            "public_swarm_generate_stream_ready",
            "public_swarm_generate_stream_endpoint_ready",
        ])
        calls: list[list[str]] = []

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir / "v2-stream-local"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--stream-generation",
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]), runner=self._local_runner(calls, usable_payload=usable_payload))

        self.assertTrue(report["ok"], report)
        self.assertEqual(len(calls), 2)
        usable_command = next(command for command in calls if "usable_swarm_inference_pack.py" in " ".join(command))
        real_p2p_command = next(command for command in calls if "real_p2p_swarm_inference_core_rc_pack.py" in " ".join(command))
        self.assertIn("--stream-generation", usable_command)
        self.assertNotIn("--stream-generation", real_p2p_command)
        local = report["readiness"]["local_p2p_generate"]
        self.assertTrue(local["stream_ready"])
        self.assertTrue(local["stream"]["stream_generation_ready"])
        self.assertTrue(report["readiness"]["real_p2p_local_route_hardening"]["ready"])
        self.assertIn("public_swarm_v2_stream_generation_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        self.assertTrue(calls)

    def test_local_mode_fresh_runs_real_p2p_route_hardening(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        calls: list[list[str]] = []

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir / "v2-real-p2p-local"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--real-p2p-port",
            "19990",
            "--real-p2p-coordinator-port",
            "19991",
            "--real-p2p-libp2p-port",
            "0",
            "--real-p2p-discovery-backend",
            "http-provider-store",
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]), runner=self._local_runner(calls))

        self.assertTrue(report["ok"], report)
        self.assertEqual(len(calls), 2)
        real_p2p_command = next(command for command in calls if "real_p2p_swarm_inference_core_rc_pack.py" in " ".join(command))
        self.assertEqual(real_p2p_command[2], "local-smoke")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--p2p-port") + 1], "19990")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--coordinator-port") + 1], "19991")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--max-new-tokens") + 1], "16")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--discovery-backend") + 1], "http-provider-store")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--failure-mode") + 1], "kill-stage1-after-claim")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--lease-seconds") + 1], "2")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--victim-compute-seconds") + 1], "8")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--miner-timeout") + 1], "420.0")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--session-queue-timeout") + 1], "420.0")
        self.assertTrue(report["readiness"]["p2p_route_hardening"]["ready"])
        self.assertTrue(report["readiness"]["real_p2p_local_route_hardening"]["ready"])
        self.assertTrue(report["readiness"]["real_p2p_local_route_hardening"]["stage_requeue_ready"])
        self.assertEqual(report["readiness"]["real_p2p_local_route_hardening"]["mode"], "local-smoke")
        self.assertEqual(report["readiness"]["real_p2p_local_route_hardening"]["stage_requeue_target"], "stage1")
        self.assertIn("real-p2p-local", report["artifacts"]["real_p2p"]["path"])
        self.assertTrue(report["artifacts"]["external_real_p2p"]["present"])
        self.assertIn("public_swarm_v2_real_p2p_local_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_real_p2p_local_requeue_ready", report["diagnosis_codes"])

    def test_local_mode_accepts_real_p2p_route_hardening_without_generation_ok(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        real_p2p_payload = check.fake_real_p2p_report(16)
        real_p2p_payload["ok"] = False
        real_p2p_payload["mode"] = "local-smoke"
        real_p2p_payload["external"] = {"external_runtime_verified": False}
        real_p2p_payload["generation"] = {}
        real_p2p_payload["p2p"].update({
            "provider_count": 5,
            "signed_provider_record_count": 5,
            "coordinator_provider_count": 1,
            "stage0_provider_count": 1,
            "stage1_provider_count": 2,
            "catalog_error": "",
        })
        real_p2p_payload["p2p"]["route"].update({
            "coordinator_url_present": True,
            "usable_now": True,
            "matched_capabilities": {
                "real_llm_sharded_stage0": "real-p2p-rc-stage0",
                "real_llm_sharded_stage1": "real-p2p-rc-local-stage1-rescue",
            },
        })
        route_codes = {
            "libp2p_or_real_p2p_discovery_ready",
            "libp2p_discovery_backend_ready",
            "real_p2p_provider_store_ready",
            "real_p2p_signed_provider_records_ready",
        }
        real_p2p_payload["diagnosis_codes"] = sorted((set(real_p2p_payload["diagnosis_codes"]) - route_codes) | {
            "real_p2p_generation_not_ready",
            "real_p2p_swarm_inference_core_rc_blocked",
        })

        report = pack.build_report(pack.parse_args([
            "local",
            "--output-dir",
            str(output_dir / "v2-real-p2p-local-no-generation-ok"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]), runner=self._local_runner([], real_p2p_payload=real_p2p_payload))

        route = report["readiness"]["p2p_route_hardening"]
        self.assertTrue(report["ok"], report)
        self.assertFalse(route["ok"])
        self.assertTrue(route["ready"])
        self.assertTrue(route["route_ready"])
        self.assertTrue(route["local_route_hardening_ready"])
        self.assertTrue(report["readiness"]["real_p2p_local_route_hardening"]["stage_requeue_ready"])
        self.assertIn("public_swarm_v2_real_p2p_local_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_real_p2p_local_requeue_ready", report["diagnosis_codes"])

    def test_local_model_variant_accepts_non_default_local_paths_without_external_claim(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        usable_payload = check.fake_usable_report(16)
        usable_payload["usable_swarm"]["hf_model_id"] = "distilgpt2"
        usable_payload["readiness"]["p2p_product_path"]["stream_ready"] = True
        usable_payload["readiness"]["p2p_product_path"]["stream"] = self._batch_stream(16)
        usable_payload["diagnosis_codes"].extend([
            "usable_real_llm_stream_ready",
            "public_swarm_generate_stream_ready",
            "public_swarm_generate_stream_endpoint_ready",
        ])
        calls: list[list[str]] = []

        def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            tokens = int(command[command.index("--max-new-tokens") + 1]) if "--max-new-tokens" in command else 16
            joined = " ".join(command)
            calls.append(command)
            if "usable_swarm_inference_pack.py" in joined:
                if "--output-dir" in command:
                    child_dir = Path(command[command.index("--output-dir") + 1])
                    child_dir.mkdir(parents=True, exist_ok=True)
                    (child_dir / "usable_swarm_inference.json").write_text(json.dumps(usable_payload) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(usable_payload) + "\n", stderr="")
            if "real_p2p_swarm_inference_core_rc_pack.py" in joined:
                payload = check.fake_real_p2p_report(tokens)
                payload["mode"] = "local-smoke"
                payload["hf_model_id"] = "distilgpt2"
                payload["expected_hf_model_id"] = "distilgpt2"
                payload["external"] = {"external_runtime_verified": False}
                if "--output-dir" in command:
                    child_dir = Path(command[command.index("--output-dir") + 1])
                    child_dir.mkdir(parents=True, exist_ok=True)
                    (child_dir / "real_p2p_swarm_inference_core_rc.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(payload) + "\n", stderr="")
            raise AssertionError(command)

        report = pack.build_report(pack.parse_args([
            "local-model-variant",
            "--output-dir",
            str(output_dir / "v2-local-distilgpt2"),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--hf-model-id",
            "distilgpt2",
            "--stream-generation",
            "--max-new-tokens",
            "16",
        ]), runner=runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "local-model-variant")
        self.assertTrue(report["public_swarm_v2"]["local_model_variant_only"])
        self.assertFalse(report["public_swarm_v2"]["external_validation_claimed"])
        self.assertTrue(report["readiness"]["local_p2p_generate"]["model"]["compatible"])
        self.assertFalse(report["readiness"]["external_validation"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["p2p_route_hardening"]["model"]["compatible"])
        self.assertIn("public_swarm_inference_v2_local_model_variant_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_local_model_variant_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_external_validation_not_claimed", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_inference_v2_ready", report["diagnosis_codes"])
        self.assertNotIn("external_runtime_verified", report["diagnosis_codes"])
        self.assertNotIn("external_stage_requeue_ready", report["diagnosis_codes"])
        self.assertNotIn("cuda_runtime_available", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_gpu_beta_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_v2_external_model_mismatch", report["diagnosis_codes"])
        self.assertNotIn("external signed/real P2P validation at token target", report["not_completed"])
        self.assertNotIn("external signed/real P2P evidence model match", report["not_completed"])
        usable_command = next(command for command in calls if "usable_swarm_inference_pack.py" in " ".join(command))
        real_p2p_command = next(command for command in calls if "real_p2p_swarm_inference_core_rc_pack.py" in " ".join(command))
        self.assertEqual(usable_command[usable_command.index("--hf-model-id") + 1], "distilgpt2")
        self.assertEqual(real_p2p_command[real_p2p_command.index("--hf-model-id") + 1], "distilgpt2")

    def test_check_script_validates_local_model_variant_contract(self) -> None:
        output_dir = self._tmp_dir()

        result = check.run_check(check.parse_args([
            "--mode",
            "local-model-variant",
            "--hf-model-id",
            "distilgpt2",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["mode"], "local-model-variant")
        self.assertFalse(result["real_local"])
        self.assertIn("public_swarm_inference_v2_check_ready", result["diagnosis_codes"])

    def test_check_script_validates_evidence_import_contract_without_local_requeue_code(self) -> None:
        output_dir = self._tmp_dir()

        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertFalse(result["real_local"])
        self.assertNotIn("missing_code:public_swarm_v2_real_p2p_local_requeue_ready", result["errors"])
        self.assertIn("public_swarm_inference_v2_check_ready", result["diagnosis_codes"])

    def test_check_script_real_local_uses_real_runner_and_forwards_options(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def real_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            tokens = int(command[command.index("--max-new-tokens") + 1])
            joined = " ".join(command)
            if "usable_swarm_inference_pack.py" in joined:
                self.assertEqual(tokens, 8)
                self.assertEqual(command[command.index("--p2p-port") + 1], "18888")
                self.assertEqual(command[command.index("--coordinator-port") + 1], "18889")
                self.assertEqual(command[command.index("--timeout-seconds") + 1], "90.0")
                self.assertIn("--stream-generation", command)
                self.assertIn("--prompt-texts", command)
                self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
                self.assertNotIn("--prompt-text", command)
                payload = check.fake_usable_report(tokens)
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "usable_swarm_inference.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(payload) + "\n", stderr="")
            if "real_p2p_swarm_inference_core_rc_pack.py" in joined:
                self.assertEqual(tokens, 8)
                self.assertEqual(command[command.index("--p2p-port") + 1], "18890")
                self.assertEqual(command[command.index("--coordinator-port") + 1], "18891")
                self.assertEqual(command[command.index("--libp2p-port") + 1], "0")
                self.assertEqual(command[command.index("--timeout-seconds") + 1], "90.0")
                payload = check.fake_real_p2p_report(tokens)
                payload["mode"] = "local-smoke"
                payload["external"] = {"external_runtime_verified": False}
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "real_p2p_swarm_inference_core_rc.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(payload) + "\n", stderr="")
            raise AssertionError(command)

        result = check.run_check(check.parse_args([
            "--mode",
            "local",
            "--real-local",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--p2p-port",
            "18888",
            "--coordinator-port",
            "18889",
            "--real-p2p-port",
            "18890",
            "--real-p2p-coordinator-port",
            "18891",
            "--real-p2p-libp2p-port",
            "0",
            "--timeout-seconds",
            "90",
            "--stream-generation",
            "--prompt-texts",
            "first prompt,second prompt",
        ]), runner=real_runner)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["real_local"])
        self.assertEqual(result["max_new_tokens"], 8)
        self.assertEqual(len(calls), 2)
        report = json.loads((output_dir / "v2" / "public_swarm_inference_v2.json").read_text(encoding="utf-8"))
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["readiness"]["local_p2p_generate"]["route_source"], "p2p-discovery")

    def test_stream_generation_blocks_v2_when_local_stream_missing(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)
        usable_payload = json.loads(sources["usable"].read_text(encoding="utf-8"))
        p2p = usable_payload["readiness"]["p2p_product_path"]
        p2p["stream_ready"] = False
        p2p.pop("stream", None)
        usable_payload["diagnosis_codes"] = [
            code for code in usable_payload.get("diagnosis_codes", [])
            if code not in {"usable_real_llm_stream_ready", "public_swarm_generate_stream_ready"}
        ]
        sources["usable"].write_text(json.dumps(usable_payload) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2-stream-missing"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--stream-generation",
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["readiness"]["local_p2p_generate"]["stream_ready"])
        self.assertIn("local safe stream progress", report["not_completed"])
        self.assertNotIn("public_swarm_v2_stream_generation_ready", report["diagnosis_codes"])

    def test_evidence_import_blocks_non_default_model_without_matching_reports(self) -> None:
        output_dir = self._tmp_dir()
        sources = self._write_sources(output_dir, external_tokens=16)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "v2-model-mismatch"),
            "--usable-report",
            str(sources["usable"]),
            "--preview-report",
            str(sources["preview"]),
            "--real-p2p-report",
            str(sources["real_p2p"]),
            "--gpu-report",
            str(sources["gpu"]),
            "--hf-model-id",
            "distilgpt2",
            "--fresh-external-report",
            "--max-new-tokens",
            "16",
        ]))

        self.assertFalse(report["ok"], report)
        readiness = report["readiness"]
        self.assertFalse(readiness["local_p2p_generate"]["model"]["compatible"])
        self.assertFalse(readiness["external_validation"]["model"]["compatible"])
        self.assertFalse(readiness["p2p_route_hardening"]["model"]["compatible"])
        self.assertIn("public_swarm_v2_local_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_external_model_mismatch", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_p2p_model_mismatch", report["diagnosis_codes"])
        self.assertIn("local evidence model match", report["not_completed"])
        self.assertIn("external signed/real P2P evidence model match", report["not_completed"])
        self.assertIn("signed or real P2P evidence model match", report["not_completed"])

    def test_cli_default_real_p2p_report_uses_16_token_import(self) -> None:
        args = cli.parse_args(["public-swarm-v2", "evidence-import"])

        self.assertEqual(args.real_p2p_report, pack.DEFAULT_REAL_P2P_REPORT)
        self.assertIn("16tok", args.real_p2p_report)


if __name__ == "__main__":
    unittest.main()
