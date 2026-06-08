from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import petals_class_p2p_candidate_pack as pack
from scripts import petals_class_p2p_candidate_check as check


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class PetalsClassP2pCandidatePackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_petals_candidate_pack_test_"))

    def _write(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def _args_for_reports(self, output_dir: Path, *, requeue_report: dict) -> pack.argparse.Namespace:
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        self._write(local_path, check.fake_real_p2p_report(mode="local-smoke"))
        self._write(runtime_path, check.fake_real_p2p_report(mode="kaggle-runtime-smoke"))
        self._write(external_path, check.fake_real_p2p_report(mode="kaggle-auto"))
        self._write(requeue_path, requeue_report)
        return pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--max-new-tokens",
            "8",
        ])

    def test_package_marks_runbook_ready_without_full_candidate_readiness(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["package", "--output-dir", str(output_dir)]))

        self.assertTrue(report["ok"], report)
        self.assertIn("petals_class_p2p_candidate_runbook_ready", report["diagnosis_codes"])
        self.assertNotIn("petals_class_p2p_candidate_ready", report["diagnosis_codes"])
        self.assertEqual(report["user_status"]["state"], "package-ready")
        self.assertEqual(report["review_summary"]["state"], "package-ready")
        self.assertFalse(report["review_summary"]["ready"])
        self.assertEqual(report["recommended_next_command"]["reason"], "verify_local_candidate_path")
        self.assertIn("petals-candidate local-smoke", report["recommended_next_command"]["command_line"])
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertGreaterEqual(report["artifact_summary"]["present_artifact_count"], 3)
        self.assertTrue((output_dir / "PETALS_CLASS_P2P_CANDIDATE.md").is_file())

    def test_evidence_import_requires_victim_result_rejection_for_requeue(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        requeue_report["live_requeue_summary"]["victim_result_accepted"] = True

        report = pack.build_report(self._args_for_reports(output_dir, requeue_report=requeue_report))

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["candidate"]["external_stage_requeue_ready"])
        self.assertFalse(report["candidate"]["victim_result_not_accepted"])
        self.assertNotIn("p2p_live_requeue_rescue_ready", report["diagnosis_codes"])
        self.assertIn("petals_class_p2p_candidate_blocked", report["diagnosis_codes"])
        self.assertIn("external requeue rescue proof with victim result rejection", report["not_completed"])

    def test_evidence_import_preserves_live_requeue_summary_when_ready(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)

        report = pack.build_report(self._args_for_reports(output_dir, requeue_report=requeue_report))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["candidate"]["p2p_live_requeue_ready"])
        self.assertTrue(report["candidate"]["victim_result_not_accepted"])
        self.assertTrue(report["candidate"]["model_id_consistent"])
        self.assertEqual(report["candidate"]["hf_model_id"], "sshleifer/tiny-gpt2")
        self.assertFalse(report["candidate"]["live_requeue_summary"]["victim_result_accepted"])
        self.assertIn("p2p_live_requeue_rescue_ready", report["diagnosis_codes"])
        self.assertIn("p2p_victim_result_not_accepted", report["diagnosis_codes"])
        self.assertIn("p2p_candidate_model_id_consistent", report["diagnosis_codes"])
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
        self.assertEqual(report["review_summary"]["schema"], "petals_class_p2p_candidate_review_summary_v1")
        self.assertEqual(report["review_summary"]["state"], "ready")
        self.assertTrue(report["review_summary"]["ready"])
        self.assertEqual(report["review_summary"]["next_step"], "review_artifacts")
        self.assertEqual(report["review_summary"]["recommended_next_command"], report["recommended_next_command"])
        self.assertEqual(report["review_summary"]["next_command"], report["recommended_next_command"]["command_line"])
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["user_status"]["recommended_label"], report["recommended_next_command"]["label"])
        self.assertEqual(report["recommended_next_command"]["reason"], "review_artifacts")
        self.assertTrue(report["recommended_next_command"]["public_artifact_safe"])
        self.assertTrue(any(item["label"] == "inspect support bundle" for item in report["next_commands"]))
        self.assertEqual(report["artifact_summary"]["schema"], "petals_class_p2p_candidate_artifact_summary_v1")
        self.assertTrue(report["artifact_summary"]["inspect_first"].endswith("petals_class_p2p_candidate.md"))
        self.assertTrue(report["artifact_summary"]["summary_json"].endswith("petals_class_p2p_candidate.json"))
        self.assertTrue(report["artifact_summary"]["support_bundle"].endswith("support_bundle.json"))
        self.assertGreaterEqual(report["artifact_summary"]["present_artifact_count"], 3)
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        markdown = (output_dir / "candidate" / "petals_class_p2p_candidate.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("## Artifact Summary", markdown)
        self.assertIn("- recommended next:", markdown)
        self.assertIn("inspect support bundle", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- output request note:", markdown)
        self.assertIn("answer text", markdown)
        self.assertIn("prompt scope: `source=prompt-text count=1", markdown)
        self.assertIn("- prompt scope note:", markdown)
        self.assertIn("raw prompt text", markdown)
        self.assertIn("state=no-local-answer", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not an answer transcript", markdown)
        self.assertIn("raw_generated_text_public=False", markdown)
        support = json.loads((output_dir / "candidate" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["review_summary"], report["review_summary"])
        self.assertEqual(support["user_status"], report["user_status"])
        self.assertEqual(support["recommended_next_command"], report["recommended_next_command"])
        self.assertEqual(support["next_commands"], report["next_commands"])
        self.assertEqual(support["artifact_summary"], report["artifact_summary"])
        self.assertEqual(support["prompt_scope"], report["prompt_scope"])
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")

    def test_evidence_import_recovers_redacted_lease_timeout_when_diagnosed(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        requeue_report["live_requeue_summary"]["lease_expired"] = "<redacted>"
        requeue_report["diagnosis_codes"].append("live_requeue_lease_timeout_observed")

        report = pack.build_report(self._args_for_reports(output_dir, requeue_report=requeue_report))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["candidate"]["p2p_live_requeue_ready"])
        self.assertIs(report["candidate"]["live_requeue_summary"]["lease_expired"], True)
        self.assertIn("p2p_live_requeue_rescue_ready", report["diagnosis_codes"])

    def test_evidence_import_does_not_recover_redacted_lease_timeout_without_diagnosis(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        requeue_report["live_requeue_summary"]["lease_expired"] = "<redacted>"

        report = pack.build_report(self._args_for_reports(output_dir, requeue_report=requeue_report))

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["candidate"]["p2p_live_requeue_ready"])
        self.assertEqual(report["candidate"]["live_requeue_summary"]["lease_expired"], "<redacted>")
        self.assertIn("external requeue rescue proof with victim result rejection", report["not_completed"])

    def test_evidence_import_preserves_safe_batch_and_stream_summaries(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        self._write(local_path, check.fake_real_p2p_report(mode="local-smoke"))
        self._write(runtime_path, check.fake_real_p2p_report(mode="kaggle-runtime-smoke"))
        self._write(external_path, check.add_safe_batch_stream(check.fake_real_p2p_report(mode="kaggle-auto")))
        self._write(requeue_path, requeue_report)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--max-new-tokens",
            "8",
        ]))
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["candidate"]["batch_ready"])
        self.assertTrue(report["candidate"]["stream_ready"])
        self.assertTrue(report["candidate"]["batch"]["batch_generation_ready"])
        self.assertTrue(report["candidate"]["stream"]["stream_generation_ready"])
        self.assertIn("p2p_candidate_batch_generation_ready", report["diagnosis_codes"])
        self.assertIn("p2p_candidate_stream_generation_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        self.assertNotIn('"generated_text":', encoded)
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn('"prompt_text":', encoded)

    def test_evidence_import_accepts_supplemental_batch_stream_report(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        batch_stream_path = source_dir / "batch-stream.json"
        self._write(local_path, check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        self._write(runtime_path, check.fake_real_p2p_report(mode="kaggle-runtime-smoke", generated_tokens=16))
        self._write(external_path, check.fake_real_p2p_report(mode="kaggle-auto", generated_tokens=16))
        self._write(requeue_path, requeue_report)
        self._write(batch_stream_path, check.add_safe_batch_stream(check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16)))

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--batch-stream-report",
            str(batch_stream_path),
            "--max-new-tokens",
            "16",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["candidate"]["batch_stream_source"], "batch_stream_report")
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertTrue(report["candidate"]["batch_ready"])
        self.assertTrue(report["candidate"]["stream_ready"])
        self.assertTrue(report["candidate"]["batch"]["batch_generation_ready"])
        self.assertTrue(report["candidate"]["stream"]["stream_generation_ready"])
        self.assertIn("batch_stream_report", report["source_reports"])
        self.assertTrue(report["artifacts"]["batch_stream_report"]["present"])
        self.assertIn("p2p_candidate_batch_generation_ready", report["diagnosis_codes"])
        self.assertIn("p2p_candidate_stream_generation_ready", report["diagnosis_codes"])

    def test_evidence_import_prefers_batch_stream_prompt_scope(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        batch_stream_path = source_dir / "batch-stream.json"
        batch_stream = check.add_safe_batch_stream(check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        batch_stream["prompt_scope"] = {
            "source": "prompt-texts",
            "prompt_count": 2,
            "inline_prompt_text": True,
            "terminal_next_commands_local_private": True,
            "terminal_logs_local_private": True,
            "saved_artifacts_prompt_placeholders": True,
            "saved_artifacts_public_safe": True,
            "prefer_prompt_file_or_stdin_for_shareable_logs": True,
            "prompt_file_path_public": False,
            "raw_prompt_public": False,
            "public_artifact_safe": True,
        }
        self._write(local_path, check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        self._write(runtime_path, check.fake_real_p2p_report(mode="kaggle-runtime-smoke", generated_tokens=16))
        self._write(external_path, check.fake_real_p2p_report(mode="kaggle-auto", generated_tokens=16))
        self._write(requeue_path, requeue_report)
        self._write(batch_stream_path, batch_stream)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--batch-stream-report",
            str(batch_stream_path),
            "--max-new-tokens",
            "16",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])

    def test_evidence_import_without_source_prompt_scope_uses_safe_fallback(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        for payload in [
            requeue_report,
        ]:
            payload.pop("prompt_scope", None)
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        local = check.fake_real_p2p_report(mode="local-smoke")
        runtime = check.fake_real_p2p_report(mode="kaggle-runtime-smoke")
        external = check.fake_real_p2p_report(mode="kaggle-auto")
        for payload in [local, runtime, external]:
            payload.pop("prompt_scope", None)
        self._write(local_path, local)
        self._write(runtime_path, runtime)
        self._write(external_path, external)
        self._write(requeue_path, requeue_report)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--max-new-tokens",
            "8",
        ]))
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt_scope"]["source"], "imported-or-built-in-validation-prompts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertFalse(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertNotIn("CrowdTensor Petals candidate", encoded)

    def test_evidence_import_rejects_batch_stream_ready_codes_without_structured_evidence(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        batch_stream_path = source_dir / "batch-stream.json"
        batch_stream = check.add_safe_batch_stream(check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        batch_stream["batch"]["observed_request_count"] = 1
        batch_stream["batch"]["result_count"] = 1
        batch_stream["batch"]["batch_generation_ready"] = False
        batch_stream["stream"]["progress"].pop("per_request_progress", None)
        batch_stream["stream"]["progress"]["per_request_progress_complete"] = False
        batch_stream["stream"]["progress"]["per_request_monotonic_progress"] = False
        batch_stream["stream"]["stream_generation_ready"] = True
        self._write(local_path, check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        self._write(runtime_path, check.fake_real_p2p_report(mode="kaggle-runtime-smoke", generated_tokens=16))
        self._write(external_path, check.fake_real_p2p_report(mode="kaggle-auto", generated_tokens=16))
        self._write(requeue_path, requeue_report)
        self._write(batch_stream_path, batch_stream)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--batch-stream-report",
            str(batch_stream_path),
            "--max-new-tokens",
            "16",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["candidate"]["batch_ready"])
        self.assertFalse(report["candidate"]["stream_ready"])
        self.assertFalse(report["candidate"]["stream"]["stream_generation_ready"])
        self.assertNotIn("p2p_candidate_batch_generation_ready", report["diagnosis_codes"])
        self.assertNotIn("p2p_candidate_stream_generation_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])

    def test_evidence_import_rejects_batch_with_duplicate_request_identity(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        batch_stream_path = source_dir / "batch-stream.json"
        batch_stream = check.add_safe_batch_stream(check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        batch_stream["batch"]["results"][1]["request_id"] = "req-0"
        batch_stream["batch"]["results"][1]["prompt_hash"] = "sha256:a"
        self._write(local_path, check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        self._write(runtime_path, check.fake_real_p2p_report(mode="kaggle-runtime-smoke", generated_tokens=16))
        self._write(external_path, check.fake_real_p2p_report(mode="kaggle-auto", generated_tokens=16))
        self._write(requeue_path, requeue_report)
        self._write(batch_stream_path, batch_stream)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--batch-stream-report",
            str(batch_stream_path),
            "--max-new-tokens",
            "16",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["candidate"]["batch_ready"])
        self.assertFalse(report["candidate"]["batch"]["batch_identity_ready"])
        self.assertNotIn("p2p_candidate_batch_generation_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_evidence_import_rejects_stale_batch_ready_without_structured_evidence(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        batch_stream_path = source_dir / "batch-stream.json"
        batch_stream = check.add_safe_batch_stream(check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        batch_stream["batch_ready"] = True
        batch_stream["batch"] = {
            "enabled": True,
            "batch_generation_ready": True,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        self._write(local_path, check.fake_real_p2p_report(mode="local-smoke", generated_tokens=16))
        self._write(runtime_path, check.fake_real_p2p_report(mode="kaggle-runtime-smoke", generated_tokens=16))
        self._write(external_path, check.fake_real_p2p_report(mode="kaggle-auto", generated_tokens=16))
        self._write(requeue_path, requeue_report)
        self._write(batch_stream_path, batch_stream)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--batch-stream-report",
            str(batch_stream_path),
            "--max-new-tokens",
            "16",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["candidate"]["batch_ready"])
        self.assertFalse(report["candidate"]["batch"]["batch_identity_ready"])
        self.assertFalse(report["candidate"]["batch"]["batch_generation_ready"])
        self.assertNotIn("p2p_candidate_batch_generation_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_evidence_import_rejects_supplemental_batch_stream_model_mismatch(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        source_dir = output_dir / "sources"
        local_path = source_dir / "local.json"
        runtime_path = source_dir / "runtime.json"
        external_path = source_dir / "external.json"
        requeue_path = source_dir / "requeue.json"
        batch_stream_path = source_dir / "batch-stream.json"
        batch_stream = check.add_safe_batch_stream(check.fake_real_p2p_report(mode="local-smoke"))
        batch_stream["hf_model_id"] = "distilgpt2"
        self._write(local_path, check.fake_real_p2p_report(mode="local-smoke"))
        self._write(runtime_path, check.fake_real_p2p_report(mode="kaggle-runtime-smoke"))
        self._write(external_path, check.fake_real_p2p_report(mode="kaggle-auto"))
        self._write(requeue_path, requeue_report)
        self._write(batch_stream_path, batch_stream)

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "candidate"),
            "--local-report",
            str(local_path),
            "--runtime-smoke-report",
            str(runtime_path),
            "--external-report",
            str(external_path),
            "--requeue-report",
            str(requeue_path),
            "--batch-stream-report",
            str(batch_stream_path),
            "--max-new-tokens",
            "8",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertTrue(report["candidate"]["batch_ready"])
        self.assertTrue(report["candidate"]["stream_ready"])
        self.assertFalse(report["candidate"]["model_id_consistent"])
        self.assertEqual(report["candidate"]["observed_hf_model_ids"], ["distilgpt2", "sshleifer/tiny-gpt2"])
        self.assertIn("p2p_candidate_model_id_mismatch", report["diagnosis_codes"])
        self.assertIn("P2P candidate model-id consistency", report["not_completed"])

    def test_evidence_import_blocks_mismatched_model_ids(self) -> None:
        output_dir = self._tmp_dir()
        requeue_report = check.fake_real_p2p_report(mode="kaggle-auto", requeue=True)
        requeue_report["hf_model_id"] = "distilgpt2"

        report = pack.build_report(self._args_for_reports(output_dir, requeue_report=requeue_report))

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["candidate"]["model_id_consistent"])
        self.assertEqual(report["candidate"]["observed_hf_model_ids"], ["distilgpt2", "sshleifer/tiny-gpt2"])
        self.assertIn("p2p_candidate_model_id_mismatch", report["diagnosis_codes"])
        self.assertIn("P2P candidate model-id consistency", report["not_completed"])

    def test_local_smoke_prunes_child_private_artifacts_after_loading_report(self) -> None:
        output_dir = self._tmp_dir()
        args = self._args_for_reports(output_dir, requeue_report=check.fake_real_p2p_report(mode="kaggle-auto", requeue=True))
        args.mode = "local-smoke"
        child_dir = Path(args.output_dir) / "real-p2p-local"
        child_payload = check.fake_real_p2p_report(mode="local-smoke", generated_tokens=8)

        def fake_runner(command: list[str], **_: object) -> object:
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "real_p2p_swarm_inference_core_rc.json").write_text(json.dumps(child_payload) + "\n", encoding="utf-8")
            (child_dir / "libp2p-bootstrap-peer-key.json").write_text('{"private":"secret"}\n', encoding="utf-8")
            (child_dir / "state").mkdir(parents=True, exist_ok=True)
            (child_dir / "state" / "tasks.jsonl").write_text('{"generated_text":"raw"}\n', encoding="utf-8")
            return completed(child_payload)

        report = pack.build_report(args, runner=fake_runner)  # type: ignore[arg-type]

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["local_child_cleanup"]["local_private_artifacts_cleaned"])
        self.assertIn("local_child_private_artifacts_cleaned", report["diagnosis_codes"])
        self.assertFalse((child_dir / "libp2p-bootstrap-peer-key.json").exists())
        self.assertFalse((child_dir / "state").exists())


if __name__ == "__main__":
    unittest.main()
