from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"
SPEC = importlib.util.spec_from_file_location("remote_real_llm_sharded_beta_pack", SCRIPT_PATH)
pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pack)


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class _HttpErrorWithBody(HTTPError):
    def __init__(self, body: str) -> None:
        super().__init__(
            url="http://127.0.0.1:8951/admin/inference-sessions",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )
        self._body = body.encode("utf-8")

    def read(self, *args: object, **kwargs: object) -> bytes:
        return self._body


class RemoteRealLlmShardedBetaPackTests(unittest.TestCase):
    def _evidence_payload(self) -> dict:
        return {
            "schema": "real_llm_sharded_cli_v1",
            "ok": True,
            "diagnosis_codes": [
                "real_llm_sharded_ready",
                "real_llm_artifact_ready",
                "stage_0_accepted",
                "stage_1_accepted",
                "activation_transport_ready",
                "baseline_match",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
            ],
            "session": {
                "schema": "real_llm_sharded_session_v1",
                "session_id": "session-test",
                "stage_count": 2,
                "stage_0_task_id": "task-0",
                "stage_1_task_id": "task-1",
                "request_count": 1,
                "model_id": "sshleifer/tiny-gpt2",
                "max_new_tokens": 1,
            },
            "artifact": {
                "schema": "real_llm_artifact_v1",
                "model_id": "sshleifer/tiny-gpt2",
                "backend": "hf_transformers_cpu",
                "artifact_hash": "sha256:artifact",
                "loaded": True,
            },
            "stage_summary": {
                "stage_0": {
                    "task_id": "task-0",
                    "miner_id": "remote-real-llm-stage0",
                    "activation_count": 1,
                },
                "stage_1": {
                    "task_id": "task-1",
                    "miner_id": "remote-real-llm-stage1",
                    "baseline_match": True,
                    "decoded_tokens_match": True,
                    "request_count": 1,
                },
            },
            "stage_assignment": {
                "mode": "split",
                "required_distinct_stage_miners": True,
                "stage0_miner_id": "remote-real-llm-stage0",
                "stage1_miner_id": "remote-real-llm-stage1",
                "distinct_stage_miners": True,
                "stage_assignment_valid": True,
            },
            "safety": {
                "read_only": True,
                "redaction_ok": True,
                "raw_activation_redacted": True,
                "not_production": True,
            },
        }

    def test_local_mode_guidance_redacts_prompt_texts(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_remote_real_llm_beta_test_"))

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("real-llm-shard-infer", command)
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "CrowdTensor routes home CPU,A miner returns one token")
            local_dir = Path(command[command.index("--output-dir") + 1])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "real_llm_sharded_cli_summary.json").write_text("{}", encoding="utf-8")
            (local_dir / "real_llm_sharded_evidence.json").write_text("{}", encoding="utf-8")
            return completed(self._evidence_payload())

        args = pack.parse_args([
            "--mode",
            "local",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "1",
            "--stage-mode",
            "split",
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "remote_real_llm_sharded_beta_v1")
        self.assertIn("remote_real_llm_sharded_ready", report["diagnosis_codes"])
        self.assertIn("local_real_llm_sharded_inference_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_real_llm_sharded_beta_json"]["present"])
        self.assertTrue(report["artifacts"]["remote_real_llm_sharded_beta_markdown"]["present"])
        self.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
        self.assertEqual(report["user_status"]["proof_level"], "local-cpu")
        self.assertEqual(report["review_summary"]["schema"], "remote_real_llm_sharded_beta_review_summary_v1")
        self.assertEqual(report["not_completed"], [])
        next_command_blob = json.dumps(report["next_commands"], sort_keys=True)
        self.assertIn("<redacted-prompts>", next_command_blob)
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("CrowdTensor routes", encoded)
        self.assertNotIn("A miner returns", encoded)
        markdown = (output_dir / "remote_real_llm_sharded_beta.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertNotIn("CrowdTensor routes", markdown)
        self.assertTrue((output_dir / "support_bundle.json").is_file())

    def test_parse_args_reads_and_forwards_prompt_texts_file(self) -> None:
        prompt_file = Path(tempfile.mkdtemp(prefix="crowdtensor_remote_real_llm_prompts_test_")) / "prompts.txt"
        prompt_file.write_text("first, comma prompt\nsecond prompt\n", encoding="utf-8")

        args = pack.parse_args([
            "--mode",
            "remote-loopback",
            "--prompt-texts-file",
            str(prompt_file),
        ])

        self.assertEqual(args.prompt_texts, "")
        self.assertEqual(args.prompt_texts_file, str(prompt_file))
        self.assertEqual(args.prompt_texts_list, ["first, comma prompt", "second prompt"])
        self.assertEqual(pack.prompt_list_from_args(args), ["first, comma prompt", "second prompt"])

    def test_parse_args_preserves_remote_existing_timeouts(self) -> None:
        args = pack.parse_args([
            "--mode",
            "remote-existing",
            "--coordinator-url",
            "http://127.0.0.1:8951",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--timeout-seconds",
            "900",
            "--remote-timeout-seconds",
            "840",
            "--hf-model-id",
            "gpt2-xl",
            "--real-llm-backend",
            "hf_transformers_cuda",
        ])

        self.assertEqual(args.timeout_seconds, 900)
        self.assertEqual(args.remote_timeout_seconds, 840)
        self.assertEqual(args.hf_model_id, "gpt2-xl")
        self.assertEqual(args.real_llm_backend, "hf_transformers_cuda")

    def test_parse_args_rejects_inline_and_file_prompt_batch(self) -> None:
        prompt_file = Path(tempfile.mkdtemp(prefix="crowdtensor_remote_real_llm_prompts_test_")) / "prompts.txt"
        prompt_file.write_text("first prompt\nsecond prompt\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as raised:
            pack.parse_args([
                "--mode",
                "remote-loopback",
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(
            str(raised.exception),
            "remote_real_llm_sharded_beta accepts either --prompt-texts or --prompt-texts-file, not both",
        )

    def test_payload_summary_extracts_nested_generation_hash(self) -> None:
        summary = pack.payload_summary({
            "schema": "remote_real_llm_sharded_beta_v1",
            "ok": True,
            "diagnosis_codes": ["remote_real_llm_sharded_ready", "multi_token_generation_ready"],
            "payload_summaries": {
                "remote_existing_real_llm_sharded_inference": {
                    "generation": {
                        "max_new_tokens": 4,
                        "generated_token_count": 4,
                        "generated_text_hash": "sha256:nested",
                        "generated_text_redacted": True,
                        "multi_token_generation_ready": True,
                    }
                }
            },
        })

        self.assertEqual(summary["generation"]["generated_token_count"], 4)
        self.assertEqual(summary["generation"]["generated_text_hash"], "sha256:nested")

    def test_remote_existing_reports_missing_hf_dependencies(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_remote_real_llm_beta_test_"))
        args = pack.parse_args([
            "--mode",
            "remote-existing",
            "--coordinator-url",
            "http://127.0.0.1:8951",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "1",
        ])
        error = _HttpErrorWithBody(
            "real_llm_sharded_infer requires optional Hugging Face dependencies: transformers. "
            "Install with: python -m pip install -e .[hf]"
        )

        with patch.object(pack.base, "request_json", side_effect=error):
            report = pack.build_report(args)

        self.assertFalse(report["ok"])
        self.assertEqual(report["schema"], "remote_real_llm_sharded_beta_v1")
        self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])
        self.assertIn("session_create_failed", report["diagnosis_codes"])
        self.assertIn("remote_real_llm_sharded_failed", report["diagnosis_codes"])
        step = report["steps"][0]
        self.assertEqual(step["http_status"], 500)
        self.assertIn("python -m pip install -e .[hf]", step["error"])

    def test_remote_existing_records_transient_state_poll_retry(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_remote_real_llm_beta_test_"))
        args = pack.parse_args([
            "--mode",
            "remote-existing",
            "--coordinator-url",
            "http://127.0.0.1:8951",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "1",
            "--remote-timeout-seconds",
            "3",
            "--poll-interval",
            "0.01",
        ])
        session = {
            "schema": "real_llm_sharded_session_v1",
            "session_id": "session-test",
            "stage_count": 2,
            "request_count": 1,
            "model_id": "sshleifer/tiny-gpt2",
            "max_new_tokens": 1,
        }
        state = {
            "tasks": [
                {
                    "status": "completed",
                    "workload_type": "real_llm_sharded_infer",
                    "workload_metadata": {"session_id": "session-test", "stage_id": 0, "generation_step": 0},
                    "validation": {"code": "ok", "activation_transport_ready": True, "activation_count": 1},
                },
                {
                    "status": "completed",
                    "workload_type": "real_llm_sharded_infer",
                    "workload_metadata": {"session_id": "session-test", "stage_id": 1, "generation_step": 0},
                    "validation": {
                        "code": "ok",
                        "baseline_match": True,
                        "decoded_tokens_match": True,
                        "activation_transport_ready": True,
                        "request_count": 1,
                    },
                },
            ],
            "model": {"model_bundle": {"version": 0, "optimizer_step": 0}, "global_step": 0},
            "model_updates": 0,
        }

        with patch.object(pack.base, "request_json", side_effect=[session, socket.timeout("state read timed out"), state, {"results": []}]), patch.object(pack.base.time, "sleep", return_value=None):
            report = pack.build_report(args)

        self.assertIn("remote_state_poll_retry", report["diagnosis_codes"])
        self.assertEqual(report["steps"][0]["remote_state_poll_error_count"], 1)
        self.assertEqual(report["steps"][0]["remote_state_poll_last_error_type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
