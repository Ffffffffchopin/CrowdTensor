from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_micro_llm_sharded_beta_pack.py"
SPEC = importlib.util.spec_from_file_location("remote_micro_llm_sharded_beta_pack", SCRIPT_PATH)
pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pack)


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class RemoteMicroLlmShardedBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_remote_micro_llm_sharded_beta_test_"))

    def _evidence_payload(self, *, requeue: bool = False) -> dict:
        codes = [
            "micro_llm_sharded_ready",
            "stage_0_accepted",
            "stage_1_accepted",
            "baseline_match",
            "decoded_tokens_match",
            "activation_transport_ready",
        ]
        if requeue:
            codes.append("stage_requeue_ready")
        codes.extend(["distinct_stage_miners", "stage_assignment_valid"])
        return {
            "schema": "micro_llm_sharded_evidence_v1",
            "ok": True,
            "diagnosis_codes": codes,
            "session": {
                "schema": "micro_llm_sharded_session_v1",
                "session_id": "session-test",
                "stage_count": 2,
                "stage_0_task_id": "task-0",
                "stage_1_task_id": "task-1",
                "request_count": 2,
                "decode_steps": 4,
            },
            "stage_summary": {
                "stage_0": {
                    "task_id": "task-0",
                    "miner_id": "remote-micro-llm-shard-miner-stage0",
                    "activation_count": 8,
                    "activation_hashes": ["hash-a", "hash-b"],
                },
                "stage_1": {
                    "task_id": "task-1",
                    "miner_id": "remote-micro-llm-shard-miner-stage1",
                    "baseline_match": True,
                    "decoded_tokens_match": True,
                    "request_count": 2,
                    "decode_steps": 4,
                },
            },
            "stage_assignment": {
                "mode": "split",
                "required_distinct_stage_miners": True,
                "stage0_miner_id": "remote-micro-llm-shard-miner-stage0",
                "stage1_miner_id": "remote-micro-llm-shard-miner-stage1",
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

    def test_remote_loopback_wraps_micro_llm_evidence_pack(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("micro_llm_sharded_inference_evidence_pack.py", command[1])
            self.assertIn("--decode-steps", command)
            self.assertIn("--stage-mode", command)
            self.assertIn("--require-distinct-stage-miners", command)
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            self.assertIn("4", command)
            evidence_json = Path(command[command.index("--json-out") + 1])
            evidence_json.parent.mkdir(parents=True, exist_ok=True)
            evidence_json.write_text("{}", encoding="utf-8")
            evidence_md = Path(command[command.index("--markdown-out") + 1])
            evidence_md.write_text("# evidence\n", encoding="utf-8")
            return completed(self._evidence_payload(requeue=True))

        args = pack.parse_args([
            "--mode",
            "remote-loopback",
            "--output-dir",
            str(output_dir),
            "--failure-mode",
            "kill-stage-after-claim",
            "--request-count",
            "2",
            "--decode-steps",
            "4",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--micro-llm-artifact",
            "dist/micro-llm-artifact",
            "--prompt-texts",
            "arn,ten",
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "remote_micro_llm_sharded_beta_v1")
        self.assertEqual(report["mode"], "remote-loopback")
        self.assertIn("remote_micro_llm_sharded_ready", report["diagnosis_codes"])
        self.assertIn("remote_micro_llm_sharded_loopback_ready", report["diagnosis_codes"])
        self.assertIn("stage_requeue_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_micro_llm_sharded_beta_json"]["present"])
        self.assertTrue(report["artifacts"]["remote_micro_llm_sharded_beta_markdown"]["present"])
        self.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["user_status"]["proof_level"], "local-loopback-remote-stand-in")
        self.assertEqual(report["review_summary"]["schema"], "remote_micro_llm_sharded_beta_review_summary_v1")
        self.assertEqual(report["not_completed"], [])
        self.assertEqual(report["prompt_text_count"], 2)
        self.assertTrue(report["prompt_texts_redacted"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn('"prompt_texts":', encoded)
        self.assertNotIn("arn,ten", encoded)
        next_command_blob = json.dumps(report["next_commands"], sort_keys=True)
        self.assertIn("<redacted-prompts>", next_command_blob)
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertTrue((output_dir / "support_bundle.json").is_file())
        markdown = (output_dir / "remote_micro_llm_sharded_beta.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("local-loopback-remote-stand-in", markdown)
        self.assertNotIn("arn,ten", markdown)
        self.assertTrue(calls)

    def test_local_mode_wraps_micro_llm_cli(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("-m", command)
            self.assertIn("crowdtensor.cli", command)
            self.assertIn("micro-llm-shard-infer", command)
            local_dir = Path(command[command.index("--output-dir") + 1])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "micro_llm_sharded_cli_summary.json").write_text("{}", encoding="utf-8")
            (local_dir / "micro_llm_sharded_evidence.json").write_text("{}", encoding="utf-8")
            payload = self._evidence_payload()
            payload["schema"] = "micro_llm_sharded_cli_v1"
            return completed(payload)

        args = pack.parse_args(["--mode", "local", "--output-dir", str(output_dir), "--request-count", "2"])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("local_micro_llm_sharded_inference_ready", report["diagnosis_codes"])
        self.assertIn("remote_micro_llm_sharded_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["local_micro_llm_sharded_cli_summary"]["present"])
        self.assertTrue(report["artifacts"]["support_bundle_json"]["present"])
        self.assertEqual(report["user_status"]["proof_level"], "local-cpu")


if __name__ == "__main__":
    unittest.main()
