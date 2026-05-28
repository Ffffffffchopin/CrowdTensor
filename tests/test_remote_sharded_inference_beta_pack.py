from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_sharded_inference_beta_pack.py"
SPEC = importlib.util.spec_from_file_location("remote_sharded_inference_beta_pack", SCRIPT_PATH)
pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pack)


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class RemoteShardedInferenceBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_remote_sharded_beta_test_"))

    def _evidence_payload(self, *, requeue: bool = False) -> dict:
        codes = [
            "sharded_inference_ready",
            "stage_0_accepted",
            "stage_1_accepted",
            "baseline_match",
            "activation_transport_ready",
        ]
        if requeue:
            codes.append("stage_requeue_ready")
        return {
            "schema": "sharded_inference_evidence_v1",
            "ok": True,
            "diagnosis_codes": codes,
            "session": {
                "schema": "sharded_inference_session_v1",
                "session_id": "session-test",
                "stage_count": 2,
                "stage_0_task_id": "task-0",
                "stage_1_task_id": "task-1",
                "request_count": 2,
                "scenario_id": "route-baseline",
            },
            "stage_summary": {
                "stage_0": {
                    "task_id": "task-0",
                    "miner_id": "remote-shard-miner-stage0",
                    "activation_count": 2,
                    "activation_hashes": ["hash-a", "hash-b"],
                },
                "stage_1": {
                    "task_id": "task-1",
                    "miner_id": "remote-shard-miner-stage1",
                    "baseline_match": True,
                    "request_count": 2,
                },
            },
            "safety": {
                "read_only": True,
                "redaction_ok": True,
                "raw_activation_redacted": True,
                "not_production": True,
            },
        }

    def test_remote_loopback_wraps_sharded_evidence_pack(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("sharded_inference_evidence_pack.py", command[1])
            self.assertIn("--failure-mode", command)
            self.assertIn("kill-stage-after-claim", command)
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
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "remote_sharded_inference_beta_v1")
        self.assertEqual(report["mode"], "remote-loopback")
        self.assertIn("remote_sharded_inference_ready", report["diagnosis_codes"])
        self.assertIn("remote_sharded_loopback_ready", report["diagnosis_codes"])
        self.assertIn("stage_requeue_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["remote_sharded_inference_beta_json"]["present"])
        self.assertTrue(calls)

    def test_local_mode_wraps_cli_shard_infer(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("-m", command)
            self.assertIn("crowdtensor.cli", command)
            self.assertIn("shard-infer", command)
            local_dir = Path(command[command.index("--output-dir") + 1])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "sharded_inference_cli_summary.json").write_text("{}", encoding="utf-8")
            (local_dir / "sharded_inference_evidence.json").write_text("{}", encoding="utf-8")
            payload = self._evidence_payload()
            payload["schema"] = "sharded_inference_cli_v1"
            return completed(payload)

        args = pack.parse_args(["--mode", "local", "--output-dir", str(output_dir), "--request-count", "2"])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("local_sharded_inference_ready", report["diagnosis_codes"])
        self.assertIn("remote_sharded_inference_ready", report["diagnosis_codes"])
        self.assertTrue(report["artifacts"]["local_sharded_inference_cli_summary"]["present"])

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
