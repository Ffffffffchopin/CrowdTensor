from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "cpu_inference_beta_rc_check.py"
SPEC = importlib.util.spec_from_file_location("cpu_inference_beta_rc_check", SCRIPT_PATH)
check = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check)


class CpuInferenceBetaRCCheckTests(unittest.TestCase):
    def _ready_payload(self) -> dict:
        return {
            "schema": "cpu_inference_beta_rc_v1",
            "ok": True,
            "diagnosis_codes": [
                "cpu_inference_beta_rc_ready",
                "local_cpu_inference_ready",
                "remote_loopback_ready",
                "two_machine_rehearsal_ready",
                "kaggle_remote_miner_artifacts_ready",
                "miner_join_pack_ready",
                "cpu_miner_beta_ready",
            ],
            "miner_join_pack": {"schema": "miner_join_pack_v1", "ready": True},
            "safety": {
                "cpu_only": True,
                "read_only": True,
                "not_production": True,
                "not_p2p": True,
                "not_gpu_tpu_workload": True,
            },
            "output_request": {
                "include_output": False,
                "raw_prompt_public": False,
                "raw_generation_public": False,
                "raw_external_llm_output_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "built-in-fixed-scenarios",
                "inline_prompt_text": False,
                "terminal_next_commands_local_private": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "raw_generation_public": False,
                "raw_external_llm_output_public": False,
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generation_public": False,
                "raw_external_llm_output_public": False,
            },
            "user_status": {
                "state": "ready",
                "public_artifact_safe": True,
            },
            "review_summary": {
                "schema": "cpu_inference_beta_rc_review_summary_v1",
                "state": "ready",
            },
            "recommended_next_command": {
                "command_line": "crowdtensor cpu-infer --mode beta-rc --json",
                "public_artifact_safe": True,
            },
            "next_commands": [
                {"label": "inspect", "command_line": "cat cpu_inference_beta_rc.md"},
                {"label": "support", "command_line": "cat support_bundle.json"},
            ],
            "not_completed": [],
            "artifact_summary": {
                "artifact_count": 10,
                "present_artifact_count": 10,
                "public_artifact_safe": True,
            },
            "artifacts": {
                name: {"present": True}
                for name in [
                    "cpu_inference_beta_rc_json",
                    "cpu_inference_beta_rc_markdown",
                    "local_cpu_inference_beta_json",
                    "remote_loopback_cpu_inference_beta_json",
                    "kaggle_remote_miner_script",
                    "kaggle_remote_miner_runbook",
                    "miner_join_script",
                    "miner_join_runbook",
                    "demo_manifest_json",
                    "support_bundle_json",
                ]
            },
        }

    def test_validate_report_requires_readiness_and_artifacts(self) -> None:
        check.validate_report(self._ready_payload())

    def test_validate_report_rejects_secret_fragment(self) -> None:
        payload = self._ready_payload()
        payload["bad"] = "lease_token"
        with self.assertRaises(SystemExit):
            check.validate_report(payload)


if __name__ == "__main__":
    unittest.main()
