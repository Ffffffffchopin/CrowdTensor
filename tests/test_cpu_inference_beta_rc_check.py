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
    def test_validate_report_requires_readiness_and_artifacts(self) -> None:
        payload = {
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
                ]
            },
        }

        check.validate_report(payload)

    def test_validate_report_rejects_secret_fragment(self) -> None:
        payload = {
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
                ]
            },
            "bad": "lease_token",
        }
        with self.assertRaises(SystemExit):
            check.validate_report(payload)


if __name__ == "__main__":
    unittest.main()
