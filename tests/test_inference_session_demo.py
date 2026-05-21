from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "inference_session_demo.py"
SPEC = importlib.util.spec_from_file_location("inference_session_demo", SCRIPT_PATH)
inference_session_demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inference_session_demo)


class InferenceSessionDemoTests(unittest.TestCase):
    def _state(self, *, raw_payload_key: str = "") -> dict:
        task = {
            "task_id": "task-demo",
            "status": "completed",
            "workload_type": "model_bundle_infer",
            "validation": {
                "code": "ok",
                "request_count": 4,
                "correct_count": 1,
                "accuracy": 0.25,
                "predicted_token": "e",
                "target_token": "n",
            },
            "metrics": {
                "request_count": 4,
                "correct_count": 1,
                "accuracy": 0.25,
                "elapsed_ms": 2.5,
                "requests_per_second": 1600.0,
            },
        }
        if raw_payload_key:
            task[raw_payload_key] = [{"request_id": "req-1"}]
        return {
            "model": {
                "global_step": 0,
                "model_bundle": {
                    "bundle_id": "builtin-char-bundle",
                    "version": 0,
                    "optimizer_step": 0,
                },
            },
            "model_updates": 0,
            "tasks": [task],
            "miner_profiles": {
                "inference-session-demo": {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "last_capabilities": {
                        "hardware_profile": {
                            "os": "Linux",
                            "platform": "Linux",
                            "machine": "x86_64",
                            "processor": "x86_64",
                            "cpu_count": 8,
                            "python_version": "3.12.0",
                        },
                    },
                },
            },
        }

    def _ledger(self) -> dict:
        return {
            "results": [
                {
                    "event_index": 1,
                    "status": "completed",
                    "model_updated": False,
                    "model_bundle_updated": False,
                    "session_metrics": {
                        "request_count": 4,
                        "correct_count": 1,
                        "accuracy": 0.25,
                        "elapsed_ms": 2.5,
                        "requests_per_second": 1600.0,
                    },
                },
            ],
        }

    def test_build_demo_report_summarizes_session_without_raw_results(self) -> None:
        report = inference_session_demo.build_demo_report(
            state=self._state(),
            ledger=self._ledger(),
            miner_summary={"accepted_tasks": 1},
            expected_request_count=4,
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["task_id"], "task-demo")
        self.assertEqual(report["request_count"], 4)
        self.assertEqual(report["requests_per_second"], 1600.0)
        self.assertTrue(report["read_only"])
        self.assertTrue(report["redaction_ok"])
        self.assertEqual(report["miner"]["runtime"], "python-cli")
        self.assertEqual(report["miner"]["hardware_profile"]["cpu_count"], 8)

    def test_build_demo_report_marks_raw_inference_payload_as_not_redacted(self) -> None:
        report = inference_session_demo.build_demo_report(
            state=self._state(raw_payload_key="inference_results"),
            ledger=self._ledger(),
            miner_summary={"accepted_tasks": 1},
            expected_request_count=4,
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["redaction_ok"])

    def test_build_demo_report_marks_model_mutation_as_not_read_only(self) -> None:
        state = self._state()
        state["model"]["model_bundle"]["version"] = 1
        report = inference_session_demo.build_demo_report(
            state=state,
            ledger=self._ledger(),
            miner_summary={"accepted_tasks": 1},
            expected_request_count=4,
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["read_only"])


if __name__ == "__main__":
    unittest.main()
