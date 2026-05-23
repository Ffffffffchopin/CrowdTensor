from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "inference_session_client.py"
SPEC = importlib.util.spec_from_file_location("inference_session_client", SCRIPT_PATH)
inference_session_client = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inference_session_client)


class InferenceSessionClientTests(unittest.TestCase):
    def _session(self) -> dict:
        return {
            "accepted": True,
            "schema": "inference_session_request_v1",
            "task_id": "task-client",
            "status": "queued",
            "workload_type": "model_bundle_infer",
            "request_count": 4,
            "result_query": "/admin/results?task_id=task-client&workload_type=model_bundle_infer",
            "task_requirements": {"runtime": "python-cli", "backend": "cpu"},
        }

    def _row(self, *, validation: dict | None = None) -> dict:
        validation = validation or {
            "accepted": True,
            "code": "ok",
            "request_count": 4,
            "correct_count": 1,
            "accuracy": 0.25,
            "request_trace_count": 4,
            "request_trace": [
                {
                    "request_id": "req-1",
                    "prompt": "crow",
                    "predicted_token": "e",
                    "target_token": "n",
                    "correct": False,
                }
            ],
        }
        return {
            "event_index": 7,
            "task_id": "task-client",
            "status": "completed",
            "accepted": True,
            "miner_id": "miner-a",
            "workload_type": "model_bundle_infer",
            "attempt": 1,
            "model_updated": False,
            "model_bundle_updated": False,
            "validation": validation,
            "session_metrics": {
                "request_count": validation.get("request_count"),
                "correct_count": validation.get("correct_count"),
                "accuracy": validation.get("accuracy"),
                "elapsed_ms": 2.5,
                "requests_per_second": 1600.0,
            },
            "lease_token": "secret-lease",
            "inference_results": [{"raw": True}],
        }

    def test_build_report_summarizes_success_without_raw_payloads(self) -> None:
        report = inference_session_client.build_report(
            session=self._session(),
            row=self._row(),
            expected_request_count=4,
            attempts=2,
            elapsed_seconds=1.234,
            observations={},
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], "inference_session_client_v1")
        self.assertEqual(report["diagnosis"]["primary_code"], "session_client_ready")
        self.assertEqual(report["diagnosis_codes"], ["session_client_ready"])
        self.assertEqual(report["session"]["task_id"], "task-client")
        self.assertEqual(report["result"]["validation"]["request_count"], 4)
        self.assertEqual(report["result"]["session_metrics"]["requests_per_second"], 1600.0)
        self.assertTrue(report["result"]["read_only"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("secret-lease", encoded)
        self.assertNotIn("inference_results", encoded)

    def test_build_report_classifies_timeout(self) -> None:
        report = inference_session_client.build_report(
            session=self._session(),
            row=None,
            expected_request_count=4,
            attempts=5,
            elapsed_seconds=10.0,
            observations={},
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["diagnosis"]["primary_code"], "session_timeout")

    def test_build_report_classifies_request_count_mismatch(self) -> None:
        validation = {
            "accepted": True,
            "code": "ok",
            "request_count": 3,
            "correct_count": 1,
            "accuracy": 1 / 3,
        }
        report = inference_session_client.build_report(
            session=self._session(),
            row=self._row(validation=validation),
            expected_request_count=4,
            attempts=1,
            elapsed_seconds=0.1,
            observations={},
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["diagnosis"]["primary_code"], "request_count_mismatch")
        self.assertEqual(report["diagnosis"]["details"]["expected_request_count"], 4)

    def test_build_report_classifies_validation_failure_before_request_mismatch(self) -> None:
        validation = {
            "accepted": False,
            "code": "model_bundle_inference_prediction_mismatch",
            "reason": "prediction mismatch",
            "request_count": 3,
        }
        report = inference_session_client.build_report(
            session=self._session(),
            row=self._row(validation=validation),
            expected_request_count=4,
            attempts=1,
            elapsed_seconds=0.1,
            observations={},
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["diagnosis"]["primary_code"], "validation_failed")
        self.assertEqual(report["diagnosis_codes"], ["validation_failed", "request_count_mismatch"])

    def test_observation_failure_code_prefers_auth_and_transport(self) -> None:
        self.assertEqual(
            inference_session_client.observation_failure_code(
                {"ok": False, "status": 403},
                default="session_create_failed",
            ),
            "admin_auth_failed",
        )
        self.assertEqual(
            inference_session_client.observation_failure_code(
                {"ok": False, "status": None},
                default="session_create_failed",
            ),
            "coordinator_unreachable",
        )


if __name__ == "__main__":
    unittest.main()
