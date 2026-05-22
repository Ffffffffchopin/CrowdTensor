from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "home_compute_demo.py"
SPEC = importlib.util.spec_from_file_location("home_compute_demo", SCRIPT_PATH)
home_compute_demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(home_compute_demo)


class HomeComputeDemoTests(unittest.TestCase):
    def _matrix(self, *, workload_status: str = "available") -> dict:
        return {
            "ok": workload_status != "blocked",
            "host_profile": {
                "python": "3.12.0",
                "os": "Linux",
                "machine": "x86_64",
                "cpu_count": 8,
            },
            "configured_runtimes": {
                "external_llm_http": {
                    "configured": False,
                    "api_key_configured": False,
                },
            },
            "workloads": [
                {
                    "name": "model_bundle_infer",
                    "status": workload_status,
                    "reason": "ready",
                    "cpu_only": True,
                },
            ],
            "summary": {
                "available": 1 if workload_status == "available" else 0,
                "optional_missing": 0,
                "blocked": 1 if workload_status == "blocked" else 0,
                "available_workloads": ["model_bundle_infer"] if workload_status == "available" else [],
                "blocked_workloads": ["model_bundle_infer"] if workload_status == "blocked" else [],
            },
        }

    def _session(self, *, ok: bool = True, read_only: bool = True, redaction_ok: bool = True) -> dict:
        return {
            "ok": ok,
            "workload_type": "model_bundle_infer",
            "request_count": 4,
            "accuracy": 0.25,
            "elapsed_ms": 2.5,
            "requests_per_second": 1600.0,
            "read_only": read_only,
            "redaction_ok": redaction_ok,
        }

    def test_build_home_compute_report_combines_matrix_and_session(self) -> None:
        report = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(),
            session_report=self._session(),
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["demo"], "home_compute_inference_v1")
        self.assertEqual(report["selected_workload"]["name"], "model_bundle_infer")
        self.assertTrue(report["selected_workload"]["cpu_only"])
        self.assertEqual(report["runtime_matrix"]["host_profile"]["cpu_count"], 8)
        self.assertTrue(report["safety"]["read_only"])
        self.assertTrue(report["safety"]["redaction_ok"])

    def test_build_home_compute_report_blocks_missing_workload(self) -> None:
        report = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(workload_status="blocked"),
            session_report=None,
        )

        self.assertFalse(report["ok"])
        self.assertIn("error", report)
        self.assertEqual(report["selected_workload"]["status"], "blocked")
        self.assertIsNone(report["inference_session"])
        self.assertFalse(report["safety"]["raw_payloads_exposed"])

    def test_build_home_compute_report_fails_unsafe_session(self) -> None:
        report = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(),
            session_report=self._session(redaction_ok=False),
        )

        self.assertFalse(report["ok"])
        self.assertTrue(report["safety"]["raw_payloads_exposed"])


if __name__ == "__main__":
    unittest.main()
