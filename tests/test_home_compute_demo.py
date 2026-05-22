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
    def _matrix(self, *, workload_status: str = "available", matrix_ok: bool | None = None, route_ok: bool | None = None) -> dict:
        if matrix_ok is None:
            matrix_ok = workload_status != "blocked"
        if route_ok is None:
            route_ok = workload_status == "available"
        return {
            "ok": matrix_ok,
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
            "hardware_targets": [
                {
                    "name": "cpu_baseline",
                    "status": "available" if route_ok else "blocked",
                    "usable_now": route_ok,
                    "supported_workloads": ["model_bundle_infer"] if route_ok else [],
                },
                {
                    "name": "nvidia_cuda",
                    "status": "optional_missing",
                    "usable_now": False,
                    "supported_workloads": [],
                },
            ],
            "recommended_routes": [
                {
                    "name": "local_cpu_model_bundle_infer",
                    "target": "cpu_baseline",
                    "workload": "model_bundle_infer",
                    "status": "available" if route_ok else "blocked",
                    "usable_now": route_ok,
                    "confidence": "ready" if route_ok else "blocked",
                    "reason": "cpu baseline can run model_bundle_infer"
                    if route_ok else
                    "cpu baseline is blocked",
                    "matched_capabilities": ["target:cpu_baseline", "workload:model_bundle_infer"]
                    if route_ok else [],
                    "missing_capabilities": []
                    if route_ok else ["target:cpu_baseline"],
                    "next_command": "python3 scripts/home_compute_demo.py --json",
                },
            ],
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
                "blocked": 0 if matrix_ok else 1,
                "available_workloads": ["model_bundle_infer"] if workload_status == "available" else [],
                "blocked_workloads": ["model_bundle_infer"] if not matrix_ok else [],
            },
        }

    def _session(
        self,
        *,
        ok: bool = True,
        read_only: bool = True,
        redaction_ok: bool = True,
        request_count: int = 4,
        trace: list[dict] | None = None,
    ) -> dict:
        request_trace = [
            {
                "request_id": "req-1",
                "prompt": "crow",
                "predicted_token": "e",
                "target_token": "n",
                "correct": False,
                "top_k": [{"token_id": 5, "token": "e", "probability": 0.2}],
            },
        ] if trace is None else trace
        return {
            "ok": ok,
            "workload_type": "model_bundle_infer",
            "request_count": request_count,
            "accuracy": 0.25,
            "elapsed_ms": 2.5,
            "requests_per_second": 1600.0,
            "request_trace": request_trace,
            "request_trace_count": len(request_trace),
            "request_trace_truncated": False,
            "read_only": read_only,
            "redaction_ok": redaction_ok,
        }

    def test_build_home_compute_report_combines_matrix_and_session(self) -> None:
        report = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(),
            session_report=self._session(),
            expected_request_count=4,
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["demo"], "home_compute_inference_v1")
        self.assertEqual(report["selected_workload"]["name"], "model_bundle_infer")
        self.assertEqual(report["capability_route"]["name"], "local_cpu_model_bundle_infer")
        self.assertTrue(report["capability_route"]["usable_now"])
        self.assertEqual(report["route_decision"]["name"], "local_cpu_model_bundle_infer")
        self.assertEqual(report["route_decision"]["confidence"], "ready")
        self.assertIn("target:cpu_baseline", report["route_decision"]["matched_capabilities"])
        self.assertEqual(report["route_decision"]["missing_capabilities"], [])
        self.assertTrue(report["runtime_matrix"]["hardware_targets"][0]["usable_now"])
        self.assertTrue(report["selected_workload"]["cpu_only"])
        self.assertEqual(report["runtime_matrix"]["host_profile"]["cpu_count"], 8)
        self.assertEqual(report["inference_session"]["request_trace_count"], 1)
        self.assertEqual(report["inference_session"]["request_trace"][0]["prompt"], "crow")
        self.assertTrue(report["safety"]["read_only"])
        self.assertTrue(report["safety"]["redaction_ok"])
        self.assertEqual(report["diagnosis"]["primary_code"], "home_compute_ready")
        self.assertEqual(report["diagnosis_codes"], ["home_compute_ready"])

    def test_build_home_compute_report_blocks_missing_workload(self) -> None:
        report = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(workload_status="blocked"),
            session_report=None,
            expected_request_count=4,
        )

        self.assertFalse(report["ok"])
        self.assertIn("error", report)
        self.assertEqual(report["selected_workload"]["status"], "blocked")
        self.assertEqual(report["route_decision"]["confidence"], "blocked")
        self.assertIn("target:cpu_baseline", report["route_decision"]["missing_capabilities"])
        self.assertIsNone(report["inference_session"])
        self.assertFalse(report["safety"]["raw_payloads_exposed"])
        self.assertEqual(report["diagnosis"]["primary_code"], "runtime_matrix_blocked")
        self.assertIn("workload_unavailable", report["diagnosis_codes"])

    def test_build_home_compute_report_fails_unsafe_session(self) -> None:
        report = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(),
            session_report=self._session(redaction_ok=False),
            expected_request_count=4,
        )

        self.assertFalse(report["ok"])
        self.assertTrue(report["safety"]["raw_payloads_exposed"])
        self.assertEqual(report["diagnosis"]["primary_code"], "unsafe_session")

    def test_diagnosis_classifies_unavailable_cpu_route(self) -> None:
        report = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(workload_status="available", route_ok=False),
            session_report=self._session(),
            expected_request_count=4,
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["diagnosis"]["primary_code"], "cpu_route_unavailable")

    def test_diagnosis_classifies_missing_and_failed_session(self) -> None:
        missing = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(),
            session_report=None,
            expected_request_count=4,
        )
        failed = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(),
            session_report=self._session(ok=False),
            expected_request_count=4,
        )

        self.assertEqual(missing["diagnosis"]["primary_code"], "session_not_run")
        self.assertEqual(failed["diagnosis"]["primary_code"], "session_failed")

    def test_diagnosis_classifies_request_count_and_trace_gaps(self) -> None:
        mismatch = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(),
            session_report=self._session(request_count=2),
            expected_request_count=4,
        )
        missing_trace = home_compute_demo.build_home_compute_report(
            matrix=self._matrix(),
            session_report=self._session(trace=[]),
            expected_request_count=4,
        )

        self.assertEqual(mismatch["diagnosis"]["primary_code"], "request_count_mismatch")
        self.assertEqual(missing_trace["diagnosis"]["primary_code"], "trace_missing")


if __name__ == "__main__":
    unittest.main()
