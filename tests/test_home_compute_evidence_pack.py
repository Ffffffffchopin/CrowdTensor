from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "home_compute_evidence_pack.py"
SPEC = importlib.util.spec_from_file_location("home_compute_evidence_pack", SCRIPT_PATH)
home_compute_evidence_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(home_compute_evidence_pack)


class HomeComputeEvidencePackTests(unittest.TestCase):
    def _matrix(self, *, workload_status: str = "available") -> dict:
        return {
            "ok": workload_status != "blocked",
            "host_profile": {
                "python": "3.12.0",
                "os": "Linux",
                "machine": "x86_64",
                "cpu_count": 8,
                "platform": "Linux-6.0",
            },
            "configured_runtimes": {
                "external_llm_http": {
                    "configured": True,
                    "url_configured": True,
                    "api_key_configured": True,
                },
            },
            "hardware_targets": [
                {
                    "name": "cpu_baseline",
                    "status": "available" if workload_status == "available" else "blocked",
                    "usable_now": workload_status == "available",
                    "supported_workloads": ["model_bundle_infer"] if workload_status == "available" else [],
                },
            ],
            "recommended_routes": [
                {
                    "name": "local_cpu_model_bundle_infer",
                    "target": "cpu_baseline",
                    "workload": "model_bundle_infer",
                    "status": workload_status,
                    "usable_now": workload_status == "available",
                    "confidence": "ready" if workload_status == "available" else "blocked",
                    "reason": "cpu baseline can run model_bundle_infer",
                    "matched_capabilities": ["target:cpu_baseline", "workload:model_bundle_infer"],
                    "missing_capabilities": [] if workload_status == "available" else ["target:cpu_baseline"],
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
                "blocked": 1 if workload_status == "blocked" else 0,
                "available_workloads": ["model_bundle_infer"] if workload_status == "available" else [],
                "blocked_workloads": ["model_bundle_infer"] if workload_status == "blocked" else [],
            },
            "recommended_next_commands": ["python3 scripts/runtime_matrix.py --json"],
        }

    def _session(self) -> dict:
        return {
            "ok": True,
            "workload_type": "model_bundle_infer",
            "request_count": 4,
            "correct_count": 1,
            "accuracy": 0.25,
            "elapsed_ms": 2.5,
            "requests_per_second": 1600.0,
            "read_only": True,
            "redaction_ok": True,
            "request_trace_count": 1,
            "request_trace_truncated": False,
            "request_trace": [
                {
                    "request_id": "req-1",
                    "prompt": "crow",
                    "predicted_token": "e",
                    "target_token": "n",
                    "correct": False,
                    "top_k": [{"token": "e", "probability": 0.2}],
                },
            ],
        }

    def _home_report(self) -> dict:
        return {
            "ok": True,
            "selected_workload": {
                "name": "model_bundle_infer",
                "status": "available",
                "reason": "ready",
                "cpu_only": True,
            },
            "route_decision": {
                "name": "local_cpu_model_bundle_infer",
                "target": "cpu_baseline",
                "workload": "model_bundle_infer",
                "status": "available",
                "usable_now": True,
                "confidence": "ready",
                "reason": "cpu baseline can run model_bundle_infer",
                "matched_capabilities": ["target:cpu_baseline"],
                "missing_capabilities": [],
                "next_command": "python3 scripts/home_compute_demo.py --json",
            },
            "inference_session": self._session(),
            "safety": {
                "read_only": True,
                "redaction_ok": True,
                "raw_payloads_exposed": False,
            },
            "recommended_next_commands": ["python3 scripts/home_compute_demo.py --json"],
        }

    def test_build_evidence_combines_safe_route_trace_and_runtime_summary(self) -> None:
        evidence = home_compute_evidence_pack.build_evidence(
            matrix=self._matrix(),
            home_report=self._home_report(),
            runtime_report={"present": True, "ok": True, "checks_total": 2, "checks_failed": []},
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["schema"], "home_compute_evidence_v1")
        self.assertEqual(evidence["generated_at"], "2026-05-22T00:00:00+00:00")
        self.assertEqual(evidence["route_decision"]["name"], "local_cpu_model_bundle_infer")
        self.assertEqual(evidence["route_decision"]["confidence"], "ready")
        self.assertEqual(evidence["inference_summary"]["request_count"], 4)
        self.assertEqual(evidence["request_trace"][0]["prompt"], "crow")
        self.assertTrue(evidence["safety"]["read_only"])
        self.assertTrue(evidence["runtime_acceptance"]["present"])
        encoded = json.dumps(evidence, sort_keys=True)
        self.assertNotIn("super-secret-key", encoded)
        self.assertIn("<redacted>", encoded)

    def test_skip_demo_falls_back_to_matrix_route(self) -> None:
        evidence = home_compute_evidence_pack.build_evidence(
            matrix=self._matrix(),
            home_report=None,
            runtime_report={"present": False, "ok": None, "checks_total": 0, "checks_failed": []},
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["selected_workload"]["name"], "model_bundle_infer")
        self.assertEqual(evidence["route_decision"]["name"], "local_cpu_model_bundle_infer")
        self.assertEqual(evidence["route_decision"]["confidence"], "ready")
        self.assertFalse(evidence["inference_summary"]["present"])

    def test_load_runtime_report_summarizes_failures(self) -> None:
        path = Path("/tmp/crowdtensor_home_compute_evidence_runtime.json")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        path.write_text(json.dumps({
            "ok": False,
            "duration_seconds": 1.25,
            "checks": [
                {"name": "readiness", "ok": True},
                {"name": "chaos", "ok": False},
            ],
        }), encoding="utf-8")

        report = home_compute_evidence_pack.load_runtime_report(str(path))

        self.assertTrue(report["present"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["checks_total"], 2)
        self.assertEqual(report["checks_failed"], ["chaos"])

    def test_markdown_renders_route_trace_safety_and_limitations(self) -> None:
        evidence = home_compute_evidence_pack.build_evidence(
            matrix=self._matrix(),
            home_report=self._home_report(),
            runtime_report={"present": False, "ok": None, "checks_total": 0, "checks_failed": []},
            generated_at="2026-05-22T00:00:00+00:00",
        )

        markdown = home_compute_evidence_pack.render_markdown(evidence)

        self.assertIn("# CrowdTensor Home Compute Evidence", markdown)
        self.assertIn("local_cpu_model_bundle_infer", markdown)
        self.assertIn("prompt=`crow`", markdown)
        self.assertIn("Read-only", markdown)
        self.assertIn("CPU-only demo evidence", markdown)


if __name__ == "__main__":
    unittest.main()
