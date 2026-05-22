from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_demo_acceptance_pack.py"
SPEC = importlib.util.spec_from_file_location("remote_demo_acceptance_pack", SCRIPT_PATH)
remote_demo_acceptance_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(remote_demo_acceptance_pack)


class RemoteDemoAcceptancePackTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            coordinator_url="https://coordinator.example",
            miner_id="remote-a",
            observer_token="observer-secret",
            admin_token="admin-secret",
            request_count=4,
            timeout_seconds=120.0,
            poll_interval=2.0,
            http_timeout=5.0,
            artifact_timeout=60.0,
            admin_results_limit=10,
            output_dir="dist/remote-demo-acceptance",
            json_out="",
            markdown_out="",
        )

    def _state(self) -> dict:
        return {
            "accepted_results": 1,
            "task_counts": {"completed": 1, "queued": 1, "leased": 0, "rejected": 0},
            "miner_profiles": {
                "remote-a": {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "accepted": 1,
                    "rejected": 0,
                    "last_capabilities": {
                        "supported_workloads": ["model_bundle_infer"],
                    },
                },
            },
            "tasks": [
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "miner_id": "remote-a",
                    "workload_type": "model_bundle_infer",
                    "validation": {
                        "code": "ok",
                        "request_count": 4,
                        "request_trace_count": 4,
                        "accuracy": 0.25,
                    },
                },
            ],
        }

    def _results(self) -> dict:
        return {
            "results": [
                {
                    "status": "completed",
                    "miner_id": "remote-a",
                    "workload_type": "model_bundle_infer",
                    "validation": {
                        "code": "ok",
                        "request_count": 4,
                        "request_trace_count": 4,
                        "accuracy": 0.25,
                    },
                },
            ],
        }

    def test_readiness_summary_requires_profile_result_validation_and_request_count(self) -> None:
        summary = remote_demo_acceptance_pack.readiness_summary(
            state=self._state(),
            results=self._results(),
            miner_id="remote-a",
            request_count=4,
        )

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["route"], "remote_python_model_bundle_infer")
        self.assertIn("runtime:python-cli", summary["matched_capabilities"])
        self.assertIn("accepted_result", summary["matched_capabilities"])
        self.assertFalse(summary["missing_capabilities"])

        failed = remote_demo_acceptance_pack.readiness_summary(
            state=self._state(),
            results={"results": []},
            miner_id="remote-a",
            request_count=4,
        )
        self.assertFalse(failed["ready"])
        self.assertIn("accepted_result", failed["missing_capabilities"])

    def test_build_report_summarizes_artifacts_without_secrets(self) -> None:
        args = self._args()
        wait = {
            "ok": True,
            "attempts": 2,
            "elapsed_seconds": 1.5,
            "status": {
                "summary": remote_demo_acceptance_pack.readiness_summary(
                    state=self._state(),
                    results=self._results(),
                    miner_id="remote-a",
                    request_count=4,
                ),
            },
        }
        artifacts = {
            "evidence": {
                "ok": True,
                "path": "/tmp/evidence.json",
                "markdown_path": "/tmp/evidence.md",
                "summary": {
                    "schema": "remote_compute_evidence_v1",
                    "ok": True,
                    "request_count": 4,
                    "read_only": True,
                    "redaction_ok": True,
                },
            },
            "support_bundle": {
                "ok": True,
                "path": "/tmp/support.json",
                "markdown_path": "/tmp/support.md",
                "summary": {
                    "online_enabled": True,
                    "health_ok": True,
                    "state_ok": True,
                    "admin_results_ok": True,
                },
            },
        }

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=wait,
            artifacts=artifacts,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], "remote_demo_acceptance_v1")
        self.assertEqual(report["route"], "remote_python_model_bundle_infer")
        self.assertEqual(report["evidence_summary"]["schema"], "remote_compute_evidence_v1")
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("observer-secret", encoded)
        self.assertNotIn("admin-secret", encoded)

    def test_markdown_renders_acceptance_sections(self) -> None:
        report = {
            "generated_at": "2026-05-22T00:00:00+00:00",
            "ok": True,
            "coordinator_url": "https://coordinator.example",
            "miner_id": "remote-a",
            "route": "remote_python_model_bundle_infer",
            "wait_summary": {"ok": True, "attempts": 1, "elapsed_seconds": 0.1},
            "evidence_summary": {"schema": "remote_compute_evidence_v1", "ok": True},
            "support_bundle_summary": {"online_enabled": True},
            "artifacts": {"evidence_json": "/tmp/evidence.json"},
            "limitations": ["Controlled demo"],
        }

        markdown = remote_demo_acceptance_pack.render_markdown(report)

        self.assertIn("# CrowdTensor Remote Demo Acceptance", markdown)
        self.assertIn("remote_python_model_bundle_infer", markdown)
        self.assertIn("Support Bundle", markdown)


if __name__ == "__main__":
    unittest.main()
