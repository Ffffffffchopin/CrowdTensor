from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_compute_evidence_pack.py"
SPEC = importlib.util.spec_from_file_location("remote_compute_evidence_pack", SCRIPT_PATH)
remote_compute_evidence_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(remote_compute_evidence_pack)


class RemoteComputeEvidencePackTests(unittest.TestCase):
    def _state(self) -> dict:
        return {
            "accepted_results": 1,
            "rejected_results": 0,
            "task_counts": {"completed": 1, "queued": 0, "leased": 0, "rejected": 0},
            "model_updates": 0,
            "model_bundle_updates": 0,
            "model": {
                "global_step": 0,
                "model_bundle": {
                    "bundle_id": "builtin-char-bundle",
                    "version": 0,
                    "optimizer_step": 0,
                },
            },
            "miner_profiles": {
                "remote-a": {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "accepted": 1,
                    "rejected": 0,
                    "last_capabilities": {
                        "supported_workloads": ["diloco_train", "model_bundle_infer"],
                        "hardware_profile": {
                            "os": "Linux",
                            "platform": "Linux-6.0",
                            "machine": "x86_64",
                            "cpu_count": 8,
                            "python_version": "3.12.0",
                        },
                    },
                    "last_runtime_status": {
                        "accepted_tasks": 1,
                        "request_retries": 0,
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
                        "correct_count": 1,
                        "accuracy": 0.25,
                        "predicted_token": "e",
                        "target_token": "n",
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
                    },
                    "metrics": {
                        "request_count": 4,
                        "elapsed_ms": 2.5,
                        "requests_per_second": 1600.0,
                    },
                },
            ],
        }

    def _ledger(self) -> dict:
        return {
            "results": [
                {
                    "event_index": 3,
                    "status": "completed",
                    "workload_type": "model_bundle_infer",
                    "miner_id": "remote-a",
                    "model_updated": False,
                    "model_bundle_updated": False,
                    "validation": {
                        "code": "ok",
                        "request_count": 4,
                        "correct_count": 1,
                        "accuracy": 0.25,
                        "predicted_token": "e",
                        "target_token": "n",
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
                    },
                    "session_metrics": {
                        "request_count": 4,
                        "elapsed_ms": 2.5,
                        "requests_per_second": 1600.0,
                    },
                },
            ],
        }

    def test_build_evidence_summarizes_remote_route_and_trace(self) -> None:
        evidence = remote_compute_evidence_pack.build_evidence(
            mode="local-loopback",
            base_url="http://127.0.0.1:8912",
            miner_id="remote-a",
            state=self._state(),
            ledger=self._ledger(),
            miner_summary={"accepted_tasks": 1, "request_retries": 0},
            request_count=4,
            invite={"token_hash": "sha256:abc", "env": {"CROWDTENSOR_MINER_TOKEN": "remote-secret"}},
            registry_path=Path("/tmp/miner_registry.json"),
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["schema"], "remote_compute_evidence_v1")
        self.assertEqual(evidence["route_decision"]["name"], "remote_python_model_bundle_infer")
        self.assertEqual(evidence["route_decision"]["confidence"], "ready")
        self.assertIn("runtime:python-cli", evidence["route_decision"]["matched_capabilities"])
        self.assertIn("backend:cpu", evidence["route_decision"]["matched_capabilities"])
        self.assertEqual(evidence["inference_summary"]["request_count"], 4)
        self.assertEqual(evidence["request_trace"][0]["prompt"], "crow")
        self.assertTrue(evidence["safety"]["read_only"])
        self.assertTrue(evidence["safety"]["redaction_ok"])
        self.assertTrue(evidence["invite"]["registry_hashed"])
        encoded = json.dumps(evidence, sort_keys=True)
        self.assertNotIn("remote-secret", encoded)
        self.assertNotIn("inference_results", encoded)

    def test_collect_mode_does_not_require_invite_but_requires_completed_result(self) -> None:
        state = self._state()
        state["tasks"] = []
        evidence = remote_compute_evidence_pack.build_evidence(
            mode="collect",
            base_url="https://coordinator.example",
            miner_id="remote-a",
            state=state,
            ledger={"results": []},
            miner_summary=None,
            request_count=4,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertFalse(evidence["ok"])
        self.assertIsNone(evidence["invite"]["registry_hashed"])
        self.assertIn("completed_result", evidence["route_decision"]["missing_capabilities"])
        self.assertFalse(evidence["inference_summary"]["present"])

    def test_markdown_renders_remote_sections(self) -> None:
        evidence = remote_compute_evidence_pack.build_evidence(
            mode="local-loopback",
            base_url="http://127.0.0.1:8912",
            miner_id="remote-a",
            state=self._state(),
            ledger=self._ledger(),
            miner_summary={"accepted_tasks": 1},
            request_count=4,
            invite={"token_hash": "sha256:abc"},
            generated_at="2026-05-22T00:00:00+00:00",
        )

        markdown = remote_compute_evidence_pack.render_markdown(evidence)

        self.assertIn("# CrowdTensor Remote Compute Evidence", markdown)
        self.assertIn("remote-a", markdown)
        self.assertIn("remote_python_model_bundle_infer", markdown)
        self.assertIn("prompt=`crow`", markdown)
        self.assertIn("Registry hashed", markdown)


if __name__ == "__main__":
    unittest.main()
