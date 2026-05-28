from __future__ import annotations

import unittest

import scripts.sharded_inference_evidence_pack as pack

class ShardedInferenceEvidencePackTests(unittest.TestCase):
    def test_build_report_requires_two_stages_baseline_and_safety(self) -> None:
        args = type("Args", (), {
            "base_url": "http://127.0.0.1:9820",
        })()
        session = {
            "schema": "sharded_inference_session_v1",
            "session_id": "shard-session-test",
            "stage_count": 2,
            "stage_0_task_id": "task-stage-0",
            "request_count": 2,
            "scenario_id": "route-baseline",
        }
        state = {
            "model": {"global_step": 0, "model_bundle": {"version": 0, "optimizer_step": 0}},
            "model_updates": 0,
            "tasks": [
                {
                    "task_id": "task-stage-0",
                    "status": "completed",
                    "miner_id": "m0",
                    "attempt": 1,
                    "workload_type": pack.WORKLOAD_TYPE,
                    "workload_metadata": {"session_id": "shard-session-test", "stage_id": 0},
                    "validation": {
                        "code": "ok",
                        "stage_id": 0,
                        "activation_transport_ready": True,
                        "activation_count": 2,
                        "activation_bytes": 128,
                        "activation_hashes": ["sha256:a", "sha256:b"],
                    },
                    "metrics": {"elapsed_ms": 1.0},
                },
                {
                    "task_id": "task-stage-1",
                    "status": "completed",
                    "miner_id": "m1",
                    "attempt": 1,
                    "workload_type": pack.WORKLOAD_TYPE,
                    "workload_metadata": {"session_id": "shard-session-test", "stage_id": 1},
                    "validation": {
                        "code": "ok",
                        "stage_id": 1,
                        "activation_transport_ready": True,
                        "baseline_match": True,
                        "request_count": 2,
                        "correct_count": 1,
                        "accuracy": 0.5,
                        "request_trace": [],
                    },
                    "metrics": {"elapsed_ms": 1.0},
                },
            ],
        }
        report = pack.build_report(
            args=args,
            session=session,
            state=state,
            stage_processes=[],
            requeue_summary={"enabled": False},
            ledger_rows=[
                {"task_id": "task-stage-0", "model_updated": False, "model_bundle_updated": False},
                {"task_id": "task-stage-1", "model_updated": False, "model_bundle_updated": False},
            ],
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "sharded_inference_evidence_v1")
        self.assertIn("sharded_inference_ready", report["diagnosis_codes"])
        self.assertIn("activation_transport_ready", report["diagnosis_codes"])
        self.assertTrue(report["safety"]["redaction_ok"])


if __name__ == "__main__":
    unittest.main()
