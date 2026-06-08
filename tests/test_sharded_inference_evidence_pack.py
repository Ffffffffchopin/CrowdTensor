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
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["review_summary"]["schema"], "sharded_inference_review_summary_v1")
        self.assertFalse(report["not_completed"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_result_public"])
        self.assertEqual(report["answer_scope"]["scope_state"], "evidence-only")
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertIn("inspect sharded inference evidence", report["recommended_next_command"]["label"])
        self.assertGreaterEqual(len(report["next_commands"]), 3)

    def test_render_markdown_includes_review_scope_and_next_steps(self) -> None:
        args = type("Args", (), {
            "base_url": "http://127.0.0.1:9820",
            "failure_mode": "none",
            "stage_mode": "both",
            "request_count": 2,
            "scenario_id": "route-baseline",
        })()
        report = {
            "schema": pack.SCHEMA,
            "ok": True,
            "workload_type": pack.WORKLOAD_TYPE,
            "session": {
                "session_id": "shard-session-md",
                "stage_count": 2,
                "request_count": 2,
                "scenario_id": "route-baseline",
            },
            "stage_summary": {
                "stage_0": {"task_id": "stage0", "miner_id": "m0", "activation_count": 2},
                "stage_1": {"task_id": "stage1", "miner_id": "m1", "baseline_match": True},
            },
            "observability": {"requeue_summary": {"enabled": False}, "processes": []},
            "diagnosis_codes": [
                "sharded_inference_ready",
                "stage_0_accepted",
                "stage_1_accepted",
                "activation_transport_ready",
                "baseline_match",
            ],
            "safety": {
                "read_only": True,
                "redaction_ok": True,
                "raw_activation_redacted": True,
                "not_production": True,
            },
            "limitations": [],
        }
        report = pack.attach_user_guidance(report, args)

        markdown = pack.render_markdown(report)

        self.assertIn("## Review", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("## Not Completed", markdown)
        self.assertIn("- none", markdown)


if __name__ == "__main__":
    unittest.main()
