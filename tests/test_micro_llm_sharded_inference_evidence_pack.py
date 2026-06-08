from __future__ import annotations

import json
import unittest

import scripts.micro_llm_sharded_inference_evidence_pack as pack


class MicroLlmShardedInferenceEvidencePackTests(unittest.TestCase):
    def test_report_redacts_generated_text_and_adds_user_guidance(self) -> None:
        args = type("Args", (), {
            "base_url": "http://127.0.0.1:9860",
            "require_distinct_stage_miners": True,
            "stage_mode": "split",
            "decode_steps": 2,
            "request_count": 1,
            "failure_mode": "none",
            "prompt_texts": "Tiny prompt",
        })()
        session = {
            "schema": "micro_llm_sharded_session_v1",
            "session_id": "micro-session-test",
            "stage_count": 2,
            "stage_0_task_id": "task-stage-0",
            "stage_1_task_id": "task-stage-1",
            "request_count": 1,
            "decode_steps": 2,
            "prompt_request_count": 1,
        }
        state = {
            "model": {
                "global_step": 0,
                "micro_transformer": {
                    "version": 0,
                    "optimizer_step": 0,
                    "artifact_schema": "micro_llm_artifact_v1",
                    "artifact_id": "tiny-micro",
                    "artifact_hash": "sha256:artifact",
                },
            },
            "model_updates": 0,
            "tasks": [
                {
                    "task_id": "task-stage-0",
                    "status": "completed",
                    "miner_id": "micro-stage0",
                    "attempt": 1,
                    "workload_type": pack.WORKLOAD_TYPE,
                    "workload_metadata": {"session_id": "micro-session-test", "stage_id": 0},
                    "capabilities": {
                        "micro_llm_sharded_stage_role": "stage0",
                        "micro_llm_sharded_stage_capabilities": [pack.MICRO_LLM_SHARDED_STAGE0_CAPABILITY],
                    },
                    "validation": {
                        "code": "ok",
                        "activation_transport_ready": True,
                        "activation_count": 1,
                        "activation_bytes": 128,
                        "activation_hashes": ["sha256:activation"],
                        "artifact_hash": "sha256:artifact",
                    },
                    "metrics": {"elapsed_ms": 1.0},
                },
                {
                    "task_id": "task-stage-1",
                    "status": "completed",
                    "miner_id": "micro-stage1",
                    "attempt": 1,
                    "workload_type": pack.WORKLOAD_TYPE,
                    "workload_metadata": {"session_id": "micro-session-test", "stage_id": 1},
                    "capabilities": {
                        "micro_llm_sharded_stage_role": "stage1",
                        "micro_llm_sharded_stage_capabilities": [pack.MICRO_LLM_SHARDED_STAGE1_CAPABILITY],
                    },
                    "validation": {
                        "code": "ok",
                        "activation_transport_ready": True,
                        "baseline_match": True,
                        "decoded_tokens_match": True,
                        "request_count": 1,
                        "decode_steps": 2,
                        "generated_token_count": 2,
                        "generated_text": "raw micro answer",
                        "request_trace": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:prompt",
                                "activation_hash": "sha256:activation",
                                "output_text": "raw micro answer",
                                "baseline_match": True,
                                "decoded_tokens_match": True,
                                "generated_token_ids": [1, 2],
                            }
                        ],
                        "artifact_hash": "sha256:artifact",
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
                {"task_id": "task-stage-0", "model_updated": False, "micro_transformer_updated": False},
                {"task_id": "task-stage-1", "model_updated": False, "micro_transformer_updated": False},
            ],
        )
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertNotIn("raw micro answer", encoded)
        self.assertNotIn('"generated_text":', encoded)
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["review_summary"]["schema"], "micro_llm_sharded_review_summary_v1")
        self.assertFalse(report["not_completed"])
        self.assertEqual(report["generation"]["generated_token_count"], 2)
        self.assertTrue(report["generation"]["generated_text_redacted"])
        self.assertTrue(report["safety"]["generated_text_redacted"])
        self.assertTrue(report["safety"]["generated_token_ids_redacted"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertEqual(report["answer_scope"]["scope_state"], "hash-only")
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertTrue(report["stage_summary"]["stage_1"]["generated_text_redacted"])
        self.assertIn("generated_text_hash", report["stage_summary"]["stage_1"])

    def test_render_markdown_includes_review_scope_and_next_steps(self) -> None:
        args = type("Args", (), {
            "base_url": "http://127.0.0.1:9860",
            "require_distinct_stage_miners": False,
            "stage_mode": "both",
            "decode_steps": 2,
            "request_count": 1,
            "failure_mode": "none",
            "prompt_texts": "",
        })()
        report = {
            "schema": pack.SCHEMA,
            "ok": True,
            "workload_type": pack.WORKLOAD_TYPE,
            "session": {"session_id": "micro-session-md", "stage_count": 2, "decode_steps": 2, "request_count": 1},
            "artifact": {"loaded": False},
            "generation": {"decode_steps": 2, "generated_token_count": 2},
            "stage_summary": {
                "stage_0": {"task_id": "stage0", "miner_id": "m0", "activation_count": 1},
                "stage_1": {"task_id": "stage1", "miner_id": "m1", "baseline_match": True, "decoded_tokens_match": True},
            },
            "stage_assignment": {"required_distinct_stage_miners": False},
            "observability": {"requeue_summary": {"enabled": False}, "processes": []},
            "diagnosis_codes": [
                "micro_llm_sharded_ready",
                "stage_0_accepted",
                "stage_1_accepted",
                "activation_transport_ready",
                "baseline_match",
                "decoded_tokens_match",
            ],
            "safety": {
                "read_only": True,
                "redaction_ok": True,
                "raw_activation_redacted": True,
                "generated_text_redacted": True,
                "generated_token_ids_redacted": True,
                "not_production": True,
            },
            "limitations": [],
        }
        report = pack.attach_micro_user_guidance(report, args)

        markdown = pack.render_markdown(report)

        self.assertIn("## Review", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("## Not Completed", markdown)
        self.assertIn("- none", markdown)


if __name__ == "__main__":
    unittest.main()
