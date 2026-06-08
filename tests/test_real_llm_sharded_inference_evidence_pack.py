from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import scripts.real_llm_sharded_inference_evidence_pack as pack


class RealLlmShardedInferenceEvidencePackTests(unittest.TestCase):
    def test_parse_args_reads_prompt_texts_file(self) -> None:
        prompt_file = Path(tempfile.mkdtemp(prefix="crowdtensor_real_llm_prompts_test_")) / "prompts.txt"
        prompt_file.write_text("first, comma prompt\nsecond prompt\n", encoding="utf-8")

        args = pack.parse_args(["--prompt-texts-file", str(prompt_file)])

        self.assertEqual(args.prompt_texts, "")
        self.assertEqual(args.prompt_texts_file, str(prompt_file))
        self.assertEqual(args.prompt_texts_list, ["first, comma prompt", "second prompt"])
        self.assertEqual(pack.prompt_list_from_args(args), ["first, comma prompt", "second prompt"])

    def test_parse_args_rejects_inline_and_file_prompt_batch(self) -> None:
        prompt_file = Path(tempfile.mkdtemp(prefix="crowdtensor_real_llm_prompts_test_")) / "prompts.txt"
        prompt_file.write_text("first prompt\nsecond prompt\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as raised:
            pack.parse_args([
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(
            str(raised.exception),
            "real_llm_sharded_inference_evidence accepts either --prompt-texts or --prompt-texts-file, not both",
        )

    def test_report_redacts_generated_text_and_token_payloads(self) -> None:
        args = type("Args", (), {
            "base_url": "http://127.0.0.1:9880",
            "require_distinct_stage_miners": True,
            "stage_mode": "split",
            "real_llm_partition_mode": "full",
        })()
        session = {
            "schema": "real_llm_sharded_session_v1",
            "session_id": "real-session-test",
            "stage_count": 2,
            "stage_0_task_id": "task-stage-0",
            "stage_1_task_id": "task-stage-1",
            "request_count": 1,
            "artifact_hash": "sha256:artifact",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cpu",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
        }
        state = {
            "model": {"global_step": 0},
            "model_updates": 0,
            "tasks": [
                {
                    "task_id": "task-stage-0",
                    "status": "completed",
                    "miner_id": "real-stage0",
                    "attempt": 1,
                    "workload_type": pack.WORKLOAD_TYPE,
                    "workload_metadata": {"session_id": "real-session-test", "stage_id": 0},
                    "capabilities": {
                        "real_llm_sharded_stage_role": "stage0",
                        "real_llm_sharded_stage_capabilities": [pack.REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY],
                        "real_llm_runtime": {"adapter_kind": "hf_transformers_cuda"},
                    },
                    "validation": {
                        "code": "ok",
                        "activation_transport_ready": True,
                        "activation_count": 1,
                        "activation_bytes": 128,
                        "activation_hashes": ["sha256:activation"],
                        "artifact_hash": "sha256:artifact",
                        "real_llm_artifact_ready": True,
                    },
                    "metrics": {"elapsed_ms": 1.0},
                },
                {
                    "task_id": "task-stage-1",
                    "status": "completed",
                    "miner_id": "real-stage1",
                    "attempt": 1,
                    "workload_type": pack.WORKLOAD_TYPE,
                    "workload_metadata": {"session_id": "real-session-test", "stage_id": 1},
                    "capabilities": {
                        "real_llm_sharded_stage_role": "stage1",
                        "real_llm_sharded_stage_capabilities": [pack.REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY],
                        "real_llm_runtime": {"adapter_kind": "hf_transformers_cuda"},
                    },
                    "validation": {
                        "code": "ok",
                        "activation_transport_ready": True,
                        "baseline_match": True,
                        "decoded_tokens_match": True,
                        "request_count": 1,
                        "generated_token_ids": [16046],
                        "generated_text": " stairs",
                        "request_trace": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:prompt",
                                "activation_hash": "sha256:activation",
                                "output_hash": "sha256:output",
                                "baseline_match": True,
                                "next_token_id": 16046,
                                "next_token_text": " stairs",
                            }
                        ],
                        "artifact_hash": "sha256:artifact",
                        "real_llm_artifact_ready": True,
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
                {"task_id": "task-stage-0", "model_updated": False},
                {"task_id": "task-stage-1", "model_updated": False},
            ],
        )
        encoded = str(report)

        self.assertTrue(report["ok"], report)
        self.assertNotIn(" stairs", encoded)
        self.assertNotIn("generated_token_ids", report["stage_summary"]["stage_1"])
        self.assertNotIn("generated_text", report["stage_summary"]["stage_1"])
        self.assertEqual(report["stage_summary"]["stage_1"]["generated_token_summary"], {"count": 1, "redacted": True})
        self.assertTrue(report["stage_summary"]["stage_1"]["request_trace"][0]["next_token_redacted"])
        self.assertTrue(report["safety"]["generated_text_redacted"])
        self.assertTrue(report["safety"]["generated_token_ids_redacted"])
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["review_summary"]["schema"], "real_llm_sharded_review_summary_v1")
        self.assertFalse(report["not_completed"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertEqual(report["answer_scope"]["scope_state"], "hash-only")
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertIn("inspect real LLM sharded evidence", report["recommended_next_command"]["label"])
        self.assertGreaterEqual(len(report["next_commands"]), 3)

    def test_stage_local_partition_evidence_is_required_and_summarized(self) -> None:
        args = type("Args", (), {
            "base_url": "http://127.0.0.1:9880",
            "require_distinct_stage_miners": True,
            "stage_mode": "split",
            "real_llm_partition_mode": "stage_local",
        })()
        session = {
            "schema": "real_llm_sharded_session_v1",
            "session_id": "real-session-stage-local",
            "stage_count": 2,
            "stage_0_task_id": "task-stage-0",
            "stage_1_task_id": "task-stage-1",
            "request_count": 1,
            "artifact_hash": "sha256:artifact",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cuda",
            "partition_mode": "stage_local",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
        }
        common_validation = {
            "code": "ok",
            "partition_mode": "stage_local",
            "stage_local_partition_ready": True,
            "partition_parameter_split_valid": True,
            "full_model_parameter_count": 4096,
            "stage_parameter_count": 1024,
            "stage_parameter_fraction": 0.25,
            "device_parameter_count": 1024,
            "stage_cpu_partition_ready": False,
            "stage_gpu_memory_reduced": True,
            "real_llm_artifact_ready": True,
            "artifact_hash": "sha256:artifact",
        }
        state = {
            "model": {"global_step": 0},
            "model_updates": 0,
            "tasks": [
                {
                    "task_id": "task-stage-0",
                    "status": "completed",
                    "miner_id": "real-stage0",
                    "attempt": 1,
                    "workload_type": pack.WORKLOAD_TYPE,
                    "workload_metadata": {"session_id": "real-session-stage-local", "stage_id": 0},
                    "capabilities": {
                        "real_llm_sharded_stage_role": "stage0",
                        "real_llm_sharded_stage_capabilities": [pack.REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY],
                        "real_llm_runtime": {"adapter_kind": "hf_transformers_cuda"},
                    },
                    "validation": {
                        **common_validation,
                        "activation_transport_ready": True,
                        "activation_count": 1,
                        "activation_bytes": 128,
                        "activation_hashes": ["sha256:activation"],
                        "stage_layer_range": [0, 1],
                        "stage0_partition_loaded": True,
                    },
                    "metrics": {"elapsed_ms": 1.0},
                },
                {
                    "task_id": "task-stage-1",
                    "status": "completed",
                    "miner_id": "real-stage1",
                    "attempt": 1,
                    "workload_type": pack.WORKLOAD_TYPE,
                    "workload_metadata": {"session_id": "real-session-stage-local", "stage_id": 1},
                    "capabilities": {
                        "real_llm_sharded_stage_role": "stage1",
                        "real_llm_sharded_stage_capabilities": [pack.REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY],
                        "real_llm_runtime": {"adapter_kind": "hf_transformers_cuda"},
                    },
                    "validation": {
                        **common_validation,
                        "activation_transport_ready": True,
                        "baseline_match": True,
                        "decoded_tokens_match": True,
                        "request_count": 1,
                        "stage_layer_range": [1, 2],
                        "stage1_partition_loaded": True,
                        "baseline_device": "cpu",
                        "request_trace": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:prompt",
                                "activation_hash": "sha256:activation",
                                "output_hash": "sha256:output",
                                "baseline_match": True,
                                "next_token_id": 16046,
                                "next_token_text": " stairs",
                            }
                        ],
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
                {"task_id": "task-stage-0", "model_updated": False},
                {"task_id": "task-stage-1", "model_updated": False},
            ],
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["artifact"]["partition_mode"], "stage_local")
        self.assertTrue(report["artifact"]["stage_local_partition_ready"])
        self.assertIn("stage_local_partition_ready", report["diagnosis_codes"])
        self.assertIn("stage0_partition_loaded", report["diagnosis_codes"])
        self.assertIn("stage1_partition_loaded", report["diagnosis_codes"])
        self.assertTrue(report["safety"]["stage_local_partition"])
        self.assertEqual(report["stage_summary"]["stage_1"]["baseline_device"], "cpu")
        self.assertFalse(report["not_completed"])

    def test_render_markdown_includes_review_scope_and_next_steps(self) -> None:
        args = type("Args", (), {
            "base_url": "http://127.0.0.1:9880",
            "require_distinct_stage_miners": False,
            "stage_mode": "both",
            "real_llm_partition_mode": "full",
        })()
        report = {
            "schema": pack.SCHEMA,
            "ok": True,
            "workload_type": pack.WORKLOAD_TYPE,
            "session": {
                "session_id": "real-session-md",
                "stage_count": 2,
                "model_id": "sshleifer/tiny-gpt2",
                "partition_mode": "full",
            },
            "artifact": {"backend": "hf_transformers_cpu", "partition_mode": "full"},
            "generation": {"max_new_tokens": 1, "generated_token_count": 1},
            "stage_summary": {
                "stage_0": {"task_id": "stage0", "miner_id": "m0", "activation_count": 1},
                "stage_1": {"task_id": "stage1", "miner_id": "m1", "baseline_match": True},
            },
            "stage_assignment": {"required_distinct_stage_miners": False},
            "observability": {"requeue_summary": {"enabled": False}, "processes": []},
            "diagnosis_codes": [
                "real_llm_sharded_ready",
                "stage_0_accepted",
                "stage_1_accepted",
                "activation_transport_ready",
                "real_llm_artifact_ready",
                "baseline_match",
                "decoded_tokens_match",
                "generation_complete",
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
        report = pack.attach_user_guidance(report, args)

        markdown = pack.render_markdown(report)

        self.assertIn("## Review", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("## Not Completed", markdown)
        self.assertIn("- none", markdown)

    def test_support_bundle_payload_is_public_safe(self) -> None:
        args = type("Args", (), {
            "base_url": "http://127.0.0.1:9880",
            "require_distinct_stage_miners": False,
            "stage_mode": "both",
            "real_llm_partition_mode": "full",
        })()
        report = {
            "schema": pack.SCHEMA,
            "ok": False,
            "diagnosis_codes": ["stage_0_missing"],
            "artifact": {},
            "generation": {"max_new_tokens": 1, "generated_token_count": 0},
            "stage_assignment": {"required_distinct_stage_miners": False},
            "observability": {"requeue_summary": {"enabled": False}, "processes": []},
            "safety": {
                "read_only": True,
                "redaction_ok": True,
                "raw_activation_redacted": True,
                "generated_text_redacted": True,
                "generated_token_ids_redacted": True,
                "not_production": True,
            },
        }
        report = pack.attach_user_guidance(report, args)
        bundle = pack.support_bundle_payload(report)
        encoded = json.dumps(bundle, sort_keys=True)

        self.assertEqual(bundle["schema"], "real_llm_sharded_support_bundle_v1")
        self.assertNotIn("generated_text\":", encoded)
        self.assertNotIn("generated_token_ids\":", encoded)


if __name__ == "__main__":
    unittest.main()
