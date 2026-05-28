from __future__ import annotations

import unittest

import scripts.real_llm_sharded_inference_evidence_pack as pack


class RealLlmShardedInferenceEvidencePackTests(unittest.TestCase):
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
                        "real_llm_sharded_stage_capabilities": [pack.REAL_LLM_SHARDED_STAGE0_CAPABILITY],
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
                        "real_llm_sharded_stage_capabilities": [pack.REAL_LLM_SHARDED_STAGE1_CAPABILITY],
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


if __name__ == "__main__":
    unittest.main()
