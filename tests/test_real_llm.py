from __future__ import annotations

import unittest
from unittest import mock

from crowdtensor import real_llm


class RealLlmTests(unittest.TestCase):
    def setUp(self) -> None:
        real_llm.clear_real_llm_runtime_caches()

    def test_default_model_metadata_only_cuda_artifact_does_not_require_hf_dependencies(self) -> None:
        with mock.patch.object(real_llm, "missing_hf_dependencies", return_value=["transformers"]):
            artifact = real_llm.inspect_real_llm_artifact(
                model_id=real_llm.DEFAULT_MODEL_ID,
                backend=real_llm.BACKEND_CUDA,
                require_runtime=False,
            )

        self.assertEqual(artifact["schema"], real_llm.REAL_LLM_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(artifact["model_id"], real_llm.DEFAULT_MODEL_ID)
        self.assertEqual(artifact["backend"], real_llm.BACKEND_CUDA)
        self.assertEqual(artifact["metadata_source"], "built_in_default_model_manifest")
        self.assertTrue(artifact["metadata_only"])
        self.assertEqual(artifact["num_hidden_layers"], 2)
        self.assertEqual(artifact["split_index"], 1)
        self.assertEqual(artifact["cuda_runtime"]["diagnosis_codes"], ["cuda_runtime_deferred_to_miner"])
        self.assertTrue(str(artifact["artifact_hash"]).startswith("sha256:"))

    def test_sharded_spec_preserves_generation_controls(self) -> None:
        artifact = {
            "schema": real_llm.REAL_LLM_ARTIFACT_SCHEMA_VERSION,
            "artifact_hash": "sha256:test",
            "model_id": real_llm.DEFAULT_MODEL_ID,
            "backend": real_llm.BACKEND_CPU,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
        }

        spec = real_llm.real_llm_sharded_inference_spec_for(
            "task-1",
            "miner-1",
            artifact,
            request_count=1,
            prompt_texts=["The future of open AI is"],
            max_new_tokens=16,
            generation_step=3,
        )

        self.assertEqual(spec["max_new_tokens"], 16)
        self.assertEqual(spec["generation_step"], 3)
        self.assertEqual(spec["requests"][0]["max_new_tokens"], 16)
        self.assertEqual(spec["requests"][0]["generation_step"], 0)
        self.assertEqual(spec["requests"][0]["generated_token_ids"], [])
        self.assertEqual(spec["requests"][0]["generated_text"], "")

    def test_gpt2_block_call_supports_new_past_key_values_signature(self) -> None:
        class NewBlock:
            def __init__(self) -> None:
                self.kwargs = {}

            def forward(self, hidden_states, past_key_values=None, use_cache=False):  # noqa: ANN001, ANN202
                self.kwargs = {"past_key_values": past_key_values, "use_cache": use_cache}
                return hidden_states, ("key", "value")

            __call__ = forward

        block = NewBlock()
        output = real_llm._call_gpt2_block(  # noqa: SLF001
            block,
            "hidden",
            dynamic_cache="cache",
            layer_past="legacy",
            use_cache=True,
        )

        self.assertEqual(output[0], "hidden")
        self.assertEqual(block.kwargs, {"past_key_values": "cache", "use_cache": True})

    def test_gpt2_block_call_supports_legacy_layer_past_signature(self) -> None:
        class LegacyBlock:
            def __init__(self) -> None:
                self.kwargs = {}

            def forward(self, hidden_states, layer_past=None, use_cache=False):  # noqa: ANN001, ANN202
                self.kwargs = {"layer_past": layer_past, "use_cache": use_cache}
                return hidden_states, ("key", "value")

            __call__ = forward

        block = LegacyBlock()
        output = real_llm._call_gpt2_block(  # noqa: SLF001
            block,
            "hidden",
            dynamic_cache="cache",
            layer_past="legacy",
            use_cache=True,
        )

        self.assertEqual(output[0], "hidden")
        self.assertEqual(block.kwargs, {"layer_past": "legacy", "use_cache": True})

    def test_stage_local_runtime_preserves_batched_activation_shape(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        artifact = real_llm.inspect_real_llm_artifact(
            model_id=real_llm.DEFAULT_MODEL_ID,
            backend=real_llm.BACKEND_CPU,
            require_runtime=True,
        )
        artifact["partition_mode"] = real_llm.PARTITION_MODE_STAGE_LOCAL
        artifact["artifact_hash"] = "sha256:test-stage-local-artifact"
        stage0_spec = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task",
            "stage0-miner",
            artifact,
            request_count=1,
            prompt_texts=["CrowdTensor routes home CPU"],
            session_id="session-test",
            stage_id=0,
            max_new_tokens=2,
            generation_step=0,
        )

        stage0_result = real_llm.run_real_llm_sharded_inference(stage0_spec)
        activation = dict(stage0_result["activation_results"][0])
        self.assertEqual(len(activation["hidden_shape"]), 3)
        self.assertEqual(activation["hidden_shape"][0], 1)

        squeezed_activation = dict(activation)
        squeezed_activation["hidden_state"] = activation["hidden_state"][0]
        squeezed_activation["hidden_shape"] = activation["hidden_shape"][1:]
        squeezed_activation["activation_hash"] = real_llm._activation_hash(squeezed_activation)  # noqa: SLF001
        stage1_spec = real_llm.real_llm_sharded_inference_spec_for(
            "stage1-task",
            "stage1-miner",
            artifact,
            request_count=1,
            session_id="session-test",
            stage_id=1,
            parent_task_id="stage0-task",
            max_new_tokens=2,
            generation_step=0,
            activation_results=[squeezed_activation],
        )

        stage1_result = real_llm.run_real_llm_sharded_inference(stage1_spec)
        self.assertTrue(stage1_result["baseline_match"])
        self.assertTrue(stage1_result["decoded_tokens_match"])
        self.assertEqual(stage1_result["generated_token_count"], 1)

    def test_stage0_uses_generated_token_ids_as_continuation(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        artifact = real_llm.inspect_real_llm_artifact(
            model_id=real_llm.DEFAULT_MODEL_ID,
            backend=real_llm.BACKEND_CPU,
            require_runtime=True,
        )
        artifact["artifact_hash"] = "sha256:test-token-continuation-artifact"
        tokenizer, _, _ = real_llm._load_model_and_tokenizer(  # noqa: SLF001
            real_llm.DEFAULT_MODEL_ID,
            backend=real_llm.BACKEND_CPU,
        )
        prompt = "CrowdTensor routes home CPU"
        first_token = 42
        prompt_len = int(real_llm._tokenize_prompt(tokenizer, prompt).shape[1])  # noqa: SLF001
        stage0_spec = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task",
            "stage0-miner",
            artifact,
            request_count=1,
            requests=[
                {
                    "request_id": "req-1",
                    "prompt": prompt,
                    "prompt_hash": real_llm._prompt_hash(prompt),  # noqa: SLF001
                    "max_new_tokens": 2,
                    "generated_token_ids": [first_token],
                    "generated_text": " continuation text is not re-tokenized",
                    "generation_step": 1,
                }
            ],
            session_id="session-test",
            stage_id=0,
            max_new_tokens=2,
            generation_step=1,
        )

        stage0_result = real_llm.run_real_llm_sharded_inference(stage0_spec)
        activation = stage0_result["activation_results"][0]

        self.assertEqual(activation["generated_token_ids"], [first_token])
        self.assertEqual(activation["prompt_token_count"], prompt_len)
        self.assertEqual(activation["generated_prefix_token_count"], 1)
        self.assertEqual(activation["input_token_count"], prompt_len + 1)
        self.assertEqual(activation["input_ids"][-1], first_token)
        self.assertTrue(activation["token_continuation_ready"])
        self.assertTrue(activation["kv_cache_ready"])

    def test_stage0_kv_cache_hits_on_incremental_token_continuation(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        artifact = real_llm.inspect_real_llm_artifact(
            model_id=real_llm.DEFAULT_MODEL_ID,
            backend=real_llm.BACKEND_CPU,
            require_runtime=True,
        )
        artifact["artifact_hash"] = "sha256:test-stage0-kv-cache-artifact"
        prompt = "CrowdTensor routes home CPU"
        first_token = 42
        stage0_step0 = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task-0",
            "stage0-miner",
            artifact,
            request_count=1,
            prompt_texts=[prompt],
            session_id="session-kv-test",
            stage_id=0,
            max_new_tokens=2,
            generation_step=0,
        )
        step0 = real_llm.run_real_llm_sharded_inference(stage0_step0)
        activation0 = step0["activation_results"][0]
        self.assertTrue(activation0["kv_cache_ready"])
        self.assertFalse(activation0["kv_cache_hit"])

        stage0_step1 = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task-1",
            "stage0-miner",
            artifact,
            request_count=1,
            requests=[
                {
                    "request_id": "req-1",
                    "prompt": prompt,
                    "prompt_hash": real_llm._prompt_hash(prompt),  # noqa: SLF001
                    "max_new_tokens": 2,
                    "generated_token_ids": [first_token],
                    "generated_text": " cached",
                    "generation_step": 1,
                }
            ],
            session_id="session-kv-test",
            stage_id=0,
            max_new_tokens=2,
            generation_step=1,
        )
        cached = real_llm.run_real_llm_sharded_inference(stage0_step1)
        cached_activation = cached["activation_results"][0]

        real_llm.clear_real_llm_runtime_caches()
        uncached = real_llm.run_real_llm_sharded_inference(stage0_step1)
        uncached_activation = uncached["activation_results"][0]

        self.assertTrue(cached_activation["kv_cache_hit"])
        self.assertEqual(cached_activation["generated_prefix_token_count"], 1)
        self.assertEqual(cached_activation["kv_cache_tokens_before"], cached_activation["input_token_count"] - 1)
        self.assertEqual(cached_activation["hidden_shape"], uncached_activation["hidden_shape"])
        self.assertEqual(cached_activation["activation_hash"], uncached_activation["activation_hash"])
        self.assertEqual(cached_activation["hidden_state"], uncached_activation["hidden_state"])

    def test_stage0_kv_cache_misses_when_miner_changes(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        artifact = real_llm.inspect_real_llm_artifact(
            model_id=real_llm.DEFAULT_MODEL_ID,
            backend=real_llm.BACKEND_CPU,
            require_runtime=True,
        )
        artifact["artifact_hash"] = "sha256:test-stage0-kv-cache-miner-artifact"
        prompt = "CrowdTensor routes home CPU"
        stage0_step0 = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task-0",
            "stage0-miner-a",
            artifact,
            request_count=1,
            prompt_texts=[prompt],
            session_id="session-kv-miner-test",
            stage_id=0,
            max_new_tokens=2,
            generation_step=0,
        )
        real_llm.run_real_llm_sharded_inference(stage0_step0)

        stage0_step1 = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task-1",
            "stage0-miner-b",
            artifact,
            request_count=1,
            requests=[
                {
                    "request_id": "req-1",
                    "prompt": prompt,
                    "prompt_hash": real_llm._prompt_hash(prompt),  # noqa: SLF001
                    "max_new_tokens": 2,
                    "generated_token_ids": [42],
                    "generated_text": " cached",
                    "generation_step": 1,
                }
            ],
            session_id="session-kv-miner-test",
            stage_id=0,
            max_new_tokens=2,
            generation_step=1,
        )
        result = real_llm.run_real_llm_sharded_inference(stage0_step1)
        activation = result["activation_results"][0]

        self.assertTrue(activation["kv_cache_ready"])
        self.assertFalse(activation["kv_cache_hit"])

    def test_stage1_kv_cache_hits_on_incremental_token_continuation(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        artifact = real_llm.inspect_real_llm_artifact(
            model_id=real_llm.DEFAULT_MODEL_ID,
            backend=real_llm.BACKEND_CPU,
            require_runtime=True,
        )
        artifact["artifact_hash"] = "sha256:test-stage1-kv-cache-artifact"
        prompt = "CrowdTensor routes home CPU"
        session_id = "session-stage1-kv-test"
        stage0_step0 = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task-0",
            "stage0-miner",
            artifact,
            request_count=1,
            prompt_texts=[prompt],
            session_id=session_id,
            stage_id=0,
            max_new_tokens=2,
            generation_step=0,
        )
        activation0 = real_llm.run_real_llm_sharded_inference(stage0_step0)["activation_results"][0]
        stage1_step0 = real_llm.real_llm_sharded_inference_spec_for(
            "stage1-task-0",
            "stage1-miner",
            artifact,
            request_count=1,
            session_id=session_id,
            stage_id=1,
            parent_task_id="stage0-task-0",
            max_new_tokens=2,
            generation_step=0,
            activation_results=[activation0],
        )
        result0 = real_llm.run_real_llm_sharded_inference(stage1_step0)["inference_result"]
        self.assertTrue(result0["kv_cache_ready"])
        self.assertFalse(result0["kv_cache_hit"])

        first_token = int(result0["next_token_id"])
        stage0_step1 = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task-1",
            "stage0-miner",
            artifact,
            request_count=1,
            requests=[
                {
                    "request_id": "req-1",
                    "prompt": prompt,
                    "prompt_hash": real_llm._prompt_hash(prompt),  # noqa: SLF001
                    "max_new_tokens": 2,
                    "generated_token_ids": [first_token],
                    "generated_text": result0["next_token_text"],
                    "generation_step": 1,
                }
            ],
            session_id=session_id,
            stage_id=0,
            max_new_tokens=2,
            generation_step=1,
        )
        activation1 = real_llm.run_real_llm_sharded_inference(stage0_step1)["activation_results"][0]
        stage1_step1 = real_llm.real_llm_sharded_inference_spec_for(
            "stage1-task-1",
            "stage1-miner",
            artifact,
            request_count=1,
            session_id=session_id,
            stage_id=1,
            parent_task_id="stage0-task-1",
            max_new_tokens=2,
            generation_step=1,
            activation_results=[activation1],
        )
        cached = real_llm.run_real_llm_sharded_inference(stage1_step1)["inference_result"]

        real_llm.clear_real_llm_runtime_caches()
        uncached = real_llm.run_real_llm_sharded_inference(stage1_step1)["inference_result"]

        self.assertTrue(cached["kv_cache_ready"])
        self.assertTrue(cached["kv_cache_hit"])
        self.assertEqual(cached["kv_cache_tokens_before"], cached["kv_cache_tokens_after"] - 1)
        self.assertTrue(cached["baseline_match"])
        self.assertEqual(cached["next_token_id"], uncached["next_token_id"])
        self.assertEqual(cached["generated_token_ids"], uncached["generated_token_ids"])
        self.assertEqual(cached["generated_text_hash"], uncached["generated_text_hash"])
        self.assertEqual(cached["output_hash"], uncached["output_hash"])

    def test_stage1_kv_cache_misses_when_miner_changes(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        artifact = real_llm.inspect_real_llm_artifact(
            model_id=real_llm.DEFAULT_MODEL_ID,
            backend=real_llm.BACKEND_CPU,
            require_runtime=True,
        )
        artifact["artifact_hash"] = "sha256:test-stage1-kv-cache-miner-artifact"
        prompt = "CrowdTensor routes home CPU"
        session_id = "session-stage1-kv-miner-test"
        stage0_step0 = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task-0",
            "stage0-miner",
            artifact,
            request_count=1,
            prompt_texts=[prompt],
            session_id=session_id,
            stage_id=0,
            max_new_tokens=2,
            generation_step=0,
        )
        activation0 = real_llm.run_real_llm_sharded_inference(stage0_step0)["activation_results"][0]
        stage1_step0 = real_llm.real_llm_sharded_inference_spec_for(
            "stage1-task-0",
            "stage1-miner-a",
            artifact,
            request_count=1,
            session_id=session_id,
            stage_id=1,
            parent_task_id="stage0-task-0",
            max_new_tokens=2,
            generation_step=0,
            activation_results=[activation0],
        )
        result0 = real_llm.run_real_llm_sharded_inference(stage1_step0)["inference_result"]

        stage0_step1 = real_llm.real_llm_sharded_inference_spec_for(
            "stage0-task-1",
            "stage0-miner",
            artifact,
            request_count=1,
            requests=[
                {
                    "request_id": "req-1",
                    "prompt": prompt,
                    "prompt_hash": real_llm._prompt_hash(prompt),  # noqa: SLF001
                    "max_new_tokens": 2,
                    "generated_token_ids": [int(result0["next_token_id"])],
                    "generated_text": result0["next_token_text"],
                    "generation_step": 1,
                }
            ],
            session_id=session_id,
            stage_id=0,
            max_new_tokens=2,
            generation_step=1,
        )
        activation1 = real_llm.run_real_llm_sharded_inference(stage0_step1)["activation_results"][0]
        stage1_step1 = real_llm.real_llm_sharded_inference_spec_for(
            "stage1-task-1",
            "stage1-miner-b",
            artifact,
            request_count=1,
            session_id=session_id,
            stage_id=1,
            parent_task_id="stage0-task-1",
            max_new_tokens=2,
            generation_step=1,
            activation_results=[activation1],
        )
        result = real_llm.run_real_llm_sharded_inference(stage1_step1)["inference_result"]

        self.assertTrue(result["kv_cache_ready"])
        self.assertFalse(result["kv_cache_hit"])


if __name__ == "__main__":
    unittest.main()
