from __future__ import annotations

import unittest
from unittest import mock

from crowdtensor import real_llm


class RealLlmTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
