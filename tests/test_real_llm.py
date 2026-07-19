from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

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
        self.assertEqual(artifact["execution_family"], real_llm.EXECUTION_FAMILY_GPT2)
        self.assertTrue(artifact["execution_support"]["current_stage_split_supported"])
        self.assertFalse(artifact["execution_support"]["large_model_sharded_execution_ready"])
        self.assertIn(
            "real_llm_true_partial_weight_loading_missing",
            artifact["execution_support"]["large_model_blockers"],
        )
        self.assertEqual(artifact["cuda_runtime"]["diagnosis_codes"], ["cuda_runtime_deferred_to_miner"])
        self.assertTrue(str(artifact["artifact_hash"]).startswith("sha256:"))

    def test_large_llama_like_metadata_without_weight_index_reports_partial_runtime_gap(self) -> None:
        summary = real_llm.real_llm_execution_support_summary(
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 28,
                "hidden_size": 3584,
                "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            }
        )

        self.assertEqual(summary["execution_family"], real_llm.EXECUTION_FAMILY_LLAMA_LIKE)
        self.assertTrue(summary["current_stage_split_supported"])
        self.assertFalse(summary["large_model_sharded_execution_ready"])
        self.assertTrue(summary["large_model_candidate"])
        self.assertEqual(summary["stage_local_load_strategy"], "full_model_cpu_load_then_stage_module_device_move")
        self.assertIn("real_llm_true_partial_weight_loading_missing", summary["large_model_blockers"])
        self.assertIn("real_llm_llama_like_runtime_execution_missing", summary["large_model_blockers"])
        self.assertIn("real_llm_llama_like_stage_runtime_adapter_ready", summary["diagnosis_codes"])

    def test_llama_like_partial_weight_plan_maps_stage_owned_safetensor_keys(self) -> None:
        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
            "model.layers.3.mlp.down_proj.weight": "model-00004-of-00004.safetensors",
            "model.norm.weight": "model-00004-of-00004.safetensors",
            "lm_head.weight": "model-00004-of-00004.safetensors",
        }
        metadata = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 4,
            "hidden_size": 3584,
            "split_index": 2,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "weight_map": weight_map,
        }

        plan = real_llm.real_llm_partial_weight_loading_plan(metadata)

        self.assertEqual(plan["schema"], real_llm.REAL_LLM_PARTIAL_WEIGHT_PLAN_SCHEMA_VERSION)
        self.assertTrue(plan["ready"])
        self.assertFalse(plan["runtime_execution_ready"])
        self.assertEqual(plan["execution_family"], real_llm.EXECUTION_FAMILY_LLAMA_LIKE)
        self.assertEqual(plan["weight_file_count"], 4)
        self.assertEqual(plan["stage_plans"][0]["stage_layer_range"], [0, 2])
        self.assertEqual(plan["stage_plans"][1]["stage_layer_range"], [2, 4])
        self.assertTrue(plan["stage_plans"][0]["loads_only_stage_weight_keys"])
        self.assertTrue(plan["stage_plans"][1]["loads_only_stage_weight_keys"])
        self.assertIn("model-00001-of-00004.safetensors", plan["stage_plans"][0]["assigned_weight_files"])
        self.assertIn("model-00004-of-00004.safetensors", plan["stage_plans"][1]["assigned_weight_files"])

        summary = real_llm.real_llm_execution_support_summary(metadata)
        self.assertTrue(summary["partial_weight_loading_plan_ready"])
        self.assertFalse(summary["true_partial_weight_loading_ready"])
        self.assertFalse(summary["large_model_sharded_execution_ready"])
        self.assertEqual(summary["stage_local_load_strategy"], "stage_weight_index_selective_load_plan")
        self.assertNotIn("real_llm_true_partial_weight_loading_missing", summary["large_model_blockers"])
        self.assertIn("real_llm_llama_like_runtime_execution_missing", summary["large_model_blockers"])
        self.assertIn("real_llm_llama_like_partial_weight_plan_ready", summary["diagnosis_codes"])

    def test_n_stage_partition_plan_maps_stage_owned_safetensor_keys(self) -> None:
        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.4.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.5.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.6.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
            "model.layers.7.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
            "model.norm.weight": "model-00004-of-00004.safetensors",
            "lm_head.weight": "model-00004-of-00004.safetensors",
        }
        metadata = {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 8,
            "hidden_size": 5120,
            "stage_count": 4,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "weight_map": weight_map,
        }

        plan = real_llm.real_llm_n_stage_partition_plan(metadata, stage_count=4)

        self.assertEqual(plan["schema"], real_llm.REAL_LLM_N_STAGE_PARTITION_PLAN_SCHEMA_VERSION)
        self.assertTrue(plan["ready"])
        self.assertFalse(plan["runtime_execution_ready"])
        self.assertEqual(plan["stage_count"], 4)
        self.assertTrue(plan["stage_ranges_valid"])
        self.assertEqual(plan["covered_decoder_layer_count"], 8)
        self.assertEqual([stage["stage_layer_range"] for stage in plan["stage_plans"]], [[0, 2], [2, 4], [4, 6], [6, 8]])
        self.assertIn("model.embed_tokens.", plan["stage_plans"][0]["expected_key_prefixes"])
        self.assertNotIn("lm_head.", plan["stage_plans"][0]["expected_key_prefixes"])
        self.assertIn("model.norm.", plan["stage_plans"][3]["expected_key_prefixes"])
        self.assertIn("lm_head.", plan["stage_plans"][3]["expected_key_prefixes"])
        self.assertTrue(all(stage["loads_only_stage_weight_keys"] for stage in plan["stage_plans"]))
        self.assertIn("real_llm_n_stage_partition_abstraction_ready", plan["diagnosis_codes"])
        self.assertFalse(plan["unassigned_weight_key_count"])

        summary = real_llm.real_llm_execution_support_summary(metadata)
        self.assertTrue(summary["n_stage_partition_plan_ready"])
        self.assertEqual(summary["n_stage_partition_plan"]["stage_count"], 4)
        self.assertIn("real_llm_n_stage_partition_abstraction_ready", summary["diagnosis_codes"])
        self.assertFalse(summary["large_model_sharded_execution_ready"])

    def test_14b_metadata_estimates_weight_budget_for_n_stage_plan(self) -> None:
        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            **{
                f"model.layers.{index}.self_attn.q_proj.weight": f"model-{(index // 10) + 1:05d}-of-00004.safetensors"
                for index in range(48)
            },
            "model.norm.weight": "model-00004-of-00004.safetensors",
            "lm_head.weight": "model-00004-of-00004.safetensors",
        }
        metadata = {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 48,
            "hidden_size": 5120,
            "stage_count": 4,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "weight_map": weight_map,
        }

        summary = real_llm.real_llm_execution_support_summary(metadata)
        plan = summary["n_stage_partition_plan"]

        self.assertEqual(summary["parameter_count_estimate"], 14_700_000_000)
        self.assertTrue(summary["large_model_candidate"])
        self.assertEqual(plan["stage_count"], 4)
        self.assertTrue(plan["ready"])
        self.assertEqual(plan["covered_decoder_layer_count"], 48)
        self.assertEqual(
            [stage["stage_layer_range"] for stage in plan["stage_plans"]],
            [[0, 12], [12, 24], [24, 36], [36, 48]],
        )
        self.assertEqual(plan["estimated_weight_bytes_fp32"], 58_800_000_000)
        self.assertLess(
            max(stage["estimated_stage_weight_bytes_fp32"] for stage in plan["stage_plans"]),
            summary["estimated_weight_bytes_fp32"],
        )
        self.assertIn("real_llm_n_stage_partition_abstraction_ready", summary["diagnosis_codes"])
        self.assertFalse(summary["large_model_sharded_execution_ready"])

    def test_stage_selective_safetensors_loader_materializes_only_stage_owned_keys(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        import torch  # type: ignore
        from safetensors.torch import save_file  # type: ignore

        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        }
        metadata = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 4,
            "hidden_size": 8,
            "split_index": 2,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "weight_map": weight_map,
        }
        with tempfile.TemporaryDirectory(prefix="crowdtensor_stage_weights_") as tmp:
            root = Path(tmp)
            save_file(
                {
                    "model.embed_tokens.weight": torch.ones((2, 2)),
                    "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 2.0),
                    "model.layers.1.self_attn.q_proj.weight": torch.full((2, 2), 3.0),
                    "model.layers.2.self_attn.q_proj.weight": torch.full((2, 2), 4.0),
                },
                root / "model-00001-of-00002.safetensors",
            )
            save_file(
                {
                    "model.layers.2.self_attn.q_proj.weight": torch.full((2, 2), 5.0),
                    "model.layers.3.self_attn.q_proj.weight": torch.full((2, 2), 6.0),
                    "model.norm.weight": torch.ones((2,)),
                    "lm_head.weight": torch.full((2, 2), 7.0),
                    "model.layers.1.self_attn.q_proj.weight": torch.full((2, 2), 8.0),
                },
                root / "model-00002-of-00002.safetensors",
            )

            stage0_tensors, stage0 = real_llm._load_stage_selective_safetensors(  # noqa: SLF001
                metadata,
                stage_id=0,
                weight_root=root,
            )
            stage1 = real_llm.real_llm_stage_selective_weight_load_summary(
                metadata,
                stage_id=1,
                weight_root=root,
            )

        self.assertEqual(stage0["schema"], real_llm.REAL_LLM_STAGE_SELECTIVE_WEIGHT_LOAD_SCHEMA_VERSION)
        self.assertTrue(stage0["ready"])
        self.assertTrue(stage0["stage_selective_tensor_materialization_ready"])
        self.assertTrue(stage0["loads_only_stage_weight_keys"])
        self.assertFalse(stage0["cross_stage_weight_keys_loaded"])
        self.assertEqual(stage0["assigned_weight_key_count"], 3)
        self.assertEqual(stage0["loaded_weight_key_count"], 3)
        self.assertGreaterEqual(stage0["candidate_file_key_count"], 4)
        self.assertGreater(stage0["skipped_non_stage_weight_key_count"], 0)
        self.assertEqual(stage0["missing_weight_file_count"], 0)
        self.assertEqual(stage0["missing_weight_key_count"], 0)
        self.assertEqual(set(stage0_tensors), {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
        })
        self.assertNotIn("model.layers.2.self_attn.q_proj.weight", stage0_tensors)
        self.assertGreater(stage0["loaded_tensor_bytes"], 0)

        self.assertTrue(stage1["ready"])
        self.assertEqual(stage1["assigned_weight_key_count"], 4)
        self.assertEqual(stage1["loaded_weight_key_count"], 4)
        self.assertTrue(stage1["loads_only_stage_weight_keys"])
        self.assertFalse(stage1["cross_stage_weight_keys_loaded"])
        self.assertIn(
            "real_llm_stage_selective_weight_materialization_ready",
            stage1["diagnosis_codes"],
        )

        support = real_llm.real_llm_execution_support_summary({
            **metadata,
            "stage_selective_weight_load_summaries": [stage0, stage1],
        })
        self.assertTrue(support["partial_weight_loading_plan_ready"])
        self.assertTrue(support["partial_weight_tensor_materialization_ready"])
        self.assertTrue(support["true_partial_weight_loading_ready"])
        self.assertFalse(support["partial_weight_runtime_execution_ready"])
        self.assertFalse(support["large_model_sharded_execution_ready"])
        self.assertEqual(
            support["stage_local_load_strategy"],
            "stage_weight_index_selective_tensor_materialization",
        )

    def test_stage_selective_tensors_apply_to_stage_owned_model_state(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        import torch  # type: ignore
        from safetensors.torch import save_file  # type: ignore
        from transformers import LlamaConfig, LlamaForCausalLM  # type: ignore

        config = LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        model = LlamaForCausalLM(config)
        state = model.state_dict()
        stage0_keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
        ]
        stage1_keys = [
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        weight_map = {
            **{key: "model-00001-of-00002.safetensors" for key in stage0_keys},
            **{key: "model-00002-of-00002.safetensors" for key in stage1_keys},
        }
        metadata = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 4,
            "hidden_size": 8,
            "split_index": 2,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "weight_map": weight_map,
        }
        with tempfile.TemporaryDirectory(prefix="crowdtensor_stage_apply_") as tmp:
            root = Path(tmp)
            save_file(
                {
                    key: torch.full_like(state[key], float(index + 1))
                    for index, key in enumerate(stage0_keys)
                },
                root / "model-00001-of-00002.safetensors",
            )
            save_file(
                {
                    key: torch.full_like(state[key], float(index + 10))
                    for index, key in enumerate(stage1_keys)
                },
                root / "model-00002-of-00002.safetensors",
            )
            stage0_tensors, load_summary = real_llm._load_stage_selective_safetensors(  # noqa: SLF001
                metadata,
                stage_id=0,
                weight_root=root,
            )
            apply_summary = real_llm._apply_stage_selective_tensors_to_model(  # noqa: SLF001
                model,
                stage0_tensors,
                metadata,
                stage_id=0,
            )

        self.assertTrue(load_summary["ready"])
        self.assertTrue(apply_summary["ready"])
        self.assertTrue(apply_summary["stage_selective_tensor_application_ready"])
        self.assertEqual(apply_summary["applied_weight_key_count"], len(stage0_keys))
        self.assertEqual(apply_summary["missing_assigned_weight_key_count"], 0)
        self.assertEqual(apply_summary["unknown_model_key_count"], 0)
        self.assertEqual(apply_summary["shape_mismatch_count"], 0)
        self.assertFalse(apply_summary["cross_stage_weight_keys_loaded"])
        self.assertTrue(apply_summary["loads_only_stage_weight_keys"])
        self.assertIn(
            "real_llm_stage_selective_weight_application_ready",
            apply_summary["diagnosis_codes"],
        )
        for index, key in enumerate(stage0_keys):
            self.assertTrue(torch.equal(model.state_dict()[key], torch.full_like(state[key], float(index + 1))))
        self.assertFalse(torch.equal(model.state_dict()[stage1_keys[0]], torch.full_like(state[stage1_keys[0]], 10.0)))

        support = real_llm.real_llm_execution_support_summary({
            **metadata,
            "stage_selective_weight_load_summaries": [load_summary],
            "stage_selective_weight_application_summaries": [apply_summary],
        })
        self.assertTrue(support["partial_weight_tensor_materialization_ready"])
        self.assertTrue(support["partial_weight_tensor_application_ready"])
        self.assertTrue(support["true_partial_weight_loading_ready"])
        self.assertFalse(support["partial_weight_runtime_execution_ready"])
        self.assertEqual(
            support["stage_local_load_strategy"],
            "stage_weight_index_selective_tensor_application",
        )

    def test_stage_selective_runtime_smoke_matches_baseline(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        import torch  # type: ignore
        from safetensors.torch import save_file  # type: ignore
        from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # type: ignore

        config = LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        seed_model = LlamaForCausalLM(config)
        state = seed_model.state_dict()
        stage0_keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
        ]
        stage1_keys = [
            "model.layers.2.self_attn.q_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        metadata = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 4,
            "hidden_size": 8,
            "split_index": 2,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "weight_map": {
                **{key: "model-00001-of-00002.safetensors" for key in stage0_keys},
                **{key: "model-00002-of-00002.safetensors" for key in stage1_keys},
            },
        }
        with tempfile.TemporaryDirectory(prefix="crowdtensor_stage_runtime_") as tmp:
            root = Path(tmp)
            save_file(
                {key: torch.full_like(state[key], float(index + 1)) for index, key in enumerate(stage0_keys)},
                root / "model-00001-of-00002.safetensors",
            )
            save_file(
                {key: torch.full_like(state[key], float(index + 10)) for index, key in enumerate(stage1_keys)},
                root / "model-00002-of-00002.safetensors",
            )
            stage0_model = LlamaForCausalLM(config)
            stage1_model = LlamaForCausalLM(config)
            baseline_model = LlamaForCausalLM(config)
            stage0_tensors, stage0_load = real_llm._load_stage_selective_safetensors(  # noqa: SLF001
                metadata,
                stage_id=0,
                weight_root=root,
            )
            stage1_tensors, stage1_load = real_llm._load_stage_selective_safetensors(  # noqa: SLF001
                metadata,
                stage_id=1,
                weight_root=root,
            )
            stage0_apply = real_llm._apply_stage_selective_tensors_to_model(  # noqa: SLF001
                stage0_model,
                stage0_tensors,
                metadata,
                stage_id=0,
            )
            stage1_apply = real_llm._apply_stage_selective_tensors_to_model(  # noqa: SLF001
                stage1_model,
                stage1_tensors,
                metadata,
                stage_id=1,
            )
            real_llm._apply_stage_selective_tensors_to_model(  # noqa: SLF001
                baseline_model,
                stage0_tensors,
                metadata,
                stage_id=0,
            )
            real_llm._apply_stage_selective_tensors_to_model(  # noqa: SLF001
                baseline_model,
                stage1_tensors,
                metadata,
                stage_id=1,
            )
        tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "CrowdTensor": 1, "routes": 2, "home": 3, "GPU": 4}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")

        runtime = real_llm.run_stage_selective_runtime_smoke(
            tokenizer=hf_tokenizer,
            stage0_model=stage0_model,
            stage1_model=stage1_model,
            baseline_model=baseline_model,
            metadata=metadata,
            prompt="CrowdTensor routes home GPU",
        )

        self.assertTrue(stage0_load["ready"])
        self.assertTrue(stage1_load["ready"])
        self.assertTrue(stage0_apply["ready"])
        self.assertTrue(stage1_apply["ready"])
        self.assertEqual(runtime["schema"], real_llm.REAL_LLM_STAGE_SELECTIVE_RUNTIME_SCHEMA_VERSION)
        self.assertTrue(runtime["ready"])
        self.assertTrue(runtime["stage_selective_runtime_execution_ready"])
        self.assertTrue(runtime["activation_transport_ready"])
        self.assertTrue(runtime["baseline_match"])
        self.assertTrue(runtime["decoded_tokens_match"])
        self.assertEqual(runtime["generated_token_count"], 1)
        self.assertFalse(runtime["large_model_validation"])
        self.assertFalse(runtime["kaggle_runtime_validation"])
        self.assertFalse(runtime["raw_prompt_public"])
        self.assertFalse(runtime["raw_generated_text_public"])
        self.assertFalse(runtime["generated_token_ids_public"])
        self.assertFalse(runtime["activation_public"])

        support = real_llm.real_llm_execution_support_summary({
            **metadata,
            "stage_selective_weight_load_summaries": [stage0_load, stage1_load],
            "stage_selective_weight_application_summaries": [stage0_apply, stage1_apply],
            "stage_selective_runtime": runtime,
        })
        self.assertTrue(support["partial_weight_runtime_execution_ready"])
        self.assertTrue(support["partial_weight_tensor_application_ready"])
        self.assertFalse(support["large_model_sharded_execution_ready"])
        self.assertEqual(
            support["stage_local_load_strategy"],
            "stage_weight_index_selective_runtime_execution",
        )

    def test_stage_selective_runtime_smoke_can_skip_baseline(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast  # type: ignore

        config = GPT2Config(
            n_layer=4,
            n_embd=16,
            n_head=4,
            n_positions=32,
            n_ctx=32,
            vocab_size=16,
            bos_token_id=0,
            eos_token_id=0,
        )
        full_model = GPT2LMHeadModel(config)
        metadata = {
            "model_id": "local-gpt2-stage-smoke",
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "num_hidden_layers": 4,
            "hidden_size": 16,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "split_index": 2,
            "weight_map": {key: "model.safetensors" for key in full_model.state_dict()},
        }
        stage0_model = GPT2LMHeadModel(config)
        stage1_model = GPT2LMHeadModel(config)
        stage0_model.load_state_dict(full_model.state_dict())
        stage1_model.load_state_dict(full_model.state_dict())
        tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "CrowdTensor": 1, "routes": 2}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")

        runtime = real_llm.run_stage_selective_runtime_smoke(
            tokenizer=hf_tokenizer,
            stage0_model=stage0_model,
            stage1_model=stage1_model,
            baseline_model=None,
            metadata=metadata,
            prompt="CrowdTensor routes",
            baseline_required=False,
        )

        self.assertTrue(runtime["ready"])
        self.assertTrue(runtime["stage_selective_runtime_execution_ready"])
        self.assertTrue(runtime["activation_transport_ready"])
        self.assertTrue(runtime["baseline_validation_skipped"])
        self.assertFalse(runtime["baseline_match"])
        self.assertEqual(runtime["generated_token_count"], 1)

    def test_stage_selective_hf_runtime_smoke_uses_meta_stage_models(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        import torch  # type: ignore
        from safetensors.torch import save_file  # type: ignore
        from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # type: ignore

        config = LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        seed_model = LlamaForCausalLM(config)
        state = seed_model.state_dict()
        stage0_keys = [
            "model.embed_tokens.weight",
            *[key for key in state if key.startswith("model.layers.0.")],
            *[key for key in state if key.startswith("model.layers.1.")],
        ]
        stage1_keys = [
            *[key for key in state if key.startswith("model.layers.2.")],
            *[key for key in state if key.startswith("model.layers.3.")],
            "model.norm.weight",
            "lm_head.weight",
        ]

        with tempfile.TemporaryDirectory(prefix="crowdtensor_stage_hf_runtime_") as tmp:
            root = Path(tmp)
            config.to_json_file(root / "config.json")
            tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "CrowdTensor": 1, "routes": 2, "home": 3, "GPU": 4}, unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
            hf_tokenizer.save_pretrained(root)
            save_file(
                {key: state[key].detach().clone() for key in stage0_keys},
                root / "model-00001-of-00002.safetensors",
            )
            save_file(
                {key: state[key].detach().clone() for key in stage1_keys},
                root / "model-00002-of-00002.safetensors",
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps({
                    "metadata": {"total_size": 1},
                    "weight_map": {
                        **{key: "model-00001-of-00002.safetensors" for key in stage0_keys},
                        **{key: "model-00002-of-00002.safetensors" for key in stage1_keys},
                    },
                }),
                encoding="utf-8",
            )

            runtime = real_llm.run_stage_selective_hf_runtime_smoke(
                model_id=str(root),
                prompt="CrowdTensor routes home GPU",
            )

        self.assertEqual(runtime["schema"], real_llm.REAL_LLM_STAGE_SELECTIVE_HF_RUNTIME_SCHEMA_VERSION)
        self.assertTrue(runtime["ready"])
        self.assertTrue(runtime["stage_selective_runtime_execution_ready"])
        self.assertEqual(runtime["runtime_execution_scope"], "real_hf_stage_selective_runtime")
        self.assertEqual(runtime["stage_devices"]["stage0"], "cpu")
        self.assertEqual(runtime["stage_devices"]["stage1"], "cpu")
        self.assertFalse(runtime["multi_device_stage_assignment"])
        self.assertTrue(runtime["activation_transport_ready"])
        self.assertTrue(runtime["baseline_match"])
        self.assertTrue(runtime["decoded_tokens_match"])
        self.assertEqual(runtime["generated_token_count"], 1)
        self.assertTrue(runtime["stage0_remaining_meta_parameter_count"] > 0)
        self.assertTrue(runtime["stage1_remaining_meta_parameter_count"] > 0)
        self.assertEqual(runtime["runtime_buffers"]["stage0"]["remaining_meta_buffer_count"], 0)
        self.assertEqual(runtime["runtime_buffers"]["stage1"]["remaining_meta_buffer_count"], 0)
        self.assertTrue(runtime["stage_summaries"][0]["loads_only_stage_weight_keys"])
        self.assertTrue(runtime["stage_application_summaries"][0]["loads_only_stage_weight_keys"])
        self.assertTrue(runtime["model_execution_support"]["partial_weight_runtime_execution_ready"])
        self.assertFalse(runtime["large_model_validation"])
        self.assertFalse(runtime["kaggle_runtime_validation"])
        self.assertFalse(runtime["local_weight_root_public"])
        self.assertFalse(runtime["raw_prompt_public"])
        self.assertFalse(runtime["raw_generated_text_public"])
        self.assertFalse(runtime["generated_token_ids_public"])
        self.assertFalse(runtime["activation_public"])

    def test_stage_selective_hf_awq_runtime_smoke_replaces_quantized_linears(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        import torch  # type: ignore
        from safetensors.torch import save_file  # type: ignore
        from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # type: ignore

        def pack_awq(weight: torch.Tensor, *, group_size: int = 4) -> dict[str, torch.Tensor]:
            source = weight.detach().t().contiguous().to(dtype=torch.int32)
            in_features, out_features = source.shape
            assert in_features % group_size == 0
            assert out_features % 8 == 0
            qweight = torch.zeros((in_features, out_features // 8), dtype=torch.int32)
            order = [0, 2, 4, 6, 1, 3, 5, 7]
            clamped = source.clamp(0, 15)
            for col in range(out_features // 8):
                for bit_index, source_index in enumerate(order):
                    qweight[:, col] |= clamped[:, col * 8 + source_index] << (bit_index * 4)
            qzeros = torch.zeros((in_features // group_size, out_features // 8), dtype=torch.int32)
            scales = torch.ones((in_features // group_size, out_features), dtype=torch.float16)
            return {"qweight": qweight, "qzeros": qzeros, "scales": scales}

        config = LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        config.quantization_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 4,
            "version": "gemm",
            "zero_point": True,
        }
        seed_model = LlamaForCausalLM(config)
        with torch.no_grad():
            for module in seed_model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.fill_(1.0)
                    if module.bias is not None:
                        module.bias.zero_()
            seed_model.model.embed_tokens.weight.fill_(1.0)
            seed_model.model.norm.weight.fill_(1.0)
            seed_model.lm_head.weight.fill_(1.0)

        state = seed_model.state_dict()
        awq_state: dict[str, torch.Tensor] = {}
        weight_map: dict[str, str] = {}
        quantized_modules = [
            name[:-len(".weight")]
            for name in state
            if name.endswith(".weight")
            and (
                ".self_attn.q_proj." in name
                or ".self_attn.k_proj." in name
                or ".self_attn.v_proj." in name
                or ".self_attn.o_proj." in name
                or ".mlp.gate_proj." in name
                or ".mlp.up_proj." in name
                or ".mlp.down_proj." in name
            )
        ]
        for module_name in quantized_modules:
            packed = pack_awq(state[f"{module_name}.weight"])
            for suffix, tensor in packed.items():
                awq_state[f"{module_name}.{suffix}"] = tensor
        for key, tensor in state.items():
            if any(key == f"{module}.weight" for module in quantized_modules):
                continue
            if key.endswith("rotary_emb.inv_freq"):
                continue
            awq_state[key] = tensor.detach().clone()

        stage0_keys = [
            key
            for key in awq_state
            if key.startswith("model.embed_tokens.")
            or key.startswith("model.layers.0.")
            or key.startswith("model.layers.1.")
        ]
        stage1_keys = [
            key
            for key in awq_state
            if key.startswith("model.layers.2.")
            or key.startswith("model.layers.3.")
            or key.startswith("model.norm.")
            or key.startswith("lm_head.")
        ]
        with tempfile.TemporaryDirectory(prefix="crowdtensor_awq_stage_hf_runtime_") as tmp:
            root = Path(tmp)
            config.to_json_file(root / "config.json")
            tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "CrowdTensor": 1, "routes": 2}, unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
            hf_tokenizer.save_pretrained(root)
            save_file({key: awq_state[key] for key in stage0_keys}, root / "model-00001-of-00002.safetensors")
            save_file({key: awq_state[key] for key in stage1_keys}, root / "model-00002-of-00002.safetensors")
            weight_map.update({key: "model-00001-of-00002.safetensors" for key in stage0_keys})
            weight_map.update({key: "model-00002-of-00002.safetensors" for key in stage1_keys})
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
                encoding="utf-8",
            )

            runtime = real_llm.run_stage_selective_hf_runtime_smoke(
                model_id=str(root),
                prompt="CrowdTensor routes",
                baseline_required=False,
            )

        self.assertTrue(runtime["ready"])
        self.assertTrue(runtime["stage_selective_runtime_execution_ready"])
        self.assertTrue(runtime["awq_stage_runtime_ready"])
        self.assertTrue(runtime["awq_stage_preparation"]["stage0"]["awq_stage_model_prepared"])
        self.assertTrue(runtime["awq_stage_preparation"]["stage1"]["awq_stage_model_prepared"])
        self.assertGreater(runtime["awq_stage_preparation"]["stage0"]["awq_linear_replacement_count"], 0)
        self.assertTrue(runtime["activation_transport_ready"])
        self.assertTrue(runtime["baseline_validation_skipped"])
        self.assertEqual(runtime["generated_token_count"], 1)
        self.assertFalse(runtime["activation_public"])

    def test_stage_selective_single_safetensors_supports_tied_lm_head(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        import torch  # type: ignore
        from safetensors.torch import save_file  # type: ignore
        from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # type: ignore

        config = LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )
        model = LlamaForCausalLM(config)
        state = model.state_dict()
        single_file_state = {
            key: tensor.detach().clone()
            for key, tensor in state.items()
            if key != "lm_head.weight" and not key.endswith("rotary_emb.inv_freq")
        }

        with tempfile.TemporaryDirectory(prefix="crowdtensor_tied_lm_head_") as tmp:
            root = Path(tmp)
            config.to_json_file(root / "config.json")
            tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "CrowdTensor": 1, "routes": 2}, unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
            hf_tokenizer.save_pretrained(root)
            save_file(single_file_state, root / "model.safetensors")

            config_loaded, metadata, weight_root = real_llm._stage_selective_hf_metadata(  # noqa: SLF001
                model_id=str(root),
                backend=real_llm.BACKEND_CPU,
            )
            self.assertTrue(config_loaded.tie_word_embeddings)
            self.assertEqual(metadata["tied_weight_aliases"]["lm_head.weight"], "model.embed_tokens.weight")
            self.assertEqual(metadata["weight_file_count"], 1)
            stage1_tensors, stage1_load = real_llm._load_stage_selective_safetensors(  # noqa: SLF001
                metadata,
                stage_id=1,
                weight_root=weight_root,
            )

        self.assertTrue(stage1_load["ready"])
        self.assertIn("lm_head.weight", stage1_tensors)
        self.assertTrue(torch.equal(stage1_tensors["lm_head.weight"], state["model.embed_tokens.weight"]))

    def test_stage_selective_hf_metadata_downloads_only_stage_owned_shards(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        from transformers import LlamaConfig  # type: ignore

        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
            "model.norm.weight": "model-00004-of-00004.safetensors",
            "lm_head.weight": "model-00004-of-00004.safetensors",
        }

        with tempfile.TemporaryDirectory(prefix="crowdtensor_stage_hf_meta_") as tmp:
            root = Path(tmp)
            config = LlamaConfig(
                vocab_size=8,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=16,
                tie_word_embeddings=False,
            )
            config.to_json_file(root / "config.json")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 1000}, "weight_map": weight_map}),
                encoding="utf-8",
            )
            requested: list[str] = []

            def fake_cached_file(_model_id: str, filename: str, **_kwargs: object) -> str:
                requested.append(filename)
                path = root / filename
                path.write_bytes(b"stage-shard-placeholder")
                return str(path)

            with mock.patch("transformers.utils.cached_file", side_effect=fake_cached_file):
                _config, metadata, weight_root = real_llm._stage_selective_hf_metadata(  # noqa: SLF001
                    model_id=str(root),
                    backend=real_llm.BACKEND_CPU,
                    stage_id=0,
                )

        self.assertEqual(weight_root, root)
        self.assertEqual(
            requested,
            ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors"],
        )
        self.assertEqual(metadata["stage_weight_download_scope"], "stage_owned_weight_files")
        self.assertEqual(metadata["stage_weight_download_stage_id"], 0)
        self.assertEqual(metadata["stage_weight_download_file_count"], 2)
        self.assertTrue(metadata["stage_weight_downloads_only_stage_files"])

    def test_n_stage_selective_safetensors_loader_materializes_only_stage_owned_keys(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        import torch  # type: ignore
        from safetensors.torch import save_file  # type: ignore

        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.4.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.5.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.6.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
            "model.layers.7.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
            "model.norm.weight": "model-00004-of-00004.safetensors",
            "lm_head.weight": "model-00004-of-00004.safetensors",
        }
        metadata = {
            "model_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 8,
            "hidden_size": 5120,
            "stage_count": 4,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "weight_map": weight_map,
        }
        with tempfile.TemporaryDirectory(prefix="crowdtensor_n_stage_weights_") as tmp:
            root = Path(tmp)
            save_file(
                {
                    "model.embed_tokens.weight": torch.ones((2, 2)),
                    "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 2.0),
                    "model.layers.1.self_attn.q_proj.weight": torch.full((2, 2), 3.0),
                    "model.layers.2.self_attn.q_proj.weight": torch.full((2, 2), 4.0),
                },
                root / "model-00001-of-00004.safetensors",
            )
            save_file(
                {
                    "model.layers.2.self_attn.q_proj.weight": torch.full((2, 2), 5.0),
                    "model.layers.3.self_attn.q_proj.weight": torch.full((2, 2), 6.0),
                    "model.layers.5.self_attn.q_proj.weight": torch.full((2, 2), 7.0),
                },
                root / "model-00002-of-00004.safetensors",
            )
            save_file(
                {
                    "model.layers.4.self_attn.q_proj.weight": torch.full((2, 2), 8.0),
                    "model.layers.5.self_attn.q_proj.weight": torch.full((2, 2), 9.0),
                },
                root / "model-00003-of-00004.safetensors",
            )
            save_file(
                {
                    "model.layers.6.self_attn.q_proj.weight": torch.full((2, 2), 10.0),
                    "model.layers.7.self_attn.q_proj.weight": torch.full((2, 2), 11.0),
                    "model.norm.weight": torch.ones((2,)),
                    "lm_head.weight": torch.full((2, 2), 12.0),
                },
                root / "model-00004-of-00004.safetensors",
            )

            stage2_tensors, stage2 = real_llm._load_stage_selective_safetensors(  # noqa: SLF001
                metadata,
                stage_id=2,
                stage_count=4,
                weight_root=root,
            )

        self.assertTrue(stage2["ready"])
        self.assertEqual(stage2["stage_id"], 2)
        self.assertEqual(stage2["stage_count"], 4)
        self.assertEqual(stage2["stage_layer_range"], [4, 6])
        self.assertEqual(stage2["assigned_weight_files"], ["model-00003-of-00004.safetensors"])
        self.assertEqual(sorted(stage2_tensors), [
            "model.layers.4.self_attn.q_proj.weight",
            "model.layers.5.self_attn.q_proj.weight",
        ])
        self.assertEqual(stage2["candidate_file_key_count"], 2)
        self.assertFalse(stage2["cross_stage_weight_keys_loaded"])
        self.assertTrue(stage2["loads_only_stage_weight_keys"])

    def test_stage_selective_hf_metadata_downloads_only_n_stage_owned_shards(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        from transformers import LlamaConfig  # type: ignore

        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.4.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.5.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.6.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
            "model.layers.7.self_attn.q_proj.weight": "model-00004-of-00004.safetensors",
            "model.norm.weight": "model-00004-of-00004.safetensors",
            "lm_head.weight": "model-00004-of-00004.safetensors",
        }

        with tempfile.TemporaryDirectory(prefix="crowdtensor_n_stage_hf_meta_") as tmp:
            root = Path(tmp)
            config = LlamaConfig(
                vocab_size=8,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=8,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=16,
                tie_word_embeddings=False,
            )
            config.to_json_file(root / "config.json")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 1000}, "weight_map": weight_map}),
                encoding="utf-8",
            )
            requested: list[str] = []

            def fake_cached_file(_model_id: str, filename: str, **_kwargs: object) -> str:
                requested.append(filename)
                path = root / filename
                path.write_bytes(b"stage-shard-placeholder")
                return str(path)

            with mock.patch("transformers.utils.cached_file", side_effect=fake_cached_file):
                _config, metadata, weight_root = real_llm._stage_selective_hf_metadata(  # noqa: SLF001
                    model_id=str(root),
                    backend=real_llm.BACKEND_CPU,
                    stage_id=2,
                    stage_count=4,
                )

        self.assertEqual(weight_root, root)
        self.assertEqual(requested, ["model-00003-of-00004.safetensors"])
        self.assertEqual(metadata["stage_count"], 4)
        self.assertEqual(metadata["stage_weight_download_scope"], "stage_owned_weight_files")
        self.assertEqual(metadata["stage_weight_download_stage_id"], 2)
        self.assertEqual(metadata["stage_weight_download_file_count"], 1)
        self.assertTrue(metadata["stage_weight_downloads_only_stage_files"])

    def test_partial_weight_plan_without_weight_index_keeps_true_partial_blocker(self) -> None:
        metadata = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
        }

        plan = real_llm.real_llm_partial_weight_loading_plan(metadata)
        summary = real_llm.real_llm_execution_support_summary(metadata)

        self.assertFalse(plan["ready"])
        self.assertIn("real_llm_partial_weight_plan_weight_map_missing", plan["blockers"])
        self.assertFalse(summary["partial_weight_loading_plan_ready"])
        self.assertIn("real_llm_true_partial_weight_loading_missing", summary["large_model_blockers"])
        self.assertIn("real_llm_llama_like_runtime_execution_missing", summary["large_model_blockers"])

    def test_gpt2_model_id_variant_stays_supported_without_config_metadata(self) -> None:
        summary = real_llm.real_llm_execution_support_summary({"model_id": "distilgpt2"})

        self.assertEqual(summary["execution_family"], real_llm.EXECUTION_FAMILY_GPT2)
        self.assertTrue(summary["current_stage_split_supported"])
        self.assertIn("real_llm_current_stage_split_supported", summary["diagnosis_codes"])

    def test_gpt2_xl_metadata_marks_small_tier_candidate_not_large_ready(self) -> None:
        summary = real_llm.real_llm_execution_support_summary(
            {
                "model_id": "gpt2-xl",
                "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            }
        )

        self.assertEqual(summary["execution_family"], real_llm.EXECUTION_FAMILY_GPT2)
        self.assertTrue(summary["current_stage_split_supported"])
        self.assertTrue(summary["small_tier_candidate"])
        self.assertTrue(summary["kaggle_small_tier_supported_by_current_split"])
        self.assertEqual(summary["parameter_count_estimate"], 1_558_000_000)
        self.assertEqual(summary["estimated_weight_bytes_fp32"], 6_232_000_000)
        self.assertFalse(summary["large_model_candidate"])
        self.assertFalse(summary["large_model_sharded_execution_ready"])
        self.assertIn("real_llm_1b_3b_small_tier_candidate_detected", summary["diagnosis_codes"])

    def test_non_gpt2_workload_fails_before_runtime_load(self) -> None:
        artifact = {
            "schema": real_llm.REAL_LLM_ARTIFACT_SCHEMA_VERSION,
            "artifact_hash": "sha256:qwen-test",
            "model_id": "bert-base-uncased",
            "backend": real_llm.BACKEND_CPU,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "model_type": "bert",
            "architectures": ["BertModel"],
            "split_index": 6,
            "num_hidden_layers": 12,
            "hidden_size": 768,
        }
        spec = real_llm.real_llm_sharded_inference_spec_for(
            "task-qwen",
            "miner-qwen",
            artifact,
            request_count=1,
            stage_id=0,
        )

        with mock.patch.object(real_llm, "_load_model_and_tokenizer") as load_model:
            with self.assertRaisesRegex(ValueError, "execution_family=unsupported_hf_causal_lm"):
                real_llm.run_real_llm_sharded_inference(spec)

        load_model.assert_not_called()

    def test_llama_like_stage_runtime_adapter_runs_tiny_random_llama(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))

        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        try:
            artifact = real_llm.inspect_real_llm_artifact(
                model_id=model_id,
                backend=real_llm.BACKEND_CPU,
                require_runtime=True,
            )
        except Exception as exc:  # pragma: no cover - depends on optional HF cache/network
            self.skipTest(f"tiny random Llama unavailable: {exc}")
        artifact["partition_mode"] = real_llm.PARTITION_MODE_STAGE_LOCAL
        artifact["artifact_hash"] = "sha256:test-llama-like-stage-runtime"
        self.assertEqual(artifact["execution_family"], real_llm.EXECUTION_FAMILY_LLAMA_LIKE)
        self.assertTrue(artifact["execution_support"]["current_stage_split_supported"])

        stage0_spec = real_llm.real_llm_sharded_inference_spec_for(
            "llama-stage0-task",
            "llama-stage0-miner",
            artifact,
            request_count=1,
            prompt_texts=["CrowdTensor routes home GPU"],
            session_id="llama-session-test",
            stage_id=0,
            max_new_tokens=1,
            generation_step=0,
        )
        stage0_result = real_llm.run_real_llm_sharded_inference(stage0_spec)
        activation = stage0_result["activation_results"][0]
        self.assertEqual(stage0_result["execution_family"], real_llm.EXECUTION_FAMILY_LLAMA_LIKE)
        self.assertTrue(stage0_result["activation_transport_ready"])
        self.assertEqual(activation["kv_cache_disabled_reason"], "llama_like_stage_cache_not_implemented")

        stage1_spec = real_llm.real_llm_sharded_inference_spec_for(
            "llama-stage1-task",
            "llama-stage1-miner",
            artifact,
            request_count=1,
            session_id="llama-session-test",
            stage_id=1,
            parent_task_id="llama-stage0-task",
            max_new_tokens=1,
            generation_step=0,
            activation_results=[activation],
        )
        stage1_result = real_llm.run_real_llm_sharded_inference(stage1_spec)

        self.assertEqual(stage1_result["execution_family"], real_llm.EXECUTION_FAMILY_LLAMA_LIKE)
        self.assertTrue(stage1_result["baseline_match"])
        self.assertTrue(stage1_result["decoded_tokens_match"])
        self.assertEqual(stage1_result["generated_token_count"], 1)

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

    def test_stage_selective_remote_validation_accepts_baseline_skipped_stage1(self) -> None:
        artifact = {
            "schema": real_llm.REAL_LLM_ARTIFACT_SCHEMA_VERSION,
            "artifact_hash": "sha256:test-stage-selective-remote-artifact",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "backend": real_llm.BACKEND_CUDA,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "execution_mode": real_llm.EXECUTION_MODE_STAGE_SELECTIVE_HF,
            "split_index": 14,
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "vocab_size": 152064,
        }
        activation = {
            "schema_version": real_llm.REAL_LLM_ACTIVATION_SCHEMA_VERSION,
            "session_id": "session-stage-selective-remote",
            "request_id": "req-1",
            "prompt_hash": "sha256:prompt",
            "model_id": artifact["model_id"],
            "artifact_hash": artifact["artifact_hash"],
            "split_index": 14,
            "generation_step": 0,
            "max_new_tokens": 1,
            "generated_token_ids": [],
            "generated_text": "",
            "input_ids": [1, 2, 3],
            "position_ids": [0, 1, 2],
            "hidden_shape": [1, 3, 4],
            "hidden_state": [[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]],
        }
        activation["activation_hash"] = real_llm._activation_hash(activation)  # noqa: SLF001
        spec = real_llm.real_llm_sharded_inference_spec_for(
            "stage1-task",
            "stage1-miner",
            artifact,
            request_count=1,
            session_id=activation["session_id"],
            stage_id=1,
            parent_task_id="stage0-task",
            max_new_tokens=1,
            generation_step=0,
            activation_results=[activation],
            execution_mode=real_llm.EXECUTION_MODE_STAGE_SELECTIVE_HF,
        )
        result_row = {
            "request_id": "req-1",
            "prompt_hash": activation["prompt_hash"],
            "model_id": artifact["model_id"],
            "artifact_hash": artifact["artifact_hash"],
            "activation_hash": activation["activation_hash"],
            "generation_step": 0,
            "max_new_tokens": 1,
            "next_token_id": 151643,
            "next_token_text": "",
            "baseline_next_token_id": None,
            "baseline_next_token_text": "",
            "generated_token_ids": [151643],
            "generated_token_count": 1,
            "generated_text": "",
            "generated_text_hash": real_llm._generated_text_hash(""),  # noqa: SLF001
            "baseline_match": False,
            "baseline_validation_skipped": True,
        }
        result_row["output_hash"] = real_llm._output_hash(result_row)  # noqa: SLF001
        sharded_result = {
            "schema_version": real_llm.REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
            "type": real_llm.WORKLOAD_TYPE,
            "session_id": activation["session_id"],
            "stage_id": 1,
            "stage_count": 2,
            "model_id": artifact["model_id"],
            "backend": real_llm.BACKEND_CUDA,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "execution_mode": real_llm.EXECUTION_MODE_STAGE_SELECTIVE_HF,
            "artifact_schema": real_llm.REAL_LLM_ARTIFACT_SCHEMA_VERSION,
            "artifact_hash": artifact["artifact_hash"],
            "split_index": 14,
            "max_new_tokens": 1,
            "generation_step": 0,
            "request_count": 1,
            "activation_count": 1,
            "activation_hashes": [activation["activation_hash"]],
            "activation_transport_ready": True,
            "stage_layer_range": [14, 28],
            "stage_parameter_count": 1_000,
            "full_model_parameter_count": 2_000,
            "stage_parameter_fraction": 0.5,
            "device_parameter_count": 1_000,
            "partition_parameter_split_valid": True,
            "stage_local_partition_ready": True,
            "stage1_partition_loaded": True,
            "stage_selective_hf_runtime_ready": True,
            "stage_selective_runtime_execution_ready": True,
            "stage_selective_weight_load_ready": True,
            "stage_selective_weight_application_ready": True,
            "stage_selective_weight_load": {
                "loaded_tensor_bytes": 2048,
                "stage_weight_download_scope": "stage_owned_weight_files",
                "stage_weight_download_file_count": 3,
                "stage_weight_downloads_only_stage_files": True,
                "public_safe": True,
            },
            "stage_selective_weight_application": {
                "applied_tensor_bytes": 2048,
                "public_safe": True,
            },
            "runtime_buffer_materialization_ready": True,
            "baseline_match": False,
            "baseline_validation_skipped": True,
            "decoded_tokens_match": True,
            "generated_token_ids": [151643],
            "generated_token_count": 1,
            "generated_text": "",
            "generated_text_hash": result_row["generated_text_hash"],
            "inference_result": result_row,
            "inference_results": [result_row],
            "real_llm_artifact_ready": True,
        }

        validation = real_llm.validate_real_llm_sharded_inference(
            sharded_result,
            expected_spec=spec,
            replay_runtime=False,
        )

        self.assertTrue(validation["accepted"])
        self.assertFalse(validation["baseline_match"])
        self.assertTrue(validation["baseline_validation_skipped"])
        self.assertTrue(validation["stage_selective_remote_validation"])
        self.assertTrue(validation["decoded_tokens_match"])
        self.assertEqual(validation["execution_mode"], real_llm.EXECUTION_MODE_STAGE_SELECTIVE_HF)
        self.assertEqual(validation["stage_selective_weight_load"]["loaded_tensor_bytes"], 2048)
        self.assertEqual(validation["stage_selective_weight_load"]["stage_weight_download_scope"], "stage_owned_weight_files")
        self.assertTrue(validation["stage_selective_weight_load"]["stage_weight_downloads_only_stage_files"])
        self.assertEqual(validation["stage_selective_weight_application"]["applied_tensor_bytes"], 2048)

    def test_stage_selective_remote_validation_accepts_second_generated_token(self) -> None:
        artifact = {
            "schema": real_llm.REAL_LLM_ARTIFACT_SCHEMA_VERSION,
            "artifact_hash": "sha256:test-stage-selective-remote-artifact",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "backend": real_llm.BACKEND_CUDA,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "execution_mode": real_llm.EXECUTION_MODE_STAGE_SELECTIVE_HF,
            "split_index": 14,
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "vocab_size": 152064,
        }
        activation = {
            "schema_version": real_llm.REAL_LLM_ACTIVATION_SCHEMA_VERSION,
            "session_id": "session-stage-selective-remote-multi-token",
            "request_id": "req-1",
            "prompt_hash": "sha256:prompt",
            "model_id": artifact["model_id"],
            "artifact_hash": artifact["artifact_hash"],
            "split_index": 14,
            "generation_step": 1,
            "max_new_tokens": 2,
            "generated_token_ids": [151643],
            "generated_text": "",
            "input_ids": [1, 2, 3, 151643],
            "position_ids": [0, 1, 2, 3],
            "hidden_shape": [1, 4, 4],
            "hidden_state": [
                [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7]]
            ],
        }
        activation["activation_hash"] = real_llm._activation_hash(activation)  # noqa: SLF001
        spec = real_llm.real_llm_sharded_inference_spec_for(
            "stage1-task-step-1",
            "stage1-miner",
            artifact,
            request_count=1,
            session_id=activation["session_id"],
            stage_id=1,
            parent_task_id="stage0-task-step-1",
            max_new_tokens=2,
            generation_step=1,
            activation_results=[activation],
            execution_mode=real_llm.EXECUTION_MODE_STAGE_SELECTIVE_HF,
        )
        result_row = {
            "request_id": "req-1",
            "prompt_hash": activation["prompt_hash"],
            "model_id": artifact["model_id"],
            "artifact_hash": artifact["artifact_hash"],
            "activation_hash": activation["activation_hash"],
            "generation_step": 1,
            "max_new_tokens": 2,
            "next_token_id": 198,
            "next_token_text": "",
            "baseline_next_token_id": None,
            "baseline_next_token_text": "",
            "generated_token_ids": [151643, 198],
            "generated_token_count": 2,
            "generated_text": "",
            "generated_text_hash": real_llm._generated_text_hash(""),  # noqa: SLF001
            "baseline_match": False,
            "baseline_validation_skipped": True,
        }
        result_row["output_hash"] = real_llm._output_hash(result_row)  # noqa: SLF001
        validation = real_llm.validate_real_llm_sharded_inference(
            {
                "schema_version": real_llm.REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
                "type": real_llm.WORKLOAD_TYPE,
                "session_id": activation["session_id"],
                "stage_id": 1,
                "stage_count": 2,
                "model_id": artifact["model_id"],
                "backend": real_llm.BACKEND_CUDA,
                "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
                "execution_mode": real_llm.EXECUTION_MODE_STAGE_SELECTIVE_HF,
                "artifact_schema": real_llm.REAL_LLM_ARTIFACT_SCHEMA_VERSION,
                "artifact_hash": artifact["artifact_hash"],
                "split_index": 14,
                "max_new_tokens": 2,
                "generation_step": 1,
                "request_count": 1,
                "activation_count": 1,
                "activation_hashes": [activation["activation_hash"]],
                "activation_transport_ready": True,
                "stage_layer_range": [14, 28],
                "stage_parameter_count": 1_000,
                "full_model_parameter_count": 2_000,
                "stage_parameter_fraction": 0.5,
                "device_parameter_count": 1_000,
                "partition_parameter_split_valid": True,
                "stage_local_partition_ready": True,
                "stage1_partition_loaded": True,
                "stage_selective_hf_runtime_ready": True,
                "stage_selective_runtime_execution_ready": True,
                "stage_selective_weight_load_ready": True,
                "stage_selective_weight_application_ready": True,
                "runtime_buffer_materialization_ready": True,
                "baseline_match": False,
                "baseline_validation_skipped": True,
                "decoded_tokens_match": True,
                "generated_token_ids": [151643, 198],
                "generated_token_count": 2,
                "generated_text": "",
                "generated_text_hash": result_row["generated_text_hash"],
                "inference_result": result_row,
                "inference_results": [result_row],
                "real_llm_artifact_ready": True,
            },
            expected_spec=spec,
            replay_runtime=False,
        )

        self.assertTrue(validation["accepted"])
        self.assertTrue(validation["stage_selective_remote_validation"])
        self.assertTrue(validation["decoded_tokens_match"])
        self.assertEqual(validation["generation_step"], 1)
        self.assertEqual(validation["generated_token_count"], 2)

    def test_full_model_validation_rejects_baseline_skipped_stage1(self) -> None:
        artifact = {
            "schema": real_llm.REAL_LLM_ARTIFACT_SCHEMA_VERSION,
            "artifact_hash": "sha256:test-full-model-remote-artifact",
            "model_id": real_llm.DEFAULT_MODEL_ID,
            "backend": real_llm.BACKEND_CUDA,
            "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
        }
        activation = {
            "schema_version": real_llm.REAL_LLM_ACTIVATION_SCHEMA_VERSION,
            "session_id": "session-full-model-remote",
            "request_id": "req-1",
            "prompt_hash": "sha256:prompt",
            "model_id": artifact["model_id"],
            "artifact_hash": artifact["artifact_hash"],
            "split_index": 1,
            "input_ids": [1, 2],
            "position_ids": [0, 1],
            "hidden_shape": [1, 2, 2],
            "hidden_state": [[[0.1, 0.2], [0.3, 0.4]]],
        }
        activation["activation_hash"] = real_llm._activation_hash(activation)  # noqa: SLF001
        spec = real_llm.real_llm_sharded_inference_spec_for(
            "stage1-task",
            "stage1-miner",
            artifact,
            request_count=1,
            session_id=activation["session_id"],
            stage_id=1,
            activation_results=[activation],
        )
        result_row = {
            "request_id": "req-1",
            "model_id": artifact["model_id"],
            "artifact_hash": artifact["artifact_hash"],
            "activation_hash": activation["activation_hash"],
            "next_token_id": 42,
            "baseline_next_token_id": None,
            "generated_token_ids": [42],
            "generated_token_count": 1,
            "generated_text": "",
            "generated_text_hash": real_llm._generated_text_hash(""),  # noqa: SLF001
            "baseline_match": False,
            "baseline_validation_skipped": True,
        }
        result_row["output_hash"] = real_llm._output_hash(result_row)  # noqa: SLF001
        validation = real_llm.validate_real_llm_sharded_inference(
            {
                "schema_version": real_llm.REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
                "session_id": activation["session_id"],
                "stage_id": 1,
                "model_id": artifact["model_id"],
                "backend": real_llm.BACKEND_CUDA,
                "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
                "artifact_hash": artifact["artifact_hash"],
                "stage_local_partition_ready": True,
                "partition_parameter_split_valid": True,
                "stage1_partition_loaded": True,
                "stage_parameter_count": 1,
                "full_model_parameter_count": 2,
                "activation_hashes": [activation["activation_hash"]],
                "activation_transport_ready": True,
                "baseline_validation_skipped": True,
                "decoded_tokens_match": True,
                "inference_results": [result_row],
            },
            expected_spec=spec,
            replay_runtime=False,
        )

        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["code"], "real_llm_baseline_mismatch")

    def test_stage_selective_partition_summary_uses_applied_stage_bytes(self) -> None:
        class Param:
            def __init__(self, count: int) -> None:
                self._count = count
                self.is_meta = True

            def numel(self) -> int:
                raise RuntimeError("meta parameter count unavailable")

        class Module:
            def __init__(self, count: int = 1) -> None:
                self.param = Param(count)

            def parameters(self):
                return [self.param]

        class Base:
            def __init__(self) -> None:
                self.embed_tokens = Module()
                self.layers = [Module() for _ in range(28)]
                self.norm = Module()

        class Model:
            def __init__(self) -> None:
                self.model = Base()
                self.lm_head = Module()

            def parameters(self):
                for module in [self.model.embed_tokens, *self.model.layers, self.model.norm, self.lm_head]:
                    yield from module.parameters()

        summary = real_llm._stage_selective_partition_summary(  # noqa: SLF001
            Model(),
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "num_hidden_layers": 28,
                "hidden_size": 3584,
                "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
            },
            {"ready": True, "applied_tensor_bytes": 8_000},
            stage_id=0,
            split_index=14,
            partition_mode=real_llm.PARTITION_MODE_STAGE_LOCAL,
            device="cuda:0",
            family=real_llm.EXECUTION_FAMILY_LLAMA_LIKE,
        )

        self.assertTrue(summary["stage_selective_partition_summary"])
        self.assertTrue(summary["partition_parameter_split_valid"])
        self.assertTrue(summary["stage_local_partition_ready"])
        self.assertTrue(summary["stage_gpu_memory_reduced"])
        self.assertEqual(summary["stage_parameter_count"], 2_000)
        self.assertGreater(summary["full_model_parameter_count"], summary["stage_parameter_count"])

    def test_stage_selective_runtime_buffer_materializes_only_stage_owned_rotary_cache(self) -> None:
        class Buffer:
            is_meta = True
            shape = (8, 4)

        class Rotary:
            def __init__(self, *, cached_shape=(8, 4)) -> None:
                self.inv_freq = Buffer()
                self.cos_cached = Buffer()
                self.cos_cached.shape = cached_shape
                self.sin_cached = Buffer()
                self.sin_cached.shape = cached_shape

            def register_buffer(self, name, value, persistent=True):  # noqa: ANN001, ANN202
                setattr(self, name, value)

            def modules(self):
                return [self]

            def named_modules(self):
                yield "", self

            def named_buffers(self, recurse=True):  # noqa: ANN001, ANN202
                yield "inv_freq", self.inv_freq
                yield "cos_cached", self.cos_cached
                yield "sin_cached", self.sin_cached

            def parameters(self):
                return []

        class Attn:
            def __init__(self, *, cached_shape=(8, 4)) -> None:
                self.rotary_emb = Rotary(cached_shape=cached_shape)

            def modules(self):
                return [self, self.rotary_emb]

            def named_modules(self):
                yield "", self
                yield "rotary_emb", self.rotary_emb

            def named_buffers(self, recurse=True):  # noqa: ANN001, ANN202
                for name, value in self.rotary_emb.named_buffers(recurse=recurse):
                    yield "rotary_emb." + name, value

            def parameters(self):
                return []

        class Block:
            def __init__(self, *, cached_shape=(8, 4)) -> None:
                self.self_attn = Attn(cached_shape=cached_shape)

            def modules(self):
                return [self, *self.self_attn.modules()]

            def named_modules(self):
                yield "", self
                for name, value in self.self_attn.named_modules():
                    yield "self_attn" + (("." + name) if name else ""), value

            def named_buffers(self, recurse=True):  # noqa: ANN001, ANN202
                for name, value in self.self_attn.named_buffers(recurse=recurse):
                    yield "self_attn." + name, value

            def parameters(self):
                return []

        class Module:
            def modules(self):
                return [self]

            def named_modules(self):
                yield "", self

            def named_buffers(self, recurse=True):  # noqa: ANN001, ANN202
                return []

            def parameters(self):
                return []

        class Base:
            def __init__(self, *, cached_shape=(8, 4)) -> None:
                self.embed_tokens = Module()
                self.layers = [Block(cached_shape=cached_shape) for _ in range(4)]
                self.norm = Module()

        class Config:
            hidden_size = 8
            num_attention_heads = 2
            max_position_embeddings = 8
            rope_theta = 10000.0

        class Model:
            def __init__(self, *, cached_shape=(8, 4)) -> None:
                self.model = Base(cached_shape=cached_shape)
                self.lm_head = Module()
                self.config = Config()

            def modules(self):
                modules = [self, self.model, self.model.embed_tokens]
                for layer in self.model.layers:
                    modules.extend(layer.modules())
                modules.extend([self.model.norm, self.lm_head])
                return modules

            def named_modules(self):
                yield "", self
                yield "model", self.model
                yield "model.embed_tokens", self.model.embed_tokens
                for index, layer in enumerate(self.model.layers):
                    yield f"model.layers.{index}", layer
                    for name, value in layer.named_modules():
                        if name:
                            yield f"model.layers.{index}.{name}", value
                yield "model.norm", self.model.norm
                yield "lm_head", self.lm_head

            def named_buffers(self, recurse=True):  # noqa: ANN001, ANN202
                for index, layer in enumerate(self.model.layers):
                    for name, value in layer.named_buffers(recurse=recurse):
                        yield f"model.layers.{index}.{name}", value

            def parameters(self):
                return []

        model = Model(cached_shape=(8, 4))
        summary = real_llm._materialize_runtime_buffers(  # noqa: SLF001
            model,
            device="cpu",
            stage_id=0,
            split_index=2,
            family=real_llm.EXECUTION_FAMILY_LLAMA_LIKE,
        )

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["remaining_meta_buffer_count"], 0)
        self.assertEqual(summary["materialized_runtime_buffer_count"], 6)
        self.assertEqual(tuple(model.model.layers[0].self_attn.rotary_emb.cos_cached.shape), (8, 4))
        self.assertEqual(tuple(model.model.layers[0].self_attn.rotary_emb.sin_cached.shape), (8, 4))

        legacy_model = Model(cached_shape=(1, 1, 8, 4))
        legacy_summary = real_llm._materialize_runtime_buffers(  # noqa: SLF001
            legacy_model,
            device="cpu",
            stage_id=0,
            split_index=2,
            family=real_llm.EXECUTION_FAMILY_LLAMA_LIKE,
        )
        self.assertTrue(legacy_summary["ready"])
        self.assertEqual(tuple(legacy_model.model.layers[0].self_attn.rotary_emb.cos_cached.shape), (1, 1, 8, 4))
        self.assertEqual(tuple(legacy_model.model.layers[0].self_attn.rotary_emb.sin_cached.shape), (1, 1, 8, 4))

    def test_stage_selective_apply_can_force_cuda_safe_dtype(self) -> None:
        missing = real_llm.missing_hf_dependencies()
        if missing:
            self.skipTest("missing optional HF dependencies: " + ", ".join(missing))
        import torch

        class TinyMeta(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Module()
                self.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty((2, 2), device="meta", dtype=torch.bfloat16)
                )
                self.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
                self.model.layers[0].weight = torch.nn.Parameter(
                    torch.empty((2, 2), device="meta", dtype=torch.bfloat16)
                )
                self.model.layers[1].weight = torch.nn.Parameter(
                    torch.empty((2, 2), device="meta", dtype=torch.bfloat16)
                )
                self.model.norm = torch.nn.Module()
                self.model.norm.weight = torch.nn.Parameter(
                    torch.empty((2, 2), device="meta", dtype=torch.bfloat16)
                )
                self.lm_head = torch.nn.Module()
                self.lm_head.weight = torch.nn.Parameter(
                    torch.empty((2, 2), device="meta", dtype=torch.bfloat16)
                )

        model = TinyMeta()
        captured = {}
        original_load_state_dict = model.load_state_dict

        def wrapped_load_state_dict(state, *args, **kwargs):  # noqa: ANN001, ANN202
            captured.update(state)
            return original_load_state_dict(state, *args, **kwargs)

        model.load_state_dict = wrapped_load_state_dict  # type: ignore[method-assign]
        summary = real_llm._apply_stage_selective_tensors_to_model(  # noqa: SLF001
            model,
            {
                "model.embed_tokens.weight": torch.ones((2, 2), dtype=torch.bfloat16),
                "model.layers.0.weight": torch.ones((2, 2), dtype=torch.bfloat16),
            },
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
                "num_hidden_layers": 2,
                "split_index": 1,
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "model.layers.0.weight": "model.safetensors",
                    "model.layers.1.weight": "model.safetensors",
                    "model.norm.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors",
                },
            },
            stage_id=0,
            partition_mode=real_llm.PARTITION_MODE_STAGE_LOCAL,
            target_device="cpu",
            target_dtype=torch.float16,
        )

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["dtype_conversion_count"], 2)
        self.assertEqual(summary["applied_parameter_count"], 8)
        self.assertEqual(captured["model.embed_tokens.weight"].dtype, torch.float16)
        self.assertEqual(captured["model.layers.0.weight"].dtype, torch.float16)

    def test_stage1_restores_activation_to_materialized_weight_dtype(self) -> None:
        import torch

        class TinyStage(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float16))

        model = TinyStage()
        self.assertEqual(real_llm._first_materialized_parameter_dtype(model), torch.float16)  # noqa: SLF001
        hidden = torch.tensor([[[0.1, 0.2]]], dtype=real_llm._first_materialized_parameter_dtype(model))  # noqa: SLF001
        self.assertEqual(hidden.dtype, torch.float16)

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
