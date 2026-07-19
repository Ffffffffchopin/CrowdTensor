from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

jax = pytest.importorskip("jax")

from crowdtensor.heterogeneous_jax_qwen_training import (  # noqa: E402
    JaxQwenStageTrainer,
    jax_adapter_hash,
)
from crowdtensor.heterogeneous_qwen_source import qwen_stage_spec  # noqa: E402
from crowdtensor.heterogeneous_qwen_training import (  # noqa: E402
    HeterogeneousStageProcessClient,
)
from crowdtensor.heterogeneous_training_checkpoint import (  # noqa: E402
    build_stage_checkpoint_archive,
    validate_stage_checkpoint_archive,
)
from crowdtensor.heterogeneous_training_manifest import (  # noqa: E402
    qwen25_7b_lora_tpu_manifest,
    validate_training_manifest,
)
from crowdtensor.qwen15b_training import load_qwen_pipeline_stage  # noqa: E402


def tiny_manifest() -> dict:
    manifest = deepcopy(qwen25_7b_lora_tpu_manifest())
    manifest.pop("content_hash")
    manifest["model"].update(
        num_hidden_layers=5,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        weight_bytes=10_000,
        parameter_count=5_000,
    )
    for stage_id, stage in enumerate(manifest["stages"]):
        stage.update(
            stage_id=stage_id,
            layer_start=stage_id,
            layer_end=stage_id + 1,
            layer_count=1,
            owns_embedding=stage_id == 0,
            owns_norm=stage_id == 4,
            owns_lm_head=stage_id == 4,
            estimated_parameter_count=1_000,
            estimated_weight_bytes=2_000,
            estimated_compute_units=1_000.0,
        )
        if stage_id == 2:
            stage["allowed_device_types"] = ["jax_tpu"]
            stage["preferred_device_type"] = "jax_tpu"
        elif stage_id == 4:
            stage["allowed_device_types"] = ["cpu"]
            stage["preferred_device_type"] = "cpu"
        else:
            stage["allowed_device_types"] = ["cpu", "cuda"]
            stage["preferred_device_type"] = "cuda"
    return validate_training_manifest(manifest)


def tiny_config() -> dict:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 5,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 128,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "vocab_size": 32,
        "tie_word_embeddings": False,
        "use_cache": False,
    }


def write_middle_stage_shard(path) -> None:
    generator = torch.Generator().manual_seed(20260713)

    def weight(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.03

    prefix = "model.layers.2"
    state = {
        f"{prefix}.input_layernorm.weight": torch.ones(8),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(8),
        f"{prefix}.self_attn.q_proj.weight": weight(8, 8),
        f"{prefix}.self_attn.q_proj.bias": weight(8),
        f"{prefix}.self_attn.k_proj.weight": weight(4, 8),
        f"{prefix}.self_attn.k_proj.bias": weight(4),
        f"{prefix}.self_attn.v_proj.weight": weight(4, 8),
        f"{prefix}.self_attn.v_proj.bias": weight(4),
        f"{prefix}.self_attn.o_proj.weight": weight(8, 8),
        f"{prefix}.mlp.gate_proj.weight": weight(16, 8),
        f"{prefix}.mlp.up_proj.weight": weight(16, 8),
        f"{prefix}.mlp.down_proj.weight": weight(8, 16),
    }
    save_file(state, path)


def torch_reference(manifest: dict, config: dict, shard, trainer):
    from peft import set_peft_model_state_dict

    module, _report = load_qwen_pipeline_stage(
        config,
        qwen_stage_spec(manifest, stage_id=2),
        shard,
        device="cpu",
        compute_dtype="bfloat16",
        inject_lora=True,
        lora_rank=manifest["lora"]["rank"],
        lora_alpha=manifest["lora"]["alpha"],
        lora_target_modules=manifest["lora"]["target_modules"],
        lora_dropout=0.0,
        lora_seed=manifest["training"]["seed"],
        gradient_checkpointing=False,
        model_id=manifest["model"]["model_id"],
        model_revision=manifest["model"]["model_revision"],
    )
    adapter = {
        name: torch.from_numpy(
            np.asarray(jax.device_get(value), dtype=np.float32).copy()
        )
        for name, value in trainer.lora.items()
    }
    incompatible = set_peft_model_state_dict(module, adapter, adapter_name="default")
    assert not list(getattr(incompatible, "unexpected_keys", []) or [])
    module.train()
    return module


def canonical_parameter_name(name: str) -> str:
    return name.replace(".default.weight", ".weight")


def test_jax_stage_matches_pytorch_forward_and_backward(tmp_path) -> None:
    manifest = tiny_manifest()
    config = tiny_config()
    shard = tmp_path / "stage2.safetensors"
    write_middle_stage_shard(shard)
    trainer = JaxQwenStageTrainer(
        training_manifest=manifest,
        config=config,
        stage_id=2,
        shard_path=shard,
        checkpoint_dir=tmp_path / "checkpoint",
        placement_generation=1,
        resume=False,
        require_tpu=False,
        expected_tpu_devices=1,
    )
    status = trainer.status()
    assert status["forward_output_sharding_explicit"] is True
    assert status["backward_output_sharding_explicit"] is True
    assert status["boundary_output_replicated"] is True
    module = torch_reference(manifest, config, shard, trainer)
    hidden = torch.linspace(-0.25, 0.25, steps=64).reshape(1, 8, 8)
    torch_input = hidden.to(torch.bfloat16).requires_grad_(True)
    torch_output = module(torch_input)

    trainer.begin_step()
    jax_forward = trainer.forward(0, hidden)
    np.testing.assert_allclose(
        jax_forward["activation"].float().numpy(),
        torch_output.detach().float().numpy(),
        rtol=0.08,
        atol=0.04,
    )

    incoming = torch.linspace(0.1, 0.2, steps=64).reshape(1, 8, 8)
    torch_output.backward(incoming.to(torch_output.dtype))
    jax_backward = trainer.backward(0, incoming)
    np.testing.assert_allclose(
        jax_backward["activation_gradient"].float().numpy(),
        torch_input.grad.detach().float().numpy(),
        rtol=0.12,
        atol=0.05,
    )
    torch_gradients = {
        canonical_parameter_name(name): parameter.grad.detach().float().numpy()
        for name, parameter in module.named_parameters()
        if "lora_" in name and parameter.grad is not None
    }
    assert set(torch_gradients) == set(trainer.accumulated_gradients)
    for name, value in trainer.accumulated_gradients.items():
        np.testing.assert_allclose(
            np.asarray(jax.device_get(value), dtype=np.float32),
            torch_gradients[name],
            rtol=0.15,
            atol=0.05,
        )
    assert any(float(np.linalg.norm(value)) > 0 for value in torch_gradients.values())


def test_jax_stage_updates_checkpoints_and_restores(tmp_path) -> None:
    manifest = tiny_manifest()
    config = tiny_config()
    shard = tmp_path / "stage2.safetensors"
    checkpoint_dir = tmp_path / "checkpoint"
    write_middle_stage_shard(shard)
    trainer = JaxQwenStageTrainer(
        training_manifest=manifest,
        config=config,
        stage_id=2,
        shard_path=shard,
        checkpoint_dir=checkpoint_dir,
        placement_generation=3,
        resume=False,
        require_tpu=False,
        expected_tpu_devices=1,
    )
    before = jax_adapter_hash(trainer.lora)
    hidden = torch.linspace(-0.25, 0.25, steps=64).reshape(1, 8, 8)
    incoming = torch.linspace(0.1, 0.2, steps=64).reshape(1, 8, 8) * 128.0
    trainer.begin_step()
    trainer.forward(0, hidden)
    trainer.backward(0, incoming)
    finished = trainer.finish_step(global_step=1, dataset_cursor=1)

    archive, report = build_stage_checkpoint_archive(
        checkpoint_dir, training_manifest=manifest, stage_id=2
    )
    validated = validate_stage_checkpoint_archive(
        archive,
        training_manifest=manifest,
        expected_stage_id=2,
        expected_step=1,
        expected_dataset_cursor=1,
        expected_placement_generation=3,
    )
    restored = JaxQwenStageTrainer(
        training_manifest=manifest,
        config=config,
        stage_id=2,
        shard_path=shard,
        checkpoint_dir=checkpoint_dir,
        placement_generation=4,
        resume=True,
        require_tpu=False,
        expected_tpu_devices=1,
    )

    assert finished["optimizer_step_applied"] is True
    assert finished["scheduler_step_applied"] is True
    assert finished["lora_gradient_norm"] > 0
    assert finished["adapter_tensor_hash"] != before
    assert validated["archive_hash"] == report["archive_hash"]
    assert validated["runtime_backend"] == "jax_tpu"
    assert restored.loaded_checkpoint["global_step"] == 1
    assert restored.optimizer_state["step"] == 1
    assert jax_adapter_hash(restored.lora) == finished["adapter_tensor_hash"]


def test_shared_stage_process_protocol_runs_jax_backend(tmp_path) -> None:
    manifest = tiny_manifest()
    config = tiny_config()
    shard = tmp_path / "stage2.safetensors"
    write_middle_stage_shard(shard)
    process = HeterogeneousStageProcessClient(
        training_manifest=manifest,
        config=config,
        stage_id=2,
        shard_path=shard,
        checkpoint_dir=tmp_path / "process-checkpoint",
        device="jax_tpu:0",
        placement_generation=1,
        resume=False,
        ready_timeout=120.0,
        require_tpu=False,
        expected_tpu_devices=1,
    )
    try:
        ready = process.public_ready()
        assert ready["runtime_backend"] == "jax_tpu"
        assert ready["jax_mesh_device_count"] == 1
        process.call("begin_step", timeout=30.0)
        hidden = torch.linspace(-0.25, 0.25, steps=64).reshape(1, 8, 8)
        forward = process.call("forward", timeout=120.0, microbatch_id=0, value=hidden)
        backward = process.call(
            "backward",
            timeout=120.0,
            microbatch_id=0,
            activation_gradient=torch.ones_like(forward["activation"]) * 128.0,
        )
        finished = process.call(
            "finish_step", timeout=120.0, global_step=1, dataset_cursor=1
        )
        status = process.call("status", timeout=30.0)
        assert backward["activation_gradient"].shape == hidden.shape
        assert finished["runtime_backend"] == "jax_tpu"
        assert finished["lora_gradient_norm"] > 0
        assert status["optimizer_step"] == 1
    finally:
        process.stop()
