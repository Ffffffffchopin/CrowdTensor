import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from crowdtensor.heterogeneous_qwen_source import (
    materialize_qwen_stage_shard,
    qwen_stage_spec,
    resolve_qwen_source,
)
from crowdtensor.heterogeneous_qwen_training import HeterogeneousQwenStageTrainer
from crowdtensor.heterogeneous_tensor_transport import (
    ChunkedTensorStore,
    decode_tensor_payload,
    encode_tensor_message,
)
from crowdtensor.heterogeneous_training_checkpoint import (
    build_stage_checkpoint_archive,
)
from crowdtensor.heterogeneous_training_manifest import (
    MANIFEST_SCHEMA,
    validate_training_manifest,
)
from crowdtensor.qwen15b_training import (
    export_qwen_standard_peft_adapter,
    load_qwen_pipeline_stage,
    qwen_stage_adapter_hash,
    qwen_stage_adapter_state,
)


def tiny_source(root: Path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(123)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    first = {
        name: tensor
        for name, tensor in state.items()
        if name.startswith("model.embed_tokens") or name.startswith("model.layers.0.")
    }
    second = {name: tensor for name, tensor in state.items() if name not in first}
    save_file(first, root / "model-00001-of-00002.safetensors")
    save_file(second, root / "model-00002-of-00002.safetensors")
    weight_map = {
        **{name: "model-00001-of-00002.safetensors" for name in first},
        **{name: "model-00002-of-00002.safetensors" for name in second},
    }
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    parameter_count = sum(tensor.numel() for tensor in state.values())
    (root / "config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    stage_bytes = [
        sum(tensor.numel() * tensor.element_size() for tensor in values.values())
        for values in (first, second)
    ]
    stage_parameters = [
        sum(tensor.numel() for tensor in values.values())
        for values in (first, second)
    ]
    manifest = validate_training_manifest(
        {
            "schema": MANIFEST_SCHEMA,
            "model": {
                "model_id": "local/tiny-qwen2",
                "model_revision": "tiny-revision-1",
                "architecture": "Qwen2ForCausalLM",
                "model_type": "qwen2",
                "parameter_count": parameter_count,
                "weight_bytes": total_bytes,
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "source_dtype": "float32",
                "trust_remote_code": False,
            },
            "lora": {
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "bias": "none",
                "learning_rate": 0.01,
                "gradient_clip_norm": 1.0,
            },
            "dataset": {
                "dataset_id": "local/tiny",
                "dataset_revision": "tiny-data-1",
                "dataset_config": "default",
                "train_split": "train",
                "validation_split": "validation",
                "data_seed": 123,
            },
            "precision": {
                "cuda_compute_dtype": "float32",
                "cpu_compute_dtype": "float32",
                "boundary_dtype": "float16",
                "optimizer_dtype": "float32",
            },
            "training": {
                "target_steps": 2,
                "microbatches_per_step": 1,
                "microbatch_size": 1,
                "sequence_length": 6,
                "gradient_accumulation_steps": 1,
                "seed": 123,
            },
            "checkpoint": {
                "backend": "local",
                "retention_steps": 2,
                "checkpoint_every_steps": 1,
                "include_optimizer": True,
                "include_scheduler": True,
                "include_rng": True,
                "atomic_global_commit": True,
            },
            "scheduler": {
                "device_policy": "cpu",
                "placement_policy": "memory-performance",
                "rebalance_policy": "failure-and-straggler",
                "max_stages_per_miner": 1,
                "memory_reserve_fraction": 0.1,
                "cuda_memory_reserve_bytes": 1,
                "cpu_memory_reserve_bytes": 1,
                "straggler_ratio": 2.0,
                "network_cost_weight": 1.0,
                "load_cost_weight": 1.0,
                "beam_width": 16,
                "required_device_types": ["cpu"],
            },
            "stages": [
                {
                    "stage_id": 0,
                    "layer_start": 0,
                    "layer_end": 1,
                    "owns_embedding": True,
                    "owns_norm": False,
                    "owns_lm_head": False,
                    "allowed_device_types": ["cpu", "cuda"],
                    "preferred_device_type": "cuda",
                    "estimated_parameter_count": stage_parameters[0],
                    "estimated_weight_bytes": stage_bytes[0],
                    "estimated_compute_units": float(stage_parameters[0]),
                },
                {
                    "stage_id": 1,
                    "layer_start": 1,
                    "layer_end": 2,
                    "owns_embedding": False,
                    "owns_norm": True,
                    "owns_lm_head": True,
                    "allowed_device_types": ["cpu", "cuda"],
                    "preferred_device_type": "cpu",
                    "estimated_parameter_count": stage_parameters[1],
                    "estimated_weight_bytes": stage_bytes[1],
                    "estimated_compute_units": float(stage_parameters[1]),
                },
            ],
        }
    )
    return manifest, config.to_dict(), model


def stage_module(manifest, config, stage_id, shard, checkpoint, *, generation, resume):
    spec = qwen_stage_spec(manifest, stage_id=stage_id)
    module, report = load_qwen_pipeline_stage(
        config,
        spec,
        shard,
        device="cpu",
        compute_dtype="float32",
        inject_lora=True,
        lora_rank=manifest["lora"]["rank"],
        lora_alpha=manifest["lora"]["alpha"],
        lora_target_modules=manifest["lora"]["target_modules"],
        lora_seed=manifest["training"]["seed"],
        gradient_checkpointing=True,
        model_id=manifest["model"]["model_id"],
        model_revision=manifest["model"]["model_revision"],
    )
    module.train()
    trainer = HeterogeneousQwenStageTrainer(
        module,
        spec,
        training_manifest=manifest,
        placement_generation=generation,
        device="cpu",
        checkpoint_dir=checkpoint,
        resume=resume,
    )
    return module, trainer, report


def transport(tmp_path, manifest, value, *, step, direction):
    source, target = (0, 1) if direction == "forward_activation" else (1, 0)
    name = "activation" if direction == "forward_activation" else "gradient"
    envelope, chunks = encode_tensor_message(
        {name: value},
        job_id="tiny-job",
        manifest_hash=manifest["content_hash"],
        global_step=step,
        microbatch_id=0,
        source_stage_id=source,
        target_stage_id=target,
        direction=direction,
        placement_generation=step,
        assignment_token_hash="sha256:" + "1" * 64,
        chunk_bytes=128,
    )
    store = ChunkedTensorStore(tmp_path / f"transport-{step}-{direction}", max_chunk_bytes=128)
    store.begin(envelope, expected_generation=step)
    for index, chunk in enumerate(chunks):
        store.put_chunk(envelope["message_id"], index, chunk, expected_generation=step)
    payload = b"".join(store.read_chunk(envelope["message_id"], index) for index in range(len(chunks)))
    return decode_tensor_payload(payload, envelope)[name], envelope


def train_step(tmp_path, manifest, trainers, tokens, *, step):
    first, second = trainers
    first.begin_step()
    second.begin_step()
    forward = first.forward(0, tokens)
    activation, activation_envelope = transport(
        tmp_path, manifest, forward["activation"], step=step, direction="forward_activation"
    )
    backward = second.loss_backward(0, activation, tokens, microbatch_count=1)
    gradient, gradient_envelope = transport(
        tmp_path,
        manifest,
        backward["activation_gradient"],
        step=step,
        direction="backward_gradient",
    )
    first_backward = first.backward(0, gradient)
    first_finish = first.finish_step(global_step=step, dataset_cursor=step)
    second_finish = second.finish_step(global_step=step, dataset_cursor=step)
    return {
        "loss": backward["loss"],
        "first_backward": first_backward,
        "finishes": [first_finish, second_finish],
        "activation_envelope": activation_envelope,
        "gradient_envelope": gradient_envelope,
    }


def test_manifest_qwen_stages_train_transport_resume_export_and_reload(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, config, full_model = tiny_source(source)
    resolved_config, index, source_report = resolve_qwen_source(
        manifest, source_root=source
    )
    assert resolved_config["num_hidden_layers"] == 2
    assert len(index["weight_map"]) > 0
    assert source_report["source_verified"] is True
    shards = []
    for stage_id in (0, 1):
        shard = tmp_path / f"stage{stage_id}.safetensors"
        progress = []
        report = materialize_qwen_stage_shard(
            manifest,
            stage_id=stage_id,
            output_path=shard,
            source_root=source,
            max_group_bytes=4096,
            progress_callback=progress.append,
        )
        assert report["stage_selective_loading"] is True
        assert report["full_model_downloaded"] is False
        assert progress[0]["phase"] == "source_resolved"
        assert progress[-1]["phase"] == "stage_shard_saved"
        assert progress[-1]["source_tensor_count"] == report["source_tensor_count"]
        assert all("source_file_name" not in item for item in progress)
        shards.append(shard)

    checkpoint_dirs = [tmp_path / "checkpoint0", tmp_path / "checkpoint1"]
    modules = []
    trainers = []
    for stage_id in (0, 1):
        module, trainer, load_report = stage_module(
            manifest,
            config,
            stage_id,
            shards[stage_id],
            checkpoint_dirs[stage_id],
            generation=1,
            resume=False,
        )
        assert load_report["model_id"] == "local/tiny-qwen2"
        assert load_report["loaded_full_model"] is False
        modules.append(module)
        trainers.append(trainer)
    initial_hashes = [qwen_stage_adapter_hash(module) for module in modules]
    tokens = torch.tensor([[1, 7, 11, 3, 9, 2]])
    first = train_step(tmp_path, manifest, trainers, tokens, step=1)
    assert first["loss"] > 0
    assert first["activation_envelope"]["tensor_specs"][0]["dtype"] == "float16"
    assert first["gradient_envelope"]["tensor_specs"][0]["dtype"] == "float16"
    assert all(item["lora_gradient_norm"] > 0 for item in first["finishes"])
    assert all(item["scheduler_step_applied"] for item in first["finishes"])
    assert [qwen_stage_adapter_hash(module) for module in modules] != initial_hashes
    for stage_id in (0, 1):
        _archive, report = build_stage_checkpoint_archive(
            checkpoint_dirs[stage_id],
            training_manifest=manifest,
            stage_id=stage_id,
        )
        assert report["scheduler_state_present"] is True
        assert report["placement_generation"] == 1

    resumed_modules = []
    resumed_trainers = []
    for stage_id in (0, 1):
        module, trainer, _load_report = stage_module(
            manifest,
            config,
            stage_id,
            shards[stage_id],
            checkpoint_dirs[stage_id],
            generation=2,
            resume=True,
        )
        assert trainer.loaded_checkpoint["global_step"] == 1
        assert trainer.loaded_checkpoint["placement_generation"] == 1
        resumed_modules.append(module)
        resumed_trainers.append(trainer)
    second = train_step(tmp_path, manifest, resumed_trainers, tokens, step=2)
    assert all(item["placement_generation"] == 2 for item in second["finishes"])
    assert all(item["lora_gradient_norm"] > 0 for item in second["finishes"])

    export = export_qwen_standard_peft_adapter(
        [qwen_stage_adapter_state(module) for module in resumed_modules],
        tmp_path / "adapter",
        lora_rank=manifest["lora"]["rank"],
        lora_alpha=manifest["lora"]["alpha"],
        lora_target_modules=manifest["lora"]["target_modules"],
        model_id=manifest["model"]["model_id"],
        model_revision=manifest["model"]["model_revision"],
    )
    from peft import PeftModel

    reloaded = PeftModel.from_pretrained(
        full_model, export["adapter_dir"], local_files_only=True
    )
    with torch.no_grad():
        logits = reloaded(input_ids=tokens, use_cache=False).logits
    assert torch.isfinite(logits).all()
    assert export["adapter_tensor_count"] == 28
    assert export["model_id"] == "local/tiny-qwen2"
