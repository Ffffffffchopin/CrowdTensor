from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from crowdtensor.qwen15b_training import (
    MODEL_ID,
    MODEL_REVISION,
    QwenStageSpec,
    assemble_qwen_standard_peft_state,
    build_stage_ownership,
    build_weight_index,
    canonical_stage_specs,
    export_qwen_standard_peft_adapter,
    load_qwen_pipeline_stage,
    load_qwen_stage_checkpoint,
    materialize_stage_shard,
    qwen_stage_adapter_hash,
    qwen_stage_adapter_state,
    save_qwen_stage_checkpoint,
    _tokenize_split,
    read_safetensors_header,
    select_stage_source_keys,
)


def _tiny_qwen_file(path: Path) -> dict[str, torch.Tensor]:
    tensors = {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "model.norm.weight": torch.ones(4, dtype=torch.float32),
    }
    for layer in range(28):
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.full(
            (4, 4), float(layer), dtype=torch.float32
        )
        tensors[f"model.layers.{layer}.mlp.down_proj.weight"] = torch.full(
            (4, 4), float(layer + 100), dtype=torch.float32
        )
    save_file(tensors, path)
    return tensors


def test_generated_weight_index_and_four_stage_ownership_cover_real_topology(tmp_path) -> None:
    source_path = tmp_path / "model.safetensors"
    tensors = _tiny_qwen_file(source_path)
    _header_length, header = read_safetensors_header(source_path)
    index = build_weight_index(header)
    ownership = build_stage_ownership({"num_hidden_layers": 28}, header)

    assert index["metadata"]["tensor_count"] == len(tensors)
    assert index["metadata"]["generated_from_single_safetensors_header"] is True
    assert index["metadata"]["official_index_present"] is False
    assert set(index["weight_map"]) == set(tensors)
    assert ownership["all_source_tensors_covered"] is True
    assert ownership["uncovered_source_keys"] == []
    assert ownership["duplicate_source_key_owners"] == {
        "model.embed_tokens.weight": [0, 3]
    }
    assert ownership["only_tied_embedding_lm_head_duplicated"] is True
    assert ownership["four_distinct_kernel_device_placements"] is True
    assert [item["layer_count"] for item in ownership["stages"]] == [7, 7, 7, 7]


def test_stage_key_selection_keeps_embedding_and_lm_head_on_required_stages(tmp_path) -> None:
    source_path = tmp_path / "model.safetensors"
    _tiny_qwen_file(source_path)
    _header_length, header = read_safetensors_header(source_path)
    specs = canonical_stage_specs()

    stage0, aliases0 = select_stage_source_keys(header, specs[0])
    stage1, aliases1 = select_stage_source_keys(header, specs[1])
    stage3, aliases3 = select_stage_source_keys(header, specs[3])

    assert "model.embed_tokens.weight" in stage0
    assert "model.norm.weight" not in stage0
    assert aliases0 == {}
    assert all("model.embed_tokens.weight" != key for key in stage1)
    assert aliases1 == {}
    assert "model.norm.weight" in stage3
    assert "model.embed_tokens.weight" in stage3
    assert aliases3 == {"lm_head.weight": "model.embed_tokens.weight"}


def test_stage_shards_materialize_only_selected_http_ranges(tmp_path) -> None:
    source_path = tmp_path / "model.safetensors"
    full = _tiny_qwen_file(source_path)
    header_length, header = read_safetensors_header(source_path)
    raw = source_path.read_bytes()

    def range_reader(_url: str, start: int, end: int) -> bytes:
        return raw[start : end + 1]

    loaded_keys: set[str] = set()
    for spec in canonical_stage_specs():
        output = tmp_path / f"stage-{spec.stage_id}.safetensors"
        report = materialize_stage_shard(
            spec=spec,
            header_length=header_length,
            header=header,
            output_path=output,
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            range_reader=range_reader,
            max_group_bytes=1024 * 1024,
        )
        selected, _aliases = select_stage_source_keys(header, spec)
        shard = load_file(output)
        assert set(shard) == set(selected)
        assert report["stage_selective_loading"] is True
        assert report["full_model_file_downloaded"] is False
        assert report["source_tensor_count"] == len(selected)
        assert report["downloaded_range_bytes"] == sum(
            full[name].numel() * full[name].element_size() for name in selected
        )
        assert "shard_path" in report
        loaded_keys.update(selected)
    assert loaded_keys == set(full)


def test_tokenizer_builds_fixed_private_sequences_without_padding() -> None:
    class Tokenizer:
        eos_token_id = 99

        @staticmethod
        def encode(text: str, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(value) % 17 for value in text]

    rows, indexes = _tokenize_split(
        ["", "alpha", "beta", "gamma", "delta", "epsilon"],
        Tokenizer(),
        sequence_length=8,
        sequence_count=2,
    )
    assert len(rows) == 2
    assert all(len(row) == 8 for row in rows)
    assert indexes == [1, 2, 3]
    assert 99 in rows[0] or 99 in rows[1]


def _tiny_transformers_qwen(tmp_path: Path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(17)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config)
    model.tie_weights()
    model.eval()
    source = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name != "lm_head.weight"
    }
    source_path = tmp_path / "tiny-qwen.safetensors"
    save_file(source, source_path)
    specs = [
        QwenStageSpec(0, "A", 0, 0, 1, owns_embedding=True),
        QwenStageSpec(1, "A", 1, 1, 2),
        QwenStageSpec(2, "B", 0, 2, 3),
        QwenStageSpec(3, "B", 1, 3, 4, owns_norm=True, owns_lm_head=True),
    ]
    header_length, header = read_safetensors_header(source_path)
    raw = source_path.read_bytes()

    def range_reader(_url: str, start: int, end: int) -> bytes:
        return raw[start : end + 1]

    shards: list[Path] = []
    for spec in specs:
        shard = tmp_path / f"tiny-stage-{spec.stage_id}.safetensors"
        materialize_stage_shard(
            spec=spec,
            header_length=header_length,
            header=header,
            output_path=shard,
            range_reader=range_reader,
            max_group_bytes=1024 * 1024,
        )
        shards.append(shard)
    return config, model, specs, shards


def test_four_stage_runtime_matches_full_transformers_qwen_logits_and_loss(tmp_path) -> None:
    config, full_model, specs, shards = _tiny_transformers_qwen(tmp_path)
    stages = []
    reports = []
    for spec, shard in zip(specs, shards, strict=True):
        stage, report = load_qwen_pipeline_stage(
            config.to_dict(),
            spec,
            shard,
            device="cpu",
            compute_dtype=torch.float32,
            inject_lora=False,
            gradient_checkpointing=False,
        )
        stage.eval()
        stages.append(stage)
        reports.append(report)

    input_ids = torch.tensor([[1, 7, 11, 3, 9, 2], [4, 5, 8, 13, 6, 10]])
    labels = input_ids.clone()
    with torch.no_grad():
        expected = full_model(input_ids=input_ids).logits
        actual = input_ids
        for stage in stages:
            actual = stage(actual)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    expected_loss = torch.nn.functional.cross_entropy(
        expected[:, :-1].reshape(-1, expected.shape[-1]),
        labels[:, 1:].reshape(-1),
    )
    actual_loss = torch.nn.functional.cross_entropy(
        actual[:, :-1].reshape(-1, actual.shape[-1]),
        labels[:, 1:].reshape(-1),
    )
    assert torch.allclose(actual_loss, expected_loss, atol=1e-6, rtol=1e-6)
    assert all(report["meta_device_construction"] for report in reports)
    assert all(report["loaded_full_model"] is False for report in reports)
    assert [report["loaded_layer_indexes"] for report in reports] == [[0], [1], [2], [3]]


def test_four_stage_lora_gradients_and_standard_peft_key_mapping(tmp_path) -> None:
    config, _full_model, specs, shards = _tiny_transformers_qwen(tmp_path)
    stages = []
    for spec, shard in zip(specs, shards, strict=True):
        stage, report = load_qwen_pipeline_stage(
            config.to_dict(),
            spec,
            shard,
            device="cpu",
            compute_dtype=torch.float32,
            inject_lora=True,
            lora_rank=2,
            lora_alpha=4,
            lora_seed=91,
            gradient_checkpointing=True,
        )
        stage.train()
        assert report["only_lora_trainable"] is True
        assert report["trainable_parameter_dtypes"] == ["float32"]
        assert report["fp32_lora_parameters_for_grad_scaler"] is True
        assert report["foreign_layer_count"] == 0
        stages.append(stage)

    input_ids = torch.tensor([[1, 7, 11, 3, 9, 2]])
    value = input_ids
    for stage in stages:
        value = stage(value)
    loss = torch.nn.functional.cross_entropy(
        value[:, :-1].reshape(-1, value.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )
    loss.backward()
    for stage in stages:
        gradient_sum = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in stage.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        assert gradient_sum > 0
    stage_states = [qwen_stage_adapter_state(stage) for stage in stages]
    standard = assemble_qwen_standard_peft_state(stage_states)
    assert len(standard) == sum(len(state) for state in stage_states)
    assert all(name.startswith("base_model.model.model.layers.") for name in standard)
    for layer_index in range(4):
        assert any(f".layers.{layer_index}." in name for name in standard)


def test_fp16_base_promotes_lora_storage_to_fp32_for_grad_scaler(tmp_path) -> None:
    config, _full_model, specs, shards = _tiny_transformers_qwen(tmp_path)
    stage, report = load_qwen_pipeline_stage(
        config.to_dict(),
        specs[0],
        shards[0],
        device="cpu",
        compute_dtype=torch.float16,
        inject_lora=True,
        lora_rank=2,
        lora_alpha=4,
        lora_seed=91,
        gradient_checkpointing=True,
    )
    trainable = [parameter for parameter in stage.parameters() if parameter.requires_grad]
    frozen = [parameter for parameter in stage.parameters() if not parameter.requires_grad]
    assert trainable and all(parameter.dtype == torch.float32 for parameter in trainable)
    assert frozen and all(parameter.dtype == torch.float16 for parameter in frozen)
    assert report["trainable_parameter_dtypes"] == ["float32"]
    assert report["frozen_parameter_dtypes"] == ["float16"]
    assert report["fp32_lora_parameters_for_grad_scaler"] is True


def test_stage_checkpoint_restores_adapter_optimizer_scaler_and_cursor(tmp_path) -> None:
    config, _full_model, specs, shards = _tiny_transformers_qwen(tmp_path)
    spec = specs[0]

    def make_stage():
        stage, _ = load_qwen_pipeline_stage(
            config.to_dict(),
            spec,
            shards[0],
            device="cpu",
            compute_dtype=torch.float32,
            inject_lora=True,
            lora_rank=2,
            lora_alpha=4,
            lora_seed=123,
            gradient_checkpointing=False,
        )
        stage.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in stage.parameters() if parameter.requires_grad],
            lr=0.01,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return stage, optimizer, scaler

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])

    def update(stage, optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = stage(tokens).float().square().mean()
        loss.backward()
        optimizer.step()

    baseline, baseline_optimizer, baseline_scaler = make_stage()
    update(baseline, baseline_optimizer)
    checkpoint = save_qwen_stage_checkpoint(
        baseline,
        baseline_optimizer,
        baseline_scaler,
        tmp_path / "checkpoint",
        spec=spec,
        global_step=1,
        dataset_cursor=4,
        device="cpu",
    )
    update(baseline, baseline_optimizer)

    resumed, resumed_optimizer, resumed_scaler = make_stage()
    restored = load_qwen_stage_checkpoint(
        resumed,
        resumed_optimizer,
        resumed_scaler,
        tmp_path / "checkpoint",
        spec=spec,
        device="cpu",
    )
    assert restored["global_step"] == 1
    assert restored["dataset_cursor"] == 4
    assert restored["content_hash"] == checkpoint["content_hash"]
    update(resumed, resumed_optimizer)
    assert qwen_stage_adapter_hash(resumed) == qwen_stage_adapter_hash(baseline)
    for name, value in qwen_stage_adapter_state(baseline).items():
        assert torch.equal(value, qwen_stage_adapter_state(resumed)[name])


def test_exported_four_stage_adapter_loads_with_standard_peft(tmp_path) -> None:
    from peft import PeftModel

    config, full_model, specs, shards = _tiny_transformers_qwen(tmp_path)
    stage_states = []
    for spec, shard in zip(specs, shards, strict=True):
        stage, _ = load_qwen_pipeline_stage(
            config.to_dict(),
            spec,
            shard,
            device="cpu",
            compute_dtype=torch.float32,
            inject_lora=True,
            lora_rank=2,
            lora_alpha=4,
            lora_seed=55,
            gradient_checkpointing=False,
        )
        for parameter in stage.parameters():
            if parameter.requires_grad:
                parameter.data.fill_(0.01)
        stage_states.append(qwen_stage_adapter_state(stage))
    exported = export_qwen_standard_peft_adapter(
        stage_states,
        tmp_path / "adapter",
        lora_rank=2,
        lora_alpha=4,
    )
    model = PeftModel.from_pretrained(full_model, exported["adapter_dir"], local_files_only=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            before = model(input_ids=tokens).logits
        after = model(input_ids=tokens).logits
    assert not torch.equal(before, after)
    assert exported["standard_peft_format"] is True
    assert exported["layer_indexes"] == [0, 1, 2, 3]
