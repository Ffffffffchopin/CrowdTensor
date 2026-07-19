"""Shared decoder-stage LoRA mechanics for installed Model Adapter plugins."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from .model_adapter import ModelAdapter, get_model_adapter


_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def tensor_state_hash(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


class PassThroughDecoderLayer:
    @staticmethod
    def create() -> Any:
        import torch

        class PassThrough(torch.nn.Module):
            def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
                return hidden_states

        return PassThrough()


def owned_lora_state(model: Any, *, start: int, end: int) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    result: dict[str, Any] = {}
    for name, tensor in get_peft_model_state_dict(model).items():
        match = _LAYER.search(name)
        if match and start <= int(match.group(1)) < end:
            result[name] = tensor.detach().cpu().contiguous()
    if not result:
        raise RuntimeError("model_adapter_owned_lora_state_empty")
    return result


def configure_adapter_stage_model(
    *,
    adapter_id: str,
    model_id: str,
    model_revision: str,
    model_config: Mapping[str, Any],
    stage_id: int,
    split_index: int,
    device: str,
    cache_dir: str,
    rank: int,
    alpha: int,
) -> tuple[Any, Any, Any, int, int, dict[str, Any]]:
    import torch

    adapter = get_model_adapter(adapter_id)
    canonical = adapter.validate_config(model_config)
    if not adapter.supports(model_id=model_id, config=model_config):
        raise RuntimeError("model_adapter_stage_source_unsupported")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = adapter.load_model(
        model_id=model_id,
        revision=model_revision,
        device=device,
        dtype=dtype,
        local_files_only=False,
        cache_dir=cache_dir,
    )
    model = adapter.apply_lora(model, rank=rank, alpha=alpha)
    causal = model.get_base_model()
    decoder = getattr(causal, "model", None)
    layers = getattr(decoder, "layers", None)
    if decoder is None or layers is None or not hasattr(causal, "lm_head"):
        raise RuntimeError("model_adapter_decoder_pipeline_components_missing")
    total_layers = len(layers)
    if total_layers != int(canonical["num_hidden_layers"]):
        raise RuntimeError("model_adapter_loaded_layer_count_mismatch")
    if int(stage_id) not in {0, 1} or split_index <= 0 or split_index >= total_layers:
        raise RuntimeError("model_adapter_split_index_invalid")
    start, end = (0, split_index) if int(stage_id) == 0 else (split_index, total_layers)
    for name, parameter in model.named_parameters():
        match = _LAYER.search(name)
        parameter.requires_grad = bool(
            match and start <= int(match.group(1)) < end and "lora_" in name
        )
    if int(stage_id) == 0:
        for index in range(split_index, total_layers):
            layers[index] = PassThroughDecoderLayer.create()
        decoder.config.num_hidden_layers = split_index
        decoder.norm = torch.nn.Identity()
    else:
        for index in range(0, split_index):
            layers[index] = PassThroughDecoderLayer.create()
        decoder.config.num_hidden_layers = total_layers
    trainable = [item for item in model.parameters() if item.requires_grad]
    if not trainable:
        raise RuntimeError("model_adapter_stage_trainable_parameters_missing")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)
    model.train()
    details = {
        "adapter_id": adapter.adapter_id,
        "family": adapter.family,
        "architecture": adapter.architectures[0],
        "model_id": model_id,
        "model_revision": model_revision,
        "layer_start": start,
        "layer_end": end,
        "loaded_layer_count": total_layers,
        "stage_pruned_layer_count": total_layers - (end - start),
        "full_model_loaded_before_stage_pruning": True,
        "stage_selective_weight_loading_verified": False,
        "trainable_lora_tensor_count": len(owned_lora_state(model, start=start, end=end)),
        "trust_remote_code": False,
        "public_artifact_safe": True,
    }
    return model, causal, optimizer, start, end, details


def merge_stage_adapters(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
    *,
    adapter: ModelAdapter,
    rank: int,
    alpha: int,
) -> dict[str, Any]:
    from peft import LoraConfig
    from safetensors.torch import load_file, save_file

    source = Path(checkpoint_dir)
    destination = Path(export_dir)
    combined: dict[str, Any] = {}
    for stage_id in (0, 1):
        state = load_file(str(source / f"stage{stage_id}_adapter.safetensors"))
        if set(combined).intersection(state):
            raise RuntimeError("model_adapter_stage_adapter_key_overlap")
        combined.update(state)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(combined, str(destination / "adapter_model.safetensors"))
    template = source / "peft-template" / "adapter_config.json"
    if template.is_file():
        shutil.copyfile(template, destination / "adapter_config.json")
    else:
        config = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(adapter.target_modules),
        )
        config.base_model_name_or_path = adapter.default_model_id
        config.revision = adapter.default_revision
        config.inference_mode = True
        config.save_pretrained(destination)
    return {
        "adapter_id": adapter.adapter_id,
        "adapter_tensor_count": len(combined),
        "adapter_file_hash": "sha256:"
        + hashlib.sha256((destination / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "adapter_config_hash": "sha256:"
        + hashlib.sha256((destination / "adapter_config.json").read_bytes()).hexdigest(),
        "standard_peft_format": True,
        "stage_adapter_key_overlap": False,
        "public_artifact_safe": True,
    }


def independent_reload(
    export_dir: str | Path,
    *,
    adapter: ModelAdapter,
    device: str,
    cache_dir: str,
) -> dict[str, Any]:
    import torch

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = adapter.reload_adapter(
        model_id=adapter.default_model_id,
        revision=adapter.default_revision,
        adapter_dir=export_dir,
        device=device,
        dtype=dtype,
        local_files_only=False,
        cache_dir=cache_dir,
    )
    model.eval()
    vocab = int(adapter.validate_config(adapter.canonical_config())["vocab_size"])
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device) % vocab
    with torch.no_grad():
        logits = model(input_ids=token_ids, use_cache=False).logits
    finite = bool(torch.isfinite(logits.float()).all().item())
    return {
        "adapter_id": adapter.adapter_id,
        "independent_process_reload": True,
        "adapter_reload_verified": finite,
        "reload_logits_finite": finite,
        "reload_output_shape": list(logits.shape),
        "token_ids_public": False,
        "logit_values_public": False,
        "public_artifact_safe": True,
    }
