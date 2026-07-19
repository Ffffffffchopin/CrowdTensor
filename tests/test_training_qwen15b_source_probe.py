from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from crowdtensor.qwen15b_training import (
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    read_safetensors_header,
)
from scripts.training_qwen15b_source_probe import build


def test_source_probe_writes_checker_ready_public_artifacts(tmp_path) -> None:
    model = tmp_path / "fixture.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.ones(2, 2),
        "model.norm.weight": torch.ones(2),
    }
    for layer in range(28):
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.ones(2, 2)
    save_file(tensors, model)
    _header_length, header = read_safetensors_header(model)
    source = {
        "source_verified": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "dataset": {"source_verified": True},
        "public_artifact_safe": True,
    }

    report = build(
        tmp_path / "output",
        resolver=lambda: (source, {"num_hidden_layers": 28}, header),
    )

    assert report["ok"] is True
    assert report["weight_index"]["metadata"]["tensor_count"] == len(tensors)
    assert report["ownership"]["all_source_tensors_covered"] is True
    assert report["ownership"]["four_distinct_kernel_device_placements"] is True
    for relative in report["artifacts"].values():
        assert (tmp_path / "output" / relative).is_file()
    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "input_ids" not in encoded
