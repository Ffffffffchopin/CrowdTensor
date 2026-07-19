import hashlib
import io
import json
import zipfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from crowdtensor.heterogeneous_training_checkpoint import (
    CHECKPOINT_SCHEMA,
    build_stage_checkpoint_archive,
    checkpoint_file_names,
    load_jax_stage_checkpoint,
    restore_stage_checkpoint_archive,
    save_jax_stage_checkpoint,
    validate_stage_checkpoint_archive,
)
from crowdtensor.heterogeneous_training_manifest import (
    qwen25_7b_lora_manifest,
    qwen25_7b_lora_tpu_manifest,
    stable_hash,
    validate_training_manifest,
)


def sha(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_hash(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(name.encode() + b"\0")
        digest.update(len(raw).to_bytes(8, "little") + raw)
    return "sha256:" + digest.hexdigest()


def tiny_manifest() -> dict:
    manifest = qwen25_7b_lora_manifest()
    manifest.pop("content_hash")
    manifest["model"].update(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )
    return validate_training_manifest(manifest)


def tiny_tpu_manifest() -> dict:
    manifest = qwen25_7b_lora_tpu_manifest()
    manifest.pop("content_hash")
    manifest["model"].update(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
    )
    return validate_training_manifest(manifest)


def jax_adapter_state(manifest: dict, stage_id: int) -> dict[str, np.ndarray]:
    stage = manifest["stages"][stage_id]
    dimensions = {
        "q_proj": (8, 8, "self_attn"),
        "k_proj": (8, 4, "self_attn"),
        "v_proj": (8, 4, "self_attn"),
        "o_proj": (8, 8, "self_attn"),
        "gate_proj": (8, 16, "mlp"),
        "up_proj": (8, 16, "mlp"),
        "down_proj": (16, 8, "mlp"),
    }
    tensors = {}
    for layer in range(stage["layer_start"], stage["layer_end"]):
        for target in manifest["lora"]["target_modules"]:
            input_size, output_size, owner = dimensions[target]
            prefix = f"model.layers.{layer}.{owner}.{target}"
            tensors[f"{prefix}.lora_A.weight"] = np.ones(
                (4, input_size), dtype=np.float32
            )
            tensors[f"{prefix}.lora_B.weight"] = np.ones(
                (output_size, 4), dtype=np.float32
            )
    return tensors


def write_checkpoint(root, *, generation: int = 3):
    manifest = tiny_manifest()
    stage = manifest["stages"][4]
    names = checkpoint_file_names(4)
    tensors = {}
    dimensions = {
        "q_proj": (8, 8, "self_attn"),
        "k_proj": (8, 4, "self_attn"),
        "v_proj": (8, 4, "self_attn"),
        "o_proj": (8, 8, "self_attn"),
        "gate_proj": (8, 16, "mlp"),
        "up_proj": (8, 16, "mlp"),
        "down_proj": (16, 8, "mlp"),
    }
    for layer in range(stage["layer_start"], stage["layer_end"]):
        for target in manifest["lora"]["target_modules"]:
            input_size, output_size, owner = dimensions[target]
            prefix = f"model.layers.{layer}.{owner}.{target}"
            tensors[f"{prefix}.lora_A.weight"] = torch.ones((4, input_size))
            tensors[f"{prefix}.lora_B.weight"] = torch.ones((output_size, 4))
    save_file(tensors, root / names["adapter"])
    torch.save({"state": {}, "param_groups": []}, root / names["optimizer"])
    torch.save({"last_epoch": 1, "_step_count": 2}, root / names["scheduler"])
    torch.save({}, root / names["scaler"])
    torch.save({"cpu": torch.random.get_rng_state()}, root / names["rng"])
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "stage_id": 4,
        "layer_start": stage["layer_start"],
        "layer_end": stage["layer_end"],
        "global_step": 1,
        "optimizer_step": 1,
        "scheduler_step": 1,
        "dataset_cursor": 1,
        "placement_generation": generation,
        "device_type": "cpu",
        "adapter_file": names["adapter"],
        "adapter_file_hash": sha(root / names["adapter"]),
        "adapter_tensor_hash": adapter_hash(tensors),
        "adapter_tensor_count": len(tensors),
        "optimizer_file": names["optimizer"],
        "optimizer_file_hash": sha(root / names["optimizer"]),
        "optimizer_state_present": True,
        "scheduler_file": names["scheduler"],
        "scheduler_file_hash": sha(root / names["scheduler"]),
        "scheduler_state_present": True,
        "grad_scaler_file": names["scaler"],
        "grad_scaler_file_hash": sha(root / names["scaler"]),
        "grad_scaler_state_present": True,
        "rng_file": names["rng"],
        "rng_file_hash": sha(root / names["rng"]),
        "rng_state_present": True,
        "tensor_values_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    value["content_hash"] = stable_hash(value)
    (root / names["manifest"]).write_text(json.dumps(value), encoding="utf-8")
    return manifest, value


def test_six_component_checkpoint_validates_scheduler_and_restores(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, checkpoint = write_checkpoint(source)

    archive, built = build_stage_checkpoint_archive(
        source, training_manifest=manifest, stage_id=4
    )
    report = validate_stage_checkpoint_archive(
        archive,
        training_manifest=manifest,
        expected_stage_id=4,
        expected_step=1,
        expected_dataset_cursor=1,
        expected_placement_generation=3,
    )
    restored = restore_stage_checkpoint_archive(
        archive,
        tmp_path / "restored",
        training_manifest=manifest,
        expected_stage_id=4,
        expected_step=1,
        expected_dataset_cursor=1,
    )

    assert built["archive_hash"] == report["archive_hash"]
    assert report["scheduler_state_present"] is True
    assert report["scheduler_safe_loaded"] is True
    assert report["adapter_tensors_finite"] is True
    assert report["placement_generation"] == 3
    assert restored["restored_file_count"] == 6
    assert len(list((tmp_path / "restored").iterdir())) == 6
    assert checkpoint["training_manifest_hash"] == manifest["content_hash"]


def test_checkpoint_rejects_stale_generation_and_unsafe_archive(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, _checkpoint = write_checkpoint(source)
    archive, _report = build_stage_checkpoint_archive(
        source, training_manifest=manifest, stage_id=4
    )

    with pytest.raises(ValueError, match="placement_generation_stale"):
        validate_stage_checkpoint_archive(
            archive,
            training_manifest=manifest,
            expected_placement_generation=4,
        )

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("../stage4_checkpoint.json", b"{}")
    with pytest.raises(ValueError, match="archive_entries_invalid"):
        validate_stage_checkpoint_archive(
            stream.getvalue(), training_manifest=manifest
        )


def test_checkpoint_rejects_adapter_shape_and_scheduler_corruption(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, checkpoint = write_checkpoint(source)
    names = checkpoint_file_names(4)
    from safetensors.torch import load_file

    tensors = load_file(source / names["adapter"])
    first = sorted(tensors)[0]
    tensors[first] = torch.ones((1, 1))
    save_file(tensors, source / names["adapter"])
    checkpoint["adapter_file_hash"] = sha(source / names["adapter"])
    checkpoint["adapter_tensor_hash"] = adapter_hash(tensors)
    checkpoint["content_hash"] = stable_hash(
        {key: value for key, value in checkpoint.items() if key != "content_hash"}
    )
    (source / names["manifest"]).write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter_shape_invalid"):
        build_stage_checkpoint_archive(
            source, training_manifest=manifest, stage_id=4
        )

    # Rebuild a valid checkpoint, then replace the scheduler with arbitrary bytes.
    source2 = tmp_path / "source2"
    source2.mkdir()
    manifest2, checkpoint2 = write_checkpoint(source2)
    (source2 / names["scheduler"]).write_bytes(b"not-a-safe-torch-state")
    checkpoint2["scheduler_file_hash"] = sha(source2 / names["scheduler"])
    checkpoint2["content_hash"] = stable_hash(
        {key: value for key, value in checkpoint2.items() if key != "content_hash"}
    )
    (source2 / names["manifest"]).write_text(json.dumps(checkpoint2), encoding="utf-8")
    with pytest.raises(ValueError, match="scheduler_state_invalid"):
        build_stage_checkpoint_archive(
            source2, training_manifest=manifest2, stage_id=4
        )


def test_jax_tpu_checkpoint_is_pickle_free_complete_and_reloadable(tmp_path) -> None:
    manifest = tiny_tpu_manifest()
    adapter = jax_adapter_state(manifest, 2)
    optimizer = {
        "step": 1,
        "exp_avg": {name: np.zeros_like(value) for name, value in adapter.items()},
        "exp_avg_sq": {
            name: np.zeros_like(value) for name, value in adapter.items()
        },
    }
    checkpoint = save_jax_stage_checkpoint(
        adapter,
        optimizer,
        {"last_epoch": 1, "learning_rate": 5e-4},
        np.asarray([7, 11], dtype=np.uint32),
        tmp_path / "jax",
        training_manifest=manifest,
        stage_spec=SimpleNamespace(stage_id=2),
        global_step=1,
        dataset_cursor=1,
        placement_generation=3,
        mesh_shape=[8],
    )

    archive, built = build_stage_checkpoint_archive(
        tmp_path / "jax", training_manifest=manifest, stage_id=2
    )
    validated = validate_stage_checkpoint_archive(
        archive,
        training_manifest=manifest,
        expected_stage_id=2,
        expected_step=1,
        expected_dataset_cursor=1,
        expected_placement_generation=3,
    )
    loaded = load_jax_stage_checkpoint(
        tmp_path / "jax",
        training_manifest=manifest,
        stage_spec=SimpleNamespace(stage_id=2),
    )

    assert checkpoint["runtime_backend"] == "jax_tpu"
    assert checkpoint["grad_scaler_state_present"] is False
    assert checkpoint["grad_scaler_state_applicable"] is False
    assert validated["archive_hash"] == built["archive_hash"]
    assert validated["optimizer_safe_loaded"] is True
    assert validated["scheduler_safe_loaded"] is True
    assert validated["rng_safe_loaded"] is True
    assert validated["jax_prng_state_present"] is True
    assert validated["jax_mesh_device_count"] == 8
    assert set(loaded["adapter_state"]) == set(adapter)
    assert loaded["optimizer_state"]["step"] == 1
    assert loaded["scheduler_state"]["last_epoch"] == 1
    assert loaded["prng_key"].tolist() == [7, 11]
