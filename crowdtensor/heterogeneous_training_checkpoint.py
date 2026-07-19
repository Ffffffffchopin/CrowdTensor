"""Validated checkpoints for manifest-driven heterogeneous Qwen stages."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .heterogeneous_training_manifest import (
    canonical_json,
    stable_hash,
    validate_training_manifest,
)


CHECKPOINT_SCHEMA = "crowdtensor_heterogeneous_qwen_stage_checkpoint_v1"
ARCHIVE_SCHEMA = "crowdtensor_heterogeneous_qwen_stage_checkpoint_archive_v1"
DEFAULT_MAX_CHECKPOINT_BYTES = 768 * 1024 * 1024
ADAPTER_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.lora_(?:A|B)(?:\.default)?\.weight$"
)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def checkpoint_file_names(stage_id: int) -> dict[str, str]:
    prefix = f"stage{int(stage_id)}"
    return {
        "manifest": f"{prefix}_checkpoint.json",
        "adapter": f"{prefix}_adapter.safetensors",
        "optimizer": f"{prefix}_optimizer.pt",
        "scheduler": f"{prefix}_scheduler.pt",
        "scaler": f"{prefix}_grad_scaler.pt",
        "rng": f"{prefix}_rng.pt",
    }


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage(manifest: dict[str, Any], stage_id: int) -> dict[str, Any]:
    try:
        stage = manifest["stages"][int(stage_id)]
    except (IndexError, TypeError) as exc:
        raise ValueError("heterogeneous_checkpoint_stage_invalid") from exc
    if int(stage["stage_id"]) != int(stage_id):
        raise ValueError("heterogeneous_checkpoint_stage_invalid")
    return dict(stage)


def _inspect_safe_state(value: bytes, *, kind: str) -> dict[str, Any]:
    try:
        import torch

        parsed = torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
    except BaseException as exc:
        raise ValueError(f"heterogeneous_checkpoint_{kind}_state_invalid") from exc
    seen: set[int] = set()
    tensor_count = 0
    tensor_bytes = 0

    def visit(item: Any, depth: int = 0) -> None:
        nonlocal tensor_count, tensor_bytes
        if depth > 32:
            raise ValueError(f"heterogeneous_checkpoint_{kind}_state_too_deep")
        if isinstance(item, torch.Tensor):
            tensor_count += 1
            tensor_bytes += int(item.numel()) * int(item.element_size())
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item).all().item()
            ):
                raise ValueError(f"heterogeneous_checkpoint_{kind}_state_non_finite")
            return
        if item is None or isinstance(item, (str, bytes, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"heterogeneous_checkpoint_{kind}_state_non_finite")
            return
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, dict):
            if len(item) > 1_000_000:
                raise ValueError(f"heterogeneous_checkpoint_{kind}_state_too_large")
            for key, child in item.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            if len(item) > 1_000_000:
                raise ValueError(f"heterogeneous_checkpoint_{kind}_state_too_large")
            for child in item:
                visit(child, depth + 1)
            return
        raise ValueError(f"heterogeneous_checkpoint_{kind}_state_type_invalid")

    visit(parsed)
    if kind == "optimizer" and (
        not isinstance(parsed, dict)
        or not isinstance(parsed.get("state"), dict)
        or not isinstance(parsed.get("param_groups"), list)
    ):
        raise ValueError("heterogeneous_checkpoint_optimizer_state_invalid")
    if kind == "scheduler" and not isinstance(parsed, dict):
        raise ValueError("heterogeneous_checkpoint_scheduler_state_invalid")
    if kind == "scaler" and not isinstance(parsed, dict):
        raise ValueError("heterogeneous_checkpoint_scaler_state_invalid")
    if kind == "rng" and (
        not isinstance(parsed, dict) or not isinstance(parsed.get("cpu"), torch.Tensor)
    ):
        raise ValueError("heterogeneous_checkpoint_rng_state_invalid")
    return {
        f"{kind}_safe_loaded": True,
        f"{kind}_tensor_count": tensor_count,
        f"{kind}_tensor_bytes": tensor_bytes,
    }


def _inspect_jax_state(
    value: bytes,
    *,
    kind: str,
    adapter_names: set[str],
) -> dict[str, Any]:
    """Validate JAX checkpoint components without pickle deserialization."""

    if kind in {"optimizer", "rng"}:
        try:
            from safetensors.torch import load

            tensors = dict(load(value))
        except BaseException as exc:
            raise ValueError(
                f"heterogeneous_checkpoint_jax_{kind}_state_invalid"
            ) from exc
        names = set(tensors)
        if kind == "optimizer":
            expected = {"step"}
            expected.update(f"exp_avg.{name}" for name in adapter_names)
            expected.update(f"exp_avg_sq.{name}" for name in adapter_names)
            if names != expected or list(tensors["step"].shape) not in ([], [1]):
                raise ValueError(
                    "heterogeneous_checkpoint_jax_optimizer_state_invalid"
                )
        elif names != {"jax_prng_key"} or list(
            tensors["jax_prng_key"].shape
        ) != [2]:
            raise ValueError("heterogeneous_checkpoint_jax_rng_state_invalid")
        tensor_bytes = 0
        for tensor in tensors.values():
            tensor_bytes += int(tensor.numel()) * int(tensor.element_size())
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                tensor.isfinite().all().item()
            ):
                raise ValueError(
                    f"heterogeneous_checkpoint_jax_{kind}_state_non_finite"
                )
        return {
            f"{kind}_safe_loaded": True,
            f"{kind}_tensor_count": len(tensors),
            f"{kind}_tensor_bytes": tensor_bytes,
            f"{kind}_encoding": "safetensors",
        }
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"heterogeneous_checkpoint_jax_{kind}_state_invalid"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"heterogeneous_checkpoint_jax_{kind}_state_invalid")
    if kind == "scheduler":
        if (
            str(parsed.get("scheduler") or "") != "constant"
            or int(parsed.get("last_epoch", -1)) < 1
            or not math.isfinite(float(parsed.get("learning_rate", float("nan"))))
        ):
            raise ValueError(
                "heterogeneous_checkpoint_jax_scheduler_state_invalid"
            )
    elif kind == "scaler":
        if parsed != {"applicable": False, "runtime_backend": "jax_tpu"}:
            raise ValueError("heterogeneous_checkpoint_jax_scaler_state_invalid")
    else:
        raise ValueError("heterogeneous_checkpoint_jax_state_kind_invalid")
    return {
        f"{kind}_safe_loaded": True,
        f"{kind}_tensor_count": 0,
        f"{kind}_tensor_bytes": 0,
        f"{kind}_encoding": "json",
    }


def _inspect_adapter(
    value: bytes,
    *,
    stage: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, Any]:
    try:
        from safetensors.torch import load

        tensors = dict(load(value))
    except BaseException as exc:
        raise ValueError("heterogeneous_checkpoint_adapter_invalid") from exc
    if not tensors:
        raise ValueError("heterogeneous_checkpoint_adapter_empty")
    manifest = validate_training_manifest(training_manifest)
    model = manifest["model"]
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    kv_width = hidden * int(model["num_key_value_heads"]) // int(
        model["num_attention_heads"]
    )
    rank = int(manifest["lora"]["rank"])
    dimensions = {
        "q_proj": (hidden, hidden, "self_attn"),
        "k_proj": (hidden, kv_width, "self_attn"),
        "v_proj": (hidden, kv_width, "self_attn"),
        "o_proj": (hidden, hidden, "self_attn"),
        "gate_proj": (hidden, intermediate, "mlp"),
        "up_proj": (hidden, intermediate, "mlp"),
        "down_proj": (intermediate, hidden, "mlp"),
    }
    expected_shapes = {}
    for layer in range(int(stage["layer_start"]), int(stage["layer_end"])):
        for target in manifest["lora"]["target_modules"]:
            input_size, output_size, owner = dimensions[str(target)]
            prefix = f"model.layers.{layer}.{owner}.{target}"
            expected_shapes[f"{prefix}.lora_A.weight"] = [rank, input_size]
            expected_shapes[f"{prefix}.lora_B.weight"] = [output_size, rank]
    names = sorted(tensors)
    canonical_names = set()
    for name, tensor in tensors.items():
        match = ADAPTER_PATTERN.fullmatch(str(name))
        if match is None:
            raise ValueError("heterogeneous_checkpoint_adapter_name_invalid")
        layer = int(match.group(1))
        if not int(stage["layer_start"]) <= layer < int(stage["layer_end"]):
            raise ValueError("heterogeneous_checkpoint_adapter_ownership_invalid")
        canonical_name = str(name).replace(".default.weight", ".weight")
        canonical_names.add(canonical_name)
        if list(tensor.shape) != expected_shapes.get(canonical_name):
            raise ValueError("heterogeneous_checkpoint_adapter_shape_invalid")
        if not bool(tensor.isfinite().all().item()):
            raise ValueError("heterogeneous_checkpoint_adapter_non_finite")
    if canonical_names != set(expected_shapes):
        raise ValueError("heterogeneous_checkpoint_adapter_coverage_invalid")
    digest = hashlib.sha256()
    for name in names:
        tensor = tensors[name].contiguous()
        raw = tensor.view(__import__("torch").uint8).numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(raw).to_bytes(8, "little") + raw)
    return {
        "adapter_tensor_count": len(tensors),
        "adapter_tensor_hash": "sha256:" + digest.hexdigest(),
        "adapter_tensor_names_hash": stable_hash(names),
        "adapter_tensors_finite": True,
        "adapter_ownership_validated": True,
    }


def validate_checkpoint_manifest(
    value: Any,
    *,
    training_manifest: dict[str, Any],
    expected_stage_id: int | None = None,
    expected_step: int | None = None,
    expected_dataset_cursor: int | None = None,
    expected_placement_generation: int | None = None,
) -> dict[str, Any]:
    manifest = validate_training_manifest(training_manifest)
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("heterogeneous_checkpoint_manifest_schema_invalid")
    result = dict(value)
    try:
        stage_id = int(result["stage_id"])
        global_step = int(result["global_step"])
        optimizer_step = int(result["optimizer_step"])
        scheduler_step = int(result["scheduler_step"])
        dataset_cursor = int(result["dataset_cursor"])
        generation = int(result["placement_generation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("heterogeneous_checkpoint_progress_invalid") from exc
    stage = _stage(manifest, stage_id)
    if (
        result.get("training_manifest_hash") != manifest["content_hash"]
        or result.get("model_id") != manifest["model"]["model_id"]
        or result.get("model_revision") != manifest["model"]["model_revision"]
        or int(result.get("layer_start", -1)) != int(stage["layer_start"])
        or int(result.get("layer_end", -1)) != int(stage["layer_end"])
    ):
        raise ValueError("heterogeneous_checkpoint_ownership_invalid")
    if (
        global_step < 1
        or optimizer_step != global_step
        or scheduler_step != global_step
        or dataset_cursor < 1
        or generation < 1
    ):
        raise ValueError("heterogeneous_checkpoint_progress_invalid")
    if expected_stage_id is not None and stage_id != int(expected_stage_id):
        raise ValueError("heterogeneous_checkpoint_stage_mismatch")
    if expected_step is not None and global_step != int(expected_step):
        raise ValueError("heterogeneous_checkpoint_step_mismatch")
    if expected_dataset_cursor is not None and dataset_cursor != int(
        expected_dataset_cursor
    ):
        raise ValueError("heterogeneous_checkpoint_cursor_mismatch")
    if expected_placement_generation is not None and generation != int(
        expected_placement_generation
    ):
        raise ValueError("heterogeneous_checkpoint_placement_generation_stale")
    names = checkpoint_file_names(stage_id)
    fields = {
        "adapter": "adapter_file",
        "optimizer": "optimizer_file",
        "scheduler": "scheduler_file",
        "scaler": "grad_scaler_file",
        "rng": "rng_file",
    }
    if any(str(result.get(field) or "") != names[kind] for kind, field in fields.items()):
        raise ValueError("heterogeneous_checkpoint_component_name_invalid")
    hash_fields = [
        "adapter_file_hash",
        "adapter_tensor_hash",
        "optimizer_file_hash",
        "scheduler_file_hash",
        "grad_scaler_file_hash",
        "rng_file_hash",
        "content_hash",
    ]
    if any(not str(result.get(field) or "").startswith("sha256:") for field in hash_fields):
        raise ValueError("heterogeneous_checkpoint_hash_contract_invalid")
    content_hash = str(result["content_hash"])
    if stable_hash({key: item for key, item in result.items() if key != "content_hash"}) != content_hash:
        raise ValueError("heterogeneous_checkpoint_manifest_content_hash_invalid")
    runtime_backend = str(result.get("runtime_backend") or "pytorch")
    if runtime_backend not in {"pytorch", "jax_tpu"}:
        raise ValueError("heterogeneous_checkpoint_runtime_backend_invalid")
    common_complete = all(
        result.get(field) is True
        for field in ("optimizer_state_present", "scheduler_state_present", "rng_state_present")
    )
    if runtime_backend == "jax_tpu":
        encodings = dict(result.get("component_encodings") or {})
        jax_complete = bool(
            result.get("device_type") == "jax_tpu"
            and result.get("grad_scaler_state_present") is False
            and result.get("grad_scaler_state_applicable") is False
            and result.get("jax_prng_state_present") is True
            and encodings
            == {
                "adapter": "safetensors",
                "optimizer": "safetensors",
                "rng": "safetensors",
                "scaler": "json_not_applicable",
                "scheduler": "json",
            }
        )
        if not common_complete or not jax_complete:
            raise ValueError("heterogeneous_checkpoint_jax_resume_state_incomplete")
    elif not common_complete or result.get("grad_scaler_state_present") is not True:
        raise ValueError("heterogeneous_checkpoint_resume_state_incomplete")
    return result


def save_stage_checkpoint(
    module: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    checkpoint_dir: str | Path,
    *,
    training_manifest: dict[str, Any],
    stage_spec: Any,
    global_step: int,
    dataset_cursor: int,
    placement_generation: int,
    device: str,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    from .qwen15b_training import qwen_stage_adapter_hash, qwen_stage_adapter_state

    manifest = validate_training_manifest(training_manifest)
    stage_id = int(stage_spec.stage_id)
    stage = _stage(manifest, stage_id)
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    names = checkpoint_file_names(stage_id)
    paths = {key: root / name for key, name in names.items()}
    adapter = qwen_stage_adapter_state(module)
    save_file(adapter, str(paths["adapter"]))
    torch.save(optimizer.state_dict(), paths["optimizer"])
    torch.save(scheduler.state_dict(), paths["scheduler"])
    torch.save(scaler.state_dict(), paths["scaler"])
    rng: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    target = torch.device(device)
    if target.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state(target)
    torch.save(rng, paths["rng"])
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "stage_id": stage_id,
        "layer_start": int(stage["layer_start"]),
        "layer_end": int(stage["layer_end"]),
        "global_step": int(global_step),
        "optimizer_step": int(global_step),
        "scheduler_step": int(global_step),
        "dataset_cursor": int(dataset_cursor),
        "placement_generation": int(placement_generation),
        "device_type": target.type,
        "adapter_file": names["adapter"],
        "adapter_file_hash": _sha256_file(paths["adapter"]),
        "adapter_tensor_hash": qwen_stage_adapter_hash(module),
        "adapter_tensor_count": len(adapter),
        "optimizer_file": names["optimizer"],
        "optimizer_file_hash": _sha256_file(paths["optimizer"]),
        "optimizer_state_present": True,
        "scheduler_file": names["scheduler"],
        "scheduler_file_hash": _sha256_file(paths["scheduler"]),
        "scheduler_state_present": True,
        "grad_scaler_file": names["scaler"],
        "grad_scaler_file_hash": _sha256_file(paths["scaler"]),
        "grad_scaler_state_present": True,
        "rng_file": names["rng"],
        "rng_file_hash": _sha256_file(paths["rng"]),
        "rng_state_present": True,
        "tensor_values_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    value["content_hash"] = stable_hash(value)
    paths["manifest"].write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["manifest"].chmod(0o600)
    return {**value, "manifest_path": str(paths["manifest"].resolve())}


def load_stage_checkpoint(
    module: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    checkpoint_dir: str | Path,
    *,
    training_manifest: dict[str, Any],
    stage_spec: Any,
    device: str,
) -> dict[str, Any]:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from .qwen15b_training import qwen_stage_adapter_hash

    manifest = validate_training_manifest(training_manifest)
    stage_id = int(stage_spec.stage_id)
    root = Path(checkpoint_dir)
    names = checkpoint_file_names(stage_id)
    parsed = json.loads((root / names["manifest"]).read_text(encoding="utf-8"))
    checkpoint = validate_checkpoint_manifest(
        parsed,
        training_manifest=manifest,
        expected_stage_id=stage_id,
    )
    components = {
        "adapter": (root / names["adapter"], checkpoint["adapter_file_hash"]),
        "optimizer": (root / names["optimizer"], checkpoint["optimizer_file_hash"]),
        "scheduler": (root / names["scheduler"], checkpoint["scheduler_file_hash"]),
        "scaler": (root / names["scaler"], checkpoint["grad_scaler_file_hash"]),
        "rng": (root / names["rng"], checkpoint["rng_file_hash"]),
    }
    if any(_sha256_file(path) != digest for path, digest in components.values()):
        raise ValueError("heterogeneous_checkpoint_component_hash_invalid")
    adapter = load_file(str(components["adapter"][0]), device="cpu")
    incompatible = set_peft_model_state_dict(module, adapter, adapter_name="default")
    if list(getattr(incompatible, "unexpected_keys", []) or []):
        raise ValueError("heterogeneous_checkpoint_adapter_keys_invalid")
    optimizer.load_state_dict(
        torch.load(components["optimizer"][0], map_location="cpu", weights_only=True)
    )
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, item in list(state.items()):
            if isinstance(item, torch.Tensor):
                state[key] = item.to(target)
    scheduler.load_state_dict(
        torch.load(components["scheduler"][0], map_location="cpu", weights_only=True)
    )
    scaler.load_state_dict(
        torch.load(components["scaler"][0], map_location="cpu", weights_only=True)
    )
    rng = torch.load(components["rng"][0], map_location="cpu", weights_only=True)
    torch.random.set_rng_state(rng["cpu"])
    if target.type == "cuda" and "cuda" in rng:
        torch.cuda.set_rng_state(rng["cuda"], target)
    if qwen_stage_adapter_hash(module) != checkpoint["adapter_tensor_hash"]:
        raise ValueError("heterogeneous_checkpoint_adapter_restore_hash_invalid")
    return {**checkpoint, "manifest_path": str((root / names["manifest"]).resolve())}


def _jax_host_array(value: Any) -> Any:
    import numpy as np

    try:
        import jax

        value = jax.device_get(value)
    except ImportError:
        pass
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def save_jax_stage_checkpoint(
    adapter_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any],
    prng_key: Any,
    checkpoint_dir: str | Path,
    *,
    training_manifest: dict[str, Any],
    stage_spec: Any,
    global_step: int,
    dataset_cursor: int,
    placement_generation: int,
    mesh_shape: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Save a pickle-free JAX TPU checkpoint in the shared six-file archive."""

    import numpy as np
    from safetensors.numpy import save_file

    manifest = validate_training_manifest(training_manifest)
    stage_id = int(stage_spec.stage_id)
    stage = _stage(manifest, stage_id)
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    names = checkpoint_file_names(stage_id)
    paths = {key: root / name for key, name in names.items()}
    adapter = {
        str(name): np.ascontiguousarray(_jax_host_array(tensor), dtype=np.float32)
        for name, tensor in adapter_state.items()
    }
    if not adapter:
        raise ValueError("heterogeneous_checkpoint_jax_adapter_empty")
    exp_avg = dict(optimizer_state.get("exp_avg") or {})
    exp_avg_sq = dict(optimizer_state.get("exp_avg_sq") or {})
    if set(exp_avg) != set(adapter) or set(exp_avg_sq) != set(adapter):
        raise ValueError("heterogeneous_checkpoint_jax_optimizer_coverage_invalid")
    optimizer_tensors: dict[str, Any] = {
        "step": np.asarray([int(optimizer_state.get("step") or 0)], dtype=np.int64)
    }
    for name in sorted(adapter):
        optimizer_tensors[f"exp_avg.{name}"] = np.ascontiguousarray(
            _jax_host_array(exp_avg[name]), dtype=np.float32
        )
        optimizer_tensors[f"exp_avg_sq.{name}"] = np.ascontiguousarray(
            _jax_host_array(exp_avg_sq[name]), dtype=np.float32
        )
    save_file(adapter, str(paths["adapter"]))
    save_file(optimizer_tensors, str(paths["optimizer"]))
    scheduler_payload = {
        "scheduler": "constant",
        "last_epoch": int(scheduler_state.get("last_epoch") or global_step),
        "learning_rate": float(scheduler_state["learning_rate"]),
    }
    _atomic_write(paths["scheduler"], canonical_json(scheduler_payload))
    _atomic_write(
        paths["scaler"],
        canonical_json({"applicable": False, "runtime_backend": "jax_tpu"}),
    )
    key = np.ascontiguousarray(_jax_host_array(prng_key), dtype=np.uint32).reshape(-1)
    if list(key.shape) != [2]:
        raise ValueError("heterogeneous_checkpoint_jax_prng_shape_invalid")
    save_file({"jax_prng_key": key}, str(paths["rng"]))
    for path in paths.values():
        if path.is_file():
            path.chmod(0o600)
    adapter_validation = _inspect_adapter(
        paths["adapter"].read_bytes(),
        stage=stage,
        training_manifest=manifest,
    )
    resolved_mesh_shape = [int(item) for item in mesh_shape]
    if not resolved_mesh_shape or math.prod(resolved_mesh_shape) < 1:
        raise ValueError("heterogeneous_checkpoint_jax_mesh_invalid")
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "stage_id": stage_id,
        "layer_start": int(stage["layer_start"]),
        "layer_end": int(stage["layer_end"]),
        "global_step": int(global_step),
        "optimizer_step": int(global_step),
        "scheduler_step": int(global_step),
        "dataset_cursor": int(dataset_cursor),
        "placement_generation": int(placement_generation),
        "device_type": "jax_tpu",
        "runtime_backend": "jax_tpu",
        "adapter_file": names["adapter"],
        "adapter_file_hash": _sha256_file(paths["adapter"]),
        "adapter_tensor_hash": adapter_validation["adapter_tensor_hash"],
        "adapter_tensor_count": len(adapter),
        "optimizer_file": names["optimizer"],
        "optimizer_file_hash": _sha256_file(paths["optimizer"]),
        "optimizer_state_present": True,
        "scheduler_file": names["scheduler"],
        "scheduler_file_hash": _sha256_file(paths["scheduler"]),
        "scheduler_state_present": True,
        "grad_scaler_file": names["scaler"],
        "grad_scaler_file_hash": _sha256_file(paths["scaler"]),
        "grad_scaler_state_present": False,
        "grad_scaler_state_applicable": False,
        "rng_file": names["rng"],
        "rng_file_hash": _sha256_file(paths["rng"]),
        "rng_state_present": True,
        "jax_prng_state_present": True,
        "jax_mesh_device_count": math.prod(resolved_mesh_shape),
        "jax_mesh_shape": resolved_mesh_shape,
        "parameter_sharding": "named_mesh_model_axis",
        "component_encodings": {
            "adapter": "safetensors",
            "optimizer": "safetensors",
            "rng": "safetensors",
            "scaler": "json_not_applicable",
            "scheduler": "json",
        },
        "tensor_values_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    value["content_hash"] = stable_hash(value)
    _atomic_write(paths["manifest"], json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return {**value, "manifest_path": str(paths["manifest"].resolve())}


def load_jax_stage_checkpoint(
    checkpoint_dir: str | Path,
    *,
    training_manifest: dict[str, Any],
    stage_spec: Any,
) -> dict[str, Any]:
    """Load and validate host-side JAX state for sharded device placement."""

    from safetensors.numpy import load_file

    manifest = validate_training_manifest(training_manifest)
    stage_id = int(stage_spec.stage_id)
    root = Path(checkpoint_dir)
    names = checkpoint_file_names(stage_id)
    parsed = json.loads((root / names["manifest"]).read_text(encoding="utf-8"))
    checkpoint = validate_checkpoint_manifest(
        parsed,
        training_manifest=manifest,
        expected_stage_id=stage_id,
    )
    if checkpoint.get("runtime_backend") != "jax_tpu":
        raise ValueError("heterogeneous_checkpoint_jax_runtime_mismatch")
    hash_fields = {
        "adapter": "adapter_file_hash",
        "optimizer": "optimizer_file_hash",
        "scheduler": "scheduler_file_hash",
        "scaler": "grad_scaler_file_hash",
        "rng": "rng_file_hash",
    }
    paths = {kind: root / names[kind] for kind in hash_fields}
    if any(_sha256_file(paths[kind]) != checkpoint[field] for kind, field in hash_fields.items()):
        raise ValueError("heterogeneous_checkpoint_component_hash_invalid")
    stage = _stage(manifest, stage_id)
    adapter_validation = _inspect_adapter(
        paths["adapter"].read_bytes(),
        stage=stage,
        training_manifest=manifest,
    )
    if adapter_validation["adapter_tensor_hash"] != checkpoint["adapter_tensor_hash"]:
        raise ValueError("heterogeneous_checkpoint_adapter_restore_hash_invalid")
    adapter = dict(load_file(str(paths["adapter"])))
    optimizer_tensors = dict(load_file(str(paths["optimizer"])))
    _inspect_jax_state(
        paths["optimizer"].read_bytes(), kind="optimizer", adapter_names=set(adapter)
    )
    _inspect_jax_state(
        paths["scheduler"].read_bytes(), kind="scheduler", adapter_names=set(adapter)
    )
    _inspect_jax_state(
        paths["scaler"].read_bytes(), kind="scaler", adapter_names=set(adapter)
    )
    _inspect_jax_state(
        paths["rng"].read_bytes(), kind="rng", adapter_names=set(adapter)
    )
    scheduler = json.loads(paths["scheduler"].read_text(encoding="utf-8"))
    rng = dict(load_file(str(paths["rng"])))
    return {
        **checkpoint,
        "adapter_state": adapter,
        "optimizer_state": {
            "step": int(optimizer_tensors.pop("step").reshape(-1)[0]),
            "exp_avg": {
                name: optimizer_tensors[f"exp_avg.{name}"] for name in adapter
            },
            "exp_avg_sq": {
                name: optimizer_tensors[f"exp_avg_sq.{name}"] for name in adapter
            },
        },
        "scheduler_state": scheduler,
        "prng_key": rng["jax_prng_key"],
        "manifest_path": str((root / names["manifest"]).resolve()),
    }


def _inspect_archive(
    archive: bytes,
    *,
    training_manifest: dict[str, Any],
    expected_stage_id: int | None = None,
    expected_step: int | None = None,
    expected_dataset_cursor: int | None = None,
    expected_placement_generation: int | None = None,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    validate_tensor_payloads: bool = True,
    include_files: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    if not archive or len(archive) > int(max_checkpoint_bytes):
        raise ValueError("heterogeneous_checkpoint_archive_size_invalid")
    try:
        bundle = zipfile.ZipFile(io.BytesIO(archive), mode="r")
    except zipfile.BadZipFile as exc:
        raise ValueError("heterogeneous_checkpoint_archive_invalid") from exc
    with bundle:
        infos = bundle.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or len(names) != 6:
            raise ValueError("heterogeneous_checkpoint_archive_entries_invalid")
        total_size = 0
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = int(info.external_attr >> 16)
            if (
                path.is_absolute()
                or len(path.parts) != 1
                or path.name in {"", ".", ".."}
                or "\\" in info.filename
                or stat.S_ISLNK(mode)
                or info.flag_bits & 0x1
            ):
                raise ValueError("heterogeneous_checkpoint_archive_path_invalid")
            total_size += int(info.file_size)
        if total_size > int(max_checkpoint_bytes):
            raise ValueError("heterogeneous_checkpoint_archive_unpacked_size_invalid")
        manifest_names = [item for item in names if item.endswith("_checkpoint.json")]
        if len(manifest_names) != 1:
            raise ValueError("heterogeneous_checkpoint_archive_manifest_missing")
        raw_manifest = bundle.read(manifest_names[0])
        if len(raw_manifest) > 1024 * 1024:
            raise ValueError("heterogeneous_checkpoint_manifest_size_invalid")
        try:
            parsed = json.loads(raw_manifest.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("heterogeneous_checkpoint_manifest_json_invalid") from exc
        checkpoint = validate_checkpoint_manifest(
            parsed,
            training_manifest=training_manifest,
            expected_stage_id=expected_stage_id,
            expected_step=expected_step,
            expected_dataset_cursor=expected_dataset_cursor,
            expected_placement_generation=expected_placement_generation,
        )
        stage = _stage(validate_training_manifest(training_manifest), int(checkpoint["stage_id"]))
        expected_names = checkpoint_file_names(int(checkpoint["stage_id"]))
        if set(names) != set(expected_names.values()):
            raise ValueError("heterogeneous_checkpoint_archive_entries_invalid")
        hash_fields = {
            "adapter": "adapter_file_hash",
            "optimizer": "optimizer_file_hash",
            "scheduler": "scheduler_file_hash",
            "scaler": "grad_scaler_file_hash",
            "rng": "rng_file_hash",
        }
        components = {}
        for kind, hash_field in hash_fields.items():
            raw = bundle.read(expected_names[kind])
            if _sha256_bytes(raw) != checkpoint[hash_field]:
                raise ValueError("heterogeneous_checkpoint_component_hash_invalid")
            components[kind] = raw
        validation: dict[str, Any] = {
            "tensor_payload_validation_enabled": bool(validate_tensor_payloads)
        }
        if validate_tensor_payloads:
            adapter = _inspect_adapter(
                components["adapter"],
                stage=stage,
                training_manifest=training_manifest,
            )
            if adapter["adapter_tensor_hash"] != checkpoint["adapter_tensor_hash"]:
                raise ValueError("heterogeneous_checkpoint_adapter_hash_invalid")
            validation.update(adapter)
            runtime_backend = str(checkpoint.get("runtime_backend") or "pytorch")
            if runtime_backend == "jax_tpu":
                from safetensors.torch import load as load_safetensors

                adapter_names = set(load_safetensors(components["adapter"]))
                for kind in ("optimizer", "scheduler", "scaler", "rng"):
                    validation.update(
                        _inspect_jax_state(
                            components[kind],
                            kind=kind,
                            adapter_names=adapter_names,
                        )
                    )
            else:
                for kind in ("optimizer", "scheduler", "scaler", "rng"):
                    validation.update(_inspect_safe_state(components[kind], kind=kind))
        files = {}
        if include_files:
            files = {
                expected_names["manifest"]: raw_manifest,
                **{expected_names[key]: value for key, value in components.items()},
            }
    component_hashes = {
        kind: checkpoint[field] for kind, field in hash_fields.items()
    }
    report = {
        "schema": ARCHIVE_SCHEMA,
        "archive_hash": _sha256_bytes(archive),
        "archive_bytes": len(archive),
        "stage_id": int(checkpoint["stage_id"]),
        "layer_start": int(checkpoint["layer_start"]),
        "layer_end": int(checkpoint["layer_end"]),
        "model_id": str(checkpoint["model_id"]),
        "model_revision": str(checkpoint["model_revision"]),
        "training_manifest_hash": str(checkpoint["training_manifest_hash"]),
        "placement_generation": int(checkpoint["placement_generation"]),
        "global_step": int(checkpoint["global_step"]),
        "optimizer_step": int(checkpoint["optimizer_step"]),
        "scheduler_step": int(checkpoint["scheduler_step"]),
        "dataset_cursor": int(checkpoint["dataset_cursor"]),
        "checkpoint_content_hash": str(checkpoint["content_hash"]),
        "adapter_tensor_hash": str(checkpoint["adapter_tensor_hash"]),
        "component_hashes_hash": stable_hash(component_hashes),
        "optimizer_state_present": True,
        "scheduler_state_present": True,
        "grad_scaler_state_present": checkpoint["grad_scaler_state_present"]
        is True,
        "grad_scaler_state_applicable": bool(
            checkpoint.get("grad_scaler_state_applicable", True)
        ),
        "rng_state_present": True,
        **validation,
        "archive_paths_validated": True,
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if str(checkpoint.get("runtime_backend") or "pytorch") == "jax_tpu":
        report.update(
            {
                "runtime_backend": "jax_tpu",
                "jax_prng_state_present": checkpoint.get(
                    "jax_prng_state_present"
                )
                is True,
                "jax_mesh_device_count": int(
                    checkpoint.get("jax_mesh_device_count") or 0
                ),
                "jax_mesh_shape": list(checkpoint.get("jax_mesh_shape") or []),
                "parameter_sharding": str(
                    checkpoint.get("parameter_sharding") or ""
                ),
            }
        )
    report["content_hash"] = stable_hash(report)
    return report, checkpoint, files


def validate_stage_checkpoint_archive(
    archive: bytes,
    *,
    training_manifest: dict[str, Any],
    expected_stage_id: int | None = None,
    expected_step: int | None = None,
    expected_dataset_cursor: int | None = None,
    expected_placement_generation: int | None = None,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    validate_tensor_payloads: bool = True,
) -> dict[str, Any]:
    report, _checkpoint, _files = _inspect_archive(
        archive,
        training_manifest=training_manifest,
        expected_stage_id=expected_stage_id,
        expected_step=expected_step,
        expected_dataset_cursor=expected_dataset_cursor,
        expected_placement_generation=expected_placement_generation,
        max_checkpoint_bytes=max_checkpoint_bytes,
        validate_tensor_payloads=validate_tensor_payloads,
    )
    return report


def build_stage_checkpoint_archive(
    checkpoint_dir: str | Path,
    *,
    training_manifest: dict[str, Any],
    stage_id: int,
    output_path: str | Path | None = None,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    root = Path(checkpoint_dir)
    names = checkpoint_file_names(stage_id)
    manifest_path = root / names["manifest"]
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("heterogeneous_checkpoint_manifest_unreadable") from exc
    checkpoint = validate_checkpoint_manifest(
        parsed,
        training_manifest=training_manifest,
        expected_stage_id=stage_id,
    )
    hash_fields = {
        "adapter": "adapter_file_hash",
        "optimizer": "optimizer_file_hash",
        "scheduler": "scheduler_file_hash",
        "scaler": "grad_scaler_file_hash",
        "rng": "rng_file_hash",
    }
    values = {names["manifest"]: manifest_path.read_bytes()}
    for kind, field in hash_fields.items():
        path = root / names[kind]
        if not path.is_file() or _sha256_file(path) != checkpoint[field]:
            raise ValueError("heterogeneous_checkpoint_component_hash_invalid")
        values[names[kind]] = path.read_bytes()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as bundle:
        for name in sorted(values):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100600 << 16
            bundle.writestr(info, values[name])
    archive = stream.getvalue()
    report = validate_stage_checkpoint_archive(
        archive,
        training_manifest=training_manifest,
        expected_stage_id=stage_id,
        expected_step=int(checkpoint["global_step"]),
        expected_dataset_cursor=int(checkpoint["dataset_cursor"]),
        expected_placement_generation=int(checkpoint["placement_generation"]),
        max_checkpoint_bytes=max_checkpoint_bytes,
    )
    if output_path is not None:
        _atomic_write(Path(output_path), archive)
    return archive, report


def restore_stage_checkpoint_archive(
    archive: bytes,
    checkpoint_dir: str | Path,
    *,
    training_manifest: dict[str, Any],
    expected_stage_id: int,
    expected_step: int,
    expected_dataset_cursor: int,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
) -> dict[str, Any]:
    report, _checkpoint, files = _inspect_archive(
        archive,
        training_manifest=training_manifest,
        expected_stage_id=expected_stage_id,
        expected_step=expected_step,
        expected_dataset_cursor=expected_dataset_cursor,
        max_checkpoint_bytes=max_checkpoint_bytes,
        validate_tensor_payloads=True,
        include_files=True,
    )
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        _atomic_write(root / name, value)
    return {
        **report,
        "restored": True,
        "restored_file_count": len(files),
        "private_paths_public": False,
    }
