"""Manifest contract for resource-aware heterogeneous pipeline training."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "crowdtensor_heterogeneous_training_manifest_v1"
TPU_MANIFEST_SCHEMA = "crowdtensor_heterogeneous_training_manifest_v2"
QWEN25_7B_MODEL_ID = "Qwen/Qwen2.5-7B"
QWEN25_7B_MODEL_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
QWEN25_7B_PARAMETER_COUNT = 7_615_616_000
QWEN25_7B_WEIGHT_BYTES = 15_231_233_024

SUPPORTED_DEVICE_TYPES = {"cpu", "cuda"}
TPU_SUPPORTED_DEVICE_TYPES = {"cpu", "cuda", "jax_tpu"}
SUPPORTED_DTYPES = {"float16", "bfloat16", "float32"}
SUPPORTED_PLACEMENT_POLICIES = {"memory-performance", "memory-first"}
SUPPORTED_REBALANCE_REASONS = {
    "initial_placement",
    "miner_joined",
    "miner_left",
    "lease_expired",
    "device_oom",
    "straggler_detected",
    "owner_requested",
    "checkpoint_recovery",
}


class TrainingManifestError(ValueError):
    """Raised when a training manifest violates the public contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingManifestError(f"heterogeneous_manifest_{field}_invalid")
    return dict(value)


def _require_string(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise TrainingManifestError(f"heterogeneous_manifest_{field}_required")
    return result


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TrainingManifestError(f"heterogeneous_manifest_{field}_invalid")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingManifestError(
            f"heterogeneous_manifest_{field}_invalid"
        ) from exc
    if result < minimum:
        raise TrainingManifestError(f"heterogeneous_manifest_{field}_invalid")
    return result


def _require_float(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise TrainingManifestError(f"heterogeneous_manifest_{field}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingManifestError(
            f"heterogeneous_manifest_{field}_invalid"
        ) from exc
    if not math.isfinite(result) or result < minimum:
        raise TrainingManifestError(f"heterogeneous_manifest_{field}_invalid")
    return result


def _validate_model(value: Any) -> dict[str, Any]:
    model = _require_dict(value, "model")
    result = {
        "model_id": _require_string(model.get("model_id"), "model_id"),
        "model_revision": _require_string(
            model.get("model_revision"), "model_revision"
        ),
        "architecture": _require_string(model.get("architecture"), "architecture"),
        "model_type": _require_string(model.get("model_type"), "model_type"),
        "parameter_count": _require_int(
            model.get("parameter_count"), "parameter_count", minimum=1
        ),
        "weight_bytes": _require_int(
            model.get("weight_bytes"), "weight_bytes", minimum=1
        ),
        "num_hidden_layers": _require_int(
            model.get("num_hidden_layers"), "num_hidden_layers", minimum=1
        ),
        "hidden_size": _require_int(
            model.get("hidden_size"), "hidden_size", minimum=1
        ),
        "intermediate_size": _require_int(
            model.get("intermediate_size"), "intermediate_size", minimum=1
        ),
        "num_attention_heads": _require_int(
            model.get("num_attention_heads"), "num_attention_heads", minimum=1
        ),
        "num_key_value_heads": _require_int(
            model.get("num_key_value_heads"), "num_key_value_heads", minimum=1
        ),
        "vocab_size": _require_int(
            model.get("vocab_size"), "vocab_size", minimum=1
        ),
        "source_dtype": _require_string(
            model.get("source_dtype"), "source_dtype"
        ).lower(),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }
    if result["source_dtype"] not in SUPPORTED_DTYPES:
        raise TrainingManifestError("heterogeneous_manifest_source_dtype_invalid")
    if result["model_type"] != "qwen2":
        raise TrainingManifestError("heterogeneous_manifest_model_type_unsupported")
    return result


def _validate_lora(value: Any) -> dict[str, Any]:
    lora = _require_dict(value, "lora")
    targets = sorted(
        {_require_string(item, "lora_target_module") for item in lora.get("target_modules") or []}
    )
    if not targets:
        raise TrainingManifestError("heterogeneous_manifest_lora_targets_required")
    bias = str(lora.get("bias") or "none")
    if bias != "none":
        raise TrainingManifestError("heterogeneous_manifest_lora_bias_unsupported")
    return {
        "rank": _require_int(lora.get("rank"), "lora_rank", minimum=1),
        "alpha": _require_int(lora.get("alpha"), "lora_alpha", minimum=1),
        "dropout": _require_float(lora.get("dropout", 0.0), "lora_dropout"),
        "target_modules": targets,
        "bias": bias,
        "learning_rate": _require_float(
            lora.get("learning_rate"), "learning_rate", minimum=1e-12
        ),
        "gradient_clip_norm": _require_float(
            lora.get("gradient_clip_norm", 1.0),
            "gradient_clip_norm",
            minimum=1e-12,
        ),
    }


def _validate_dataset(value: Any) -> dict[str, Any]:
    dataset = _require_dict(value, "dataset")
    return {
        "dataset_id": _require_string(dataset.get("dataset_id"), "dataset_id"),
        "dataset_revision": _require_string(
            dataset.get("dataset_revision"), "dataset_revision"
        ),
        "dataset_config": _require_string(
            dataset.get("dataset_config"), "dataset_config"
        ),
        "train_split": _require_string(
            dataset.get("train_split", "train"), "train_split"
        ),
        "validation_split": _require_string(
            dataset.get("validation_split", "validation"), "validation_split"
        ),
        "data_seed": _require_int(dataset.get("data_seed", 0), "data_seed"),
        "raw_training_text_public": False,
        "token_ids_public": False,
    }


def _validate_precision(value: Any, *, schema: str) -> dict[str, Any]:
    precision = _require_dict(value, "precision")
    result = {}
    for key in (
        "cuda_compute_dtype",
        "cpu_compute_dtype",
        "boundary_dtype",
        "optimizer_dtype",
    ):
        dtype = _require_string(precision.get(key), key).lower()
        if dtype not in SUPPORTED_DTYPES:
            raise TrainingManifestError(f"heterogeneous_manifest_{key}_invalid")
        result[key] = dtype
    if schema == TPU_MANIFEST_SCHEMA:
        dtype = _require_string(
            precision.get("jax_tpu_compute_dtype"), "jax_tpu_compute_dtype"
        ).lower()
        if dtype not in SUPPORTED_DTYPES:
            raise TrainingManifestError(
                "heterogeneous_manifest_jax_tpu_compute_dtype_invalid"
            )
        result["jax_tpu_compute_dtype"] = dtype
    return result


def _validate_training(value: Any) -> dict[str, Any]:
    training = _require_dict(value, "training")
    target_steps = _require_int(training.get("target_steps"), "target_steps", minimum=1)
    microbatches = _require_int(
        training.get("microbatches_per_step"),
        "microbatches_per_step",
        minimum=1,
    )
    batch_size = _require_int(
        training.get("microbatch_size"), "microbatch_size", minimum=1
    )
    sequence_length = _require_int(
        training.get("sequence_length"), "sequence_length", minimum=2
    )
    return {
        "target_steps": target_steps,
        "microbatches_per_step": microbatches,
        "microbatch_size": batch_size,
        "sequence_length": sequence_length,
        "gradient_accumulation_steps": _require_int(
            training.get("gradient_accumulation_steps", microbatches),
            "gradient_accumulation_steps",
            minimum=1,
        ),
        "seed": _require_int(training.get("seed", 0), "training_seed"),
    }


def _validate_checkpoint(value: Any) -> dict[str, Any]:
    checkpoint = _require_dict(value, "checkpoint")
    backend = str(checkpoint.get("backend") or "local")
    if backend not in {"local", "s3"}:
        raise TrainingManifestError("heterogeneous_manifest_checkpoint_backend_invalid")
    return {
        "backend": backend,
        "retention_steps": _require_int(
            checkpoint.get("retention_steps", 2),
            "checkpoint_retention_steps",
            minimum=1,
        ),
        "checkpoint_every_steps": _require_int(
            checkpoint.get("checkpoint_every_steps", 1),
            "checkpoint_every_steps",
            minimum=1,
        ),
        "include_optimizer": bool(checkpoint.get("include_optimizer", True)),
        "include_scheduler": bool(checkpoint.get("include_scheduler", True)),
        "include_rng": bool(checkpoint.get("include_rng", True)),
        "atomic_global_commit": bool(checkpoint.get("atomic_global_commit", True)),
    }


def _validate_scheduler(value: Any, *, schema: str) -> dict[str, Any]:
    scheduler = _require_dict(value, "scheduler")
    policy = str(scheduler.get("placement_policy") or "memory-performance")
    if policy not in SUPPORTED_PLACEMENT_POLICIES:
        raise TrainingManifestError("heterogeneous_manifest_placement_policy_invalid")
    device_policy = str(scheduler.get("device_policy") or "mixed")
    if device_policy not in {"cpu", "gpu", "mixed"}:
        raise TrainingManifestError("heterogeneous_manifest_device_policy_invalid")
    reserve_fraction = _require_float(
        scheduler.get("memory_reserve_fraction", 0.1),
        "memory_reserve_fraction",
    )
    if reserve_fraction >= 0.9:
        raise TrainingManifestError(
            "heterogeneous_manifest_memory_reserve_fraction_invalid"
        )
    result = {
        "device_policy": device_policy,
        "placement_policy": policy,
        "rebalance_policy": str(
            scheduler.get("rebalance_policy") or "failure-and-straggler"
        ),
        "max_stages_per_miner": _require_int(
            scheduler.get("max_stages_per_miner", 4),
            "max_stages_per_miner",
            minimum=1,
        ),
        "memory_reserve_fraction": reserve_fraction,
        "cuda_memory_reserve_bytes": _require_int(
            scheduler.get("cuda_memory_reserve_bytes", 512 * 1024 * 1024),
            "cuda_memory_reserve_bytes",
        ),
        "cpu_memory_reserve_bytes": _require_int(
            scheduler.get("cpu_memory_reserve_bytes", 1024 * 1024 * 1024),
            "cpu_memory_reserve_bytes",
        ),
        "straggler_ratio": _require_float(
            scheduler.get("straggler_ratio", 2.0),
            "straggler_ratio",
            minimum=1.0,
        ),
        "network_cost_weight": _require_float(
            scheduler.get("network_cost_weight", 1.0),
            "network_cost_weight",
        ),
        "load_cost_weight": _require_float(
            scheduler.get("load_cost_weight", 1.0),
            "load_cost_weight",
        ),
        "beam_width": _require_int(
            scheduler.get("beam_width", 256), "beam_width", minimum=1
        ),
        "required_device_types": sorted(
            {
                _require_string(item, "required_device_type").lower()
                for item in scheduler.get("required_device_types") or []
            }
        ),
    }
    if schema == TPU_MANIFEST_SCHEMA:
        result.update(
            {
                "tpu_memory_reserve_bytes": _require_int(
                    scheduler.get("tpu_memory_reserve_bytes", 8 * 1024**3),
                    "tpu_memory_reserve_bytes",
                ),
                "tpu_compile_cost_weight": _require_float(
                    scheduler.get("tpu_compile_cost_weight", 1.0),
                    "tpu_compile_cost_weight",
                ),
                "tpu_steady_state_cost_weight": _require_float(
                    scheduler.get("tpu_steady_state_cost_weight", 1.0),
                    "tpu_steady_state_cost_weight",
                ),
            }
        )
    return result


def _validate_stages(
    value: Any, *, model: dict[str, Any], schema: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TrainingManifestError("heterogeneous_manifest_stages_required")
    stages: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        stage = _require_dict(raw, f"stage_{index}")
        stage_id = _require_int(stage.get("stage_id"), "stage_id")
        layer_start = _require_int(stage.get("layer_start"), "layer_start")
        layer_end = _require_int(stage.get("layer_end"), "layer_end", minimum=1)
        if stage_id != index or layer_end <= layer_start:
            raise TrainingManifestError("heterogeneous_manifest_stage_order_invalid")
        allowed = sorted(
            {
                _require_string(item, "allowed_device_type").lower()
                for item in stage.get("allowed_device_types") or []
            }
        )
        supported = (
            TPU_SUPPORTED_DEVICE_TYPES
            if schema == TPU_MANIFEST_SCHEMA
            else SUPPORTED_DEVICE_TYPES
        )
        if not allowed or not set(allowed).issubset(supported):
            raise TrainingManifestError(
                "heterogeneous_manifest_stage_device_types_invalid"
            )
        preferred = str(stage.get("preferred_device_type") or allowed[0]).lower()
        if preferred not in allowed:
            raise TrainingManifestError(
                "heterogeneous_manifest_stage_preferred_device_invalid"
            )
        stages.append(
            {
                "stage_id": stage_id,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "layer_count": layer_end - layer_start,
                "owns_embedding": bool(stage.get("owns_embedding", False)),
                "owns_norm": bool(stage.get("owns_norm", False)),
                "owns_lm_head": bool(stage.get("owns_lm_head", False)),
                "allowed_device_types": allowed,
                "preferred_device_type": preferred,
                "estimated_parameter_count": _require_int(
                    stage.get("estimated_parameter_count"),
                    "estimated_parameter_count",
                    minimum=1,
                ),
                "estimated_weight_bytes": _require_int(
                    stage.get("estimated_weight_bytes"),
                    "estimated_weight_bytes",
                    minimum=1,
                ),
                "estimated_compute_units": _require_float(
                    stage.get(
                        "estimated_compute_units",
                        stage.get("estimated_parameter_count"),
                    ),
                    "estimated_compute_units",
                    minimum=1.0,
                ),
            }
        )
    if [item["layer_start"] for item in stages] != [
        0,
        *[item["layer_end"] for item in stages[:-1]],
    ]:
        raise TrainingManifestError("heterogeneous_manifest_stage_layers_not_contiguous")
    if stages[-1]["layer_end"] != int(model["num_hidden_layers"]):
        raise TrainingManifestError("heterogeneous_manifest_stage_layers_incomplete")
    if sum(bool(item["owns_embedding"]) for item in stages) != 1 or not stages[0][
        "owns_embedding"
    ]:
        raise TrainingManifestError("heterogeneous_manifest_embedding_owner_invalid")
    if sum(bool(item["owns_norm"]) for item in stages) != 1 or not stages[-1][
        "owns_norm"
    ]:
        raise TrainingManifestError("heterogeneous_manifest_norm_owner_invalid")
    if sum(bool(item["owns_lm_head"]) for item in stages) != 1 or not stages[-1][
        "owns_lm_head"
    ]:
        raise TrainingManifestError("heterogeneous_manifest_lm_head_owner_invalid")
    return stages


def validate_training_manifest(value: Any) -> dict[str, Any]:
    """Validate and return the canonical public manifest representation."""

    source = _require_dict(value, "root")
    schema = str(source.get("schema") or "")
    if schema not in {MANIFEST_SCHEMA, TPU_MANIFEST_SCHEMA}:
        raise TrainingManifestError("heterogeneous_manifest_schema_invalid")
    model = _validate_model(source.get("model"))
    manifest = {
        "schema": schema,
        "model": model,
        "lora": _validate_lora(source.get("lora")),
        "dataset": _validate_dataset(source.get("dataset")),
        "precision": _validate_precision(source.get("precision"), schema=schema),
        "training": _validate_training(source.get("training")),
        "checkpoint": _validate_checkpoint(source.get("checkpoint")),
        "scheduler": _validate_scheduler(source.get("scheduler"), schema=schema),
        "stages": _validate_stages(source.get("stages"), model=model, schema=schema),
        "raw_training_text_public": False,
        "token_ids_public": False,
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    required_devices = set(manifest["scheduler"]["required_device_types"])
    supported = (
        TPU_SUPPORTED_DEVICE_TYPES
        if schema == TPU_MANIFEST_SCHEMA
        else SUPPORTED_DEVICE_TYPES
    )
    if not required_devices.issubset(supported):
        raise TrainingManifestError(
            "heterogeneous_manifest_required_device_types_invalid"
        )
    stage_devices = {
        device
        for stage in manifest["stages"]
        for device in stage["allowed_device_types"]
    }
    if not required_devices.issubset(stage_devices):
        raise TrainingManifestError(
            "heterogeneous_manifest_required_device_types_unreachable"
        )
    supplied_hash = str(source.get("content_hash") or "")
    manifest["content_hash"] = stable_hash(manifest)
    if supplied_hash and supplied_hash != manifest["content_hash"]:
        raise TrainingManifestError("heterogeneous_manifest_content_hash_mismatch")
    return manifest


def load_training_manifest(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingManifestError("heterogeneous_manifest_file_invalid") from exc
    return validate_training_manifest(value)


def write_training_manifest(path: str | Path, value: Any) -> Path:
    manifest = validate_training_manifest(value)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def _qwen_layer_parameter_count(hidden: int, intermediate: int, kv_heads: int, heads: int) -> int:
    head_dim = hidden // heads
    kv_width = kv_heads * head_dim
    attention = hidden * hidden * 2 + hidden * kv_width * 2
    mlp = hidden * intermediate * 3
    norms = hidden * 2
    return attention + mlp + norms


def qwen25_7b_lora_manifest(*, target_steps: int = 6) -> dict[str, Any]:
    """Return the pinned five-stage Qwen2.5-7B heterogeneous LoRA gate."""

    hidden = 3584
    intermediate = 18944
    vocab = 152064
    layer_parameters = _qwen_layer_parameter_count(hidden, intermediate, 4, 28)
    boundaries = (0, 7, 14, 20, 26, 28)
    stages = []
    for stage_id, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        parameter_count = (end - start) * layer_parameters
        if stage_id == 0:
            parameter_count += vocab * hidden
        if stage_id == len(boundaries) - 2:
            parameter_count += hidden + vocab * hidden
        weight_bytes = parameter_count * 2
        stages.append(
            {
                "stage_id": stage_id,
                "layer_start": start,
                "layer_end": end,
                "owns_embedding": stage_id == 0,
                "owns_norm": stage_id == len(boundaries) - 2,
                "owns_lm_head": stage_id == len(boundaries) - 2,
                "allowed_device_types": (
                    ["cpu"] if stage_id == len(boundaries) - 2 else ["cpu", "cuda"]
                ),
                "preferred_device_type": (
                    "cpu" if stage_id == len(boundaries) - 2 else "cuda"
                ),
                "estimated_parameter_count": parameter_count,
                "estimated_weight_bytes": weight_bytes,
                "estimated_compute_units": float(parameter_count),
            }
        )
    # Keep estimates tied to the exact public model size even though the formula
    # above omits a few small non-layer buffers.
    difference = QWEN25_7B_WEIGHT_BYTES - sum(
        int(stage["estimated_weight_bytes"]) for stage in stages
    )
    stages[-1]["estimated_weight_bytes"] += difference
    stages[-1]["estimated_parameter_count"] += difference // 2
    value = {
        "schema": MANIFEST_SCHEMA,
        "model": {
            "model_id": QWEN25_7B_MODEL_ID,
            "model_revision": QWEN25_7B_MODEL_REVISION,
            "architecture": "Qwen2ForCausalLM",
            "model_type": "qwen2",
            "parameter_count": QWEN25_7B_PARAMETER_COUNT,
            "weight_bytes": QWEN25_7B_WEIGHT_BYTES,
            "num_hidden_layers": 28,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "vocab_size": vocab,
            "source_dtype": "bfloat16",
            "trust_remote_code": False,
        },
        "lora": {
            "rank": 4,
            "alpha": 8,
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
            "learning_rate": 0.0005,
            "gradient_clip_norm": 1.0,
        },
        "dataset": {
            "dataset_id": "Salesforce/wikitext",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "dataset_config": "wikitext-2-raw-v1",
            "train_split": "train",
            "validation_split": "validation",
            "data_seed": 20260713,
        },
        "precision": {
            "cuda_compute_dtype": "float32",
            "cpu_compute_dtype": "float32",
            "boundary_dtype": "float16",
            "optimizer_dtype": "float32",
        },
        "training": {
            "target_steps": int(target_steps),
            "microbatches_per_step": 1,
            "microbatch_size": 1,
            "sequence_length": 8,
            "gradient_accumulation_steps": 1,
            "seed": 20260713,
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
            "device_policy": "mixed",
            "placement_policy": "memory-performance",
            "rebalance_policy": "failure-and-straggler",
            "max_stages_per_miner": 2,
            "memory_reserve_fraction": 0.1,
            "cuda_memory_reserve_bytes": 512 * 1024 * 1024,
            "cpu_memory_reserve_bytes": 1024 * 1024 * 1024,
            "straggler_ratio": 2.0,
            "network_cost_weight": 1.0,
            "load_cost_weight": 1.0,
            "beam_width": 256,
            "required_device_types": ["cpu", "cuda"],
        },
        "stages": stages,
    }
    return validate_training_manifest(deepcopy(value))


def qwen25_7b_lora_tpu_manifest(*, target_steps: int = 6) -> dict[str, Any]:
    """Return the pinned CPU/CUDA/JAX-TPU five-stage Qwen2.5-7B gate."""

    value = deepcopy(qwen25_7b_lora_manifest(target_steps=target_steps))
    value.pop("content_hash", None)
    value["schema"] = TPU_MANIFEST_SCHEMA
    value["precision"]["jax_tpu_compute_dtype"] = "bfloat16"
    value["scheduler"].update(
        {
            "tpu_memory_reserve_bytes": 8 * 1024**3,
            "tpu_compile_cost_weight": 1.0,
            "tpu_steady_state_cost_weight": 1.0,
            "required_device_types": ["cpu", "cuda", "jax_tpu"],
        }
    )
    # Keep the achieved topology fixed while forcing the middle six-layer stage
    # onto one JAX TPU resource group. The remaining stages preserve the
    # three-CUDA plus pure-CPU placement used by the live gate.
    value["stages"][2]["allowed_device_types"] = ["jax_tpu"]
    value["stages"][2]["preferred_device_type"] = "jax_tpu"
    return validate_training_manifest(value)
