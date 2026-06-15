"""Optional Hugging Face tiny GPT sharded inference workload.

This module is deliberately outside the default dependency path. It provides a
CPU-first, read-only two-stage proof for a real small LLM runtime when
``transformers`` and ``torch`` are installed via the optional ``hf`` extra.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any


REAL_LLM_ARTIFACT_SCHEMA_VERSION = "real_llm_artifact_v1"
REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION = "real_llm_sharded_infer_v1"
REAL_LLM_ACTIVATION_SCHEMA_VERSION = "real_llm_activation_v1"
REAL_LLM_PARTIAL_WEIGHT_PLAN_SCHEMA_VERSION = "real_llm_partial_weight_plan_v1"
REAL_LLM_STAGE_SELECTIVE_WEIGHT_LOAD_SCHEMA_VERSION = "real_llm_stage_selective_weight_load_v1"
REAL_LLM_STAGE_SELECTIVE_WEIGHT_APPLY_SCHEMA_VERSION = "real_llm_stage_selective_weight_apply_v1"
REAL_LLM_STAGE_SELECTIVE_RUNTIME_SCHEMA_VERSION = "real_llm_stage_selective_runtime_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
BACKEND_CPU = "hf_transformers_cpu"
BACKEND_CUDA = "hf_transformers_cuda"
BACKEND_AUTO = "auto"
SUPPORTED_BACKENDS = {BACKEND_CPU, BACKEND_CUDA, BACKEND_AUTO}
PARTITION_MODE_FULL = "full"
PARTITION_MODE_STAGE_LOCAL = "stage_local"
PARTITION_MODE_STAGE_LOCAL_ALIAS = "stage-local"
SUPPORTED_PARTITION_MODES = {
    PARTITION_MODE_FULL,
    PARTITION_MODE_STAGE_LOCAL,
    PARTITION_MODE_STAGE_LOCAL_ALIAS,
}
DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"
DEFAULT_MODEL_MANIFEST = {
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "tokenizer_class": "GPT2TokenizerFast",
    "num_hidden_layers": 2,
    "hidden_size": 2,
    "vocab_size": 50257,
}
DEFAULT_PROMPTS = [
    "CrowdTensor routes home CPU",
    "A miner returns one token",
]
EXECUTION_FAMILY_GPT2 = "gpt2"
EXECUTION_FAMILY_LLAMA_LIKE = "llama_like"
EXECUTION_FAMILY_UNSUPPORTED_HF_CAUSAL_LM = "unsupported_hf_causal_lm"
EXECUTION_FAMILY_UNKNOWN = "unknown"
SUPPORTED_EXECUTION_FAMILIES = {EXECUTION_FAMILY_GPT2, EXECUTION_FAMILY_LLAMA_LIKE}
PARTIAL_WEIGHT_PLAN_FAMILIES = {EXECUTION_FAMILY_GPT2, EXECUTION_FAMILY_LLAMA_LIKE}
LLAMA_LIKE_MODEL_TYPES = {
    "gemma",
    "gemma2",
    "llama",
    "mistral",
    "mixtral",
    "phi",
    "phi3",
    "qwen2",
    "qwen3",
}
GPT2_PARAMETER_ESTIMATE_BY_MODEL_ID = {
    "distilgpt2": 82_000_000,
    "gpt2": 124_000_000,
    "openai-community/gpt2": 124_000_000,
    "gpt2-medium": 355_000_000,
    "openai-community/gpt2-medium": 355_000_000,
    "gpt2-large": 774_000_000,
    "openai-community/gpt2-large": 774_000_000,
    "gpt2-xl": 1_558_000_000,
    "openai-community/gpt2-xl": 1_558_000_000,
}
LLAMA_LIKE_PARAMETER_ESTIMATE_BY_MODEL_ID = {
    "qwen/qwen2.5-7b-instruct": 7_615_000_000,
    "qwen/qwen2.5-7b": 7_615_000_000,
    "qwen2.5-7b-instruct": 7_615_000_000,
    "qwen2.5-7b": 7_615_000_000,
    "meta-llama/llama-2-7b-hf": 6_738_000_000,
    "meta-llama/meta-llama-3-8b": 8_030_000_000,
    "meta-llama/meta-llama-3-8b-instruct": 8_030_000_000,
    "meta-llama/llama-3.1-8b": 8_030_000_000,
    "meta-llama/llama-3.1-8b-instruct": 8_030_000_000,
    "mistralai/mistral-7b-v0.1": 7_240_000_000,
    "mistralai/mistral-7b-instruct-v0.2": 7_240_000_000,
}
SMALL_TIER_MIN_PARAMETERS = 1_000_000_000
SMALL_TIER_MAX_PARAMETERS = 3_000_000_000
LARGE_TIER_MIN_PARAMETERS = 6_000_000_000
FP32_BYTES_PER_PARAMETER = 4
MAX_REQUESTS = 4
MAX_PROMPT_CHARS = 256
MAX_NEW_TOKENS = 32
ROUND_DIGITS = 8
_MODEL_CACHE: dict[tuple[str, str, str, bool], tuple[Any, Any, Any]] = {}
_STAGE0_KV_CACHE: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
_STAGE1_KV_CACHE: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}


def missing_hf_dependencies() -> list[str]:
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        missing.append("transformers")
    try:
        import safetensors  # noqa: F401
    except ModuleNotFoundError:
        missing.append("safetensors")
    return missing


def hf_available() -> bool:
    return not missing_hf_dependencies()


def require_hf_dependencies() -> None:
    missing = missing_hf_dependencies()
    if missing:
        raise RuntimeError(
            "real_llm_sharded_infer requires optional Hugging Face dependencies: "
            + ", ".join(missing)
            + ". Install with: python -m pip install -e .[hf]"
        )


def normalize_backend(backend: str | None = None) -> str:
    normalized = str(backend or BACKEND_CPU).strip().lower()
    if normalized in {"", "cpu"}:
        return BACKEND_CPU
    if normalized in {"cuda", "gpu"}:
        return BACKEND_CUDA
    if normalized in SUPPORTED_BACKENDS:
        return normalized
    raise ValueError(f"unsupported real_llm_sharded_infer backend: {backend}")


def torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def cuda_runtime_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "backend": BACKEND_CUDA,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "vram_total_mb": [],
        "torch_cuda_version": "",
        "diagnosis_codes": ["cuda_runtime_unavailable"],
    }
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        summary["missing_dependencies"] = ["torch"]
        return summary
    summary["torch_cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
    available = bool(torch.cuda.is_available())
    summary["cuda_available"] = available
    if not available:
        return summary
    try:
        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    summary["gpu_count"] = count
    names: list[str] = []
    vram: list[int] = []
    for index in range(count):
        try:
            names.append(str(torch.cuda.get_device_name(index)))
        except Exception:
            names.append("unknown")
        try:
            props = torch.cuda.get_device_properties(index)
            vram.append(int(getattr(props, "total_memory", 0) // (1024 * 1024)))
        except Exception:
            vram.append(0)
    summary["gpu_names"] = names
    summary["vram_total_mb"] = vram
    summary["diagnosis_codes"] = ["cuda_runtime_available", "gpu_runtime_ready"]
    return summary


def resolve_backend(backend: str | None = None) -> str:
    normalized = normalize_backend(backend)
    if normalized == BACKEND_AUTO:
        return BACKEND_CUDA if torch_cuda_available() else BACKEND_CPU
    if normalized == BACKEND_CUDA and not torch_cuda_available():
        raise RuntimeError("hf_transformers_cuda requires torch CUDA runtime, but torch.cuda.is_available() is false")
    return normalized


def normalize_partition_mode(mode: str | None = None) -> str:
    normalized = str(mode or PARTITION_MODE_FULL).strip().lower().replace("_", "-")
    if normalized in {"", "full", "full-model"}:
        return PARTITION_MODE_FULL
    if normalized in {"stage-local", "stage", "partitioned"}:
        return PARTITION_MODE_STAGE_LOCAL
    raise ValueError(f"unsupported real_llm_sharded_infer partition_mode: {mode}")


def execution_family_from_metadata(
    metadata: dict[str, Any] | None = None,
    *,
    model_id: str = "",
    model_type: str = "",
    architectures: list[Any] | tuple[Any, ...] | None = None,
) -> str:
    """Return the execution family supported by the current stage splitter."""

    source = dict(metadata or {})
    raw_model_id = str(model_id or source.get("model_id") or "").strip().lower()
    raw_model_type = str(model_type or source.get("model_type") or "").strip().lower()
    raw_architectures = architectures if architectures is not None else source.get("architectures")
    arch_text = " ".join(str(item).strip().lower() for item in list(raw_architectures or []))
    combined = " ".join(item for item in [raw_model_id, raw_model_type, arch_text] if item)
    if raw_model_type == "gpt2" or "gpt2" in arch_text or "gpt2" in raw_model_id:
        return EXECUTION_FAMILY_GPT2
    if raw_model_type in LLAMA_LIKE_MODEL_TYPES or any(
        marker in combined
        for marker in (
            "gemma",
            "llama",
            "mistral",
            "mixtral",
            "phi",
            "qwen",
        )
    ):
        return EXECUTION_FAMILY_LLAMA_LIKE
    if raw_model_type or arch_text:
        return EXECUTION_FAMILY_UNSUPPORTED_HF_CAUSAL_LM
    return EXECUTION_FAMILY_UNKNOWN


def _large_model_candidate_from_metadata(metadata: dict[str, Any], *, family: str) -> bool:
    model_id = str(metadata.get("model_id") or "").lower()
    if any(marker in model_id for marker in ("7b", "8b", "13b", "14b", "70b")):
        return True
    estimated_parameters = estimate_parameter_count_from_metadata(metadata, family=family)
    if estimated_parameters >= LARGE_TIER_MIN_PARAMETERS:
        return True
    try:
        layer_count = int(metadata.get("num_hidden_layers") or 0)
    except (TypeError, ValueError):
        layer_count = 0
    try:
        hidden_size = int(metadata.get("hidden_size") or 0)
    except (TypeError, ValueError):
        hidden_size = 0
    return bool(family == EXECUTION_FAMILY_LLAMA_LIKE or (layer_count >= 16 and hidden_size >= 2048))


def _first_positive_int(metadata: dict[str, Any], *names: str) -> int:
    for name in names:
        try:
            value = int(metadata.get(name) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def estimate_parameter_count_from_metadata(metadata: dict[str, Any], *, family: str | None = None) -> int:
    source = dict(metadata or {})
    resolved_family = family or execution_family_from_metadata(source)
    model_id = str(source.get("model_id") or "").strip().lower()
    if model_id in GPT2_PARAMETER_ESTIMATE_BY_MODEL_ID:
        return int(GPT2_PARAMETER_ESTIMATE_BY_MODEL_ID[model_id])
    if model_id in LLAMA_LIKE_PARAMETER_ESTIMATE_BY_MODEL_ID:
        return int(LLAMA_LIKE_PARAMETER_ESTIMATE_BY_MODEL_ID[model_id])
    if resolved_family != EXECUTION_FAMILY_GPT2:
        return 0
    layer_count = _first_positive_int(source, "num_hidden_layers", "n_layer", "n_layers")
    hidden_size = _first_positive_int(source, "hidden_size", "n_embd")
    vocab_size = _first_positive_int(source, "vocab_size")
    position_count = _first_positive_int(source, "n_positions", "max_position_embeddings")
    if layer_count <= 0 or hidden_size <= 0 or vocab_size <= 0:
        return 0
    embedding_parameters = vocab_size * hidden_size
    position_parameters = position_count * hidden_size if position_count > 0 else 0
    block_parameters = layer_count * (12 * hidden_size * hidden_size + 13 * hidden_size)
    final_norm_parameters = 2 * hidden_size
    return int(embedding_parameters + position_parameters + block_parameters + final_norm_parameters)


def _weight_map_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    raw_map = (
        metadata.get("weight_map")
        or metadata.get("safetensors_weight_map")
        or metadata.get("hf_weight_map")
        or {}
    )
    if not isinstance(raw_map, dict):
        return {}
    weight_map: dict[str, str] = {}
    for key, value in raw_map.items():
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()
        if key_text and value_text:
            weight_map[key_text] = Path(value_text).name
    return weight_map


def _layer_count_from_metadata(metadata: dict[str, Any]) -> int:
    return _first_positive_int(metadata, "num_hidden_layers", "n_layer", "n_layers")


def _split_index_for_layer_count(metadata: dict[str, Any], layer_count: int) -> int:
    if layer_count < 2:
        return 0
    try:
        raw_split = int(metadata.get("split_index") or 0)
    except (TypeError, ValueError):
        raw_split = 0
    split = raw_split if raw_split > 0 else max(1, layer_count // 2)
    return max(1, min(split, layer_count - 1))


def _stage_weight_prefixes(family: str, *, stage_id: int, split_index: int, layer_count: int) -> tuple[list[str], tuple[int, int], list[str]]:
    stage = int(stage_id)
    split = max(1, min(int(split_index), int(layer_count) - 1)) if int(layer_count) >= 2 else 0
    if family == EXECUTION_FAMILY_GPT2:
        if stage == 0:
            return (
                ["transformer.wte.", "transformer.wpe.", *[f"transformer.h.{index}." for index in range(split)]],
                (0, split),
                ["token_embedding", "position_embedding", "transformer_blocks_prefix"],
            )
        return (
            [*[f"transformer.h.{index}." for index in range(split, layer_count)], "transformer.ln_f.", "lm_head."],
            (split, layer_count),
            ["transformer_blocks_suffix", "final_norm", "lm_head"],
        )
    if family == EXECUTION_FAMILY_LLAMA_LIKE:
        if stage == 0:
            return (
                ["model.embed_tokens.", *[f"model.layers.{index}." for index in range(split)]],
                (0, split),
                ["token_embedding", "decoder_layers_prefix"],
            )
        return (
            [*[f"model.layers.{index}." for index in range(split, layer_count)], "model.norm.", "lm_head."],
            (split, layer_count),
            ["decoder_layers_suffix", "final_norm", "lm_head"],
        )
    return ([], (0, 0), [])


def real_llm_partial_weight_loading_plan(
    metadata: dict[str, Any] | None = None,
    *,
    partition_mode: str | None = None,
) -> dict[str, Any]:
    """Return a public-safe plan for loading only stage-owned HF weights.

    This is intentionally a planning contract, not runtime proof. It maps a
    Hugging Face safetensors-style weight index to the exact key prefixes and
    shard filenames each two-stage miner would need. Runtime readiness remains
    false until the stage runner actually executes from this plan.
    """

    source = dict(metadata or {})
    mode = normalize_partition_mode(partition_mode or source.get("partition_mode") or PARTITION_MODE_FULL)
    family = execution_family_from_metadata(source)
    layer_count = _layer_count_from_metadata(source)
    split = _split_index_for_layer_count(source, layer_count)
    weight_map = _weight_map_from_metadata(source)
    parameter_count_estimate = estimate_parameter_count_from_metadata(source, family=family)
    estimated_weight_bytes = parameter_count_estimate * FP32_BYTES_PER_PARAMETER
    stage_plans: list[dict[str, Any]] = []
    all_keys = set(weight_map)
    all_files = set(weight_map.values())
    assigned_all_keys: set[str] = set()
    plan_family_supported = family in PARTIAL_WEIGHT_PLAN_FAMILIES
    for stage_id in (0, 1):
        prefixes, layer_range, module_kinds = _stage_weight_prefixes(
            family,
            stage_id=stage_id,
            split_index=split,
            layer_count=layer_count,
        )
        assigned = sorted(
            key for key in weight_map if any(key.startswith(prefix) for prefix in prefixes)
        )
        assigned_files = sorted({weight_map[key] for key in assigned if weight_map.get(key)})
        assigned_all_keys.update(assigned)
        missing_prefixes = sorted(
            prefix for prefix in prefixes if not any(key.startswith(prefix) for key in assigned)
        )
        expected_fraction = 0.0
        if layer_count > 0:
            expected_fraction = round((layer_range[1] - layer_range[0]) / float(layer_count), 8)
        stage_plans.append({
            "stage_id": stage_id,
            "stage_layer_range": [int(layer_range[0]), int(layer_range[1])],
            "stage_layer_range_format": "start_inclusive_end_exclusive",
            "stage_module_kinds": module_kinds,
            "expected_key_prefixes": prefixes,
            "assigned_weight_key_count": len(assigned),
            "assigned_weight_file_count": len(assigned_files),
            "assigned_weight_files": assigned_files,
            "missing_required_prefixes": missing_prefixes,
            "loads_only_stage_weight_keys": bool(assigned and not missing_prefixes),
            "expected_decoder_layer_fraction": expected_fraction,
            "estimated_stage_weight_bytes_fp32": int(round(estimated_weight_bytes * max(expected_fraction, 0.5 if layer_count <= 0 else 0))),
        })
    unassigned = sorted(all_keys - assigned_all_keys)
    plan_ready = bool(
        mode == PARTITION_MODE_STAGE_LOCAL
        and plan_family_supported
        and layer_count >= 2
        and weight_map
        and all(not stage["missing_required_prefixes"] for stage in stage_plans)
    )
    diagnosis_codes: list[str] = []
    blockers: list[str] = []
    if plan_ready:
        diagnosis_codes.append("real_llm_partial_weight_plan_ready")
        if family == EXECUTION_FAMILY_LLAMA_LIKE:
            diagnosis_codes.append("real_llm_llama_like_partial_weight_plan_ready")
    else:
        diagnosis_codes.append("real_llm_partial_weight_plan_not_ready")
        if mode != PARTITION_MODE_STAGE_LOCAL:
            blockers.append("real_llm_partial_weight_plan_requires_stage_local")
        if not plan_family_supported:
            blockers.append("real_llm_partial_weight_plan_family_unsupported")
        if layer_count < 2:
            blockers.append("real_llm_partial_weight_plan_layer_metadata_missing")
        if not weight_map:
            blockers.append("real_llm_partial_weight_plan_weight_map_missing")
        if any(stage["missing_required_prefixes"] for stage in stage_plans):
            blockers.append("real_llm_partial_weight_plan_required_keys_missing")
    return {
        "schema": REAL_LLM_PARTIAL_WEIGHT_PLAN_SCHEMA_VERSION,
        "ready": plan_ready,
        "runtime_execution_ready": False,
        "model_id": str(source.get("model_id") or ""),
        "execution_family": family,
        "partition_mode": mode,
        "stage_count": 2,
        "num_hidden_layers": layer_count,
        "split_index": split,
        "parameter_count_estimate": parameter_count_estimate,
        "estimated_weight_bytes_fp32": estimated_weight_bytes,
        "weight_index_available": bool(weight_map),
        "weight_key_count": len(weight_map),
        "weight_file_count": len(all_files),
        "unassigned_weight_key_count": len(unassigned),
        "unassigned_weight_key_samples": unassigned[:8],
        "stage_plans": stage_plans,
        "diagnosis_codes": sorted(set(diagnosis_codes)),
        "blockers": sorted(set(blockers)),
        "public_safe": True,
    }


def _normalize_stage_id(stage_id: int) -> int:
    try:
        stage = int(stage_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("real LLM stage_id must be 0 or 1") from exc
    if stage not in {0, 1}:
        raise ValueError("real LLM stage_id must be 0 or 1")
    return stage


def _stage_weight_selection(
    metadata: dict[str, Any],
    *,
    stage_id: int,
    partition_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[str]]:
    stage = _normalize_stage_id(stage_id)
    plan = real_llm_partial_weight_loading_plan(
        metadata,
        partition_mode=partition_mode or metadata.get("partition_mode") or PARTITION_MODE_STAGE_LOCAL,
    )
    stage_plan = next(
        (
            row
            for row in list(plan.get("stage_plans") or [])
            if isinstance(row, dict) and int(row.get("stage_id", -1)) == stage
        ),
        {},
    )
    weight_map = _weight_map_from_metadata(metadata)
    prefixes = [str(prefix) for prefix in list(stage_plan.get("expected_key_prefixes") or [])]
    assigned = sorted(
        key
        for key in weight_map
        if any(str(key).startswith(prefix) for prefix in prefixes)
    )
    return plan, stage_plan, weight_map, assigned


def _tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _load_stage_selective_safetensors(
    metadata: dict[str, Any] | None = None,
    *,
    stage_id: int,
    weight_root: str | Path,
    partition_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only stage-owned safetensors keys into CPU tensors.

    The returned tensor dictionary is intentionally kept private to callers.
    The summary is public-safe: it contains counts, basenames, byte totals, and
    hashes of loaded key names, but no tensor values or local cache paths.
    """

    source = dict(metadata or {})
    stage = _normalize_stage_id(stage_id)
    plan, stage_plan, weight_map, assigned_keys = _stage_weight_selection(
        source,
        stage_id=stage,
        partition_mode=partition_mode,
    )
    assigned_key_set = set(assigned_keys)
    assigned_files = sorted({weight_map[key] for key in assigned_keys if weight_map.get(key)})
    summary: dict[str, Any] = {
        "schema": REAL_LLM_STAGE_SELECTIVE_WEIGHT_LOAD_SCHEMA_VERSION,
        "ready": False,
        "stage_selective_tensor_materialization_ready": False,
        "runtime_execution_ready": False,
        "model_id": str(source.get("model_id") or ""),
        "execution_family": plan.get("execution_family") or execution_family_from_metadata(source),
        "partition_mode": plan.get("partition_mode") or normalize_partition_mode(partition_mode or source.get("partition_mode") or PARTITION_MODE_STAGE_LOCAL),
        "stage_id": stage,
        "stage_count": 2,
        "stage_layer_range": list(stage_plan.get("stage_layer_range") or []),
        "stage_layer_range_format": "start_inclusive_end_exclusive",
        "stage_module_kinds": list(stage_plan.get("stage_module_kinds") or []),
        "assigned_weight_key_count": len(assigned_keys),
        "assigned_weight_file_count": len(assigned_files),
        "assigned_weight_files": assigned_files,
        "opened_weight_file_count": 0,
        "loaded_weight_key_count": 0,
        "loaded_weight_file_count": 0,
        "loaded_tensor_bytes": 0,
        "candidate_file_key_count": 0,
        "skipped_non_stage_weight_key_count": 0,
        "missing_weight_file_count": 0,
        "missing_weight_key_count": 0,
        "missing_weight_files": [],
        "missing_weight_key_count_by_file": {},
        "loaded_weight_key_digest": _hash_payload([]),
        "loaded_weight_file_digest": _hash_payload([]),
        "loads_only_stage_weight_keys": False,
        "cross_stage_weight_keys_loaded": False,
        "public_safe": True,
        "diagnosis_codes": [],
        "blockers": [],
    }
    blockers = set(str(item) for item in list(plan.get("blockers") or []))
    diagnosis_codes = set(str(item) for item in list(plan.get("diagnosis_codes") or []))
    if not plan.get("ready"):
        blockers.add("real_llm_stage_selective_weight_plan_not_ready")
        diagnosis_codes.add("real_llm_stage_selective_weight_load_not_ready")
        summary["blockers"] = sorted(blockers)
        summary["diagnosis_codes"] = sorted(diagnosis_codes)
        return {}, summary
    try:
        from safetensors.torch import safe_open  # type: ignore
    except ModuleNotFoundError:
        blockers.add("safetensors_dependency_missing")
        diagnosis_codes.add("real_llm_stage_selective_weight_load_not_ready")
        summary["blockers"] = sorted(blockers)
        summary["diagnosis_codes"] = sorted(diagnosis_codes)
        return {}, summary

    root = Path(weight_root)
    loaded: dict[str, Any] = {}
    opened_files: set[str] = set()
    loaded_files: set[str] = set()
    missing_files: list[str] = []
    missing_key_count_by_file: dict[str, int] = {}
    candidate_file_key_count = 0
    skipped_non_stage_key_count = 0
    for filename in assigned_files:
        safe_filename = Path(str(filename)).name
        path = root / safe_filename
        if not path.is_file():
            missing_files.append(safe_filename)
            missing_key_count_by_file[safe_filename] = len(
                [key for key in assigned_keys if weight_map.get(key) == safe_filename]
            )
            continue
        opened_files.add(safe_filename)
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                available_keys = set(str(key) for key in handle.keys())
                candidate_file_key_count += len(available_keys)
                skipped_non_stage_key_count += len(available_keys - assigned_key_set)
                expected_in_file = [
                    key for key in assigned_keys if weight_map.get(key) == safe_filename
                ]
                missing_in_file = [key for key in expected_in_file if key not in available_keys]
                if missing_in_file:
                    missing_key_count_by_file[safe_filename] = len(missing_in_file)
                for key in expected_in_file:
                    if key not in available_keys:
                        continue
                    loaded[key] = handle.get_tensor(key)
                    loaded_files.add(safe_filename)
        except Exception:
            blockers.add("real_llm_stage_selective_weight_file_load_failed")
            missing_files.append(safe_filename)
            missing_key_count_by_file[safe_filename] = len(
                [key for key in assigned_keys if weight_map.get(key) == safe_filename]
            )
    loaded_keys = sorted(loaded)
    loaded_key_set = set(loaded_keys)
    unexpected_loaded = sorted(loaded_key_set - assigned_key_set)
    missing_key_count = sum(int(value) for value in missing_key_count_by_file.values())
    tensor_bytes = sum(_tensor_nbytes(tensor) for tensor in loaded.values())
    ready = bool(
        loaded_keys
        and not missing_files
        and missing_key_count == 0
        and not unexpected_loaded
        and loaded_key_set.issubset(assigned_key_set)
    )
    if ready:
        diagnosis_codes.update({
            "real_llm_stage_selective_weight_materialization_ready",
            "real_llm_stage_selective_weight_load_ready",
        })
    else:
        diagnosis_codes.add("real_llm_stage_selective_weight_load_not_ready")
        if not loaded_keys:
            blockers.add("real_llm_stage_selective_weight_keys_not_loaded")
        if missing_files:
            blockers.add("real_llm_stage_selective_weight_files_missing")
        if missing_key_count:
            blockers.add("real_llm_stage_selective_weight_keys_missing")
        if unexpected_loaded:
            blockers.add("real_llm_stage_selective_cross_stage_weight_loaded")
    summary.update({
        "ready": ready,
        "stage_selective_tensor_materialization_ready": ready,
        "opened_weight_file_count": len(opened_files),
        "loaded_weight_key_count": len(loaded_keys),
        "loaded_weight_file_count": len(loaded_files),
        "loaded_tensor_bytes": int(tensor_bytes),
        "candidate_file_key_count": int(candidate_file_key_count),
        "skipped_non_stage_weight_key_count": int(skipped_non_stage_key_count),
        "missing_weight_file_count": len(missing_files),
        "missing_weight_key_count": int(missing_key_count),
        "missing_weight_files": sorted(set(missing_files))[:8],
        "missing_weight_key_count_by_file": {
            key: int(value) for key, value in sorted(missing_key_count_by_file.items())
        },
        "loaded_weight_key_digest": _hash_payload(loaded_keys),
        "loaded_weight_file_digest": _hash_payload(sorted(loaded_files)),
        "loads_only_stage_weight_keys": bool(ready and not unexpected_loaded),
        "cross_stage_weight_keys_loaded": bool(unexpected_loaded),
        "diagnosis_codes": sorted(diagnosis_codes),
        "blockers": sorted(blockers),
    })
    return loaded, summary


def real_llm_stage_selective_weight_load_summary(
    metadata: dict[str, Any] | None = None,
    *,
    stage_id: int,
    weight_root: str | Path,
    partition_mode: str | None = None,
) -> dict[str, Any]:
    _, summary = _load_stage_selective_safetensors(
        metadata,
        stage_id=stage_id,
        weight_root=weight_root,
        partition_mode=partition_mode,
    )
    return summary


def _apply_stage_selective_tensors_to_model(
    model: Any,
    tensors: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    stage_id: int,
    partition_mode: str | None = None,
) -> dict[str, Any]:
    """Apply already-loaded stage tensors to matching model state entries.

    This is the bridge between selective safetensors materialization and a
    runtime that can instantiate only stage-owned modules. The summary is
    public-safe and intentionally omits tensor values.
    """

    source = dict(metadata or {})
    stage = _normalize_stage_id(stage_id)
    plan, stage_plan, _weight_map, assigned_keys = _stage_weight_selection(
        source,
        stage_id=stage,
        partition_mode=partition_mode,
    )
    assigned_key_set = set(assigned_keys)
    provided_keys = sorted(str(key) for key in dict(tensors or {}))
    model_state = model.state_dict() if hasattr(model, "state_dict") else {}
    model_keys = set(str(key) for key in model_state)
    summary: dict[str, Any] = {
        "schema": REAL_LLM_STAGE_SELECTIVE_WEIGHT_APPLY_SCHEMA_VERSION,
        "ready": False,
        "stage_selective_tensor_application_ready": False,
        "runtime_execution_ready": False,
        "model_id": str(source.get("model_id") or ""),
        "execution_family": plan.get("execution_family") or execution_family_from_metadata(source),
        "partition_mode": plan.get("partition_mode") or normalize_partition_mode(partition_mode or source.get("partition_mode") or PARTITION_MODE_STAGE_LOCAL),
        "stage_id": stage,
        "stage_count": 2,
        "stage_layer_range": list(stage_plan.get("stage_layer_range") or []),
        "stage_layer_range_format": "start_inclusive_end_exclusive",
        "stage_module_kinds": list(stage_plan.get("stage_module_kinds") or []),
        "assigned_weight_key_count": len(assigned_keys),
        "provided_weight_key_count": len(provided_keys),
        "applied_weight_key_count": 0,
        "missing_assigned_weight_key_count": 0,
        "unknown_model_key_count": 0,
        "shape_mismatch_count": 0,
        "dtype_conversion_count": 0,
        "cross_stage_weight_keys_loaded": False,
        "applied_weight_key_digest": _hash_payload([]),
        "applied_tensor_bytes": 0,
        "loads_only_stage_weight_keys": False,
        "public_safe": True,
        "diagnosis_codes": [],
        "blockers": [],
    }
    blockers = set(str(item) for item in list(plan.get("blockers") or []))
    diagnosis_codes = set(str(item) for item in list(plan.get("diagnosis_codes") or []))
    if not plan.get("ready"):
        blockers.add("real_llm_stage_selective_weight_plan_not_ready")
        diagnosis_codes.add("real_llm_stage_selective_weight_application_not_ready")
        summary["blockers"] = sorted(blockers)
        summary["diagnosis_codes"] = sorted(diagnosis_codes)
        return summary
    unexpected_keys = sorted(set(provided_keys) - assigned_key_set)
    missing_assigned = sorted(assigned_key_set - set(provided_keys))
    unknown_model_keys: list[str] = []
    shape_mismatches: list[str] = []
    dtype_conversions = 0
    applied: list[str] = []
    applied_bytes = 0
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        torch = None  # type: ignore
    for key in provided_keys:
        tensor = tensors[key]
        if key not in assigned_key_set:
            continue
        target = model_state.get(key)
        if target is None:
            unknown_model_keys.append(key)
            continue
        if tuple(getattr(target, "shape", ())) != tuple(getattr(tensor, "shape", ())):
            shape_mismatches.append(key)
            continue
        if str(getattr(target, "dtype", "")) != str(getattr(tensor, "dtype", "")):
            dtype_conversions += 1
        if torch is not None:
            with torch.no_grad():
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        else:
            target.copy_(tensor)
        applied.append(key)
        applied_bytes += _tensor_nbytes(target)
    ready = bool(
        applied
        and not unexpected_keys
        and not missing_assigned
        and not unknown_model_keys
        and not shape_mismatches
        and set(applied) == assigned_key_set
    )
    if ready:
        diagnosis_codes.update({
            "real_llm_stage_selective_weight_application_ready",
            "real_llm_stage_owned_state_dict_ready",
        })
    else:
        diagnosis_codes.add("real_llm_stage_selective_weight_application_not_ready")
        if unexpected_keys:
            blockers.add("real_llm_stage_selective_cross_stage_weight_loaded")
        if missing_assigned:
            blockers.add("real_llm_stage_selective_assigned_weight_keys_missing")
        if unknown_model_keys:
            blockers.add("real_llm_stage_selective_model_state_keys_missing")
        if shape_mismatches:
            blockers.add("real_llm_stage_selective_weight_shape_mismatch")
        if not applied:
            blockers.add("real_llm_stage_selective_weight_keys_not_applied")
    summary.update({
        "ready": ready,
        "stage_selective_tensor_application_ready": ready,
        "applied_weight_key_count": len(applied),
        "missing_assigned_weight_key_count": len(missing_assigned),
        "unknown_model_key_count": len(unknown_model_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "dtype_conversion_count": int(dtype_conversions),
        "cross_stage_weight_keys_loaded": bool(unexpected_keys),
        "applied_weight_key_digest": _hash_payload(sorted(applied)),
        "applied_tensor_bytes": int(applied_bytes),
        "loads_only_stage_weight_keys": bool(ready and not unexpected_keys),
        "diagnosis_codes": sorted(diagnosis_codes),
        "blockers": sorted(blockers),
    })
    return summary


def run_stage_selective_runtime_smoke(
    *,
    tokenizer: Any,
    stage0_model: Any,
    stage1_model: Any,
    baseline_model: Any,
    metadata: dict[str, Any] | None = None,
    prompt: str = "CrowdTensor routes home GPU",
    backend: str = BACKEND_CPU,
) -> dict[str, Any]:
    """Execute a two-stage Llama-like smoke using separately loaded stage models.

    The result is public-safe. It proves that the stage-owned weights can drive
    the existing activation/decode path, while keeping 7B/Kaggle readiness as a
    separate external-runtime requirement.
    """

    source = dict(metadata or {})
    family = execution_family_from_metadata(source)
    split_index = _split_index_for_layer_count(source, _layer_count_from_metadata(source))
    if family not in SUPPORTED_EXECUTION_FAMILIES:
        return {
            "schema": REAL_LLM_STAGE_SELECTIVE_RUNTIME_SCHEMA_VERSION,
            "ready": False,
            "stage_selective_runtime_execution_ready": False,
            "model_id": str(source.get("model_id") or ""),
            "execution_family": family,
            "partition_mode": normalize_partition_mode(source.get("partition_mode") or PARTITION_MODE_STAGE_LOCAL),
            "diagnosis_codes": ["real_llm_stage_selective_runtime_execution_not_ready"],
            "blockers": ["real_llm_execution_architecture_unsupported"],
            "public_safe": True,
        }
    import torch  # type: ignore

    device = torch.device("cpu")
    stage0_model.eval()
    stage1_model.eval()
    baseline_model.eval()
    spec = {
        "schema_version": REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
        "type": WORKLOAD_TYPE,
        "session_id": "stage-selective-runtime-smoke",
        "task_id": "stage-selective-runtime-smoke-stage0",
        "miner_id": "stage-selective-runtime-stage0",
        "model_id": str(source.get("model_id") or ""),
        "artifact_hash": "sha256:stage-selective-runtime-smoke",
        "split_index": split_index,
        "stage_id": 0,
        "max_new_tokens": 1,
        "generation_step": 0,
    }
    request = {
        "request_id": "req-1",
        "prompt": str(prompt)[:MAX_PROMPT_CHARS],
        "prompt_hash": _prompt_hash(str(prompt)[:MAX_PROMPT_CHARS]),
        "max_new_tokens": 1,
        "generated_token_ids": [],
        "generated_text": "",
        "generation_step": 0,
    }
    activation = _stage0_activation(
        tokenizer=tokenizer,
        model=stage0_model,
        request=request,
        spec=spec,
        split_index=split_index,
        device=device,
        family=family,
    )
    stage1_spec = {
        **spec,
        "task_id": "stage-selective-runtime-smoke-stage1",
        "miner_id": "stage-selective-runtime-stage1",
        "stage_id": 1,
    }
    result = _stage1_result(
        tokenizer=tokenizer,
        model=stage1_model,
        baseline_model=baseline_model,
        activation=activation,
        spec=stage1_spec,
        device=device,
        baseline_device=device,
        family=family,
    )
    ready = bool(
        activation.get("activation_hash")
        and result.get("baseline_match")
        and result.get("generated_token_count") == 1
    )
    return {
        "schema": REAL_LLM_STAGE_SELECTIVE_RUNTIME_SCHEMA_VERSION,
        "ready": ready,
        "stage_selective_runtime_execution_ready": ready,
        "runtime_execution_scope": "local_synthetic_stage_selective_runtime",
        "model_id": str(source.get("model_id") or ""),
        "execution_family": family,
        "backend": backend,
        "partition_mode": normalize_partition_mode(source.get("partition_mode") or PARTITION_MODE_STAGE_LOCAL),
        "stage_count": 2,
        "split_index": split_index,
        "generated_token_count": int(result.get("generated_token_count") or 0),
        "baseline_match": bool(result.get("baseline_match")),
        "decoded_tokens_match": bool(result.get("baseline_match")),
        "activation_transport_ready": bool(activation.get("activation_hash")),
        "activation_hash": str(activation.get("activation_hash") or ""),
        "output_hash": str(result.get("output_hash") or ""),
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "public_safe": True,
        "large_model_validation": False,
        "kaggle_runtime_validation": False,
        "diagnosis_codes": [
            "real_llm_stage_selective_runtime_execution_ready"
            if ready
            else "real_llm_stage_selective_runtime_execution_not_ready",
            "activation_transport_ready" if activation.get("activation_hash") else "activation_transport_missing",
            "baseline_match" if result.get("baseline_match") else "baseline_mismatch",
            "decoded_tokens_match" if result.get("baseline_match") else "decoded_tokens_mismatch",
        ],
        "blockers": [] if ready else ["real_llm_stage_selective_runtime_execution_failed"],
    }


def _partial_weight_tensor_materialization_ready(source: dict[str, Any]) -> bool:
    if source.get("partial_weight_tensor_materialization_ready") is True:
        return True
    if source.get("stage_selective_weight_load_ready") is True:
        return True
    raw = source.get("stage_selective_weight_load")
    if isinstance(raw, dict):
        return bool(
            raw.get("ready")
            and raw.get("stage_selective_tensor_materialization_ready")
            and raw.get("loads_only_stage_weight_keys")
        )
    raw_list = source.get("stage_selective_weight_load_summaries")
    if isinstance(raw_list, list) and raw_list:
        return all(
            isinstance(item, dict)
            and item.get("ready")
            and item.get("stage_selective_tensor_materialization_ready")
            and item.get("loads_only_stage_weight_keys")
            for item in raw_list
        )
    return False


def _partial_weight_tensor_application_ready(source: dict[str, Any]) -> bool:
    if source.get("partial_weight_tensor_application_ready") is True:
        return True
    raw = source.get("stage_selective_weight_application")
    if isinstance(raw, dict):
        return bool(
            raw.get("ready")
            and raw.get("stage_selective_tensor_application_ready")
            and raw.get("loads_only_stage_weight_keys")
        )
    raw_list = source.get("stage_selective_weight_application_summaries")
    if isinstance(raw_list, list) and raw_list:
        return all(
            isinstance(item, dict)
            and item.get("ready")
            and item.get("stage_selective_tensor_application_ready")
            and item.get("loads_only_stage_weight_keys")
            for item in raw_list
        )
    return False


def _partial_weight_runtime_execution_ready(source: dict[str, Any]) -> bool:
    if source.get("partial_weight_runtime_execution_ready") is True:
        return True
    raw = source.get("stage_selective_runtime")
    if isinstance(raw, dict):
        return bool(
            raw.get("ready")
            and raw.get("stage_selective_runtime_execution_ready")
            and raw.get("baseline_match")
            and raw.get("decoded_tokens_match")
        )
    return False


def real_llm_execution_support_summary(
    metadata: dict[str, Any] | None = None,
    *,
    partition_mode: str | None = None,
) -> dict[str, Any]:
    source = dict(metadata or {})
    mode = normalize_partition_mode(partition_mode or source.get("partition_mode") or PARTITION_MODE_FULL)
    family = execution_family_from_metadata(source)
    current_supported = family in SUPPORTED_EXECUTION_FAMILIES
    parameter_count_estimate = estimate_parameter_count_from_metadata(source, family=family)
    partial_plan = real_llm_partial_weight_loading_plan(source, partition_mode=mode)
    partial_plan_ready = bool(partial_plan.get("ready"))
    tensor_materialization_ready = _partial_weight_tensor_materialization_ready(source)
    tensor_application_ready = _partial_weight_tensor_application_ready(source)
    runtime_execution_ready = _partial_weight_runtime_execution_ready(source)
    small_tier_candidate = bool(
        SMALL_TIER_MIN_PARAMETERS <= parameter_count_estimate <= SMALL_TIER_MAX_PARAMETERS
    )
    large_candidate = _large_model_candidate_from_metadata(source, family=family)
    diagnosis_codes: list[str] = []
    blockers: list[str] = []
    large_model_blockers: list[str] = [] if partial_plan_ready else ["real_llm_true_partial_weight_loading_missing"]
    if family == EXECUTION_FAMILY_GPT2:
        diagnosis_codes.extend([
            "real_llm_gpt2_execution_family",
            "real_llm_current_stage_split_supported",
        ])
    elif family == EXECUTION_FAMILY_LLAMA_LIKE:
        diagnosis_codes.extend([
            "real_llm_llama_like_execution_family",
            "real_llm_current_stage_split_supported",
            "real_llm_llama_like_stage_runtime_adapter_ready",
        ])
        large_model_blockers.append("real_llm_llama_like_runtime_execution_missing")
    elif family == EXECUTION_FAMILY_UNKNOWN:
        diagnosis_codes.extend([
            "real_llm_unknown_execution_family",
            "real_llm_current_stage_split_unsupported",
        ])
        blockers.append("real_llm_execution_family_unknown")
        large_model_blockers.append("real_llm_execution_family_unknown")
    else:
        diagnosis_codes.extend([
            "real_llm_unsupported_hf_execution_family",
            "real_llm_current_stage_split_unsupported",
        ])
        blockers.append("real_llm_execution_architecture_unsupported")
        large_model_blockers.append("real_llm_execution_architecture_unsupported")
    if mode == PARTITION_MODE_STAGE_LOCAL:
        if runtime_execution_ready:
            diagnosis_codes.append("real_llm_stage_selective_runtime_execution_ready")
        if tensor_application_ready or runtime_execution_ready:
            diagnosis_codes.append("real_llm_stage_selective_weight_application_ready")
        if tensor_materialization_ready or tensor_application_ready or runtime_execution_ready:
            diagnosis_codes.append("real_llm_stage_selective_weight_materialization_ready")
        if partial_plan_ready:
            diagnosis_codes.append("real_llm_stage_local_partial_weight_plan_ready")
        else:
            diagnosis_codes.append("real_llm_stage_local_full_model_cpu_load_required")
    diagnosis_codes.extend(str(code) for code in partial_plan.get("diagnosis_codes") or [])
    if large_candidate:
        diagnosis_codes.append("real_llm_large_model_candidate_detected")
        if not current_supported:
            diagnosis_codes.append(
                "real_llm_large_model_runtime_stage_adapter_missing"
                if partial_plan_ready
                else "real_llm_large_model_stage_adapter_missing"
            )
        elif partial_plan_ready and not runtime_execution_ready and not bool(partial_plan.get("runtime_execution_ready")):
            diagnosis_codes.append("real_llm_large_model_partial_weight_runtime_missing")
    else:
        diagnosis_codes.append("real_llm_tiny_or_small_model_candidate")
    if small_tier_candidate:
        diagnosis_codes.append("real_llm_1b_3b_small_tier_candidate_detected")
        if current_supported:
            diagnosis_codes.append("real_llm_1b_3b_small_tier_current_split_supported")
    estimated_weight_bytes = parameter_count_estimate * FP32_BYTES_PER_PARAMETER
    estimated_stage_weight_bytes = int(estimated_weight_bytes // 2) if estimated_weight_bytes else 0
    return {
        "execution_family": family,
        "current_stage_split_supported": current_supported,
        "supported_execution_families": sorted(SUPPORTED_EXECUTION_FAMILIES),
        "parameter_count_estimate": parameter_count_estimate,
        "estimated_weight_bytes_fp32": estimated_weight_bytes,
        "stage_local_estimated_stage_weight_bytes_fp32": estimated_stage_weight_bytes,
        "stage_local_load_strategy": (
            "stage_weight_index_selective_runtime_execution"
            if mode == PARTITION_MODE_STAGE_LOCAL and runtime_execution_ready
            else
            "stage_weight_index_selective_tensor_application"
            if mode == PARTITION_MODE_STAGE_LOCAL and (tensor_application_ready or runtime_execution_ready)
            else "stage_weight_index_selective_tensor_materialization"
            if mode == PARTITION_MODE_STAGE_LOCAL and (tensor_materialization_ready or tensor_application_ready or runtime_execution_ready)
            else "stage_weight_index_selective_load_plan"
            if mode == PARTITION_MODE_STAGE_LOCAL and partial_plan_ready
            else "full_model_cpu_load_then_stage_module_device_move"
            if mode == PARTITION_MODE_STAGE_LOCAL
            else "full_model_load"
        ),
        "partial_weight_loading_plan_ready": partial_plan_ready,
        "partial_weight_loading_plan": partial_plan,
        "partial_weight_tensor_materialization_ready": bool(tensor_materialization_ready or tensor_application_ready or runtime_execution_ready),
        "partial_weight_tensor_application_ready": bool(tensor_application_ready or runtime_execution_ready),
        "true_partial_weight_loading_ready": bool(tensor_materialization_ready or tensor_application_ready or runtime_execution_ready),
        "partial_weight_runtime_execution_ready": runtime_execution_ready,
        "small_tier_candidate": small_tier_candidate,
        "kaggle_small_tier_supported_by_current_split": bool(small_tier_candidate and current_supported),
        "large_model_candidate": large_candidate,
        "large_model_sharded_execution_ready": False,
        "large_model_blockers": sorted(set(large_model_blockers)),
        "blockers": sorted(set(blockers)),
        "diagnosis_codes": sorted(set(diagnosis_codes)),
    }


def _json_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_payload(value).encode("utf-8")).hexdigest()


def _round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, ROUND_DIGITS)
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    return value


def _prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def _ensure_batched_hidden(hidden: Any) -> Any:
    if int(getattr(hidden, "ndim", 0)) == 2:
        return hidden.unsqueeze(0)
    return hidden


def _stage0_cache_key(*, spec: dict[str, Any], request: dict[str, Any], split_index: int) -> tuple[str, str, str, int, str]:
    return (
        str(spec.get("session_id") or ""),
        str(request.get("request_id") or ""),
        str(spec.get("artifact_hash") or ""),
        int(split_index),
        str(spec.get("miner_id") or ""),
    )


def _stage1_cache_key(*, spec: dict[str, Any], activation: dict[str, Any], split_index: int) -> tuple[str, str, str, int, str]:
    return (
        str(spec.get("session_id") or activation.get("session_id") or ""),
        str(activation.get("request_id") or ""),
        str(spec.get("artifact_hash") or activation.get("artifact_hash") or ""),
        int(split_index),
        str(spec.get("miner_id") or ""),
    )


def _block_output_hidden_and_present(output: Any) -> tuple[Any, Any | None]:
    if not isinstance(output, (tuple, list)):
        return output, None
    hidden = output[0]
    present = getattr(output, "present", None)
    if present is None:
        present = getattr(output, "past_key_values", None)
    if present is None and isinstance(output, (tuple, list)) and len(output) > 1:
        present = output[1]
    return hidden, present


def _decoder_output_hidden(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _new_dynamic_cache(
    model: Any,
    stored_layers: list[Any] | None = None,
    device: Any | None = None,
    *,
    layer_indices: list[int] | None = None,
) -> Any | None:
    try:
        from transformers.cache_utils import DynamicCache  # type: ignore
    except Exception:
        return None
    try:
        cache = DynamicCache(config=getattr(model, "config", None))
    except Exception:
        try:
            cache = DynamicCache()
        except Exception:
            return None
    if not stored_layers:
        return cache
    indices = list(layer_indices or range(len(stored_layers)))
    try:
        layers = list(cache.layers)
    except Exception:
        layers = []
    if not layers:
        ddp_cache_data = []
        for layer in stored_layers:
            values = list(layer or [])
            if len(values) < 2 or values[0] is None or values[1] is None:
                return None
            values = [value.to(device) for value in values[:2]] if device is not None else values[:2]
            ddp_cache_data.append(tuple(values))
        try:
            return DynamicCache(ddp_cache_data=ddp_cache_data)
        except Exception:
            return None
    if len(indices) != len(stored_layers):
        return None
    try:
        for layer_index, stored in zip(indices, stored_layers):
            values = list(stored or [])
            if len(values) < 2 or values[0] is None or values[1] is None:
                return None
            key_value = [value.to(device) for value in values[:2]] if device is not None else values[:2]
            layers[int(layer_index)].update(key_value[0], key_value[1])
    except Exception:
        return None
    return cache


def _cache_layer_values(layer: Any) -> list[Any]:
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if keys is not None and values is not None:
        return [keys, values]
    try:
        return list(layer or [])
    except TypeError:
        return []


def _cache_layers(cache: Any, *, split: int, layer_indices: list[int] | None = None) -> list[Any]:
    rows: list[Any] = []
    indices = list(layer_indices or range(split))
    try:
        layers = list(cache.layers)
    except Exception:
        try:
            layers = list(iter(cache))
        except TypeError:
            return rows
    for index in indices:
        if int(index) < 0 or int(index) >= len(layers):
            return []
        layer = layers[int(index)]
        values = _cache_layer_values(layer)
        if len(values) < 2 or values[0] is None or values[1] is None:
            return []
        rows.append(tuple(value.detach().cpu() for value in values[:2]))
    return rows if len(rows) == len(indices) else []


def _block_cache_argument(block: Any) -> str:
    try:
        parameters = inspect.signature(block.forward).parameters
    except (TypeError, ValueError, AttributeError):
        return "none"
    if "past_key_values" in parameters:
        return "past_key_values"
    if "layer_past" in parameters:
        return "layer_past"
    return "none"


def _move_past_layer(layer: Any, device: Any) -> Any | None:
    values = list(layer or [])
    if len(values) < 2 or values[0] is None or values[1] is None:
        return None
    return tuple(value.to(device) for value in values[:2])


def _detach_present_layer(present: Any) -> Any | None:
    values = list(present or [])
    if len(values) < 2 or values[0] is None or values[1] is None:
        return None
    return tuple(value.detach().cpu() for value in values[:2])


def _call_gpt2_block(
    block: Any,
    hidden: Any,
    *,
    dynamic_cache: Any | None = None,
    layer_past: Any | None = None,
    use_cache: bool = True,
) -> Any:
    cache_argument = _block_cache_argument(block)
    if cache_argument == "past_key_values" and dynamic_cache is not None:
        return block(hidden, past_key_values=dynamic_cache, use_cache=use_cache)
    if cache_argument == "layer_past":
        return block(hidden, layer_past=layer_past, use_cache=use_cache)
    return block(hidden, use_cache=use_cache)


def _causal_attention_mask(*, token_count: int, dtype: Any, device: Any) -> Any:
    import torch  # type: ignore

    count = max(1, int(token_count))
    mask = torch.full((count, count), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _cache_position(*, token_count: int, device: Any) -> Any:
    import torch  # type: ignore

    return torch.arange(max(1, int(token_count)), dtype=torch.long, device=device)


def _llama_like_position_embeddings(base_model: Any, hidden: Any, position_ids: Any) -> Any | None:
    rotary = getattr(base_model, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden, position_ids)
    except TypeError:
        try:
            return rotary(position_ids)
        except TypeError:
            return None


def _call_llama_like_layer(
    layer: Any,
    hidden: Any,
    *,
    attention_mask: Any | None = None,
    position_ids: Any | None = None,
    cache_position: Any | None = None,
    position_embeddings: Any | None = None,
) -> Any:
    try:
        parameters = inspect.signature(layer.forward).parameters
    except (TypeError, ValueError, AttributeError):
        parameters = {}
    kwargs: dict[str, Any] = {}
    if "attention_mask" in parameters:
        kwargs["attention_mask"] = attention_mask
    if "position_ids" in parameters:
        kwargs["position_ids"] = position_ids
    if "past_key_value" in parameters:
        kwargs["past_key_value"] = None
    if "output_attentions" in parameters:
        kwargs["output_attentions"] = False
    if "use_cache" in parameters:
        kwargs["use_cache"] = False
    if "cache_position" in parameters:
        kwargs["cache_position"] = cache_position
    if "position_embeddings" in parameters and position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    return layer(hidden, **kwargs)


def clear_real_llm_runtime_caches() -> None:
    """Clear in-process model/runtime caches used by tests and short-lived Miners."""

    _MODEL_CACHE.clear()
    _STAGE0_KV_CACHE.clear()
    _STAGE1_KV_CACHE.clear()


def _activation_hash(activation: dict[str, Any]) -> str:
    payload = {
        "schema_version": activation.get("schema_version"),
        "session_id": activation.get("session_id"),
        "request_id": activation.get("request_id"),
        "model_id": activation.get("model_id"),
        "artifact_hash": activation.get("artifact_hash"),
        "split_index": activation.get("split_index"),
        "input_ids": activation.get("input_ids"),
        "position_ids": activation.get("position_ids"),
        "hidden_shape": activation.get("hidden_shape"),
        "hidden_state": activation.get("hidden_state"),
    }
    return _hash_payload(payload)


def _output_hash(result: dict[str, Any]) -> str:
    return _hash_payload({
        "request_id": result.get("request_id"),
        "model_id": result.get("model_id"),
        "artifact_hash": result.get("artifact_hash"),
        "activation_hash": result.get("activation_hash"),
        "next_token_id": result.get("next_token_id"),
        "baseline_next_token_id": result.get("baseline_next_token_id"),
        "baseline_match": result.get("baseline_match"),
    })


def _generated_text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _cache_kwargs(cache_dir: str = "") -> dict[str, str]:
    normalized = str(cache_dir or "").strip()
    return {"cache_dir": normalized} if normalized else {}


def _safe_hf_weight_index_metadata(model_id: str, *, cache_dir: str = "") -> dict[str, Any]:
    """Load public-safe HF weight index metadata when it is already available.

    The returned data contains key-to-filename mappings only. It never includes
    local cache paths, raw tensor data, credentials, or repository tokens.
    """

    filenames = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    cache_kwargs = _cache_kwargs(cache_dir)
    for filename in filenames:
        index_path: Path | None = None
        local_candidate = Path(str(model_id)) / filename
        if local_candidate.is_file():
            index_path = local_candidate
        else:
            try:
                from transformers.utils import cached_file  # type: ignore

                cached = cached_file(model_id, filename, **cache_kwargs)
            except Exception:
                cached = None
            if cached:
                candidate = Path(str(cached))
                if candidate.is_file():
                    index_path = candidate
        if index_path is None:
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else {}
        if not isinstance(weight_map, dict):
            continue
        safe_map = _weight_map_from_metadata({"weight_map": weight_map})
        if not safe_map:
            continue
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return {
            "weight_index_schema": filename,
            "weight_index_available": True,
            "weight_map": safe_map,
            "weight_key_count": len(safe_map),
            "weight_file_count": len(set(safe_map.values())),
            "total_size_bytes": int(metadata.get("total_size") or metadata.get("total_size_bytes") or 0),
        }
    return {
        "weight_index_available": False,
        "weight_key_count": 0,
        "weight_file_count": 0,
    }


def _default_model_metadata_artifact(*, model_id: str, split_index: int | None, backend: str) -> dict[str, Any]:
    layer_count = int(DEFAULT_MODEL_MANIFEST["num_hidden_layers"])
    split = int(split_index) if split_index is not None else max(1, layer_count // 2)
    split = max(1, min(split, layer_count - 1))
    artifact: dict[str, Any] = {
        "schema": REAL_LLM_ARTIFACT_SCHEMA_VERSION,
        "model_id": model_id,
        "backend": backend,
        "partition_mode": PARTITION_MODE_FULL,
        "model_type": str(DEFAULT_MODEL_MANIFEST["model_type"]),
        "architectures": list(DEFAULT_MODEL_MANIFEST["architectures"]),
        "tokenizer_class": str(DEFAULT_MODEL_MANIFEST["tokenizer_class"]),
        "num_hidden_layers": layer_count,
        "hidden_size": int(DEFAULT_MODEL_MANIFEST["hidden_size"]),
        "vocab_size": int(DEFAULT_MODEL_MANIFEST["vocab_size"]),
        "split_index": split,
        "max_request_count": MAX_REQUESTS,
        "read_only": True,
        "metadata_only": True,
        "metadata_source": "built_in_default_model_manifest",
    }
    artifact["execution_support"] = real_llm_execution_support_summary(artifact)
    artifact["execution_family"] = artifact["execution_support"]["execution_family"]
    artifact["partial_weight_loading_plan"] = artifact["execution_support"]["partial_weight_loading_plan"]
    if backend == BACKEND_CUDA:
        artifact["cuda_runtime"] = {
            "backend": BACKEND_CUDA,
            "cuda_available": False,
            "coordinator_runtime_required": False,
            "miner_runtime_required": True,
            "diagnosis_codes": ["cuda_runtime_deferred_to_miner"],
        }
    artifact["artifact_hash"] = _hash_payload(artifact)
    return artifact


def inspect_real_llm_artifact(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: str = "",
    split_index: int | None = None,
    backend: str = BACKEND_CPU,
    require_runtime: bool = True,
) -> dict[str, Any]:
    """Inspect a tiny HF causal LM and return a safe public artifact manifest."""

    normalized_model_id = str(model_id or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    resolved_backend = resolve_backend(backend) if require_runtime else normalize_backend(backend)
    if resolved_backend == BACKEND_AUTO:
        resolved_backend = BACKEND_CPU
    if not require_runtime and normalized_model_id == DEFAULT_MODEL_ID and not hf_available():
        return _default_model_metadata_artifact(
            model_id=normalized_model_id,
            split_index=split_index,
            backend=resolved_backend,
        )

    require_hf_dependencies()
    from transformers import AutoConfig, AutoTokenizer  # type: ignore

    config = AutoConfig.from_pretrained(normalized_model_id, **_cache_kwargs(cache_dir))
    tokenizer = AutoTokenizer.from_pretrained(normalized_model_id, **_cache_kwargs(cache_dir))
    layer_count = int(
        getattr(config, "n_layer", None)
        or getattr(config, "num_hidden_layers", None)
        or 0
    )
    hidden_size = int(
        getattr(config, "n_embd", None)
        or getattr(config, "hidden_size", None)
        or 0
    )
    vocab_size = int(getattr(config, "vocab_size", 0) or 0)
    if layer_count < 2:
        raise ValueError("real_llm_sharded_infer requires a GPT-like model with at least two layers")
    if hidden_size <= 0 or vocab_size <= 0:
        raise ValueError("real_llm_sharded_infer could not inspect model hidden/vocab sizes")
    split = int(split_index) if split_index is not None else max(1, layer_count // 2)
    split = max(1, min(split, layer_count - 1))
    artifact = {
        "schema": REAL_LLM_ARTIFACT_SCHEMA_VERSION,
        "model_id": normalized_model_id,
        "backend": resolved_backend,
        "partition_mode": PARTITION_MODE_FULL,
        "model_type": str(getattr(config, "model_type", "") or ""),
        "architectures": list(getattr(config, "architectures", []) or []),
        "tokenizer_class": tokenizer.__class__.__name__,
        "num_hidden_layers": layer_count,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "split_index": split,
        "max_request_count": MAX_REQUESTS,
        "read_only": True,
        "metadata_only": not require_runtime,
    }
    weight_index = _safe_hf_weight_index_metadata(normalized_model_id, cache_dir=cache_dir)
    artifact.update({key: value for key, value in weight_index.items() if key != "weight_map"})
    if weight_index.get("weight_map"):
        artifact["weight_map"] = weight_index["weight_map"]
    artifact["execution_support"] = real_llm_execution_support_summary(artifact)
    artifact["execution_family"] = artifact["execution_support"]["execution_family"]
    artifact["partial_weight_loading_plan"] = artifact["execution_support"]["partial_weight_loading_plan"]
    if resolved_backend == BACKEND_CUDA:
        artifact["cuda_runtime"] = (
            cuda_runtime_summary()
            if require_runtime
            else {
                "backend": BACKEND_CUDA,
                "cuda_available": False,
                "coordinator_runtime_required": False,
                "miner_runtime_required": True,
                "diagnosis_codes": ["cuda_runtime_deferred_to_miner"],
            }
        )
    artifact["artifact_hash"] = _hash_payload(artifact)
    return artifact


def _load_model_and_tokenizer(
    model_id: str,
    *,
    cache_dir: str = "",
    backend: str = BACKEND_CPU,
    move_model: bool = True,
):
    require_hf_dependencies()
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    resolved_backend = resolve_backend(backend)
    device = torch.device("cuda:0" if resolved_backend == BACKEND_CUDA else "cpu")
    cache_key = (str(model_id), str(cache_dir or ""), resolved_backend, bool(move_model))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, **_cache_kwargs(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(model_id, **_cache_kwargs(cache_dir))
    if move_model:
        model.to(device)
    model.eval()
    loaded = (tokenizer, model, device)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def _gpt2_parts(model: Any) -> tuple[Any, list[Any]]:
    transformer = getattr(model, "transformer", None)
    blocks = list(getattr(transformer, "h", []) or []) if transformer is not None else []
    if transformer is None or not blocks:
        raise ValueError("real_llm_sharded_infer currently supports GPT-2 style causal LM modules")
    if not hasattr(transformer, "wte") or not hasattr(transformer, "wpe") or not hasattr(transformer, "ln_f"):
        raise ValueError("real_llm_sharded_infer could not find GPT-2 embedding/normalization modules")
    if not hasattr(model, "lm_head"):
        raise ValueError("real_llm_sharded_infer could not find model lm_head")
    return transformer, blocks


def _llama_like_parts(model: Any) -> tuple[Any, list[Any]]:
    base_model = getattr(model, "model", None)
    layers = list(getattr(base_model, "layers", []) or []) if base_model is not None else []
    if base_model is None or not layers:
        raise ValueError("real_llm_sharded_infer could not find Llama-like decoder layers")
    if not hasattr(base_model, "embed_tokens") or not hasattr(base_model, "norm"):
        raise ValueError("real_llm_sharded_infer could not find Llama-like embedding/normalization modules")
    if not hasattr(model, "lm_head"):
        raise ValueError("real_llm_sharded_infer could not find model lm_head")
    return base_model, layers


def _model_parts(model: Any, family: str) -> tuple[Any, list[Any]]:
    if family == EXECUTION_FAMILY_LLAMA_LIKE:
        return _llama_like_parts(model)
    return _gpt2_parts(model)


def _parameter_count(module: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        marker = id(parameter)
        if marker in seen:
            continue
        seen.add(marker)
        total += int(parameter.numel())
    return total


def _module_parameter_count(modules: list[Any]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            marker = id(parameter)
            if marker in seen:
                continue
            seen.add(marker)
            total += int(parameter.numel())
    return total


def _stage_modules(model: Any, *, stage_id: int, split_index: int, family: str = EXECUTION_FAMILY_GPT2) -> tuple[list[Any], tuple[int, int], list[str]]:
    transformer, blocks = _model_parts(model, family)
    split = max(1, min(int(split_index), len(blocks) - 1))
    if family == EXECUTION_FAMILY_LLAMA_LIKE:
        if int(stage_id) == 0:
            return (
                [transformer.embed_tokens, *blocks[:split]],
                (0, split),
                ["token_embedding", "decoder_layers_prefix"],
            )
        return (
            [*blocks[split:], transformer.norm, model.lm_head],
            (split, len(blocks)),
            ["decoder_layers_suffix", "final_norm", "lm_head"],
        )
    if int(stage_id) == 0:
        return (
            [transformer.wte, transformer.wpe, *blocks[:split]],
            (0, split),
            ["token_embedding", "position_embedding", "transformer_blocks_prefix"],
        )
    return (
        [*blocks[split:], transformer.ln_f, model.lm_head],
        (split, len(blocks)),
        ["transformer_blocks_suffix", "final_norm", "lm_head"],
    )


def _move_stage_modules(model: Any, *, stage_id: int, split_index: int, device: Any, family: str = EXECUTION_FAMILY_GPT2) -> None:
    modules, _, _ = _stage_modules(model, stage_id=stage_id, split_index=split_index, family=family)
    for module in modules:
        module.to(device)
        module.eval()


def _partition_summary(
    model: Any,
    *,
    stage_id: int,
    split_index: int,
    partition_mode: str,
    device: Any,
    family: str = EXECUTION_FAMILY_GPT2,
    baseline_device: str = "",
) -> dict[str, Any]:
    transformer, blocks = _model_parts(model, family)
    split = max(1, min(int(split_index), len(blocks) - 1))
    mode = normalize_partition_mode(partition_mode)
    full_count = _parameter_count(model)
    modules, layer_range, module_kinds = _stage_modules(model, stage_id=stage_id, split_index=split, family=family)
    stage_count = _module_parameter_count(modules) if mode == PARTITION_MODE_STAGE_LOCAL else full_count
    split_valid = bool(
        len(blocks) >= 2
        and 0 <= layer_range[0] < layer_range[1] <= len(blocks)
        and (mode != PARTITION_MODE_STAGE_LOCAL or (0 < stage_count < full_count))
    )
    fraction = round(float(stage_count) / float(full_count), 8) if full_count else 0.0
    device_name = str(device)
    summary: dict[str, Any] = {
        "partition_mode": mode,
        "stage_id": int(stage_id),
        "stage_layer_range": [int(layer_range[0]), int(layer_range[1])],
        "stage_layer_range_format": "start_inclusive_end_exclusive",
        "stage_module_kinds": module_kinds,
        "stage_parameter_count": int(stage_count),
        "full_model_parameter_count": int(full_count),
        "stage_parameter_fraction": fraction,
        "device_parameter_count": int(stage_count),
        "partition_parameter_split_valid": split_valid,
        "stage_local_partition_ready": bool(mode == PARTITION_MODE_STAGE_LOCAL and split_valid),
        "stage_gpu_memory_reduced": bool(
            mode == PARTITION_MODE_STAGE_LOCAL
            and split_valid
            and device_name.startswith("cuda")
            and stage_count < full_count
        ),
        "stage_cpu_partition_ready": bool(
            mode == PARTITION_MODE_STAGE_LOCAL
            and split_valid
            and not device_name.startswith("cuda")
        ),
    }
    if baseline_device:
        summary["baseline_device"] = baseline_device
    if stage_id == 0:
        summary["stage0_partition_loaded"] = bool(mode == PARTITION_MODE_STAGE_LOCAL and split_valid)
    if stage_id == 1:
        summary["stage1_partition_loaded"] = bool(mode == PARTITION_MODE_STAGE_LOCAL and split_valid)
    return summary


def _normalized_requests(
    *,
    request_count: int,
    max_new_tokens: int = 1,
    prompt_texts: list[str] | None = None,
    requests: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    count = max(1, min(int(request_count), MAX_REQUESTS))
    generation_limit = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))
    rows: list[dict[str, Any]] = []
    if requests:
        for index, row in enumerate(list(requests)[:count]):
            prompt = str(row.get("prompt") or "")
            if not prompt:
                prompt = DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)]
            prompt = prompt[:MAX_PROMPT_CHARS]
            rows.append({
                "request_id": str(row.get("request_id") or f"req-{index + 1}"),
                "prompt": prompt,
                "prompt_hash": str(row.get("prompt_hash") or _prompt_hash(prompt)),
                "max_new_tokens": max(1, min(int(row.get("max_new_tokens", generation_limit)), MAX_NEW_TOKENS)),
                "generated_token_ids": list(row.get("generated_token_ids") or []),
                "generated_text": str(row.get("generated_text") or ""),
                "generation_step": int(row.get("generation_step", 0)),
            })
    source_prompts = list(prompt_texts or DEFAULT_PROMPTS)
    while len(rows) < count:
        index = len(rows)
        prompt = str(source_prompts[index % len(source_prompts)] or DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)])
        prompt = prompt[:MAX_PROMPT_CHARS]
        rows.append({
            "request_id": f"req-{index + 1}",
            "prompt": prompt,
            "prompt_hash": _prompt_hash(prompt),
            "max_new_tokens": generation_limit,
            "generated_token_ids": [],
            "generated_text": "",
            "generation_step": 0,
        })
    return rows


def real_llm_sharded_inference_spec_for(
    task_id: str,
    miner_id: str,
    artifact: dict[str, Any],
    *,
    request_count: int = 1,
    prompt_texts: list[str] | None = None,
    session_id: str = "",
    stage_id: int = 0,
    parent_task_id: str = "",
    max_new_tokens: int = 1,
    generation_step: int = 0,
    requests: list[dict[str, Any]] | None = None,
    activation_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stage = int(stage_id)
    if stage not in {0, 1}:
        raise ValueError("real LLM sharded inference stage_id must be 0 or 1")
    partition_mode = normalize_partition_mode(artifact.get("partition_mode") or PARTITION_MODE_FULL)
    artifact_snapshot = dict(artifact)
    artifact_snapshot["partition_mode"] = partition_mode
    execution_support = real_llm_execution_support_summary(
        artifact_snapshot,
        partition_mode=partition_mode,
    )
    artifact_snapshot["execution_support"] = execution_support
    artifact_snapshot["execution_family"] = execution_support["execution_family"]
    prompt_rows = _normalized_requests(
        request_count=request_count,
        max_new_tokens=max_new_tokens,
        prompt_texts=prompt_texts,
        requests=requests,
    )
    generation_limit = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))
    step = max(0, min(int(generation_step), generation_limit - 1))
    spec = {
        "type": WORKLOAD_TYPE,
        "schema_version": REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
        "artifact_schema": artifact.get("schema") or REAL_LLM_ARTIFACT_SCHEMA_VERSION,
        "artifact_hash": artifact.get("artifact_hash"),
        "artifact": artifact_snapshot,
        "model_id": artifact.get("model_id") or DEFAULT_MODEL_ID,
        "backend": artifact.get("backend") or "hf_transformers_cpu",
        "partition_mode": partition_mode,
        "execution_family": execution_support["execution_family"],
        "execution_support": execution_support,
        "session_id": str(session_id or task_id),
        "stage_id": stage,
        "stage_count": 2,
        "parent_task_id": str(parent_task_id or ""),
        "task_id": str(task_id),
        "miner_id": str(miner_id or "anonymous"),
        "request_count": len(prompt_rows),
        "requests": prompt_rows,
        "split_index": int(artifact.get("split_index", 1)),
        "num_hidden_layers": int(artifact.get("num_hidden_layers", 0)),
        "hidden_size": int(artifact.get("hidden_size", 0)),
        "max_new_tokens": generation_limit,
        "generation_step": step,
        "read_only": True,
    }
    if stage == 1:
        spec["activation_results"] = list(activation_results or [])
        spec["activation_hashes"] = [
            str(row.get("activation_hash") or "")
            for row in spec["activation_results"]
            if isinstance(row, dict)
        ]
    return spec


def _tokenize_prompt(tokenizer: Any, prompt: str, *, generated_token_ids: list[int] | None = None):
    import torch  # type: ignore

    encoded = tokenizer(str(prompt), return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.get("input_ids")
    if input_ids is None or int(input_ids.numel()) <= 0:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        input_ids = torch.tensor([[int(eos_id or 0)]], dtype=torch.long)
    continuation = [int(value) for value in list(generated_token_ids or [])]
    if continuation:
        continuation_ids = torch.tensor([continuation], dtype=torch.long)
        input_ids = torch.cat([input_ids, continuation_ids], dim=1)
    return input_ids


def _stage0_activation(
    *,
    tokenizer: Any,
    model: Any,
    request: dict[str, Any],
    spec: dict[str, Any],
    split_index: int,
    device: Any,
    family: str = EXECUTION_FAMILY_GPT2,
) -> dict[str, Any]:
    import torch  # type: ignore

    transformer, blocks = _model_parts(model, family)
    split = max(1, min(int(split_index), len(blocks) - 1))
    generated_prefix_token_ids = [int(value) for value in list(request.get("generated_token_ids") or [])]
    input_ids = _tokenize_prompt(
        tokenizer,
        str(request.get("prompt") or ""),
        generated_token_ids=generated_prefix_token_ids,
    ).to(device)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
    cache_key = _stage0_cache_key(spec=spec, request=request, split_index=split)
    cached = _STAGE0_KV_CACHE.get(cache_key) if generated_prefix_token_ids else None
    cache_ready = False
    cache_hit = False
    cache_tokens_before = 0
    hidden = None
    with torch.no_grad():
        if family == EXECUTION_FAMILY_LLAMA_LIKE:
            position_embeddings = _llama_like_position_embeddings(transformer, transformer.embed_tokens(input_ids), position_ids)
            attention_mask = _causal_attention_mask(token_count=int(input_ids.shape[1]), dtype=transformer.embed_tokens(input_ids).dtype, device=device)
            cache_pos = _cache_position(token_count=int(input_ids.shape[1]), device=device)
            hidden = transformer.embed_tokens(input_ids)
            for layer in blocks[:split]:
                output = _call_llama_like_layer(
                    layer,
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_pos,
                    position_embeddings=position_embeddings,
                )
                hidden = _ensure_batched_hidden(_decoder_output_hidden(output))
        elif cached and int(cached.get("input_token_count") or 0) == int(input_ids.shape[1]) - 1:
            previous_hidden = cached.get("hidden")
            past_key_values = list(cached.get("past_key_values") or [])
            cache = _new_dynamic_cache(model, stored_layers=past_key_values, device=device)
            if previous_hidden is not None and (cache is not None or len(past_key_values) == split):
                cache_hit = True
                cache_tokens_before = int(cached.get("input_token_count") or 0)
                next_input_ids = input_ids[:, -1:]
                next_position_ids = position_ids[:, -1:]
                hidden_delta = transformer.wte(next_input_ids) + transformer.wpe(next_position_ids)
                legacy_past: list[Any] = []
                for block, past in zip(blocks[:split], past_key_values):
                    output = _call_gpt2_block(
                        block,
                        hidden_delta,
                        dynamic_cache=cache,
                        layer_past=_move_past_layer(past, device),
                        use_cache=True,
                    )
                    block_hidden, _present = _block_output_hidden_and_present(output)
                    detached_present = _detach_present_layer(_present)
                    if detached_present is not None:
                        legacy_past.append(detached_present)
                    hidden_delta = _ensure_batched_hidden(block_hidden)
                next_past_key_values = _cache_layers(cache, split=split)
                if len(next_past_key_values) != split and len(legacy_past) == split:
                    next_past_key_values = legacy_past
                if len(next_past_key_values) == split:
                    hidden = torch.cat([previous_hidden.to(device), hidden_delta], dim=1)
                    _STAGE0_KV_CACHE[cache_key] = {
                        "input_token_count": int(input_ids.shape[1]),
                        "hidden": hidden.detach().cpu(),
                        "past_key_values": next_past_key_values,
                    }
                    cache_ready = True
        if hidden is None:
            hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)
            cache = _new_dynamic_cache(model)
            legacy_past: list[Any] = []
            for block in blocks[:split]:
                output = _call_gpt2_block(block, hidden, dynamic_cache=cache, use_cache=True)
                block_hidden, _present = _block_output_hidden_and_present(output)
                detached_present = _detach_present_layer(_present)
                if detached_present is not None:
                    legacy_past.append(detached_present)
                hidden = _ensure_batched_hidden(block_hidden)
            past_key_values = _cache_layers(cache, split=split) if cache is not None else []
            if len(past_key_values) != split and len(legacy_past) == split:
                past_key_values = legacy_past
            if len(past_key_values) == split:
                _STAGE0_KV_CACHE[cache_key] = {
                    "input_token_count": int(input_ids.shape[1]),
                    "hidden": hidden.detach().cpu(),
                    "past_key_values": past_key_values,
                }
                cache_ready = True
    hidden_state = _round_nested(hidden.detach().cpu().tolist())
    activation = {
        "schema_version": REAL_LLM_ACTIVATION_SCHEMA_VERSION,
        "session_id": spec.get("session_id"),
        "request_id": request.get("request_id"),
        "prompt_hash": request.get("prompt_hash"),
        "model_id": spec.get("model_id"),
        "artifact_hash": spec.get("artifact_hash"),
        "split_index": split,
        "generation_step": int(spec.get("generation_step", request.get("generation_step", 0))),
        "max_new_tokens": int(spec.get("max_new_tokens", request.get("max_new_tokens", 1))),
        "generated_token_ids": list(request.get("generated_token_ids") or []),
        "generated_text": str(request.get("generated_text") or ""),
        "input_ids": [int(value) for value in input_ids.detach().cpu().tolist()[0]],
        "position_ids": [int(value) for value in position_ids.detach().cpu().tolist()[0]],
        "hidden_shape": [int(value) for value in hidden.shape],
        "hidden_state": hidden_state,
        "prompt_token_count": int(input_ids.shape[1] - len(generated_prefix_token_ids)),
        "generated_prefix_token_count": len(generated_prefix_token_ids),
        "input_token_count": int(input_ids.shape[1]),
        "token_continuation_ready": bool(generated_prefix_token_ids),
        "kv_cache_schema": "real_llm_stage0_kv_cache_v1",
        "kv_cache_ready": cache_ready,
        "kv_cache_hit": cache_hit,
        "kv_cache_tokens_before": cache_tokens_before,
        "kv_cache_tokens_after": int(input_ids.shape[1]),
        "kv_cache_stage": "stage0_prefix",
        "kv_cache_disabled_reason": "llama_like_stage_cache_not_implemented" if family == EXECUTION_FAMILY_LLAMA_LIKE else "",
    }
    activation["activation_hash"] = _activation_hash(activation)
    return activation


def _stage1_result(
    *,
    tokenizer: Any,
    model: Any,
    baseline_model: Any | None = None,
    activation: dict[str, Any],
    spec: dict[str, Any],
    device: Any,
    baseline_device: Any | None = None,
    family: str = EXECUTION_FAMILY_GPT2,
) -> dict[str, Any]:
    import torch  # type: ignore

    transformer, blocks = _model_parts(model, family)
    split = max(1, min(int(activation.get("split_index", spec.get("split_index", 1))), len(blocks) - 1))
    input_ids = torch.tensor([list(activation.get("input_ids") or [])], dtype=torch.long, device=device)
    if input_ids.numel() <= 0:
        raise ValueError("real LLM activation input_ids are empty")
    hidden = _ensure_batched_hidden(torch.tensor(activation.get("hidden_state"), dtype=torch.float32, device=device))
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("real LLM activation hidden_state has invalid shape")
    cache_key = _stage1_cache_key(spec=spec, activation=activation, split_index=split)
    input_token_ids = [int(value) for value in list(activation.get("input_ids") or [])]
    generated_prefix_token_ids = [int(value) for value in list(activation.get("generated_token_ids") or [])]
    suffix_layer_indices = list(range(split, len(blocks)))
    cached = _STAGE1_KV_CACHE.get(cache_key) if generated_prefix_token_ids else None
    cache_ready = False
    cache_hit = False
    cache_tokens_before = 0
    with torch.no_grad():
        if family == EXECUTION_FAMILY_LLAMA_LIKE:
            position_ids = torch.tensor([list(activation.get("position_ids") or range(hidden.shape[1]))], dtype=torch.long, device=device)
            attention_mask = _causal_attention_mask(token_count=int(hidden.shape[1]), dtype=hidden.dtype, device=device)
            cache_pos = _cache_position(token_count=int(hidden.shape[1]), device=device)
            position_embeddings = _llama_like_position_embeddings(transformer, hidden, position_ids)
            for layer in blocks[split:]:
                output = _call_llama_like_layer(
                    layer,
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_pos,
                    position_embeddings=position_embeddings,
                )
                hidden = _ensure_batched_hidden(_decoder_output_hidden(output))
            cache_ready = False
        elif (
            cached
            and int(cached.get("input_token_count") or 0) == int(hidden.shape[1]) - 1
            and list(cached.get("input_token_ids") or []) == input_token_ids[:-1]
        ):
            previous_hidden = cached.get("hidden")
            past_key_values = list(cached.get("past_key_values") or [])
            cache = _new_dynamic_cache(
                model,
                stored_layers=past_key_values,
                device=device,
                layer_indices=suffix_layer_indices,
            )
            if previous_hidden is not None and (cache is not None or len(past_key_values) == len(suffix_layer_indices)):
                cache_tokens_before = int(cached.get("input_token_count") or 0)
                hidden_delta = hidden[:, -1:, :]
                legacy_past: list[Any] = []
                for block, past in zip(blocks[split:], past_key_values):
                    output = _call_gpt2_block(
                        block,
                        hidden_delta,
                        dynamic_cache=cache,
                        layer_past=_move_past_layer(past, device),
                        use_cache=True,
                    )
                    block_hidden, _present = _block_output_hidden_and_present(output)
                    detached_present = _detach_present_layer(_present)
                    if detached_present is not None:
                        legacy_past.append(detached_present)
                    hidden_delta = _ensure_batched_hidden(block_hidden)
                next_past_key_values = _cache_layers(cache, split=len(blocks), layer_indices=suffix_layer_indices)
                if len(next_past_key_values) != len(suffix_layer_indices) and len(legacy_past) == len(suffix_layer_indices):
                    next_past_key_values = legacy_past
                if len(next_past_key_values) == len(suffix_layer_indices):
                    hidden = torch.cat([previous_hidden.to(device), hidden_delta], dim=1)
                    _STAGE1_KV_CACHE[cache_key] = {
                        "input_token_count": int(hidden.shape[1]),
                        "input_token_ids": input_token_ids,
                        "hidden": hidden.detach().cpu(),
                        "past_key_values": next_past_key_values,
                    }
                    cache_ready = True
                    cache_hit = True
        if family != EXECUTION_FAMILY_LLAMA_LIKE and not cache_ready:
            cache = _new_dynamic_cache(model)
            legacy_past: list[Any] = []
            for block in blocks[split:]:
                output = _call_gpt2_block(block, hidden, dynamic_cache=cache, use_cache=True)
                block_hidden, _present = _block_output_hidden_and_present(output)
                detached_present = _detach_present_layer(_present)
                if detached_present is not None:
                    legacy_past.append(detached_present)
                hidden = _ensure_batched_hidden(block_hidden)
            past_key_values = (
                _cache_layers(cache, split=len(blocks), layer_indices=suffix_layer_indices)
                if cache is not None
                else []
            )
            if len(past_key_values) != len(suffix_layer_indices) and len(legacy_past) == len(suffix_layer_indices):
                past_key_values = legacy_past
            if len(past_key_values) == len(suffix_layer_indices):
                _STAGE1_KV_CACHE[cache_key] = {
                    "input_token_count": int(hidden.shape[1]),
                    "input_token_ids": input_token_ids,
                    "hidden": hidden.detach().cpu(),
                    "past_key_values": past_key_values,
                }
                cache_ready = True
        hidden = transformer.norm(hidden) if family == EXECUTION_FAMILY_LLAMA_LIKE else transformer.ln_f(hidden)
        logits = model.lm_head(hidden)
        next_token_id = int(torch.argmax(logits[0, -1, :]).item())
        baseline_target = baseline_model if baseline_model is not None else model
        baseline_input_ids = input_ids.to(baseline_device) if baseline_device is not None else input_ids
        baseline = baseline_target(input_ids=baseline_input_ids)
        baseline_next_token_id = int(torch.argmax(baseline.logits[0, -1, :]).item())
    next_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
    baseline_text = tokenizer.decode([baseline_next_token_id], skip_special_tokens=False)
    prior_token_ids = [int(value) for value in list(activation.get("generated_token_ids") or [])]
    prior_text = str(activation.get("generated_text") or "")
    generated_token_ids = [*prior_token_ids, next_token_id]
    generated_text = prior_text + next_text
    result = {
        "request_id": activation.get("request_id"),
        "prompt_hash": activation.get("prompt_hash"),
        "model_id": spec.get("model_id"),
        "artifact_hash": spec.get("artifact_hash"),
        "activation_hash": activation.get("activation_hash"),
        "generation_step": int(activation.get("generation_step", spec.get("generation_step", 0))),
        "max_new_tokens": int(activation.get("max_new_tokens", spec.get("max_new_tokens", 1))),
        "next_token_id": next_token_id,
        "next_token_text": next_text,
        "baseline_next_token_id": baseline_next_token_id,
        "baseline_next_token_text": baseline_text,
        "generated_token_ids": generated_token_ids,
        "generated_token_count": len(generated_token_ids),
        "generated_text": generated_text,
        "generated_text_hash": _generated_text_hash(generated_text),
        "baseline_match": next_token_id == baseline_next_token_id and next_text == baseline_text,
        "baseline_device": str(baseline_device or device),
        "kv_cache_schema": "real_llm_stage1_kv_cache_v1",
        "kv_cache_ready": cache_ready,
        "kv_cache_hit": cache_hit,
        "kv_cache_tokens_before": cache_tokens_before,
        "kv_cache_tokens_after": int(input_ids.shape[1]),
        "kv_cache_stage": "stage1_suffix",
        "kv_cache_disabled_reason": "llama_like_stage_cache_not_implemented" if family == EXECUTION_FAMILY_LLAMA_LIKE else "",
    }
    result["output_hash"] = _output_hash(result)
    return result


def run_real_llm_sharded_inference(workload_spec: dict[str, Any], *, cache_dir: str = "") -> dict[str, Any]:
    start = time.monotonic()
    spec = dict(workload_spec or {})
    if str(spec.get("schema_version")) != REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION:
        raise ValueError("real LLM sharded workload spec schema mismatch")
    stage_id = int(spec.get("stage_id", -1))
    if stage_id not in {0, 1}:
        raise ValueError("real LLM sharded inference stage_id must be 0 or 1")
    model_id = str(spec.get("model_id") or DEFAULT_MODEL_ID)
    backend = resolve_backend(str(spec.get("backend") or BACKEND_CPU))
    partition_mode = normalize_partition_mode(spec.get("partition_mode") or PARTITION_MODE_FULL)
    artifact_metadata = dict(spec.get("artifact") or {})
    artifact_metadata.setdefault("model_id", model_id)
    artifact_metadata.setdefault("partition_mode", partition_mode)
    artifact_metadata.setdefault("model_type", spec.get("model_type") or "")
    artifact_metadata.setdefault("architectures", spec.get("architectures") or [])
    artifact_metadata.setdefault("num_hidden_layers", spec.get("num_hidden_layers") or 0)
    artifact_metadata.setdefault("hidden_size", spec.get("hidden_size") or 0)
    execution_support = real_llm_execution_support_summary(
        artifact_metadata,
        partition_mode=partition_mode,
    )
    if not execution_support["current_stage_split_supported"]:
        raise ValueError(
            "real_llm_sharded_infer currently supports GPT-2 style causal LM modules; "
            f"execution_family={execution_support['execution_family']} blockers="
            + ",".join(execution_support["blockers"])
        )
    tokenizer, model, device = _load_model_and_tokenizer(
        model_id,
        cache_dir=cache_dir,
        backend=backend,
        move_model=partition_mode == PARTITION_MODE_FULL,
    )
    split_index = int(spec.get("split_index", 1))
    max_new_tokens = max(1, min(int(spec.get("max_new_tokens", 1)), MAX_NEW_TOKENS))
    generation_step = max(0, min(int(spec.get("generation_step", 0)), max_new_tokens - 1))
    if partition_mode == PARTITION_MODE_STAGE_LOCAL:
        _move_stage_modules(
            model,
            stage_id=stage_id,
            split_index=split_index,
            device=device,
            family=execution_support["execution_family"],
        )
    partition = _partition_summary(
        model,
        stage_id=stage_id,
        split_index=split_index,
        partition_mode=partition_mode,
        device=device,
        family=execution_support["execution_family"],
        baseline_device="cpu" if partition_mode == PARTITION_MODE_STAGE_LOCAL and stage_id == 1 else "",
    )

    if stage_id == 0:
        activations = [
            _stage0_activation(
                tokenizer=tokenizer,
                model=model,
                request=dict(request),
                spec=spec,
                split_index=split_index,
                device=device,
                family=execution_support["execution_family"],
            )
            for request in list(spec.get("requests") or [])
        ]
        activation_bytes = len(_json_payload(activations).encode("utf-8"))
        return {
            "schema_version": REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
            "type": WORKLOAD_TYPE,
            "session_id": spec.get("session_id"),
            "stage_id": 0,
            "stage_count": 2,
            "model_id": model_id,
            "backend": backend,
            "device": str(device),
            **partition,
            "execution_family": execution_support["execution_family"],
            "execution_support": execution_support,
            "artifact_schema": spec.get("artifact_schema"),
            "artifact_hash": spec.get("artifact_hash"),
            "split_index": split_index,
            "max_new_tokens": max_new_tokens,
            "generation_step": generation_step,
            "request_count": len(activations),
            "activation_count": len(activations),
            "activation_bytes": activation_bytes,
            "activation_hashes": [row["activation_hash"] for row in activations],
            "activation_transport_ready": bool(activations),
            "activation_results": activations,
            "real_llm_artifact_ready": True,
            "elapsed_ms": round((time.monotonic() - start) * 1000.0, 6),
        }

    activations = list(spec.get("activation_results") or [])
    baseline_device = None
    baseline_model = None
    if partition_mode == PARTITION_MODE_STAGE_LOCAL:
        import torch  # type: ignore

        baseline_device = torch.device("cpu")
        _, baseline_model, _ = _load_model_and_tokenizer(
            model_id,
            cache_dir=cache_dir,
            backend=BACKEND_CPU,
            move_model=True,
        )
    results = [
        _stage1_result(
            tokenizer=tokenizer,
            model=model,
            baseline_model=baseline_model,
            activation=dict(activation),
            spec=spec,
            device=device,
            baseline_device=baseline_device,
            family=execution_support["execution_family"],
        )
        for activation in activations
    ]
    baseline_match = bool(results) and all(bool(row.get("baseline_match")) for row in results)
    return {
        "schema_version": REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
        "type": WORKLOAD_TYPE,
        "session_id": spec.get("session_id"),
        "stage_id": 1,
        "stage_count": 2,
        "model_id": model_id,
        "backend": backend,
        "device": str(device),
        **partition,
        "execution_family": execution_support["execution_family"],
        "execution_support": execution_support,
        "artifact_schema": spec.get("artifact_schema"),
        "artifact_hash": spec.get("artifact_hash"),
        "split_index": split_index,
        "max_new_tokens": max_new_tokens,
        "generation_step": generation_step,
        "request_count": len(results),
        "activation_count": len(activations),
        "activation_bytes": len(_json_payload(activations).encode("utf-8")),
        "activation_hashes": [str(row.get("activation_hash") or "") for row in activations],
        "activation_transport_ready": bool(activations),
        "inference_results": results,
        "inference_result": results[0] if results else {},
        "baseline_device": str(baseline_device or device),
        "baseline_match": baseline_match,
        "decoded_tokens_match": baseline_match,
        "generated_token_ids": list((results[0] if results else {}).get("generated_token_ids") or []),
        "generated_token_count": int((results[0] if results else {}).get("generated_token_count") or 0),
        "generated_text": str((results[0] if results else {}).get("generated_text") or ""),
        "generated_text_hash": str((results[0] if results else {}).get("generated_text_hash") or _generated_text_hash("")),
        "real_llm_artifact_ready": True,
        "elapsed_ms": round((time.monotonic() - start) * 1000.0, 6),
    }


def _reject(code: str, reason: str, result: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "accepted": False,
        "code": code,
        "reason": reason,
        "sharded_inference_result": result,
    }


def _safe_trace(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": result.get("request_id"),
        "prompt_hash": result.get("prompt_hash"),
        "generation_step": result.get("generation_step"),
        "max_new_tokens": result.get("max_new_tokens"),
        "next_token_redacted": "next_token_id" in result or "next_token_text" in result,
        "generated_token_count": result.get("generated_token_count"),
        "generated_text_hash": result.get("generated_text_hash"),
        "baseline_match": result.get("baseline_match"),
        "activation_hash": result.get("activation_hash"),
        "output_hash": result.get("output_hash"),
    }


def validate_real_llm_sharded_inference(
    sharded_result: dict[str, Any] | None,
    *,
    expected_spec: dict[str, Any],
    cache_dir: str = "",
    replay_runtime: bool | None = None,
) -> dict[str, Any]:
    if not isinstance(sharded_result, dict):
        return _reject(
            "real_llm_sharded_result_missing",
            "real_llm_sharded_infer requires a sharded_inference_result object",
            sharded_result,
        )
    if str(sharded_result.get("schema_version")) != REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION:
        return _reject(
            "real_llm_sharded_schema_mismatch",
            "sharded result schema_version does not match real_llm_sharded_infer_v1",
            sharded_result,
        )
    try:
        stage_id = int(expected_spec.get("stage_id", sharded_result.get("stage_id", -1)))
    except (TypeError, ValueError):
        stage_id = -1
    if stage_id not in {0, 1} or int(sharded_result.get("stage_id", -1)) != stage_id:
        return _reject(
            "real_llm_sharded_stage_mismatch",
            "sharded result stage_id does not match claim stage",
            sharded_result,
        )
    if str(sharded_result.get("session_id", "")) != str(expected_spec.get("session_id", "")):
        return _reject(
            "real_llm_sharded_session_mismatch",
            "sharded result does not match claim-time session",
            sharded_result,
        )
    if str(sharded_result.get("artifact_hash", "")) != str(expected_spec.get("artifact_hash", "")):
        return _reject(
            "real_llm_artifact_hash_mismatch",
            "sharded result artifact_hash does not match claim-time artifact",
            sharded_result,
        )
    expected_partition_mode = normalize_partition_mode(expected_spec.get("partition_mode") or PARTITION_MODE_FULL)
    observed_partition_mode = normalize_partition_mode(sharded_result.get("partition_mode") or PARTITION_MODE_FULL)
    max_new_tokens = max(1, min(int(expected_spec.get("max_new_tokens", 1)), MAX_NEW_TOKENS))
    generation_step = max(0, min(int(expected_spec.get("generation_step", 0)), max_new_tokens - 1))
    if observed_partition_mode != expected_partition_mode:
        return _reject(
            "real_llm_partition_mode_mismatch",
            "sharded result partition_mode does not match claim-time partition mode",
            sharded_result,
        )
    if expected_partition_mode == PARTITION_MODE_STAGE_LOCAL:
        if not bool(sharded_result.get("stage_local_partition_ready")):
            return _reject(
                "real_llm_stage_local_partition_missing",
                "stage-local partition evidence is missing from sharded result",
                sharded_result,
            )
        if not bool(sharded_result.get("partition_parameter_split_valid")):
            return _reject(
                "real_llm_partition_parameter_split_invalid",
                "stage-local parameter split is not valid",
                sharded_result,
            )
        if int(sharded_result.get("stage_parameter_count", 0)) >= int(sharded_result.get("full_model_parameter_count", 0)):
            return _reject(
                "real_llm_partition_parameter_count_invalid",
                "stage-local parameter count must be smaller than the full model parameter count",
                sharded_result,
            )
        if stage_id == 0 and not bool(sharded_result.get("stage0_partition_loaded")):
            return _reject(
                "real_llm_stage0_partition_missing",
                "stage 0 did not report stage-local partition loading",
                sharded_result,
            )
        if stage_id == 1 and not bool(sharded_result.get("stage1_partition_loaded")):
            return _reject(
                "real_llm_stage1_partition_missing",
                "stage 1 did not report stage-local partition loading",
                sharded_result,
            )

    backend = normalize_backend(str(expected_spec.get("backend") or BACKEND_CPU))
    if replay_runtime is None:
        replay_runtime = backend != BACKEND_CUDA or torch_cuda_available()
    expected: dict[str, Any] = {}
    if replay_runtime:
        try:
            expected = run_real_llm_sharded_inference(expected_spec, cache_dir=cache_dir)
        except Exception as exc:
            return _reject(
                "real_llm_validation_runtime_failed",
                f"real LLM validator could not replay tiny HF runtime: {exc}",
                sharded_result,
            )

    if stage_id == 0:
        observed = list(sharded_result.get("activation_results") or [])
        expected_activations = list(expected.get("activation_results") or [])
        expected_count = len(expected_activations) if replay_runtime else len(list(expected_spec.get("requests") or []))
        if len(observed) != expected_count:
            return _reject(
                "real_llm_activation_count_mismatch",
                "stage 0 activation count does not match claim-time requests",
                sharded_result,
            )
        for index, actual in enumerate(observed):
            if not isinstance(actual, dict):
                return _reject("real_llm_activation_invalid", "activation entry is not an object", sharded_result)
            if str(actual.get("schema_version")) != REAL_LLM_ACTIVATION_SCHEMA_VERSION:
                return _reject("real_llm_activation_schema_mismatch", "activation schema mismatch", sharded_result)
            if replay_runtime:
                wanted = expected_activations[index]
                wanted_request_id = str(wanted.get("request_id"))
            else:
                request_rows = list(expected_spec.get("requests") or [])
                wanted_request_id = str((request_rows[index] if index < len(request_rows) else {}).get("request_id") or "")
            if str(actual.get("request_id")) != wanted_request_id:
                return _reject("real_llm_activation_request_mismatch", "activation request_id mismatch", sharded_result)
            if str(actual.get("session_id")) != str(expected_spec.get("session_id", "")):
                return _reject("real_llm_activation_session_mismatch", "activation session_id mismatch", sharded_result)
            if str(actual.get("artifact_hash")) != str(expected_spec.get("artifact_hash", "")):
                return _reject("real_llm_activation_artifact_hash_mismatch", "activation artifact_hash mismatch", sharded_result)
            recomputed_hash = _activation_hash(actual)
            if str(actual.get("activation_hash")) != recomputed_hash:
                return _reject("real_llm_activation_hash_invalid", "activation hash does not match payload", sharded_result)
            if replay_runtime and str(actual.get("activation_hash")) != str(wanted.get("activation_hash")):
                return _reject(
                    "real_llm_activation_mismatch",
                    f"activation {index} does not match replayed stage 0 output",
                    sharded_result,
                )
        activation_bytes = len(_json_payload(observed).encode("utf-8"))
        return {
            "accepted": True,
            "code": "ok",
            "reason": "accepted",
            "workload_type": WORKLOAD_TYPE,
            "schema_version": REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
            "session_id": expected_spec.get("session_id"),
            "stage_id": 0,
            "stage_count": 2,
            "model_id": expected_spec.get("model_id"),
            "backend": expected_spec.get("backend"),
            "partition_mode": str(sharded_result.get("partition_mode") or expected_spec.get("partition_mode") or PARTITION_MODE_FULL),
            "max_new_tokens": max_new_tokens,
            "generation_step": generation_step,
            "stage_layer_range": list(sharded_result.get("stage_layer_range") or []),
            "stage_parameter_count": int(sharded_result.get("stage_parameter_count", 0)),
            "full_model_parameter_count": int(sharded_result.get("full_model_parameter_count", 0)),
            "stage_parameter_fraction": sharded_result.get("stage_parameter_fraction"),
            "device_parameter_count": int(sharded_result.get("device_parameter_count", 0)),
            "partition_parameter_split_valid": bool(sharded_result.get("partition_parameter_split_valid", False)),
            "stage_local_partition_ready": bool(sharded_result.get("stage_local_partition_ready", False)),
            "stage0_partition_loaded": bool(sharded_result.get("stage0_partition_loaded", False)),
            "stage_gpu_memory_reduced": bool(sharded_result.get("stage_gpu_memory_reduced", False)),
            "stage_cpu_partition_ready": bool(sharded_result.get("stage_cpu_partition_ready", False)),
            "artifact_schema": expected_spec.get("artifact_schema"),
            "artifact_hash": expected_spec.get("artifact_hash"),
            "split_index": int(expected_spec.get("split_index", 1)),
            "request_count": len(observed),
            "activation_count": len(observed),
            "activation_bytes": activation_bytes,
            "activation_hashes": [str(row.get("activation_hash") or "") for row in observed],
            "activation_transport_ready": bool(observed),
            "real_llm_artifact_ready": True,
            "runtime_replay_performed": bool(replay_runtime),
            "remote_runtime_validation": not bool(replay_runtime),
            "sharded_inference_result": sharded_result,
            "activation_results": observed,
            "elapsed_ms": sharded_result.get("elapsed_ms"),
        }

    observed_results = list(sharded_result.get("inference_results") or [])
    if not observed_results and isinstance(sharded_result.get("inference_result"), dict):
        observed_results = [dict(sharded_result["inference_result"])]
    expected_results = list(expected.get("inference_results") or [])
    expected_result_count = len(expected_results) if replay_runtime else len(list(expected_spec.get("activation_results") or []))
    if len(observed_results) != expected_result_count:
        return _reject(
            "real_llm_result_count_mismatch",
            "stage 1 inference result count does not match activations",
            sharded_result,
        )
    expected_activations_by_request = {
        str(row.get("request_id") or ""): dict(row)
        for row in list(expected_spec.get("activation_results") or [])
        if isinstance(row, dict)
    }
    for index, actual in enumerate(observed_results):
        if not isinstance(actual, dict):
            return _reject("real_llm_result_invalid", "inference result entry is not an object", sharded_result)
        if replay_runtime:
            wanted = expected_results[index]
            wanted_request_id = str(wanted.get("request_id"))
        else:
            activation_rows = list(expected_spec.get("activation_results") or [])
            wanted_request_id = str((activation_rows[index] if index < len(activation_rows) else {}).get("request_id") or "")
        request_id = str(actual.get("request_id"))
        if request_id != wanted_request_id:
            return _reject("real_llm_result_request_mismatch", "inference result request_id mismatch", sharded_result)
        if str(actual.get("model_id")) != str(expected_spec.get("model_id")):
            return _reject("real_llm_result_model_mismatch", "inference result model_id mismatch", sharded_result)
        if str(actual.get("artifact_hash")) != str(expected_spec.get("artifact_hash")):
            return _reject("real_llm_result_artifact_hash_mismatch", "inference result artifact_hash mismatch", sharded_result)
        expected_activation = expected_activations_by_request.get(request_id, {})
        if expected_activation and str(actual.get("activation_hash")) != str(expected_activation.get("activation_hash")):
            return _reject("real_llm_result_activation_hash_mismatch", "inference result activation_hash mismatch", sharded_result)
        recomputed_hash = _output_hash(actual)
        if str(actual.get("output_hash")) != recomputed_hash:
            return _reject("real_llm_output_hash_invalid", "output hash does not match payload", sharded_result)
        if replay_runtime and str(actual.get("output_hash")) != str(wanted.get("output_hash")):
            return _reject(
                "real_llm_output_mismatch",
                f"stage 1 output {index} does not match replayed tiny HF runtime",
                sharded_result,
            )
    baseline_match = bool(observed_results) and all(bool(row.get("baseline_match")) for row in observed_results)
    first_result = observed_results[0] if observed_results else {}
    generated_token_ids = [int(value) for value in list(first_result.get("generated_token_ids") or [])]
    generated_text = str(first_result.get("generated_text") or "")
    return {
        "accepted": bool(baseline_match),
        "code": "ok" if baseline_match else "real_llm_baseline_mismatch",
        "reason": "accepted" if baseline_match else "stage 1 output does not match single-runtime baseline",
        "workload_type": WORKLOAD_TYPE,
        "schema_version": REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
        "session_id": expected_spec.get("session_id"),
        "stage_id": 1,
        "stage_count": 2,
        "model_id": expected_spec.get("model_id"),
        "backend": expected_spec.get("backend"),
        "partition_mode": str(sharded_result.get("partition_mode") or expected_spec.get("partition_mode") or PARTITION_MODE_FULL),
        "max_new_tokens": max_new_tokens,
        "generation_step": generation_step,
        "stage_layer_range": list(sharded_result.get("stage_layer_range") or []),
        "stage_parameter_count": int(sharded_result.get("stage_parameter_count", 0)),
        "full_model_parameter_count": int(sharded_result.get("full_model_parameter_count", 0)),
        "stage_parameter_fraction": sharded_result.get("stage_parameter_fraction"),
        "device_parameter_count": int(sharded_result.get("device_parameter_count", 0)),
        "partition_parameter_split_valid": bool(sharded_result.get("partition_parameter_split_valid", False)),
        "stage_local_partition_ready": bool(sharded_result.get("stage_local_partition_ready", False)),
        "stage1_partition_loaded": bool(sharded_result.get("stage1_partition_loaded", False)),
        "stage_gpu_memory_reduced": bool(sharded_result.get("stage_gpu_memory_reduced", False)),
        "stage_cpu_partition_ready": bool(sharded_result.get("stage_cpu_partition_ready", False)),
        "baseline_device": str(sharded_result.get("baseline_device") or ""),
        "artifact_schema": expected_spec.get("artifact_schema"),
        "artifact_hash": expected_spec.get("artifact_hash"),
        "split_index": int(expected_spec.get("split_index", 1)),
        "request_count": len(observed_results),
        "activation_count": int(sharded_result.get("activation_count", len(expected_spec.get("activation_results") or []))),
        "activation_bytes": int(sharded_result.get("activation_bytes", 0)),
        "activation_hashes": list(sharded_result.get("activation_hashes") or []),
        "activation_transport_ready": bool(sharded_result.get("activation_transport_ready", False)),
        "baseline_match": baseline_match,
        "decoded_tokens_match": baseline_match,
        "generated_token_ids": generated_token_ids,
        "generated_token_count": len(generated_token_ids),
        "generated_text": generated_text,
        "generated_text_hash": str(first_result.get("generated_text_hash") or _generated_text_hash(generated_text)),
        "request_trace": [_safe_trace(row) for row in observed_results],
        "request_trace_count": len(observed_results),
        "request_trace_truncated": False,
        "real_llm_artifact_ready": True,
        "runtime_replay_performed": bool(replay_runtime),
        "remote_runtime_validation": not bool(replay_runtime),
        "sharded_inference_result": sharded_result,
        "inference_result": observed_results[0] if observed_results else {},
        "inference_results": observed_results,
        "elapsed_ms": sharded_result.get("elapsed_ms"),
    }
