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
from typing import Any


REAL_LLM_ARTIFACT_SCHEMA_VERSION = "real_llm_artifact_v1"
REAL_LLM_SHARDED_INFERENCE_SCHEMA_VERSION = "real_llm_sharded_infer_v1"
REAL_LLM_ACTIVATION_SCHEMA_VERSION = "real_llm_activation_v1"
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


def _stage_modules(model: Any, *, stage_id: int, split_index: int) -> tuple[list[Any], tuple[int, int], list[str]]:
    transformer, blocks = _gpt2_parts(model)
    split = max(1, min(int(split_index), len(blocks) - 1))
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


def _move_stage_modules(model: Any, *, stage_id: int, split_index: int, device: Any) -> None:
    modules, _, _ = _stage_modules(model, stage_id=stage_id, split_index=split_index)
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
    baseline_device: str = "",
) -> dict[str, Any]:
    transformer, blocks = _gpt2_parts(model)
    split = max(1, min(int(split_index), len(blocks) - 1))
    mode = normalize_partition_mode(partition_mode)
    full_count = _parameter_count(model)
    modules, layer_range, module_kinds = _stage_modules(model, stage_id=stage_id, split_index=split)
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
        "artifact": dict(artifact),
        "model_id": artifact.get("model_id") or DEFAULT_MODEL_ID,
        "backend": artifact.get("backend") or "hf_transformers_cpu",
        "partition_mode": normalize_partition_mode(artifact.get("partition_mode") or PARTITION_MODE_FULL),
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
) -> dict[str, Any]:
    import torch  # type: ignore

    transformer, blocks = _gpt2_parts(model)
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
        if cached and int(cached.get("input_token_count") or 0) == int(input_ids.shape[1]) - 1:
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
) -> dict[str, Any]:
    import torch  # type: ignore

    transformer, blocks = _gpt2_parts(model)
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
        if (
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
        if not cache_ready:
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
        hidden = transformer.ln_f(hidden)
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
        _move_stage_modules(model, stage_id=stage_id, split_index=split_index, device=device)
    partition = _partition_summary(
        model,
        stage_id=stage_id,
        split_index=split_index,
        partition_mode=partition_mode,
        device=device,
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
