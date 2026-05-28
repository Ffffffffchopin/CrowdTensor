"""Dependency-free micro Transformer language-model workload.

This is intentionally tiny and slow-but-deterministic. It gives CrowdTensorD a
real next-token training contract without pulling in PyTorch, NumPy, or GPU
runtime dependencies.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Iterable


MICRO_TRANSFORMER_SCHEMA_VERSION = "micro_transformer_lm_v1"
MICRO_LLM_SHARDED_INFERENCE_SCHEMA_VERSION = "micro_llm_sharded_infer_v1"
MICRO_LLM_ACTIVATION_SCHEMA_VERSION = "micro_llm_activation_v1"
WORKLOAD_TYPE = "micro_transformer_lm"
MICRO_LLM_SHARDED_WORKLOAD_TYPE = "micro_llm_sharded_infer"
CORPUS = "tensor swarm tensor route tensor swarm learn "
VOCAB = sorted(set(CORPUS))
TOKEN_IDS = [VOCAB.index(char) for char in CORPUS]
CONTEXT_LENGTH = 3
EMBEDDING_DIM = 3
DEFAULT_INNER_LR = 0.08
DEFAULT_LOCAL_DELTA_SCALE = 0.1
DEFAULT_OUTER_LR = 0.7
DEFAULT_OUTER_MOMENTUM = 0.75
DEFAULT_MAX_DELTA_NORM = 3.0
DEFAULT_MAX_LOSS_DELTA = 0.75
FINITE_DIFF_EPS = 1e-4
MICRO_LLM_TRACE_LIMIT = 8


def _stable_offset(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def json_repr(payload: dict) -> str:
    return repr(_round_floats(payload))


def _artifact_hash(values: Iterable[float], config: dict) -> str:
    payload = {
        "schema_version": MICRO_TRANSFORMER_SCHEMA_VERSION,
        "config": normalize_config(config),
        "weights": [round(float(value), 12) for value in values],
    }
    return "sha256:" + hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def micro_transformer_artifact_hash(model: dict | None) -> str:
    current = normalize_micro_transformer_model(model)
    return str(current.get("artifact_hash") or _artifact_hash(current["weights"], current["config"]))


def _activation_hash(payload: dict) -> str:
    public = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "session_id",
            "request_id",
            "model_version",
            "artifact_hash",
            "prompt_token_ids",
            "decode_index",
            "hidden_state",
        )
    }
    return "sha256:" + hashlib.sha256(json_repr(public).encode("utf-8")).hexdigest()


def default_config() -> dict:
    return {
        "vocab": list(VOCAB),
        "vocab_size": len(VOCAB),
        "context_length": CONTEXT_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
    }


def parameter_count(config: dict | None = None) -> int:
    cfg = default_config() if config is None else normalize_config(config)
    vocab_size = int(cfg["vocab_size"])
    dim = int(cfg["embedding_dim"])
    return (
        vocab_size * dim
        + dim * dim * 4
        + dim * vocab_size
        + vocab_size
    )


def _initial_weight(index: int) -> float:
    return 0.05 * math.sin((index + 1) * 1.61803398875)


def default_micro_transformer_model() -> dict:
    config = default_config()
    weights = [_initial_weight(index) for index in range(parameter_count(config))]
    return {
        "schema_version": MICRO_TRANSFORMER_SCHEMA_VERSION,
        "version": 0,
        "config": config,
        "weights": weights,
        "outer_lr": DEFAULT_OUTER_LR,
        "outer_momentum": DEFAULT_OUTER_MOMENTUM,
        "outer_velocity": [0.0 for _ in weights],
        "optimizer_step": 0,
        "last_loss": micro_transformer_loss_for(weights, config, TOKEN_IDS),
    }


def normalize_config(config: dict | None) -> dict:
    source = dict(config or {})
    vocab = list(source.get("vocab") or VOCAB)
    if not vocab:
        vocab = list(VOCAB)
    vocab_size = int(source.get("vocab_size", len(vocab)))
    if vocab_size != len(vocab):
        vocab_size = len(vocab)
    context_length = max(1, int(source.get("context_length", CONTEXT_LENGTH)))
    embedding_dim = max(1, int(source.get("embedding_dim", EMBEDDING_DIM)))
    return {
        "vocab": vocab,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "embedding_dim": embedding_dim,
    }


def normalize_micro_transformer_model(model: dict | None) -> dict:
    source = dict(model or {})
    if "micro_transformer" in source and isinstance(source.get("micro_transformer"), dict):
        source = dict(source["micro_transformer"])
    config = normalize_config(source.get("config"))
    expected = parameter_count(config)
    try:
        weights = [float(value) for value in source.get("weights", [])]
    except (TypeError, ValueError):
        weights = []
    if len(weights) != expected:
        weights = [_initial_weight(index) for index in range(expected)]
    try:
        velocity = [float(value) for value in source.get("outer_velocity", [])]
    except (TypeError, ValueError):
        velocity = []
    if len(velocity) != expected:
        velocity = [0.0 for _ in weights]
    token_ids = _token_ids_from_spec({"token_ids": TOKEN_IDS}, config)
    normalized = {
        "schema_version": source.get("schema_version", MICRO_TRANSFORMER_SCHEMA_VERSION),
        "version": int(source.get("version", 0)),
        "config": config,
        "weights": weights,
        "outer_lr": float(source.get("outer_lr", DEFAULT_OUTER_LR)),
        "outer_momentum": float(source.get("outer_momentum", DEFAULT_OUTER_MOMENTUM)),
        "outer_velocity": velocity,
        "optimizer_step": int(source.get("optimizer_step", source.get("version", 0))),
        "last_loss": float(source.get("last_loss", micro_transformer_loss_for(weights, config, token_ids))),
    }
    for key in (
        "artifact_schema",
        "artifact_id",
        "artifact_version",
        "artifact_hash",
        "artifact_source",
        "tokenizer_schema",
    ):
        if key in source:
            normalized[key] = source[key]
    return normalized


def micro_transformer_training_spec_for(task_id: str, miner_id: str, model: dict) -> dict:
    current = normalize_micro_transformer_model(model)
    token_ids = list(TOKEN_IDS)
    return {
        "type": WORKLOAD_TYPE,
        "schema_version": MICRO_TRANSFORMER_SCHEMA_VERSION,
        "config": dict(current["config"]),
        "weights": list(current["weights"]),
        "model_version": int(current["version"]),
        "token_ids": token_ids,
        "inner_lr": DEFAULT_INNER_LR,
        "local_delta_scale": DEFAULT_LOCAL_DELTA_SCALE,
        "finite_diff_eps": FINITE_DIFF_EPS,
        "batch_size": _example_count(token_ids, current["config"]),
        "sample_offset": _stable_offset(task_id, miner_id, current["version"]) % _example_count(token_ids, current["config"]),
    }


def micro_llm_sharded_inference_spec_for(
    task_id: str,
    miner_id: str,
    model: dict,
    *,
    request_count: int = 1,
    decode_steps: int = 1,
    session_id: str = "",
    stage_id: int = 0,
    parent_task_id: str = "",
    requests: list[dict] | None = None,
    activation_results: list[dict] | None = None,
) -> dict:
    stage = int(stage_id)
    if stage not in {0, 1}:
        raise ValueError("micro LLM sharded inference stage_id must be 0 or 1")
    current = normalize_micro_transformer_model(model)
    config = dict(current["config"])
    token_ids = list(TOKEN_IDS)
    count = max(1, min(int(request_count), _example_count(token_ids, config)))
    steps = max(1, min(int(decode_steps), 4))
    example_count = _example_count(token_ids, config)
    sample_offset = _stable_offset(task_id, miner_id, current["version"], "micro-llm-sharded") % example_count
    request_rows: list[dict] = []
    if requests:
        for index, row in enumerate(list(requests)[:count]):
            prompt = [int(value) % int(config["vocab_size"]) for value in row.get("prompt_token_ids", [])]
            if len(prompt) != int(config["context_length"]):
                context, target = _example_at(token_ids, config, sample_offset + index)
                prompt = list(context)
            else:
                target = int(row.get("target_token_id", 0)) % int(config["vocab_size"])
            request_rows.append({
                "request_id": str(row.get("request_id") or f"req-{index + 1}"),
                "prompt_token_ids": prompt,
                "target_token_id": int(target),
                "sample_offset": int(row.get("sample_offset", sample_offset + index)),
            })
    while len(request_rows) < count:
        index = len(request_rows)
        context, target = _example_at(token_ids, config, sample_offset + index)
        request_rows.append({
            "request_id": f"req-{index + 1}",
            "prompt_token_ids": list(context),
            "target_token_id": int(target),
            "sample_offset": sample_offset + index,
        })
    spec = {
        "type": MICRO_LLM_SHARDED_WORKLOAD_TYPE,
        "schema_version": MICRO_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
        "base_model_schema_version": MICRO_TRANSFORMER_SCHEMA_VERSION,
        "session_id": str(session_id or task_id),
        "stage_id": stage,
        "stage_count": 2,
        "parent_task_id": str(parent_task_id or ""),
        "shard_role": "embedding_attention_trunk" if stage == 0 else "lm_head_decoder",
        "model_version": int(current["version"]),
        "artifact_hash": micro_transformer_artifact_hash(current),
        "config": config,
        "weights": list(current["weights"]),
        "requests": request_rows,
        "request_count": len(request_rows),
        "decode_steps": steps,
        "top_k": 3,
        "sample_offset": sample_offset,
    }
    for key in ("artifact_schema", "artifact_id", "artifact_version", "tokenizer_schema"):
        if key in current:
            spec[key] = current[key]
    if stage == 1:
        spec["activation_results"] = list(activation_results or [])
        spec["activation_count"] = len(spec["activation_results"])
    return spec


def micro_transformer_version(model: dict) -> int:
    return int(normalize_micro_transformer_model(model).get("version", 0))


def _offsets(config: dict) -> dict[str, int]:
    vocab_size = int(config["vocab_size"])
    dim = int(config["embedding_dim"])
    embedding_end = vocab_size * dim
    wq_end = embedding_end + dim * dim
    wk_end = wq_end + dim * dim
    wv_end = wk_end + dim * dim
    wo_end = wv_end + dim * dim
    head_end = wo_end + dim * vocab_size
    return {
        "embedding": 0,
        "wq": embedding_end,
        "wk": wq_end,
        "wv": wk_end,
        "wo": wv_end,
        "head": wo_end,
        "bias": head_end,
        "end": head_end + vocab_size,
    }


def _matrix_vector(vector: list[float], matrix: list[float], rows: int, cols: int) -> list[float]:
    return [
        sum(vector[row] * matrix[row * cols + col] for row in range(rows))
        for col in range(cols)
    ]


def _add_outer_product(
    target: list[float],
    start: int,
    vector: list[float],
    grad: list[float],
    rows: int,
    cols: int,
) -> None:
    for row in range(rows):
        base = start + row * cols
        for col in range(cols):
            target[base + col] += vector[row] * grad[col]


def _matvec_backward_vector(matrix: list[float], grad: list[float], rows: int, cols: int) -> list[float]:
    return [
        sum(matrix[row * cols + col] * grad[col] for col in range(cols))
        for row in range(rows)
    ]


def _add_into(target: list[float], source: list[float]) -> None:
    for index, value in enumerate(source):
        target[index] += value


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(max(-60.0, min(60.0, value - peak))) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(values) for _ in values]
    return [value / total for value in exps]


def _token_text(config: dict, token_id: int) -> str:
    vocab = list(normalize_config(config)["vocab"])
    if not vocab:
        return ""
    return str(vocab[int(token_id) % len(vocab)])


def _decode_token_ids(config: dict, token_ids: list[int]) -> str:
    return "".join(_token_text(config, int(token_id)) for token_id in token_ids)


def _safe_top_k_trace(config: dict, rows: list[dict]) -> list[dict]:
    trace = []
    for row in rows:
        token_id = int(row.get("token_id", 0))
        trace.append({
            "token_id": token_id,
            "token": _token_text(config, token_id),
            "probability": float(row.get("probability", 0.0)),
        })
    return trace


def _cross_entropy(logits: list[float], target: int) -> float:
    peak = max(logits)
    exps = [math.exp(max(-60.0, min(60.0, value - peak))) for value in logits]
    total = sum(exps)
    probability = exps[target] / total
    return -math.log(max(probability, 1e-12))


def _token_ids_from_spec(spec: dict, config: dict) -> list[int]:
    vocab_size = int(config["vocab_size"])
    values = spec.get("token_ids", TOKEN_IDS)
    token_ids = [int(value) % vocab_size for value in values]
    if len(token_ids) <= int(config["context_length"]):
        token_ids = [int(value) % vocab_size for value in TOKEN_IDS]
    return token_ids


def _example_count(token_ids: list[int], config: dict) -> int:
    return max(1, len(token_ids) - int(config["context_length"]))


def _example_at(token_ids: list[int], config: dict, sample_index: int) -> tuple[list[int], int]:
    context_length = int(config["context_length"])
    count = _example_count(token_ids, config)
    start = int(sample_index) % count
    return token_ids[start:start + context_length], token_ids[start + context_length]


def _slice(weights: list[float], start: int, length: int) -> list[float]:
    return weights[start:start + length]


def logits_for(weights: Iterable[float], config: dict, context: list[int]) -> list[float]:
    return _forward_cache([float(value) for value in weights], normalize_config(config), context)["logits"]


def hidden_state_for(weights: Iterable[float], config: dict, context: list[int]) -> list[float]:
    return list(_forward_cache([float(value) for value in weights], normalize_config(config), context)["outputs"][-1])


def logits_from_hidden(weights: Iterable[float], config: dict, hidden_state: list[float]) -> list[float]:
    values = [float(value) for value in weights]
    cfg = normalize_config(config)
    offsets = _offsets(cfg)
    vocab_size = int(cfg["vocab_size"])
    dim = int(cfg["embedding_dim"])
    if len(hidden_state) != dim:
        raise ValueError("micro LLM hidden_state length does not match embedding_dim")
    head = _slice(values, offsets["head"], dim * vocab_size)
    bias = _slice(values, offsets["bias"], vocab_size)
    return [
        bias[token] + sum(float(hidden_state[row]) * head[row * vocab_size + token] for row in range(dim))
        for token in range(vocab_size)
    ]


def _rank_logits(config: dict, logits: list[float], *, top_k: int) -> list[dict]:
    probabilities = _softmax(logits)
    ranked = sorted(enumerate(probabilities), key=lambda item: (-float(item[1]), int(item[0])))[:top_k]
    return [
        {
            "token_id": int(token_id),
            "token": _token_text(config, int(token_id)),
            "probability": float(probability),
        }
        for token_id, probability in ranked
    ]


def _forward_cache(weights: list[float], config: dict, context: list[int]) -> dict:
    cfg = normalize_config(config)
    offsets = _offsets(cfg)
    vocab_size = int(cfg["vocab_size"])
    dim = int(cfg["embedding_dim"])
    embeddings = _slice(weights, offsets["embedding"], vocab_size * dim)
    wq = _slice(weights, offsets["wq"], dim * dim)
    wk = _slice(weights, offsets["wk"], dim * dim)
    wv = _slice(weights, offsets["wv"], dim * dim)
    wo = _slice(weights, offsets["wo"], dim * dim)
    head = _slice(weights, offsets["head"], dim * vocab_size)
    bias = _slice(weights, offsets["bias"], vocab_size)

    token_ids = [int(token) % vocab_size for token in context]
    states = [_slice(embeddings, token_id * dim, dim) for token_id in token_ids]
    queries = [_matrix_vector(state, wq, dim, dim) for state in states]
    keys = [_matrix_vector(state, wk, dim, dim) for state in states]
    values_v = [_matrix_vector(state, wv, dim, dim) for state in states]
    scale = 1.0 / math.sqrt(dim)
    scores: list[list[float]] = []
    probs: list[list[float]] = []
    attended_values: list[list[float]] = []
    outputs: list[list[float]] = []
    for index, query in enumerate(queries):
        row_scores = [
            sum(query[col] * keys[past][col] for col in range(dim)) * scale
            for past in range(index + 1)
        ]
        row_probs = _softmax(row_scores)
        attended = [
            sum(row_probs[past] * values_v[past][col] for past in range(index + 1))
            for col in range(dim)
        ]
        scores.append(row_scores)
        probs.append(row_probs)
        attended_values.append(attended)
        outputs.append(_matrix_vector(attended, wo, dim, dim))

    final = outputs[-1]
    logits = [
        bias[token] + sum(final[row] * head[row * vocab_size + token] for row in range(dim))
        for token in range(vocab_size)
    ]
    return {
        "config": cfg,
        "offsets": offsets,
        "embedding": embeddings,
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "head": head,
        "bias": bias,
        "tokens": token_ids,
        "states": states,
        "queries": queries,
        "keys": keys,
        "values": values_v,
        "scores": scores,
        "probs": probs,
        "attended": attended_values,
        "outputs": outputs,
        "logits": logits,
    }


def sample_loss_for(weights: Iterable[float], config: dict, token_ids: list[int], sample_index: int) -> float:
    context, target = _example_at(token_ids, normalize_config(config), sample_index)
    return _cross_entropy(logits_for(weights, config, context), target)


def micro_transformer_loss_for(weights: Iterable[float], config: dict, token_ids: list[int] | None = None) -> float:
    cfg = normalize_config(config)
    ids = _token_ids_from_spec({"token_ids": token_ids or TOKEN_IDS}, cfg)
    count = _example_count(ids, cfg)
    return sum(sample_loss_for(weights, cfg, ids, index) for index in range(count)) / count


def micro_transformer_loss(model: dict) -> float:
    current = normalize_micro_transformer_model(model)
    return micro_transformer_loss_for(current["weights"], current["config"], TOKEN_IDS)


def finite_difference_gradient(
    weights: list[float],
    config: dict,
    token_ids: list[int],
    sample_index: int,
    *,
    eps: float,
) -> list[float]:
    gradient: list[float] = []
    for index, original in enumerate(weights):
        weights[index] = original + eps
        plus = sample_loss_for(weights, config, token_ids, sample_index)
        weights[index] = original - eps
        minus = sample_loss_for(weights, config, token_ids, sample_index)
        weights[index] = original
        gradient.append((plus - minus) / (2.0 * eps))
    return gradient


def analytic_gradient_for_sample(
    weights: list[float],
    config: dict,
    token_ids: list[int],
    sample_index: int,
) -> list[float]:
    cfg = normalize_config(config)
    context, target = _example_at(token_ids, cfg, sample_index)
    cache = _forward_cache(weights, cfg, context)
    offsets = cache["offsets"]
    vocab_size = int(cfg["vocab_size"])
    dim = int(cfg["embedding_dim"])
    grad = [0.0 for _ in weights]

    probabilities = _softmax(cache["logits"])
    d_logits = list(probabilities)
    d_logits[target] -= 1.0

    d_final = [0.0 for _ in range(dim)]
    for token, value in enumerate(d_logits):
        grad[offsets["bias"] + token] += value
        for row in range(dim):
            grad[offsets["head"] + row * vocab_size + token] += cache["outputs"][-1][row] * value
            d_final[row] += cache["head"][row * vocab_size + token] * value

    d_outputs = [[0.0 for _ in range(dim)] for _ in cache["outputs"]]
    d_outputs[-1] = d_final
    d_attended = [[0.0 for _ in range(dim)] for _ in cache["attended"]]
    d_queries = [[0.0 for _ in range(dim)] for _ in cache["queries"]]
    d_keys = [[0.0 for _ in range(dim)] for _ in cache["keys"]]
    d_values = [[0.0 for _ in range(dim)] for _ in cache["values"]]

    for index in reversed(range(len(cache["outputs"]))):
        d_output = d_outputs[index]
        _add_outer_product(grad, offsets["wo"], cache["attended"][index], d_output, dim, dim)
        d_attended[index] = _matvec_backward_vector(cache["wo"], d_output, dim, dim)

        row_probs = cache["probs"][index]
        d_probs = [0.0 for _ in row_probs]
        for past in range(index + 1):
            for col in range(dim):
                d_probs[past] += d_attended[index][col] * cache["values"][past][col]
                d_values[past][col] += row_probs[past] * d_attended[index][col]

        dot = sum(d_probs[past] * row_probs[past] for past in range(len(row_probs)))
        d_scores = [
            row_probs[past] * (d_probs[past] - dot)
            for past in range(len(row_probs))
        ]
        scale = 1.0 / math.sqrt(dim)
        for past, d_score in enumerate(d_scores):
            scaled = d_score * scale
            for col in range(dim):
                d_queries[index][col] += scaled * cache["keys"][past][col]
                d_keys[past][col] += scaled * cache["queries"][index][col]

    d_states = [[0.0 for _ in range(dim)] for _ in cache["states"]]
    for index, state in enumerate(cache["states"]):
        _add_outer_product(grad, offsets["wq"], state, d_queries[index], dim, dim)
        _add_outer_product(grad, offsets["wk"], state, d_keys[index], dim, dim)
        _add_outer_product(grad, offsets["wv"], state, d_values[index], dim, dim)
        _add_into(d_states[index], _matvec_backward_vector(cache["wq"], d_queries[index], dim, dim))
        _add_into(d_states[index], _matvec_backward_vector(cache["wk"], d_keys[index], dim, dim))
        _add_into(d_states[index], _matvec_backward_vector(cache["wv"], d_values[index], dim, dim))

    for token_id, d_state in zip(cache["tokens"], d_states):
        base = offsets["embedding"] + token_id * dim
        for col, value in enumerate(d_state):
            grad[base + col] += value

    return grad


def analytic_gradient_for_batch(
    weights: list[float],
    config: dict,
    token_ids: list[int],
    *,
    sample_offset: int = 0,
    batch_size: int | None = None,
) -> list[float]:
    cfg = normalize_config(config)
    count = _example_count(token_ids, cfg)
    size = count if batch_size is None else max(1, int(batch_size))
    gradient = [0.0 for _ in weights]
    for batch_index in range(size):
        sample_index = int(sample_offset) + batch_index
        sample_gradient = analytic_gradient_for_sample(weights, cfg, token_ids, sample_index)
        for index, value in enumerate(sample_gradient):
            gradient[index] += value / size
    return gradient


def run_micro_transformer_inner_loop(
    workload_spec: dict,
    *,
    inner_steps: int,
    compute_seconds: float = 0.0,
) -> dict:
    config = normalize_config(workload_spec.get("config"))
    initial = [float(value) for value in workload_spec["weights"]]
    if len(initial) != parameter_count(config):
        raise ValueError("micro_transformer_lm workload_spec has an invalid weight length")
    local = list(initial)
    token_ids = _token_ids_from_spec(workload_spec, config)
    steps = max(1, int(inner_steps))
    inner_lr = float(workload_spec.get("inner_lr", DEFAULT_INNER_LR))
    delta_scale = float(workload_spec.get("local_delta_scale", DEFAULT_LOCAL_DELTA_SCALE))
    eps = float(workload_spec.get("finite_diff_eps", FINITE_DIFF_EPS))
    sample_offset = int(workload_spec.get("sample_offset", 0))
    batch_size = int(workload_spec.get("batch_size", _example_count(token_ids, config)))

    started_at = time.monotonic()
    deadline = started_at + max(0.0, compute_seconds)
    step = 0
    loss_start = micro_transformer_loss_for(local, config, token_ids)
    while step < steps or (compute_seconds > 0 and time.monotonic() < deadline):
        gradient = analytic_gradient_for_batch(
            local,
            config,
            token_ids,
            sample_offset=sample_offset + step * batch_size,
            batch_size=batch_size,
        )
        for index, value in enumerate(gradient):
            local[index] -= inner_lr * value
        step += 1
        if compute_seconds > 0 and step % 5 == 0:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.02, remaining))

    local_delta = [
        (local_value - initial_value) * delta_scale
        for local_value, initial_value in zip(local, initial)
    ]
    return {
        "schema_version": MICRO_TRANSFORMER_SCHEMA_VERSION,
        "local_delta": local_delta,
        "lm_loss_start": loss_start,
        "lm_loss_end": micro_transformer_loss_for(local, config, token_ids),
        "tokens_seen": step * int(config["context_length"]),
        "samples_seen": step,
        "inner_steps": steps,
        "inner_lr": inner_lr,
        "local_delta_scale": delta_scale,
        "finite_diff_eps": eps,
        "gradient_mode": "analytic",
        "batch_size": batch_size,
        "sample_offset": sample_offset,
        "context_length": int(config["context_length"]),
        "vocab_size": int(config["vocab_size"]),
    }


def _micro_llm_requests_from_spec(spec: dict, config: dict) -> list[dict]:
    rows = spec.get("requests")
    if rows is None:
        rows = [{
            "request_id": "req-1",
            "prompt_token_ids": spec.get("prompt_token_ids", []),
            "target_token_id": spec.get("target_token_id", 0),
            "sample_offset": spec.get("sample_offset", 0),
        }]
    requests = []
    for index, row in enumerate(list(rows)):
        if not isinstance(row, dict):
            raise ValueError("micro LLM sharded requests must be objects")
        prompt = [int(value) % int(config["vocab_size"]) for value in row.get("prompt_token_ids", [])]
        if len(prompt) != int(config["context_length"]):
            raise ValueError("micro LLM prompt length does not match context_length")
        requests.append({
            "request_id": str(row.get("request_id") or f"req-{index + 1}"),
            "prompt_token_ids": prompt,
            "target_token_id": int(row.get("target_token_id", 0)) % int(config["vocab_size"]),
            "sample_offset": int(row.get("sample_offset", index)),
        })
    if not requests:
        raise ValueError("micro LLM sharded inference requires at least one request")
    return requests


def _micro_llm_decode_baseline_for_request(
    *,
    request: dict,
    weights: list[float],
    config: dict,
    decode_steps: int,
    top_k: int,
) -> dict:
    context = list(request["prompt_token_ids"])
    generated: list[int] = []
    steps = []
    for decode_index in range(max(1, min(int(decode_steps), 4))):
        step_context = list(context[-int(config["context_length"]):])
        logits = logits_for(weights, config, step_context)
        ranked = _rank_logits(config, logits, top_k=top_k)
        predicted = int(ranked[0]["token_id"])
        steps.append({
            "decode_index": decode_index,
            "context_token_ids": step_context,
            "predicted_token_id": predicted,
            "predicted_token": _token_text(config, predicted),
            "top_k": ranked,
        })
        generated.append(predicted)
        context.append(predicted)
    return {
        "schema_version": MICRO_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
        "request_id": str(request["request_id"]),
        "prompt_token_ids": list(request["prompt_token_ids"]),
        "target_token_id": int(request["target_token_id"]),
        "target_token": _token_text(config, int(request["target_token_id"])),
        "generated_token_ids": generated,
        "generated_text": _decode_token_ids(config, generated),
        "decode_steps": len(generated),
        "steps": steps,
    }


def run_micro_llm_full_inference(workload_spec: dict) -> dict:
    spec = dict(workload_spec or {})
    config = normalize_config(spec.get("config"))
    weights = [float(value) for value in spec.get("weights", [])]
    expected = parameter_count(config)
    if len(weights) != expected:
        raise ValueError(f"micro LLM weights length {len(weights)} does not match expected {expected}")
    requests = _micro_llm_requests_from_spec(spec, config)
    decode_steps = max(1, min(int(spec.get("decode_steps", 1)), 4))
    top_k = max(1, min(int(spec.get("top_k", 3)), int(config["vocab_size"])))
    start = time.monotonic()
    results = [
        _micro_llm_decode_baseline_for_request(
            request=request,
            weights=weights,
            config=config,
            decode_steps=decode_steps,
            top_k=top_k,
        )
        for request in requests
    ]
    elapsed_ms = (time.monotonic() - start) * 1000.0
    elapsed_seconds = max(elapsed_ms / 1000.0, 1e-9)
    return {
        "schema_version": MICRO_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
        "inference_result": results[0],
        "inference_results": results,
        "request_count": len(results),
        "decode_steps": decode_steps,
        "generated_token_count": sum(len(row["generated_token_ids"]) for row in results),
        "elapsed_ms": elapsed_ms,
        "requests_per_second": len(results) / elapsed_seconds,
    }


def run_micro_llm_sharded_inference(workload_spec: dict) -> dict:
    spec = dict(workload_spec or {})
    stage_id = int(spec.get("stage_id", 0))
    if stage_id not in {0, 1}:
        raise ValueError("micro LLM sharded inference stage_id must be 0 or 1")
    config = normalize_config(spec.get("config"))
    weights = [float(value) for value in spec.get("weights", [])]
    expected = parameter_count(config)
    if len(weights) != expected:
        raise ValueError(f"micro LLM sharded weights length {len(weights)} does not match expected {expected}")
    requests = _micro_llm_requests_from_spec(spec, config)
    session_id = str(spec.get("session_id") or "")
    model_version = int(spec.get("model_version", 0))
    artifact_hash = str(spec.get("artifact_hash") or _artifact_hash(weights, config))
    decode_steps = max(1, min(int(spec.get("decode_steps", 1)), 4))
    top_k = max(1, min(int(spec.get("top_k", 3)), int(config["vocab_size"])))
    start = time.monotonic()

    if stage_id == 0:
        activations = []
        for request in requests:
            baseline = _micro_llm_decode_baseline_for_request(
                request=request,
                weights=weights,
                config=config,
                decode_steps=decode_steps,
                top_k=top_k,
            )
            for step in baseline["steps"]:
                hidden_state = hidden_state_for(weights, config, list(step["context_token_ids"]))
                activation = {
                    "schema_version": MICRO_LLM_ACTIVATION_SCHEMA_VERSION,
                    "session_id": session_id,
                    "request_id": str(request["request_id"]),
                    "model_version": model_version,
                    "artifact_hash": artifact_hash,
                    "prompt_token_ids": list(step["context_token_ids"]),
                    "decode_index": int(step["decode_index"]),
                    "hidden_state": hidden_state,
                    "hidden_dim": len(hidden_state),
                }
                activation["activation_hash"] = _activation_hash(activation)
                activations.append(activation)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        activation_bytes = len(json_repr({"activation_results": activations}).encode("utf-8"))
        return {
            "schema_version": MICRO_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
            "stage_id": 0,
            "stage_count": 2,
            "session_id": session_id,
            "activation_result": activations[0],
            "activation_results": activations,
            "activation_count": len(activations),
            "activation_bytes": activation_bytes,
            "activation_hashes": [row["activation_hash"] for row in activations],
            "request_count": len(requests),
            "decode_steps": decode_steps,
            "elapsed_ms": elapsed_ms,
        }

    activations = list(spec.get("activation_results") or [])
    if not activations and isinstance(spec.get("activation_result"), dict):
        activations = [dict(spec["activation_result"])]
    expected_activation_count = len(requests) * decode_steps
    if len(activations) != expected_activation_count:
        raise ValueError("stage-1 micro LLM activation count does not match request_count * decode_steps")
    by_key = {
        (str(row.get("request_id")), int(row.get("decode_index", -1))): row
        for row in activations
        if isinstance(row, dict)
    }
    results = []
    for request in requests:
        generated: list[int] = []
        steps = []
        context = list(request["prompt_token_ids"])
        for decode_index in range(decode_steps):
            key = (str(request["request_id"]), decode_index)
            activation = by_key.get(key)
            if not isinstance(activation, dict):
                raise ValueError("stage-1 micro LLM missing activation for request/decode step")
            if str(activation.get("schema_version")) != MICRO_LLM_ACTIVATION_SCHEMA_VERSION:
                raise ValueError("micro LLM activation schema_version mismatch")
            if str(activation.get("session_id")) != session_id:
                raise ValueError("micro LLM activation session_id mismatch")
            if int(activation.get("model_version", -1)) != model_version:
                raise ValueError("micro LLM activation model_version mismatch")
            if str(activation.get("artifact_hash")) != artifact_hash:
                raise ValueError("micro LLM activation artifact_hash mismatch")
            step_context = list(context[-int(config["context_length"]):])
            if [int(value) for value in activation.get("prompt_token_ids", [])] != step_context:
                raise ValueError("micro LLM activation context mismatch")
            hidden = [float(value) for value in activation.get("hidden_state", [])]
            if str(activation.get("activation_hash")) != _activation_hash({**activation, "hidden_state": hidden}):
                raise ValueError("micro LLM activation_hash mismatch")
            logits = logits_from_hidden(weights, config, hidden)
            ranked = _rank_logits(config, logits, top_k=top_k)
            predicted = int(ranked[0]["token_id"])
            steps.append({
                "decode_index": decode_index,
                "context_token_ids": step_context,
                "predicted_token_id": predicted,
                "predicted_token": _token_text(config, predicted),
                "top_k": ranked,
                "activation_hash": str(activation.get("activation_hash")),
            })
            generated.append(predicted)
            context.append(predicted)
        results.append({
            "schema_version": MICRO_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
            "request_id": str(request["request_id"]),
            "prompt_token_ids": list(request["prompt_token_ids"]),
            "target_token_id": int(request["target_token_id"]),
            "target_token": _token_text(config, int(request["target_token_id"])),
            "generated_token_ids": generated,
            "generated_text": _decode_token_ids(config, generated),
            "decode_steps": len(generated),
            "steps": steps,
        })
    elapsed_ms = (time.monotonic() - start) * 1000.0
    elapsed_seconds = max(elapsed_ms / 1000.0, 1e-9)
    return {
        "schema_version": MICRO_LLM_SHARDED_INFERENCE_SCHEMA_VERSION,
        "stage_id": 1,
        "stage_count": 2,
        "session_id": session_id,
        "inference_result": results[0],
        "inference_results": results,
        "request_count": len(results),
        "decode_steps": decode_steps,
        "generated_token_count": sum(len(row["generated_token_ids"]) for row in results),
        "elapsed_ms": elapsed_ms,
        "requests_per_second": len(results) / elapsed_seconds,
        "activation_count": len(activations),
        "activation_hashes": [str(row.get("activation_hash")) for row in activations],
    }


def _safe_micro_llm_trace(config: dict, result: dict) -> dict:
    prompt = [int(value) for value in result.get("prompt_token_ids", [])]
    generated = [int(value) for value in result.get("generated_token_ids", [])]
    return {
        "request_id": str(result.get("request_id", "")),
        "prompt": _decode_token_ids(config, prompt),
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "generated_text": _decode_token_ids(config, generated),
        "decode_steps": int(result.get("decode_steps", len(generated))),
        "steps": [
            {
                "decode_index": int(step.get("decode_index", 0)),
                "predicted_token_id": int(step.get("predicted_token_id", 0)),
                "predicted_token": _token_text(config, int(step.get("predicted_token_id", 0))),
                "top_k": _safe_top_k_trace(config, list(step.get("top_k") or [])),
                "activation_hash": step.get("activation_hash"),
            }
            for step in list(result.get("steps") or [])[:MICRO_LLM_TRACE_LIMIT]
        ],
    }


def validate_micro_llm_sharded_inference(
    model: dict,
    sharded_result: dict | None,
    *,
    expected_spec: dict | None = None,
) -> dict:
    spec = dict(expected_spec or {})
    if not isinstance(sharded_result, dict):
        return {
            "accepted": False,
            "code": "micro_llm_sharded_result_missing",
            "reason": "micro_llm_sharded_infer requires a sharded_inference_result object",
            "sharded_inference_result": sharded_result,
        }
    if str(sharded_result.get("schema_version")) != MICRO_LLM_SHARDED_INFERENCE_SCHEMA_VERSION:
        return {
            "accepted": False,
            "code": "micro_llm_sharded_schema_mismatch",
            "reason": "sharded result schema_version does not match micro_llm_sharded_infer_v1",
            "sharded_inference_result": sharded_result,
        }
    try:
        stage_id = int(spec.get("stage_id", sharded_result.get("stage_id", -1)))
    except (TypeError, ValueError):
        stage_id = -1
    if stage_id not in {0, 1} or int(sharded_result.get("stage_id", -1)) != stage_id:
        return {
            "accepted": False,
            "code": "micro_llm_sharded_stage_mismatch",
            "reason": "sharded result stage_id does not match claim stage",
            "sharded_inference_result": sharded_result,
        }
    current = normalize_micro_transformer_model(model)
    config = current["config"]
    weights = current["weights"]
    artifact_hash = micro_transformer_artifact_hash(current)
    session_id = str(spec.get("session_id", ""))
    if str(sharded_result.get("session_id", session_id)) != session_id:
        return {
            "accepted": False,
            "code": "micro_llm_sharded_session_mismatch",
            "reason": "sharded result does not match claim-time session",
            "sharded_inference_result": sharded_result,
        }
    expected_requests = list(spec.get("requests") or [])
    decode_steps = max(1, min(int(spec.get("decode_steps", 1)), 4))
    top_k = max(1, min(int(spec.get("top_k", 3)), int(config["vocab_size"])))

    if stage_id == 0:
        activations = list(sharded_result.get("activation_results") or [])
        if not activations and isinstance(sharded_result.get("activation_result"), dict):
            activations = [dict(sharded_result["activation_result"])]
        if len(activations) != len(expected_requests) * decode_steps:
            return {
                "accepted": False,
                "code": "micro_llm_activation_count_mismatch",
                "reason": "stage-0 activation count does not match request_count * decode_steps",
                "activation_count": len(activations),
                "expected_activation_count": len(expected_requests) * decode_steps,
            }
        expected_contexts = {}
        for request in expected_requests:
            baseline = _micro_llm_decode_baseline_for_request(
                request=request,
                weights=weights,
                config=config,
                decode_steps=decode_steps,
                top_k=top_k,
            )
            for step in baseline["steps"]:
                expected_contexts[(str(request["request_id"]), int(step["decode_index"]))] = list(step["context_token_ids"])
        normalized_activations = []
        for index, activation in enumerate(activations):
            if not isinstance(activation, dict):
                return {
                    "accepted": False,
                    "code": "micro_llm_activation_invalid",
                    "reason": "activation rows must be objects",
                    "request_index": index,
                }
            key = (str(activation.get("request_id")), int(activation.get("decode_index", -1)))
            expected_context = expected_contexts.get(key)
            if expected_context is None:
                return {
                    "accepted": False,
                    "code": "micro_llm_activation_step_mismatch",
                    "reason": "activation request_id/decode_index does not match claim-time decode plan",
                    "request_index": index,
                }
            if (
                str(activation.get("schema_version")) != MICRO_LLM_ACTIVATION_SCHEMA_VERSION
                or str(activation.get("session_id")) != session_id
                or int(activation.get("model_version", -1)) != int(current["version"])
                or str(activation.get("artifact_hash")) != artifact_hash
                or [int(value) for value in activation.get("prompt_token_ids", [])] != expected_context
            ):
                return {
                    "accepted": False,
                    "code": "micro_llm_activation_mismatch",
                    "reason": "activation row does not match claim-time context or model identity",
                    "request_index": index,
                    "activation": activation,
                }
            try:
                hidden = [float(value) for value in activation.get("hidden_state", [])]
            except (TypeError, ValueError):
                return {
                    "accepted": False,
                    "code": "micro_llm_activation_not_numeric",
                    "reason": "activation hidden_state must be numeric",
                    "request_index": index,
                }
            expected_hidden = hidden_state_for(weights, config, expected_context)
            if len(hidden) != len(expected_hidden) or any(abs(a - b) > 1e-12 for a, b in zip(hidden, expected_hidden)):
                return {
                    "accepted": False,
                    "code": "micro_llm_activation_hidden_mismatch",
                    "reason": "activation hidden_state does not match Coordinator recomputation",
                    "request_index": index,
                }
            if str(activation.get("activation_hash")) != _activation_hash({**activation, "hidden_state": hidden}):
                return {
                    "accepted": False,
                    "code": "micro_llm_activation_hash_mismatch",
                    "reason": "activation_hash does not match activation payload",
                    "request_index": index,
                }
            normalized_activations.append({**activation, "hidden_state": hidden})
        activation_bytes = len(json_repr({"activation_results": normalized_activations}).encode("utf-8"))
        return {
            "accepted": True,
            "code": "ok",
            "reason": "accepted",
            "stage_id": 0,
            "stage_count": 2,
            "session_id": session_id,
            "base_model_version": int(current["version"]),
            "artifact_hash": artifact_hash,
            "artifact_schema": current.get("artifact_schema"),
            "artifact_id": current.get("artifact_id"),
            "artifact_version": current.get("artifact_version"),
            "tokenizer_schema": current.get("tokenizer_schema"),
            "request_count": len(expected_requests),
            "decode_steps": decode_steps,
            "activation_result": normalized_activations[0],
            "activation_results": normalized_activations,
            "activation_count": len(normalized_activations),
            "activation_bytes": activation_bytes,
            "activation_hashes": [row["activation_hash"] for row in normalized_activations],
            "activation_transport_ready": True,
        }

    baseline = run_micro_llm_full_inference({
        "config": config,
        "weights": weights,
        "requests": expected_requests,
        "decode_steps": decode_steps,
        "top_k": top_k,
    })
    observed_results = list(sharded_result.get("inference_results") or [])
    if not observed_results and isinstance(sharded_result.get("inference_result"), dict):
        observed_results = [dict(sharded_result["inference_result"])]
    if len(observed_results) != len(expected_requests):
        return {
            "accepted": False,
            "code": "micro_llm_result_count_mismatch",
            "reason": "stage-1 result count does not match claim-time requests",
            "stage_id": 1,
            "request_count": len(observed_results),
            "expected_request_count": len(expected_requests),
        }
    normalized_results = []
    for index, row in enumerate(observed_results):
        if not isinstance(row, dict):
            return {
                "accepted": False,
                "code": "micro_llm_result_invalid",
                "reason": "stage-1 inference_results rows must be objects",
                "stage_id": 1,
                "request_index": index,
            }
        normalized = {
            **row,
            "prompt_token_ids": [int(value) for value in row.get("prompt_token_ids", [])],
            "target_token_id": int(row.get("target_token_id", 0)),
            "generated_token_ids": [int(value) for value in row.get("generated_token_ids", [])],
            "decode_steps": int(row.get("decode_steps", 0)),
        }
        normalized_results.append(normalized)
    baseline_results = baseline.get("inference_results") or []
    baseline_compact = [
        {
            "request_id": row.get("request_id"),
            "prompt_token_ids": row.get("prompt_token_ids"),
            "target_token_id": row.get("target_token_id"),
            "generated_token_ids": row.get("generated_token_ids"),
            "steps": [
                {
                    "decode_index": step.get("decode_index"),
                    "context_token_ids": step.get("context_token_ids"),
                    "predicted_token_id": step.get("predicted_token_id"),
                    "top_k": [
                        {
                            "token_id": top.get("token_id"),
                            "probability": top.get("probability"),
                        }
                        for top in step.get("top_k", [])
                    ],
                }
                for step in row.get("steps", [])
            ],
        }
        for row in baseline_results
    ]
    observed_compact = [
        {
            "request_id": row.get("request_id"),
            "prompt_token_ids": row.get("prompt_token_ids"),
            "target_token_id": row.get("target_token_id"),
            "generated_token_ids": row.get("generated_token_ids"),
            "steps": [
                {
                    "decode_index": step.get("decode_index"),
                    "context_token_ids": step.get("context_token_ids"),
                    "predicted_token_id": step.get("predicted_token_id"),
                    "top_k": [
                        {
                            "token_id": top.get("token_id"),
                            "probability": top.get("probability"),
                        }
                        for top in step.get("top_k", [])
                    ],
                }
                for step in row.get("steps", [])
            ],
        }
        for row in normalized_results
    ]
    baseline_match = observed_compact == baseline_compact
    if not baseline_match:
        return {
            "accepted": False,
            "code": "micro_llm_baseline_mismatch",
            "reason": "stage-1 micro LLM output does not match single-process baseline",
            "stage_id": 1,
            "baseline_match": False,
            "decoded_tokens_match": False,
        }
    generated_token_count = sum(len(row.get("generated_token_ids", [])) for row in normalized_results)
    request_trace = [_safe_micro_llm_trace(config, result) for result in normalized_results[:MICRO_LLM_TRACE_LIMIT]]
    first = normalized_results[0]
    validation = {
        "accepted": True,
        "code": "ok",
        "reason": "accepted",
        "stage_id": 1,
        "stage_count": 2,
        "session_id": session_id,
        "base_model_version": int(current["version"]),
        "artifact_hash": artifact_hash,
        "artifact_schema": current.get("artifact_schema"),
        "artifact_id": current.get("artifact_id"),
        "artifact_version": current.get("artifact_version"),
        "tokenizer_schema": current.get("tokenizer_schema"),
        "inference_result": normalized_results[0],
        "inference_results": normalized_results,
        "request_count": len(normalized_results),
        "decode_steps": decode_steps,
        "generated_token_count": generated_token_count,
        "generated_token_ids": list(first.get("generated_token_ids") or []),
        "generated_text": _decode_token_ids(config, list(first.get("generated_token_ids") or [])),
        "request_trace": request_trace,
        "request_trace_count": len(request_trace),
        "request_trace_truncated": len(normalized_results) > len(request_trace),
        "activation_count": int(sharded_result.get("activation_count", 0)),
        "activation_hashes": list(sharded_result.get("activation_hashes") or []),
        "activation_transport_ready": True,
        "baseline_match": True,
        "decoded_tokens_match": True,
    }
    return validation


def apply_micro_transformer_update(model: dict, local_delta: Iterable[float]) -> dict:
    root = dict(model or {})
    current = normalize_micro_transformer_model(root)
    delta = [float(value) for value in local_delta]
    if len(delta) != len(current["weights"]):
        raise ValueError(
            f"micro_transformer local_delta length {len(delta)} "
            f"does not match weights length {len(current['weights'])}"
        )
    next_velocity = [
        current["outer_momentum"] * velocity + update
        for velocity, update in zip(current["outer_velocity"], delta)
    ]
    next_weights = [
        weight + current["outer_lr"] * velocity
        for weight, velocity in zip(current["weights"], next_velocity)
    ]
    next_micro = {
        **current,
        "version": int(current["version"]) + 1,
        "weights": next_weights,
        "outer_velocity": next_velocity,
        "optimizer_step": int(current["optimizer_step"]) + 1,
        "last_loss": micro_transformer_loss_for(next_weights, current["config"], TOKEN_IDS),
    }
    root["micro_transformer"] = next_micro
    return root


def _validation_result(*, accepted: bool, code: str, reason: str, values: list[float] | None = None) -> dict:
    return {
        "accepted": accepted,
        "code": code,
        "reason": reason,
        "local_delta": values,
        "delta_norm": None,
        "loss_before": None,
        "loss_after": None,
        "loss_delta": None,
    }


def validate_micro_transformer_delta(
    model: dict,
    local_delta: Iterable[float] | None,
    *,
    max_delta_norm: float = DEFAULT_MAX_DELTA_NORM,
    max_loss_delta: float = DEFAULT_MAX_LOSS_DELTA,
) -> dict:
    current = normalize_micro_transformer_model(model)
    try:
        values = [float(value) for value in local_delta]
    except (TypeError, ValueError):
        return _validation_result(
            accepted=False,
            code="micro_transformer_delta_not_numeric",
            reason="micro_transformer_lm local_delta must be a numeric iterable",
        )
    if len(values) != len(current["weights"]):
        return _validation_result(
            accepted=False,
            code="micro_transformer_delta_length_mismatch",
            reason=(
                f"micro_transformer local_delta length {len(values)} "
                f"does not match weights length {len(current['weights'])}"
            ),
            values=values,
        )
    if not all(math.isfinite(value) for value in values):
        return _validation_result(
            accepted=False,
            code="micro_transformer_delta_non_finite",
            reason="micro_transformer local_delta contains NaN or infinite values",
            values=values,
        )
    delta_norm = math.sqrt(sum(value * value for value in values))
    if delta_norm > max_delta_norm:
        result = _validation_result(
            accepted=False,
            code="micro_transformer_delta_norm_too_large",
            reason=f"micro_transformer local_delta norm {delta_norm:.6f} exceeds max_delta_norm {max_delta_norm:.6f}",
            values=values,
        )
        result["delta_norm"] = delta_norm
        return result

    loss_before = micro_transformer_loss(current)
    candidate = apply_micro_transformer_update({"micro_transformer": current}, values)
    loss_after = micro_transformer_loss(candidate)
    loss_delta = loss_after - loss_before
    return {
        "accepted": loss_delta <= max_loss_delta,
        "code": "ok" if loss_delta <= max_loss_delta else "micro_transformer_loss_spike",
        "reason": "accepted" if loss_delta <= max_loss_delta else (
            f"candidate micro_transformer loss increase {loss_delta:.6f} exceeds max_loss_delta {max_loss_delta:.6f}"
        ),
        "local_delta": values,
        "delta_norm": delta_norm,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_delta": loss_delta,
        "max_delta_norm": max_delta_norm,
        "max_loss_delta": max_loss_delta,
    }
