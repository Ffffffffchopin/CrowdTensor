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
WORKLOAD_TYPE = "micro_transformer_lm"
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


def _stable_offset(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


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
    return {
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
