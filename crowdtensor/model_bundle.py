"""Dependency-free model bundle language-model workload.

The bundle contract is a small CPU-only adapter boundary. It keeps model
artifact identity, versioning, validation, and replay audit semantics explicit
without depending on external model files or accelerator runtimes.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Iterable


MODEL_BUNDLE_SCHEMA_VERSION = "model_bundle_lm_v1"
MODEL_BUNDLE_INFERENCE_SCHEMA_VERSION = "model_bundle_infer_v1"
WORKLOAD_TYPE = "model_bundle_lm"
INFERENCE_WORKLOAD_TYPE = "model_bundle_infer"
BUNDLE_ID = "builtin-char-bundle"
CORPUS = "crowd tensor nodes route gradients safely "
VOCAB = sorted(set(CORPUS))
TOKEN_IDS = [VOCAB.index(char) for char in CORPUS]
CONTEXT_LENGTH = 4
DEFAULT_INNER_LR = 0.06
DEFAULT_LOCAL_DELTA_SCALE = 0.12
DEFAULT_OUTER_LR = 0.65
DEFAULT_OUTER_MOMENTUM = 0.7
DEFAULT_MAX_DELTA_NORM = 4.0
DEFAULT_MAX_LOSS_DELTA = 1.0


def _stable_offset(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _artifact_hash(values: Iterable[float], config: dict) -> str:
    payload = {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "bundle_id": BUNDLE_ID,
        "config": normalize_config(config),
        "weights": [round(float(value), 12) for value in values],
    }
    raw = repr(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def default_config() -> dict:
    return {
        "vocab": list(VOCAB),
        "vocab_size": len(VOCAB),
        "context_length": CONTEXT_LENGTH,
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
    return {
        "vocab": vocab,
        "vocab_size": vocab_size,
        "context_length": context_length,
    }


def parameter_count(config: dict | None = None) -> int:
    cfg = default_config() if config is None else normalize_config(config)
    vocab_size = int(cfg["vocab_size"])
    return int(cfg["context_length"]) * vocab_size + vocab_size


def _initial_weight(index: int) -> float:
    return 0.04 * math.sin((index + 1) * 1.324717957244746)


def default_model_bundle() -> dict:
    config = default_config()
    weights = [_initial_weight(index) for index in range(parameter_count(config))]
    return {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "bundle_id": BUNDLE_ID,
        "version": 0,
        "config": config,
        "weights": weights,
        "artifact_hash": _artifact_hash(weights, config),
        "outer_lr": DEFAULT_OUTER_LR,
        "outer_momentum": DEFAULT_OUTER_MOMENTUM,
        "outer_velocity": [0.0 for _ in weights],
        "optimizer_step": 0,
        "last_loss": bundle_loss_for(weights, config, TOKEN_IDS),
    }


def normalize_model_bundle(model: dict | None) -> dict:
    source = dict(model or {})
    if "model_bundle" in source and isinstance(source.get("model_bundle"), dict):
        source = dict(source["model_bundle"])
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
    artifact_hash = str(source.get("artifact_hash") or _artifact_hash(weights, config))
    return {
        "schema_version": source.get("schema_version", MODEL_BUNDLE_SCHEMA_VERSION),
        "bundle_id": str(source.get("bundle_id", BUNDLE_ID) or BUNDLE_ID),
        "version": int(source.get("version", 0)),
        "config": config,
        "weights": weights,
        "artifact_hash": artifact_hash,
        "outer_lr": float(source.get("outer_lr", DEFAULT_OUTER_LR)),
        "outer_momentum": float(source.get("outer_momentum", DEFAULT_OUTER_MOMENTUM)),
        "outer_velocity": velocity,
        "optimizer_step": int(source.get("optimizer_step", source.get("version", 0))),
        "last_loss": float(source.get("last_loss", bundle_loss_for(weights, config, token_ids))),
    }


def model_bundle_training_spec_for(task_id: str, miner_id: str, model: dict) -> dict:
    current = normalize_model_bundle(model)
    config = dict(current["config"])
    token_ids = list(TOKEN_IDS)
    return {
        "type": WORKLOAD_TYPE,
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "bundle_id": current["bundle_id"],
        "bundle_version": int(current["version"]),
        "artifact_hash": current["artifact_hash"],
        "config": config,
        "weights": list(current["weights"]),
        "token_ids": token_ids,
        "inner_lr": DEFAULT_INNER_LR,
        "local_delta_scale": DEFAULT_LOCAL_DELTA_SCALE,
        "max_delta_norm": DEFAULT_MAX_DELTA_NORM,
        "max_loss_delta": DEFAULT_MAX_LOSS_DELTA,
        "batch_size": _example_count(token_ids, config),
        "sample_offset": _stable_offset(
            task_id,
            miner_id,
            current["bundle_id"],
            current["version"],
        ) % _example_count(token_ids, config),
    }


def model_bundle_inference_spec_for(
    task_id: str,
    miner_id: str,
    model: dict,
    *,
    request_count: int = 1,
) -> dict:
    current = normalize_model_bundle(model)
    config = dict(current["config"])
    token_ids = list(TOKEN_IDS)
    example_count = _example_count(token_ids, config)
    sample_offset = _stable_offset(
        task_id,
        miner_id,
        current["bundle_id"],
        current["version"],
        "infer",
    ) % example_count
    count = max(1, min(int(request_count), example_count))
    requests = []
    for index in range(count):
        context, target = _example_at(token_ids, config, sample_offset + index)
        requests.append({
            "request_id": f"req-{index + 1}",
            "prompt_token_ids": context,
            "target_token_id": target,
            "top_k": 3,
            "sample_offset": sample_offset + index,
        })
    first = requests[0]
    return {
        "type": INFERENCE_WORKLOAD_TYPE,
        "schema_version": MODEL_BUNDLE_INFERENCE_SCHEMA_VERSION,
        "bundle_id": current["bundle_id"],
        "bundle_version": int(current["version"]),
        "artifact_hash": current["artifact_hash"],
        "config": config,
        "weights": list(current["weights"]),
        "requests": requests,
        "request_count": len(requests),
        "prompt_token_ids": list(first["prompt_token_ids"]),
        "target_token_id": int(first["target_token_id"]),
        "top_k": int(first["top_k"]),
        "sample_offset": sample_offset,
    }


def model_bundle_version(model: dict) -> int:
    return int(normalize_model_bundle(model).get("version", 0))


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


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(max(-60.0, min(60.0, value - peak))) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(values) for _ in values]
    return [value / total for value in exps]


def logits_for(weights: Iterable[float], config: dict, context: list[int]) -> list[float]:
    values = [float(value) for value in weights]
    cfg = normalize_config(config)
    vocab_size = int(cfg["vocab_size"])
    context_length = int(cfg["context_length"])
    logits = list(values[context_length * vocab_size:context_length * vocab_size + vocab_size])
    for position, token_id in enumerate(context[:context_length]):
        base = position * vocab_size
        for token in range(vocab_size):
            logits[token] += values[base + token] * (1.0 if token == token_id else 0.0)
    return logits


def _cross_entropy(logits: list[float], target: int) -> float:
    probabilities = _softmax(logits)
    return -math.log(max(probabilities[target], 1e-12))


def bundle_loss_for(weights: Iterable[float], config: dict, token_ids: list[int]) -> float:
    values = [float(value) for value in weights]
    cfg = normalize_config(config)
    count = _example_count(token_ids, cfg)
    total = 0.0
    for index in range(count):
        context, target = _example_at(token_ids, cfg, index)
        total += _cross_entropy(logits_for(values, cfg, context), target)
    return total / count


def model_bundle_loss(model: dict) -> float:
    bundle = normalize_model_bundle(model)
    return bundle_loss_for(bundle["weights"], bundle["config"], TOKEN_IDS)


def _gradient_for(weights: list[float], config: dict, context: list[int], target: int) -> list[float]:
    cfg = normalize_config(config)
    vocab_size = int(cfg["vocab_size"])
    context_length = int(cfg["context_length"])
    probabilities = _softmax(logits_for(weights, cfg, context))
    grad_logits = list(probabilities)
    grad_logits[target] -= 1.0
    gradient = [0.0 for _ in weights]
    head_start = context_length * vocab_size
    for token, value in enumerate(grad_logits):
        gradient[head_start + token] += value
    for position, token_id in enumerate(context[:context_length]):
        gradient[position * vocab_size + token_id] += grad_logits[token_id]
    return gradient


def _l2(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def run_model_bundle_inner_loop(
    workload_spec: dict,
    *,
    inner_steps: int,
    compute_seconds: float = 0.0,
) -> dict:
    spec = dict(workload_spec or {})
    config = normalize_config(spec.get("config"))
    initial = [float(value) for value in spec.get("weights", [])]
    expected = parameter_count(config)
    if len(initial) != expected:
        raise ValueError(f"model bundle weights length {len(initial)} does not match expected {expected}")
    local = list(initial)
    token_ids = _token_ids_from_spec(spec, config)
    inner_lr = float(spec.get("inner_lr", DEFAULT_INNER_LR))
    local_delta_scale = float(spec.get("local_delta_scale", DEFAULT_LOCAL_DELTA_SCALE))
    sample_offset = int(spec.get("sample_offset", 0))
    steps = max(1, int(inner_steps))
    start = time.monotonic()
    deadline = start + max(0.0, compute_seconds)
    step = 0
    loss_start = bundle_loss_for(local, config, token_ids)

    while step < steps or (compute_seconds > 0 and time.monotonic() < deadline):
        context, target = _example_at(token_ids, config, sample_offset + step)
        gradient = _gradient_for(local, config, context, target)
        for index, value in enumerate(gradient):
            local[index] -= inner_lr * value
        step += 1
        if compute_seconds > 0 and step % 200 == 0:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.05, remaining))

    values = [
        (local_value - initial_value) * local_delta_scale
        for local_value, initial_value in zip(local, initial)
    ]
    return {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "bundle_delta": {
            "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
            "bundle_id": str(spec.get("bundle_id", BUNDLE_ID)),
            "base_bundle_version": int(spec.get("bundle_version", 0)),
            "artifact_hash": str(spec.get("artifact_hash", "")),
            "values": values,
        },
        "bundle_loss_start": loss_start,
        "bundle_loss_end": bundle_loss_for(local, config, token_ids),
        "samples_seen": step,
        "inner_steps": steps,
        "inner_lr": inner_lr,
        "local_delta_scale": local_delta_scale,
        "sample_offset": sample_offset,
        "delta_norm": _l2(values),
    }


def _token_text(config: dict, token_id: int) -> str:
    cfg = normalize_config(config)
    vocab = list(cfg["vocab"])
    if not vocab:
        return ""
    return str(vocab[int(token_id) % len(vocab)])


def _inference_requests_from_spec(spec: dict, config: dict) -> list[dict]:
    rows = spec.get("requests")
    if rows is None:
        rows = [{
            "request_id": "req-1",
            "prompt_token_ids": spec.get("prompt_token_ids", []),
            "target_token_id": spec.get("target_token_id", 0),
            "top_k": spec.get("top_k", 3),
            "sample_offset": spec.get("sample_offset", 0),
        }]
    requests = []
    for index, row in enumerate(list(rows)):
        if not isinstance(row, dict):
            raise ValueError("model bundle inference request rows must be objects")
        prompt = [int(value) % int(config["vocab_size"]) for value in row.get("prompt_token_ids", [])]
        if len(prompt) != int(config["context_length"]):
            raise ValueError("model bundle inference prompt length does not match context_length")
        target = int(row.get("target_token_id", 0)) % int(config["vocab_size"])
        top_k = max(1, min(int(row.get("top_k", spec.get("top_k", 3))), int(config["vocab_size"])))
        requests.append({
            "request_id": str(row.get("request_id") or f"req-{index + 1}"),
            "prompt_token_ids": prompt,
            "target_token_id": target,
            "top_k": top_k,
            "sample_offset": int(row.get("sample_offset", index)),
        })
    if not requests:
        raise ValueError("model bundle inference requires at least one request")
    return requests


def _run_single_model_bundle_inference_request(
    *,
    request: dict,
    config: dict,
    weights: list[float],
    bundle_id: str,
    bundle_version: int,
    artifact_hash: str,
) -> dict:
    prompt = list(request["prompt_token_ids"])
    target = int(request["target_token_id"])
    top_k = int(request["top_k"])
    logits = logits_for(weights, config, prompt)
    probabilities = _softmax(logits)
    ranked = sorted(
        enumerate(probabilities),
        key=lambda item: (-float(item[1]), int(item[0])),
    )[:top_k]
    predicted_id = int(ranked[0][0])
    return {
        "schema_version": MODEL_BUNDLE_INFERENCE_SCHEMA_VERSION,
        "request_id": str(request["request_id"]),
        "bundle_id": bundle_id,
        "base_bundle_version": int(bundle_version),
        "artifact_hash": artifact_hash,
        "prompt_token_ids": prompt,
        "target_token_id": target,
        "target_token": _token_text(config, target),
        "predicted_token_id": predicted_id,
        "predicted_token": _token_text(config, predicted_id),
        "top_k": [
            {
                "token_id": int(token_id),
                "token": _token_text(config, int(token_id)),
                "probability": float(probability),
            }
            for token_id, probability in ranked
        ],
        "correct": predicted_id == target,
    }


def run_model_bundle_inference(workload_spec: dict) -> dict:
    spec = dict(workload_spec or {})
    config = normalize_config(spec.get("config"))
    weights = [float(value) for value in spec.get("weights", [])]
    expected = parameter_count(config)
    if len(weights) != expected:
        raise ValueError(f"model bundle inference weights length {len(weights)} does not match expected {expected}")
    start = time.monotonic()
    requests = _inference_requests_from_spec(spec, config)
    bundle_id = str(spec.get("bundle_id", BUNDLE_ID))
    bundle_version = int(spec.get("bundle_version", 0))
    artifact_hash = str(spec.get("artifact_hash", ""))
    results = [
        _run_single_model_bundle_inference_request(
            request=request,
            config=config,
            weights=weights,
            bundle_id=bundle_id,
            bundle_version=bundle_version,
            artifact_hash=artifact_hash,
        )
        for request in requests
    ]
    elapsed_ms = (time.monotonic() - start) * 1000.0
    correct_count = sum(1 for result in results if bool(result["correct"]))
    elapsed_seconds = max(elapsed_ms / 1000.0, 1e-9)
    return {
        "schema_version": MODEL_BUNDLE_INFERENCE_SCHEMA_VERSION,
        "inference_result": results[0],
        "inference_results": results,
        "prediction_correct": correct_count == len(results),
        "request_count": len(results),
        "correct_count": correct_count,
        "accuracy": correct_count / len(results),
        "elapsed_ms": elapsed_ms,
        "requests_per_second": len(results) / elapsed_seconds,
        "top_k": max(int(request["top_k"]) for request in requests),
        "sample_offset": int(spec.get("sample_offset", 0)),
    }


def _validate_single_model_bundle_inference_result(
    current: dict,
    inference_result: dict,
    *,
    expected_request_id: str | None = None,
) -> dict:
    config = current["config"]
    if str(inference_result.get("schema_version")) != MODEL_BUNDLE_INFERENCE_SCHEMA_VERSION:
        return {
            "accepted": False,
            "code": "model_bundle_inference_schema_mismatch",
            "reason": "inference_result schema_version does not match model_bundle_infer_v1",
            "inference_result": inference_result,
        }
    request_id = str(inference_result.get("request_id") or expected_request_id or "req-1")
    if expected_request_id is not None and request_id != str(expected_request_id):
        return {
            "accepted": False,
            "code": "model_bundle_inference_request_id_mismatch",
            "reason": "inference_result request_id does not match expected request order",
            "inference_result": inference_result,
        }
    if str(inference_result.get("bundle_id")) != current["bundle_id"]:
        return {
            "accepted": False,
            "code": "model_bundle_inference_bundle_id_mismatch",
            "reason": "inference_result bundle_id does not match current bundle",
            "inference_result": inference_result,
        }
    try:
        base_bundle_version = int(inference_result.get("base_bundle_version", -1))
    except (TypeError, ValueError):
        return {
            "accepted": False,
            "code": "model_bundle_inference_version_not_numeric",
            "reason": "inference_result base_bundle_version must be an integer",
            "inference_result": inference_result,
        }
    if base_bundle_version != int(current["version"]):
        return {
            "accepted": False,
            "code": "model_bundle_inference_version_mismatch",
            "reason": "inference_result base_bundle_version does not match current bundle version",
            "inference_result": inference_result,
        }
    if str(inference_result.get("artifact_hash")) != current["artifact_hash"]:
        return {
            "accepted": False,
            "code": "model_bundle_inference_artifact_hash_mismatch",
            "reason": "inference_result artifact_hash does not match current bundle artifact_hash",
            "inference_result": inference_result,
        }
    try:
        prompt = [int(value) % int(config["vocab_size"]) for value in inference_result.get("prompt_token_ids", [])]
        target = int(inference_result.get("target_token_id"))
        predicted = int(inference_result.get("predicted_token_id"))
        top_k_rows = list(inference_result.get("top_k") or [])
    except (TypeError, ValueError):
        return {
            "accepted": False,
            "code": "model_bundle_inference_not_numeric",
            "reason": "inference_result contains non-numeric token fields",
            "inference_result": inference_result,
        }
    if len(prompt) != int(config["context_length"]):
        return {
            "accepted": False,
            "code": "model_bundle_inference_prompt_length_mismatch",
            "reason": "inference_result prompt length does not match current bundle context_length",
            "inference_result": inference_result,
        }
    if not top_k_rows:
        return {
            "accepted": False,
            "code": "model_bundle_inference_top_k_missing",
            "reason": "inference_result top_k must contain at least one row",
            "inference_result": inference_result,
        }
    expected = run_model_bundle_inference({
        "bundle_id": current["bundle_id"],
        "bundle_version": current["version"],
        "artifact_hash": current["artifact_hash"],
        "config": config,
        "weights": current["weights"],
        "requests": [{
            "request_id": request_id,
            "prompt_token_ids": prompt,
            "target_token_id": target,
            "top_k": len(top_k_rows),
        }],
    })["inference_result"]
    try:
        observed_top = [
            {
                "token_id": int(row.get("token_id")),
                "probability": float(row.get("probability")),
            }
            for row in top_k_rows
        ]
    except (AttributeError, TypeError, ValueError):
        return {
            "accepted": False,
            "code": "model_bundle_inference_top_k_invalid",
            "reason": "inference_result top_k rows must contain numeric token_id and probability",
            "inference_result": inference_result,
        }
    expected_top = [
        {
            "token_id": int(row["token_id"]),
            "probability": float(row["probability"]),
        }
        for row in expected["top_k"]
    ]
    if predicted != int(expected["predicted_token_id"]) or observed_top != expected_top:
        return {
            "accepted": False,
            "code": "model_bundle_inference_prediction_mismatch",
            "reason": "inference_result prediction does not match Coordinator recomputation",
            "expected": expected,
            "inference_result": inference_result,
        }
    correct = predicted == target
    normalized = {
        **inference_result,
        "request_id": request_id,
        "prompt_token_ids": prompt,
        "target_token_id": target,
        "predicted_token_id": predicted,
        "correct": correct,
    }
    return {
        "accepted": True,
        "code": "ok",
        "reason": "accepted",
        "inference_result": normalized,
        "bundle_id": current["bundle_id"],
        "base_bundle_version": base_bundle_version,
        "artifact_hash": current["artifact_hash"],
        "request_id": request_id,
        "predicted_token_id": predicted,
        "predicted_token": _token_text(config, predicted),
        "target_token_id": target,
        "target_token": _token_text(config, target),
        "correct": correct,
    }


def validate_model_bundle_inference(
    model: dict,
    inference_result: dict | None,
    *,
    inference_results: list[dict] | None = None,
    expected_requests: list[dict] | None = None,
) -> dict:
    current = normalize_model_bundle(model)
    if inference_results is None and inference_result is not None:
        inference_results = [inference_result]
    if not isinstance(inference_results, list) or not inference_results:
        return {
            "accepted": False,
            "code": "model_bundle_inference_missing",
            "reason": "model_bundle_infer requires inference_result or inference_results",
            "inference_result": inference_result,
            "inference_results": inference_results,
        }
    normalized_results = []
    row_validations = []
    if expected_requests is not None and len(inference_results) != len(expected_requests):
        return {
            "accepted": False,
            "code": "model_bundle_inference_request_count_mismatch",
            "reason": "inference_results length does not match claim-time requests",
            "expected_request_count": len(expected_requests),
            "request_count": len(inference_results),
            "inference_results": inference_results,
        }
    for index, row in enumerate(inference_results):
        if not isinstance(row, dict):
            return {
                "accepted": False,
                "code": "model_bundle_inference_result_invalid",
                "reason": "inference_results rows must be objects",
                "inference_results": inference_results,
            }
        expected_row = (expected_requests or [])[index] if expected_requests is not None else None
        expected_request_id = str(
            (expected_row or {}).get("request_id")
            or row.get("request_id")
            or f"req-{index + 1}"
        )
        row_validation = _validate_single_model_bundle_inference_result(
            current,
            row,
            expected_request_id=expected_request_id,
        )
        if not row_validation["accepted"]:
            return {
                **row_validation,
                "request_index": index,
                "inference_results": inference_results,
            }
        if expected_row is not None:
            expected_prompt = [
                int(value) % int(current["config"]["vocab_size"])
                for value in expected_row.get("prompt_token_ids", [])
            ]
            expected_target = int(expected_row.get("target_token_id", -1)) % int(current["config"]["vocab_size"])
            normalized = row_validation["inference_result"]
            if (
                normalized.get("prompt_token_ids") != expected_prompt
                or int(normalized.get("target_token_id", -1)) != expected_target
                or len(normalized.get("top_k") or []) != int(expected_row.get("top_k", len(normalized.get("top_k") or [])))
            ):
                return {
                    "accepted": False,
                    "code": "model_bundle_inference_request_mismatch",
                    "reason": "inference_result does not match claim-time request prompt, target, or top_k",
                    "request_index": index,
                    "expected_request": expected_row,
                    "inference_result": row,
                    "inference_results": inference_results,
                }
        row_validations.append(row_validation)
        normalized_results.append(row_validation["inference_result"])

    first = row_validations[0]
    correct_count = sum(1 for validation in row_validations if bool(validation["correct"]))
    request_count = len(row_validations)
    return {
        "accepted": True,
        "code": "ok",
        "reason": "accepted",
        "inference_result": normalized_results[0],
        "inference_results": normalized_results,
        "bundle_id": current["bundle_id"],
        "base_bundle_version": first["base_bundle_version"],
        "artifact_hash": current["artifact_hash"],
        "request_count": request_count,
        "correct_count": correct_count,
        "accuracy": correct_count / request_count,
        "request_id": first["request_id"],
        "predicted_token_id": first["predicted_token_id"],
        "predicted_token": first["predicted_token"],
        "target_token_id": first["target_token_id"],
        "target_token": first["target_token"],
        "correct": bool(first["correct"]),
    }


def validate_model_bundle_delta(model: dict, bundle_delta: dict | None) -> dict:
    current = normalize_model_bundle(model)
    if not isinstance(bundle_delta, dict):
        return {
            "accepted": False,
            "code": "model_bundle_delta_missing",
            "reason": "model_bundle_lm requires a bundle_delta object",
            "bundle_delta": bundle_delta,
        }
    values = bundle_delta.get("values")
    if not isinstance(values, list):
        return {
            "accepted": False,
            "code": "model_bundle_delta_values_missing",
            "reason": "bundle_delta.values must be a list",
            "bundle_delta": bundle_delta,
        }
    try:
        delta = [float(value) for value in values]
    except (TypeError, ValueError):
        return {
            "accepted": False,
            "code": "model_bundle_delta_not_numeric",
            "reason": "bundle_delta values must be numeric",
            "bundle_delta": bundle_delta,
        }
    if len(delta) != len(current["weights"]):
        return {
            "accepted": False,
            "code": "model_bundle_delta_length_mismatch",
            "reason": f"bundle_delta length {len(delta)} does not match weights length {len(current['weights'])}",
            "bundle_delta": {**bundle_delta, "values": delta},
        }
    if any(not math.isfinite(value) for value in delta):
        return {
            "accepted": False,
            "code": "model_bundle_delta_non_finite",
            "reason": "bundle_delta contains non-finite values",
            "bundle_delta": {**bundle_delta, "values": delta},
        }
    if str(bundle_delta.get("schema_version")) != MODEL_BUNDLE_SCHEMA_VERSION:
        return {
            "accepted": False,
            "code": "model_bundle_schema_mismatch",
            "reason": "bundle_delta schema_version does not match model_bundle_lm_v1",
            "bundle_delta": {**bundle_delta, "values": delta},
        }
    if str(bundle_delta.get("bundle_id")) != current["bundle_id"]:
        return {
            "accepted": False,
            "code": "model_bundle_id_mismatch",
            "reason": "bundle_delta bundle_id does not match current bundle",
            "bundle_delta": {**bundle_delta, "values": delta},
        }
    try:
        base_bundle_version = int(bundle_delta.get("base_bundle_version", -1))
    except (TypeError, ValueError):
        return {
            "accepted": False,
            "code": "model_bundle_version_not_numeric",
            "reason": "bundle_delta base_bundle_version must be an integer",
            "bundle_delta": {**bundle_delta, "values": delta},
        }
    if base_bundle_version != int(current["version"]):
        return {
            "accepted": False,
            "code": "model_bundle_version_mismatch",
            "reason": "bundle_delta base_bundle_version does not match current bundle version",
            "bundle_delta": {**bundle_delta, "values": delta},
        }
    if str(bundle_delta.get("artifact_hash")) != current["artifact_hash"]:
        return {
            "accepted": False,
            "code": "model_bundle_artifact_hash_mismatch",
            "reason": "bundle_delta artifact_hash does not match current bundle artifact_hash",
            "bundle_delta": {**bundle_delta, "values": delta},
        }

    delta_norm = _l2(delta)
    max_delta_norm = DEFAULT_MAX_DELTA_NORM
    loss_before = model_bundle_loss(model)
    candidate = apply_model_bundle_update(model, {**bundle_delta, "values": delta})
    loss_after = model_bundle_loss(candidate)
    loss_delta = loss_after - loss_before
    accepted = delta_norm <= max_delta_norm and loss_delta <= DEFAULT_MAX_LOSS_DELTA
    code = "ok"
    reason = "accepted"
    if delta_norm > max_delta_norm:
        code = "model_bundle_delta_norm_too_large"
        reason = f"bundle_delta norm {delta_norm:.6f} exceeds max {max_delta_norm:.6f}"
    elif loss_delta > DEFAULT_MAX_LOSS_DELTA:
        code = "model_bundle_loss_spike"
        reason = f"bundle loss delta {loss_delta:.6f} exceeds max {DEFAULT_MAX_LOSS_DELTA:.6f}"
    return {
        "accepted": accepted,
        "code": code,
        "reason": reason,
        "bundle_delta": {**bundle_delta, "values": delta},
        "bundle_id": current["bundle_id"],
        "base_bundle_version": base_bundle_version,
        "artifact_hash": current["artifact_hash"],
        "delta_norm": delta_norm,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_delta": loss_delta,
        "max_delta_norm": max_delta_norm,
        "max_loss_delta": DEFAULT_MAX_LOSS_DELTA,
    }


def apply_model_bundle_update(model: dict, bundle_delta: dict) -> dict:
    current = normalize_model_bundle(model)
    delta = [float(value) for value in bundle_delta.get("values", [])]
    momentum = float(current["outer_momentum"])
    lr = float(current["outer_lr"])
    velocity = [
        momentum * old_velocity + update
        for old_velocity, update in zip(current["outer_velocity"], delta)
    ]
    weights = [
        weight + lr * update
        for weight, update in zip(current["weights"], velocity)
    ]
    next_bundle = {
        **current,
        "version": int(current["version"]) + 1,
        "weights": weights,
        "artifact_hash": _artifact_hash(weights, current["config"]),
        "outer_velocity": velocity,
        "optimizer_step": int(current["optimizer_step"]) + 1,
    }
    next_bundle["last_loss"] = bundle_loss_for(weights, current["config"], TOKEN_IDS)
    root = dict(model)
    root["model_bundle"] = next_bundle
    return root
