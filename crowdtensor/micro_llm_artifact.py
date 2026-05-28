"""File-backed artifact format for the dependency-free micro LLM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .micro_transformer import (
    MICRO_TRANSFORMER_SCHEMA_VERSION,
    VOCAB,
    default_micro_transformer_model,
    micro_transformer_loss_for,
    normalize_config,
    parameter_count,
)


ARTIFACT_SCHEMA_VERSION = "micro_llm_artifact_v1"
TOKENIZER_SCHEMA_VERSION = "char_tokenizer_v1"
CONFIG_SCHEMA_VERSION = "micro_transformer_config_v1"
WEIGHTS_SCHEMA_VERSION = "micro_transformer_weights_v1"
DEFAULT_ARTIFACT_ID = "crowdtensor-micro-llm-alpha"
DEFAULT_PROMPTS = ["arn", "ten"]


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_artifact_hash(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    tokenizer: dict[str, Any],
    weights: dict[str, Any],
) -> str:
    public_manifest = dict(manifest)
    public_manifest.pop("artifact_hash", None)
    payload = {
        "manifest": public_manifest,
        "config": config,
        "tokenizer": tokenizer,
        "weights": weights,
    }
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_canonical(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_artifact_payloads(
    *,
    artifact_id: str = DEFAULT_ARTIFACT_ID,
    version: int = 1,
) -> dict[str, dict[str, Any]]:
    model = default_micro_transformer_model()
    config = {
        "schema": CONFIG_SCHEMA_VERSION,
        **normalize_config(model["config"]),
    }
    tokenizer = {
        "schema": TOKENIZER_SCHEMA_VERSION,
        "type": "char_vocab",
        "vocab": list(config["vocab"]),
    }
    weights = {
        "schema": WEIGHTS_SCHEMA_VERSION,
        "values": [float(value) for value in model["weights"]],
        "count": len(model["weights"]),
    }
    manifest = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": str(artifact_id or DEFAULT_ARTIFACT_ID),
        "version": int(version),
        "model_schema_version": MICRO_TRANSFORMER_SCHEMA_VERSION,
        "config_path": "config.json",
        "tokenizer_path": "tokenizer.json",
        "weights_path": "weights.json",
        "format": "json",
    }
    manifest["artifact_hash"] = compute_artifact_hash(
        manifest=manifest,
        config=config,
        tokenizer=tokenizer,
        weights=weights,
    )
    return {
        "manifest": manifest,
        "config": config,
        "tokenizer": tokenizer,
        "weights": weights,
    }


def build_default_micro_llm_artifact(
    output_dir: str | Path,
    *,
    artifact_id: str = DEFAULT_ARTIFACT_ID,
    version: int = 1,
) -> dict[str, Any]:
    root = Path(output_dir)
    payloads = default_artifact_payloads(artifact_id=artifact_id, version=version)
    write_json(root / "config.json", payloads["config"])
    write_json(root / "tokenizer.json", payloads["tokenizer"])
    write_json(root / "weights.json", payloads["weights"])
    write_json(root / "manifest.json", payloads["manifest"])
    return inspect_micro_llm_artifact(root)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def manifest_path_for(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "manifest.json" if candidate.is_dir() else candidate


def load_micro_llm_artifact(path: str | Path) -> dict[str, Any]:
    manifest_path = manifest_path_for(path)
    root = manifest_path.parent
    manifest = _load_json(manifest_path, label="micro LLM artifact manifest")
    if manifest.get("schema") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("micro LLM artifact manifest schema mismatch")
    config = _load_json(root / str(manifest.get("config_path") or "config.json"), label="micro LLM config")
    tokenizer = _load_json(root / str(manifest.get("tokenizer_path") or "tokenizer.json"), label="micro LLM tokenizer")
    weights = _load_json(root / str(manifest.get("weights_path") or "weights.json"), label="micro LLM weights")
    if config.get("schema") != CONFIG_SCHEMA_VERSION:
        raise ValueError("micro LLM config schema mismatch")
    if tokenizer.get("schema") != TOKENIZER_SCHEMA_VERSION:
        raise ValueError("micro LLM tokenizer schema mismatch")
    if weights.get("schema") != WEIGHTS_SCHEMA_VERSION:
        raise ValueError("micro LLM weights schema mismatch")
    normalized_config = normalize_config(config)
    vocab = [str(item) for item in tokenizer.get("vocab") or []]
    if vocab != list(normalized_config["vocab"]):
        raise ValueError("micro LLM tokenizer vocab does not match config vocab")
    values = [float(value) for value in weights.get("values") or []]
    expected = parameter_count(normalized_config)
    if len(values) != expected:
        raise ValueError(f"micro LLM weights count {len(values)} does not match expected {expected}")
    computed_hash = compute_artifact_hash(
        manifest=manifest,
        config=config,
        tokenizer=tokenizer,
        weights=weights,
    )
    if manifest.get("artifact_hash") and manifest["artifact_hash"] != computed_hash:
        raise ValueError("micro LLM artifact_hash mismatch")
    artifact_hash = str(manifest.get("artifact_hash") or computed_hash)
    model = {
        "schema_version": MICRO_TRANSFORMER_SCHEMA_VERSION,
        "version": 0,
        "config": normalized_config,
        "weights": values,
        "outer_velocity": [0.0 for _ in values],
        "optimizer_step": 0,
        "last_loss": micro_transformer_loss_for(values, normalized_config),
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": str(manifest.get("artifact_id") or DEFAULT_ARTIFACT_ID),
        "artifact_version": int(manifest.get("version", 1)),
        "artifact_hash": artifact_hash,
        "artifact_source": str(manifest_path),
        "tokenizer_schema": TOKENIZER_SCHEMA_VERSION,
    }
    return {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": model["artifact_id"],
        "artifact_version": model["artifact_version"],
        "artifact_hash": artifact_hash,
        "manifest_path": str(manifest_path),
        "config": normalized_config,
        "tokenizer": tokenizer,
        "weight_count": len(values),
        "model": model,
    }


def inspect_micro_llm_artifact(path: str | Path) -> dict[str, Any]:
    artifact = load_micro_llm_artifact(path)
    return {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "ok": True,
        "artifact_id": artifact["artifact_id"],
        "artifact_version": artifact["artifact_version"],
        "artifact_hash": artifact["artifact_hash"],
        "manifest_path": artifact["manifest_path"],
        "tokenizer_schema": TOKENIZER_SCHEMA_VERSION,
        "config": artifact["config"],
        "weight_count": artifact["weight_count"],
        "default_prompts": list(DEFAULT_PROMPTS),
    }


def encode_prompt_text(prompt: str, config: dict[str, Any], tokenizer: dict[str, Any] | None = None) -> list[int]:
    text = str(prompt)
    cfg = normalize_config(config)
    context_length = int(cfg["context_length"])
    if len(text) != context_length:
        raise ValueError(f"micro LLM prompt must be exactly {context_length} characters")
    vocab = list((tokenizer or {}).get("vocab") or cfg["vocab"])
    index = {str(token): pos for pos, token in enumerate(vocab)}
    unknown = sorted({char for char in text if char not in index})
    if unknown:
        raise ValueError("micro LLM prompt contains characters outside the artifact tokenizer")
    return [index[char] for char in text]


def prompt_requests_for_model(
    model: dict[str, Any],
    *,
    prompt_texts: list[str] | None = None,
) -> list[dict[str, Any]]:
    prompts = list(prompt_texts or DEFAULT_PROMPTS)
    config = normalize_config((model or {}).get("config"))
    tokenizer = {
        "schema": TOKENIZER_SCHEMA_VERSION,
        "vocab": list(config["vocab"]),
    }
    requests = []
    for index, prompt in enumerate(prompts):
        prompt_ids = encode_prompt_text(str(prompt), config, tokenizer)
        requests.append({
            "request_id": f"req-{index + 1}",
            "prompt_token_ids": prompt_ids,
            "target_token_id": 0,
            "sample_offset": index,
            "prompt_hash": "sha256:" + hashlib.sha256(str(prompt).encode("utf-8")).hexdigest(),
        })
    return requests
