#!/usr/bin/env python3
"""Build a public-safe Qwen/Llama-like TPU stage adapter plan for 32B RC work."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "gpu_tpu_qwen_stage_adapter_plan_v1"
DEFAULT_OUTPUT_DIR = "dist/gpu-tpu-qwen-stage-adapter-plan"
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-32B-Instruct"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY=",
    "KAGGLE_USERNAME=",
    "HF_TOKEN=",
    "HUGGING_FACE_HUB_TOKEN=",
    "Bearer ",
    "kaggle-cookies.json",
    "kaggle-web-storage-state.json",
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"lease_token":',
    '"idempotency_key":',
    "operator.private.env",
    "miner.private.env",
    "kernel.py",
)


LAYER_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def fetch_hf_json(model_repo: str, filename: str, *, timeout_seconds: float = 120.0) -> dict[str, Any]:
    url = f"https://huggingface.co/{model_repo}/resolve/main/{filename}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def fixture_config() -> dict[str, Any]:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 5120,
        "intermediate_size": 27648,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "num_hidden_layers": 64,
        "vocab_size": 152064,
        "torch_dtype": "bfloat16",
        "rope_theta": 1000000.0,
    }


def fixture_weight_index(config: dict[str, Any]) -> dict[str, Any]:
    weight_map: dict[str, str] = {
        "model.embed_tokens.weight": "model-00001-of-00010.safetensors",
        "model.norm.weight": "model-00010-of-00010.safetensors",
        "lm_head.weight": "model-00010-of-00010.safetensors",
    }
    layer_count = _int(config.get("num_hidden_layers"), 64)
    suffixes = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    for layer in range(layer_count):
        filename = f"model-{min(10, layer // 7 + 1):05d}-of-00010.safetensors"
        for suffix in suffixes:
            weight_map[f"model.layers.{layer}.{suffix}"] = filename
    return {"metadata": {"total_size": 64_000_000_000}, "weight_map": weight_map}


def default_tpu_range(layer_count: int) -> tuple[int, int]:
    start = max(0, layer_count // 3)
    end = min(layer_count, (layer_count * 2) // 3)
    return start, max(start + 1, end)


def map_layer_suffix(suffix: str, *, local_layer: int) -> dict[str, Any]:
    prefix = f"decoder.layers.{local_layer}"
    mapping: dict[str, tuple[str, str, str]] = {
        "input_layernorm.weight": (f"{prefix}.pre_self_attention_layer_norm.scale", "copy_1d", "layer_norm_scale"),
        "post_attention_layernorm.weight": (f"{prefix}.post_self_attention_layer_norm.scale", "copy_1d", "layer_norm_scale"),
        "self_attn.q_proj.weight": (f"{prefix}.self_attention.query.kernel", "transpose_2d", "attention_query_kernel"),
        "self_attn.q_proj.bias": (f"{prefix}.self_attention.query.bias", "copy_1d", "attention_query_bias"),
        "self_attn.k_proj.weight": (f"{prefix}.self_attention.key.kernel", "transpose_2d", "attention_key_kernel"),
        "self_attn.k_proj.bias": (f"{prefix}.self_attention.key.bias", "copy_1d", "attention_key_bias"),
        "self_attn.v_proj.weight": (f"{prefix}.self_attention.value.kernel", "transpose_2d", "attention_value_kernel"),
        "self_attn.v_proj.bias": (f"{prefix}.self_attention.value.bias", "copy_1d", "attention_value_bias"),
        "self_attn.o_proj.weight": (f"{prefix}.self_attention.out.kernel", "transpose_2d", "attention_out_kernel"),
        "self_attn.o_proj.bias": (f"{prefix}.self_attention.out.bias", "copy_1d", "attention_out_bias"),
        "self_attn.q_norm.weight": (f"{prefix}.self_attention.query_norm.scale", "copy_1d", "attention_query_norm"),
        "self_attn.k_norm.weight": (f"{prefix}.self_attention.key_norm.scale", "copy_1d", "attention_key_norm"),
        "mlp.gate_proj.weight": (f"{prefix}.mlp.gate.kernel", "transpose_2d", "mlp_gate_kernel"),
        "mlp.up_proj.weight": (f"{prefix}.mlp.up.kernel", "transpose_2d", "mlp_up_kernel"),
        "mlp.down_proj.weight": (f"{prefix}.mlp.down.kernel", "transpose_2d", "mlp_down_kernel"),
    }
    if suffix not in mapping:
        return {
            "mapped": False,
            "jax_path": "",
            "transform": "unsupported",
            "semantic": "unsupported",
        }
    jax_path, transform, semantic = mapping[suffix]
    return {
        "mapped": True,
        "jax_path": jax_path,
        "transform": transform,
        "semantic": semantic,
    }


def build_mapping(config: dict[str, Any], weight_index: dict[str, Any], *, tpu_start: int, tpu_end: int) -> dict[str, Any]:
    weight_map = {str(key): str(value) for key, value in _dict(weight_index.get("weight_map")).items()}
    assigned: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    assigned_files: set[str] = set()
    for key in sorted(weight_map):
        match = LAYER_KEY_PATTERN.match(key)
        if not match:
            continue
        layer = int(match.group(1))
        if layer < tpu_start or layer >= tpu_end:
            continue
        suffix = match.group(2)
        local_layer = layer - tpu_start
        mapped = map_layer_suffix(suffix, local_layer=local_layer)
        filename = weight_map[key]
        assigned_files.add(filename)
        row = {
            "torch_key_hash": stable_hash(key),
            "torch_key_suffix": suffix,
            "global_layer": layer,
            "local_tpu_layer": local_layer,
            "source_file": filename,
            "mapped": bool(mapped.get("mapped")),
            "jax_path": mapped.get("jax_path"),
            "transform": mapped.get("transform"),
            "semantic": mapped.get("semantic"),
        }
        assigned.append(row)
        if not mapped.get("mapped"):
            unsupported.append(row)
    return {
        "assigned_key_count": len(assigned),
        "assigned_file_count": len(assigned_files),
        "assigned_files": sorted(assigned_files),
        "mapped_key_count": sum(1 for item in assigned if item.get("mapped")),
        "unsupported_key_count": len(unsupported),
        "unsupported_key_suffixes": sorted({str(item.get("torch_key_suffix") or "") for item in unsupported}),
        "mapping_samples": assigned[:24],
        "all_assigned_keys_mapped": bool(assigned and not unsupported),
    }


def build_shape_protocol(config: dict[str, Any], *, tpu_start: int, tpu_end: int, activation_dtype: str, context_length: int) -> dict[str, Any]:
    hidden_size = _int(config.get("hidden_size"))
    layer_count = max(0, tpu_end - tpu_start)
    num_heads = _int(config.get("num_attention_heads"))
    num_kv_heads = _int(config.get("num_key_value_heads"), num_heads)
    head_dim = hidden_size // num_heads if hidden_size and num_heads else 0
    bytes_per = {"float16": 2, "bfloat16": 2, "float32": 4}.get(activation_dtype, 2)
    return {
        "schema": "gpu_tpu_qwen_stage_adapter_shape_protocol_v1",
        "activation_metadata": {
            "layout": "batch_seq_hidden",
            "dtype": activation_dtype,
            "shape": [1, context_length, hidden_size],
            "shape_public": True,
            "activation_payload_public": False,
            "bytes_per_token": hidden_size * bytes_per,
        },
        "stage_local_kv_cache": {
            "stage_local_only": True,
            "transport_public": False,
            "cache_tensors_public": False,
            "layer_count": layer_count,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": activation_dtype,
            "estimated_kv_bytes_per_token": 2 * layer_count * num_kv_heads * head_dim * bytes_per,
        },
        "public_artifact_safe": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config_json)
    index_path = Path(args.weight_index_json)
    blockers: list[str] = []
    fetch_error: dict[str, Any] = {}
    if args.mode == "fixture":
        config = fixture_config()
        weight_index = fixture_weight_index(config)
    else:
        try:
            config = load_json(config_path) if args.config_json else fetch_hf_json(args.model_repo, "config.json")
            weight_index = load_json(index_path) if args.weight_index_json else fetch_hf_json(args.model_repo, "model.safetensors.index.json")
        except Exception as exc:
            config = {}
            weight_index = {}
            blockers.append("hf_metadata_fetch_failed")
            fetch_error = {"type": type(exc).__name__, "digest": stable_hash(str(exc))}
    layer_count = _int(config.get("num_hidden_layers"))
    tpu_start, tpu_end = default_tpu_range(layer_count)
    if args.tpu_layer_start >= 0:
        tpu_start = args.tpu_layer_start
    if args.tpu_layer_end >= 0:
        tpu_end = args.tpu_layer_end
    if layer_count < 1:
        blockers.append("config_num_hidden_layers_missing")
    if tpu_start < 0 or tpu_end <= tpu_start or (layer_count and tpu_end > layer_count):
        blockers.append("invalid_tpu_layer_range")
    mapping = build_mapping(config, weight_index, tpu_start=tpu_start, tpu_end=tpu_end) if not blockers else {
        "assigned_key_count": 0,
        "assigned_file_count": 0,
        "assigned_files": [],
        "mapped_key_count": 0,
        "unsupported_key_count": 0,
        "unsupported_key_suffixes": [],
        "mapping_samples": [],
        "all_assigned_keys_mapped": False,
    }
    if mapping.get("assigned_key_count", 0) < 1:
        blockers.append("tpu_stage_assigned_keys_missing")
    if not mapping.get("all_assigned_keys_mapped"):
        blockers.append("tpu_stage_key_mapping_incomplete")
    shape_protocol = build_shape_protocol(
        config,
        tpu_start=tpu_start,
        tpu_end=tpu_end,
        activation_dtype=args.activation_dtype,
        context_length=args.context_length,
    )
    plan_ready = bool(
        not any(item in blockers for item in ["hf_metadata_fetch_failed", "config_num_hidden_layers_missing", "invalid_tpu_layer_range", "tpu_stage_assigned_keys_missing", "tpu_stage_key_mapping_incomplete"])
        and mapping.get("all_assigned_keys_mapped")
    )
    runtime_ready = False
    if not runtime_ready:
        blockers.append("jax_tpu_runtime_execution_not_performed")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": plan_ready,
        "mode": args.mode,
        "model_repo": args.model_repo,
        "model_type": str(config.get("model_type") or ""),
        "architectures": list(config.get("architectures") or []),
        "target_model_tier": "32b",
        "decoder_layer_count": layer_count,
        "hidden_size": _int(config.get("hidden_size")),
        "intermediate_size": _int(config.get("intermediate_size")),
        "num_attention_heads": _int(config.get("num_attention_heads")),
        "num_key_value_heads": _int(config.get("num_key_value_heads"), _int(config.get("num_attention_heads"))),
        "tpu_stage": {
            "stage_id": args.tpu_stage_id,
            "backend": "jax_tpu",
            "layer_range": [tpu_start, tpu_end],
            "layer_count": max(0, tpu_end - tpu_start),
            "stage_owned_middle_layers": True,
        },
        "checkpoint_bridge_plan_ready": plan_ready,
        "stage_owned_tpu_loader_plan_ready": plan_ready,
        "qwen_llama_like_stage_runtime_planned": True,
        "tpu_32b_runtime_adapter_ready": False,
        "jax_tpu_runtime_execution_ready": runtime_ready,
        "same_request_live_heterogeneous_verified": False,
        "mapping": mapping,
        "shape_protocol": shape_protocol,
        "source_metadata": {
            "config_json": str(config_path) if args.config_json else f"https://huggingface.co/{args.model_repo}/resolve/main/config.json",
            "weight_index_json": str(index_path) if args.weight_index_json else f"https://huggingface.co/{args.model_repo}/resolve/main/model.safetensors.index.json",
            "metadata_only_no_weight_download": True,
            "weight_tensor_values_public": False,
        },
        "fetch_error": fetch_error,
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": True,
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "past_key_values_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "private_runtime_state_public": False,
        },
        "diagnosis_codes": [
            "gpu_tpu_qwen_stage_adapter_plan_ready" if plan_ready else "gpu_tpu_qwen_stage_adapter_plan_blocked",
            "gpu_tpu_qwen_stage_owned_mapping_ready" if mapping.get("all_assigned_keys_mapped") else "gpu_tpu_qwen_stage_owned_mapping_incomplete",
            "gpu_tpu_qwen_runtime_execution_not_performed",
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["diagnosis_codes"].append("gpu_tpu_qwen_stage_adapter_public_redaction_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "gpu_tpu_qwen_stage_adapter_plan.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen/Llama-like TPU stage adapter plan for 32B heterogeneous RC work.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["fixture", "fetch"], default="fetch")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--config-json", default="")
    parser.add_argument("--weight-index-json", default="")
    parser.add_argument("--tpu-stage-id", type=int, default=1)
    parser.add_argument("--tpu-layer-start", type=int, default=-1)
    parser.add_argument("--tpu-layer-end", type=int, default=-1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--activation-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.context_length < 1 or args.context_length > 4096:
        raise SystemExit("--context-length must be between 1 and 4096")
    if args.tpu_stage_id < 0:
        raise SystemExit("--tpu-stage-id must be non-negative")
    if args.tpu_layer_start < -1 or args.tpu_layer_end < -1:
        raise SystemExit("--tpu-layer-start and --tpu-layer-end must be -1 or non-negative")
    if args.config_json and not Path(args.config_json).is_file():
        raise SystemExit("--config-json must point to an existing JSON file")
    if args.weight_index_json and not Path(args.weight_index_json).is_file():
        raise SystemExit("--weight-index-json must point to an existing JSON file")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Qwen TPU stage adapter plan ready: {report.get('ok')}")
        print(f"output: {Path(args.output_dir) / 'gpu_tpu_qwen_stage_adapter_plan.json'}")
        print(f"tpu runtime adapter ready: {report.get('tpu_32b_runtime_adapter_ready')}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
