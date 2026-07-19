#!/usr/bin/env python3
"""Preflight the GLM 5.2 transformers decode adapter foundation.

This is not inference success. It checks whether the project can reuse the
installed Transformers GLM-MoE-DSA implementation, normalize the public AWQ
config, and map stage-selective AWQ `pack-quantized` tensors to the modules
that a real decode worker must load.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_transformers_decode_adapter_preflight_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-transformers-decode-adapter-preflight"
MODEL_ID = "zai-org/GLM-5.2"
DEFAULT_MODEL_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
PACK_FIELDS = ["weight_packed", "weight_scale", "weight_zero_point", "weight_shape"]
PACK_RUNTIME_MODULES = ["compressed_tensors", "autoawq", "awq", "llmcompressor"]
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"raw_generated_text":',
    '"generated_token_ids":',
    '"input_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"weight_tensor_values":',
    '"safetensors_header_payload":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_hf_json(repo: str, filename: str, *, timeout_seconds: float) -> dict[str, Any]:
    quoted = urllib.parse.quote(filename)
    request = urllib.request.Request(
        f"https://huggingface.co/{repo}/resolve/main/{quoted}",
        headers={"User-Agent": "crowdtensor-glm52-transformers-preflight/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def sha_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "available": False,
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
        }
    return {
        "module": module_name,
        "available": True,
        "version": str(getattr(module, "__version__", "")),
    }


def normalize_awq_config(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    normalized = dict(config)
    layer_types = [str(item) for item in _list(normalized.get("layer_types"))]
    invalid_layer_types = [item for item in layer_types if item not in {"attention", "sparse", "dense", "moe", "hybrid"}]
    if invalid_layer_types:
        normalized.pop("layer_types", None)
        return normalized, "removed_invalid_layer_types"
    return normalized, "unchanged"


def transformers_status(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "transformers_available": False,
        "glm_moe_dsa_config_class_available": False,
        "glm_moe_dsa_model_class_available": False,
        "awq_config_original_ready": False,
        "awq_config_normalized_ready": False,
        "tiny_forward_ready": False,
        "blockers": [],
    }
    try:
        import torch
        import transformers
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

        result.update(
            {
                "transformers_available": True,
                "transformers_version": str(getattr(transformers, "__version__", "")),
                "torch_version": str(getattr(torch, "__version__", "")),
                "glm_moe_dsa_config_class_available": True,
                "glm_moe_dsa_model_class_available": True,
                "model_class": "GlmMoeDsaForCausalLM",
            }
        )
        try:
            original = GlmMoeDsaConfig(**dict(config))
            result["awq_config_original_ready"] = True
            result["original_model_type"] = str(original.model_type)
        except Exception as exc:
            result["awq_config_original_error_type"] = type(exc).__name__
            result["awq_config_original_error_digest"] = sha_payload(str(exc))
            result["blockers"].append("glm52_awq_config_original_not_transformers_loadable")

        normalized, action = normalize_awq_config(config)
        result["awq_config_normalization_action"] = action
        try:
            normalized_config = GlmMoeDsaConfig(**normalized)
            result["awq_config_normalized_ready"] = True
            result["normalized_model_type"] = str(normalized_config.model_type)
            result["normalized_layer_count"] = _int(getattr(normalized_config, "num_hidden_layers", 0))
            result["normalized_mlp_layer_type_count"] = len(getattr(normalized_config, "mlp_layer_types", []) or [])
            result["normalized_indexer_type_count"] = len(getattr(normalized_config, "indexer_types", []) or [])
        except Exception as exc:
            result["awq_config_normalized_error_type"] = type(exc).__name__
            result["awq_config_normalized_error_digest"] = sha_payload(str(exc))
            result["blockers"].append("glm52_awq_config_normalized_not_transformers_loadable")

        try:
            tiny = GlmMoeDsaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=8,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=2,
                n_routed_experts=4,
                num_experts_per_tok=2,
                n_shared_experts=1,
                kv_lora_rank=4,
                q_lora_rank=4,
                qk_rope_head_dim=4,
                qk_nope_head_dim=4,
                v_head_dim=4,
                index_head_dim=4,
                index_n_heads=2,
                index_topk=2,
                max_position_embeddings=16,
                mlp_layer_types=["dense", "dense", "dense", "sparse"],
                indexer_types=["full", "full", "full", "full"],
            )
            model = GlmMoeDsaForCausalLM(tiny).eval()
            with torch.no_grad():
                output = model(input_ids=torch.tensor([[1]], dtype=torch.long), use_cache=False)
            result["tiny_forward_ready"] = True
            result["tiny_forward_logits_shape"] = [int(item) for item in output.logits.shape]
            result["tiny_forward_output_hash"] = sha_payload(
                {
                    "shape": result["tiny_forward_logits_shape"],
                    "sum_rounded": round(float(output.logits.detach().float().sum().item()), 6),
                }
            )
        except Exception as exc:
            result["tiny_forward_error_type"] = type(exc).__name__
            result["tiny_forward_error_digest"] = sha_payload(str(exc))
            result["blockers"].append("glm52_tiny_transformers_forward_failed")
    except Exception as exc:
        result["transformers_error_type"] = type(exc).__name__
        result["transformers_error_digest"] = sha_payload(str(exc))
        result["blockers"].append("glm52_transformers_glm_moe_dsa_unavailable")
    return result


def stage_layer_ranges(layer_count: int, stage_count: int) -> list[tuple[int, int]]:
    count = max(1, int(stage_count))
    base = int(layer_count) // count
    remainder = int(layer_count) % count
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for stage_id in range(count):
        width = base + (1 if stage_id < remainder else 0)
        ranges.append((cursor, cursor + width))
        cursor += width
    return ranges


def pack_keys(prefix: str) -> list[str]:
    return [f"{prefix}.{field}" for field in PACK_FIELDS]


def layer_expected_keys(config: dict[str, Any], layer_id: int) -> dict[str, list[str]]:
    mlp_types = [str(item) for item in _list(config.get("mlp_layer_types"))]
    indexer_types = [str(item) for item in _list(config.get("indexer_types"))]
    layer_prefix = f"model.layers.{layer_id}"
    mlp_type = mlp_types[layer_id] if layer_id < len(mlp_types) else ("dense" if layer_id < 3 else "sparse")
    indexer_type = indexer_types[layer_id] if layer_id < len(indexer_types) else "full"
    common = [
        f"{layer_prefix}.input_layernorm.weight",
        f"{layer_prefix}.post_attention_layernorm.weight",
        f"{layer_prefix}.self_attn.q_a_proj.weight",
        f"{layer_prefix}.self_attn.q_a_layernorm.weight",
        f"{layer_prefix}.self_attn.kv_a_proj_with_mqa.weight",
        f"{layer_prefix}.self_attn.kv_a_layernorm.weight",
    ]
    if mlp_type == "sparse":
        attention_pack = []
        for name in ["q_b_proj", "kv_b_proj", "o_proj"]:
            attention_pack.extend(pack_keys(f"{layer_prefix}.self_attn.{name}"))
        mlp_direct = [
            f"{layer_prefix}.mlp.gate.weight",
            f"{layer_prefix}.mlp.gate.e_score_correction_bias",
            f"{layer_prefix}.mlp.shared_experts.gate_proj.weight",
            f"{layer_prefix}.mlp.shared_experts.up_proj.weight",
            f"{layer_prefix}.mlp.shared_experts.down_proj.weight",
        ]
        expert_pack = []
        expert_count = _int(config.get("n_routed_experts"), 0)
        for expert_id in range(expert_count):
            for projection in ["gate_proj", "up_proj", "down_proj"]:
                expert_pack.extend(pack_keys(f"{layer_prefix}.mlp.experts.{expert_id}.{projection}"))
    else:
        attention_pack = []
        common.extend(
            [
                f"{layer_prefix}.self_attn.q_b_proj.weight",
                f"{layer_prefix}.self_attn.kv_b_proj.weight",
                f"{layer_prefix}.self_attn.o_proj.weight",
            ]
        )
        mlp_direct = [
            f"{layer_prefix}.mlp.gate_proj.weight",
            f"{layer_prefix}.mlp.up_proj.weight",
            f"{layer_prefix}.mlp.down_proj.weight",
        ]
        expert_pack = []
    indexer_direct = []
    if indexer_type == "full":
        indexer_direct = [
            f"{layer_prefix}.self_attn.indexer.wq_b.weight",
            f"{layer_prefix}.self_attn.indexer.wk.weight",
            f"{layer_prefix}.self_attn.indexer.k_norm.weight",
            f"{layer_prefix}.self_attn.indexer.k_norm.bias",
            f"{layer_prefix}.self_attn.indexer.weights_proj.weight",
        ]
    return {
        "common_direct": common,
        "indexer_direct": indexer_direct,
        "attention_pack": attention_pack,
        "mlp_direct": mlp_direct,
        "expert_pack": expert_pack,
    }


def mapping_summary(config: dict[str, Any], index: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    weight_map = {str(key): str(value) for key, value in _dict(index.get("weight_map")).items()}
    layer_count = _int(config.get("num_hidden_layers"))
    ranges = stage_layer_ranges(layer_count, int(args.stage_count))
    if int(args.stage_id) >= 0:
        selected_ranges = [ranges[int(args.stage_id)]]
        selected_stage_ids = [int(args.stage_id)]
    else:
        selected_ranges = [(0, layer_count)]
        selected_stage_ids = [-1]
    mlp_types = [str(item) for item in _list(config.get("mlp_layer_types"))]
    indexer_types = [str(item) for item in _list(config.get("indexer_types"))]
    missing_keys: list[str] = []
    direct_required = 0
    pack_required = 0
    dense_layers = 0
    sparse_layers = 0
    full_indexer_layers = 0
    shared_indexer_layers = 0
    layer_summaries: list[dict[str, Any]] = []
    for stage_id, layer_range in zip(selected_stage_ids, selected_ranges, strict=False):
        for layer_id in range(layer_range[0], layer_range[1]):
            expected = layer_expected_keys(config, layer_id)
            direct = expected["common_direct"] + expected["indexer_direct"] + expected["mlp_direct"]
            pack = expected["attention_pack"] + expected["expert_pack"]
            layer_missing = [key for key in [*direct, *pack] if key not in weight_map]
            missing_keys.extend(layer_missing)
            direct_required += len(direct)
            pack_required += len(pack)
            mlp_type = mlp_types[layer_id] if layer_id < len(mlp_types) else ("dense" if layer_id < 3 else "sparse")
            indexer_type = indexer_types[layer_id] if layer_id < len(indexer_types) else "full"
            dense_layers += 1 if mlp_type != "sparse" else 0
            sparse_layers += 1 if mlp_type == "sparse" else 0
            full_indexer_layers += 1 if indexer_type == "full" else 0
            shared_indexer_layers += 1 if indexer_type == "shared" else 0
            if len(layer_summaries) < int(args.layer_summary_limit):
                layer_summaries.append(
                    {
                        "stage_id": stage_id,
                        "layer_id": layer_id,
                        "mlp_layer_type": mlp_type,
                        "indexer_type": indexer_type,
                        "direct_key_count": len(direct),
                        "pack_key_count": len(pack),
                        "missing_key_count": len(layer_missing),
                        "missing_key_digest": sha_payload(layer_missing[:50]),
                    }
                )
    present_required = direct_required + pack_required - len(missing_keys)
    return {
        "stage_id": int(args.stage_id),
        "stage_count": int(args.stage_count),
        "selected_stage_ranges": [[int(a), int(b)] for a, b in selected_ranges],
        "selected_layer_count": sum(b - a for a, b in selected_ranges),
        "dense_layer_count": dense_layers,
        "sparse_layer_count": sparse_layers,
        "full_indexer_layer_count": full_indexer_layers,
        "shared_indexer_layer_count": shared_indexer_layers,
        "direct_required_key_count": direct_required,
        "pack_required_key_count": pack_required,
        "required_key_count": direct_required + pack_required,
        "present_required_key_count": present_required,
        "missing_required_key_count": len(missing_keys),
        "missing_required_key_digest": sha_payload(missing_keys[:200]),
        "weight_key_count": len(weight_map),
        "stage_weight_mapping_ready": bool(weight_map and not missing_keys and direct_required + pack_required > 0),
        "layer_summaries": layer_summaries,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: set[str] = set()
    errors: list[dict[str, Any]] = []
    try:
        config = fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
        normalized_config, normalization_action = normalize_awq_config(config)
    except Exception as exc:
        config = {}
        normalized_config = {}
        normalization_action = "config_fetch_failed"
        blockers.add("glm52_awq_config_fetch_failed")
        errors.append({"phase": "config", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})
    try:
        index = fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    except Exception as exc:
        index = {}
        blockers.add("glm52_awq_index_fetch_failed")
        errors.append({"phase": "index", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})

    transformer = transformers_status(config)
    for item in _list(transformer.get("blockers")):
        blockers.add(str(item))
    mapping = mapping_summary(normalized_config or config, index, args) if config and index else {
        "stage_weight_mapping_ready": False,
        "missing_required_key_count": 0,
    }
    if mapping.get("stage_weight_mapping_ready") is not True:
        blockers.add("glm52_stage_weight_mapping_not_ready")

    dependency_statuses = [import_status(module) for module in PACK_RUNTIME_MODULES]
    pack_runtime_ready = any(item.get("available") is True for item in dependency_statuses)
    if not pack_runtime_ready:
        blockers.add("glm52_pack_quantized_runtime_dependency_missing")

    quantization = _dict(config.get("quantization_config"))
    quant_groups = _dict(quantization.get("config_groups"))
    quant_weight_bits = [
        _int(_dict(_dict(group).get("weights")).get("num_bits"))
        for group in quant_groups.values()
        if isinstance(group, dict)
    ]
    if "quant" not in str(quantization.get("format") or "").lower() or 4 not in quant_weight_bits:
        blockers.add("glm52_awq_pack_quantized_config_not_ready")

    foundation_ready = bool(
        transformer.get("glm_moe_dsa_model_class_available") is True
        and transformer.get("awq_config_normalized_ready") is True
        and transformer.get("tiny_forward_ready") is True
        and mapping.get("stage_weight_mapping_ready") is True
    )
    if not foundation_ready:
        blockers.add("glm52_transformers_decode_adapter_foundation_not_ready")
    blockers.add("glm52_transformers_preflight_is_not_full_decode")
    decode_adapter_ready = bool(foundation_ready and pack_runtime_ready and False)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "glm52_transformers_decode_adapter_preflight_ready": True,
        "decode_adapter_ready": decode_adapter_ready,
        "adapter_foundation_ready": foundation_ready,
        "model": {
            "model_id": MODEL_ID,
            "model_repo": str(args.model_repo),
            "config_ready": bool(config),
            "index_ready": bool(index),
            "model_type": str((normalized_config or config).get("model_type") or ""),
            "architecture": [str(item) for item in _list((normalized_config or config).get("architectures"))],
            "num_hidden_layers": _int((normalized_config or config).get("num_hidden_layers")),
            "n_routed_experts": _int((normalized_config or config).get("n_routed_experts")),
            "num_experts_per_tok": _int((normalized_config or config).get("num_experts_per_tok")),
            "normalization_action": normalization_action,
            "quantization_format": str(quantization.get("format") or ""),
            "quantization_weight_bits": sorted(set(bit for bit in quant_weight_bits if bit)),
            "quantization_ignore_count": len(_list(quantization.get("ignore"))),
        },
        "transformers_runtime": transformer,
        "pack_quantized_runtime": {
            "ready": pack_runtime_ready,
            "dependencies": dependency_statuses,
        },
        "stage_weight_mapping": mapping,
        "errors": errors,
        "blockers": [] if decode_adapter_ready else sorted(blockers),
        "completion_boundary": {
            "preflight_is_not_decode_success": True,
            "tiny_random_forward_is_not_glm52_inference": True,
            "weight_mapping_is_not_weight_loading": True,
            "requires_pack_quantized_dequant_runtime": True,
            "requires_stage_decode_verified": True,
            "requires_same_request_generated_token_hash": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["decode_adapter_ready"] = False
        report["adapter_foundation_ready"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--stage-id", type=int, default=-1, help="-1 means all layers")
    parser.add_argument("--stage-count", type=int, default=3)
    parser.add_argument("--layer-summary-limit", type=int, default=12)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.stage_count <= 0:
        raise SystemExit("--stage-count must be positive")
    if args.stage_id >= args.stage_count:
        raise SystemExit("--stage-id must be -1 or less than --stage-count")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_transformers_decode_adapter_preflight.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Adapter foundation ready: {report.get('adapter_foundation_ready')}")
        print(f"Decode adapter ready: {report.get('decode_adapter_ready')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
