#!/usr/bin/env python3
"""Measure the gap between GLM 5.2 stage runtime evidence and full decode.

This probe is intentionally a blocker artifact until a real GLM 5.2 decode
adapter exists. It consumes public model metadata plus optional stage and
same-request reports, then records the concrete capabilities that must be
implemented before the Kaggle CPU/GPU/TPU path can count as inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_decode_adapter_gap_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-decode-adapter-gap-probe"
MODEL_ID = "zai-org/GLM-5.2"
DEFAULT_MODEL_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
REQUIRED_PROVIDERS = ["kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"]
REQUIRED_CAPABILITIES = [
    "glm_moe_dsa_transformer_block_runtime",
    "glm_moe_dsa_attention_q_lora_kv_lora_rope_nope",
    "glm_moe_dsa_dense_and_moe_mlp_runtime",
    "glm_moe_dsa_topk_router_and_expert_gather",
    "awq_int4_dequant_linear_runtime",
    "stage_activation_handoff_runtime",
    "stage_local_kv_cache_runtime",
    "lm_head_logits_token_selection_runtime",
    "coordinator_same_request_decode_runtime",
]
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


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
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


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def fetch_hf_json(repo: str, filename: str, *, timeout_seconds: float) -> dict[str, Any]:
    quoted = urllib.parse.quote(filename)
    request = urllib.request.Request(
        f"https://huggingface.co/{repo}/resolve/main/{quoted}",
        headers={"User-Agent": "crowdtensor-glm52-decode-gap-probe/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def model_metadata(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    try:
        config = fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    except Exception as exc:
        config = {}
        errors.append({"phase": "config", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})
    try:
        index = fetch_hf_json(
            args.model_repo,
            "model.safetensors.index.json",
            timeout_seconds=float(args.hf_timeout_seconds),
        )
    except Exception as exc:
        index = {}
        errors.append({"phase": "index", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})

    weight_map = _dict(index.get("weight_map"))
    key_sample = sorted(str(key) for key in weight_map)[:400]
    quantization = _dict(config.get("quantization_config"))
    quant_groups = _dict(quantization.get("config_groups"))
    quant_weight_bits = [
        _int(_dict(_dict(group).get("weights")).get("num_bits"))
        for group in quant_groups.values()
        if isinstance(group, dict)
    ]
    family_hits = {
        "attention_low_rank": any("q_a_proj" in key or "q_b_proj" in key or "kv_a_proj" in key or "kv_b_proj" in key for key in key_sample),
        "rope_nope_attention": _int(config.get("qk_rope_head_dim")) > 0 and _int(config.get("qk_nope_head_dim")) > 0,
        "dense_mlp": any(".mlp.gate_proj." in key or ".mlp.up_proj." in key for key in key_sample),
        "moe_experts": any(".mlp.experts." in str(key) for key in weight_map),
        "moe_router": any(".mlp.gate." in str(key) or ".mlp.router." in str(key) for key in weight_map),
        "awq_int4_tensors": bool(
            "quant" in str(quantization.get("format") or "").lower()
            and (4 in quant_weight_bits or not quant_weight_bits)
        )
        or any(any(fragment in str(key) for fragment in ["qweight", "qzeros", "scales", "g_idx"]) for key in weight_map),
        "lm_head": any(str(key).startswith("lm_head.") for key in weight_map),
    }
    return {
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "config_ready": bool(config),
        "index_ready": bool(index),
        "model_type": str(config.get("model_type") or ""),
        "architectures": [str(item) for item in _list(config.get("architectures"))],
        "num_hidden_layers": _int(config.get("num_hidden_layers")),
        "hidden_size": _int(config.get("hidden_size")),
        "num_attention_heads": _int(config.get("num_attention_heads")),
        "num_key_value_heads": _int(config.get("num_key_value_heads")),
        "q_lora_rank": _int(config.get("q_lora_rank")),
        "kv_lora_rank": _int(config.get("kv_lora_rank")),
        "qk_rope_head_dim": _int(config.get("qk_rope_head_dim")),
        "qk_nope_head_dim": _int(config.get("qk_nope_head_dim")),
        "v_head_dim": _int(config.get("v_head_dim")),
        "first_k_dense_replace": _int(config.get("first_k_dense_replace")),
        "n_routed_experts": _int(config.get("n_routed_experts")),
        "num_experts_per_tok": _int(config.get("num_experts_per_tok")),
        "moe_intermediate_size": _int(config.get("moe_intermediate_size")),
        "quantization_format": str(quantization.get("format") or ""),
        "quantization_weight_bits": sorted(set(bit for bit in quant_weight_bits if bit)),
        "weight_key_count": len(weight_map),
        "total_weight_size_bytes": _int(_dict(index.get("metadata")).get("total_size")),
        "total_weight_size_gb": round(_int(_dict(index.get("metadata")).get("total_size")) / 1_000_000_000, 6),
        "family_hits": family_hits,
        "metadata_errors": errors,
    }


def stage_summary(report: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    provider = str(report.get("provider") or report.get("backend") or report.get("stage_provider") or "")
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    return {
        "provider": provider,
        "stage_id": _int(report.get("stage_id"), ordinal),
        "stage_layer_range": _list(report.get("stage_layer_range")),
        "model_id": str(report.get("model_id") or _dict(report.get("model")).get("model_id") or ""),
        "stage_runtime_kind": str(report.get("stage_runtime_kind") or ""),
        "stage_execution_verified": report.get("stage_execution_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
        "stage_output_hash_present": _hash_ok(report.get("stage_output_hash") or report.get("output_hash")),
        "weight_tensor_values_loaded": report.get("stage_owned_weight_values_loaded") is True
        or report.get("weight_tensor_values_loaded") is True,
        "live_run_performed": report.get("live_run_performed") is True,
        "stage_smoke_only": report.get("stage_smoke_only") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "blockers": blockers,
    }


def same_request_summary(report: dict[str, Any]) -> dict[str, Any]:
    success = _dict(report.get("success"))
    same = _dict(report.get("same_request"))
    generated_count = _int(success.get("generated_token_count") or report.get("generated_token_count"))
    generated_hash = str(success.get("generated_token_hash") or report.get("generated_token_hash") or "")
    return {
        "present": bool(report),
        "same_request_decode_verified": report.get("same_request_decode_verified") is True
        or report.get("glm52_kaggle_same_request_verified") is True
        or success.get("same_request_decode_verified") is True,
        "model_id": str(report.get("model_id") or _dict(report.get("model")).get("model_id") or same.get("model_id") or ""),
        "generated_token_count": generated_count,
        "generated_token_hash_present": _hash_ok(generated_hash),
        "accepted_providers": [str(item) for item in _list(success.get("accepted_providers") or report.get("accepted_providers"))],
        "coordinator_request_verified": same.get("coordinator_request_verified") is True,
        "live_run_performed": report.get("live_run_performed") is True,
        "cleanup_verified": _dict(report.get("cleanup")).get("temporary_kaggle_kernels_deleted") is True
        and _dict(report.get("cleanup")).get("temporary_private_packages_removed") is True
        and _dict(report.get("cleanup")).get("live_resources_left_running") is False,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True if report else True,
    }


def activation_handoff_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "present": False,
            "stage_activation_handoff_runtime_verified": False,
            "stage_activation_handoff_contract_verified": False,
            "handoff_count": 0,
            "stage_runtime_provider_coverage": [],
            "same_request_decode_verified": False,
            "blockers": [],
            "public_artifact_safe": True,
        }
    return {
        "present": True,
        "schema": str(report.get("schema") or ""),
        "ok": report.get("ok") is True,
        "stage_activation_handoff_runtime_verified": report.get("stage_activation_handoff_runtime_verified") is True,
        "stage_activation_handoff_contract_verified": report.get("stage_activation_handoff_contract_verified") is True,
        "handoff_count": _int(report.get("handoff_count")),
        "stage_runtime_provider_coverage": [str(item) for item in _list(report.get("stage_runtime_provider_coverage"))],
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def component_report_summary(name: str, report: dict[str, Any], required_true_fields: list[str]) -> dict[str, Any]:
    if not report:
        return {
            "name": name,
            "present": False,
            "verified": False,
            "schema": "",
            "model_id": "",
            "required_true_fields": required_true_fields,
            "missing_true_fields": required_true_fields,
            "blockers": [f"{name}_report_missing"],
            "public_artifact_safe": True,
        }
    missing = [field for field in required_true_fields if report.get(field) is not True]
    verified = bool(
        report.get("ok") is True
        and str(report.get("model_id") or "") == MODEL_ID
        and report.get("public_artifact_safe") is True
        and not missing
    )
    return {
        "name": name,
        "present": True,
        "verified": verified,
        "schema": str(report.get("schema") or ""),
        "ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "required_true_fields": required_true_fields,
        "missing_true_fields": missing,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "generated_token_verified": report.get("generated_token_verified") is True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def component_capability_evidence(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
    reports = {
        "attention_projection": load_json(args.attention_projection_report),
        "attention_single_token": load_json(args.attention_single_token_report),
        "kv_cache_decode": load_json(args.kv_cache_decode_report),
        "dsa_masked_layer_decode": load_json(args.dsa_masked_layer_decode_report),
        "lm_head_token": load_json(args.lm_head_token_report),
        "stage_hidden_lm_head": load_json(args.stage_hidden_lm_head_report),
        "pack_quantized_dequant": load_json(args.pack_quantized_dequant_report),
        "pack_quantized_expert_mlp": load_json(args.pack_quantized_expert_mlp_report),
        "pack_quantized_router_gather": load_json(args.pack_quantized_router_gather_report),
        "pack_quantized_moe_mlp": load_json(args.pack_quantized_moe_mlp_report),
    }
    components = {
        "attention_projection": component_report_summary(
            "attention_projection",
            reports["attention_projection"],
            ["attention_projection_verified", "input_layernorm_verified", "q_lora_projection_verified", "kv_lora_projection_verified"],
        ),
        "attention_single_token": component_report_summary(
            "attention_single_token",
            reports["attention_single_token"],
            ["single_token_attention_verified", "rope_applied", "attention_scores_verified", "attention_weights_verified", "o_proj_verified"],
        ),
        "kv_cache_decode": component_report_summary(
            "kv_cache_decode",
            reports["kv_cache_decode"],
            ["kv_cache_prefill_verified", "kv_cache_update_verified", "kv_cache_decode_attention_verified", "o_proj_verified"],
        ),
        "dsa_masked_layer_decode": component_report_summary(
            "dsa_masked_layer_decode",
            reports["dsa_masked_layer_decode"],
            ["layer_decode_verified", "attention_decode_verified", "dsa_masked_attention_integrated", "dsa_indexer_verified", "full_moe_mlp_verified", "kv_cache_prefill_verified", "kv_cache_update_verified"],
        ),
        "lm_head_token": component_report_summary(
            "lm_head_token",
            reports["lm_head_token"],
            ["lm_head_logits_token_selection_verified", "lm_head_streamed_full_vocab"],
        ),
        "stage_hidden_lm_head": component_report_summary(
            "stage_hidden_lm_head",
            reports["stage_hidden_lm_head"],
            ["stage_hidden_lm_head_token_selection_verified", "stage_hidden_to_lm_head_verified", "stage_layer_decode_verified", "lm_head_streamed_full_vocab"],
        ),
        "pack_quantized_dequant": component_report_summary(
            "pack_quantized_dequant",
            reports["pack_quantized_dequant"],
            ["pack_quantized_dequant_verified", "pack_quantized_linear_slice_verified"],
        ),
        "pack_quantized_expert_mlp": component_report_summary(
            "pack_quantized_expert_mlp",
            reports["pack_quantized_expert_mlp"],
            ["pack_quantized_expert_mlp_verified", "single_expert_mlp_verified"],
        ),
        "pack_quantized_router_gather": component_report_summary(
            "pack_quantized_router_gather",
            reports["pack_quantized_router_gather"],
            ["router_topk_verified", "routed_expert_subset_verified"],
        ),
        "pack_quantized_moe_mlp": component_report_summary(
            "pack_quantized_moe_mlp",
            reports["pack_quantized_moe_mlp"],
            ["full_moe_mlp_verified", "shared_experts_mlp_verified", "router_topk_verified", "routed_expert_gather_verified"],
        ),
    }
    experts_per_token = max(1, _int(metadata.get("num_experts_per_tok"), 1))
    router_expert_count = _int(reports["pack_quantized_router_gather"].get("executed_expert_count"))
    moe_expert_count = _int(reports["pack_quantized_moe_mlp"].get("executed_expert_count"))
    if router_expert_count < experts_per_token:
        components["pack_quantized_router_gather"]["verified"] = False
        components["pack_quantized_router_gather"]["missing_true_fields"].append("executed_expert_count>=num_experts_per_tok")
    if moe_expert_count < experts_per_token:
        components["pack_quantized_moe_mlp"]["verified"] = False
        components["pack_quantized_moe_mlp"]["missing_true_fields"].append("executed_expert_count>=num_experts_per_tok")

    def cap(verified: bool, evidence: list[str]) -> dict[str, Any]:
        return {
            "verified": bool(verified),
            "evidence": evidence if verified else [],
            "public_artifact_safe": all(components[name].get("public_artifact_safe") is True for name in evidence),
        }

    evidence = {
        "glm_moe_dsa_attention_q_lora_kv_lora_rope_nope": cap(
            components["attention_projection"]["verified"]
            and components["attention_single_token"]["verified"]
            and components["dsa_masked_layer_decode"]["verified"],
            ["attention_projection", "attention_single_token", "dsa_masked_layer_decode"],
        ),
        "glm_moe_dsa_dense_and_moe_mlp_runtime": cap(
            components["pack_quantized_expert_mlp"]["verified"]
            and components["pack_quantized_moe_mlp"]["verified"],
            ["pack_quantized_expert_mlp", "pack_quantized_moe_mlp"],
        ),
        "glm_moe_dsa_topk_router_and_expert_gather": cap(
            components["pack_quantized_router_gather"]["verified"]
            and components["pack_quantized_moe_mlp"]["verified"],
            ["pack_quantized_router_gather", "pack_quantized_moe_mlp"],
        ),
        "awq_int4_dequant_linear_runtime": cap(
            components["pack_quantized_dequant"]["verified"],
            ["pack_quantized_dequant"],
        ),
        "stage_local_kv_cache_runtime": cap(
            components["kv_cache_decode"]["verified"]
            and components["dsa_masked_layer_decode"]["verified"],
            ["kv_cache_decode", "dsa_masked_layer_decode"],
        ),
        "lm_head_logits_token_selection_runtime": cap(
            components["lm_head_token"]["verified"]
            and components["stage_hidden_lm_head"]["verified"],
            ["lm_head_token", "stage_hidden_lm_head"],
        ),
        "glm_moe_dsa_transformer_block_runtime": cap(
            components["dsa_masked_layer_decode"]["verified"]
            and components["pack_quantized_moe_mlp"]["verified"]
            and components["kv_cache_decode"]["verified"],
            ["dsa_masked_layer_decode", "pack_quantized_moe_mlp", "kv_cache_decode"],
        ),
    }
    return {
        "present": any(component.get("present") for component in components.values()),
        "components": components,
        "capabilities": evidence,
        "completion_boundary": {
            "component_runtime_evidence_is_not_stage_decode": True,
            "component_runtime_evidence_is_not_same_request_decode": True,
            "component_runtime_evidence_is_not_generated_token": True,
        },
        "public_artifact_safe": all(component.get("public_artifact_safe") is True for component in components.values() if component.get("present")),
    }


def required_capability_status(
    metadata: dict[str, Any],
    stages: list[dict[str, Any]],
    same: dict[str, Any],
    activation_handoff: dict[str, Any],
    component_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    providers_with_decode = {str(stage.get("provider")) for stage in stages if stage.get("stage_decode_verified") is True}
    same_verified = same.get("same_request_decode_verified") is True
    config_ready = metadata.get("config_ready") is True and metadata.get("index_ready") is True
    full_decode_verified = bool(same_verified and config_ready and set(REQUIRED_PROVIDERS).issubset(providers_with_decode))
    activation_handoff_verified = bool(
        config_ready
        and activation_handoff.get("stage_activation_handoff_runtime_verified") is True
        and set(REQUIRED_PROVIDERS).issubset(set(str(item) for item in _list(activation_handoff.get("stage_runtime_provider_coverage"))))
        and activation_handoff.get("public_artifact_safe") is True
    )
    statuses: list[dict[str, Any]] = []
    for capability in REQUIRED_CAPABILITIES:
        verified = full_decode_verified
        evidence = "same_request_decode" if full_decode_verified else ""
        if capability == "stage_activation_handoff_runtime" and activation_handoff_verified:
            verified = True
            evidence = "stage_activation_handoff_probe"
        component_status = _dict(_dict(component_evidence.get("capabilities")).get(capability))
        if capability != "coordinator_same_request_decode_runtime" and component_status.get("verified") is True:
            verified = True
            evidence = "component_runtime_probe:" + ",".join(str(item) for item in _list(component_status.get("evidence")))
        statuses.append(
            {
                "capability": capability,
                "required": True,
                "verified": verified,
                "evidence": evidence,
            }
        )
    return statuses


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    metadata = model_metadata(args)
    stage_reports = [load_json(path) for path in args.stage_report]
    stages = [stage_summary(report, ordinal=index) for index, report in enumerate(stage_reports) if report]
    same = same_request_summary(load_json(args.same_request_report))
    activation_handoff = activation_handoff_summary(load_json(args.activation_handoff_report))
    component_evidence = component_capability_evidence(args, metadata)
    capability_status = required_capability_status(metadata, stages, same, activation_handoff, component_evidence)

    blockers: set[str] = set()
    if metadata.get("config_ready") is not True or metadata.get("index_ready") is not True:
        blockers.add("glm52_decode_model_metadata_not_ready")
    if metadata.get("model_type") != "glm_moe_dsa":
        blockers.add("glm52_decode_model_type_not_glm_moe_dsa")
    if metadata.get("n_routed_experts", 0) <= 0 or metadata.get("num_experts_per_tok", 0) <= 0:
        blockers.add("glm52_decode_moe_config_not_observed")
    if metadata.get("weight_key_count", 0) <= 0:
        blockers.add("glm52_decode_weight_index_not_observed")

    providers = {stage["provider"] for stage in stages if stage.get("stage_execution_verified") is True}
    providers_with_decode = {stage["provider"] for stage in stages if stage.get("stage_decode_verified") is True}
    for provider in REQUIRED_PROVIDERS:
        if provider not in providers:
            blockers.add(f"glm52_stage_runtime_provider_missing:{provider}")
        if provider not in providers_with_decode:
            blockers.add(f"glm52_stage_decode_provider_missing:{provider}")
    for stage in stages:
        blockers.update(str(item) for item in _list(stage.get("blockers")) if item)
        if stage.get("stage_runtime_kind") == "glm52_awq_stage_value_provider_op":
            blockers.add("glm52_stage_value_provider_op_is_not_full_decode")
        if stage.get("stage_decode_verified") is not True:
            blockers.add("glm52_stage_decode_not_verified")
    if same.get("same_request_decode_verified") is not True:
        blockers.add("glm52_same_request_decode_not_verified")
    if same.get("generated_token_count", 0) < 1 or same.get("generated_token_hash_present") is not True:
        blockers.add("glm52_generated_token_not_verified")
    if activation_handoff.get("present") and activation_handoff.get("public_artifact_safe") is not True:
        blockers.add("glm52_stage_activation_handoff_public_artifact_unsafe")
    if activation_handoff.get("present") and activation_handoff.get("stage_activation_handoff_runtime_verified") is not True:
        blockers.add("glm52_stage_activation_handoff_not_verified")
    if component_evidence.get("present") and component_evidence.get("public_artifact_safe") is not True:
        blockers.add("glm52_component_capability_public_artifact_unsafe")

    for item in capability_status:
        if item.get("verified") is not True:
            blockers.add(f"glm52_decode_capability_missing:{item['capability']}")

    ready = bool(not blockers)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "glm52_decode_adapter_gap_probe_ready": True,
        "decode_adapter_ready": ready,
        "same_request_decode_ready": ready,
        "model": metadata,
        "stage_runtime_reports": stages,
        "same_request": same,
        "stage_activation_handoff": activation_handoff,
        "component_capability_evidence": component_evidence,
        "required_capabilities": capability_status,
        "required_provider_coverage": REQUIRED_PROVIDERS,
        "stage_runtime_provider_coverage": sorted(providers),
        "stage_decode_provider_coverage": sorted(providers_with_decode),
        "blockers": [] if ready else sorted(blockers),
        "completion_boundary": {
            "stage_runtime_value_op_is_not_decode": True,
            "requires_transformer_block_semantics": True,
            "requires_awq_dequant_linear_runtime": True,
            "requires_moe_router_and_expert_runtime": True,
            "requires_generated_token_hash": True,
            "requires_cpu_gpu_tpu_same_request": True,
            "stage_activation_handoff_evidence_is_not_same_request_success": True,
            "component_runtime_evidence_is_not_same_request_success": True,
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
        report["same_request_decode_ready"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--stage-report", action="append", default=[])
    parser.add_argument("--same-request-report", default="")
    parser.add_argument("--activation-handoff-report", default="")
    parser.add_argument("--attention-projection-report", default="")
    parser.add_argument("--attention-single-token-report", default="")
    parser.add_argument("--kv-cache-decode-report", default="")
    parser.add_argument("--dsa-masked-layer-decode-report", default="")
    parser.add_argument("--lm-head-token-report", default="")
    parser.add_argument("--stage-hidden-lm-head-report", default="")
    parser.add_argument("--pack-quantized-dequant-report", default="")
    parser.add_argument("--pack-quantized-expert-mlp-report", default="")
    parser.add_argument("--pack-quantized-router-gather-report", default="")
    parser.add_argument("--pack-quantized-moe-mlp-report", default="")
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_decode_adapter_gap_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Decode adapter ready: {report.get('decode_adapter_ready')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
