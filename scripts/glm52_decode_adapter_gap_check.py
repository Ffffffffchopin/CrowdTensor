#!/usr/bin/env python3
"""Validate GLM 5.2 decode adapter gap reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_decode_adapter_gap_probe as probe  # noqa: E402


SCHEMA = "glm52_decode_adapter_gap_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_report(report: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("glm52_decode_adapter_gap_probe_ready") is not True:
        errors.append("probe_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    model = _dict(report.get("model"))
    if model.get("model_id") != probe.MODEL_ID:
        errors.append("model_id_mismatch")
    if model.get("config_ready") is not True:
        errors.append("model_config_not_ready")
    if model.get("index_ready") is not True:
        errors.append("model_index_not_ready")
    if model.get("model_type") != "glm_moe_dsa":
        errors.append("model_type_not_glm_moe_dsa")
    if _int(model.get("num_hidden_layers")) <= 0:
        errors.append("layer_count_missing")
    if _int(model.get("n_routed_experts")) <= 0:
        errors.append("moe_expert_count_missing")
    if _int(model.get("weight_key_count")) <= 0:
        errors.append("weight_key_count_missing")
    family_hits = _dict(model.get("family_hits"))
    for key in ["attention_low_rank", "rope_nope_attention", "moe_experts", "awq_int4_tensors", "lm_head"]:
        if family_hits.get(key) is not True:
            errors.append(f"model_family_missing:{key}")

    capabilities = [item for item in _list(report.get("required_capabilities")) if isinstance(item, dict)]
    capability_names = {str(item.get("capability") or "") for item in capabilities}
    for capability in probe.REQUIRED_CAPABILITIES:
        if capability not in capability_names:
            errors.append(f"required_capability_missing:{capability}")
    component_evidence = _dict(report.get("component_capability_evidence"))
    if component_evidence:
        if component_evidence.get("public_artifact_safe") is not True:
            errors.append("component_capability_evidence_public_artifact_unsafe")
        component_caps = _dict(component_evidence.get("capabilities"))
        for capability, item in component_caps.items():
            if capability not in probe.REQUIRED_CAPABILITIES:
                errors.append(f"component_capability_unknown:{capability}")
            if _dict(item).get("verified") is True and not _list(_dict(item).get("evidence")):
                errors.append(f"component_capability_verified_without_evidence:{capability}")
        component_boundary = _dict(component_evidence.get("completion_boundary"))
        for key in [
            "component_runtime_evidence_is_not_stage_decode",
            "component_runtime_evidence_is_not_same_request_decode",
            "component_runtime_evidence_is_not_generated_token",
        ]:
            if component_boundary.get(key) is not True:
                errors.append(f"component_capability_boundary_missing:{key}")
    if report.get("decode_adapter_ready") is True:
        for item in capabilities:
            if item.get("verified") is not True:
                errors.append(f"decode_ready_with_unverified_capability:{item.get('capability')}")
        if set(report.get("stage_decode_provider_coverage") or []) != set(probe.REQUIRED_PROVIDERS):
            errors.append("decode_ready_without_required_provider_coverage")
        same = _dict(report.get("same_request"))
        if same.get("same_request_decode_verified") is not True:
            errors.append("decode_ready_without_same_request_verified")
        if same.get("generated_token_count", 0) < 1 or same.get("generated_token_hash_present") is not True:
            errors.append("decode_ready_without_generated_token_hash")
        if report.get("blockers"):
            errors.append("decode_ready_with_blockers")
    else:
        if not report.get("blockers"):
            errors.append("not_ready_missing_blockers")
    if require_ready and report.get("decode_adapter_ready") is not True:
        errors.append("decode_adapter_not_ready")

    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "stage_runtime_value_op_is_not_decode",
        "requires_transformer_block_semantics",
        "requires_awq_dequant_linear_runtime",
        "requires_moe_router_and_expert_runtime",
        "requires_generated_token_hash",
        "requires_cpu_gpu_tpu_same_request",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")

    safety = _dict(report.get("safety"))
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_missing")
    for key in [
        "credentials_public",
        "cookies_public",
        "signed_url_public",
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "weight_tensor_values_public",
        "safetensors_header_payload_public",
    ]:
        if safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    return sorted(set(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_ready=bool(args.require_ready))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "decode_adapter_ready": report.get("decode_adapter_ready") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_decode_adapter_gap_check: ok={result['ok']} "
            f"errors={len(errors)} ready={result['decode_adapter_ready']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
