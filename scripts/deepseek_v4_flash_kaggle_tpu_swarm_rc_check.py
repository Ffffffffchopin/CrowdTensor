#!/usr/bin/env python3
"""Validate DeepSeek-V4-Flash Kaggle GPU+WebTPU+CPU swarm RC evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import deepseek_v4_flash_kaggle_tpu_swarm_rc_pack as pack  # noqa: E402


SCHEMA = "deepseek_v4_flash_kaggle_tpu_swarm_rc_check_v1"
REQUIRED_PROVIDERS = set(pack.REQUIRED_PROVIDERS)


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


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("deepseek_v4_flash_kaggle_tpu_swarm_rc_ready") is not True:
        errors.append("swarm_rc_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    model = _dict(report.get("model"))
    if model.get("model_id") != "deepseek-ai/DeepSeek-V4-Flash":
        errors.append("model_id_mismatch")
    if model.get("architecture_class") != "moe":
        errors.append("architecture_class_mismatch")

    source = _dict(report.get("source_resolver"))
    if source.get("present") is not True:
        errors.append("source_resolver_missing")
    if source.get("resolver_ready") is not True:
        errors.append("source_resolver_not_ready")
    if not source.get("recommended_repo"):
        errors.append("recommended_repo_missing")
    if not source.get("recommended_quant"):
        errors.append("recommended_quant_missing")

    success = _dict(report.get("success"))
    same = _dict(report.get("same_request"))
    success_verified = success.get("same_request_decode_verified") is True
    accepted = set(str(item) for item in _list(success.get("accepted_providers")))
    generated = _int(success.get("generated_token_count"))
    same_blockers = set(str(item) for item in _list(same.get("blockers")))
    report_blockers = set(str(item) for item in _list(report.get("blockers")))
    slice_only_three_provider_evidence = bool(
        same.get("deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified") is True
        and generated >= 1
        and REQUIRED_PROVIDERS.issubset(accepted)
        and "deepseek_v4_full_same_request_decode_not_verified" in same_blockers.union(report_blockers)
    )
    if _list(success.get("required_providers")) != pack.REQUIRED_PROVIDERS:
        errors.append("required_providers_mismatch")

    adapter = _dict(report.get("deepseek_tpu_adapter"))
    if adapter.get("present") is True:
        if adapter.get("public_artifact_safe") is not True:
            errors.append("deepseek_tpu_adapter_public_artifact_unsafe")
        if adapter.get("deepseek_v4_real_weight_tpu_tensor_load_ready") is True:
            if _int(adapter.get("real_weight_sample_loaded_tensor_count")) < 1:
                errors.append("adapter_real_weight_sample_count_missing")
            if _int(adapter.get("real_weight_sample_total_loaded_tensor_bytes")) <= 0:
                errors.append("adapter_real_weight_sample_bytes_missing")
            if adapter.get("real_weight_sample_values_public") is not False:
                errors.append("adapter_real_weight_sample_values_public_unsafe")
            dtype_counts = _dict(adapter.get("real_weight_sample_dtype_counts"))
            for dtype_name in ["BF16", "F8_E4M3", "F8_E8M0", "I8"]:
                if dtype_counts and _int(dtype_counts.get(dtype_name)) < 1:
                    errors.append(f"adapter_real_weight_sample_dtype_missing:{dtype_name}")
            if adapter.get("real_i8_expert_dequant_smoke_ready") is True and _int(dtype_counts.get("I8")) < 1:
                errors.append("adapter_i8_expert_smoke_without_i8_sample")
            if adapter.get("real_i8_expert_mlp_slice_smoke_ready") is True and _int(dtype_counts.get("I8")) < 3:
                errors.append("adapter_i8_expert_mlp_slice_without_w1_w2_w3_samples")
            if adapter.get("real_fp4_topk_expert_mlp_forward_ready") is True:
                if _int(adapter.get("real_fp4_topk_count")) < 1:
                    errors.append("adapter_fp4_topk_count_missing")
                if _int(adapter.get("real_fp4_topk_loaded_tensor_count")) < 1:
                    errors.append("adapter_fp4_topk_loaded_tensor_count_missing")
                if _int(adapter.get("real_fp4_topk_total_loaded_tensor_bytes")) <= 0:
                    errors.append("adapter_fp4_topk_loaded_tensor_bytes_missing")
                if _list(adapter.get("real_fp4_topk_final_output_shape")) != [4096]:
                    errors.append("adapter_fp4_topk_final_output_shape_mismatch")
                if not str(adapter.get("real_fp4_topk_final_output_hash") or "").startswith("sha256:"):
                    errors.append("adapter_fp4_topk_final_output_hash_missing")
                if adapter.get("real_fp4_topk_finite_output") is not True:
                    errors.append("adapter_fp4_topk_output_not_finite")
                if adapter.get("real_fp4_topk_weight_tensor_values_public") is not False:
                    errors.append("adapter_fp4_topk_weight_values_public_unsafe")
                if adapter.get("real_fp4_topk_activation_payload_public") is not False:
                    errors.append("adapter_fp4_topk_activation_public_unsafe")
        if adapter.get("metadata_ready") is True:
            if adapter.get("stage_key_mapping_ready") is not True:
                errors.append("adapter_metadata_without_stage_mapping")
            if _int(adapter.get("stage_selected_key_count")) < 1:
                errors.append("adapter_stage_keys_missing")
            family_hits = _dict(adapter.get("stage_family_hits"))
            for family in ["mla_attention", "moe_router", "shared_experts", "routed_experts"]:
                if family_hits.get(family) is not True:
                    errors.append(f"adapter_stage_family_missing:{family}")

    torch_smoke = _dict(report.get("deepseek_torch_stage_smoke"))
    if torch_smoke.get("present") is True:
        if torch_smoke.get("public_artifact_safe") is not True:
            errors.append("deepseek_torch_stage_smoke_public_artifact_unsafe")
        if torch_smoke.get("torch_stage_adapter_smoke_ready") is not True:
            errors.append("deepseek_torch_stage_smoke_not_ready")
        if torch_smoke.get("transformers_reference_used") is not True:
            errors.append("deepseek_torch_stage_smoke_reference_missing")
        if torch_smoke.get("real_deepseek_weights_loaded") is not False:
            errors.append("deepseek_torch_stage_smoke_weight_overclaim")
        if torch_smoke.get("jax_tpu_translation_ready") is not False:
            errors.append("deepseek_torch_stage_smoke_jax_tpu_overclaim")
        components = _dict(torch_smoke.get("components_exercised"))
        for component in ["manifold_hyper_connections", "compressed_attention", "mla_shared_kv_attention", "moe_router"]:
            if components.get(component) is not True:
                errors.append(f"deepseek_torch_stage_smoke_component_missing:{component}")

    jax_smoke = _dict(report.get("deepseek_jax_stage_smoke"))
    if jax_smoke.get("present") is True:
        if jax_smoke.get("public_artifact_safe") is not True:
            errors.append("deepseek_jax_stage_smoke_public_artifact_unsafe")
        if jax_smoke.get("numpy_reference_ready") is not True:
            errors.append("deepseek_jax_stage_smoke_numpy_reference_missing")
        if jax_smoke.get("real_deepseek_weights_loaded") is not False:
            errors.append("deepseek_jax_stage_smoke_weight_overclaim")
        if jax_smoke.get("deepseek_v4_jax_stage_forward_ready") is True and jax_smoke.get("jax_runtime_execution_ready") is not True:
            errors.append("deepseek_jax_stage_smoke_jax_forward_overclaim")
        if jax_smoke.get("deepseek_v4_jax_tpu_stage_forward_ready") is True and jax_smoke.get("tpu_runtime_ready") is not True:
            errors.append("deepseek_jax_stage_smoke_tpu_forward_overclaim")
        components = _dict(jax_smoke.get("components_exercised"))
        for component in ["manifold_hyper_connections", "mla_shared_kv_attention", "topk_moe_router", "shared_experts"]:
            if components.get(component) is not True:
                errors.append(f"deepseek_jax_stage_smoke_component_missing:{component}")

    safetensors_header = _dict(report.get("deepseek_safetensors_stage_header"))
    if safetensors_header.get("present") is True:
        if safetensors_header.get("public_artifact_safe") is not True:
            errors.append("deepseek_safetensors_stage_header_public_artifact_unsafe")
        if safetensors_header.get("real_weight_tensor_values_loaded") is not False:
            errors.append("deepseek_safetensors_stage_header_weight_value_overclaim")
        if safetensors_header.get("real_weight_tensor_values_public") is not False:
            errors.append("deepseek_safetensors_stage_header_weight_value_public_unsafe")
        if safetensors_header.get("safetensors_header_payload_public") is not False:
            errors.append("deepseek_safetensors_stage_header_payload_public_unsafe")
        if _int(safetensors_header.get("stage_selected_key_count")) < 1:
            errors.append("deepseek_safetensors_stage_header_keys_missing")
        if _int(safetensors_header.get("stage_selected_file_count")) < 1:
            errors.append("deepseek_safetensors_stage_header_files_missing")
        header_family_hits = _dict(safetensors_header.get("stage_family_hits"))
        for family in ["mla_attention", "moe_router", "shared_experts", "routed_experts", "norms"]:
            if header_family_hits.get(family) is not True:
                errors.append(f"deepseek_safetensors_stage_header_family_missing:{family}")
        if safetensors_header.get("safetensors_header_ready") is True:
            if safetensors_header.get("stage_header_shape_ready") is not True:
                errors.append("deepseek_safetensors_stage_header_shape_not_ready")
            if _int(safetensors_header.get("header_file_count")) != _int(safetensors_header.get("stage_selected_file_count")):
                errors.append("deepseek_safetensors_stage_header_file_count_mismatch")
            if _int(safetensors_header.get("header_fetch_error_count")) != 0:
                errors.append("deepseek_safetensors_stage_header_fetch_errors_present")
            if _int(safetensors_header.get("missing_header_key_count")) != 0:
                errors.append("deepseek_safetensors_stage_header_missing_keys_present")
            if not _dict(safetensors_header.get("dtype_counts")):
                errors.append("deepseek_safetensors_stage_header_dtype_counts_missing")
            if _int(safetensors_header.get("total_selected_tensor_storage_bytes")) <= 0:
                errors.append("deepseek_safetensors_stage_header_storage_bytes_missing")
        elif success_verified:
            errors.append("success_without_safetensors_stage_header_ready")
        elif "deepseek_v4_flash_safetensors_stage_header_not_ready" not in set(report.get("blockers") or []):
            errors.append("deepseek_safetensors_stage_header_not_ready_without_blocker")

    web_tpu = _dict(report.get("web_tpu_execution_channel"))
    if web_tpu.get("present") is True and web_tpu.get("public_artifact_safe") is not True:
        errors.append("web_tpu_execution_public_artifact_unsafe")

    cpu_fp4_topk = _dict(report.get("deepseek_cpu_fp4_topk_expert_forward"))
    if cpu_fp4_topk.get("present") is True:
        if cpu_fp4_topk.get("public_artifact_safe") is not True:
            errors.append("cpu_fp4_topk_public_artifact_unsafe")
        if cpu_fp4_topk.get("model_id") != "deepseek-ai/DeepSeek-V4-Flash":
            errors.append("cpu_fp4_topk_model_id_mismatch")
        if cpu_fp4_topk.get("stage_selective_fp4_topk_expert_forward_ready") is True:
            if _int(cpu_fp4_topk.get("topk")) < 1:
                errors.append("cpu_fp4_topk_count_missing")
            if _int(cpu_fp4_topk.get("loaded_tensor_count")) < 1:
                errors.append("cpu_fp4_topk_loaded_tensor_count_missing")
            if _int(cpu_fp4_topk.get("total_loaded_tensor_bytes")) <= 0:
                errors.append("cpu_fp4_topk_loaded_tensor_bytes_missing")
            if _list(cpu_fp4_topk.get("final_output_shape")) != [4096]:
                errors.append("cpu_fp4_topk_final_output_shape_mismatch")
            if not str(cpu_fp4_topk.get("final_output_hash") or "").startswith("sha256:"):
                errors.append("cpu_fp4_topk_final_output_hash_missing")
            if cpu_fp4_topk.get("finite_output") is not True:
                errors.append("cpu_fp4_topk_output_not_finite")
        elif success_verified:
            errors.append("success_without_cpu_fp4_topk_forward")

    if success_verified:
        if same.get("same_request_decode_verified") is not True:
            errors.append("success_without_same_request_proof")
        if generated < 1:
            errors.append("success_without_generated_token")
        if not REQUIRED_PROVIDERS.issubset(accepted):
            errors.append("success_missing_required_provider")
        if adapter.get("present") is not True or adapter.get("deepseek_v4_jax_tpu_stage_forward_ready") is not True:
            errors.append("success_without_deepseek_tpu_stage_forward")
        if report.get("failure_stage"):
            errors.append("success_with_failure_stage")
        if _list(report.get("blockers")):
            errors.append("success_with_blockers")
    else:
        if generated >= 1 and REQUIRED_PROVIDERS.issubset(accepted) and not slice_only_three_provider_evidence:
            errors.append("same_request_evidence_present_but_success_false")
        if same.get("deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified") is True:
            if generated < 1:
                errors.append("slice_evidence_without_generated_token")
            if not REQUIRED_PROVIDERS.issubset(accepted):
                errors.append("slice_evidence_missing_required_provider")
            if "deepseek_v4_full_same_request_decode_not_verified" not in same_blockers.union(report_blockers):
                errors.append("slice_evidence_without_full_decode_blocker")
            if same.get("deepseek_v4_distinct_backend_stage_layer_ranges_verified") is True:
                ranges = _dict(same.get("deepseek_v4_stage_layer_ranges"))
                if set(ranges) != {"cuda", "jax_tpu", "cpu"}:
                    errors.append("distinct_stage_ranges_missing_backend")
                if _int(same.get("deepseek_v4_stage_layer_coverage_count")) < 3:
                    errors.append("distinct_stage_ranges_coverage_too_small")
        if "deepseek_v4_flash_kaggle_tpu_same_request_decode_not_verified" not in set(report.get("blockers") or []):
            errors.append("failure_blocker_missing")
        if not str(report.get("failure_stage") or "").strip():
            errors.append("failure_stage_missing")

    safety = _dict(report.get("safety"))
    for flag in [
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "past_key_values_public",
        "credentials_public",
        "cookies_public",
        "jupyter_proxy_token_public",
        "private_runtime_state_public",
        "private_kaggle_payload_public",
        "weight_tensor_values_public",
    ]:
        if safety.get(flag) is not False:
            errors.append(f"safety_flag_mismatch:{flag}")
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_missing")

    artifacts = _dict(report.get("artifacts"))
    for name in ("summary_json", "support_bundle_json"):
        if _dict(artifacts.get(name)).get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return sorted(set(errors))


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    report = load_json(Path(args.report))
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": args.report,
        "same_request_decode_verified": _dict(report.get("success")).get("same_request_decode_verified") is True,
        "generated_token_count": _dict(report.get("success")).get("generated_token_count"),
        "accepted_providers": _dict(report.get("success")).get("accepted_providers") or [],
        "failure_stage": report.get("failure_stage"),
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DeepSeek-V4-Flash Kaggle GPU+WebTPU+CPU swarm RC evidence.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Check ok: {result['ok']}")
        print(f"Report: {result['report_path']}")
        if result["errors"]:
            print("Errors: " + ", ".join(result["errors"]))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
