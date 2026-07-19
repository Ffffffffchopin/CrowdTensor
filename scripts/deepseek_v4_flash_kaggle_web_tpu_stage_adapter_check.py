#!/usr/bin/env python3
"""Validate DeepSeek-V4-Flash Kaggle Web TPU stage-adapter probe artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe as probe  # noqa: E402


SCHEMA = "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_check_v1"


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
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    for field in [
        "metadata_ready",
        "kaggle_web_tpu_runtime_ready",
        "deepseek_v4_jax_tpu_stage_forward_ready",
        "model",
        "stage_plan",
        "deepseek_metadata",
        "web_tpu_cell",
        "failure_stage",
        "blockers",
        "cleanup_status",
        "safety",
        "artifacts",
    ]:
        if field not in report:
            errors.append(f"required_field_missing:{field}")

    model = _dict(report.get("model"))
    if model.get("model_id") != "deepseek-ai/DeepSeek-V4-Flash":
        errors.append("model_id_mismatch")
    if model.get("architecture_class") != "moe":
        errors.append("architecture_class_mismatch")
    if model.get("expected_model_type") != "deepseek_v4":
        errors.append("expected_model_type_mismatch")

    cell = _dict(report.get("web_tpu_cell"))
    if cell.get("schema") != "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_cell_summary_v1":
        errors.append("cell_summary_schema_mismatch")
    metadata = _dict(report.get("deepseek_metadata"))
    config = _dict(metadata.get("model_config"))
    if report.get("metadata_ready") is True:
        if metadata.get("metadata_ready") is not True:
            errors.append("metadata_ready_without_metadata")
        if metadata.get("stage_key_mapping_ready") is not True:
            errors.append("metadata_ready_without_stage_key_mapping")
        if config.get("model_type") != "deepseek_v4":
            errors.append("deepseek_model_type_missing")
        if "DeepseekV4ForCausalLM" not in _list(config.get("architectures")):
            errors.append("deepseek_architecture_missing")
        if _int(config.get("num_hidden_layers")) < 1:
            errors.append("num_hidden_layers_missing")
        stage_mapping = _dict(metadata.get("stage_mapping"))
        if _int(stage_mapping.get("selected_key_count")) < 1:
            errors.append("stage_selected_keys_missing")
        family_hits = _dict(stage_mapping.get("family_hits"))
        for family in [
            "mla_attention",
            "moe_router",
            "shared_experts",
            "routed_experts",
            "hybrid_compression",
            "norms",
        ]:
            if family_hits.get(family) is not True:
                errors.append(f"stage_family_missing:{family}")

    if report.get("kaggle_web_tpu_runtime_ready") is True:
        if cell.get("tpu_runtime_ready") is not True:
            errors.append("web_tpu_ready_without_cell_tpu")
        if _int(cell.get("tpu_device_count")) < 1:
            errors.append("web_tpu_ready_without_tpu_device")
    fixture_ready = report.get("deepseek_v4_jax_tpu_fixture_stage_forward_ready") is True
    real_tensor_load_ready = report.get("deepseek_v4_real_weight_tpu_tensor_load_ready") is True
    if real_tensor_load_ready:
        real_load = _dict(cell.get("deepseek_v4_real_weight_tpu_tensor_load"))
        if _int(real_load.get("loaded_tensor_count")) < 1:
            errors.append("real_weight_tpu_tensor_load_count_missing")
        if _int(real_load.get("device_put_count")) < _int(real_load.get("loaded_tensor_count")):
            errors.append("real_weight_tpu_tensor_device_put_missing")
        if real_load.get("weight_tensor_values_public") is not False:
            errors.append("real_weight_tpu_tensor_values_public_unsafe")
        if real_load.get("real_weight_tensor_values_loaded") is not True:
            errors.append("real_weight_tpu_tensor_loaded_flag_missing")
        router = _dict(real_load.get("real_router_smoke"))
        if real_load.get("real_router_smoke_ready") is True:
            if router.get("ready") is not True:
                errors.append("real_weight_router_smoke_ready_mismatch")
            if router.get("weight_tensor_values_public") is not False:
                errors.append("real_weight_router_smoke_weight_public_unsafe")
            if router.get("activation_payload_public") is not False:
                errors.append("real_weight_router_smoke_activation_public_unsafe")
            if not str(router.get("topk_index_digest") or "").startswith("sha256:"):
                errors.append("real_weight_router_smoke_index_digest_missing")
            if not str(router.get("topk_value_hash") or "").startswith("sha256:"):
                errors.append("real_weight_router_smoke_value_hash_missing")
        fp8_smoke = _dict(real_load.get("real_fp8_block_dequant_smoke"))
        if real_load.get("real_fp8_block_dequant_smoke_ready") is True:
            if fp8_smoke.get("ready") is not True:
                errors.append("real_fp8_block_dequant_smoke_ready_mismatch")
            if fp8_smoke.get("weight_tensor_values_public") is not False:
                errors.append("real_fp8_block_dequant_weight_public_unsafe")
            if fp8_smoke.get("activation_payload_public") is not False:
                errors.append("real_fp8_block_dequant_activation_public_unsafe")
            if not str(fp8_smoke.get("output_hash") or "").startswith("sha256:"):
                errors.append("real_fp8_block_dequant_output_hash_missing")
            if _int(_list(fp8_smoke.get("weight_block_shape"))[0] if _list(fp8_smoke.get("weight_block_shape")) else 0) <= 0:
                errors.append("real_fp8_block_dequant_shape_missing")
        i8_expert_smoke = _dict(real_load.get("real_i8_expert_dequant_smoke"))
        if real_load.get("real_i8_expert_dequant_smoke_ready") is True:
            if i8_expert_smoke.get("ready") is not True:
                errors.append("real_i8_expert_dequant_smoke_ready_mismatch")
            if i8_expert_smoke.get("weight_tensor_values_public") is not False:
                errors.append("real_i8_expert_dequant_weight_public_unsafe")
            if i8_expert_smoke.get("activation_payload_public") is not False:
                errors.append("real_i8_expert_dequant_activation_public_unsafe")
            if not str(i8_expert_smoke.get("output_hash") or "").startswith("sha256:"):
                errors.append("real_i8_expert_dequant_output_hash_missing")
            if _int(_list(i8_expert_smoke.get("weight_block_shape"))[0] if _list(i8_expert_smoke.get("weight_block_shape")) else 0) <= 0:
                errors.append("real_i8_expert_dequant_shape_missing")
            if _int(i8_expert_smoke.get("scale_group_size")) <= 0:
                errors.append("real_i8_expert_dequant_scale_group_missing")
        i8_expert_mlp = _dict(real_load.get("real_i8_expert_mlp_slice_smoke"))
        if real_load.get("real_i8_expert_mlp_slice_smoke_ready") is True:
            if i8_expert_mlp.get("ready") is not True:
                errors.append("real_i8_expert_mlp_slice_smoke_ready_mismatch")
            if i8_expert_mlp.get("weight_tensor_values_public") is not False:
                errors.append("real_i8_expert_mlp_slice_weight_public_unsafe")
            if i8_expert_mlp.get("activation_payload_public") is not False:
                errors.append("real_i8_expert_mlp_slice_activation_public_unsafe")
            if not str(i8_expert_mlp.get("output_hash") or "").startswith("sha256:"):
                errors.append("real_i8_expert_mlp_slice_output_hash_missing")
            for name in ["w1_block_shape", "w2_block_shape", "w3_block_shape"]:
                if _int(_list(i8_expert_mlp.get(name))[0] if _list(i8_expert_mlp.get(name)) else 0) <= 0:
                    errors.append(f"real_i8_expert_mlp_slice_shape_missing:{name}")
        fp4_topk = _dict(real_load.get("real_fp4_topk_expert_mlp_forward"))
        if real_load.get("real_fp4_topk_expert_mlp_forward_ready") is True:
            if fp4_topk.get("ready") is not True:
                errors.append("real_fp4_topk_expert_forward_ready_mismatch")
            if fp4_topk.get("weight_tensor_values_public") is not False:
                errors.append("real_fp4_topk_expert_forward_weight_public_unsafe")
            if fp4_topk.get("activation_payload_public") is not False:
                errors.append("real_fp4_topk_expert_forward_activation_public_unsafe")
            if _int(fp4_topk.get("topk")) < 1:
                errors.append("real_fp4_topk_expert_forward_topk_missing")
            if _int(fp4_topk.get("loaded_tensor_count")) < 1:
                errors.append("real_fp4_topk_expert_forward_loaded_tensor_count_missing")
            if _int(fp4_topk.get("total_loaded_tensor_bytes")) <= 0:
                errors.append("real_fp4_topk_expert_forward_loaded_tensor_bytes_missing")
            if _list(fp4_topk.get("final_output_shape")) != [4096]:
                errors.append("real_fp4_topk_expert_forward_output_shape_mismatch")
            if fp4_topk.get("finite_output") is not True:
                errors.append("real_fp4_topk_expert_forward_output_not_finite")
            if not str(fp4_topk.get("final_output_hash") or "").startswith("sha256:"):
                errors.append("real_fp4_topk_expert_forward_output_hash_missing")
        for item in _list(real_load.get("tensor_summaries")):
            item_dict = _dict(item)
            if item_dict.get("weight_tensor_values_public") is not False:
                errors.append("real_weight_tpu_tensor_item_values_public_unsafe")
            if item_dict.get("device_put_ready") is not True:
                errors.append("real_weight_tpu_tensor_item_device_put_missing")
            if not str(item_dict.get("raw_payload_sha256") or "").startswith("sha256:"):
                errors.append("real_weight_tpu_tensor_item_hash_missing")
    if fixture_ready:
        fixture = _dict(cell.get("deepseek_v4_jax_tpu_fixture_stage_forward"))
        components = _dict(fixture.get("components_exercised"))
        for component in [
            "manifold_hyper_connections",
            "mla_shared_kv_attention",
            "grouped_output_projection",
            "attention_sink",
            "topk_moe_router",
            "routed_experts",
            "shared_experts",
            "stage_local_kv_cache_shape",
        ]:
            if components.get(component) is not True:
                errors.append(f"fixture_component_missing:{component}")
        if fixture.get("fixture_weights") is not True:
            errors.append("fixture_weights_flag_missing")
        if fixture.get("real_weight_tensor_values_loaded") is not False:
            errors.append("fixture_real_weight_overclaim")
        if fixture.get("activation_payload_public") is not False:
            errors.append("fixture_activation_public_unsafe")
        kv_cache = _dict(fixture.get("stage_local_kv_cache_metadata"))
        if kv_cache.get("stage_local_only") is not True:
            errors.append("fixture_stage_local_kv_cache_missing")
        if kv_cache.get("kv_payload_public") is not False:
            errors.append("fixture_kv_payload_public_unsafe")

    adapter_ready = report.get("deepseek_v4_jax_tpu_stage_forward_ready") is True
    if report.get("deepseek_v4_flash_kaggle_web_tpu_stage_adapter_ready") is True:
        if report.get("ok") is not True:
            errors.append("adapter_ready_report_not_ok")
        if report.get("metadata_ready") is not True:
            errors.append("adapter_ready_without_metadata")
        if report.get("kaggle_web_tpu_runtime_ready") is not True:
            errors.append("adapter_ready_without_web_tpu")
        if not adapter_ready:
            errors.append("adapter_ready_without_stage_forward")
        if report.get("failure_stage"):
            errors.append("adapter_ready_has_failure_stage")
        if _list(report.get("blockers")):
            errors.append("adapter_ready_has_blockers")
    else:
        if report.get("ok") is True:
            errors.append("adapter_not_ready_report_ok")
        if not _list(report.get("blockers")):
            errors.append("adapter_not_ready_without_blocker")
        if not str(report.get("failure_stage") or "").strip():
            errors.append("adapter_not_ready_without_failure_stage")
        if (
            report.get("metadata_ready") is True
            and report.get("kaggle_web_tpu_runtime_ready") is True
            and not adapter_ready
            and not (
                "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented" in set(report.get("blockers") or [])
                or "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented" in set(report.get("blockers") or [])
            )
        ):
            errors.append("adapter_gap_blocker_missing")

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
        "weight_tensor_values_public",
        "credentials_public",
        "cookies_public",
        "jupyter_proxy_token_public",
        "private_runtime_state_public",
    ]:
        if safety.get(flag) is not False:
            errors.append(f"safety_flag_mismatch:{flag}")
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_mismatch")

    cleanup = _dict(report.get("cleanup_status"))
    if cleanup.get("temporary_kaggle_kernels_deleted") is not True:
        errors.append("cleanup_kernel_deleted_missing")
    if cleanup.get("temporary_private_packages_removed") is not True:
        errors.append("cleanup_private_packages_removed_missing")
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_left_running")
    if cleanup.get("cookie_file_public") is not False:
        errors.append("cleanup_cookie_public_flag_missing")
    if cleanup.get("storage_state_file_public") is not False:
        errors.append("cleanup_storage_state_public_flag_missing")

    artifacts = _dict(report.get("artifacts"))
    if _dict(artifacts.get("summary_json")).get("present") is not True:
        errors.append("artifact_missing:summary_json")
    return sorted(set(errors))


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    report = load_json(Path(args.report))
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": args.report,
        "metadata_ready": report.get("metadata_ready") is True,
        "kaggle_web_tpu_runtime_ready": report.get("kaggle_web_tpu_runtime_ready") is True,
        "deepseek_v4_jax_tpu_stage_forward_ready": report.get("deepseek_v4_jax_tpu_stage_forward_ready") is True,
        "failure_stage": report.get("failure_stage"),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DeepSeek-V4-Flash Kaggle Web TPU stage-adapter probe artifact.")
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
