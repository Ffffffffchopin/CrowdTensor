#!/usr/bin/env python3
"""Validate dense Qwen GPU+TPU+CPU frontier artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import three_accelerator_dense_qwen_frontier_pack as pack  # noqa: E402


SCHEMA = "three_accelerator_dense_qwen_frontier_check_v1"


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
    if report.get("three_accelerator_dense_qwen_frontier_ready") is not True:
        errors.append("frontier_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    scope = _dict(report.get("goal_scope"))
    if scope.get("requires_gpu_tpu_cpu_same_request_for_success") is not True:
        errors.append("goal_scope_same_request_requirement_missing")
    if scope.get("dense_full_precision_main_path") is not True:
        errors.append("goal_scope_dense_main_path_missing")
    if scope.get("quantized_large_model_main_path_allowed") is not False:
        errors.append("goal_scope_quantized_large_model_not_disallowed")
    if scope.get("kaggle_models_attach_preferred") is not True:
        errors.append("goal_scope_kaggle_attach_missing")

    for field in [
        "largest_dense_model_attempted",
        "largest_dense_model_attached",
        "largest_dense_model_stage_preflighted",
        "largest_dense_model_loaded",
        "largest_dense_model_1token_decoded",
        "all_three_accelerators_same_request_verified",
        "same_request_dense_32b_success",
        "kaggle_model_attach_used",
        "tpu_jax_qwen_stage_runtime_ready",
        "gpu_stage_runtime_ready",
        "cpu_stage_runtime_ready",
        "generated_token_count",
        "cleanup_status",
        "blocker_codes",
    ]:
        if field not in report:
            errors.append(f"required_field_missing:{field}")

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
        "private_runtime_state_public",
        "private_kaggle_payload_public",
        "weight_tensor_values_public",
    ]:
        if safety.get(flag) is not False:
            errors.append(f"safety_flag_mismatch:{flag}")
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_mismatch")

    model_sources = _dict(report.get("model_sources"))
    candidates = [item for item in _list(model_sources.get("candidates")) if isinstance(item, dict)]
    if not candidates:
        errors.append("model_source_candidates_missing")
    required_sizes = {"72b", "32b", "14b", "7b"}
    present_sizes = {str(item.get("parameter_class") or "") for item in candidates}
    for size in required_sizes:
        if size not in present_sizes:
            errors.append(f"dense_candidate_missing:{size}")
    for item in candidates:
        if item.get("framework") != "Transformers":
            errors.append(f"candidate_framework_not_transformers:{item.get('parameter_class')}")
        slug = str(item.get("instance_slug") or "").lower()
        if any(term in slug for term in ["awq", "gptq", "4bit", "8bit", "fp8", "bnb", "gguf"]):
            errors.append(f"candidate_not_dense_full_precision:{item.get('parameter_class')}")
        if item.get("attach_can_avoid_runtime_download") is not True:
            errors.append(f"candidate_attach_not_runtime_download_safe:{item.get('parameter_class')}")
        if not str(item.get("kaggle_kernel_model_source") or "").strip():
            errors.append(f"candidate_model_source_missing:{item.get('parameter_class')}")
        if not str(item.get("attached_runtime_path") or "").startswith("/kaggle/input/"):
            errors.append(f"candidate_attached_path_invalid:{item.get('parameter_class')}")
    if pack.parameter_value(str(report.get("largest_dense_model_attempted") or "")) < 70:
        errors.append("largest_dense_model_attempted_below_72b")
    if pack.parameter_value(str(report.get("largest_dense_model_loaded") or "")) > 32:
        attached_loaded = [
            item
            for item in candidates
            if item.get("parameter_class") == report.get("largest_dense_model_loaded")
            and item.get("attach_path_present") is True
        ]
        if not attached_loaded:
            errors.append("larger_dense_loaded_claim_without_live_or_attached_proof")
    attach_probe = _dict(report.get("kaggle_model_attach_probe"))
    if report.get("kaggle_model_attach_used") is True:
        if attach_probe.get("schema") != "three_accelerator_dense_qwen_kaggle_model_attach_import_v1":
            errors.append("kaggle_attach_probe_schema_missing")
        if attach_probe.get("kaggle_model_attach_probe_ready") is not True:
            errors.append("kaggle_attach_used_without_ready_probe")
        if attach_probe.get("path_present") is not True:
            errors.append("kaggle_attach_used_without_runtime_path")
        if attach_probe.get("config_json_present") is not True:
            errors.append("kaggle_attach_used_without_config")
        if attach_probe.get("weight_index_present") is not True:
            errors.append("kaggle_attach_used_without_weight_index")
        if _int(attach_probe.get("safetensors_file_count")) < 1:
            errors.append("kaggle_attach_used_without_safetensors")
        if attach_probe.get("temporary_kaggle_kernel_deleted") is not True:
            errors.append("kaggle_attach_probe_kernel_not_deleted")
        if attach_probe.get("temporary_private_package_removed") is not True:
            errors.append("kaggle_attach_probe_private_package_not_removed")
        if report.get("largest_dense_model_attached") != attach_probe.get("parameter_class"):
            errors.append("largest_attached_not_reflecting_attach_probe")
        if attach_probe.get("dense_full_precision") is not True:
            errors.append("kaggle_attach_probe_not_dense_full_precision")
        if attach_probe.get("stage_owned_preflight_verified") is True:
            if attach_probe.get("stage_plan_schema") != "kaggle_model_attach_stage_plan_v1":
                errors.append("stage_preflight_schema_missing")
            if _int(attach_probe.get("stage_plan_stage_count")) < 2:
                errors.append("stage_preflight_stage_count_too_small")
            backends = set(attach_probe.get("stage_plan_backends") or [])
            if not {"cuda", "jax_tpu", "cpu"}.issubset(backends):
                errors.append("stage_preflight_backends_missing")
            if _int(attach_probe.get("stage_plan_assigned_key_count_total")) < 1:
                errors.append("stage_preflight_assigned_keys_missing")
            if _int(attach_probe.get("stage_plan_present_key_count_total")) != _int(attach_probe.get("stage_plan_assigned_key_count_total")):
                errors.append("stage_preflight_present_key_mismatch")
            stage_summaries = [item for item in _list(attach_probe.get("stage_plan_stage_summaries")) if isinstance(item, dict)]
            if len(stage_summaries) != _int(attach_probe.get("stage_plan_stage_count")):
                errors.append("stage_preflight_stage_summaries_mismatch")
            if any(item.get("stage_owned_header_verified") is not True for item in stage_summaries):
                errors.append("stage_preflight_stage_not_verified")
            if report.get("largest_dense_model_stage_preflighted") != attach_probe.get("parameter_class"):
                errors.append("largest_stage_preflighted_not_reflecting_attach_probe")
        elif report.get("largest_dense_model_stage_preflighted"):
            errors.append("stage_preflight_claim_without_attach_probe")

    adapter = _dict(report.get("tpu_dense_qwen_adapter"))
    if adapter.get("dense_full_precision_only") is not True:
        errors.append("adapter_dense_full_precision_only_missing")
    if adapter.get("quantized_weight_adapter_used") is not False:
        errors.append("adapter_quantized_weight_used")
    if adapter.get("torch_reference_forward_ready") is not True:
        errors.append("adapter_torch_reference_forward_missing")
    components = _dict(adapter.get("qwen_components_exercised"))
    for component in ["rms_norm", "rope", "causal_attention", "swiglu_mlp", "stage_local_kv_cache"]:
        if components.get(component) is not True:
            errors.append(f"adapter_component_missing:{component}")
    if report.get("tpu_jax_qwen_stage_runtime_ready") is True and adapter.get("tpu_runtime_ready") is not True:
        errors.append("tpu_stage_runtime_ready_without_tpu")
    tpu_loader = _dict(report.get("retained_tpu_dense_qwen_stage"))
    if report.get("tpu_jax_qwen_stage_runtime_ready") is True:
        if tpu_loader.get("schema") != "three_accelerator_dense_qwen_tpu_loader_import_v1":
            errors.append("tpu_loader_import_schema_missing")
        if tpu_loader.get("tpu_jax_qwen_stage_runtime_ready") is not True:
            errors.append("tpu_ready_without_real_loader_import")
        if tpu_loader.get("full_stage_owned_tpu_loader_ready") is not True:
            errors.append("tpu_ready_without_full_stage_loader")
        if tpu_loader.get("quantization") != "none":
            errors.append("tpu_loader_not_dense")
        if _int(tpu_loader.get("executed_layer_count")) < 1:
            errors.append("tpu_loader_executed_layer_missing")
        if _int(tpu_loader.get("loaded_execution_tensor_key_count")) < 1:
            errors.append("tpu_loader_weight_keys_missing")
        if _int(tpu_loader.get("tpu_device_count")) < 1:
            errors.append("tpu_loader_device_count_missing")
        if tpu_loader.get("stage_local_kv_cache_verified") is not True:
            errors.append("tpu_loader_kv_cache_missing")
    elif tpu_loader.get("schema") == "three_accelerator_dense_qwen_tpu_loader_import_v1" and tpu_loader.get("tpu_jax_qwen_stage_runtime_ready") is True:
        errors.append("tpu_loader_ready_not_reflected")

    baseline = _dict(report.get("baseline_32b_three_accelerator"))
    if baseline.get("schema") != "three_accelerator_dense_qwen_32b_bridge_import_v1":
        errors.append("baseline_schema_mismatch")
    if baseline.get("all_three_accelerators_same_request_verified") is True:
        if not {"cuda", "jax_tpu", "cpu"}.issubset(set(baseline.get("accepted_stage_backends") or [])):
            errors.append("baseline_required_backends_missing")
        if _int(baseline.get("generated_token_count")) < 1:
            errors.append("baseline_generated_token_count_missing")
        if report.get("largest_dense_model_1token_decoded") != "32b":
            errors.append("baseline_decode_not_reflected")
    else:
        if report.get("largest_dense_model_1token_decoded"):
            errors.append("decode_claim_without_baseline")

    gpu_cpu_fallback = _dict(report.get("baseline_32b_gpu_cpu_dense_fallback"))
    if gpu_cpu_fallback.get("schema") != "three_accelerator_dense_qwen_gpu_cpu_fallback_import_v1":
        errors.append("gpu_cpu_dense_fallback_schema_mismatch")
    if gpu_cpu_fallback.get("gpu_cpu_dense_fallback_verified") is not True:
        errors.append("gpu_cpu_dense_fallback_missing")
    if gpu_cpu_fallback.get("quantization") != "none":
        errors.append("gpu_cpu_dense_fallback_not_dense")
    if not {"gpu", "cpu"}.issubset(set(gpu_cpu_fallback.get("resource_kinds") or [])):
        errors.append("gpu_cpu_dense_fallback_backends_missing")

    if pack.parameter_value(str(report.get("largest_dense_model_1token_decoded") or "")) > 32:
        ladder = [item for item in _list(report.get("bounded_experiment_ladder")) if isinstance(item, dict)]
        matching = [item for item in ladder if item.get("parameter_class") == report.get("largest_dense_model_1token_decoded")]
        if not matching or matching[0].get("one_token_decode_verified") is not True:
            errors.append("larger_dense_decode_claim_without_ladder_proof")
    if pack.parameter_value(str(report.get("largest_dense_model_stage_preflighted") or "")) > pack.parameter_value(str(report.get("largest_dense_model_loaded") or "")):
        if pack.parameter_value(str(report.get("largest_dense_model_loaded") or "")) > 32:
            errors.append("stage_preflight_unexpectedly_promoted_loaded_claim")
    if report.get("same_request_dense_frontier_success") is True:
        if report.get("tpu_jax_qwen_stage_runtime_ready") is not True:
            errors.append("frontier_success_without_tpu_adapter")
        if report.get("kaggle_model_attach_available") is not True or report.get("kaggle_model_attach_used") is not True:
            errors.append("frontier_success_without_model_attach")
        if report.get("largest_dense_model_1token_decoded") != report.get("largest_dense_model_attempted"):
            errors.append("frontier_success_without_largest_attempt_decode")
        if report.get("largest_dense_model_attached") != report.get("largest_dense_model_attempted"):
            errors.append("frontier_success_without_largest_attempt_attach")
    else:
        if not _list(report.get("blocker_codes")):
            errors.append("blocked_frontier_without_blockers")
    if report.get("same_request_dense_32b_success") is True:
        if report.get("largest_dense_model_1token_decoded") != "32b":
            errors.append("dense_32b_success_decode_mismatch")
        if report.get("tpu_jax_qwen_stage_runtime_ready") is not True:
            errors.append("dense_32b_success_without_tpu_adapter")

    cleanup = _dict(report.get("cleanup_status"))
    if cleanup.get("temporary_kaggle_kernels_deleted") is not True:
        errors.append("cleanup_kernel_deleted_missing")
    if cleanup.get("temporary_private_packages_removed") is not True:
        errors.append("cleanup_private_packages_removed_missing")
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_left_running")

    artifacts = _dict(report.get("artifacts"))
    required_artifacts = ["summary_json", "support_bundle_json", "model_source_resolver_json", "adapter_smoke_json", "gpu_cpu_dense_fallback_json", "tpu_dense_loader_json"]
    if report.get("kaggle_model_attach_used") is True:
        required_artifacts.append("kaggle_model_attach_probe_json")
    for artifact_name in required_artifacts:
        if _dict(artifacts.get(artifact_name)).get("present") is not True:
            errors.append(f"artifact_missing:{artifact_name}")
    return errors


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report = load_json(Path(args.report))
        report_path = args.report
    else:
        pack_args = [
            "--output-dir",
            args.output_dir,
            "--baseline-32b-bridge-report",
            args.baseline_32b_bridge_report,
            "--gpu-cpu-dense-fallback-report",
            args.gpu_cpu_dense_fallback_report,
            "--tpu-dense-loader-report",
            args.tpu_dense_loader_report,
            "--kaggle-model-attach-probe-report",
            args.kaggle_model_attach_probe_report,
            "--dense-adapter-report",
            args.dense_adapter_report,
            "--kaggle-input-root",
            args.kaggle_input_root,
            "--hf-timeout-seconds",
            str(args.hf_timeout_seconds),
            "--adapter-sequence-length",
            str(args.adapter_sequence_length),
        ]
        if args.fetch_hf_metadata:
            pack_args.append("--fetch-hf-metadata")
        if args.run_jax_adapter:
            pack_args.append("--run-jax-adapter")
        if args.require_tpu_adapter:
            pack_args.append("--require-tpu-adapter")
        report = pack.build_report(pack.parse_args(pack_args))
        report_path = str(Path(args.output_dir) / "three_accelerator_dense_qwen_frontier.json")
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": report_path,
        "three_accelerator_dense_qwen_frontier_ready": report.get("three_accelerator_dense_qwen_frontier_ready") is True,
        "largest_dense_model_attempted": report.get("largest_dense_model_attempted"),
        "largest_dense_model_attached": report.get("largest_dense_model_attached"),
        "largest_dense_model_loaded": report.get("largest_dense_model_loaded"),
        "largest_dense_model_1token_decoded": report.get("largest_dense_model_1token_decoded"),
        "all_three_accelerators_same_request_verified": report.get("all_three_accelerators_same_request_verified") is True,
        "kaggle_model_attach_used": report.get("kaggle_model_attach_used") is True,
        "tpu_jax_qwen_stage_runtime_ready": report.get("tpu_jax_qwen_stage_runtime_ready") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dense Qwen GPU+TPU+CPU frontier artifact.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-32b-bridge-report", default=pack.DEFAULT_32B_BRIDGE_REPORT)
    parser.add_argument("--gpu-cpu-dense-fallback-report", default=pack.DEFAULT_GPU_CPU_DENSE_FALLBACK_REPORT)
    parser.add_argument("--tpu-dense-loader-report", default=pack.DEFAULT_TPU_DENSE_LOADER_REPORT)
    parser.add_argument("--kaggle-model-attach-probe-report", default=pack.DEFAULT_KAGGLE_MODEL_ATTACH_PROBE_REPORT)
    parser.add_argument("--dense-adapter-report", default=pack.DEFAULT_DENSE_ADAPTER_REPORT)
    parser.add_argument("--kaggle-input-root", default="/kaggle/input")
    parser.add_argument("--fetch-hf-metadata", action="store_true")
    parser.add_argument("--hf-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--adapter-sequence-length", type=int, default=4)
    parser.add_argument("--run-jax-adapter", action="store_true")
    parser.add_argument("--require-tpu-adapter", action="store_true")
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
