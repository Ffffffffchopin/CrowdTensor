#!/usr/bin/env python3
"""Validate DeepSeek-V4-Flash Quantized Swarm RC evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import deepseek_v4_flash_quantized_swarm_rc_pack as pack  # noqa: E402


SCHEMA = "deepseek_v4_flash_quantized_swarm_rc_check_v1"
REQUIRED_PROVIDERS = {"kaggle_cuda", "colab_cuda", "cpu"}


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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("deepseek_v4_flash_quantized_swarm_rc_ready") is not True:
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
    if abs(_float(model.get("total_params_b")) - 284.0) > 0.001:
        errors.append("total_params_mismatch")
    if abs(_float(model.get("active_params_b")) - 13.0) > 0.001:
        errors.append("active_params_mismatch")
    if model.get("quantized") is not True:
        errors.append("quantized_flag_missing")

    source = _dict(report.get("source_resolver"))
    if source.get("schema") != "deepseek_v4_flash_quantized_source_summary_v1":
        errors.append("source_summary_schema_mismatch")
    if source.get("present") is not True:
        errors.append("source_resolver_missing")
    if source.get("source_ok") is not True:
        errors.append("source_resolver_not_ok")
    if source.get("resolver_ready") is not True:
        errors.append("source_resolver_not_ready")
    if source.get("public_artifact_safe") is not True:
        errors.append("source_resolver_public_artifact_unsafe")
    if not str(source.get("recommended_repo") or ""):
        errors.append("recommended_repo_missing")
    if not str(source.get("recommended_quant") or ""):
        errors.append("recommended_quant_missing")
    if _float(source.get("recommended_total_size_gb")) <= 0:
        errors.append("recommended_size_missing")
    if _int(source.get("recommended_split_file_count")) < 1:
        errors.append("recommended_split_files_missing")
    if not _list(source.get("recommended_files")):
        errors.append("recommended_files_missing")
    if source.get("recommended_runtime_backend") != "llama_cpp_v4_fork":
        errors.append("recommended_runtime_backend_mismatch")
    if "deepseek_v4_flash_requires_v4_aware_llama_cpp_fork" not in set(report.get("blockers") or []):
        errors.append("v4_aware_llama_cpp_blocker_missing")

    success = _dict(report.get("success"))
    same = _dict(report.get("same_request"))
    success_verified = success.get("same_request_decode_verified") is True
    same_verified = same.get("same_request_decode_verified") is True
    generated = _int(success.get("generated_token_count"))
    accepted = set(str(item) for item in _list(success.get("accepted_providers")))
    same_accepted = set(str(item) for item in _list(same.get("accepted_providers")))

    if success_verified:
        if same_verified is not True:
            errors.append("success_without_same_request_summary_proof")
        if same.get("present") is not True:
            errors.append("success_without_same_request_report")
        if generated < 1:
            errors.append("success_without_generated_token")
        if not REQUIRED_PROVIDERS.issubset(accepted):
            errors.append("success_missing_required_provider")
        if not REQUIRED_PROVIDERS.issubset(same_accepted):
            errors.append("same_request_summary_missing_required_provider")
        if same.get("public_artifact_safe") is not True:
            errors.append("same_request_public_artifact_unsafe")
        if str(report.get("failure_stage") or ""):
            errors.append("success_with_failure_stage")
        if "deepseek_v4_flash_quantized_same_request_decode_verified" not in set(report.get("diagnosis_codes") or []):
            errors.append("success_diagnosis_missing")
    else:
        if generated >= 1 and REQUIRED_PROVIDERS.issubset(accepted):
            errors.append("same_request_evidence_present_but_success_false")
        if "deepseek_v4_flash_quantized_same_request_decode_not_verified" not in set(report.get("blockers") or []):
            errors.append("failure_blocker_missing")
        if not str(report.get("failure_stage") or "").strip():
            errors.append("failure_stage_missing")
        if "deepseek_v4_flash_quantized_same_request_decode_not_verified" not in set(report.get("diagnosis_codes") or []):
            errors.append("failure_diagnosis_missing")
    if _list(success.get("required_providers")) != ["kaggle_cuda", "colab_cuda", "cpu"]:
        errors.append("required_providers_mismatch")

    single = _dict(report.get("single_kernel_probe"))
    if single.get("present") is True and single.get("public_artifact_safe") is not True:
        errors.append("single_kernel_public_artifact_unsafe")
    if single.get("one_token_generation_verified") is True and success_verified and same.get("present") is not True:
        errors.append("single_kernel_generation_counted_as_swarm_success")

    kaggle_gpu = _dict(report.get("kaggle_gpu_preflight"))
    if kaggle_gpu.get("present") is True:
        if kaggle_gpu.get("schema") != "deepseek_v4_flash_kaggle_gpu_preflight_summary_v1":
            errors.append("kaggle_gpu_preflight_schema_mismatch")
        if kaggle_gpu.get("public_artifact_safe") is not True:
            errors.append("kaggle_gpu_preflight_public_artifact_unsafe")
        if kaggle_gpu.get("owner_public") is not False:
            errors.append("kaggle_gpu_owner_public_flag_mismatch")
        if kaggle_gpu.get("evidence_ready") is not True:
            errors.append("kaggle_gpu_preflight_evidence_not_ready")
        if success_verified and kaggle_gpu.get("kaggle_cuda_ready") is not True:
            errors.append("success_with_kaggle_cuda_preflight_not_ready")
        if kaggle_gpu.get("kaggle_cuda_ready") is not True:
            if "kaggle_cuda_preflight_not_ready" not in set(report.get("blockers") or []):
                errors.append("kaggle_cuda_preflight_blocker_missing")

    colab_cuda = _dict(report.get("colab_cuda_preflight"))
    if colab_cuda.get("present") is True:
        if colab_cuda.get("schema") != "deepseek_v4_flash_colab_cuda_preflight_summary_v1":
            errors.append("colab_cuda_preflight_schema_mismatch")
        if colab_cuda.get("public_artifact_safe") is not True:
            errors.append("colab_cuda_preflight_public_artifact_unsafe")
        if success_verified and colab_cuda.get("colab_cuda_ready") is not True:
            errors.append("success_with_colab_cuda_preflight_not_ready")
        if colab_cuda.get("colab_cuda_ready") is not True:
            if "colab_cuda_preflight_not_ready" not in set(report.get("blockers") or []):
                errors.append("colab_cuda_preflight_blocker_missing")

    llama_v4 = _dict(report.get("llama_v4_build_preflight"))
    if llama_v4.get("present") is True:
        if llama_v4.get("schema") != "deepseek_v4_flash_llama_v4_build_preflight_summary_v1":
            errors.append("llama_v4_build_preflight_schema_mismatch")
        if llama_v4.get("public_artifact_safe") is not True:
            errors.append("llama_v4_build_preflight_public_artifact_unsafe")
        if success_verified and llama_v4.get("llama_v4_runtime_build_ready") is not True:
            errors.append("success_with_llama_v4_runtime_build_not_ready")
        if llama_v4.get("llama_v4_runtime_build_ready") is not True:
            if "deepseek_v4_flash_llama_v4_runtime_build_not_ready" not in set(report.get("blockers") or []):
                errors.append("llama_v4_build_blocker_missing")
        else:
            if llama_v4.get("llama_cli_present") is not True:
                errors.append("llama_v4_ready_without_llama_cli")
            if llama_v4.get("rpc_server_present") is not True:
                errors.append("llama_v4_ready_without_rpc_server")
            if llama_v4.get("llama_cli_supports_rpc") is not True:
                errors.append("llama_v4_ready_without_rpc_flag")
            if llama_v4.get("cmake_configure_ok") is not True:
                errors.append("llama_v4_ready_without_cmake_configure")
            if llama_v4.get("cmake_build_ok") is not True:
                errors.append("llama_v4_ready_without_cmake_build")
            if llama_v4.get("provider") == "kaggle_cuda":
                if llama_v4.get("fresh_kaggle_run_performed") is not True:
                    errors.append("kaggle_llama_v4_ready_without_fresh_run")
                if llama_v4.get("kaggle_kernel_deleted") is not True:
                    errors.append("kaggle_llama_v4_ready_without_kernel_cleanup")
                if llama_v4.get("private_package_removed") is not True:
                    errors.append("kaggle_llama_v4_ready_without_private_cleanup")

    artifacts = _dict(report.get("artifacts"))
    for name in ("summary_json", "support_bundle_json"):
        if _dict(artifacts.get(name)).get("present") is not True:
            errors.append(f"artifact_missing:{name}")

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
        errors.append("safety_public_artifact_safe_missing")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_check(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"deepseek_v4_flash_quantized_swarm_rc_check: ok={result['ok']} errors={','.join(result['errors']) or 'none'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
