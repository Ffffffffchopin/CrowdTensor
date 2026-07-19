#!/usr/bin/env python3
"""Validate Kaggle CUDA + Colab CUDA + CPU max-parameter search evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_colab_gpu_cpu_max_parameter_search_pack as pack  # noqa: E402


SCHEMA = "kaggle_colab_gpu_cpu_max_parameter_search_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("kaggle_colab_gpu_cpu_max_parameter_search_ready") is not True:
        errors.append("max_search_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    scope = _dict(report.get("goal_scope"))
    if scope.get("accelerator_path") != "kaggle_t4x2_colab_t4_kaggle_cpu":
        errors.append("accelerator_path_mismatch")
    if scope.get("dense_full_precision_main_path") is not True:
        errors.append("dense_full_precision_scope_missing")
    if scope.get("quantized_success_allowed") is not False:
        errors.append("quantized_success_not_disallowed")
    if scope.get("same_request_full_layer_decode_required_for_success") is not True:
        errors.append("same_request_full_layer_gate_missing")

    attempts = [item for item in _list(report.get("attempts")) if isinstance(item, dict)]
    if not attempts:
        errors.append("attempts_missing")
    baseline = [item for item in attempts if item.get("parameter_class") == "32b"]
    if not baseline:
        errors.append("baseline_32b_missing")
    elif baseline[0].get("same_request_decode_verified") is not True:
        errors.append("baseline_32b_not_verified")
    if not any(item.get("parameter_class") == "72b" for item in attempts):
        errors.append("attempt_72b_missing")

    for item in attempts:
        if item.get("schema") != "kaggle_colab_gpu_cpu_max_search_attempt_v1":
            errors.append("attempt_schema_mismatch")
        if item.get("quantization") != "none":
            errors.append(f"attempt_quantization_not_none:{item.get('parameter_class')}")
        if item.get("public_artifact_safe") is not True:
            errors.append(f"attempt_public_artifact_unsafe:{item.get('parameter_class')}")
        if item.get("same_request_decode_verified") is True:
            if not {"kaggle_cuda", "colab_cuda", "cpu"}.issubset(set(item.get("accepted_providers") or [])):
                errors.append(f"attempt_success_missing_provider:{item.get('parameter_class')}")
            if int(item.get("generated_token_count") or 0) < 1:
                errors.append(f"attempt_success_missing_token:{item.get('parameter_class')}")
            if item.get("kernels_deleted") is not True:
                errors.append(f"attempt_success_cleanup_missing:{item.get('parameter_class')}")
            if item.get("private_packages_removed") is not True:
                errors.append(f"attempt_success_private_cleanup_missing:{item.get('parameter_class')}")
            if item.get("parameter_class") == "72b" and item.get("full_layer_coverage_verified") is not True:
                errors.append("attempt_72b_success_without_full_layer_coverage")
        else:
            if item.get("parameter_class") != "32b" and not item.get("blockers"):
                errors.append(f"attempt_failure_without_blockers:{item.get('parameter_class')}")

    max_success = str(report.get("max_successful_same_request_decode_parameter_class") or "")
    verified = [
        str(item.get("parameter_class") or "")
        for item in attempts
        if item.get("same_request_decode_verified") is True
    ]
    expected = max(verified, key=pack.parameter_value, default="")
    if max_success != expected:
        errors.append("max_successful_decode_mismatch")
    if pack.parameter_value(max_success) > 32:
        larger = [item for item in attempts if item.get("parameter_class") == max_success]
        if not larger or not any(item.get("same_request_decode_verified") is True for item in larger):
            errors.append("larger_decode_overclaim")
    dense_success = [
        str(item.get("parameter_class") or "")
        for item in attempts
        if item.get("same_request_decode_verified") is True
        and str(item.get("architecture_class") or "dense") == "dense"
        and str(item.get("quantization") or "") == "none"
    ]
    expected_dense = max(dense_success, key=pack.parameter_value, default="")
    if str(report.get("max_successful_dense_full_precision_parameter_class") or "") != expected_dense:
        errors.append("max_successful_dense_full_precision_mismatch")
    moe_success = [
        item for item in attempts
        if item.get("same_request_decode_verified") is True
        and str(item.get("architecture_class") or "") in {"moe", "hybrid_moe"}
    ]
    expected_moe_total = max((float(item.get("moe_total_parameter_count_b") or 0.0) for item in moe_success), default=0.0)
    expected_moe_active = max((float(item.get("moe_active_parameter_count_b") or 0.0) for item in moe_success), default=0.0)
    actual_moe_total = pack.parameter_value(str(report.get("max_successful_moe_total_parameter_class") or ""))
    actual_moe_active = pack.parameter_value(str(report.get("max_successful_moe_activated_parameter_class") or ""))
    if abs(actual_moe_total - expected_moe_total) > 0.001:
        errors.append("max_successful_moe_total_mismatch")
    if abs(actual_moe_active - expected_moe_active) > 0.001:
        errors.append("max_successful_moe_active_mismatch")

    max_attempted = str(report.get("max_attempted_parameter_class") or "")
    attempted = [str(item.get("parameter_class") or "") for item in attempts if item.get("parameter_class")]
    if max_attempted != max(attempted, key=pack.parameter_value, default=""):
        errors.append("max_attempted_mismatch")
    if pack.parameter_value(max_attempted) > pack.parameter_value(max_success):
        if not _list(report.get("blocker_codes")):
            errors.append("larger_attempt_without_blockers")
        if not str(report.get("failure_stage") or "").strip():
            errors.append("larger_attempt_without_failure_stage")

    conclusions = _dict(report.get("conclusions"))
    if conclusions.get("max_stably_verified_dense_full_precision_parameter_class") != expected_dense:
        errors.append("conclusion_max_success_mismatch")
    if conclusions.get("max_successful_dense_full_precision_parameter_class") != report.get("max_successful_dense_full_precision_parameter_class"):
        errors.append("conclusion_max_dense_success_mismatch")
    if conclusions.get("max_successful_moe_total_parameter_class") != report.get("max_successful_moe_total_parameter_class"):
        errors.append("conclusion_max_moe_total_mismatch")
    if conclusions.get("max_successful_moe_activated_parameter_class") != report.get("max_successful_moe_activated_parameter_class"):
        errors.append("conclusion_max_moe_active_mismatch")
    dense_attempted = [
        str(item.get("parameter_class") or "")
        for item in attempts
        if str(item.get("architecture_class") or "dense") == "dense"
        and str(item.get("quantization") or "") == "none"
    ]
    expected_dense_attempted = max(dense_attempted, key=pack.parameter_value, default="")
    if conclusions.get("max_attempted_dense_full_precision_parameter_class") != expected_dense_attempted:
        errors.append("conclusion_max_dense_attempted_mismatch")
    if pack.parameter_value(max_attempted) > pack.parameter_value(max_success) and not _list(conclusions.get("next_bottleneck")):
        errors.append("next_bottleneck_missing")

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
        "max_successful_same_request_decode_parameter_class": report.get("max_successful_same_request_decode_parameter_class"),
        "max_attempted_parameter_class": report.get("max_attempted_parameter_class"),
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
        print(f"kaggle_colab_gpu_cpu_max_parameter_search_check: ok={result['ok']} errors={','.join(result['errors']) or 'none'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
