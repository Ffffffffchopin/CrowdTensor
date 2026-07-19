#!/usr/bin/env python3
"""Validate GPU+TPU+CPU heterogeneous capacity frontier evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import heterogeneous_capacity_frontier_pack as pack  # noqa: E402


SCHEMA = "heterogeneous_capacity_frontier_check_v1"


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
    if report.get("heterogeneous_capacity_frontier_ready") is not True:
        errors.append("heterogeneous_capacity_frontier_ready_missing")
    if report.get("execution_mode") not in pack.EXECUTION_MODES:
        errors.append("execution_mode_invalid")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")

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
        errors.append("safety_public_artifact_safe_mismatch")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    baseline = _dict(report.get("baseline_32b"))
    if baseline.get("schema") != "heterogeneous_capacity_32b_baseline_v1":
        errors.append("baseline_32b_schema_mismatch")
    if baseline.get("gpu_tpu_cpu_same_request_verified") is not True:
        errors.append("baseline_32b_same_request_missing")
    if _int(baseline.get("generated_token_count")) < 4:
        errors.append("baseline_32b_generated_token_count_below_four")
    if not {"cuda", "jax_tpu", "cpu"}.issubset(set(baseline.get("accepted_stage_backends") or [])):
        errors.append("baseline_32b_required_backends_missing")

    candidates = [item for item in _list(report.get("candidates")) if isinstance(item, dict)]
    if not candidates:
        errors.append("candidates_missing")
    if not any(str(item.get("parameter_class") or "").startswith("72b") for item in candidates):
        errors.append("candidate_72b_missing")
    if not any(pack.parameter_class_value(str(item.get("parameter_class") or "")) >= 100 for item in candidates):
        errors.append("candidate_100b_plus_missing")

    stage_imports = [item for item in _list(report.get("larger_stage_load_imports")) if isinstance(item, dict)]
    live_load_candidates = [item for item in candidates if item.get("stage_owned_load_verified") is True]
    for imported in stage_imports:
        if imported.get("schema") != "heterogeneous_capacity_larger_stage_load_import_v1":
            errors.append("larger_stage_load_import_schema_mismatch")
        if imported.get("stage_owned_load_verified") is True:
            if _int(imported.get("stage_count")) < 3:
                errors.append("larger_stage_load_import_stage_count_too_low")
            if _int(imported.get("stage_report_count")) != _int(imported.get("stage_count")):
                errors.append("larger_stage_load_import_stage_report_count_mismatch")
            if _int(imported.get("covered_weight_key_count")) != _int(imported.get("weight_key_count")):
                errors.append("larger_stage_load_import_key_coverage_mismatch")
            if imported.get("kernels_deleted") is not True:
                errors.append("larger_stage_load_import_kernel_cleanup_missing")
            if imported.get("private_packages_removed") is not True:
                errors.append("larger_stage_load_import_private_cleanup_missing")
    partial_imports = [item for item in _list(report.get("partial_stage_load_imports")) if isinstance(item, dict)]
    partial_live_candidates = [item for item in candidates if item.get("partial_stage_owned_load_verified") is True]
    for imported in partial_imports:
        if imported.get("schema") != "heterogeneous_capacity_partial_stage_load_import_v1":
            errors.append("partial_stage_load_import_schema_mismatch")
        if imported.get("partial_stage_load_verified") is True:
            if _int(imported.get("stage_count")) < 3:
                errors.append("partial_stage_load_import_stage_count_too_low")
            if _int(imported.get("assigned_weight_key_count")) < 1:
                errors.append("partial_stage_load_import_assigned_keys_missing")
            if _int(imported.get("loaded_weight_key_count")) != _int(imported.get("assigned_weight_key_count")):
                errors.append("partial_stage_load_import_loaded_key_count_mismatch")
            if imported.get("temp_cleanup_ok") is not True:
                errors.append("partial_stage_load_import_temp_cleanup_missing")

    for candidate in candidates:
        if candidate.get("schema") != "heterogeneous_capacity_candidate_v1":
            errors.append("candidate_schema_mismatch")
        for field in ["model_id", "parameter_class", "quantization", "weight_format", "blocked_reason"]:
            if not str(candidate.get(field) or "").strip():
                errors.append(f"candidate_{candidate.get('parameter_class')}_field_missing:{field}")
        topology = _dict(candidate.get("topology"))
        if candidate.get("blocked_reason") != "candidate_metadata_preflight_failed":
            if topology.get("stage_count", 0) < 3:
                errors.append(f"candidate_{candidate.get('parameter_class')}_stage_count_too_low")
            backends = set(topology.get("stage_backends") or [])
            if not {"cuda", "jax_tpu", "cpu"}.issubset(backends):
                errors.append(f"candidate_{candidate.get('parameter_class')}_required_backends_missing")
            stages = [item for item in _list(candidate.get("stage_plans")) if isinstance(item, dict)]
            if len(stages) != _int(topology.get("stage_count")):
                errors.append(f"candidate_{candidate.get('parameter_class')}_stage_plan_count_mismatch")
            if candidate.get("stage_owned_load_preflight_verified") is True:
                if not all(stage.get("stage_owned_header_verified") is True for stage in stages):
                    errors.append(f"candidate_{candidate.get('parameter_class')}_preflight_overclaim")
                if not all(stage.get("loads_only_stage_weight_keys_preflight") is True for stage in stages):
                    errors.append(f"candidate_{candidate.get('parameter_class')}_stage_key_scope_missing")
        if candidate.get("stage_owned_load_verified") is True and candidate.get("stage_owned_load_preflight_verified") is not True:
            errors.append(f"candidate_{candidate.get('parameter_class')}_load_without_preflight")
        if candidate.get("stage_owned_load_verified") is True:
            imported = _dict(candidate.get("stage_owned_load_live_import"))
            if imported.get("stage_owned_load_verified") is not True:
                errors.append(f"candidate_{candidate.get('parameter_class')}_live_load_import_missing")
            if candidate.get("blocked_reason") and candidate.get("target") == "stage_load":
                errors.append(f"candidate_{candidate.get('parameter_class')}_stage_load_still_blocked_after_import")
        if candidate.get("partial_stage_owned_load_verified") is True:
            imported = _dict(candidate.get("partial_stage_owned_load_live_import"))
            if imported.get("partial_stage_load_verified") is not True:
                errors.append(f"candidate_{candidate.get('parameter_class')}_partial_live_load_import_missing")
        if candidate.get("one_token_decode_verified") is True:
            if candidate.get("gpu_tpu_cpu_same_request_verified") is not True:
                errors.append(f"candidate_{candidate.get('parameter_class')}_decode_without_same_request")
            if not str(candidate.get("live_decode_proof_path") or ""):
                errors.append(f"candidate_{candidate.get('parameter_class')}_decode_proof_missing")
        else:
            if not candidate.get("blockers"):
                errors.append(f"candidate_{candidate.get('parameter_class')}_blocked_without_blockers")

    conclusions = _dict(report.get("conclusions"))
    for field in [
        "max_stage_owned_load_parameter_class",
        "max_stage_owned_load_preflight_parameter_class",
        "max_partial_stage_owned_load_parameter_class",
        "max_1token_decode_parameter_class",
        "max_multitoken_decode_parameter_class",
        "max_gpu_tpu_cpu_same_request_parameter_class",
        "best_successful_topology",
        "largest_failed_candidate",
        "next_bottleneck",
    ]:
        if field not in conclusions:
            errors.append(f"conclusion_missing:{field}")
    if conclusions.get("max_gpu_tpu_cpu_same_request_parameter_class") != "32b":
        errors.append("max_same_request_decode_should_remain_32b_without_larger_live_proof")
    if live_load_candidates and pack.parameter_class_value(str(conclusions.get("max_stage_owned_load_parameter_class") or "")) < 70:
        errors.append("max_stage_owned_load_should_reflect_larger_live_import")
    if partial_live_candidates and pack.parameter_class_value(str(conclusions.get("max_partial_stage_owned_load_parameter_class") or "")) < 100:
        errors.append("max_partial_stage_owned_load_should_reflect_100b_live_import")
    if pack.parameter_class_value(str(conclusions.get("max_stage_owned_load_preflight_parameter_class") or "")) < 70:
        errors.append("max_stage_owned_preflight_below_70b")
    if not _dict(conclusions.get("largest_failed_candidate")).get("blocked_reason"):
        errors.append("largest_failed_candidate_blocker_missing")
    if not _list(conclusions.get("next_bottleneck")):
        errors.append("next_bottleneck_missing")

    resources = _dict(report.get("resource_bounds"))
    if resources.get("single_account_policy") != "respect_kaggle_limits_no_multi_account_bypass":
        errors.append("single_account_policy_missing")

    artifacts = _dict(report.get("artifacts"))
    for name in ["summary_json", "support_bundle_json"]:
        if _dict(artifacts.get(name)).get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return errors


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report = load_json(Path(args.report))
    else:
        pack_args = [
            "--output-dir",
            args.output_dir,
            "--execution-mode",
            args.execution_mode,
            "--baseline-32b-bridge-report",
            args.baseline_32b_bridge_report,
            "--baseline-32b-serving-report",
            args.baseline_32b_serving_report,
            "--stage-count",
            str(args.stage_count),
            "--stage-backends",
            args.stage_backends,
            "--hf-timeout-seconds",
            str(args.hf_timeout_seconds),
            "--max-header-bytes",
            str(args.max_header_bytes),
        ]
        for item in args.candidate or []:
            pack_args.extend(["--candidate", item])
        for item in args.larger_stage_owned_load_report or []:
            pack_args.extend(["--larger-stage-owned-load-report", item])
        for item in args.partial_stage_owned_load_report or []:
            pack_args.extend(["--partial-stage-owned-load-report", item])
        if args.skip_safetensors_headers:
            pack_args.append("--skip-safetensors-headers")
        report = pack.build_report(pack.parse_args(pack_args))
    errors = validate_report(report)
    conclusions = _dict(report.get("conclusions"))
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "output_dir": report.get("output_dir") or args.output_dir,
        "report_path": args.report or str(Path(args.output_dir) / "heterogeneous_capacity_frontier.json"),
        "heterogeneous_capacity_frontier_ready": report.get("heterogeneous_capacity_frontier_ready") is True,
        "max_stage_owned_load_parameter_class": conclusions.get("max_stage_owned_load_parameter_class"),
        "max_stage_owned_load_preflight_parameter_class": conclusions.get("max_stage_owned_load_preflight_parameter_class"),
        "max_partial_stage_owned_load_parameter_class": conclusions.get("max_partial_stage_owned_load_parameter_class"),
        "max_1token_decode_parameter_class": conclusions.get("max_1token_decode_parameter_class"),
        "max_multitoken_decode_parameter_class": conclusions.get("max_multitoken_decode_parameter_class"),
        "max_gpu_tpu_cpu_same_request_parameter_class": conclusions.get("max_gpu_tpu_cpu_same_request_parameter_class"),
        "largest_failed_candidate": conclusions.get("largest_failed_candidate"),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
        "diagnosis_codes": ["heterogeneous_capacity_frontier_check_ready"] if not errors else ["heterogeneous_capacity_frontier_check_failed"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPU+TPU+CPU heterogeneous capacity frontier evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=pack.EXECUTION_MODES, default="metadata-preflight")
    parser.add_argument("--baseline-32b-bridge-report", default=pack.DEFAULT_32B_BRIDGE_REPORT)
    parser.add_argument("--baseline-32b-serving-report", default=pack.DEFAULT_32B_SERVING_REPORT)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--larger-stage-owned-load-report", action="append", default=[])
    parser.add_argument("--partial-stage-owned-load-report", action="append", default=[])
    parser.add_argument("--stage-count", type=int, default=len(pack.DEFAULT_STAGE_BACKENDS))
    parser.add_argument("--stage-backends", default=",".join(pack.DEFAULT_STAGE_BACKENDS))
    parser.add_argument("--hf-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-header-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--skip-safetensors-headers", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print(
            "heterogeneous_capacity_frontier_check: ok "
            f"max_decode={result.get('max_gpu_tpu_cpu_same_request_parameter_class')} "
            f"max_preflight={result.get('max_stage_owned_load_preflight_parameter_class')}"
        )
    else:
        print("heterogeneous_capacity_frontier_check: failed")
        for error in result["errors"]:
            print(f"- {error}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
