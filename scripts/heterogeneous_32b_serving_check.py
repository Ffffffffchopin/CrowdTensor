#!/usr/bin/env python3
"""Validate 32B GPU+TPU+CPU heterogeneous serving deployment evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import heterogeneous_32b_serving_pack as pack  # noqa: E402


SCHEMA = "heterogeneous_32b_serving_check_v1"


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
    required_true_fields = [
        "heterogeneous_32b_serving_ready",
        "production_like_serving_path_ready",
        "gpu_tpu_cpu_32b_same_request_source_verified",
        "multi_token_generation_ready",
        "streaming_response_contract_ready",
        "stage_local_kv_cache_ready",
        "latency_metrics_ready",
        "failure_requeue_ready",
        "public_artifact_safe",
    ]
    for field in required_true_fields:
        if report.get(field) is not True:
            errors.append(f"{field}_missing")
    if "live_external_runtime_verified" not in report:
        errors.append("live_external_runtime_verified_missing")
    if "blocked_reason" not in report:
        errors.append("blocked_reason_missing")
    if report.get("serving_mode") not in pack.SERVING_MODES:
        errors.append("serving_mode_invalid")
    if report.get("live_run_mode") not in pack.LIVE_RUN_MODES:
        errors.append("live_run_mode_invalid")
    if _int(report.get("generated_token_count")) < pack.MIN_TARGET_TOKENS:
        errors.append("generated_token_count_below_four")
    if report.get("fallback_model_used") is not False:
        errors.append("fallback_model_used")

    safety = _dict(report.get("safety"))
    for name, expected in pack.default_safety_flags().items():
        if safety.get(name) is not expected:
            errors.append(f"safety_flag_mismatch:{name}")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    source = _dict(report.get("source_32b_summary"))
    if source.get("source_verified") is not True:
        errors.append("source_32b_not_verified")
    if not {"cuda", "jax_tpu", "cpu"}.issubset(set(source.get("accepted_stage_backends") or [])):
        errors.append("source_missing_required_backends")
    if source.get("stage_local_kv_cache_verified") is not True:
        errors.append("source_kv_cache_missing")
    if source.get("tpu_32b_runtime_adapter_ready") is not True:
        errors.append("source_tpu_adapter_missing")

    plan = _dict(report.get("deployment_plan"))
    if plan.get("schema") != pack.SERVING_PLAN_SCHEMA:
        errors.append("deployment_plan_schema_mismatch")
    if plan.get("deployment_path_ready") is not True:
        errors.append("deployment_path_not_ready")
    coordinator = _dict(plan.get("coordinator"))
    if not str(coordinator.get("start_command") or "").startswith("crowdtensor heterogeneous-serve"):
        errors.append("coordinator_start_command_missing")
    miner_backends = {str(item.get("backend") or "") for item in _list(plan.get("miners")) if isinstance(item, dict)}
    if not {"cuda", "jax_tpu", "cpu"}.issubset(miner_backends):
        errors.append("deployment_miners_missing_required_backends")
    user_request = _dict(plan.get("user_request"))
    if "heterogeneous-generate" not in str(user_request.get("generate_command") or ""):
        errors.append("generate_command_missing")

    streaming = _dict(report.get("streaming_response_contract"))
    if streaming.get("schema") != pack.STREAMING_CONTRACT_SCHEMA:
        errors.append("streaming_schema_mismatch")
    if streaming.get("streaming_response_contract_ready") is not True:
        errors.append("streaming_contract_not_ready")
    events = [item for item in _list(streaming.get("events")) if isinstance(item, dict)]
    if len(events) < pack.MIN_TARGET_TOKENS:
        errors.append("streaming_token_events_below_target")
    for item in events:
        if not str(item.get("token_hash") or "").startswith("sha256:"):
            errors.append("streaming_token_hash_missing")
        if item.get("text_public") is not False or item.get("token_id_public") is not False:
            errors.append("streaming_token_payload_public")
        if len(_list(item.get("stage_handoff_hashes"))) < 2:
            errors.append("streaming_handoff_hashes_missing")

    kv_cache = _dict(report.get("stage_local_kv_cache"))
    if kv_cache.get("stage_local_kv_cache_ready") is not True:
        errors.append("kv_cache_not_ready")
    if kv_cache.get("kv_cache_public") is not False or kv_cache.get("past_key_values_public") is not False:
        errors.append("kv_cache_public")
    for stage, payload in _dict(kv_cache.get("per_stage")).items():
        if _dict(payload).get("kv_payload_public") is not False:
            errors.append(f"kv_payload_public:{stage}")

    metrics = _dict(report.get("latency_metrics"))
    if metrics.get("schema") != pack.METRICS_SCHEMA:
        errors.append("metrics_schema_mismatch")
    if metrics.get("latency_metrics_ready") is not True:
        errors.append("latency_metrics_not_ready")
    if float(metrics.get("ttft_ms") or 0.0) <= 0:
        errors.append("ttft_missing")
    if float(metrics.get("token_throughput_tps") or 0.0) <= 0:
        errors.append("throughput_missing")
    activation = _dict(metrics.get("activation_transport"))
    if _int(activation.get("total_activation_bytes")) <= 0:
        errors.append("activation_bytes_missing")
    if activation.get("activation_payload_public") is not False:
        errors.append("activation_payload_public")

    failure = _dict(report.get("failure_requeue"))
    if failure.get("schema") != pack.FAILURE_REQUEUE_SCHEMA:
        errors.append("failure_requeue_schema_mismatch")
    if failure.get("failure_requeue_ready") is not True:
        errors.append("failure_requeue_not_ready")
    for field in ["bounded_failure_injected", "stale_result_rejected", "replacement_claim_accepted", "request_completed_after_requeue"]:
        if failure.get(field) is not True:
            errors.append(f"failure_requeue_{field}_missing")

    live = _dict(report.get("live_external_summary"))
    if live.get("schema") != "heterogeneous_32b_live_external_summary_v1":
        errors.append("live_external_summary_schema_mismatch")
    live_attempt = _dict(report.get("live_external_multitoken_attempt"))
    if live_attempt.get("schema") != pack.LIVE_ATTEMPT_SCHEMA:
        errors.append("live_external_attempt_schema_mismatch")
    if live_attempt.get("requested_generated_token_count") != report.get("target_generated_token_count"):
        errors.append("live_external_attempt_target_mismatch")
    if report.get("live_run_mode") == "external":
        if live_attempt.get("fresh_live_run_attempted") is not True:
            errors.append("live_external_attempt_flag_missing")
    else:
        if live_attempt.get("fresh_live_run_attempted") is not False:
            errors.append("live_external_attempt_false_expected")
        if live_attempt.get("live_external_runtime_verified") is True:
            errors.append("live_external_attempt_overclaims_without_run")
    if report.get("live_external_runtime_verified") is True:
        if report.get("live_run_mode") != "external":
            errors.append("live_external_true_without_external_mode")
        if live.get("live_external_runtime_verified") is not True:
            errors.append("live_external_top_level_mismatch")
        if _int(live.get("generated_token_count")) < _int(report.get("target_generated_token_count"), pack.MIN_TARGET_TOKENS):
            errors.append("live_external_generated_token_count_too_low")
        if live.get("fallback_model_used") is True:
            errors.append("live_external_fallback_model_used")
    else:
        if report.get("live_run_mode") == "external" and not live.get("blockers"):
            errors.append("live_external_blockers_missing")

    blocker = _dict(report.get("blocker_report"))
    if blocker.get("schema") != pack.BLOCKER_SCHEMA:
        errors.append("blocker_report_schema_mismatch")
    if report.get("blocked_reason") and report.get("live_run_mode") != "external":
        errors.append("blocked_reason_set_without_external_live_blocker")
    if report.get("live_run_mode") == "external" and report.get("live_external_runtime_verified") is not True:
        if blocker.get("live_external_runtime_blocked") is not True:
            errors.append("live_external_blocker_flag_missing")
    if blocker.get("deployment_engineering_complete") is not True:
        errors.append("deployment_engineering_not_complete")

    boundaries = _dict(report.get("boundaries"))
    for field in [
        "not_production_sla",
        "not_p2p_nat_traversal",
        "not_billing_or_settlement",
        "not_training_or_finetuning",
        "not_unbounded_kaggle_service",
        "not_larger_model_exploration",
        "live_external_multitoken_requires_fresh_report",
        "fixture_or_fallback_is_not_live_external_success",
    ]:
        if boundaries.get(field) is not True:
            errors.append(f"boundary_missing:{field}")

    artifacts = _dict(report.get("artifacts"))
    for name in [
        "summary_json",
        "summary_markdown",
        "support_bundle_json",
        "deployment_plan_json",
        "streaming_response_contract_json",
        "latency_metrics_json",
        "stage_local_kv_cache_json",
        "failure_requeue_json",
        "live_external_multitoken_attempt_json",
        "blocker_report_json",
    ]:
        if not isinstance(artifacts.get(name), dict) or artifacts[name].get("present") is not True:
            errors.append(f"artifact_missing:{name}")

    diagnosis = set(report.get("diagnosis_codes") or [])
    for code in [
        "heterogeneous_32b_serving_ready",
        "production_like_serving_path_ready",
        "gpu_tpu_cpu_32b_same_request_source_verified",
        "multi_token_generation_ready",
        "streaming_response_contract_ready",
        "stage_local_kv_cache_ready",
        "latency_metrics_ready",
        "failure_requeue_ready",
        "fallback_model_not_used",
        "heterogeneous_32b_serving_public_artifact_redaction_ready",
    ]:
        if code not in diagnosis:
            errors.append(f"diagnosis_missing:{code}")
    if report.get("live_external_runtime_verified") is True:
        if "live_external_runtime_verified" not in diagnosis:
            errors.append("diagnosis_live_external_verified_missing")
    else:
        if "live_external_runtime_not_verified" not in diagnosis:
            errors.append("diagnosis_live_external_not_verified_missing")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 32B heterogeneous serving deployment evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--serving-mode", choices=pack.SERVING_MODES, default="evidence-import")
    parser.add_argument("--rc-report", default=pack.DEFAULT_RC_REPORT)
    parser.add_argument("--live-run-mode", choices=pack.LIVE_RUN_MODES, default="none")
    parser.add_argument("--live-serving-report", default="")
    parser.add_argument("--target-32b-model-id", default=pack.TARGET_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=pack.MIN_TARGET_TOKENS)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--failure-injection", choices=pack.FAILURE_INJECTIONS, default="tpu-timeout")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report_path = Path(args.report)
        report = load_json(report_path)
    else:
        pack_args = pack.parse_args(
            [
                "--output-dir",
                args.output_dir,
                "--serving-mode",
                args.serving_mode,
                "--rc-report",
                args.rc_report,
                "--live-run-mode",
                args.live_run_mode,
                "--target-32b-model-id",
                args.target_32b_model_id,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--context-length",
                str(args.context_length),
                "--failure-injection",
                args.failure_injection,
                *(["--live-serving-report", args.live_serving_report] if args.live_serving_report else []),
            ]
        )
        report = pack.build_report(pack_args)
        report_path = Path(args.output_dir) / "heterogeneous_32b_serving.json"
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": str(report_path),
        "output_dir": str(report.get("output_dir") or ""),
        "heterogeneous_32b_serving_ready": report.get("heterogeneous_32b_serving_ready") is True,
        "production_like_serving_path_ready": report.get("production_like_serving_path_ready") is True,
        "gpu_tpu_cpu_32b_same_request_source_verified": report.get("gpu_tpu_cpu_32b_same_request_source_verified") is True,
        "multi_token_generation_ready": report.get("multi_token_generation_ready") is True,
        "streaming_response_contract_ready": report.get("streaming_response_contract_ready") is True,
        "stage_local_kv_cache_ready": report.get("stage_local_kv_cache_ready") is True,
        "latency_metrics_ready": report.get("latency_metrics_ready") is True,
        "failure_requeue_ready": report.get("failure_requeue_ready") is True,
        "live_external_runtime_verified": report.get("live_external_runtime_verified") is True,
        "blocked_reason": str(report.get("blocked_reason") or ""),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
        "diagnosis_codes": ["heterogeneous_32b_serving_check_ready"] if not errors else ["heterogeneous_32b_serving_check_failed"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Heterogeneous 32B serving check ok: {result.get('ok')}")
        print(f"report: {result.get('report_path')}")
        if result.get("errors"):
            print("errors:")
            for error in result.get("errors") or []:
                print(f"- {error}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
