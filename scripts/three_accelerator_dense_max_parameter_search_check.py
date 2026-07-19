#!/usr/bin/env python3
"""Validate dense GPU+TPU+CPU max-parameter search artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import three_accelerator_dense_max_parameter_search_pack as pack  # noqa: E402


SCHEMA = "three_accelerator_dense_max_parameter_search_check_v1"


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
    if report.get("three_accelerator_dense_max_parameter_search_ready") is not True:
        errors.append("max_search_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    scope = _dict(report.get("goal_scope"))
    if scope.get("single_kaggle_account_only") is not True:
        errors.append("single_kaggle_account_scope_missing")
    if scope.get("multi_account_limit_bypass_allowed") is not False:
        errors.append("multi_account_limit_bypass_not_disallowed")
    if scope.get("dense_full_precision_main_path") is not True:
        errors.append("dense_full_precision_scope_missing")
    if scope.get("quantized_main_path_success_allowed") is not False:
        errors.append("quantized_success_not_disallowed")
    if set(scope.get("accelerators_required_for_frontier_success") or []) != {"cuda", "jax_tpu", "cpu"}:
        errors.append("accelerator_scope_mismatch")

    for field in [
        "max_successful_same_request_decode_parameter_class",
        "max_attempted_parameter_class",
        "max_attached_parameter_class",
        "max_stage_loaded_parameter_class",
        "max_tpu_executed_parameter_class",
        "generated_token_count",
        "accepted_stage_backends",
        "blocker_codes",
        "cleanup_status",
        "frontier_import",
        "same_request_32b_import",
        "same_request_72b_import",
        "dense_72b_attach_stage_plan_import",
        "web_tpu_execution_channel_import",
        "web_tpu_active_event_import",
        "web_tpu_start_wait_import",
        "dense_72b_tpu_stage_load_attempt",
        "colab_tpu_reacquire_import",
        "colab_tpu_runtime_stability_import",
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

    backends = set(report.get("accepted_stage_backends") or [])
    same_32b = _dict(report.get("same_request_32b_import"))
    same_72b = _dict(report.get("same_request_72b_import"))
    if same_32b.get("schema") != "dense_max_search_32b_same_request_import_v1":
        errors.append("same_request_32b_import_schema_mismatch")
    if report.get("max_successful_same_request_decode_parameter_class") == "32b":
        if same_32b.get("same_request_decode_verified") is not True:
            errors.append("max_decode_32b_without_bridge_proof")
        if _int(report.get("generated_token_count")) < 1:
            errors.append("max_decode_without_generated_token")
        if not {"cuda", "jax_tpu", "cpu"}.issubset(backends):
            errors.append("max_decode_without_three_backends")
    elif report.get("max_successful_same_request_decode_parameter_class") == "72b":
        if same_72b.get("schema") != "dense_max_search_72b_same_request_import_v1":
            errors.append("same_request_72b_import_schema_mismatch")
        if same_72b.get("same_request_full_model_decode_verified") is not True:
            errors.append("max_decode_72b_without_full_model_bridge_proof")
        if same_72b.get("full_72b_weight_loading_public_claim") is not True:
            errors.append("max_decode_72b_without_full_weight_claim")
        if _int(report.get("generated_token_count")) < 1:
            errors.append("max_decode_without_generated_token")
        if not {"cuda", "jax_tpu", "cpu"}.issubset(backends):
            errors.append("max_decode_without_three_backends")
    elif report.get("max_successful_same_request_decode_parameter_class"):
        errors.append("unknown_max_successful_decode_class")
    if same_72b and same_72b.get("schema") != "dense_max_search_72b_same_request_import_v1":
        errors.append("same_request_72b_import_schema_mismatch")
    if same_72b.get("same_request_stage_decode_verified") is True and same_72b.get("same_request_full_model_decode_verified") is not True:
        if "dense_72b_same_request_stage_verified_but_full_model_decode_not_verified" not in set(report.get("blocker_codes") or []):
            errors.append("missing_72b_stage_bridge_not_full_decode_blocker")
        if report.get("max_successful_same_request_decode_parameter_class") == "72b":
            errors.append("max_decode_72b_from_stage_bridge_only")

    if pack.parameter_value(str(report.get("max_successful_same_request_decode_parameter_class") or "")) > 32:
        ladder = [item for item in _list(report.get("attempt_ladder")) if isinstance(item, dict)]
        matching = [
            item
            for item in ladder
            if item.get("parameter_class") == report.get("max_successful_same_request_decode_parameter_class")
        ]
        if not matching or matching[0].get("same_request_decode_verified") is not True:
            errors.append("larger_decode_overclaim_without_same_request_ladder_proof")

    attach = _dict(report.get("dense_72b_attach_stage_plan_import"))
    if attach.get("schema") != "dense_max_search_72b_attach_stage_plan_import_v1":
        errors.append("attach_72b_import_schema_mismatch")
    if report.get("max_attached_parameter_class") == "72b":
        if attach.get("attach_verified") is not True:
            errors.append("max_attached_72b_without_attach_proof")
        if attach.get("dense_full_precision") is not True:
            errors.append("max_attached_72b_not_dense")
    if report.get("max_stage_preflighted_parameter_class") == "72b":
        if attach.get("stage_owned_preflight_verified") is not True:
            errors.append("max_stage_preflighted_72b_without_stage_plan")
        if not {"cuda", "jax_tpu", "cpu"}.issubset(set(attach.get("stage_plan_backends") or [])):
            errors.append("stage_preflight_72b_missing_backend")

    tpu_72b = _dict(report.get("dense_72b_tpu_stage_load_attempt"))
    if tpu_72b.get("schema") != "dense_max_search_72b_tpu_stage_load_attempt_import_v1":
        errors.append("tpu_72b_attempt_schema_mismatch")
    tpu_72b_ready = bool(tpu_72b.get("tpu_72b_stage_load_and_forward_verified") is True)
    if report.get("max_tpu_executed_parameter_class") == "72b":
        if not tpu_72b_ready:
            errors.append("max_tpu_executed_72b_without_tpu_stage_forward")
        if _int(tpu_72b.get("loaded_execution_tensor_key_count")) < 1:
            errors.append("max_tpu_executed_72b_without_loaded_keys")
        if _int(tpu_72b.get("executed_layer_count")) < 1:
            errors.append("max_tpu_executed_72b_without_layer_forward")
        if _int(tpu_72b.get("tpu_device_count")) < 1:
            errors.append("max_tpu_executed_72b_without_tpu_devices")
        if tpu_72b.get("stage_output_hash_present") is not True:
            errors.append("max_tpu_executed_72b_without_output_hash")
    elif report.get("max_tpu_executed_parameter_class") == "32b":
        frontier = _dict(report.get("frontier_import"))
        if frontier.get("same_request_dense_32b_success") is not True:
            errors.append("max_tpu_executed_32b_without_frontier_32b_success")
    elif report.get("max_tpu_executed_parameter_class"):
        errors.append("unknown_max_tpu_executed_class")

    if report.get("max_stage_loaded_parameter_class") == "72b" and not tpu_72b_ready:
        errors.append("max_stage_loaded_72b_without_72b_tpu_load_forward")
    if pack.parameter_value(str(report.get("max_stage_loaded_parameter_class") or "")) > 32 and not tpu_72b_ready:
        errors.append("larger_stage_loaded_overclaim_without_live_load")
    if report.get("max_attempted_parameter_class") == "72b" and report.get("max_successful_same_request_decode_parameter_class") != "72b":
        if not _list(report.get("blocker_codes")):
            errors.append("blocked_72b_attempt_without_blockers")
        if not str(report.get("failure_stage") or "").strip():
            errors.append("blocked_72b_attempt_without_failure_stage")
    if tpu_72b.get("tpu_72b_stage_load_and_forward_verified") is False:
        if "dense_72b_tpu_stage_load_and_forward_not_verified" not in set(report.get("blocker_codes") or []):
            errors.append("missing_72b_tpu_load_blocker")

    channel = _dict(report.get("web_tpu_execution_channel_import"))
    if channel.get("schema") != "dense_max_search_web_tpu_channel_import_v1":
        errors.append("web_tpu_channel_import_schema_mismatch")
    if channel.get("imported") is True:
        if channel.get("public_artifact_safe") is not True:
            errors.append("web_tpu_channel_public_artifact_unsafe")
        if channel.get("web_tpu_execution_channel_ready") is True:
            if channel.get("small_jax_cell_ready") is not True:
                errors.append("web_tpu_channel_ready_without_small_jax")
            if channel.get("tiny_qwen_like_cell_ready") is not True:
                errors.append("web_tpu_channel_ready_without_tiny_qwen")
            if _int(channel.get("tpu_device_count")) < 1:
                errors.append("web_tpu_channel_ready_without_tpu_device")
        else:
            if "web_tpu_execution_channel_not_ready" not in set(report.get("blocker_codes") or []):
                errors.append("web_tpu_channel_not_ready_without_blocker")
            if not str(channel.get("failure_stage") or "").strip():
                errors.append("web_tpu_channel_not_ready_without_failure_stage")
            if (
                report.get("failure_stage")
                and not str(report.get("failure_stage")).startswith("web_tpu_channel_")
                and not str(report.get("failure_stage")).startswith("web_tpu_active_event_")
                and report.get("failure_stage") != "colab_tpu_reacquire_not_ready"
            ):
                errors.append("report_failure_stage_should_reflect_current_channel_blocker")

    active_event = _dict(report.get("web_tpu_active_event_import"))
    active_event_overridden = bool(
        report.get("web_tpu_active_event_overridden_by_execution_channel") is True
        and channel.get("web_tpu_execution_channel_ready") is True
    )
    if active_event.get("schema") != "dense_max_search_web_tpu_active_event_import_v1":
        errors.append("web_tpu_active_event_import_schema_mismatch")
    if active_event.get("imported") is True:
        if active_event.get("public_artifact_safe") is not True:
            errors.append("web_tpu_active_event_public_artifact_unsafe")
        if _int(active_event.get("active_event_count")) < 1:
            errors.append("web_tpu_active_event_import_without_events")
        if active_event.get("active_event_runtime_ready") is True:
            if active_event.get("active_event_running") is not True:
                errors.append("web_tpu_active_event_ready_without_running_event")
            if active_event.get("jupyter_frame_visible") is not True:
                errors.append("web_tpu_active_event_ready_without_jupyter_frame")
            if active_event.get("jupyter_session_or_kernel_visible") is not True:
                errors.append("web_tpu_active_event_ready_without_session_or_kernel")
        elif active_event_overridden:
            if "web_tpu_active_event_not_ready" in set(report.get("blocker_codes") or []):
                errors.append("active_event_override_still_has_not_ready_blocker")
        else:
            if "web_tpu_active_event_not_ready" not in set(report.get("blocker_codes") or []):
                errors.append("web_tpu_active_event_not_ready_without_blocker")
            if not str(active_event.get("failure_stage") or "").strip():
                errors.append("web_tpu_active_event_not_ready_without_failure_stage")
            if (
                report.get("failure_stage")
                and not str(report.get("failure_stage")).startswith("web_tpu_active_event_")
                and not str(report.get("failure_stage")).startswith("web_tpu_channel_")
                and report.get("failure_stage") != "colab_tpu_reacquire_not_ready"
            ):
                errors.append("report_failure_stage_should_reflect_current_active_event_blocker")

    start_wait = _dict(report.get("web_tpu_start_wait_import"))
    start_wait_overridden = bool(
        report.get("web_tpu_start_wait_overridden_by_execution_channel") is True
        and channel.get("web_tpu_execution_channel_ready") is True
    )
    if start_wait.get("schema") != "dense_max_search_web_tpu_start_wait_import_v1":
        errors.append("web_tpu_start_wait_import_schema_mismatch")
    if start_wait.get("imported") is True:
        if start_wait.get("public_artifact_safe") is not True:
            errors.append("web_tpu_start_wait_public_artifact_unsafe")
        start_wait_blockers = set(start_wait.get("blockers") or [])
        start_click_missing_is_reported = (
            start_wait.get("web_tpu_ui_runtime_ready") is not True
            and "kaggle_web_tpu_start_session_not_clicked" in start_wait_blockers
        )
        if start_wait.get("start_clicked") is not True and not start_click_missing_is_reported:
            errors.append("web_tpu_start_wait_import_without_start_click")
        if start_wait.get("web_tpu_ui_runtime_ready") is not True:
            if start_wait_overridden:
                if "web_tpu_start_wait_runtime_not_ready" in set(report.get("blocker_codes") or []):
                    errors.append("web_tpu_start_wait_override_still_has_not_ready_blocker")
            elif "web_tpu_start_wait_runtime_not_ready" not in set(report.get("blocker_codes") or []):
                errors.append("web_tpu_start_wait_not_ready_without_blocker")
            if not start_wait_overridden and not str(start_wait.get("failure_stage") or "").strip():
                errors.append("web_tpu_start_wait_not_ready_without_failure_stage")

    colab_reacquire = _dict(report.get("colab_tpu_reacquire_import"))
    colab_reacquire_overridden = bool(
        report.get("colab_tpu_reacquire_overridden_by_web_tpu_channel") is True
        and channel.get("web_tpu_execution_channel_ready") is True
    )
    if colab_reacquire.get("schema") != "dense_max_search_colab_tpu_reacquire_import_v1":
        errors.append("colab_tpu_reacquire_import_schema_mismatch")
    if colab_reacquire.get("imported") is True:
        if colab_reacquire.get("public_artifact_safe") is not True:
            errors.append("colab_tpu_reacquire_public_artifact_unsafe")
        if _int(colab_reacquire.get("attempts_completed")) < 1:
            errors.append("colab_tpu_reacquire_import_without_attempts")
        if colab_reacquire.get("colab_tpu_reacquire_ready") is True:
            if _int(colab_reacquire.get("successful_attempt_index")) < 1:
                errors.append("colab_tpu_reacquire_ready_without_success_index")
            if colab_reacquire.get("endpoint_hash_present") is not True:
                errors.append("colab_tpu_reacquire_ready_without_endpoint_hash")
            if colab_reacquire.get("runtime_proxy_host_hash_present") is not True:
                errors.append("colab_tpu_reacquire_ready_without_proxy_host_hash")
        elif colab_reacquire_overridden:
            if "colab_tpu_reacquire_not_ready" in set(report.get("blocker_codes") or []):
                errors.append("colab_reacquire_override_still_has_not_ready_blocker")
        else:
            if "colab_tpu_reacquire_not_ready" not in set(report.get("blocker_codes") or []):
                errors.append("colab_tpu_reacquire_not_ready_without_blocker")
            if report.get("failure_stage") and report.get("failure_stage") != "colab_tpu_reacquire_not_ready":
                errors.append("report_failure_stage_should_reflect_current_colab_reacquire_blocker")

    colab_runtime = _dict(report.get("colab_tpu_runtime_stability_import"))
    colab_runtime_overridden = bool(report.get("colab_tpu_runtime_stability_overridden") is True)
    if colab_runtime.get("schema") != "dense_max_search_colab_tpu_runtime_stability_import_v1":
        errors.append("colab_tpu_runtime_stability_import_schema_mismatch")
    if colab_runtime.get("imported") is True:
        if colab_runtime.get("public_artifact_safe") is not True:
            errors.append("colab_tpu_runtime_stability_public_artifact_unsafe")
        if colab_runtime.get("colab_tpu_runtime_stably_acquired") is True:
            if _int(colab_runtime.get("rounds_completed")) < 1:
                errors.append("colab_tpu_runtime_ready_without_completed_round")
            if _int(colab_runtime.get("rounds_ready")) < 1:
                errors.append("colab_tpu_runtime_ready_without_ready_round")
            if _int(colab_runtime.get("observed_device_count_max")) < 1:
                errors.append("colab_tpu_runtime_ready_without_tpu_device")
            if colab_runtime.get("endpoint_hash_present") is not True:
                errors.append("colab_tpu_runtime_ready_without_endpoint_hash")
            if colab_runtime.get("runtime_proxy_host_hash_present") is not True:
                errors.append("colab_tpu_runtime_ready_without_proxy_host_hash")
        elif colab_runtime_overridden:
            if "colab_tpu_runtime_stability_not_ready" in set(report.get("blocker_codes") or []):
                errors.append("colab_runtime_override_still_has_not_ready_blocker")
        else:
            if "colab_tpu_runtime_stability_not_ready" not in set(report.get("blocker_codes") or []):
                errors.append("colab_tpu_runtime_stability_not_ready_without_blocker")

    cleanup = _dict(report.get("cleanup_status"))
    if cleanup.get("temporary_kaggle_kernels_deleted") is not True:
        errors.append("cleanup_kernel_deleted_missing")
    if cleanup.get("temporary_private_packages_removed") is not True:
        errors.append("cleanup_private_packages_removed_missing")
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_left_running")

    artifacts = _dict(report.get("artifacts"))
    for artifact_name in [
        "summary_json",
        "support_bundle_json",
        "frontier_json",
        "same_request_32b_json",
        "attach_72b_stage_plan_json",
        "tpu_72b_stage_load_attempt_json",
    ]:
        if _dict(artifacts.get(artifact_name)).get("present") is not True:
            errors.append(f"artifact_missing:{artifact_name}")
    if channel.get("imported") is True and _dict(artifacts.get("web_tpu_channel_json")).get("present") is not True:
        errors.append("artifact_missing:web_tpu_channel_json")
    if active_event.get("imported") is True and _dict(artifacts.get("web_tpu_active_event_json")).get("present") is not True:
        errors.append("artifact_missing:web_tpu_active_event_json")
    if start_wait.get("imported") is True and _dict(artifacts.get("web_tpu_start_wait_json")).get("present") is not True:
        errors.append("artifact_missing:web_tpu_start_wait_json")
    if colab_reacquire.get("imported") is True and _dict(artifacts.get("colab_tpu_reacquire_json")).get("present") is not True:
        errors.append("artifact_missing:colab_tpu_reacquire_json")
    if colab_runtime.get("imported") is True and _dict(artifacts.get("colab_tpu_runtime_stability_json")).get("present") is not True:
        errors.append("artifact_missing:colab_tpu_runtime_stability_json")
    return errors


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report = load_json(Path(args.report))
        report_path = args.report
    else:
        pack_args = [
            "--output-dir",
            args.output_dir,
            "--frontier-report",
            args.frontier_report,
            "--bridge-32b-report",
            args.bridge_32b_report,
            "--bridge-72b-report",
            args.bridge_72b_report,
            "--attach-72b-stage-plan-report",
            args.attach_72b_stage_plan_report,
            "--tpu-72b-stage-load-report",
            args.tpu_72b_stage_load_report,
        ]
        if args.web_tpu_channel_report:
            pack_args.extend(["--web-tpu-channel-report", args.web_tpu_channel_report])
        if args.web_tpu_active_event_report:
            pack_args.extend(["--web-tpu-active-event-report", args.web_tpu_active_event_report])
        if args.web_tpu_start_wait_report:
            pack_args.extend(["--web-tpu-start-wait-report", args.web_tpu_start_wait_report])
        if args.colab_tpu_reacquire_report:
            pack_args.extend(["--colab-tpu-reacquire-report", args.colab_tpu_reacquire_report])
        if args.colab_tpu_runtime_stability_report:
            pack_args.extend(["--colab-tpu-runtime-stability-report", args.colab_tpu_runtime_stability_report])
        report = pack.build_report(pack.parse_args(pack_args))
        report_path = str(Path(args.output_dir) / "three_accelerator_dense_max_parameter_search.json")
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": report_path,
        "three_accelerator_dense_max_parameter_search_ready": report.get(
            "three_accelerator_dense_max_parameter_search_ready"
        )
        is True,
        "max_successful_same_request_decode_parameter_class": report.get(
            "max_successful_same_request_decode_parameter_class"
        ),
        "max_attempted_parameter_class": report.get("max_attempted_parameter_class"),
        "max_attached_parameter_class": report.get("max_attached_parameter_class"),
        "max_stage_loaded_parameter_class": report.get("max_stage_loaded_parameter_class"),
        "max_tpu_executed_parameter_class": report.get("max_tpu_executed_parameter_class"),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dense GPU+TPU+CPU max-parameter search artifact.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frontier-report", default=pack.DEFAULT_FRONTIER_REPORT)
    parser.add_argument("--bridge-32b-report", default=pack.DEFAULT_32B_BRIDGE_REPORT)
    parser.add_argument("--bridge-72b-report", default=pack.DEFAULT_72B_BRIDGE_REPORT)
    parser.add_argument("--attach-72b-stage-plan-report", default=pack.DEFAULT_72B_ATTACH_STAGE_PLAN_REPORT)
    parser.add_argument("--tpu-72b-stage-load-report", default=pack.DEFAULT_72B_TPU_STAGE_LOAD_REPORT)
    parser.add_argument("--web-tpu-channel-report", default=pack.DEFAULT_WEB_TPU_CHANNEL_REPORT)
    parser.add_argument("--web-tpu-active-event-report", default=pack.DEFAULT_WEB_TPU_ACTIVE_EVENT_REPORT)
    parser.add_argument("--web-tpu-start-wait-report", default=pack.DEFAULT_WEB_TPU_START_WAIT_REPORT)
    parser.add_argument("--colab-tpu-reacquire-report", default=pack.DEFAULT_COLAB_TPU_REACQUIRE_REPORT)
    parser.add_argument("--colab-tpu-runtime-stability-report", default=pack.DEFAULT_COLAB_TPU_RUNTIME_STABILITY_REPORT)
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
