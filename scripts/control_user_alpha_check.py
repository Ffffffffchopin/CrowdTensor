#!/usr/bin/env python3
"""Validate the Core-backed Control/User Alpha evidence pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import control_user_alpha_pack as pack  # noqa: E402


SCHEMA = "control_user_alpha_check_v1"


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
    for field in [
        "core_handoff_imported",
        "core_validation_status_imported",
        "control_layer_ready",
        "user_layer_ready",
        "model_catalog_ready",
        "session_lifecycle_ready",
        "user_safe_inference_entrypoint_ready",
        "public_artifact_safe",
    ]:
        if report.get(field) is not True:
            errors.append(f"{field}_missing")
    if report.get("ok") is not True:
        errors.append("report_not_ok")

    diagnosis = set(report.get("diagnosis_codes") or [])
    required_codes = {
        "core_handoff_imported",
        "core_validation_status_imported",
        "control_layer_ready",
        "user_layer_ready",
        "model_catalog_ready",
        "session_lifecycle_ready",
        "user_safe_inference_entrypoint_ready",
        "public_artifact_redaction_ready",
        "control_user_alpha_ready",
    }
    for code in sorted(required_codes):
        if code not in diagnosis:
            errors.append(f"diagnosis_missing:{code}")

    boundaries = _dict(report.get("boundaries"))
    for name in pack.BOUNDARIES:
        if boundaries.get(name) is not True:
            errors.append(f"boundary_missing:{name}")

    catalog = _dict(report.get("model_catalog"))
    models = [item for item in _list(catalog.get("models")) if isinstance(item, dict)]
    model_ids = {str(item.get("model_id") or "") for item in models}
    if "Qwen/Qwen2.5-7B-Instruct" not in model_ids:
        errors.append("model_catalog_missing_7b")
    if "Qwen/Qwen2.5-14B-Instruct" not in model_ids:
        errors.append("model_catalog_missing_14b")
    capabilities = _dict(catalog.get("capabilities"))
    for field in [
        "large_model_stage_selective_ready",
        "n_stage_partition_plan_ready",
        "stage_selective_performance_report_ready",
    ]:
        if capabilities.get(field) is not True:
            errors.append(f"capability_missing:{field}")

    control = _dict(report.get("control_layer"))
    scheduler = _dict(control.get("scheduler"))
    route = _dict(scheduler.get("route"))
    lifecycle = _dict(control.get("session_lifecycle"))
    miner_status = _dict(control.get("miner_status"))
    if scheduler.get("scheduler_ready") is not True:
        errors.append("scheduler_not_ready")
    if route.get("usable_now") is not True:
        errors.append("route_not_usable")
    events = [item.get("event") for item in _list(lifecycle.get("events")) if isinstance(item, dict)]
    for event in ["create", "list", "get", "cancel"]:
        if event not in events:
            errors.append(f"lifecycle_event_missing:{event}")
    operations = [item.get("operation") for item in _list(lifecycle.get("operations")) if isinstance(item, dict)]
    for operation in ["create", "list", "get", "cancel"]:
        if operation not in operations:
            errors.append(f"lifecycle_operation_missing:{operation}")
    if miner_status.get("stage_assignment_valid") is not True:
        errors.append("stage_assignment_invalid")
    if miner_status.get("distinct_stage_miners") is not True:
        errors.append("distinct_stage_miners_missing")
    stage_caps = {
        str(item.get("capability") or "")
        for item in _list(miner_status.get("stage_capabilities"))
        if isinstance(item, dict)
    }
    for capability in ["real_llm_sharded_cuda_stage0", "real_llm_sharded_cuda_stage1"]:
        if capability not in stage_caps:
            errors.append(f"stage_capability_missing:{capability}")
    if _dict(miner_status.get("failure_status")).get("failure_recovery_report_ready") is not True:
        errors.append("failure_status_missing")

    user = _dict(report.get("user_layer"))
    if user.get("user_safe_inference_entrypoint_ready") is not True:
        errors.append("user_entrypoint_not_ready")
    if _dict(user.get("entrypoint")).get("one_command_smoke_ready") is not True:
        errors.append("one_command_smoke_missing")
    answer_scope = _dict(user.get("answer_scope"))
    if answer_scope.get("scope_state") != "saved-terminal-redacted":
        errors.append("answer_scope_state_mismatch")
    for field in ["raw_generated_text_public", "generated_token_ids_public", "activation_public"]:
        if answer_scope.get(field) is not False:
            errors.append(f"answer_scope_public_flag_mismatch:{field}")
    prompt_scope = _dict(user.get("prompt_scope"))
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("raw_prompt_public_mismatch")

    safety = _dict(report.get("safety"))
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_mismatch")
    if safety.get("input_public_leak_paths"):
        errors.append("input_public_leak_paths_present")
    if safety.get("report_public_leak_paths"):
        errors.append("report_public_leak_paths_present")
    leak_errors = pack.public_redaction_errors(report)
    if leak_errors:
        errors.append("public_redaction_scan_failed:" + ",".join(leak_errors[:8]))

    artifact_summary = _dict(report.get("artifact_summary"))
    if artifact_summary.get("public_artifact_safe") is not True:
        errors.append("artifact_summary_public_artifact_safe_mismatch")
    artifacts = _dict(report.get("artifacts"))
    for name in ["summary_json", "summary_markdown", "support_bundle_json"]:
        artifact = _dict(artifacts.get(name))
        if artifact.get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return errors


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report = load_json(Path(args.report))
    else:
        pack_args = pack.parse_args([
            "--output-dir",
            args.output_dir,
            "--core-handoff-report",
            args.core_handoff_report,
            "--core-status-report",
            args.core_status_report,
            "--mode",
            args.mode,
            "--model-id",
            args.model_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--request-label",
            args.request_label,
            "--prompt",
            args.prompt,
        ])
        report = pack.build_report(pack_args)
    errors = validate_report(report)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "output_dir": report.get("output_dir") or args.output_dir,
        "report_path": args.report or str(Path(args.output_dir) / "control_user_alpha.json"),
        "control_layer_ready": report.get("control_layer_ready") is True,
        "user_layer_ready": report.get("user_layer_ready") is True,
        "core_handoff_imported": report.get("core_handoff_imported") is True,
        "core_validation_status_imported": report.get("core_validation_status_imported") is True,
        "model_catalog_ready": report.get("model_catalog_ready") is True,
        "session_lifecycle_ready": report.get("session_lifecycle_ready") is True,
        "user_safe_inference_entrypoint_ready": report.get("user_safe_inference_entrypoint_ready") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
        "diagnosis_codes": ["control_user_alpha_check_ready"] if not errors else ["control_user_alpha_check_failed"],
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Core-backed Control/User Alpha evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--core-handoff-report", default=pack.DEFAULT_CORE_HANDOFF_REPORT)
    parser.add_argument("--core-status-report", default=pack.DEFAULT_CORE_STATUS_REPORT)
    parser.add_argument(
        "--mode",
        choices=["evidence-import", "local-fixture", "external-existing", "live-ready"],
        default="evidence-import",
    )
    parser.add_argument("--model-id", default=pack.DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--request-label", default="control-user-alpha-check")
    parser.add_argument("--prompt", default="CrowdTensor user alpha check request")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.report and not Path(args.report).is_file():
        raise SystemExit("--report must point to an existing JSON file")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Control/User Alpha check ready: {result.get('ok')}")
        if result.get("errors"):
            print("errors: " + ", ".join(result.get("errors") or []))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
