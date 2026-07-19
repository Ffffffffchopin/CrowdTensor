#!/usr/bin/env python3
"""Validate GPU Swarm Usability Alpha evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_swarm_usability_alpha_pack as pack  # noqa: E402


SCHEMA = "gpu_swarm_usability_alpha_check_v1"


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
    for field in [
        "gpu_swarm_usability_alpha_ready",
        "user_gpu_swarm_entrypoint_ready",
        "gpu_miner_join_pack_ready",
        "coordinator_workflow_ready",
        "two_gpu_stage_route_ready",
        "inference_request_lifecycle_ready",
        "model_catalog_imported",
        "control_user_alpha_imported",
        "core_handoff_imported",
        "public_artifact_safe",
    ]:
        if report.get(field) is not True:
            errors.append(f"{field}_missing")
    if report.get("execution_mode") not in pack.EXECUTION_MODES:
        errors.append("execution_mode_invalid")
    if report.get("action") not in pack.ACTIONS:
        errors.append("action_invalid")
    if report.get("selected_stage") not in pack.STAGES:
        errors.append("selected_stage_invalid")
    if not isinstance(report.get("external_runtime_verified"), bool):
        errors.append("external_runtime_verified_not_boolean")

    boundaries = _dict(report.get("boundaries"))
    for field in pack.BOUNDARIES:
        if boundaries.get(field) is not True:
            errors.append(f"boundary_missing:{field}")

    diagnosis = set(report.get("diagnosis_codes") or [])
    for code in [
        "gpu_swarm_usability_alpha_ready",
        "user_gpu_swarm_entrypoint_ready",
        "gpu_miner_join_pack_ready",
        "coordinator_workflow_ready",
        "two_gpu_stage_route_ready",
        "inference_request_lifecycle_ready",
        "model_catalog_imported",
        "control_user_alpha_imported",
        "core_handoff_imported",
        "gpu_swarm_public_artifact_redaction_ready",
    ]:
        if code not in diagnosis:
            errors.append(f"diagnosis_missing:{code}")

    catalog = _dict(report.get("model_catalog"))
    model_ids = {str(item.get("model_id") or "") for item in _list(catalog.get("models")) if isinstance(item, dict)}
    if "Qwen/Qwen2.5-7B-Instruct" not in model_ids:
        errors.append("model_catalog_missing_7b")
    if "Qwen/Qwen2.5-14B-Instruct" not in model_ids:
        errors.append("model_catalog_missing_14b")
    capabilities = _dict(catalog.get("capabilities"))
    for field in ["n_stage_partition_plan_ready", "stage_selective_performance_report_ready"]:
        if capabilities.get(field) is not True:
            errors.append(f"capability_missing:{field}")

    coordinator = _dict(report.get("coordinator_workflow"))
    if coordinator.get("coordinator_workflow_ready") is not True:
        errors.append("coordinator_workflow_not_ready")
    if not coordinator.get("start_command"):
        errors.append("coordinator_start_command_missing")

    join = _dict(report.get("miner_join_packs"))
    stages = [item for item in _list(join.get("stages")) if isinstance(item, dict)]
    if len(stages) != 2:
        errors.append("stage_join_pack_count_mismatch")
    stage_roles = {str(item.get("stage") or "") for item in stages}
    if stage_roles != {"stage0", "stage1"}:
        errors.append("stage_roles_mismatch")
    capabilities_seen = {str(item.get("required_capability") or "") for item in stages}
    if capabilities_seen != {"real_llm_sharded_cuda_stage0", "real_llm_sharded_cuda_stage1"}:
        errors.append("stage_capabilities_mismatch")
    for item in stages:
        if item.get("backend") != "hf_transformers_cuda":
            errors.append(f"stage_backend_mismatch:{item.get('stage')}")
        if item.get("stage_owned_weight_loading_required") is not True:
            errors.append(f"stage_weight_scope_missing:{item.get('stage')}")
        if f"${{{pack.SAFE_MINER_TOKEN_ENV}}}" not in str(item.get("private_token_placeholder") or ""):
            errors.append(f"stage_token_placeholder_missing:{item.get('stage')}")
        if item.get("private_token_env") != pack.SAFE_MINER_TOKEN_ENV:
            errors.append(f"stage_token_env_missing:{item.get('stage')}")
        if pack.SAFE_MINER_TOKEN_ENV not in str(item.get("command_template") or ""):
            errors.append(f"stage_command_token_placeholder_missing:{item.get('stage')}")
        if not item.get("recommended_command"):
            errors.append(f"stage_recommended_command_missing:{item.get('stage')}")

    readiness = _dict(report.get("gpu_readiness"))
    if readiness.get("gpu_readiness_report_ready") is not True:
        errors.append("gpu_readiness_report_missing")
    if len(_list(readiness.get("checks"))) != 2:
        errors.append("gpu_readiness_stage_count_mismatch")

    lifecycle = _dict(report.get("inference_lifecycle"))
    if lifecycle.get("inference_request_lifecycle_ready") is not True:
        errors.append("inference_lifecycle_not_ready")
    events = {str(item.get("event") or "") for item in _list(lifecycle.get("events")) if isinstance(item, dict)}
    for event in ["prepare", "coordinator_plan", "miner_join_plan", "infer_request", "status", "collect"]:
        if event not in events:
            errors.append(f"lifecycle_event_missing:{event}")
    result_scope = _dict(lifecycle.get("result_scope"))
    for field in ["raw_prompt_public", "raw_generated_text_public", "generated_token_ids_public", "activation_public"]:
        if result_scope.get(field) is not False:
            errors.append(f"result_scope_public_flag_mismatch:{field}")

    workflow = _dict(report.get("user_workflow"))
    if workflow.get("user_gpu_swarm_entrypoint_ready") is not True:
        errors.append("user_workflow_not_ready")
    labels = {str(item.get("label") or "") for item in _list(workflow.get("next_commands")) if isinstance(item, dict)}
    for label in ["prepare", "coordinator", "miner-stage0", "miner-stage1", "infer", "status", "collect"]:
        if label not in labels:
            errors.append(f"next_command_missing:{label}")

    cleanup = _dict(report.get("cleanup_plan"))
    if cleanup.get("cleanup_ready") is not True:
        errors.append("cleanup_plan_missing")
    if cleanup.get("dry_run_default") is not True:
        errors.append("cleanup_dry_run_default_missing")
    if cleanup.get("private_env_written") is not False:
        errors.append("cleanup_private_env_flag_mismatch")

    safety = _dict(report.get("safety"))
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_mismatch")
    if safety.get("report_public_leak_paths"):
        errors.append("report_public_leak_paths_present")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    artifacts = _dict(report.get("artifacts"))
    for name in [
        "summary_json",
        "runbook_markdown",
        "support_bundle_json",
        "stage0_join_script",
        "stage1_join_script",
        "stage0_join_runbook",
        "stage1_join_runbook",
    ]:
        if _dict(artifacts.get(name)).get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return errors


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report = load_json(Path(args.report))
    else:
        pack_args = pack.parse_args([
            "--output-dir",
            args.output_dir,
            "--control-user-alpha-report",
            args.control_user_alpha_report,
            "--core-handoff-report",
            args.core_handoff_report,
            "--core-status-report",
            args.core_status_report,
            "--execution-mode",
            args.execution_mode,
            "--coordinator-url",
            args.coordinator_url,
            "--port",
            str(args.port),
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
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "output_dir": report.get("output_dir") or args.output_dir,
        "report_path": args.report or str(Path(args.output_dir) / "gpu_swarm_usability_alpha.json"),
        "gpu_swarm_usability_alpha_ready": report.get("gpu_swarm_usability_alpha_ready") is True,
        "user_gpu_swarm_entrypoint_ready": report.get("user_gpu_swarm_entrypoint_ready") is True,
        "gpu_miner_join_pack_ready": report.get("gpu_miner_join_pack_ready") is True,
        "coordinator_workflow_ready": report.get("coordinator_workflow_ready") is True,
        "two_gpu_stage_route_ready": report.get("two_gpu_stage_route_ready") is True,
        "inference_request_lifecycle_ready": report.get("inference_request_lifecycle_ready") is True,
        "model_catalog_imported": report.get("model_catalog_imported") is True,
        "control_user_alpha_imported": report.get("control_user_alpha_imported") is True,
        "core_handoff_imported": report.get("core_handoff_imported") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "execution_mode": report.get("execution_mode"),
        "external_runtime_verified": report.get("external_runtime_verified") is True,
        "errors": errors,
        "diagnosis_codes": ["gpu_swarm_usability_alpha_check_ready"] if not errors else ["gpu_swarm_usability_alpha_check_failed"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPU Swarm Usability Alpha evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--control-user-alpha-report", default=pack.DEFAULT_CONTROL_USER_ALPHA_REPORT)
    parser.add_argument("--core-handoff-report", default=pack.DEFAULT_CORE_HANDOFF_REPORT)
    parser.add_argument("--core-status-report", default=pack.DEFAULT_CORE_STATUS_REPORT)
    parser.add_argument("--execution-mode", choices=pack.EXECUTION_MODES, default="evidence-import")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:9300")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--model-id", default=pack.DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--request-label", default="gpu-swarm-alpha-check")
    parser.add_argument("--prompt", default="CrowdTensor GPU swarm alpha check request")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.report and not Path(args.report).is_file():
        raise SystemExit("--report must point to an existing JSON file")
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"GPU Swarm Usability Alpha check ready: {result.get('ok')}")
        if result.get("errors"):
            print("errors: " + ", ".join(result.get("errors") or []))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
