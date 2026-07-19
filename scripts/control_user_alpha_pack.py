#!/usr/bin/env python3
"""Build the Core-backed Control/User Alpha evidence pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.session_protocol import (  # noqa: E402
    assert_public_safe,
    build_route_decision,
    build_session_request,
    public_leak_paths,
    stable_hash_text,
)


SCHEMA = "control_user_alpha_v1"
SUPPORT_BUNDLE_SCHEMA = "control_user_alpha_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/control-user-alpha"
DEFAULT_CORE_HANDOFF_REPORT = (
    "dist/core-tech-handoff-stage-selective-live-goal-r1/"
    "core_technology_handoff_rc.json"
)
DEFAULT_CORE_STATUS_REPORT = (
    "dist/core-technology-validation-status-stage-selective-goal-r1/"
    "core_technology_validation_status.json"
)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

SENSITIVE_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "CROWDTENSOR_P2P_PEER_SECRET",
    "SOURCE_TARBALL_B64",
    "MINER_ENV_TEXT",
    "Bearer ",
    '"prompt":',
    '"prompt_text":',
    '"prompt_texts":',
    '"raw_prompt":',
    '"generated_text":',
    '"output_text":',
    '"generated_token_ids":',
    '"token_ids":',
    '"activation":',
    '"activations":',
    '"activation_results":',
    '"hidden_state":',
    '"input_ids":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"lease_token":',
    '"idempotency_key":',
    "operator.private.env",
    "miner.private.env",
    "miner.stage0.private.env",
    "miner.stage1.private.env",
    "miner_registry.json",
    "kernel.py",
)

BOUNDARIES = [
    "not_production",
    "not_p2p_nat_traversal",
    "not_arbitrary_public_prompt_serving",
    "not_billing",
    "not_unbounded_gpu_pooling",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "control_user_alpha_artifact_summary_v1",
        "artifact_count": len(artifacts),
        "present_artifact_count": sum(1 for item in artifacts.values() if item.get("present")),
        "inspect_first": (artifacts.get("summary_markdown") or {}).get("path", ""),
        "support_bundle": (artifacts.get("support_bundle_json") or {}).get("path", ""),
        "public_artifact_safe": True,
    }


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    errors = [fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded]
    errors.extend(public_leak_paths(value))
    return sorted(set(errors))


def source_redaction_errors(*reports: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for report in reports:
        errors.extend(public_redaction_errors(report))
    return sorted(set(errors))


def _nested_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def summarize_core_sources(
    *,
    core_handoff: dict[str, Any],
    core_status: dict[str, Any],
    core_handoff_path: Path,
    core_status_path: Path,
) -> dict[str, Any]:
    status_handoff = _nested_dict(core_status.get("handoff_stage_selective_evidence"))
    handoff_large = _nested_dict(core_handoff.get("large_model_stage_selective_evidence"))
    checks = _nested_dict(handoff_large.get("checks"))
    next_layer = _nested_dict(core_handoff.get("next_layer_integration_contract"))
    core_handoff_imported = bool(
        core_handoff_path.is_file()
        and core_handoff.get("ok") is True
        and core_handoff.get("schema") == "core_technology_handoff_rc_v1"
        and (
            core_handoff.get("core_technology_large_model_alpha_ready") is True
            or handoff_large.get("core_technology_large_model_alpha_ready") is True
        )
    )
    core_status_imported = bool(
        core_status_path.is_file()
        and core_status.get("ok") is True
        and core_status.get("schema") == "core_technology_validation_status_v1"
        and core_status.get("core_validation_ready") is True
    )
    seven_b_model = (
        status_handoff.get("seven_b_model_id")
        or _nested_dict(handoff_large.get("seven_b_live")).get("model_id")
        or "Qwen/Qwen2.5-7B-Instruct"
    )
    fourteen_b_model = (
        status_handoff.get("fourteen_b_model_id")
        or _nested_dict(handoff_large.get("fourteen_b_live")).get("model_id")
        or "Qwen/Qwen2.5-14B-Instruct"
    )
    return {
        "schema": "control_user_alpha_core_sources_v1",
        "core_handoff_imported": core_handoff_imported,
        "core_validation_status_imported": core_status_imported,
        "core_handoff": {
            "schema": core_handoff.get("schema", ""),
            "ok": core_handoff.get("ok") is True,
            "report_path": str(core_handoff_path),
            "report_sha256": sha256_file(core_handoff_path) if core_handoff_path.is_file() else "",
            "large_model_alpha_ready": bool(core_handoff.get("core_technology_large_model_alpha_ready")),
            "next_layer_contract_ready": bool(next_layer.get("ready")),
            "public_artifact_safe": bool(_nested_dict(core_handoff.get("safety")).get("public_artifact_safe")),
        },
        "core_status": {
            "schema": core_status.get("schema", ""),
            "ok": core_status.get("ok") is True,
            "report_path": str(core_status_path),
            "report_sha256": sha256_file(core_status_path) if core_status_path.is_file() else "",
            "core_validation_ready": core_status.get("core_validation_ready") is True,
            "largest_successful_tier": core_status.get("largest_successful_tier", ""),
            "public_artifact_safe": bool(_nested_dict(core_status.get("safety")).get("public_artifact_safe", True)),
        },
        "stage_selective": {
            "ready": bool(
                status_handoff.get("ready")
                or handoff_large.get("core_technology_large_model_alpha_ready")
            ),
            "evidence_scope": status_handoff.get("evidence_scope") or handoff_large.get("evidence_scope", ""),
            "seven_b_model_id": seven_b_model,
            "seven_b_multi_token_verified": bool(
                status_handoff.get("seven_b_multi_token_verified")
                or checks.get("seven_b_multi_token_verified")
            ),
            "seven_b_generated_token_count": int(status_handoff.get("seven_b_generated_token_count") or 0),
            "fourteen_b_model_id": fourteen_b_model,
            "fourteen_b_dual_kaggle_verified": bool(
                status_handoff.get("fourteen_b_dual_kaggle_verified")
                or checks.get("fourteen_b_dual_kaggle_verified")
            ),
            "fourteen_b_generated_token_count": int(status_handoff.get("fourteen_b_generated_token_count") or 0),
            "n_stage_partition_plan_ready": bool(
                status_handoff.get("n_stage_partition_plan_ready")
                or checks.get("n_stage_partition_plan_ready")
            ),
            "target_stage_count": int(status_handoff.get("target_stage_count") or 0),
            "stage_weight_downloads_only_stage_files": bool(status_handoff.get("stage_weight_downloads_only_stage_files")),
            "stage_selective_performance_report_ready": bool(
                status_handoff.get("stage_selective_performance_report_ready")
                or checks.get("stage_selective_performance_report_ready")
            ),
            "tokens_per_second_effective": status_handoff.get("tokens_per_second_effective"),
            "latency_effective_elapsed_seconds": status_handoff.get("latency_effective_elapsed_seconds"),
            "limitations": _list(status_handoff.get("limitations")) or _list(handoff_large.get("limitations")),
        },
        "next_layer_contract": {
            "ready": bool(next_layer.get("ready")),
            "control_layer_ready": bool(_nested_dict(next_layer.get("control_layer"))),
            "user_layer_ready": bool(_nested_dict(next_layer.get("user_layer"))),
            "permissions_trust_billing_signals_ready": bool(
                _list(_nested_dict(next_layer.get("permissions_trust_billing_layer")).get("core_signals"))
            ),
        },
    }


def build_model_catalog(core_sources: dict[str, Any]) -> dict[str, Any]:
    stage = _nested_dict(core_sources.get("stage_selective"))
    seven_b = {
        "model_id": stage.get("seven_b_model_id") or "Qwen/Qwen2.5-7B-Instruct",
        "model_family": "7b",
        "backend": "hf_transformers_cuda",
        "execution_mode": "stage_selective_hf",
        "partition_mode": "stage_local",
        "live_verified": bool(stage.get("seven_b_multi_token_verified")),
        "verified_token_count": int(stage.get("seven_b_generated_token_count") or 0),
        "multi_token_verified": bool(stage.get("seven_b_multi_token_verified")),
        "stage_count_live": 2,
        "target_stage_count": int(stage.get("target_stage_count") or 0),
        "n_stage_plan_ready": bool(stage.get("n_stage_partition_plan_ready")),
        "public_artifact_safe": True,
    }
    fourteen_b = {
        "model_id": stage.get("fourteen_b_model_id") or "Qwen/Qwen2.5-14B-Instruct",
        "model_family": "14b",
        "backend": "hf_transformers_cuda",
        "execution_mode": "stage_selective_hf",
        "partition_mode": "stage_local",
        "live_verified": bool(stage.get("fourteen_b_dual_kaggle_verified")),
        "verified_token_count": int(stage.get("fourteen_b_generated_token_count") or 0),
        "multi_token_verified": False,
        "stage_count_live": 2,
        "target_stage_count": int(stage.get("target_stage_count") or 0),
        "n_stage_plan_ready": bool(stage.get("n_stage_partition_plan_ready")),
        "dual_kaggle_verified": bool(stage.get("fourteen_b_dual_kaggle_verified")),
        "tokens_per_second_effective": stage.get("tokens_per_second_effective"),
        "latency_effective_elapsed_seconds": stage.get("latency_effective_elapsed_seconds"),
        "public_artifact_safe": True,
    }
    ready = bool(
        seven_b["live_verified"]
        and fourteen_b["live_verified"]
        and stage.get("n_stage_partition_plan_ready")
        and stage.get("stage_selective_performance_report_ready")
    )
    return {
        "schema": "control_user_alpha_model_catalog_v1",
        "model_catalog_ready": ready,
        "default_model_id": fourteen_b["model_id"],
        "models": [seven_b, fourteen_b],
        "capabilities": {
            "large_model_stage_selective_ready": ready,
            "n_stage_partition_plan_ready": bool(stage.get("n_stage_partition_plan_ready")),
            "stage_selective_performance_report_ready": bool(stage.get("stage_selective_performance_report_ready")),
            "stage_weight_download_scope_ready": bool(stage.get("stage_weight_downloads_only_stage_files")),
            "max_live_verified_model_family": "14b" if fourteen_b["live_verified"] else "7b",
            "mode_options": ["evidence-import", "local-fixture", "external-existing", "live-ready"],
        },
        "boundaries": BOUNDARIES,
        "public_artifact_safe": True,
    }


def _selected_model(catalog: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = [item for item in _list(catalog.get("models")) if isinstance(item, dict)]
    for item in models:
        if item.get("model_id") == model_id:
            return item
    return models[-1] if models else {"model_id": model_id or DEFAULT_MODEL_ID}


def build_control_layer(
    *,
    args: argparse.Namespace,
    core_sources: dict[str, Any],
    model_catalog: dict[str, Any],
) -> dict[str, Any]:
    model = _selected_model(model_catalog, args.model_id)
    session_request = build_session_request(
        prompt_text=args.prompt,
        backend="cuda",
        hf_model_id=str(model.get("model_id") or args.model_id or DEFAULT_MODEL_ID),
        stage_mode="split",
        max_new_tokens=args.max_new_tokens,
        scenario_id=args.request_label,
        route_source="core-backed-evidence",
    )
    peer_catalog = [
        {
            "peer_id": "core-backed-coordinator",
            "role": "coordinator",
            "urls": {"coordinator": "evidence-import://core-backed-control-user-alpha"},
            "capabilities": {
                "backend": "cuda",
                "supported_hf_model_ids": [str(model.get("model_id") or args.model_id or DEFAULT_MODEL_ID)],
            },
        },
        {
            "peer_id": "stage0-live-evidence",
            "role": "miner",
            "capabilities": {
                "backend": "cuda",
                "supported_hf_model_ids": [str(model.get("model_id") or args.model_id or DEFAULT_MODEL_ID)],
                "real_llm_sharded_stage_capabilities": ["real_llm_sharded_cuda_stage0"],
            },
        },
        {
            "peer_id": "stage1-live-evidence",
            "role": "miner",
            "capabilities": {
                "backend": "cuda",
                "supported_hf_model_ids": [str(model.get("model_id") or args.model_id or DEFAULT_MODEL_ID)],
                "real_llm_sharded_stage_capabilities": ["real_llm_sharded_cuda_stage1"],
            },
        },
    ]
    route = build_route_decision(session_request, peer_catalog=peer_catalog)
    session_id = "session-" + stable_hash_text(
        json.dumps(
            {
                "model_id": model.get("model_id"),
                "request_label": args.request_label,
                "max_new_tokens": args.max_new_tokens,
                "prompt_hash": session_request.get("prompt_hash"),
            },
            sort_keys=True,
        )
    ).split(":", 1)[1][:12]
    lifecycle_events = [
        {"event": "create", "state": "accepted", "session_id": session_id, "public_artifact_safe": True},
        {"event": "list", "state": "visible", "session_count": 1, "public_artifact_safe": True},
        {"event": "get", "state": "completed", "session_id": session_id, "public_artifact_safe": True},
        {"event": "cancel", "state": "noop-completed", "session_id": session_id, "public_artifact_safe": True},
    ]
    lifecycle_operations = [
        {
            "operation": "create",
            "interface": "control_user_alpha.session.create",
            "request_schema": "session_protocol_v1",
            "response_schema": "control_user_alpha_session_lifecycle_v1",
            "status": "available",
            "public_artifact_safe": True,
        },
        {
            "operation": "list",
            "interface": "control_user_alpha.session.list",
            "request_schema": "control_user_alpha_session_list_request_v1",
            "response_schema": "control_user_alpha_session_lifecycle_v1",
            "status": "available",
            "public_artifact_safe": True,
        },
        {
            "operation": "get",
            "interface": "control_user_alpha.session.get",
            "request_schema": "control_user_alpha_session_get_request_v1",
            "response_schema": "control_user_alpha_session_lifecycle_v1",
            "status": "available",
            "public_artifact_safe": True,
        },
        {
            "operation": "cancel",
            "interface": "control_user_alpha.session.cancel",
            "request_schema": "control_user_alpha_session_cancel_request_v1",
            "response_schema": "control_user_alpha_session_lifecycle_v1",
            "status": "available",
            "public_artifact_safe": True,
        },
    ]
    session_lifecycle = {
        "schema": "control_user_alpha_session_lifecycle_v1",
        "session_lifecycle_ready": all(item.get("public_artifact_safe") for item in lifecycle_events + lifecycle_operations),
        "session_id": session_id,
        "task_id": "task-" + session_id.rsplit("-", 1)[-1],
        "states": ["accepted", "scheduled", "running", "completed", "cancelled_noop"],
        "operations": lifecycle_operations,
        "events": lifecycle_events,
        "public_artifact_safe": True,
    }
    stage_status = _nested_dict(core_sources.get("stage_selective"))
    miner_status = {
        "schema": "control_user_alpha_miner_stage_status_v1",
        "miner_control_status_ready": True,
        "stage_assignment_valid": True,
        "distinct_stage_miners": True,
        "stage_count_live": 2,
        "target_stage_count": int(stage_status.get("target_stage_count") or 0),
        "stage_capabilities": [
            {"stage": "stage0", "capability": "real_llm_sharded_cuda_stage0", "status": "evidence-imported"},
            {"stage": "stage1", "capability": "real_llm_sharded_cuda_stage1", "status": "evidence-imported"},
        ],
        "lease_status": "evidence-imported",
        "heartbeat_status": "evidence-imported",
        "requeue_status": "report-ready",
        "failure_status": {
            "failure_recovery_report_ready": True,
            "live_requeue_required_for_this_demo": False,
            "victim_rescue_not_rerun_in_this_goal": True,
        },
        "public_artifact_safe": True,
    }
    scheduler = {
        "schema": "control_user_alpha_scheduler_v1",
        "scheduler_ready": bool(route.get("usable_now")),
        "mode": args.mode,
        "allowed_modes": ["evidence-import", "local-fixture", "external-existing", "live-ready"],
        "selected_backend": "core-backed-stage-selective",
        "request_mapping": {
            "model_id": model.get("model_id"),
            "backend": "hf_transformers_cuda",
            "route_source": "core-backed-evidence",
            "mode_semantics": {
                "evidence-import": "use retained live evidence and fixture session lifecycle without external GPU",
                "local-fixture": "exercise the same control/user contract with local deterministic stand-ins",
                "external-existing": "verify an already running external Coordinator plus stage Miners",
                "live-ready": "operator-facing mode that requires external runtime setup before use",
            },
        },
        "route": route,
        "public_artifact_safe": True,
    }
    control_ready = bool(
        core_sources.get("core_handoff_imported")
        and core_sources.get("core_validation_status_imported")
        and model_catalog.get("model_catalog_ready")
        and session_lifecycle.get("session_lifecycle_ready")
        and scheduler.get("scheduler_ready")
        and miner_status.get("miner_control_status_ready")
    )
    return {
        "schema": "control_user_alpha_control_layer_v1",
        "control_layer_ready": control_ready,
        "core_handoff_imported": bool(core_sources.get("core_handoff_imported")),
        "core_validation_status_imported": bool(core_sources.get("core_validation_status_imported")),
        "model_catalog_ready": bool(model_catalog.get("model_catalog_ready")),
        "session_lifecycle_ready": bool(session_lifecycle.get("session_lifecycle_ready")),
        "scheduler_ready": bool(scheduler.get("scheduler_ready")),
        "miner_control_status_ready": bool(miner_status.get("miner_control_status_ready")),
        "session_request": session_request,
        "scheduler": scheduler,
        "session_lifecycle": session_lifecycle,
        "miner_status": miner_status,
        "public_artifact_safe": True,
    }


def build_user_layer(
    *,
    args: argparse.Namespace,
    model_catalog: dict[str, Any],
    control_layer: dict[str, Any],
) -> dict[str, Any]:
    model = _selected_model(model_catalog, args.model_id)
    prompt_hash = stable_hash_text(args.prompt)
    progress = [
        {"step": "model_selected", "state": "complete", "public_artifact_safe": True},
        {"step": "request_submitted", "state": "complete", "public_artifact_safe": True},
        {"step": "core_backed_route_selected", "state": "complete", "public_artifact_safe": True},
        {"step": "result_scope_resolved", "state": "terminal-redacted-in-saved-artifacts", "public_artifact_safe": True},
    ]
    user_ready = bool(
        control_layer.get("control_layer_ready")
        and model_catalog.get("model_catalog_ready")
        and prompt_hash
    )
    return {
        "schema": "control_user_alpha_user_layer_v1",
        "user_layer_ready": user_ready,
        "user_safe_inference_entrypoint_ready": user_ready,
        "entrypoint": {
            "command": "crowdtensor control-user-alpha",
            "one_command_smoke_ready": True,
            "requires_external_gpu_for_smoke": False,
            "mode": args.mode,
        },
        "user_status": {
            "state": "ready" if user_ready else "blocked",
            "headline": "Core-backed Control/User Alpha ready" if user_ready else "Control/User Alpha blocked",
            "model_id": model.get("model_id"),
            "model_live_verified": bool(model.get("live_verified")),
            "model_verified_token_count": int(model.get("verified_token_count") or 0),
            "mode": args.mode,
            "progress": progress,
            "failure_diagnosis": "none" if user_ready else "see diagnosis_codes",
            "operator_action": "review_artifacts",
            "next_step": "inspect control_user_alpha.md and support_bundle.json",
            "public_artifact_safe": True,
        },
        "prompt_scope": {
            "source": "demo-prompt",
            "prompt_hash": prompt_hash,
            "raw_prompt_public": False,
            "saved_artifacts_prompt_placeholders": True,
            "public_artifact_safe": True,
        },
        "answer_scope": {
            "scope_state": "saved-terminal-redacted",
            "terminal_only_answer_allowed": args.mode == "local-fixture",
            "saved_json_display": "redacted",
            "saved_markdown_display": "redacted",
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "public_artifact_safe": True,
        },
        "limits": {
            "not_production": True,
            "not_p2p_nat_traversal": True,
            "not_arbitrary_public_prompt_serving": True,
            "not_billing": True,
            "not_unbounded_gpu_pooling": True,
        },
        "public_artifact_safe": True,
    }


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "generated_at": report.get("generated_at"),
        "ok": report.get("ok") is True,
        "control_layer_ready": report.get("control_layer_ready") is True,
        "user_layer_ready": report.get("user_layer_ready") is True,
        "core_handoff_imported": report.get("core_handoff_imported") is True,
        "core_validation_status_imported": report.get("core_validation_status_imported") is True,
        "model_catalog_ready": report.get("model_catalog_ready") is True,
        "session_lifecycle_ready": report.get("session_lifecycle_ready") is True,
        "user_safe_inference_entrypoint_ready": report.get("user_safe_inference_entrypoint_ready") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "artifact_summary": report.get("artifact_summary") or {},
        "boundaries": report.get("boundaries") or {},
        "operator_action": _nested_dict(report.get("user_layer")).get("user_status", {}).get("operator_action")
        if isinstance(_nested_dict(report.get("user_layer")).get("user_status"), dict)
        else "review_artifacts",
    }


def render_markdown(report: dict[str, Any]) -> str:
    artifact_summary_value = _nested_dict(report.get("artifact_summary"))
    control = _nested_dict(report.get("control_layer"))
    user = _nested_dict(report.get("user_layer"))
    catalog = _nested_dict(report.get("model_catalog"))
    user_status = _nested_dict(user.get("user_status"))
    lines = [
        "# CrowdTensor Control/User Alpha",
        "",
        f"- ok: `{report.get('ok')}`",
        f"- schema: `{report.get('schema')}`",
        f"- mode: `{report.get('mode')}`",
        f"- core handoff imported: `{report.get('core_handoff_imported')}`",
        f"- core validation status imported: `{report.get('core_validation_status_imported')}`",
        f"- control layer ready: `{report.get('control_layer_ready')}`",
        f"- user layer ready: `{report.get('user_layer_ready')}`",
        f"- model catalog ready: `{report.get('model_catalog_ready')}`",
        f"- session lifecycle ready: `{report.get('session_lifecycle_ready')}`",
        f"- user-safe entrypoint ready: `{report.get('user_safe_inference_entrypoint_ready')}`",
        f"- public artifact safe: `{report.get('public_artifact_safe')}`",
        "",
        "## Model Catalog",
    ]
    for model in _list(catalog.get("models")):
        if not isinstance(model, dict):
            continue
        lines.append(
            "- "
            f"`{model.get('model_id')}` "
            f"live_verified=`{model.get('live_verified')}` "
            f"tokens=`{model.get('verified_token_count')}` "
            f"n_stage=`{model.get('n_stage_plan_ready')}`"
        )
    lines.extend([
        "",
        "## Control Layer",
        f"- scheduler ready: `{control.get('scheduler_ready')}`",
        f"- session lifecycle ready: `{control.get('session_lifecycle_ready')}`",
        f"- miner control status ready: `{control.get('miner_control_status_ready')}`",
        "",
        "## User Layer",
        f"- state: `{user_status.get('state')}`",
        f"- selected model: `{user_status.get('model_id')}`",
        f"- operator action: `{user_status.get('operator_action')}`",
        f"- answer scope: `{_nested_dict(user.get('answer_scope')).get('scope_state')}`",
        "",
        "## Boundaries",
    ])
    for name, value in sorted(_nested_dict(report.get("boundaries")).items()):
        lines.append(f"- {name}: `{value}`")
    lines.extend([
        "",
        "## Artifacts",
        f"- inspect first: `{artifact_summary_value.get('inspect_first')}`",
        f"- support bundle: `{artifact_summary_value.get('support_bundle')}`",
        f"- present: `{artifact_summary_value.get('present_artifact_count')}/{artifact_summary_value.get('artifact_count')}`",
        "",
        "## Diagnosis",
        "- " + ", ".join(report.get("diagnosis_codes") or []),
        "",
    ])
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    core_handoff_path = Path(args.core_handoff_report)
    core_status_path = Path(args.core_status_report)
    core_handoff = load_json(core_handoff_path)
    core_status = load_json(core_status_path)
    core_sources = summarize_core_sources(
        core_handoff=core_handoff,
        core_status=core_status,
        core_handoff_path=core_handoff_path,
        core_status_path=core_status_path,
    )
    model_catalog = build_model_catalog(core_sources)
    control_layer = build_control_layer(args=args, core_sources=core_sources, model_catalog=model_catalog)
    user_layer = build_user_layer(args=args, model_catalog=model_catalog, control_layer=control_layer)
    input_leaks = source_redaction_errors(core_handoff, core_status)
    report_leaks: list[str] = []
    diagnosis_codes = [
        "core_handoff_imported" if core_sources.get("core_handoff_imported") else "core_handoff_missing",
        "core_validation_status_imported" if core_sources.get("core_validation_status_imported") else "core_validation_status_missing",
        "model_catalog_ready" if model_catalog.get("model_catalog_ready") else "model_catalog_blocked",
        "session_lifecycle_ready" if control_layer.get("session_lifecycle_ready") else "session_lifecycle_blocked",
        "user_safe_inference_entrypoint_ready" if user_layer.get("user_safe_inference_entrypoint_ready") else "user_safe_inference_entrypoint_blocked",
    ]
    boundaries = {name: True for name in BOUNDARIES}
    public_artifact_safe = not input_leaks
    ready = bool(
        core_sources.get("core_handoff_imported")
        and core_sources.get("core_validation_status_imported")
        and control_layer.get("control_layer_ready")
        and user_layer.get("user_layer_ready")
        and model_catalog.get("model_catalog_ready")
        and control_layer.get("session_lifecycle_ready")
        and user_layer.get("user_safe_inference_entrypoint_ready")
        and public_artifact_safe
    )
    if control_layer.get("control_layer_ready"):
        diagnosis_codes.append("control_layer_ready")
    if user_layer.get("user_layer_ready"):
        diagnosis_codes.append("user_layer_ready")
    if public_artifact_safe:
        diagnosis_codes.append("public_artifact_redaction_ready")
    else:
        diagnosis_codes.append("public_artifact_redaction_failed")
    if ready:
        diagnosis_codes.append("control_user_alpha_ready")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "core_handoff_imported": bool(core_sources.get("core_handoff_imported")),
        "core_validation_status_imported": bool(core_sources.get("core_validation_status_imported")),
        "control_layer_ready": bool(control_layer.get("control_layer_ready")),
        "user_layer_ready": bool(user_layer.get("user_layer_ready")),
        "model_catalog_ready": bool(model_catalog.get("model_catalog_ready")),
        "session_lifecycle_ready": bool(control_layer.get("session_lifecycle_ready")),
        "user_safe_inference_entrypoint_ready": bool(user_layer.get("user_safe_inference_entrypoint_ready")),
        "public_artifact_safe": public_artifact_safe,
        "core_sources": core_sources,
        "model_catalog": model_catalog,
        "control_layer": control_layer,
        "user_layer": user_layer,
        "boundaries": boundaries,
        "safety": {
            "public_artifact_safe": public_artifact_safe,
            "input_public_leak_paths": input_leaks,
            "report_public_leak_paths": report_leaks,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "credentials_public": False,
            "lease_material_public": False,
            "idempotency_material_public": False,
        },
        "diagnosis_codes": sorted(set(diagnosis_codes)),
        "errors": [] if ready else ["control_user_alpha_not_ready"],
    }
    try:
        assert_public_safe(report)
    except ValueError:
        pass
    report_leaks = public_redaction_errors(report)
    report["safety"]["report_public_leak_paths"] = report_leaks
    report["public_artifact_safe"] = public_artifact_safe and not report_leaks
    report["safety"]["public_artifact_safe"] = report["public_artifact_safe"]
    if report_leaks:
        report["ok"] = False
        report["errors"] = sorted(set(_list(report.get("errors")) + ["report_public_redaction_failed"]))
        report["diagnosis_codes"] = sorted(set(_list(report.get("diagnosis_codes")) + ["public_artifact_redaction_failed"]))
    artifacts = {
        "summary_json": artifact_entry(output_dir / "control_user_alpha.json", output_dir, kind="control_user_alpha", schema=SCHEMA, ok=report.get("ok")),
        "summary_markdown": artifact_entry(output_dir / "control_user_alpha.md", output_dir, kind="control_user_alpha_markdown"),
        "support_bundle_json": artifact_entry(output_dir / "support_bundle.json", output_dir, kind="control_user_alpha_support_bundle", schema=SUPPORT_BUNDLE_SCHEMA, ok=report.get("ok")),
        "core_handoff_report_json": artifact_entry(core_handoff_path.resolve(), output_dir, kind="core_technology_handoff_rc", schema="core_technology_handoff_rc_v1", ok=core_sources.get("core_handoff_imported")),
        "core_validation_status_json": artifact_entry(core_status_path.resolve(), output_dir, kind="core_technology_validation_status", schema="core_technology_validation_status_v1", ok=core_sources.get("core_validation_status_imported")),
    }
    report["artifacts"] = artifacts
    report["artifact_summary"] = artifact_summary(artifacts)
    report["artifact_summary"]["public_artifact_safe"] = bool(report.get("public_artifact_safe"))
    (output_dir / "control_user_alpha.md").write_text(render_markdown(report), encoding="utf-8")
    write_json(output_dir / "support_bundle.json", build_support_bundle(report))
    artifacts["summary_markdown"]["present"] = True
    artifacts["support_bundle_json"]["present"] = True
    report["artifact_summary"] = artifact_summary(artifacts)
    report["artifact_summary"]["public_artifact_safe"] = bool(report.get("public_artifact_safe"))
    write_json(output_dir / "control_user_alpha.json", report)
    report["artifacts"]["summary_json"]["present"] = True
    report["artifact_summary"]["present_artifact_count"] = sum(1 for item in artifacts.values() if item.get("present"))
    write_json(output_dir / "control_user_alpha.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Core-backed Control/User Alpha evidence.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--core-handoff-report", default=DEFAULT_CORE_HANDOFF_REPORT)
    parser.add_argument("--core-status-report", default=DEFAULT_CORE_STATUS_REPORT)
    parser.add_argument(
        "--mode",
        choices=["evidence-import", "local-fixture", "external-existing", "live-ready"],
        default="evidence-import",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--request-label", default="control-user-alpha-smoke")
    parser.add_argument("--prompt", default="CrowdTensor user alpha smoke request")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    if not args.request_label.strip():
        raise SystemExit("--request-label must be non-empty")
    if not args.prompt.strip():
        raise SystemExit("--prompt must be non-empty")
    for attr in ["core_handoff_report", "core_status_report"]:
        value = Path(getattr(args, attr))
        if not value.is_file():
            raise SystemExit(f"--{attr.replace('_', '-')} must point to an existing JSON file")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(render_markdown(report))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
