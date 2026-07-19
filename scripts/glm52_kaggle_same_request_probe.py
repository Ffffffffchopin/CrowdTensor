#!/usr/bin/env python3
"""Assemble public-safe GLM 5.2 Kaggle CPU/GPU/TPU same-request proof.

This script is intentionally conservative. In preflight mode it only writes a
not-started blocker report. In assemble mode it accepts already produced stage,
Coordinator, and cleanup reports, strips them down to public-safe proof fields,
and marks success only when all three Kaggle provider families participated in
one live Coordinator request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_same_request_probe_v1"
STAGE_SCHEMA = "glm52_kaggle_stage_runtime_report_v1"
MODEL_ID = "zai-org/GLM-5.2"
COMPATIBLE_WEIGHT_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
REQUIRED_PROVIDERS = ["kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"]
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-same-request"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"raw_generated_text":',
    '"generated_token_ids":',
    '"input_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
)
STAGE_SMOKE_SCHEMAS = {
    "glm52_awq_tpu_stage_smoke_v1",
    "glm52_kaggle_tpu_awq_stage_smoke_watch_v1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def safety_flags() -> dict[str, bool]:
    return {
        "public_artifact_safe": True,
        "credentials_public": False,
        "cookies_public": False,
        "signed_url_public": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "hidden_state_public": False,
        "logits_public": False,
        "kv_cache_public": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
    }


def _model_id(report: dict[str, Any]) -> str:
    return str(
        report.get("model_id")
        or _dict(report.get("model")).get("model_id")
        or _dict(report.get("same_request")).get("model_id")
        or ""
    )


def _request_hash(report: dict[str, Any]) -> str:
    same = _dict(report.get("same_request"))
    return str(
        report.get("coordinator_request_id_hash")
        or report.get("request_id_hash")
        or same.get("coordinator_request_id_hash")
        or same.get("request_id_hash")
        or ""
    )


def _stage_output_hash(report: dict[str, Any]) -> str:
    return str(
        report.get("stage_output_hash")
        or report.get("output_hash")
        or report.get("activation_handoff_hash")
        or _dict(report.get("stage")).get("stage_output_hash")
        or ""
    )


def _stage_weight_value_hash(report: dict[str, Any]) -> str:
    return str(
        report.get("stage_weight_value_hash")
        or report.get("weight_value_sha256")
        or report.get("weight_value_hash")
        or _dict(report.get("stage")).get("stage_weight_value_hash")
        or ""
    )


def _stage_weight_values_loaded(report: dict[str, Any]) -> bool:
    return bool(
        (
            report.get("stage_owned_weight_values_loaded") is True
            or report.get("weight_tensor_values_loaded") is True
        )
        and _hash_ok(_stage_weight_value_hash(report))
        and _int(report.get("weight_value_byte_count") or report.get("stage_weight_value_byte_count")) > 0
        and report.get("weight_tensor_values_public") is not True
    )


def normalize_stage_report(report: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    source_schema = str(report.get("schema") or "")
    provider = str(report.get("provider") or report.get("backend") or report.get("stage_provider") or "")
    stage_id = _int(report.get("stage_id"), ordinal)
    stage_output_hash = _stage_output_hash(report)
    weight_value_hash = _stage_weight_value_hash(report)
    weight_values_loaded = _stage_weight_values_loaded(report)
    request_hash = _request_hash(report)
    model_id = _model_id(report)
    blockers = [str(item) for item in _list(report.get("blockers")) if item]

    stage_smoke_only = source_schema in STAGE_SMOKE_SCHEMAS or report.get("stage_smoke_only") is True
    if stage_smoke_only:
        blockers.append("stage_report_is_stage_smoke_only")
    if provider not in REQUIRED_PROVIDERS:
        blockers.append(f"stage_provider_not_required:{provider or 'missing'}")
    if model_id != MODEL_ID:
        blockers.append("stage_model_id_not_glm52" if model_id else "stage_model_id_missing")
    if not _hash_ok(request_hash):
        blockers.append("stage_coordinator_request_hash_missing")
    if not _hash_ok(stage_output_hash):
        blockers.append("stage_output_hash_missing")
    if report.get("public_artifact_safe") is not True:
        blockers.append("stage_public_artifact_unsafe")
    if report.get("live_run_performed") is not True:
        blockers.append("stage_live_run_not_performed")
    if not weight_values_loaded:
        blockers.append("stage_weight_values_not_loaded")
    if report.get("stage_decode_verified") is not True:
        blockers.append("stage_decode_not_verified")

    stage_ready = bool(
        not stage_smoke_only
        and report.get("stage_decode_verified") is True
        and weight_values_loaded
        and _hash_ok(stage_output_hash)
        and _hash_ok(request_hash)
        and provider in REQUIRED_PROVIDERS
        and model_id == MODEL_ID
        and report.get("live_run_performed") is True
        and report.get("public_artifact_safe") is True
    )
    return {
        "schema": STAGE_SCHEMA,
        "source_schema": source_schema,
        "stage_id": stage_id,
        "provider": provider,
        "model_id": model_id,
        "compatible_weight_repo": str(report.get("compatible_weight_repo") or report.get("model_repo") or ""),
        "coordinator_request_id_hash": request_hash,
        "stage_layer_range": _list(report.get("stage_layer_range")),
        "stage_execution_verified": stage_ready,
        "stage_decode_verified": report.get("stage_decode_verified") is True and stage_ready,
        "stage_output_hash": stage_output_hash if _hash_ok(stage_output_hash) else "",
        "weight_value_hash_present": _hash_ok(weight_value_hash),
        "weight_value_byte_count": _int(report.get("weight_value_byte_count") or report.get("stage_weight_value_byte_count")),
        "weight_tensor_values_loaded": weight_values_loaded,
        "weight_tensor_values_public": report.get("weight_tensor_values_public") is True,
        "live_run_performed": report.get("live_run_performed") is True,
        "stage_smoke_only": stage_smoke_only,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "activation_public": False,
        "kv_cache_public": False,
        "blockers": sorted(set(blockers)),
    }


def summarize_coordinator(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    success = _dict(report.get("success"))
    same = _dict(report.get("same_request"))
    generated_count = _int(
        args.generated_token_count
        or success.get("generated_token_count")
        or report.get("generated_token_count")
    )
    generated_hash = str(
        args.generated_token_hash
        or success.get("generated_token_hash")
        or report.get("generated_token_hash")
        or ""
    )
    request_hash = str(
        args.coordinator_request_id_hash
        or same.get("coordinator_request_id_hash")
        or report.get("coordinator_request_id_hash")
        or ""
    )
    model_id = str(report.get("model_id") or _dict(report.get("model")).get("model_id") or success.get("model_id") or same.get("model_id") or MODEL_ID)
    return {
        "schema": "glm52_kaggle_same_request_coordinator_summary_v1",
        "present": bool(report) or bool(args.coordinator_request_id_hash or args.generated_token_hash),
        "source_schema": str(report.get("schema") or ""),
        "model_id": model_id,
        "coordinator_request_id_hash": request_hash,
        "coordinator_request_verified": _hash_ok(request_hash),
        "generated_token_count": generated_count,
        "generated_token_hash": generated_hash if _hash_ok(generated_hash) else "",
        "live_run_performed": report.get("live_run_performed") is True or _dict(report.get("runtime")).get("live_run_performed") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True if report else True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
    }


def summarize_cleanup(report: dict[str, Any]) -> dict[str, Any]:
    temporary_deleted = report.get("temporary_kaggle_kernels_deleted") is True or report.get("temporary_resources_deleted") is True
    packages_removed = report.get("temporary_private_packages_removed") is True or report.get("private_packages_removed") is True
    live_left = report.get("live_resources_left_running")
    return {
        "schema": "glm52_kaggle_same_request_cleanup_summary_v1",
        "present": bool(report),
        "temporary_kaggle_kernels_deleted": temporary_deleted,
        "temporary_private_packages_removed": packages_removed,
        "live_resources_left_running": live_left if isinstance(live_left, bool) else None,
        "public_artifact_safe": report.get("public_artifact_safe") is True if report else True,
        "blockers": [str(item) for item in _list(report.get("blockers"))],
    }


def build_report(
    args: argparse.Namespace,
    *,
    stage_reports: list[dict[str, Any]] | None = None,
    coordinator_report: dict[str, Any] | None = None,
    cleanup_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_summaries = [
        normalize_stage_report(report, ordinal=index)
        for index, report in enumerate(stage_reports or [])
    ]
    coordinator = summarize_coordinator(coordinator_report or {}, args)
    cleanup = summarize_cleanup(cleanup_report or {})

    blockers: set[str] = set()
    if args.mode == "preflight":
        blockers.add("glm52_same_request_live_run_not_started")
    if not stage_summaries:
        blockers.add("glm52_stage_reports_missing")

    request_hashes = {
        str(stage.get("coordinator_request_id_hash"))
        for stage in stage_summaries
        if _hash_ok(stage.get("coordinator_request_id_hash"))
    }
    coordinator_hash = str(coordinator.get("coordinator_request_id_hash") or "")
    if coordinator_hash and _hash_ok(coordinator_hash):
        request_hashes.add(coordinator_hash)
    if len(request_hashes) != 1:
        blockers.add("glm52_same_request_hash_not_unique")

    ready_providers = {
        str(stage.get("provider"))
        for stage in stage_summaries
        if stage.get("stage_execution_verified") is True
    }
    for provider in REQUIRED_PROVIDERS:
        if provider not in ready_providers:
            blockers.add(f"same_request_provider_missing:{provider}")

    for stage in stage_summaries:
        blockers.update(str(item) for item in _list(stage.get("blockers")) if item)
        if coordinator_hash and _hash_ok(coordinator_hash) and stage.get("coordinator_request_id_hash") != coordinator_hash:
            blockers.add("stage_coordinator_request_hash_mismatch")

    if coordinator.get("model_id") != MODEL_ID:
        blockers.add("same_request_model_id_not_glm52")
    if coordinator.get("generated_token_count", 0) < 1 or not _hash_ok(coordinator.get("generated_token_hash")):
        blockers.add("same_request_generated_token_missing")
    if coordinator.get("coordinator_request_verified") is not True:
        blockers.add("same_request_coordinator_request_missing")
    if coordinator.get("live_run_performed") is not True:
        blockers.add("same_request_coordinator_live_run_not_performed")
    if coordinator.get("public_artifact_safe") is not True:
        blockers.add("same_request_coordinator_public_artifact_unsafe")
    blockers.update(str(item) for item in _list(coordinator.get("blockers")) if item)

    if cleanup.get("temporary_kaggle_kernels_deleted") is not True:
        blockers.add("cleanup_kernel_delete_missing")
    if cleanup.get("temporary_private_packages_removed") is not True:
        blockers.add("cleanup_private_package_removal_missing")
    if cleanup.get("live_resources_left_running") is not False:
        blockers.add("cleanup_live_resources_left_unknown")
    if cleanup.get("public_artifact_safe") is not True:
        blockers.add("cleanup_public_artifact_unsafe")
    blockers.update(str(item) for item in _list(cleanup.get("blockers")) if item)

    verified = bool(args.mode == "assemble" and not blockers)
    accepted = sorted(ready_providers) if verified else sorted(ready_providers)
    request_hash = sorted(request_hashes)[0] if len(request_hashes) == 1 else ""

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": verified,
        "mode": args.mode,
        "model": {
            "model_id": MODEL_ID,
            "compatible_weight_repo": args.compatible_weight_repo,
            "fallback_model_used": False,
        },
        "glm52_kaggle_same_request_verified": verified,
        "same_request_decode_verified": verified,
        "live_run_performed": verified,
        "success": {
            "same_request_decode_verified": verified,
            "model_id": MODEL_ID,
            "generated_token_count": coordinator.get("generated_token_count", 0) if verified else 0,
            "generated_token_hash": coordinator.get("generated_token_hash", "") if verified else "",
            "accepted_providers": REQUIRED_PROVIDERS if verified else accepted,
            "blockers": [] if verified else sorted(blockers),
        },
        "same_request": {
            "coordinator_request_verified": verified,
            "coordinator_request_id_hash": request_hash,
            "model_id": MODEL_ID,
            "stage_count": len(stage_summaries),
            "blockers": [] if verified else sorted(blockers),
        },
        "coordinator": coordinator,
        "stage_reports": stage_summaries,
        "cleanup": cleanup,
        "generated_token_count": coordinator.get("generated_token_count", 0) if verified else 0,
        "generated_token_hash": coordinator.get("generated_token_hash", "") if verified else "",
        "accepted_providers": REQUIRED_PROVIDERS if verified else accepted,
        "fallback_model_used": False,
        "queue_only_evidence": False,
        "metadata_only": False,
        "stage_smoke_only": False,
        "failure_stage": "" if verified else "glm52_same_request_decode_not_verified",
        "blockers": [] if verified else sorted(blockers),
        "diagnosis_codes": [
            "glm52_same_request_verified" if verified else "glm52_same_request_not_verified",
            "glm52_required_providers_ready" if set(REQUIRED_PROVIDERS).issubset(ready_providers) else "glm52_required_providers_not_ready",
        ],
        "safety": safety_flags(),
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
        report["success"]["same_request_decode_verified"] = False
        report["glm52_kaggle_same_request_verified"] = False
        report["same_request_decode_verified"] = False
        report["ok"] = False
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "assemble"], default="preflight")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-report", action="append", default=[])
    parser.add_argument("--coordinator-report", default="")
    parser.add_argument("--cleanup-report", default="")
    parser.add_argument("--compatible-weight-repo", default=COMPATIBLE_WEIGHT_REPO)
    parser.add_argument("--generated-token-count", type=int, default=0)
    parser.add_argument("--generated-token-hash", default="")
    parser.add_argument("--coordinator-request-id-hash", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    stage_reports = [load_json(path) for path in args.stage_report]
    report = build_report(
        args,
        stage_reports=stage_reports,
        coordinator_report=load_json(args.coordinator_report),
        cleanup_report=load_json(args.cleanup_report),
    )
    report_path = output_dir / "glm52_kaggle_same_request_probe.json"
    write_json(report_path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {report_path}")
        print(f"Same-request verified: {report.get('same_request_decode_verified')}")
        print(f"Failure stage: {report.get('failure_stage')}")
    return 0 if report.get("same_request_decode_verified") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
