#!/usr/bin/env python3
"""Validate GLM 5.2 Kaggle CPU/GPU/TPU same-request live attempt reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_kaggle_same_request_live_probe as probe


SCHEMA = "glm52_kaggle_same_request_live_check_v1"
PROBE_SCHEMA = probe.SCHEMA
MODEL_ID = probe.same_request_probe.MODEL_ID
REQUIRED_PROVIDERS = set(probe.REQUIRED_PROVIDERS)


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


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != PROBE_SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    safety = _dict(report.get("safety"))
    for key in [
        "credentials_public",
        "cookies_public",
        "signed_url_public",
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "weight_tensor_values_public",
        "safetensors_header_payload_public",
    ]:
        if key in safety and safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    if report.get("coordinator_url_public") is not False:
        errors.append("coordinator_url_public_not_false")
    if report.get("coordinator_token_public") is not False:
        errors.append("coordinator_token_public_not_false")
    if report.get("model_id") != MODEL_ID:
        errors.append("model_id_not_glm52")
    stage_count = _int(report.get("stage_count"))
    if stage_count < 2:
        errors.append("stage_count_too_small")
    stage_order = _list(report.get("stage_order"))
    if len(stage_order) != stage_count:
        errors.append("stage_order_count_mismatch")

    verified = report.get("same_request_decode_verified") is True
    if require_verified and not verified:
        errors.append("same_request_live_not_verified")
    if not verified:
        return sorted(set(errors))

    if report.get("ok") is not True:
        errors.append("verified_but_report_not_ok")
    if _list(report.get("blockers")):
        errors.append("verified_but_blockers_present")
    if report.get("full_stage_count_verified") is not True:
        errors.append("full_stage_count_not_verified")
    if _int(report.get("generated_token_count")) < 1:
        errors.append("generated_token_missing")
    target_tokens = max(1, _int(report.get("target_generated_token_count"), 1))
    expected_task_count = max(stage_count, _int(report.get("expected_stage_task_count"), stage_count * target_tokens))
    if target_tokens > 1 and _int(report.get("generated_token_count")) < target_tokens:
        errors.append("generated_token_count_below_target")
    token_hashes = _list(report.get("generated_token_hashes"))
    if target_tokens > 1 and len(token_hashes) < target_tokens:
        errors.append("generated_token_hashes_below_target")
    providers = {str(item) for item in _list(report.get("accepted_providers"))}
    for provider in sorted(REQUIRED_PROVIDERS - providers):
        errors.append(f"required_provider_missing:{provider}")
    if _int(report.get("stage_runtime_reports_collected")) != stage_count:
        errors.append("stage_runtime_reports_collected_count_mismatch")
    if _int(report.get("stage_runtime_reports_verified")) != stage_count:
        errors.append("stage_runtime_reports_verified_count_mismatch")
    if _int(report.get("coordinator_stage_reports_collected")) != expected_task_count:
        errors.append("coordinator_stage_reports_count_mismatch")
    if _int(report.get("worker_stage_decode_reports_collected")) != stage_count:
        errors.append("worker_stage_decode_reports_count_mismatch")
    if _int(report.get("worker_stage_decode_task_count"), stage_count) < expected_task_count:
        errors.append("worker_stage_decode_task_count_below_expected")

    status = _dict(report.get("coordinator_status"))
    if status.get("ready") is not True:
        errors.append("coordinator_not_ready")
    if _int(status.get("generated_token_count")) < target_tokens:
        errors.append("coordinator_generated_token_missing")
    if _int(status.get("completed_task_count")) != expected_task_count:
        errors.append("coordinator_completed_task_count_mismatch")
    if _int(status.get("pending_count")) != 0:
        errors.append("coordinator_pending_not_empty")

    cleanup = _dict(report.get("cleanup_status"))
    if cleanup.get("temporary_kaggle_kernels_deleted") is not True:
        errors.append("cleanup_kernel_delete_missing")
    if cleanup.get("temporary_private_packages_removed") is not True:
        errors.append("cleanup_private_package_removal_missing")
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_left_unknown")

    same_check = _dict(report.get("same_request_check"))
    if same_check.get("ok") is not True:
        errors.append("same_request_check_not_ok")
    if _int(same_check.get("error_count")) != 0:
        errors.append("same_request_check_errors_present")
    return sorted(set(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_verified=bool(args.require_verified))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "stage_count": _int(report.get("stage_count")),
        "generated_token_count": _int(report.get("generated_token_count")),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_same_request_live_check: ok={result['ok']} "
            f"errors={len(errors)} verified={result['same_request_decode_verified']} "
            f"stage_count={result['stage_count']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
