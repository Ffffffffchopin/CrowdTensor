#!/usr/bin/env python3
"""Validate GLM 5.2 stage activation handoff reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_stage_activation_handoff_probe as probe  # noqa: E402


SCHEMA = "glm52_stage_activation_handoff_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("glm52_stage_activation_handoff_probe_ready") is not True:
        errors.append("probe_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    if report.get("model_id") != probe.MODEL_ID:
        errors.append("model_id_mismatch")
    stages = [item for item in _list(report.get("stages")) if isinstance(item, dict)]
    handoffs = [item for item in _list(report.get("activation_handoffs")) if isinstance(item, dict)]
    if len(stages) < 3:
        errors.append("stage_count_below_three")
    if len(handoffs) < 2:
        errors.append("handoff_count_below_two")
    provider_coverage = set(str(item) for item in _list(report.get("stage_runtime_provider_coverage")))
    if not set(probe.REQUIRED_PROVIDERS).issubset(provider_coverage):
        errors.append("provider_coverage_incomplete")
    if report.get("coordinator_request_id_hash_present") is not True:
        errors.append("coordinator_request_id_hash_missing")
    request_hash = str(report.get("coordinator_request_id_hash") or "")
    if not probe._hash_ok(request_hash):
        errors.append("coordinator_request_id_hash_invalid")
    for stage in stages:
        if stage.get("model_id") != probe.MODEL_ID:
            errors.append(f"stage_model_id_mismatch:{stage.get('stage_id')}")
        if stage.get("stage_handoff_endpoint_ready") is not True:
            errors.append(f"stage_endpoint_not_ready:{stage.get('stage_id')}")
        if stage.get("activation_payload_public") is not False:
            errors.append(f"stage_activation_payload_public:{stage.get('stage_id')}")
        if not probe._hash_ok(stage.get("stage_output_hash")):
            errors.append(f"stage_output_hash_missing:{stage.get('stage_id')}")
    for handoff in handoffs:
        if handoff.get("handoff_verified") is not True:
            errors.append(f"handoff_not_verified:{handoff.get('from_stage_id')}->{handoff.get('to_stage_id')}")
        if handoff.get("same_request_hash_verified") is not True:
            errors.append(f"handoff_request_hash_not_verified:{handoff.get('from_stage_id')}->{handoff.get('to_stage_id')}")
        if handoff.get("contiguous_layer_boundary") is not True:
            errors.append(f"handoff_layer_boundary_not_contiguous:{handoff.get('from_stage_id')}->{handoff.get('to_stage_id')}")
        if handoff.get("activation_payload_public") is not False:
            errors.append(f"handoff_activation_payload_public:{handoff.get('from_stage_id')}->{handoff.get('to_stage_id')}")
    if report.get("stage_activation_handoff_runtime_verified") is True:
        if report.get("same_request_decode_verified") is True:
            errors.append("handoff_overclaims_same_request_decode")
        if report.get("generated_token_verified") is True:
            errors.append("handoff_overclaims_generated_token")
    else:
        if not report.get("blockers"):
            errors.append("not_verified_missing_blockers")
    if require_verified and report.get("stage_activation_handoff_runtime_verified") is not True:
        errors.append("stage_activation_handoff_not_verified")
    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "activation_handoff_evidence_is_not_same_request_decode",
        "activation_handoff_evidence_is_not_generated_token",
        "activation_handoff_evidence_is_not_stage_decode",
        "requires_coordinator_same_request_decode",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")
    safety = _dict(report.get("safety"))
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_missing")
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
        if safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
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
        "stage_activation_handoff_runtime_verified": report.get("stage_activation_handoff_runtime_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_stage_activation_handoff_check: ok={result['ok']} "
            f"errors={len(errors)} verified={result['stage_activation_handoff_runtime_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
