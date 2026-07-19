#!/usr/bin/env python3
"""Validate the GLM 5.2 Kaggle Coordinator decode bridge contract artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_kaggle_coordinator_decode_bridge_probe as probe


SCHEMA = "glm52_kaggle_coordinator_decode_bridge_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def validate_report(report: dict[str, Any], *, require_contract: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
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
    if require_contract and report.get("coordinator_bridge_contract_ready") is not True:
        errors.append("coordinator_bridge_contract_not_ready")
    if report.get("model_id") != probe.MODEL_ID:
        errors.append("model_id_not_glm52")
    if not _hash_ok(report.get("coordinator_request_id_hash")):
        errors.append("coordinator_request_hash_missing")
    if int(report.get("stage_count") or 0) < 2:
        errors.append("stage_count_too_small")
    if report.get("same_request_decode_verified") is True:
        errors.append("contract_artifact_overclaims_same_request_success")
    if report.get("live_run_performed") is True:
        errors.append("contract_artifact_overclaims_live_run")
    boundary = _dict(report.get("completion_boundary"))
    if boundary.get("contract_is_not_live_success") is not True:
        errors.append("completion_boundary_missing_contract_limit")
    if "glm52_live_kaggle_same_request_not_run" not in list(report.get("blockers") or []):
        errors.append("live_same_request_blocker_missing")
    return sorted(set(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-contract", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_contract=bool(args.require_contract))
    payload = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "coordinator_bridge_contract_ready": report.get("coordinator_bridge_contract_ready") is True,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ok={payload['ok']} errors={len(errors)}")
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
