#!/usr/bin/env python3
"""Validate a Kaggle Web TPU queue monitor probe report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_web_tpu_queue_monitor_probe_v1"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate(report: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_not_true")
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    for key in ("credentials_public", "cookies_public", "jupyter_proxy_token_public", "private_runtime_state_public"):
        if safety.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if report.get("kaggle_notebook_url_public") is not False:
        errors.append("kaggle_notebook_url_public_must_be_false")
    if report.get("queue_monitor_probe_ready") is not True:
        errors.append("queue_monitor_probe_not_ready")
    observations = _list(report.get("observations"))
    if int(report.get("observation_count") or 0) < 1 or not observations:
        errors.append("observations_missing")
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            errors.append(f"observation_{index}_not_object")
            continue
        if "body_text_hash" not in observation:
            errors.append(f"observation_{index}_missing_body_hash")
        if observation.get("queue_prompt_visible") and not observation.get("queue_prompt_hash"):
            errors.append(f"observation_{index}_missing_queue_prompt_hash")
    progress = report.get("queue_progress") if isinstance(report.get("queue_progress"), dict) else {}
    if progress.get("queue_position_observed") is True:
        if progress.get("first_queue_position") is None or progress.get("last_queue_position") is None:
            errors.append("queue_position_observed_without_first_last")
        unique = progress.get("unique_queue_positions")
        if not isinstance(unique, list) or not unique:
            errors.append("queue_position_observed_without_unique_positions")
    ready = report.get("web_tpu_runtime_ready") is True
    if require_ready and not ready:
        errors.append("web_tpu_runtime_not_ready")
    if ready and report.get("blocked_reason"):
        errors.append("ready_report_has_blocked_reason")
    if not ready and not _list(report.get("blocker_codes")):
        errors.append("not_ready_report_missing_blockers")
    if report.get("read_only") is True and report.get("start_clicked") is True:
        errors.append("read_only_report_clicked_start")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    errors = validate(report if isinstance(report, dict) else {}, require_ready=args.require_ready)
    result = {"ok": not errors, "errors": errors, "report": args.report}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("OK" if result["ok"] else "FAILED")
        for error in errors:
            print(error)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
