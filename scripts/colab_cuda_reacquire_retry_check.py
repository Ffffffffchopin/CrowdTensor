#!/usr/bin/env python3
"""Validate a public-safe bounded Colab CUDA reacquire retry report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA = "colab_cuda_reacquire_retry_probe_v1"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate(report: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("report is not public-safe")
    for key in (
        "oauth_token_public",
        "runtime_proxy_token_public",
        "runtime_proxy_url_public",
        "endpoint_public",
        "credentials_public",
        "private_runtime_state_public",
    ):
        if report.get(key) is not False:
            errors.append(f"{key} must be false")

    attempts_requested = _int(report.get("attempts_requested"))
    attempts_completed = _int(report.get("attempts_completed"))
    if attempts_requested < 1:
        errors.append("attempts requested must be positive")
    if attempts_completed < 1:
        errors.append("no attempts completed")
    if attempts_completed > attempts_requested:
        errors.append("attempts completed exceeds requested")

    attempts = _list(report.get("attempts"))
    if not attempts:
        errors.append("attempt summaries missing")
    elif len(attempts) != attempts_completed:
        errors.append("attempt summary count mismatch")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            errors.append(f"attempt {index} is not an object")
            continue
        if _int(attempt.get("attempt_index")) != index + 1:
            errors.append(f"attempt {index} index mismatch")
        if not str(attempt.get("accelerator_requested") or ""):
            errors.append(f"attempt {index} missing accelerator")
        if not str(attempt.get("authuser") or "").isdigit():
            errors.append(f"attempt {index} invalid authuser")
        if attempt.get("public_artifact_safe") is not True:
            errors.append(f"attempt {index} is not public-safe")
        if attempt.get("runtime_proxy_token_public") is not False:
            errors.append(f"attempt {index} exposes proxy token")
        if attempt.get("runtime_proxy_url_public") is not False:
            errors.append(f"attempt {index} exposes proxy URL")
        if attempt.get("endpoint_public") is not False:
            errors.append(f"attempt {index} exposes endpoint")
        if attempt.get("ok") is True and not attempt.get("report_path"):
            errors.append(f"attempt {index} success missing report path")

    ready = report.get("colab_cuda_reacquire_ready") is True
    if require_ready and not ready:
        errors.append("Colab CUDA reacquire was not verified")
    if ready:
        if report.get("ok") is not True:
            errors.append("ready report is not ok")
        if _int(report.get("successful_attempt_index")) < 1:
            errors.append("successful attempt index missing")
        if not report.get("successful_report_path"):
            errors.append("successful report path missing")
        if not str(report.get("accelerator") or ""):
            errors.append("successful accelerator missing")
        if not str(report.get("authuser") or "").isdigit():
            errors.append("successful authuser missing")
        if not report.get("endpoint_hash"):
            errors.append("endpoint hash missing")
        if not report.get("runtime_proxy_host_hash"):
            errors.append("runtime proxy host hash missing")
    else:
        if report.get("ok") is True:
            errors.append("not-ready report must not be ok")
        if not _list(report.get("blockers")):
            errors.append("not-ready report missing blockers")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    errors = validate(report, require_ready=args.require_ready)
    result = {"ok": not errors, "errors": errors, "report": args.report}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            print(error, file=sys.stderr)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
