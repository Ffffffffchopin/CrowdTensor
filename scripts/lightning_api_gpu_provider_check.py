#!/usr/bin/env python3
"""Check a Lightning AI API GPU provider probe report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "lightning_api_gpu_provider_probe_v1"


def check_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("unexpected_schema")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_not_true")
    if report.get("credentials_public") is not False:
        errors.append("credentials_public_not_false")
    token = report.get("token_file") if isinstance(report.get("token_file"), dict) else {}
    if token.get("secret_values_public") is not False:
        errors.append("token_secret_values_public_not_false")
    if report.get("api_auth_verified") is not True:
        errors.append("api_auth_not_verified")
    accelerators = report.get("default_accelerators") if isinstance(report.get("default_accelerators"), dict) else {}
    if int(accelerators.get("gpu_accelerator_count") or 0) <= 0:
        errors.append("no_gpu_accelerators_reported")
    if report.get("create_or_start_attempted") is not False:
        errors.append("create_or_start_was_attempted")
    if not isinstance(report.get("blockers"), list):
        errors.append("blockers_not_list")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    path = Path(args.report)
    report = json.loads(path.read_text(encoding="utf-8"))
    errors = check_report(report if isinstance(report, dict) else {})
    result = {"ok": not errors, "errors": errors, "report": str(path)}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("OK" if result["ok"] else "FAILED")
        for error in errors:
            print(error)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
