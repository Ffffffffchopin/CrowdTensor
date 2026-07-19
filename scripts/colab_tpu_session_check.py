#!/usr/bin/env python3
"""Validate a public-safe Colab TPU session allocation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCHEMA = "colab_tpu_session_probe_v1"


def validate(report: dict) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if report.get("ok") is not True:
        errors.append("session allocation did not succeed")
    if report.get("colab_tpu_session_allocated") is not True:
        errors.append("Colab TPU session was not allocated")
    if report.get("public_artifact_safe") is not True:
        errors.append("report is not public-safe")
    for key in ("oauth_token_public", "runtime_proxy_token_public", "runtime_proxy_url_public", "endpoint_public"):
        if report.get(key) is not False:
            errors.append(f"{key} must be false")
    if not report.get("endpoint_hash"):
        errors.append("endpoint hash missing")
    if not report.get("runtime_proxy_host_hash"):
        errors.append("runtime proxy host hash missing")
    if str(report.get("accelerator") or "") not in {"V5E1", "V6E1"}:
        errors.append("unexpected accelerator")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    errors = validate(report)
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
