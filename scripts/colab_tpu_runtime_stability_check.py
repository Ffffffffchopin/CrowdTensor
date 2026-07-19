#!/usr/bin/env python3
"""Validate a public-safe Colab TPU runtime stability report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCHEMA = "colab_tpu_runtime_stability_probe_v1"


def validate(report: dict) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if not report.get("public_artifact_safe"):
        errors.append("report is not marked public-safe")
    if report.get("runtime_proxy_token_public") is not False:
        errors.append("runtime proxy token must not be public")
    if report.get("runtime_proxy_url_public") is not False:
        errors.append("runtime proxy URL must not be public")
    if report.get("endpoint_public") is not False:
        errors.append("endpoint must not be public")
    if report.get("ok") is not True:
        errors.append("report did not complete successfully")
    if report.get("colab_tpu_runtime_stably_acquired") is not True:
        errors.append("stable Colab TPU acquisition was not verified")
    if report.get("runtime_proxy_connected") is not True:
        errors.append("runtime proxy connection was not verified")
    if int(report.get("rounds_requested", 0)) < 3:
        errors.append("at least 3 rounds are required")
    if report.get("rounds_completed") != report.get("rounds_requested"):
        errors.append("not all requested rounds completed")
    if report.get("rounds_ready") != report.get("rounds_requested"):
        errors.append("not all requested rounds were TPU-ready")
    if int(report.get("observed_device_count_max", 0)) < 1:
        errors.append("no TPU device was observed")
    if not report.get("endpoint_hash"):
        errors.append("endpoint hash missing")
    if not report.get("runtime_proxy_host_hash"):
        errors.append("runtime proxy host hash missing")
    observations = report.get("observations", [])
    if not isinstance(observations, list) or not observations:
        errors.append("observations missing")
    else:
        for index, obs in enumerate(observations):
            if obs.get("matmul_ready") is not True:
                errors.append(f"round {index} matmul not ready")
            if int(obs.get("device_count", 0)) < 1:
                errors.append(f"round {index} has no TPU device")
            if obs.get("matmul_dtype") != "bfloat16":
                errors.append(f"round {index} did not run bfloat16 workload")
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
