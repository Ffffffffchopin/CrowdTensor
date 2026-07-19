#!/usr/bin/env python3
"""Validate public-safe Colab Qwen stage loader reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCHEMA = "colab_tpu_qwen_stage_loader_probe_v1"


def validate(report: dict, *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("report is not public-safe")
    for key in ("runtime_proxy_token_public", "runtime_proxy_url_public", "endpoint_public"):
        if report.get(key) is not False:
            errors.append(f"{key} must be false")
    if not report.get("endpoint_hash"):
        errors.append("endpoint hash missing")
    if not report.get("runtime_proxy_host_hash"):
        errors.append("runtime proxy host hash missing")
    if require_ready:
        if report.get("ok") is not True:
            errors.append("loader probe did not succeed")
        if report.get("colab_qwen_stage_loader_ready") is not True:
            errors.append("Colab Qwen stage loader is not ready")
        if int(report.get("executed_layer_count", 0)) < 1:
            errors.append("no layer execution verified")
        if int(report.get("missing_stage_key_count", 0)) != 0:
            errors.append("stage keys are missing")
        if int(report.get("tpu_device_count", 0)) < 1:
            errors.append("no TPU device was observed")
    if report.get("gpu_tpu_cpu_72b_same_request_verified") is True:
        errors.append("stage loader report must not claim full 72B same-request success")
    if report.get("same_request_72b_full_model_verified") is True:
        errors.append("stage loader report must not claim full 72B model success")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
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
