#!/usr/bin/env python3
"""Validate a public-safe Colab CUDA runtime report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "colab_cuda_runtime_probe_v1"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def validate(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("report is not public safe")
    for key in (
        "runtime_proxy_token_public",
        "runtime_proxy_url_public",
        "endpoint_public",
        "credentials_public",
        "private_runtime_state_public",
    ):
        if report.get(key) is not False:
            errors.append(f"{key} must be false")
    if report.get("ok") is True or report.get("colab_cuda_runtime_ready") is True:
        if report.get("ok") is not True or report.get("colab_cuda_runtime_ready") is not True:
            errors.append("ok and colab_cuda_runtime_ready must agree")
        if report.get("cuda_available") is not True:
            errors.append("CUDA runtime was not available")
        if _int(report.get("cuda_device_count")) < 1:
            errors.append("no CUDA device was reported")
        if report.get("cuda_matmul_ready") is not True:
            errors.append("CUDA matmul was not verified")
        if not str(report.get("endpoint_hash") or ""):
            errors.append("ready report missing endpoint hash")
        if not str(report.get("runtime_proxy_host_hash") or ""):
            errors.append("ready report missing runtime proxy host hash")
        devices = report.get("devices")
        if not isinstance(devices, list) or not devices:
            errors.append("ready report missing device summaries")
        else:
            for index, device in enumerate(devices):
                if not isinstance(device, dict):
                    errors.append(f"device {index} summary is not an object")
                    continue
                if device.get("name_public") is not False:
                    errors.append(f"device {index} name must not be public")
                if not str(device.get("name_hash") or ""):
                    errors.append(f"device {index} missing name hash")
                if _int(device.get("total_memory_mb")) <= 0:
                    errors.append(f"device {index} missing total memory")
    else:
        blockers = report.get("blockers")
        if not isinstance(blockers, list) or not blockers:
            errors.append("not-ready report missing blockers")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    errors = validate(report)
    payload = {
        "schema": "colab_cuda_runtime_check_v1",
        "ok": not errors,
        "errors": errors,
        "report_path": args.report,
        "cuda_device_count": _int(report.get("cuda_device_count")),
        "colab_cuda_runtime_ready": report.get("colab_cuda_runtime_ready") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
