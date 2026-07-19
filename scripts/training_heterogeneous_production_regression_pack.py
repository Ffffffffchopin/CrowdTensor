#!/usr/bin/env python3
"""Pack a pytest JUnit result into a public Training Production summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from crowdtensor.heterogeneous_training_manifest import stable_hash


SCHEMA = "crowdtensor_heterogeneous_training_production_regression_summary_v1"
REPORT_NAME = "training_heterogeneous_production_regression_summary.json"


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def pack(
    *,
    junit_path: str | Path,
    output_dir: str | Path,
    warning_count: int = 0,
) -> dict[str, Any]:
    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError("training_production_regression_junit_invalid") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("training_production_regression_junit_invalid")
    tests = sum(int(item.get("tests") or 0) for item in suites)
    failures = sum(int(item.get("failures") or 0) for item in suites)
    errors = sum(int(item.get("errors") or 0) for item in suites)
    skipped = sum(int(item.get("skipped") or 0) for item in suites)
    duration = sum(float(item.get("time") or 0.0) for item in suites)
    failed = failures + errors
    passed = tests - failed - skipped
    test_files = set()
    for item in root.iter("testcase"):
        identity = str(item.get("classname") or item.get("name") or "")
        parts = identity.split(".")
        if identity.startswith("tests.") and len(parts) >= 2:
            test_files.add(parts[1])
    test_files = sorted(test_files)
    report = {
        "schema": SCHEMA,
        "ok": bool(tests > 0 and failed == 0),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "warnings": max(0, int(warning_count)),
        "duration_seconds": duration,
        "test_count": tests,
        "test_file_count": len(test_files),
        "test_manifest_hash": stable_hash(test_files),
        "junit_file_hash": _file_hash(junit_path),
        "scheduler_tests_included": "test_heterogeneous_training_scheduler" in test_files,
        "checkpoint_and_journal_tests_included": bool(
            "test_heterogeneous_training_checkpoint" in test_files
            and "test_elastic_training_runtime" in test_files
        ),
        "cli_and_workflow_tests_included": bool(
            "test_training_cli" in test_files
            and "test_training_heterogeneous_production_rc" in test_files
        ),
        "cpu_cuda_tpu_tests_included": bool(
            "test_cuda_training_worker" in test_files
            and "test_training_heterogeneous_tpu_beta" in test_files
            and "test_heterogeneous_qwen_training" in test_files
        ),
        "transport_and_privacy_tests_included": bool(
            "test_heterogeneous_tensor_transport" in test_files
            and "test_training_public_safety" in test_files
        ),
        "raw_commands_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warning-count", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        junit_path=args.junit,
        output_dir=args.output_dir,
        warning_count=args.warning_count,
    )
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
