#!/usr/bin/env python3
"""Run the bounded local quality gate for the Model Adapter ecosystem."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash


SCHEMA = "crowdtensor_model_ecosystem_beta_local_gate_v1"
COMPILE_FILES = (
    "crowdtensor/model_adapter.py",
    "crowdtensor/adapter_stage_training.py",
    "crowdtensor/community_live_training.py",
    "scripts/model_adapter_conformance_check.py",
    "scripts/model_adapter_plugin_smoke.py",
    "scripts/mistral_kaggle_live_ledger.py",
    "scripts/mistral_kaggle_live_package.py",
    "scripts/mistral_kaggle_live_probe.py",
    "scripts/mistral_kaggle_live_check.py",
    "plugins/mistral_adapter/src/crowdtensor_mistral_adapter/__init__.py",
)
TEST_FILES = (
    "tests/test_model_adapter.py",
    "tests/test_model_adapter_plugins.py",
    "tests/test_mistral_adapter.py",
    "tests/test_model_adapter_plugin_smoke.py",
    "tests/test_community_workflow.py",
    "tests/test_community_live_training.py",
    "tests/test_community_kaggle_live_package.py",
    "tests/test_mistral_kaggle_live_ledger.py",
    "tests/test_mistral_kaggle_live_package.py",
    "tests/test_mistral_kaggle_live_probe.py",
    "tests/test_mistral_kaggle_live_check.py",
    "tests/test_model_ecosystem_beta_rc.py",
    "tests/test_community_docs_check.py",
)


def _run(command: list[str], *, root: Path, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    output = completed.stdout or ""
    summary = next(
        (
            line.strip()
            for line in reversed(output.splitlines())
            if re.search(r"\d+ passed", line)
        ),
        "",
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "summary": summary[:200],
        "output_public": False,
    }


def run_gate(output_dir: str | Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    compile_result = _run(
        [sys.executable, "-m", "py_compile", *COMPILE_FILES],
        root=root,
        timeout=120,
    )
    pytest_result = _run(
        [sys.executable, "-m", "pytest", "-q", *TEST_FILES],
        root=root,
        timeout=600,
    )
    match = re.search(r"(\d+) passed", str(pytest_result.get("summary") or ""))
    passed = int(match.group(1)) if match else 0
    report = {
        "schema": SCHEMA,
        "ok": bool(compile_result["ok"] and pytest_result["ok"] and passed >= 40),
        "py_compile": compile_result,
        "pytest": pytest_result,
        "passed": passed,
        "failed": 0 if pytest_result["ok"] else 1,
        "test_file_count": len(TEST_FILES),
        "plugin_registry_tests_included": True,
        "mistral_architecture_tests_included": True,
        "dual_wheel_service_tests_included": True,
        "live_report_checker_tests_included": True,
        "community_regression_tests_included": True,
        "workspace_paths_public": False,
        "subprocess_output_public": False,
        "public_artifact_safe": True,
    }
    safety = scan_public_value(report)
    report["public_safety"] = safety
    report["public_artifact_safe"] = safety["ok"] is True
    report["ok"] = bool(report["ok"] and safety["ok"])
    report["content_hash"] = stable_hash(report)
    (output / "model_ecosystem_beta_local_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_gate(args.output_dir)
    print(json.dumps(report, sort_keys=True) if args.json else f"ok={report['ok']} passed={report['passed']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
