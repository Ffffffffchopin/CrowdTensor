#!/usr/bin/env python3
"""Run and record the bounded CUDA Training RC regression suites."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "crowdtensor_cuda_training_test_summary_v1"


SUITES = {
    "cuda_training": [
        "tests/test_cuda_training_contract.py",
        "tests/test_cuda_training_rendezvous.py",
        "tests/test_cuda_training_worker.py",
        "tests/test_cuda_training_remote_delta.py",
        "tests/test_training_cuda_single_kernel_package.py",
        "tests/test_training_cuda_single_kernel_probe.py",
        "tests/test_training_cuda_two_node_package.py",
        "tests/test_training_cuda_two_node_probe.py",
        "tests/test_training_cuda_two_node_rc.py",
    ],
    "cpu_training": [
        "tests/test_hf_lora_training.py",
        "tests/test_pipeline_lora_training.py",
        "tests/test_training_contract.py",
        "tests/test_named_tensor_optimizer.py",
        "tests/test_training_cli.py",
        "tests/test_training_foundation_rc.py",
        "tests/test_training_public_safety.py",
    ],
    "control_plane": [
        "tests/test_state_store.py",
        "tests/test_miner_cli.py",
        "tests/test_coordinator_api.py",
    ],
}


def _summary_line(output: str) -> str:
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if " passed" in line or " failed" in line or " error" in line:
            return line[:240]
    return lines[-1][:240] if lines else "no pytest summary"


def run_suite(name: str, tests: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    return {
        "suite": name,
        "ok": process.returncode == 0,
        "returncode": process.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "test_file_count": len(tests),
        "summary": _summary_line(process.stdout or ""),
        "raw_test_output_public": False,
        "public_artifact_safe": True,
    }


def build(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suites = {name: run_suite(name, tests) for name, tests in SUITES.items()}
    report = {
        "schema": SCHEMA,
        "ok": all(item["ok"] for item in suites.values()),
        "suites": suites,
        "cuda_training_tests_passed": suites["cuda_training"]["ok"],
        "cpu_training_regressions_passed": suites["cpu_training"]["ok"],
        "state_store_miner_coordinator_regressions_passed": suites["control_plane"]["ok"],
        "raw_test_output_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    path = output / "training_cuda_test_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build(args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"training_cuda_test_suite ok={report['ok']}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
