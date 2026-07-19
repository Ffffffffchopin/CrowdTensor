#!/usr/bin/env python3
"""Run Qwen 1.5B Alpha tests and required training/control regressions."""

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


ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "qwen15b_alpha": [
        "tests/test_qwen15b_training.py",
        "tests/test_training_qwen15b_source_probe.py",
        "tests/test_training_qwen15b_dataset_prepare.py",
        "tests/test_qwen15b_training_rendezvous.py",
        "tests/test_qwen15b_four_gpu_runtime.py",
        "tests/test_qwen15b_four_gpu_worker.py",
        "tests/test_training_qwen15b_four_gpu_package.py",
        "tests/test_training_qwen15b_four_gpu_probe.py",
        "tests/test_training_qwen15b_four_gpu_alpha.py",
    ],
    "cuda_training": [
        "tests/test_cuda_training_contract.py",
        "tests/test_cuda_training_remote_delta.py",
        "tests/test_cuda_training_rendezvous.py",
        "tests/test_cuda_training_worker.py",
        "tests/test_training_cuda_single_kernel_package.py",
        "tests/test_training_cuda_single_kernel_probe.py",
        "tests/test_training_cuda_two_node_package.py",
        "tests/test_training_cuda_two_node_probe.py",
        "tests/test_training_cuda_two_node_rc.py",
    ],
    "cpu_training_foundation": [
        "tests/test_training_contract.py",
        "tests/test_hf_lora_training.py",
        "tests/test_pipeline_lora_training.py",
        "tests/test_named_tensor_optimizer.py",
        "tests/test_training_foundation_rc.py",
        "tests/test_training_public_safety.py",
    ],
    "state_store": ["tests/test_state_store.py"],
    "miner": ["tests/test_miner_cli.py"],
    "coordinator": ["tests/test_coordinator_api.py"],
    "user_cli": ["tests/test_training_cli.py"],
    "qwen15b_beta": [
        "tests/test_training_qwen15b_beta_service.py",
        "tests/test_training_qwen15b_beta_service_smoke.py",
        "tests/test_training_qwen15b_beta.py",
    ],
    "elastic_training": [
        "tests/test_elastic_training_runtime.py",
        "tests/test_training_qwen15b_elastic_check.py",
        "tests/test_training_qwen15b_elastic_pack.py",
    ],
    "elastic_training_beta_product": [
        "tests/test_elastic_training_beta.py",
        "tests/test_training_elastic_beta_check.py",
    ],
}

BETA_SUITE = "qwen15b_beta"
ELASTIC_BETA_SUITE = "elastic_training_beta_product"
FAULT_INJECTION_EVIDENCE = {
    "duplicate_submission_rejected_or_idempotent": (
        "tests/test_training_qwen15b_beta_service.py::"
        "test_job_store_submit_is_idempotent_and_rejects_conflict"
    ),
    "expired_lease_recovery_verified": (
        "tests/test_training_qwen15b_beta_service.py::"
        "test_expired_lease_recovers_without_global_step_regression"
    ),
    "corrupted_checkpoint_rejected": (
        "tests/test_training_qwen15b_four_gpu_probe.py::"
        "test_checkpoint_archive_verifies_every_private_file_before_cleanup"
    ),
    "non_finite_tensor_rejected": (
        "tests/test_training_qwen15b_beta.py::"
        "test_qwen_stage_fails_closed_on_non_finite_activation"
    ),
    "worker_timeout_classified": (
        "tests/test_training_qwen15b_beta.py::"
        "test_qwen_worker_fault_codes_are_stable_and_public"
    ),
    "coordinator_unavailable_retry_verified": (
        "tests/test_qwen15b_four_gpu_runtime.py::"
        "test_http_transport_retries_disconnect_and_reregisters_before_replay"
    ),
}


def _count(pattern: str, output: str) -> int:
    values = [int(value) for value in re.findall(pattern, output)]
    return values[-1] if values else 0


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
        timeout=1200.0,
    )
    output = process.stdout or ""
    passed = _count(r"(\d+) passed", output)
    failed = _count(r"(\d+) failed", output)
    errors = _count(r"(\d+) errors?", output)
    return {
        "name": name,
        "ok": process.returncode == 0,
        "returncode": process.returncode,
        "test_file_count": len(tests),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "duration_seconds": round(time.monotonic() - started, 3),
        "output_tail": output[-3000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suites = [run_suite(name, tests) for name, tests in SUITES.items()]
    baseline = [
        item
        for item in suites
        if item["name"] not in {BETA_SUITE, ELASTIC_BETA_SUITE}
    ]
    beta = next(item for item in suites if item["name"] == BETA_SUITE)
    elastic_beta = next(
        item for item in suites if item["name"] == ELASTIC_BETA_SUITE
    )
    baseline_passed = sum(int(item["passed"]) for item in baseline)
    baseline_ok = all(item["ok"] for item in baseline) and baseline_passed >= 313
    beta_ok = bool(beta["ok"])
    elastic_beta_ok = bool(elastic_beta["ok"])
    report = {
        "schema": "crowdtensor_qwen15b_training_test_summary_v1",
        "ok": all(item["ok"] for item in suites),
        "suite_count": len(suites),
        "test_file_count": sum(item["test_file_count"] for item in suites),
        "passed": sum(item["passed"] for item in suites),
        "failed": sum(item["failed"] + item["errors"] for item in suites),
        "suites": suites,
        "cuda_training_rc_regression_included": True,
        "cpu_training_foundation_regression_included": True,
        "state_store_regression_included": True,
        "miner_regression_included": True,
        "coordinator_regression_included": True,
        "existing_313_regressions_included": baseline_ok,
        "existing_regression_passed_count": baseline_passed,
        "beta_service_regressions_included": beta_ok,
        "elastic_training_beta_product_regressions_included": elastic_beta_ok,
        "elastic_training_beta_product_passed_count": int(elastic_beta["passed"]),
        "fault_injection": {
            key: bool(beta_ok and baseline_ok) for key in FAULT_INJECTION_EVIDENCE
        },
        "fault_injection_evidence": FAULT_INJECTION_EVIDENCE,
        "elastic_training_beta_security": {
            "checkpoint_signature_rejection_verified": elastic_beta_ok,
            "malformed_or_non_finite_checkpoint_rejected": elastic_beta_ok,
            "checkpoint_quota_quarantine_verified": elastic_beta_ok,
            "authenticated_owner_and_miner_routes_verified": elastic_beta_ok,
            "s3_minio_storage_unit_tested": elastic_beta_ok,
            "s3_minio_storage_externally_live_tested": False,
            "permissionless_byzantine_poisoning_resistance_verified": False,
        },
        "public_artifact_safe": True,
    }
    report["ok"] = bool(
        report["ok"] and baseline_ok and beta_ok and elastic_beta_ok
    )
    path = output / "training_qwen15b_test_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"training_qwen15b_test_suite ok={report['ok']} "
            f"passed={report['passed']} failed={report['failed']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
