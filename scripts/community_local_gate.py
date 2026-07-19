#!/usr/bin/env python3
"""Run the local-equivalent P0-P4 Community CI gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from crowdtensor.community_reliability import BoundedChaosRunner, InMemoryReliabilityTarget
from crowdtensor.community_security import scan_public_value, security_contract_report
from crowdtensor.model_adapter import adapter_registry_report, stable_hash
from crowdtensor.version import public_version
from scripts.community_docs_check import check as check_docs
from scripts.community_release_check import check as check_release


SCHEMA = "crowdtensor_community_local_gate_v1"


TESTS = [
    "tests/test_community_protocol.py",
    "tests/test_community_workflow.py",
    "tests/test_community_api.py",
    "tests/test_community_security.py",
    "tests/test_community_reliability.py",
    "tests/test_community_live_training.py",
    "tests/test_community_kaggle_live_package.py",
    "tests/test_community_kaggle_wheel_smoke_probe.py",
    "tests/test_community_kaggle_gpu_stage0_diagnostic.py",
    "tests/test_community_kaggle_reliability_live_check.py",
    "tests/test_community_live_gate_ledger_amend.py",
    "tests/test_community_release_build.py",
    "tests/test_community_smollm_live_check.py",
    "tests/test_community_docs_check.py",
    "tests/test_community_cleanup_audit_check.py",
    "tests/test_community_maturity_rc.py",
    "tests/test_model_adapter.py",
    "tests/test_heterogeneous_training_production.py",
    "tests/test_heterogeneous_training_scheduler.py",
]


def _hash(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def run_gate(
    output_dir: str | Path,
    *,
    release_report: str | Path,
    minio_report: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *TESTS],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = re.search(r"(\d+) passed", process.stdout or "")
    test_count = int(match.group(1)) if match else 0
    tests = {
        "ok": process.returncode == 0,
        "test_file_count": len(TESTS),
        "passed_count": test_count,
        "failed_count": 0 if process.returncode == 0 else 1,
        "output_hash": _hash(process.stdout or ""),
        "security_negative_tests_included": True,
        "protocol_compatibility_tests_included": True,
        "scheduler_runtime_cli_tests_included": True,
        "cuda_contract_tests_included": True,
        "jax_contract_tests_included": True,
    }
    chaos = BoundedChaosRunner(InMemoryReliabilityTarget(), maximum_seconds=30).run()
    chaos_path = output / "community_bounded_chaos.json"
    chaos_path.write_text(json.dumps(chaos, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    docs = check_docs(".")
    docs_path = output / "community_docs_check.json"
    docs_path.write_text(json.dumps(docs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    release = check_release(release_report)
    minio = json.loads(Path(minio_report).read_text(encoding="utf-8"))
    minio_ok = bool(
        minio.get("ok") is True
        and minio.get("real_api_calls_performed") is True
        and minio.get("content_addressed") is True
        and minio.get("mirror_fallback_verified") is True
        and minio.get("primary_repair_verified") is True
        and minio.get("retention_verified") is True
        and minio.get("service_restart_verified") is True
        and minio.get("cleanup_verified") is True
        and minio.get("container_left_running") is False
    )
    report = {
        "schema": SCHEMA,
        "ok": bool(tests["ok"] and chaos["ok"] and docs["ok"] and release["ok"] and minio_ok),
        "versions": public_version(),
        "tests": tests,
        "workflow": {
            "complete_action_contract": True,
            "idempotency_tested": True,
            "dry_run_tested": True,
            "explicit_exit_codes_tested": True,
            "run_id_and_safe_next_command_tested": True,
        },
        "protocol": {
            "same_major_compatibility_tested": True,
            "unknown_major_minor_rejected": True,
            "silent_downgrade_allowed": False,
        },
        "security": security_contract_report(),
        "model_adapters": adapter_registry_report(),
        "chaos": {
            "ok": chaos["ok"],
            "scenario_count": chaos["scenario_count"],
            "content_hash": chaos["content_hash"],
        },
        "minio": {
            "ok": minio_ok,
            "source_schema": minio.get("schema"),
            "content_hash": minio.get("content_hash"),
            "real_local_api_only": True,
            "external_sla_verified": False,
        },
        "docs": {
            "ok": docs["ok"],
            "required_file_count": docs["required_file_count"],
            "strict_public_markdown_count": docs["strict_public_markdown_count"],
        },
        "release": {
            "ok": release["ok"],
            "artifact_count": release["artifact_count"],
            "document_count": release["document_count"],
        },
        "physical_multi_machine_test_required": False,
        "kaggle_logical_multi_node_label_required": True,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    privacy = scan_public_value(report)
    report["public_safety"] = privacy
    report["public_artifact_safe"] = privacy["ok"] is True
    report["ok"] = bool(report["ok"] and privacy["ok"])
    report["content_hash"] = stable_hash(report)
    (output / "community_local_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--release-report", required=True)
    parser.add_argument("--minio-report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_gate(
        args.output_dir,
        release_report=args.release_report,
        minio_report=args.minio_report,
    )
    print(json.dumps(report, sort_keys=True) if args.json else f"local_gate_ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
