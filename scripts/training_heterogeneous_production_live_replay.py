#!/usr/bin/env python3
"""Re-derive public Training Production live gates from retained raw evidence."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

from crowdtensor.heterogeneous_training_manifest import stable_hash
from scripts.training_cuda_kaggle_common import public_safety_errors
from scripts.training_heterogeneous_production_live_probe import (
    _effective_kernel_evidence,
    _flatten_workers,
    _replacement_evidence,
)
from scripts.training_heterogeneous_production_rc_check import (
    LIVE_SCHEMA,
    REQUIRED_PROVIDERS,
    replacement_evidence_ready,
)


REPLAY_SCHEMA = "crowdtensor_heterogeneous_training_production_live_replay_v1"
REPORT_NAME = "training_heterogeneous_production_live_probe.json"


def _read(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("training_production_live_replay_source_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("training_production_live_replay_source_invalid")
    return value


def _content_hash_valid(value: dict[str, Any]) -> bool:
    supplied = str(value.get("content_hash") or "")
    return bool(
        supplied
        and supplied
        == stable_hash(
            {key: item for key, item in value.items() if key != "content_hash"}
        )
    )


def replay(
    *,
    source_live_path: str | Path,
    kernel_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    source = _read(source_live_path)
    if (
        source.get("schema") != LIVE_SCHEMA
        or source.get("live_run_performed") is not True
        or not _content_hash_valid(source)
    ):
        raise ValueError("training_production_live_replay_source_invalid")
    kernel_reports = [_read(path) for path in kernel_paths]
    roles = [str(item.get("kernel_role") or "") for item in kernel_reports]
    if sorted(roles) != ["cpu", "gpu_a", "gpu_b", "tpu"]:
        raise ValueError("training_production_live_replay_kernel_coverage_invalid")

    workers = _flatten_workers(kernel_reports)
    replacements = {
        kind: _replacement_evidence(workers, kind)
        for kind in ("cpu", "cuda", "jax_tpu")
    }
    kernel_evidence, effective_kernel_verified = _effective_kernel_evidence(
        kernel_reports, replacements
    )
    replacement_verified = all(
        replacement_evidence_ready(item) for item in replacements.values()
    )

    report = copy.deepcopy(source)
    source_hash = str(source.get("content_hash") or "")
    report["worker_replacements"] = replacements
    report["kernel_evidence"] = kernel_evidence
    report["effective_kernel_evidence_verified"] = effective_kernel_verified
    report["external_runtime_verified"] = bool(
        sorted(report.get("accepted_providers") or []) == REQUIRED_PROVIDERS
        and replacement_verified
        and effective_kernel_verified
    )
    blockers = set(str(item) for item in report.get("blockers") or [])
    if replacement_verified:
        blockers.discard("training_production_worker_replacement_gate_failed")
    else:
        blockers.add("training_production_worker_replacement_gate_failed")
    report["blockers"] = sorted(blockers)
    report["evidence_replay"] = {
        "schema": REPLAY_SCHEMA,
        "source_schema": LIVE_SCHEMA,
        "source_content_hash": source_hash,
        "source_content_hash_valid": True,
        "raw_kernel_report_count": len(kernel_reports),
        "raw_kernel_report_hashes": sorted(
            str(item.get("kernel_report_hash") or "") for item in kernel_evidence
        ),
        "derived_fields": [
            "effective_kernel_evidence_verified",
            "external_runtime_verified",
            "kernel_evidence",
            "worker_replacements",
        ],
        "source_live_run_reused": True,
        "live_run_reexecuted": False,
        "training_measurements_changed": False,
        "public_artifact_safe": True,
    }
    report.pop("content_hash", None)
    safety = public_safety_errors(report)
    report["public_safety_errors"] = safety
    report["public_artifact_safe"] = not safety
    if safety:
        report["blockers"] = sorted(
            set(report["blockers"]) | {"training_production_public_safety_failed"}
        )
    report["content_hash"] = stable_hash(report)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    kernel_output = output / "kernels"
    kernel_output.mkdir(parents=True, exist_ok=True)
    for source_path, role in zip(kernel_paths, roles):
        shutil.copyfile(source_path, kernel_output / f"{role}.json")
    (output / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-live-report", required=True)
    parser.add_argument("--kernel-report", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = replay(
        source_live_path=args.source_live_report,
        kernel_paths=args.kernel_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if not report.get("blockers") else 2


if __name__ == "__main__":
    raise SystemExit(main())
