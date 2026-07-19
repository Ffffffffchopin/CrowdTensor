#!/usr/bin/env python3
"""Build a canonical public-safe heterogeneous Training Beta artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.training_heterogeneous_beta_check import (
    SCHEMA,
    build_acceptance_gates,
)


LIVE_SCHEMA = "crowdtensor_heterogeneous_training_beta_live_probe_v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("heterogeneous_training_beta_source_invalid")
    return value


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def pack(
    live_report: str | Path,
    output_dir: str | Path,
    *,
    regression_summary: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(live_report).resolve()
    source = _load(source_path)
    if source.get("schema") != LIVE_SCHEMA:
        raise ValueError("heterogeneous_training_beta_live_schema_invalid")
    report = copy.deepcopy(source)
    report["schema"] = SCHEMA
    report["source_evidence"] = {
        "live_report_hash": _file_hash(source_path),
        "runtime_measurements_changed": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if regression_summary is not None:
        regression_path = Path(regression_summary).resolve()
        regression = _load(regression_path)
        report["regression_summary"] = {
            **regression,
            "source_report_hash": _file_hash(regression_path),
        }
    report["blockers"] = sorted(
        {str(item) for item in report.get("blockers") or [] if str(item)}
    )
    report.pop("content_hash", None)
    report["acceptance_gates"] = build_acceptance_gates(report)
    report["heterogeneous_training_beta_ready"] = bool(
        report.get("live_run_performed") is True
        and all(report["acceptance_gates"].values())
        and not report["blockers"]
    )
    report["content_hash"] = _stable_hash(report)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "training_heterogeneous_beta.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-report", required=True)
    parser.add_argument("--regression-summary", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        args.live_report,
        args.output_dir,
        regression_summary=args.regression_summary or None,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "training_heterogeneous_beta_pack "
            f"ready={report['heterogeneous_training_beta_ready']} "
            f"blockers={len(report['blockers'])}"
        )


if __name__ == "__main__":
    main()
