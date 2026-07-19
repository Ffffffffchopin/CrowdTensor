#!/usr/bin/env python3
"""Pack public Volunteer Training Internet Beta evidence into an Engineering RC."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_training_protocol import with_public_safety


SCHEMA = "crowdtensor_volunteer_training_internet_beta_engineering_rc_v1"
PROBE_SCHEMA = "crowdtensor_volunteer_training_internet_beta_probe_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Volunteer Training Beta probe must be an object")
    return value


def pack(probe_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(probe_path).expanduser().resolve()
    probe = _read(source)
    if probe.get("schema") != PROBE_SCHEMA:
        raise ValueError("Volunteer Training Beta probe schema mismatch")
    output = Path(output_dir).expanduser().resolve()
    evidence = output / "evidence"
    if output.exists():
        shutil.rmtree(output)
    evidence.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    sources = {"probe": source, **{
        str(name): source.parent / str(relative)
        for name, relative in (probe.get("artifacts") or {}).items()
    }}
    for name, artifact_source in sources.items():
        if not artifact_source.is_file():
            raise FileNotFoundError(f"Volunteer Training Beta artifact missing: {name}")
        suffix = ".jsonl" if artifact_source.suffix == ".jsonl" else ".json"
        destination = evidence / f"{name}{suffix}"
        shutil.copyfile(artifact_source, destination)
        destination.chmod(0o644)
        artifacts[name] = destination.relative_to(output).as_posix()
        artifact_hashes[name] = sha256_file(destination)

    ready = bool(
        probe.get("ok") is True
        and probe.get("volunteer_training_internet_beta_engineering_verified") is True
        and (probe.get("cleanup") or {}).get("cleanup_verified") is True
    )
    report = with_public_safety(
        {
            "schema": SCHEMA,
            "volunteer_training_internet_beta_engineering_rc_ready": ready,
            "goal_achieved": ready,
            "evidence_scope": "local_independent_process_real_peft_engineering_rc",
            "campaign_id": probe.get("campaign_id"),
            "campaign_manifest_hash": probe.get("campaign_manifest_hash"),
            "campaign_source": probe.get("campaign_source"),
            "round_progress": probe.get("round_progress"),
            "real_training": probe.get("real_training"),
            "transport_security": probe.get("transport_security"),
            "fault_recovery": probe.get("fault_recovery"),
            "checkpoint_lineage": probe.get("checkpoint_lineage"),
            "centralized_baseline": probe.get("centralized_baseline"),
            "communication": probe.get("communication"),
            "independent_replay": probe.get("independent_replay"),
            "contributor_workflow": probe.get("contributor_workflow"),
            "cleanup": probe.get("cleanup"),
            "public_artifact_scan_ok": probe.get("public_artifact_scan_ok") is True,
            "artifacts": artifacts,
            "artifact_hashes": artifact_hashes,
            "limitations": probe.get("limitations"),
        }
    )
    report["content_hash"] = sha256_json(report)
    (output / "volunteer_training_internet_beta_engineering_rc.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = pack(args.probe, args.output_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["goal_achieved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
