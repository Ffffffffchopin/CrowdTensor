#!/usr/bin/env python3
"""Pack public Volunteer Training Protocol Alpha evidence into one RC manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_training_protocol import with_public_safety


SCHEMA = "crowdtensor_volunteer_training_alpha_rc_v1"
PROBE_SCHEMA = "crowdtensor_volunteer_training_alpha_probe_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("volunteer Alpha probe must be a JSON object")
    return value


def pack(probe_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(probe_path).resolve()
    probe = _read(source)
    if probe.get("schema") != PROBE_SCHEMA:
        raise ValueError("volunteer Alpha probe schema mismatch")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        probe_relative = source.relative_to(output).as_posix()
    except ValueError:
        probe_relative = Path(__import__("os").path.relpath(source, output)).as_posix()

    artifacts: dict[str, str] = {"probe": probe_relative}
    artifact_hashes: dict[str, str] = {"probe": sha256_file(source)}
    for name, relative in (probe.get("artifacts") or {}).items():
        artifact = source.parent / str(relative)
        if not artifact.is_file():
            raise FileNotFoundError(f"volunteer Alpha artifact missing: {name}")
        try:
            packed_relative = artifact.resolve().relative_to(output).as_posix()
        except ValueError:
            packed_relative = Path(
                __import__("os").path.relpath(artifact.resolve(), output)
            ).as_posix()
        artifacts[str(name)] = packed_relative
        artifact_hashes[str(name)] = sha256_file(artifact)

    ready = bool(
        probe.get("ok")
        and probe.get("volunteer_training_protocol_alpha_verified")
        and (probe.get("cleanup") or {}).get("cleanup_verified")
        and probe.get("public_artifact_scan_ok")
    )
    report = with_public_safety(
        {
            "schema": SCHEMA,
            "volunteer_training_protocol_alpha_ready": ready,
            "goal_achieved": ready,
            "evidence_scope": "local_http_real_peft_protocol_alpha",
            "campaign_id": probe.get("campaign_id"),
            "campaign_manifest_hash": probe.get("campaign_manifest_hash"),
            "protocol_version": probe.get("protocol_version"),
            "real_training": probe.get("real_training"),
            "round_progress": probe.get("round_progress"),
            "churn_proof": probe.get("churn_proof"),
            "update_validation": probe.get("update_validation"),
            "centralized_baseline": probe.get("centralized_baseline"),
            "communication": probe.get("communication"),
            "contributor_workflow": probe.get("contributor_workflow"),
            "http_service": probe.get("http_service"),
            "audit_ledger": probe.get("audit_ledger"),
            "cleanup": probe.get("cleanup"),
            "limitations": probe.get("limitations"),
            "artifacts": artifacts,
            "artifact_hashes": artifact_hashes,
        }
    )
    report["content_hash"] = sha256_json(report)
    path = output / "volunteer_training_alpha_rc.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    else:
        print(
            "volunteer_training_protocol_alpha_ready="
            + str(report["volunteer_training_protocol_alpha_ready"])
        )
    return 0 if report["goal_achieved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
