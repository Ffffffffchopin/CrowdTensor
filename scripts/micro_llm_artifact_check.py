#!/usr/bin/env python3
"""Acceptance check for file-backed micro-LLM artifact sharded inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "micro_llm_artifact_check_v1"
REQUIRED_CODES = {
    "micro_llm_artifact_ready",
    "artifact_loaded",
    "micro_llm_sharded_ready",
    "stage_0_accepted",
    "stage_1_accepted",
    "baseline_match",
    "decoded_tokens_match",
    "activation_transport_ready",
    "distinct_stage_miners",
    "stage_assignment_valid",
}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        return json.loads(line)
    raise ValueError("no JSON payload found on stdout")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate artifact-backed micro-LLM sharded inference.")
    parser.add_argument("--base-port", type=int, default=9880)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--json-out", default="")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report_path = Path(args.json_out) if args.json_out else Path("dist/micro-llm-artifact-local/micro_llm_artifact_check.json")
    root = report_path.parent.resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else root / "artifact"
    build = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "micro_llm_artifact_pack.py"),
            "--output-dir",
            str(artifact_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    artifact_payload: dict[str, Any] = {}
    errors: list[str] = []
    try:
        artifact_payload = json_from_stdout(build.stdout)
    except ValueError as exc:
        errors.append(f"artifact_build:{exc}")
    if build.returncode != 0:
        errors.append(f"artifact_build_returncode:{build.returncode}")

    evidence_json = root / "micro_llm_artifact_evidence.json"
    evidence = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "micro_llm_sharded_inference_evidence_pack.py"),
            "--port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--decode-steps",
            str(args.decode_steps),
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--micro-llm-artifact",
            str(artifact_dir),
            "--json-out",
            str(evidence_json),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
    )
    evidence_payload: dict[str, Any] = {}
    try:
        evidence_payload = json_from_stdout(evidence.stdout)
    except ValueError as exc:
        errors.append(f"evidence:{exc}")
    if evidence.returncode != 0:
        errors.append(f"evidence_returncode:{evidence.returncode}")

    codes = set(evidence_payload.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        errors.append("missing_codes:" + ",".join(missing))
    artifact_hash = artifact_payload.get("artifact_hash")
    evidence_hash = (evidence_payload.get("artifact") or {}).get("artifact_hash")
    if not artifact_hash or artifact_hash != evidence_hash:
        errors.append("artifact_hash_mismatch")
    report = {
        "schema": SCHEMA,
        "ok": not errors,
        "artifact": {
            "artifact_id": artifact_payload.get("artifact_id"),
            "artifact_hash": artifact_hash,
            "manifest_path": artifact_payload.get("manifest_path"),
        },
        "evidence_schema": evidence_payload.get("schema"),
        "diagnosis_codes": sorted(codes),
        "artifact_paths": {
            "artifact_dir": str(artifact_dir),
            "evidence_json": str(evidence_json),
        },
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
