#!/usr/bin/env python3
"""Acceptance check for CPU-only pipeline-sharded inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sharded_inference_check_v1"
REQUIRED_CODES = {
    "sharded_inference_ready",
    "stage_0_accepted",
    "stage_1_accepted",
    "baseline_match",
    "activation_transport_ready",
}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        return json.loads(line)
    raise ValueError("no JSON payload found on stdout")


def validate_report(payload: dict[str, Any], *, require_requeue: bool) -> list[str]:
    errors = []
    if payload.get("schema") != "sharded_inference_evidence_v1":
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("payload_not_ok")
    codes = set(payload.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        errors.append("missing_codes:" + ",".join(missing))
    if require_requeue and "stage_requeue_ready" not in codes:
        errors.append("missing_stage_requeue_ready")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    if safety.get("read_only") is not True or safety.get("redaction_ok") is not True:
        errors.append("safety_failed")
    stage = payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {}
    if not (stage.get("stage_1") or {}).get("baseline_match"):
        errors.append("baseline_match_false")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CPU-only pipeline-sharded inference.")
    parser.add_argument("--base-port", type=int, default=9820)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--failure-mode", choices=["none", "kill-stage-after-claim"], default="none")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_sharded_check_") as temp:
        output_dir = Path(temp)
        evidence_json = output_dir / "sharded_inference_evidence.json"
        evidence_md = output_dir / "sharded_inference_evidence.md"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "sharded_inference_evidence_pack.py"),
            "--port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--failure-mode",
            args.failure_mode,
            "--json-out",
            str(evidence_json),
            "--markdown-out",
            str(evidence_md),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
        payload: dict[str, Any] = {}
        errors: list[str] = []
        try:
            payload = json_from_stdout(completed.stdout)
        except ValueError as exc:
            errors.append(str(exc))
        if completed.returncode != 0:
            errors.append(f"pack_returncode:{completed.returncode}")
        errors.extend(validate_report(payload, require_requeue=args.failure_mode == "kill-stage-after-claim"))
        report = {
            "schema": SCHEMA,
            "ok": not errors,
            "failure_mode": args.failure_mode,
            "evidence_schema": payload.get("schema"),
            "diagnosis_codes": sorted(set(payload.get("diagnosis_codes") or [])),
            "artifact_paths": {
                "json": str(evidence_json),
                "markdown": str(evidence_md),
            },
            "errors": errors,
        }
        if args.json_out:
            output = Path(args.json_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
