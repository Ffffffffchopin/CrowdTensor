#!/usr/bin/env python3
"""Acceptance check for CPU-only micro-LLM pipeline-sharded inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "micro_llm_sharded_check_v1"
REQUIRED_CODES = {
    "micro_llm_sharded_ready",
    "stage_0_accepted",
    "stage_1_accepted",
    "baseline_match",
    "decoded_tokens_match",
    "activation_transport_ready",
}
STAGE_AWARE_CODES = {
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


def validate_report(
    payload: dict[str, Any],
    *,
    require_requeue: bool,
    require_distinct_stage_miners: bool,
) -> list[str]:
    errors = []
    if payload.get("schema") != "micro_llm_sharded_evidence_v1":
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("payload_not_ok")
    codes = set(payload.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        errors.append("missing_codes:" + ",".join(missing))
    if require_requeue and "stage_requeue_ready" not in codes:
        errors.append("missing_stage_requeue_ready")
    if require_distinct_stage_miners:
        missing_stage = sorted(STAGE_AWARE_CODES - codes)
        if missing_stage:
            errors.append("missing_stage_aware_codes:" + ",".join(missing_stage))
        assignment = payload.get("stage_assignment") if isinstance(payload.get("stage_assignment"), dict) else {}
        if assignment.get("distinct_stage_miners") is not True:
            errors.append("distinct_stage_miners_false")
        if assignment.get("stage_assignment_valid") is not True:
            errors.append("stage_assignment_valid_false")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    if safety.get("read_only") is not True or safety.get("redaction_ok") is not True:
        errors.append("safety_failed")
    stage = payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {}
    if not (stage.get("stage_1") or {}).get("baseline_match"):
        errors.append("baseline_match_false")
    if not (stage.get("stage_1") or {}).get("decoded_tokens_match"):
        errors.append("decoded_tokens_match_false")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in ["activation_results", "hidden_state", "sharded_inference_result", "lease_token", "idempotency_key"]:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CPU-only micro-LLM pipeline-sharded inference.")
    parser.add_argument("--base-port", type=int, default=9860)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--micro-llm-artifact", default="")
    parser.add_argument("--prompt-texts", default="")
    parser.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_micro_llm_sharded_check_") as temp:
        output_dir = Path(temp)
        evidence_json = output_dir / "micro_llm_sharded_evidence.json"
        evidence_md = output_dir / "micro_llm_sharded_evidence.md"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "micro_llm_sharded_inference_evidence_pack.py"),
            "--port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--decode-steps",
            str(args.decode_steps),
            "--micro-llm-artifact",
            args.micro_llm_artifact,
            "--failure-mode",
            args.failure_mode,
            "--stage-mode",
            args.stage_mode,
            "--json-out",
            str(evidence_json),
            "--markdown-out",
            str(evidence_md),
        ]
        if args.prompt_texts:
            command.extend(["--prompt-texts", args.prompt_texts])
        if args.require_distinct_stage_miners:
            command.append("--require-distinct-stage-miners")
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
        errors.extend(validate_report(
            payload,
            require_requeue=args.failure_mode != "none",
            require_distinct_stage_miners=args.require_distinct_stage_miners or args.stage_mode == "split",
        ))
        report = {
            "schema": SCHEMA,
            "ok": not errors,
            "failure_mode": args.failure_mode,
            "stage_mode": args.stage_mode,
            "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
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
