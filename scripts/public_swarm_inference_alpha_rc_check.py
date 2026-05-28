#!/usr/bin/env python3
"""CI-safe check for Public Swarm Inference Alpha RC evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "public_swarm_inference_alpha_rc_check_v1"
SECRET_FRAGMENTS = (
    "lease_token",
    "idempotency_key",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "real_llm_sharded_result",
    "generated_text",
    "generated_token_ids",
    "Bearer ",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
)


def run_json(command: list[str], *, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed([line for line in completed.stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"command emitted no JSON: {' '.join(command)}\nstdout:\n{completed.stdout}")


def assert_no_sensitive_output(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"Public Swarm Inference Alpha RC check leaked sensitive fragment: {fragment}")


def validate_report(payload: dict[str, Any], *, mode: str) -> None:
    if payload.get("schema") != "public_swarm_inference_alpha_rc_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected Public Swarm Inference Alpha RC report: {json.dumps(payload, sort_keys=True)}")
    codes = payload.get("diagnosis_codes") or []
    if mode == "evidence-import":
        for code in [
            "public_swarm_inference_alpha_rc_ready",
            "public_swarm_alpha_rc_evidence_imported",
            "stage0_live_requeue_evidence_ready",
            "stage1_live_requeue_evidence_ready",
            "public_swarm_live_requeue_evidence_ready",
            "public_swarm_live_requeue_summary_ready",
            "public_swarm_alpha_private_artifacts_absent",
        ]:
            if code not in codes:
                raise SystemExit(f"missing RC readiness diagnosis {code}: {codes}")
        imported = payload.get("imported_reports") or {}
        for stage in ["stage0", "stage1"]:
            stage_summary = imported.get(stage) or {}
            if stage_summary.get("ready") is not True:
                raise SystemExit(f"{stage} retained live evidence is not ready: {stage_summary}")
            requeue = stage_summary.get("live_requeue_summary") or {}
            for key in ["claim_observed", "victim_kernel_deleted", "rescued_result"]:
                if requeue.get(key) is not True:
                    raise SystemExit(f"{stage} missing live requeue summary {key}: {requeue}")
            if not requeue.get("lease_expired"):
                raise SystemExit(f"{stage} missing live requeue summary lease_expired: {requeue}")
            if requeue.get("victim_result_accepted") is not False:
                raise SystemExit(f"{stage} victim result was not rejected: {requeue}")
        summary = imported.get("summary") or {}
        if summary.get("ready") is not True:
            raise SystemExit(f"live requeue summary is not ready: {summary}")
    else:
        for code in ["public_swarm_alpha_rc_local_smoke_ready", "public_swarm_alpha_rc_contract_ready"]:
            if code not in codes:
                raise SystemExit(f"missing local smoke readiness diagnosis {code}: {codes}")
    safety = payload.get("safety") or {}
    for key in ["cpu_only", "not_production", "not_p2p", "not_large_model_serving"]:
        if safety.get(key) is not True:
            raise SystemExit(f"missing safety flag {key}: {safety}")
    artifacts = payload.get("artifacts") or {}
    for name in ["public_swarm_inference_alpha_rc_json", "public_swarm_inference_alpha_rc_markdown"]:
        artifact = artifacts.get(name) or {}
        if artifact.get("present") is not True:
            raise SystemExit(f"missing RC artifact {name}: {artifact}")
    assert_no_sensitive_output(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Swarm Inference Alpha RC evidence.")
    parser.add_argument("--mode", choices=["evidence-import", "local-smoke"], default="evidence-import")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stage0-report", default="")
    parser.add_argument("--stage1-report", default="")
    parser.add_argument("--summary-report", default="")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_public_swarm_alpha_rc_") as temp:
        output_dir = Path(args.output_dir) if args.output_dir else Path(temp) / "alpha-rc"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "public_swarm_inference_alpha_rc_pack.py"),
            "--mode",
            args.mode,
            "--output-dir",
            str(output_dir),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ]
        if args.stage0_report:
            command.extend(["--stage0-report", args.stage0_report])
        if args.stage1_report:
            command.extend(["--stage1-report", args.stage1_report])
        if args.summary_report:
            command.extend(["--summary-report", args.summary_report])
        payload = run_json(command, timeout=args.timeout_seconds + 30.0)
        validate_report(payload, mode=args.mode)
        print(json.dumps({
            "ok": True,
            "schema": SCHEMA,
            "mode": args.mode,
            "rc_schema": payload.get("schema"),
            "diagnosis_codes": payload.get("diagnosis_codes"),
            "artifact_count": len(payload.get("artifacts") or {}),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
