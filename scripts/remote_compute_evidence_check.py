#!/usr/bin/env python3
"""Acceptance check for the safe remote-compute evidence pack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CrowdTensorD remote-compute evidence pack.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8912)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--miner-id", default="remote-evidence-miner")
    parser.add_argument("--admin-token", default="remote-evidence-admin")
    parser.add_argument("--observer-token", default="remote-evidence-observer")
    args = parser.parse_args()
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_compute_evidence_pack.py"),
        "--mode",
        "local-loopback",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--miner-id",
        args.miner_id,
        "--admin-token",
        args.admin_token,
        "--observer-token",
        args.observer_token,
    ]
    if args.state_dir:
        command.extend(["--state-dir", args.state_dir])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=90,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "remote_compute_evidence_pack.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("remote_compute_evidence_pack.py emitted no JSON evidence")
    evidence = json.loads(lines[-1])
    if not evidence.get("ok"):
        raise SystemExit(f"remote-compute evidence failed: {json.dumps(evidence, sort_keys=True)}")
    if evidence.get("schema") != "remote_compute_evidence_v1":
        raise SystemExit(f"unexpected evidence schema: {evidence.get('schema')}")

    route = evidence.get("route_decision") or {}
    if (
        route.get("name") != "remote_python_model_bundle_infer"
        or route.get("target") != "remote_python_miner"
        or route.get("workload") != "model_bundle_infer"
        or route.get("confidence") != "ready"
        or not route.get("usable_now")
    ):
        raise SystemExit(f"unexpected remote evidence route: {route}")
    for capability in ["runtime:python-cli", "backend:cpu", "workload:model_bundle_infer", "completed_result"]:
        if capability not in (route.get("matched_capabilities") or []):
            raise SystemExit(f"remote evidence route missing capability {capability}: {route}")
    if route.get("missing_capabilities"):
        raise SystemExit(f"remote evidence route should not miss capabilities: {route}")

    miner = evidence.get("miner") or {}
    profile = miner.get("profile") or {}
    if miner.get("miner_id") != args.miner_id:
        raise SystemExit(f"unexpected miner id: {miner}")
    if profile.get("runtime") != "python-cli" or profile.get("backend") != "cpu":
        raise SystemExit(f"unexpected remote miner profile: {profile}")
    if "model_bundle_infer" not in (profile.get("supported_workloads") or []):
        raise SystemExit(f"remote miner did not advertise model_bundle_infer: {profile}")

    summary = evidence.get("inference_summary") or {}
    if summary.get("workload_type") != "model_bundle_infer":
        raise SystemExit(f"unexpected inference workload: {summary}")
    if int(summary.get("request_count", 0)) != args.request_count:
        raise SystemExit(f"request count mismatch: {summary}")
    expected_trace_count = min(args.request_count, 8)
    if int(summary.get("request_trace_count", 0)) != expected_trace_count:
        raise SystemExit(f"request trace count mismatch: {summary}")
    if float(summary.get("requests_per_second", 0.0)) <= 0.0:
        raise SystemExit(f"invalid throughput: {summary}")

    request_trace = evidence.get("request_trace") or []
    if len(request_trace) != expected_trace_count:
        raise SystemExit(f"request trace length mismatch: {request_trace}")
    if not request_trace or not request_trace[0].get("prompt") or not request_trace[0].get("top_k"):
        raise SystemExit(f"request trace is not readable: {request_trace}")

    invite = evidence.get("invite") or {}
    safety = evidence.get("safety") or {}
    if invite.get("registry_hashed") is not True:
        raise SystemExit(f"invite registry was not hashed: {invite}")
    if not safety.get("read_only") or not safety.get("redaction_ok") or safety.get("raw_payloads_exposed"):
        raise SystemExit(f"unsafe remote evidence report: {safety}")
    payload = json.dumps(evidence, sort_keys=True)
    for secret_fragment in [
        "remote-evidence-token",
        "CROWDTENSOR_MINER_TOKEN=",
        "lease_token",
        "idempotency_key",
        "inference_results",
        "Bearer ",
    ]:
        if secret_fragment in payload:
            raise SystemExit(f"remote-compute evidence leaked secret-like material: {secret_fragment}")

    print(json.dumps({
        "ok": True,
        "schema": evidence["schema"],
        "mode": evidence.get("mode"),
        "miner_id": miner.get("miner_id"),
        "route": route["name"],
        "route_confidence": route.get("confidence"),
        "request_count": summary.get("request_count"),
        "request_trace_count": summary.get("request_trace_count"),
        "requests_per_second": summary.get("requests_per_second"),
        "registry_hashed": invite.get("registry_hashed"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
