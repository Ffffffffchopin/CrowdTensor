#!/usr/bin/env python3
"""Acceptance check for the matrix-guided home-compute demo."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local CrowdTensorD home-compute demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8909)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--admin-token", default="local-admin")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args()
    activate_miner_token(args)
    activate_observer_token(args)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "home_compute_demo.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--admin-token",
        args.admin_token,
        "--json",
    ]
    if args.state_dir:
        command.extend(["--state-dir", args.state_dir])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "home_compute_demo.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("home_compute_demo.py emitted no JSON report")
    report = json.loads(lines[-1])
    if not report.get("ok"):
        raise SystemExit(f"home-compute demo failed: {json.dumps(report, sort_keys=True)}")
    if report.get("demo") != "home_compute_inference_v1":
        raise SystemExit(f"unexpected demo id: {report.get('demo')}")
    diagnosis = report.get("diagnosis") or {}
    if diagnosis.get("primary_code") != "home_compute_ready":
        raise SystemExit(f"unexpected home-compute diagnosis: {diagnosis}")
    if report.get("diagnosis_codes") != ["home_compute_ready"]:
        raise SystemExit(f"unexpected home-compute diagnosis codes: {report.get('diagnosis_codes')}")
    selected = report.get("selected_workload") or {}
    if selected.get("name") != "model_bundle_infer" or selected.get("status") not in {"available", "configured"}:
        raise SystemExit(f"unexpected selected workload: {selected}")
    route = report.get("capability_route") or {}
    if (
        route.get("name") != "local_cpu_model_bundle_infer"
        or route.get("target") != "cpu_baseline"
        or route.get("workload") != "model_bundle_infer"
        or not route.get("usable_now")
    ):
        raise SystemExit(f"unexpected capability route: {route}")
    decision = report.get("route_decision") or {}
    if (
        decision.get("name") != route.get("name")
        or decision.get("target") != route.get("target")
        or decision.get("workload") != route.get("workload")
        or decision.get("confidence") != "ready"
        or not decision.get("reason")
    ):
        raise SystemExit(f"unexpected route decision: {decision}")
    if "target:cpu_baseline" not in (decision.get("matched_capabilities") or []):
        raise SystemExit(f"route decision missing CPU target match: {decision}")
    if decision.get("missing_capabilities"):
        raise SystemExit(f"home-compute route should not miss capabilities: {decision}")
    matrix = report.get("runtime_matrix") or {}
    summary = matrix.get("summary") or {}
    if not matrix.get("ok") or int(summary.get("blocked", 0)) != 0:
        raise SystemExit(f"runtime matrix is not ready: {matrix}")
    targets = {target.get("name"): target for target in matrix.get("hardware_targets", [])}
    if not targets.get("cpu_baseline", {}).get("usable_now"):
        raise SystemExit(f"CPU baseline target is not usable: {targets}")
    session = report.get("inference_session") or {}
    if session.get("workload_type") != "model_bundle_infer":
        raise SystemExit(f"unexpected session workload: {session}")
    if int(session.get("request_count", 0)) != args.request_count:
        raise SystemExit(f"request count mismatch: {session}")
    request_trace = session.get("request_trace") or []
    expected_trace_count = min(args.request_count, 8)
    if int(session.get("request_trace_count", 0)) != expected_trace_count or len(request_trace) != expected_trace_count:
        raise SystemExit(f"request trace count mismatch: {session}")
    if not request_trace or not request_trace[0].get("prompt") or not request_trace[0].get("top_k"):
        raise SystemExit(f"request trace is not readable: {session}")
    if float(session.get("requests_per_second", 0.0)) <= 0.0:
        raise SystemExit(f"invalid throughput: {session}")
    safety = report.get("safety") or {}
    if not safety.get("read_only") or not safety.get("redaction_ok") or safety.get("raw_payloads_exposed"):
        raise SystemExit(f"unsafe demo report: {safety}")
    payload = json.dumps(report, sort_keys=True)
    for secret_fragment in ["local-runtime-key", "CROWDTENSOR_LLM_RUNTIME_API_KEY=", "Bearer "]:
        if secret_fragment in payload:
            raise SystemExit("home-compute demo leaked secret-like material")
    print(json.dumps({
        "ok": True,
        "demo": report["demo"],
        "route": route["name"],
        "route_confidence": decision.get("confidence"),
        "workload": selected["name"],
        "request_count": session.get("request_count"),
        "request_trace_count": session.get("request_trace_count"),
        "requests_per_second": session.get("requests_per_second"),
        "cpu_count": matrix.get("host_profile", {}).get("cpu_count"),
        "diagnosis": diagnosis.get("primary_code"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
