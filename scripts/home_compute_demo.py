#!/usr/bin/env python3
"""Run a matrix-guided local home-compute inference demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg  # noqa: E402
import inference_session_demo  # noqa: E402
import runtime_matrix  # noqa: E402


WORKLOAD_TYPE = "model_bundle_infer"


def workload_status(matrix: dict[str, Any], name: str) -> dict[str, Any]:
    for workload in matrix.get("workloads", []):
        if workload.get("name") == name:
            return workload
    return {
        "name": name,
        "status": "blocked",
        "reason": f"{name} is missing from runtime matrix",
        "next_command": "python3 scripts/runtime_matrix.py --json",
    }


def selected_route(matrix: dict[str, Any]) -> dict[str, Any]:
    routes = matrix.get("recommended_routes") or []
    for route in routes:
        if route.get("name") == "local_cpu_model_bundle_infer":
            return route
    for route in routes:
        if route.get("usable_now"):
            return route
    return {
        "name": "local_cpu_model_bundle_infer",
        "target": "cpu_baseline",
        "workload": WORKLOAD_TYPE,
        "status": "blocked",
        "usable_now": False,
        "confidence": "blocked",
        "reason": "local CPU model bundle inference route is missing from runtime matrix",
        "matched_capabilities": [],
        "missing_capabilities": ["route:local_cpu_model_bundle_infer"],
        "next_command": "python3 scripts/runtime_matrix.py --json",
    }


def route_decision(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": route.get("name"),
        "target": route.get("target"),
        "workload": route.get("workload"),
        "status": route.get("status"),
        "usable_now": bool(route.get("usable_now")),
        "confidence": route.get("confidence", "blocked"),
        "reason": route.get("reason"),
        "matched_capabilities": list(route.get("matched_capabilities") or []),
        "missing_capabilities": list(route.get("missing_capabilities") or []),
        "next_command": route.get("next_command"),
    }


def build_home_compute_report(
    *,
    matrix: dict[str, Any],
    session_report: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = workload_status(matrix, WORKLOAD_TYPE)
    route = selected_route(matrix)
    available = selected.get("status") in {"available", "configured"}
    session_ok = bool(session_report and session_report.get("ok"))
    read_only = bool(session_report and session_report.get("read_only"))
    redaction_ok = bool(session_report and session_report.get("redaction_ok"))
    host = matrix.get("host_profile", {})
    summary = matrix.get("summary", {})
    report = {
        "ok": bool(matrix.get("ok") and available and session_ok and read_only and redaction_ok),
        "demo": "home_compute_inference_v1",
        "selected_workload": {
            "name": WORKLOAD_TYPE,
            "status": selected.get("status"),
            "reason": selected.get("reason"),
            "cpu_only": bool(selected.get("cpu_only", True)),
        },
        "runtime_matrix": {
            "ok": bool(matrix.get("ok")),
            "host_profile": {
                "python": host.get("python"),
                "os": host.get("os"),
                "machine": host.get("machine"),
                "cpu_count": host.get("cpu_count"),
            },
            "summary": {
                "available": summary.get("available", 0),
                "optional_missing": summary.get("optional_missing", 0),
                "blocked": summary.get("blocked", 0),
                "available_workloads": summary.get("available_workloads", []),
                "blocked_workloads": summary.get("blocked_workloads", []),
            },
            "configured_runtimes": matrix.get("configured_runtimes", {}),
            "hardware_targets": [
                {
                    "name": target.get("name"),
                    "status": target.get("status"),
                    "usable_now": bool(target.get("usable_now")),
                    "supported_workloads": target.get("supported_workloads", []),
                }
                for target in matrix.get("hardware_targets", [])
            ],
        },
        "capability_route": route,
        "route_decision": route_decision(route),
        "inference_session": session_report,
        "safety": {
            "read_only": read_only,
            "redaction_ok": redaction_ok,
            "raw_payloads_exposed": bool(session_report) and not redaction_ok,
        },
        "recommended_next_commands": [
            "python3 scripts/runtime_matrix.py --json",
            "python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
        ],
    }
    if not available:
        report["error"] = selected.get("reason") or f"{WORKLOAD_TYPE} is not available"
    elif session_report is None:
        report["error"] = "inference session did not run"
    elif not session_ok:
        report["error"] = "inference session failed"
    return report


def print_human_report(report: dict[str, Any]) -> None:
    matrix = report["runtime_matrix"]
    host = matrix["host_profile"]
    selected = report["selected_workload"]
    route = report.get("capability_route") or {}
    decision = report.get("route_decision") or {}
    session = report.get("inference_session") or {}
    print("CrowdTensor home-compute demo")
    print(f"  ok: {report['ok']}")
    print(
        "  host: "
        f"python={host.get('python')} os={host.get('os')} "
        f"machine={host.get('machine')} cpu_count={host.get('cpu_count')}"
    )
    print(f"  selected workload: {selected['name']} ({selected['status']})")
    print(
        "  capability route: "
        f"{route.get('name')} target={route.get('target')} "
        f"status={route.get('status')} confidence={decision.get('confidence')}"
    )
    if decision.get("reason"):
        print(f"  route reason: {decision.get('reason')}")
    matched = decision.get("matched_capabilities") or []
    missing = decision.get("missing_capabilities") or []
    if matched:
        print(f"  matched capabilities: {', '.join(matched)}")
    if missing:
        print(f"  missing capabilities: {', '.join(missing)}")
    targets = report["runtime_matrix"].get("hardware_targets") or []
    if targets:
        usable = [target["name"] for target in targets if target.get("usable_now")]
        detected = [target["name"] for target in targets if target.get("status") == "detected"]
        print(f"  usable targets: {', '.join(usable) if usable else 'none'}")
        print(f"  detected future targets: {', '.join(detected) if detected else 'none'}")
    if session:
        print(
            "  session: "
            f"{session.get('request_count')} requests, "
            f"accuracy={float(session.get('accuracy') or 0.0):.3f}, "
            f"elapsed_ms={float(session.get('elapsed_ms') or 0.0):.3f}, "
            f"requests_per_second={float(session.get('requests_per_second') or 0.0):.3f}"
        )
        trace = session.get("request_trace") or []
        if trace:
            print("  trace:")
            for row in trace:
                print(
                    "    "
                    f"{row.get('request_id')} "
                    f"prompt={row.get('prompt')!r} "
                    f"predicted={row.get('predicted_token')!r} "
                    f"target={row.get('target_token')!r} "
                    f"correct={bool(row.get('correct'))}"
                )
            if session.get("request_trace_truncated"):
                print(f"    ... truncated at {session.get('request_trace_count')} rows")
    print(
        "  safety: "
        f"read_only={report['safety']['read_only']} "
        f"redaction_ok={report['safety']['redaction_ok']}"
    )
    print("  next:")
    for command in report["recommended_next_commands"]:
        print(f"    - {command}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matrix-guided local CrowdTensorD home-compute demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8909)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--admin-token", default="local-admin")
    parser.add_argument("--json", action="store_true", help="emit only the machine-readable demo report")
    parser.add_argument("--browser-path", default="")
    parser.add_argument("--llm-runtime-cmd", default="")
    parser.add_argument("--llm-runtime-url", default="")
    parser.add_argument("--llm-runtime-api-key", default="")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args(argv)
    activate_miner_token(args)
    activate_observer_token(args)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    matrix = runtime_matrix.build_matrix(
        root=ROOT,
        browser_path=args.browser_path,
        llm_runtime_cmd=args.llm_runtime_cmd,
        llm_runtime_url=args.llm_runtime_url,
        llm_runtime_api_key=args.llm_runtime_api_key,
    )
    selected = workload_status(matrix, WORKLOAD_TYPE)
    if selected.get("status") not in {"available", "configured"}:
        return build_home_compute_report(matrix=matrix, session_report=None)
    session_report = inference_session_demo.run_demo(args)
    return build_home_compute_report(matrix=matrix, session_report=session_report)


def main() -> None:
    args = parse_args()
    report = run_demo(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human_report(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
