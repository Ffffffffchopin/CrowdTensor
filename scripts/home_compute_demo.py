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


def build_home_compute_report(
    *,
    matrix: dict[str, Any],
    session_report: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = workload_status(matrix, WORKLOAD_TYPE)
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
        },
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
    session = report.get("inference_session") or {}
    print("CrowdTensor home-compute demo")
    print(f"  ok: {report['ok']}")
    print(
        "  host: "
        f"python={host.get('python')} os={host.get('os')} "
        f"machine={host.get('machine')} cpu_count={host.get('cpu_count')}"
    )
    print(f"  selected workload: {selected['name']} ({selected['status']})")
    if session:
        print(
            "  session: "
            f"{session.get('request_count')} requests, "
            f"accuracy={float(session.get('accuracy') or 0.0):.3f}, "
            f"elapsed_ms={float(session.get('elapsed_ms') or 0.0):.3f}, "
            f"requests_per_second={float(session.get('requests_per_second') or 0.0):.3f}"
        )
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
