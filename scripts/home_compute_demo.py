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
DIAGNOSIS_ORDER = [
    "runtime_matrix_blocked",
    "workload_unavailable",
    "cpu_route_unavailable",
    "session_not_run",
    "session_failed",
    "request_count_mismatch",
    "trace_missing",
    "unsafe_session",
    "demo_skipped",
    "home_compute_ready",
]
DIAGNOSIS_LIBRARY = {
    "home_compute_ready": {
        "severity": "info",
        "summary": "Home-compute demo completed on the local CPU route.",
        "next_steps": [
            "Share the home-compute evidence report for local reproducibility.",
            "Run the runtime acceptance pack before making broader capability claims.",
        ],
    },
    "runtime_matrix_blocked": {
        "severity": "error",
        "summary": "The runtime matrix reports that required local dependencies are blocked.",
        "next_steps": [
            "Run python3 scripts/runtime_matrix.py --json to inspect missing core dependencies.",
            "Install the required Python/runtime dependencies before rerunning the demo.",
        ],
    },
    "workload_unavailable": {
        "severity": "error",
        "summary": "The model_bundle_infer workload is not available in the runtime matrix.",
        "next_steps": [
            "Confirm project files and CPU-only workload contracts are present.",
            "Run python3 scripts/runtime_acceptance_pack.py after fixing the blocked workload.",
        ],
    },
    "cpu_route_unavailable": {
        "severity": "error",
        "summary": "The local CPU model bundle inference route is unavailable.",
        "next_steps": [
            "Inspect route_decision.missing_capabilities for the blocked requirement.",
            "Rerun scripts/runtime_matrix.py after fixing the CPU baseline route.",
        ],
    },
    "session_not_run": {
        "severity": "warning",
        "summary": "The inference session did not run.",
        "next_steps": [
            "Rerun scripts/home_compute_demo.py without skipping the session path.",
            "Check Coordinator/Miner startup settings and port availability.",
        ],
    },
    "session_failed": {
        "severity": "error",
        "summary": "The inference session ran but returned a failed report.",
        "next_steps": [
            "Inspect the inference_session summary for validation or runtime failures.",
            "Rerun scripts/inference_session_demo.py directly with the same request count.",
        ],
    },
    "request_count_mismatch": {
        "severity": "error",
        "summary": "The inference session used a different request count than requested.",
        "next_steps": [
            "Rerun the demo with a clean state directory and the intended --request-count value.",
            "Check for stale session output from an earlier run.",
        ],
    },
    "trace_missing": {
        "severity": "error",
        "summary": "The inference session did not expose a readable request trace.",
        "next_steps": [
            "Rerun the demo and confirm model_bundle_infer produced capped request_trace rows.",
            "Inspect inference_session_demo.py output for validation errors.",
        ],
    },
    "unsafe_session": {
        "severity": "error",
        "summary": "The session failed read-only or redaction safety checks.",
        "next_steps": [
            "Do not share the report until raw payload exposure is fixed.",
            "Verify read_only, redaction_ok, and raw_payloads_exposed in the report.",
        ],
    },
    "demo_skipped": {
        "severity": "warning",
        "summary": "The home-compute evidence was generated without running the demo session.",
        "next_steps": [
            "Rerun without --skip-demo when you need measured inference evidence.",
            "Treat the report as route readiness evidence only.",
        ],
    },
}


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
        "diagnosis_codes": ["cpu_baseline_blocked"],
        "operator_action": "fix_blocker",
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
        "diagnosis_codes": list(route.get("diagnosis_codes") or []),
        "operator_action": route.get("operator_action"),
        "next_command": route.get("next_command"),
    }


def observed_summary(
    *,
    matrix: dict[str, Any],
    selected: dict[str, Any],
    route: dict[str, Any],
    session_report: dict[str, Any] | None,
    safety: dict[str, Any],
    expected_request_count: int | None,
) -> dict[str, Any]:
    summary = matrix.get("summary") or {}
    return {
        "matrix_ok": bool(matrix.get("ok")),
        "blocked_workloads": summary.get("blocked_workloads", []),
        "selected_workload": {
            "name": selected.get("name"),
            "status": selected.get("status"),
        },
        "route": {
            "name": route.get("name"),
            "status": route.get("status"),
            "usable_now": bool(route.get("usable_now")),
            "confidence": route.get("confidence"),
            "missing_capabilities": list(route.get("missing_capabilities") or []),
        },
        "session": {
            "present": bool(session_report),
            "ok": (session_report or {}).get("ok"),
            "request_count": (session_report or {}).get("request_count"),
            "request_trace_count": (session_report or {}).get("request_trace_count"),
            "scenario_id": (session_report or {}).get("scenario_id"),
        },
        "expected_request_count": expected_request_count,
        "safety": safety,
    }


def diagnosis_payload(
    code: str,
    *,
    details: dict[str, Any] | None = None,
    observed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    template = DIAGNOSIS_LIBRARY[code]
    return {
        "primary_code": code,
        "severity": template["severity"],
        "summary": template["summary"],
        "details": details or {},
        "next_steps": list(template["next_steps"]),
        "observed": observed or {},
    }


def append_diagnosis(
    diagnoses: list[dict[str, Any]],
    code: str,
    *,
    observed: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> None:
    if any(item.get("primary_code") == code for item in diagnoses):
        return
    diagnoses.append(diagnosis_payload(code, details=details, observed=observed))


def sorted_diagnoses(diagnoses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {code: index for index, code in enumerate(DIAGNOSIS_ORDER)}
    return sorted(diagnoses, key=lambda item: order.get(str(item.get("primary_code")), len(order)))


def diagnose_home_compute(
    *,
    matrix: dict[str, Any],
    selected: dict[str, Any],
    route: dict[str, Any],
    session_report: dict[str, Any] | None,
    safety: dict[str, Any],
    expected_request_count: int | None = None,
    demo_skipped: bool = False,
) -> dict[str, Any]:
    observed = observed_summary(
        matrix=matrix,
        selected=selected,
        route=route,
        session_report=session_report,
        safety=safety,
        expected_request_count=expected_request_count,
    )
    diagnoses: list[dict[str, Any]] = []
    if not matrix.get("ok"):
        append_diagnosis(
            diagnoses,
            "runtime_matrix_blocked",
            observed=observed,
            details={"summary": matrix.get("summary", {})},
        )
    if selected.get("status") not in {"available", "configured"}:
        append_diagnosis(
            diagnoses,
            "workload_unavailable",
            observed=observed,
            details={
                "workload": selected.get("name"),
                "status": selected.get("status"),
                "reason": selected.get("reason"),
            },
        )
    if (
        route.get("name") != "local_cpu_model_bundle_infer"
        or route.get("target") != "cpu_baseline"
        or route.get("workload") != WORKLOAD_TYPE
        or not route.get("usable_now")
    ):
        append_diagnosis(
            diagnoses,
            "cpu_route_unavailable",
            observed=observed,
            details={
                "route": route.get("name"),
                "target": route.get("target"),
                "workload": route.get("workload"),
                "missing_capabilities": list(route.get("missing_capabilities") or []),
            },
        )
    if demo_skipped:
        append_diagnosis(diagnoses, "demo_skipped", observed=observed)
    elif session_report is None:
        append_diagnosis(diagnoses, "session_not_run", observed=observed)
    elif not session_report.get("ok"):
        append_diagnosis(
            diagnoses,
            "session_failed",
            observed=observed,
            details={"workload_type": session_report.get("workload_type"), "error": session_report.get("error")},
        )
    if (
        session_report
        and expected_request_count is not None
        and int(session_report.get("request_count") or 0) != int(expected_request_count)
    ):
        append_diagnosis(
            diagnoses,
            "request_count_mismatch",
            observed=observed,
            details={
                "expected_request_count": int(expected_request_count),
                "actual_request_count": session_report.get("request_count"),
            },
        )
    trace = (session_report or {}).get("request_trace") or []
    if session_report and (not isinstance(trace, list) or not trace or not trace[0].get("prompt")):
        append_diagnosis(
            diagnoses,
            "trace_missing",
            observed=observed,
            details={"request_trace_count": (session_report or {}).get("request_trace_count")},
        )
    if (
        not demo_skipped
        and (
            not safety.get("read_only")
            or not safety.get("redaction_ok")
            or safety.get("raw_payloads_exposed")
        )
    ):
        append_diagnosis(diagnoses, "unsafe_session", observed=observed, details={"safety": safety})
    if not diagnoses:
        append_diagnosis(
            diagnoses,
            "home_compute_ready",
            observed=observed,
            details={"workload_type": WORKLOAD_TYPE, "route": route.get("name")},
        )
    diagnoses = sorted_diagnoses(diagnoses)
    return {
        "primary": diagnoses[0],
        "all": diagnoses,
        "codes": [str(item.get("primary_code")) for item in diagnoses],
    }


def build_home_compute_report(
    *,
    matrix: dict[str, Any],
    session_report: dict[str, Any] | None,
    expected_request_count: int | None = None,
) -> dict[str, Any]:
    selected = workload_status(matrix, WORKLOAD_TYPE)
    route = selected_route(matrix)
    available = selected.get("status") in {"available", "configured"}
    route_ok = (
        route.get("name") == "local_cpu_model_bundle_infer"
        and route.get("target") == "cpu_baseline"
        and route.get("workload") == WORKLOAD_TYPE
        and bool(route.get("usable_now"))
    )
    session_ok = bool(session_report and session_report.get("ok"))
    read_only = bool(session_report and session_report.get("read_only"))
    redaction_ok = bool(session_report and session_report.get("redaction_ok"))
    safety = {
        "read_only": read_only,
        "redaction_ok": redaction_ok,
        "raw_payloads_exposed": bool(session_report) and not redaction_ok,
    }
    diagnosis = diagnose_home_compute(
        matrix=matrix,
        selected=selected,
        route=route,
        session_report=session_report,
        safety=safety,
        expected_request_count=expected_request_count,
    )
    host = matrix.get("host_profile", {})
    summary = matrix.get("summary", {})
    report = {
        "ok": bool(matrix.get("ok") and available and route_ok and session_ok and read_only and redaction_ok),
        "demo": "home_compute_inference_v1",
        "selected_workload": {
            "name": WORKLOAD_TYPE,
            "status": selected.get("status"),
            "reason": selected.get("reason"),
            "cpu_only": bool(selected.get("cpu_only", True)),
        },
        "scenario": {
            "scenario_schema": (session_report or {}).get("scenario_schema"),
            "scenario_id": (session_report or {}).get("scenario_id"),
            "scenario_description": (session_report or {}).get("scenario_description"),
            "scenario_request_count": (session_report or {}).get("scenario_request_count"),
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
            "diagnosis_summary": matrix.get("diagnosis_summary", {}),
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
        "diagnosis": diagnosis["primary"],
        "diagnosis_codes": diagnosis["codes"],
        "diagnoses": diagnosis["all"],
        "safety": safety,
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
    diagnosis = report.get("diagnosis") or {}
    print("CrowdTensor home-compute demo")
    print(f"  ok: {report['ok']}")
    print(
        "  diagnosis: "
        f"{diagnosis.get('primary_code')} "
        f"severity={diagnosis.get('severity')} "
        f"summary={diagnosis.get('summary')}"
    )
    print(
        "  host: "
        f"python={host.get('python')} os={host.get('os')} "
        f"machine={host.get('machine')} cpu_count={host.get('cpu_count')}"
    )
    print(f"  selected workload: {selected['name']} ({selected['status']})")
    scenario = report.get("scenario") or {}
    if scenario.get("scenario_id"):
        print(f"  scenario: {scenario.get('scenario_id')} ({scenario.get('scenario_schema')})")
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
    for step in diagnosis.get("next_steps") or []:
        print(f"    - {step}")
    for command in report["recommended_next_commands"]:
        print(f"    - {command}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matrix-guided local CrowdTensorD home-compute demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8909)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="")
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
        return build_home_compute_report(
            matrix=matrix,
            session_report=None,
            expected_request_count=args.request_count,
        )
    session_report = inference_session_demo.run_demo(args)
    return build_home_compute_report(
        matrix=matrix,
        session_report=session_report,
        expected_request_count=args.request_count,
    )


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
