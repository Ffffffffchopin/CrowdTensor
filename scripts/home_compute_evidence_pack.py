#!/usr/bin/env python3
"""Build a safe, shareable home-compute evidence report."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg  # noqa: E402
import home_compute_demo  # noqa: E402
import runtime_matrix  # noqa: E402
import support_bundle  # noqa: E402


EVIDENCE_SCHEMA = "home_compute_evidence_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_runtime_report(path: str) -> dict[str, Any]:
    if not path:
        return {"present": False, "ok": None, "checks_total": 0, "checks_failed": []}
    report_path = Path(path)
    if not report_path.is_file():
        return {
            "present": False,
            "ok": False,
            "path": str(report_path),
            "error": "runtime report file does not exist",
            "checks_total": 0,
            "checks_failed": [],
        }
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    checks = payload.get("checks") if isinstance(payload, dict) else []
    if not isinstance(checks, list):
        checks = []
    failed = [
        str(check.get("name") or "<unnamed>")
        for check in checks
        if isinstance(check, dict) and check.get("ok") is not True
    ]
    return {
        "present": True,
        "path": str(report_path),
        "ok": bool(payload.get("ok")),
        "duration_seconds": payload.get("duration_seconds"),
        "checks_total": len(checks),
        "checks_failed": failed,
    }


def inference_summary(session_report: dict[str, Any] | None) -> dict[str, Any]:
    session = session_report or {}
    return {
        "present": bool(session_report),
        "ok": session.get("ok"),
        "workload_type": session.get("workload_type"),
        "request_count": session.get("request_count"),
        "correct_count": session.get("correct_count"),
        "accuracy": session.get("accuracy"),
        "elapsed_ms": session.get("elapsed_ms"),
        "requests_per_second": session.get("requests_per_second"),
        "read_only": session.get("read_only"),
        "redaction_ok": session.get("redaction_ok"),
        "request_trace_count": session.get("request_trace_count"),
        "request_trace_truncated": session.get("request_trace_truncated"),
    }


def safe_host_profile(matrix: dict[str, Any]) -> dict[str, Any]:
    host = matrix.get("host_profile") or {}
    return {
        "python": host.get("python"),
        "os": host.get("os"),
        "machine": host.get("machine"),
        "cpu_count": host.get("cpu_count"),
        "platform": host.get("platform"),
    }


def fallback_workload(matrix: dict[str, Any]) -> dict[str, Any]:
    selected = home_compute_demo.workload_status(matrix, home_compute_demo.WORKLOAD_TYPE)
    return {
        "name": home_compute_demo.WORKLOAD_TYPE,
        "status": selected.get("status"),
        "reason": selected.get("reason"),
        "cpu_only": bool(selected.get("cpu_only", True)),
    }


def fallback_route_decision(matrix: dict[str, Any]) -> dict[str, Any]:
    return home_compute_demo.route_decision(home_compute_demo.selected_route(matrix))


def evidence_diagnosis(
    *,
    matrix: dict[str, Any],
    home_report: dict[str, Any] | None,
    route: dict[str, Any],
) -> dict[str, Any]:
    if home_report is not None and home_report.get("diagnosis"):
        return {
            "primary": home_report.get("diagnosis"),
            "codes": list(home_report.get("diagnosis_codes") or []),
            "all": list(home_report.get("diagnoses") or [home_report.get("diagnosis")]),
        }
    selected = home_compute_demo.workload_status(matrix, home_compute_demo.WORKLOAD_TYPE)
    safety = {"read_only": None, "redaction_ok": None, "raw_payloads_exposed": False}
    return home_compute_demo.diagnose_home_compute(
        matrix=matrix,
        selected=selected,
        route=route,
        session_report=None,
        safety=safety,
        demo_skipped=True,
    )


def build_evidence(
    *,
    matrix: dict[str, Any],
    home_report: dict[str, Any] | None,
    runtime_report: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    report = home_report or {}
    session = report.get("inference_session") if isinstance(report.get("inference_session"), dict) else None
    route = report.get("route_decision") or report.get("capability_route") or fallback_route_decision(matrix)
    safety = report.get("safety") or {}
    diagnosis = evidence_diagnosis(matrix=matrix, home_report=home_report, route=route)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "ok": bool(report.get("ok")) if home_report is not None else bool(matrix.get("ok")),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "host_profile": safe_host_profile(matrix),
        "runtime_matrix": {
            "ok": bool(matrix.get("ok")),
            "summary": matrix.get("summary", {}),
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
            "recommended_routes": matrix.get("recommended_routes", []),
        },
        "selected_workload": report.get("selected_workload") or fallback_workload(matrix),
        "route_decision": route,
        "inference_summary": inference_summary(session),
        "request_trace": list((session or {}).get("request_trace") or []),
        "diagnosis": diagnosis["primary"],
        "diagnosis_codes": diagnosis["codes"],
        "diagnoses": diagnosis["all"],
        "safety": {
            "read_only": safety.get("read_only"),
            "redaction_ok": safety.get("redaction_ok"),
            "raw_payloads_exposed": bool(safety.get("raw_payloads_exposed", False)),
        },
        "runtime_acceptance": runtime_report,
        "recommended_next_commands": report.get("recommended_next_commands") or matrix.get("recommended_next_commands", []),
        "limitations": [
            "CPU-only demo evidence; not production LLM serving",
            "No GPU pooling, WebGPU model shards, P2P routing, or token incentives are claimed",
        ],
    }
    return support_bundle.sanitize(evidence)


def build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    matrix = runtime_matrix.build_matrix(
        root=Path(args.root),
        browser_path=args.browser_path,
        llm_runtime_cmd=args.llm_runtime_cmd,
        llm_runtime_url=args.llm_runtime_url,
        llm_runtime_api_key=args.llm_runtime_api_key,
    )
    home_report = None
    if not args.skip_demo:
        demo_args = argparse.Namespace(**vars(args))
        demo_args.base_url = f"http://{args.host}:{args.port}"
        home_report = home_compute_demo.run_demo(demo_args)
    return build_evidence(
        matrix=matrix,
        home_report=home_report,
        runtime_report=load_runtime_report(args.runtime_report),
    )


def render_markdown(payload: dict[str, Any]) -> str:
    route = payload.get("route_decision") or {}
    summary = payload.get("inference_summary") or {}
    diagnosis = payload.get("diagnosis") or {}
    safety = payload.get("safety") or {}
    lines = [
        "# CrowdTensor Home Compute Evidence",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        "",
        "## Route",
        "",
        f"- Selected route: `{route.get('name', '')}`",
        f"- Target: `{route.get('target', '')}`",
        f"- Workload: `{route.get('workload', '')}`",
        f"- Confidence: `{route.get('confidence', '')}`",
        f"- Reason: {route.get('reason', '')}",
        f"- Matched capabilities: `{', '.join(route.get('matched_capabilities') or [])}`",
        f"- Missing capabilities: `{', '.join(route.get('missing_capabilities') or [])}`",
        "",
        "## Diagnosis",
        "",
        f"- Primary code: `{diagnosis.get('primary_code')}`",
        f"- Severity: `{diagnosis.get('severity')}`",
        f"- Summary: {diagnosis.get('summary', '')}",
        f"- Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "### Next Steps",
        "",
    ]
    for step in diagnosis.get("next_steps") or []:
        lines.append(f"- {step}")
    lines.extend([
        "",
        "## Inference Session",
        "",
        f"- Requests: `{summary.get('request_count')}`",
        f"- Accuracy: `{summary.get('accuracy')}`",
        f"- Elapsed ms: `{summary.get('elapsed_ms')}`",
        f"- Requests/sec: `{summary.get('requests_per_second')}`",
        f"- Read-only: `{summary.get('read_only')}`",
        f"- Redaction OK: `{summary.get('redaction_ok')}`",
        "",
        "## Trace",
        "",
    ])
    for row in payload.get("request_trace") or []:
        lines.append(
            f"- `{row.get('request_id')}` prompt=`{row.get('prompt')}` "
            f"predicted=`{row.get('predicted_token')}` target=`{row.get('target_token')}` "
            f"correct=`{row.get('correct')}`"
        )
    if not payload.get("request_trace"):
        lines.append("- No request trace captured.")
    lines.extend([
        "",
        "## Safety",
        "",
        f"- Read-only: `{safety.get('read_only')}`",
        f"- Redaction OK: `{safety.get('redaction_ok')}`",
        f"- Raw payloads exposed: `{safety.get('raw_payloads_exposed')}`",
        "",
        "## Limitations",
        "",
    ])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def write_json(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a safe CrowdTensor home-compute evidence report.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8909)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--admin-token", default="local-admin")
    parser.add_argument("--browser-path", default="")
    parser.add_argument("--llm-runtime-cmd", default="")
    parser.add_argument("--llm-runtime-url", default="")
    parser.add_argument("--llm-runtime-api-key", default="")
    parser.add_argument("--runtime-report", default="")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args(argv)
    activate_miner_token(args)
    activate_observer_token(args)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    try:
        args = parse_args()
        payload = build_from_args(args)
        write_json(payload, args.json_out)
        write_markdown(payload, args.markdown_out)
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(0 if payload.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
