#!/usr/bin/env python3
"""Build a safe CrowdTensorD operator support bundle."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import doctor  # noqa: E402
import release_evidence_pack  # noqa: E402
import release_gate  # noqa: E402


SENSITIVE_FRAGMENTS = (
    "token",
    "secret",
    "key",
    "delta",
    "weights",
    "lease",
    "idempotency",
)
SAFE_SENSITIVE_NAMED_FIELDS = {
    "all_token_events_ready",
    "complete_token_count",
    "external_generated_token_count",
    "generated_token_count",
    "generated_token_count_ready",
    "generated_token_ids_public",
    "lease_expired",
    "max_observed_token_count",
    "multi_token_generation_ready",
    "tiny_gpt2_multi_token_ready",
    "next_token_redacted",
    "observed_token_counts",
    "peer_secret_gossiped",
    "required_generated_token_count",
    "target_token_count",
    "token_target_ready",
    "token_rotation_required",
    "private_kernel_payload_contains_miner_token",
    "private_kernel_payload_contains_peer_secret",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_pyproject(root: Path) -> dict[str, Any]:
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload.get("project", {})
    return {
        "name": project.get("name", ""),
        "version": project.get("version", ""),
        "requires_python": project.get("requires-python", ""),
    }


def is_sensitive_key(key: str) -> bool:
    lowered = str(key).lower()
    parts = [part for part in re.split(r"[^a-z0-9]+", lowered) if part]
    return any(fragment in parts for fragment in SENSITIVE_FRAGMENTS)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in SAFE_SENSITIVE_NAMED_FIELDS:
                sanitized[str(key)] = sanitize(item)
            elif is_sensitive_key(str(key)):
                sanitized[str(key)] = "<redacted>"
            else:
                sanitized[str(key)] = sanitize(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def summarize_checks(report: dict[str, Any], *, label_key: str = "name") -> dict[str, Any]:
    checks = report.get("checks") if isinstance(report, dict) else []
    if not isinstance(checks, list):
        checks = []
    failed = [
        str(check.get(label_key) or check.get("id") or "<unnamed>")
        for check in checks
        if isinstance(check, dict) and check.get("ok") is not True
    ]
    return {
        "ok": bool(report.get("ok")) if isinstance(report, dict) else False,
        "total": len(checks),
        "failed": failed,
    }


def summarize_diagnosis(report: dict[str, Any]) -> dict[str, Any]:
    direct_codes = report.get("diagnosis_codes") if isinstance(report, dict) else []
    direct_by_check = report.get("diagnosis_by_check") if isinstance(report, dict) else {}
    direct_failed = report.get("failed_checks") if isinstance(report, dict) else []
    if direct_codes or direct_by_check or direct_failed:
        return {
            "codes": sorted(set(str(code) for code in direct_codes or [] if code)),
            "by_check": {
                str(name): [str(code) for code in codes if code]
                for name, codes in dict(direct_by_check or {}).items()
            },
            "failed_checks": [str(name) for name in direct_failed or [] if name],
        }
    diagnosis = report.get("diagnosis_summary") if isinstance(report, dict) else {}
    if isinstance(diagnosis, dict) and diagnosis:
        return {
            "codes": list(diagnosis.get("codes") or []),
            "by_check": dict(diagnosis.get("by_check") or {}),
            "failed_checks": list(diagnosis.get("failed_checks") or []),
        }
    nested_reports = report.get("reports") if isinstance(report, dict) else {}
    if isinstance(nested_reports, dict) and nested_reports:
        codes: list[str] = []
        by_check: dict[str, list[str]] = {}
        failed: list[str] = []
        for report_name, nested_report in nested_reports.items():
            if not isinstance(nested_report, dict):
                continue
            nested = summarize_diagnosis(nested_report)
            codes.extend(nested["codes"])
            failed.extend(f"{report_name}.{name}" for name in nested["failed_checks"])
            for check_name, check_codes in nested["by_check"].items():
                by_check[f"{report_name}.{check_name}"] = check_codes
        return {"codes": sorted(set(codes)), "by_check": by_check, "failed_checks": failed}
    checks = report.get("checks") if isinstance(report, dict) else []
    if not isinstance(checks, list):
        checks = []
    by_check: dict[str, list[str]] = {}
    codes: list[str] = []
    failed: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "<unnamed>")
        if check.get("ok") is not True:
            failed.append(name)
        check_codes = [str(code) for code in check.get("diagnosis_codes") or [] if code]
        if check_codes:
            by_check[name] = check_codes
            codes.extend(check_codes)
    return {"codes": sorted(set(codes)), "by_check": by_check, "failed_checks": failed}


def safe_observability_summary(summary: dict[str, Any]) -> dict[str, Any]:
    schema = summary.get("schema")
    if schema == "remote_compute_observability_v1":
        route = summary.get("route") if isinstance(summary.get("route"), dict) else {}
        miner = summary.get("miner") if isinstance(summary.get("miner"), dict) else {}
        work_queue = summary.get("work_queue") if isinstance(summary.get("work_queue"), dict) else {}
        inference = summary.get("inference") if isinstance(summary.get("inference"), dict) else {}
        safety = summary.get("safety") if isinstance(summary.get("safety"), dict) else {}
        return {
            "schema": schema,
            "route": {
                "name": route.get("name"),
                "confidence": route.get("confidence"),
                "usable_now": route.get("usable_now"),
                "matched_capabilities": list(route.get("matched_capabilities") or []),
                "missing_capabilities": list(route.get("missing_capabilities") or []),
            },
            "miner": {
                "runtime": miner.get("runtime"),
                "backend": miner.get("backend"),
                "accepted": miner.get("accepted"),
                "rejected": miner.get("rejected"),
            },
            "work_queue": {
                "task_counts": work_queue.get("task_counts", {}),
                "accepted_results": work_queue.get("accepted_results"),
                "rejected_results": work_queue.get("rejected_results"),
                "ledger_rows": work_queue.get("ledger_rows"),
            },
            "inference": {
                "ok": inference.get("ok"),
                "request_count": inference.get("request_count"),
                "expected_request_count": inference.get("expected_request_count"),
                "request_trace_count": inference.get("request_trace_count"),
                "accuracy": inference.get("accuracy"),
                "elapsed_ms": inference.get("elapsed_ms"),
                "requests_per_second": inference.get("requests_per_second"),
            },
            "safety": {
                "read_only": safety.get("read_only"),
                "redaction_ok": safety.get("redaction_ok"),
                "registry_hashed": safety.get("registry_hashed"),
                "raw_payloads_exposed": safety.get("raw_payloads_exposed"),
            },
        }
    if schema == "remote_demo_observability_v1":
        availability = summary.get("availability") if isinstance(summary.get("availability"), dict) else {}
        work_queue = summary.get("work_queue") if isinstance(summary.get("work_queue"), dict) else {}
        miner = summary.get("miner") if isinstance(summary.get("miner"), dict) else {}
        inference = summary.get("inference") if isinstance(summary.get("inference"), dict) else {}
        artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
        return {
            "schema": schema,
            "route": summary.get("route"),
            "miner_id": summary.get("miner_id"),
            "availability": {
                "health_ok": availability.get("health_ok"),
                "ready_ok": availability.get("ready_ok"),
                "state_ok": availability.get("state_ok"),
                "admin_results_ok": availability.get("admin_results_ok"),
                "acceptance_ready": availability.get("acceptance_ready"),
                "attempts": availability.get("attempts"),
                "elapsed_seconds": availability.get("elapsed_seconds"),
            },
            "work_queue": {
                "task_counts": work_queue.get("task_counts", {}),
                "accepted_results": work_queue.get("accepted_results"),
                "task_id": work_queue.get("task_id"),
            },
            "miner": {
                "runtime": miner.get("runtime"),
                "backend": miner.get("backend"),
                "accepted": miner.get("accepted"),
                "rejected": miner.get("rejected"),
                "supported_workloads": list(miner.get("supported_workloads") or []),
            },
            "inference": {
                "ok": inference.get("ok"),
                "request_count": inference.get("request_count"),
                "request_trace_count": inference.get("request_trace_count"),
                "accuracy": inference.get("accuracy"),
                "requests_per_second": inference.get("requests_per_second"),
            },
            "artifacts": {
                "evidence_ok": artifacts.get("evidence_ok"),
                "support_bundle_ok": artifacts.get("support_bundle_ok"),
                "evidence_observability_schema": artifacts.get("evidence_observability_schema"),
                "support_online_enabled": artifacts.get("support_online_enabled"),
            },
            "diagnosis_codes": list(summary.get("diagnosis_codes") or []),
        }
    return {}


def summarize_observability(report: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    summary = report.get("observability_summary") if isinstance(report, dict) else {}
    if isinstance(summary, dict):
        safe = safe_observability_summary(summary)
        if safe:
            summaries.append(safe)
    nested_reports = report.get("reports") if isinstance(report, dict) else {}
    if isinstance(nested_reports, dict):
        for nested_report in nested_reports.values():
            if isinstance(nested_report, dict):
                summaries.extend(summarize_observability(nested_report))
    return summaries


def load_json_report(path: str, *, name: str, required: bool = False) -> dict[str, Any]:
    if not path:
        return {"name": name, "present": False, "required": required, "ok": not required}
    report_path = Path(path)
    if not report_path.is_file():
        return {
            "name": name,
            "path": str(report_path),
            "present": False,
            "required": required,
            "ok": not required,
            "error": "report file does not exist",
        }
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{name} report is not valid JSON: {exc}") from exc
    summary = summarize_checks(payload)
    diagnosis = summarize_diagnosis(payload)
    observability = summarize_observability(payload)
    release_status = payload.get("release_status") if isinstance(payload.get("release_status"), dict) else {}
    if release_status:
        report_ok = bool(release_status.get("ready")) or release_status.get("status") == "ready"
    else:
        report_ok = bool(payload.get("ok", summary["ok"]))
    return {
        "name": name,
        "path": str(report_path),
        "present": True,
        "required": required,
        "ok": report_ok,
        "skipped": bool(payload.get("skipped")),
        "skip_reason": payload.get("skip_reason", ""),
        "duration_seconds": payload.get("duration_seconds"),
        "checks_total": summary["total"],
        "checks_failed": summary["failed"],
        "diagnosis_codes": diagnosis["codes"],
        "diagnosis_by_check": diagnosis["by_check"],
        "failed_checks": diagnosis["failed_checks"],
        "observability_summaries": observability,
        "status": payload.get("release_status", {}).get("status", ""),
        "blocking_reasons": payload.get("release_status", {}).get("blocking_reasons", []),
    }


def request_text(
    base_url: str,
    path: str,
    *,
    token_header: tuple[str, str] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    headers = {}
    if token_header is not None and token_header[1]:
        headers[token_header[0]] = token_header[1]
    request = Request(f"{base_url.rstrip('/')}{path}", headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "content_type": response.headers.get("content-type", ""),
                "body": raw,
            }
    except HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "content_type": exc.headers.get("content-type", "") if exc.headers else "",
            "body": exc.read().decode("utf-8"),
        }
    except (OSError, URLError) as exc:
        return {"ok": False, "status": None, "error": str(exc), "body": ""}


def request_json(
    base_url: str,
    path: str,
    *,
    token_header: tuple[str, str] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    response = request_text(base_url, path, token_header=token_header, timeout=timeout)
    if not response.get("body"):
        return response
    try:
        response["json"] = sanitize(json.loads(str(response["body"])))
        response.pop("body", None)
    except json.JSONDecodeError:
        response["body"] = str(response["body"])[:1000]
    return response


def state_digest(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("json") if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        return payload
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    return {
        "ok": payload.get("ok"),
        "status": payload.get("status"),
        "event_index": data.get("event_index"),
        "task_counts": data.get("task_counts", {}),
        "accepted_results": data.get("accepted_results"),
        "rejected_results": data.get("rejected_results"),
        "model": {
            "global_step": model.get("global_step"),
            "optimizer_step": model.get("optimizer_step"),
            "adapter_step": model.get("adapter_step"),
            "micro_transformer_step": (model.get("micro_transformer") or {}).get("optimizer_step")
            if isinstance(model.get("micro_transformer"), dict)
            else None,
            "model_bundle_step": (model.get("model_bundle") or {}).get("optimizer_step")
            if isinstance(model.get("model_bundle"), dict)
            else None,
        },
        "miner_profiles": len(data.get("miner_profiles") or {}),
        "miner_workload_scores": len(data.get("miner_workload_scores") or {}),
        "task_lanes": len(data.get("task_lanes") or []),
    }


def metrics_digest(payload: dict[str, Any]) -> dict[str, Any]:
    body = str(payload.get("body") or "")
    return {
        "ok": payload.get("ok"),
        "status": payload.get("status"),
        "content_type": payload.get("content_type", ""),
        "line_count": len([line for line in body.splitlines() if line.strip()]),
        "text": body,
        "error": payload.get("error", ""),
    }


def collect_online(args: argparse.Namespace) -> dict[str, Any]:
    if not args.coordinator:
        return {"enabled": False}
    observer_header = ("x-crowdtensor-observer-token", args.observer_token)
    admin_header = ("x-crowdtensor-admin-token", args.admin_token)
    online: dict[str, Any] = {
        "enabled": True,
        "base_url": args.coordinator.rstrip("/"),
        "health": request_json(args.coordinator, "/health", timeout=args.timeout),
        "version": request_json(args.coordinator, "/version", timeout=args.timeout),
        "ready": request_json(args.coordinator, "/ready", timeout=args.timeout),
        "metrics": metrics_digest(
            request_text(
                args.coordinator,
                "/metrics",
                token_header=observer_header if args.observer_token else None,
                timeout=args.timeout,
            )
        ),
    }
    if args.observer_token:
        online["state"] = state_digest(
            request_json(args.coordinator, "/state", token_header=observer_header, timeout=args.timeout)
        )
    else:
        online["state"] = {"present": False, "reason": "observer token not provided"}
    if args.admin_token:
        online["admin_results"] = request_json(
            args.coordinator,
            f"/admin/results?limit={args.admin_results_limit}",
            token_header=admin_header,
            timeout=args.timeout,
        )
    else:
        online["admin_results"] = {"present": False, "reason": "admin token not provided"}
    return sanitize(online)


def doctor_args(args: argparse.Namespace) -> argparse.Namespace:
    return doctor.parse_args([
        "--root",
        str(args.root),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--state-dir",
        args.state_dir,
    ])


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    doctor_report = doctor.build_report(doctor_args(args))
    release_gate_report = release_gate.run_release_gate(root)
    reports = {
        "runtime": load_json_report(args.runtime_report, name="runtime"),
        "browser": load_json_report(args.browser_report, name="browser"),
        "remote": load_json_report(args.remote_report, name="remote"),
        "release_evidence": load_json_report(args.release_evidence, name="release_evidence"),
    }
    return sanitize({
        "generated_at": utc_now(),
        "project": read_pyproject(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "root": str(root),
        },
        "git": release_evidence_pack.collect_git_info(
            root,
            git_dir=args.git_dir,
            work_tree=args.work_tree,
        ),
        "doctor": {
            "ok": doctor_report.get("ok"),
            "summary": doctor_report.get("summary", {}),
            "checks": doctor_report.get("checks", []),
        },
        "release_gate": summarize_checks(release_gate_report),
        "reports": reports,
        "online": collect_online(args),
    })


def write_json(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensorD Support Bundle",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Project: `{payload.get('project', {}).get('name', '')}` `{payload.get('project', {}).get('version', '')}`",
        f"Commit: `{payload.get('git', {}).get('commit', '')}`",
        "",
        "## Summary",
        "",
        f"- Doctor: `{payload.get('doctor', {}).get('ok')}`",
        f"- Release gate: `{payload.get('release_gate', {}).get('ok')}`",
        f"- Online collection: `{payload.get('online', {}).get('enabled')}`",
        "",
        "## Reports",
        "",
    ]
    for name, report in payload.get("reports", {}).items():
        state = "missing" if not report.get("present") else "ok" if report.get("ok") else "failed"
        lines.append(f"- `{name}`: {state}, checks={report.get('checks_total', 0)}")
        diagnosis_codes = report.get("diagnosis_codes") or []
        if diagnosis_codes:
            lines.append(f"  - diagnosis: `{', '.join(diagnosis_codes)}`")
        for observed in report.get("observability_summaries") or []:
            inference = observed.get("inference") or {}
            availability = observed.get("availability") or {}
            route = observed.get("route") or {}
            route_name = route.get("name") if isinstance(route, dict) else route
            detail = f"requests={inference.get('request_count')}, rps={inference.get('requests_per_second')}"
            if availability:
                detail += f", health={availability.get('health_ok')}, state={availability.get('state_ok')}"
            lines.append(f"  - observability `{observed.get('schema')}` route=`{route_name}` {detail}")
    lines.append("")
    return "\n".join(lines)


def write_markdown(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a safe CrowdTensorD operator support bundle.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default="state/support-bundle")
    parser.add_argument("--coordinator", default="")
    parser.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    parser.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    parser.add_argument("--admin-results-limit", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--runtime-report", default="")
    parser.add_argument("--browser-report", default="")
    parser.add_argument("--remote-report", default="")
    parser.add_argument("--release-evidence", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--git-dir", default="")
    parser.add_argument("--work-tree", default="")
    return parser.parse_args(argv)


def main() -> None:
    try:
        args = parse_args()
        bundle = build_bundle(args)
        write_json(bundle, args.json_out)
        write_markdown(bundle, args.markdown_out)
        print(json.dumps(bundle, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
