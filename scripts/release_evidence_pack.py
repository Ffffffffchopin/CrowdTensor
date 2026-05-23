#!/usr/bin/env python3
"""Build a release evidence report for a CrowdTensorD checkout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import release_gate  # noqa: E402
import security_preflight  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_pyproject(root: Path) -> dict[str, Any]:
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload.get("project", {})
    return {
        "name": project.get("name", ""),
        "version": project.get("version", ""),
        "requires_python": project.get("requires-python", ""),
        "scripts": project.get("scripts", {}),
    }


def git_command(root: Path, git_dir: str, work_tree: str, args: list[str]) -> tuple[bool, str]:
    command = ["git"]
    if git_dir:
        command.append(f"--git-dir={git_dir}")
    if work_tree:
        command.append(f"--work-tree={work_tree}")
    command.extend(args)
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
    )
    output = (completed.stdout or "").strip()
    if completed.returncode == 0:
        return True, output
    return False, (completed.stderr or output or "git command failed").strip()


def collect_git_info(root: Path, *, git_dir: str = "", work_tree: str = "") -> dict[str, Any]:
    ok, commit = git_command(root, git_dir, work_tree, ["rev-parse", "HEAD"])
    if not ok:
        return {
            "available": False,
            "commit": "",
            "branch": "",
            "remote_origin": "",
            "dirty": True,
            "status": [],
            "error": commit,
        }
    branch_ok, branch = git_command(root, git_dir, work_tree, ["rev-parse", "--abbrev-ref", "HEAD"])
    remote_ok, remote = git_command(root, git_dir, work_tree, ["config", "--get", "remote.origin.url"])
    status_ok, status = git_command(root, git_dir, work_tree, ["status", "--porcelain"])
    status_lines = [line for line in status.splitlines() if line.strip()] if status_ok else []
    return {
        "available": True,
        "commit": commit,
        "branch": branch if branch_ok else "",
        "remote_origin": remote if remote_ok else "",
        "dirty": bool(status_lines),
        "status": status_lines,
        "error": "" if status_ok else status,
    }


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


def load_acceptance_report(path: str, *, name: str, required: bool) -> dict[str, Any]:
    if not path:
        return {
            "name": name,
            "path": "",
            "present": False,
            "required": required,
            "ok": not required,
            "skipped": False,
            "error": "missing required report path" if required else "",
            "checks_total": 0,
            "checks_failed": [],
        }
    report_path = Path(path)
    if not report_path.is_file():
        return {
            "name": name,
            "path": str(report_path),
            "present": False,
            "required": required,
            "ok": not required,
            "skipped": False,
            "error": "report file does not exist" if required else "",
            "checks_total": 0,
            "checks_failed": [],
        }
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "name": name,
            "path": str(report_path),
            "present": True,
            "required": required,
            "ok": False,
            "skipped": False,
            "error": f"could not parse report JSON: {exc}",
            "checks_total": 0,
            "checks_failed": [],
        }
    summary = summarize_checks(payload)
    diagnosis = summarize_diagnosis(payload)
    observability = summarize_observability(payload)
    return {
        "name": name,
        "path": str(report_path),
        "present": True,
        "required": required,
        "ok": summary["ok"],
        "skipped": bool(payload.get("skipped")),
        "skip_reason": payload.get("skip_reason", ""),
        "duration_seconds": payload.get("duration_seconds"),
        "started_at": payload.get("started_at", ""),
        "finished_at": payload.get("finished_at", ""),
        "checks_total": summary["total"],
        "checks_failed": summary["failed"],
        "diagnosis_codes": diagnosis["codes"],
        "diagnosis_by_check": diagnosis["by_check"],
        "failed_checks": diagnosis["failed_checks"],
        "observability_summaries": observability,
    }


def security_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        host=args.security_host,
        miner_token=args.miner_token,
        observer_token=args.observer_token,
        admin_token=args.admin_token,
        miner_token_registry=args.miner_token_registry,
        cors_origins=args.cors_origins,
        strict=args.security_strict,
    )


def blocking_reasons(
    *,
    git_info: dict[str, Any],
    release_gate_report: dict[str, Any],
    security_report: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    allow_dirty: bool,
) -> list[str]:
    reasons: list[str] = []
    if not git_info.get("available"):
        reasons.append(f"git metadata unavailable: {git_info.get('error', '')}")
    if git_info.get("dirty") and not allow_dirty:
        reasons.append("git worktree is dirty")
    if not release_gate_report.get("ok"):
        failed = summarize_checks(release_gate_report)["failed"]
        reasons.append("release gate failed" + (f": {', '.join(failed)}" if failed else ""))
    if not security_report.get("ok"):
        reasons.append("security preflight failed")
    for name, report in reports.items():
        if report.get("required") and not report.get("present"):
            reasons.append(f"{name} acceptance report is missing")
        elif report.get("present") and not report.get("ok"):
            reasons.append(f"{name} acceptance report is not ok")
    return reasons


def build_evidence(
    args: argparse.Namespace,
    *,
    release_gate_report: dict[str, Any] | None = None,
    security_report: dict[str, Any] | None = None,
    git_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(args.root).resolve()
    release_gate_report = release_gate_report if release_gate_report is not None else release_gate.run_release_gate(root)
    security_report = security_report if security_report is not None else security_preflight.run_preflight(security_args(args))
    git_info = git_info if git_info is not None else collect_git_info(
        root,
        git_dir=args.git_dir,
        work_tree=args.work_tree,
    )
    optional_required = bool(args.strict_optional and not args.allow_missing_optional)
    reports = {
        "runtime": load_acceptance_report(args.runtime_report, name="runtime", required=True),
        "browser": load_acceptance_report(args.browser_report, name="browser", required=optional_required),
        "remote": load_acceptance_report(args.remote_report, name="remote", required=optional_required),
    }
    reasons = blocking_reasons(
        git_info=git_info,
        release_gate_report=release_gate_report,
        security_report=security_report,
        reports=reports,
        allow_dirty=bool(args.allow_dirty),
    )
    return {
        "generated_at": utc_now(),
        "project": read_pyproject(root),
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "git": git_info,
        "checks": {
            "release_gate": summarize_checks(release_gate_report),
            "security_preflight": summarize_checks(security_report, label_key="id"),
        },
        "reports": reports,
        "release_status": {
            "ready": not reasons,
            "status": "ready" if not reasons else "blocked",
            "blocking_reasons": reasons,
            "allow_dirty": bool(args.allow_dirty),
            "allow_missing_optional": bool(args.allow_missing_optional),
            "strict_optional": optional_required,
        },
    }


def write_json(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_status(value: bool) -> str:
    return "ok" if value else "blocked"


def render_markdown(payload: dict[str, Any]) -> str:
    status = payload["release_status"]
    git = payload["git"]
    project = payload["project"]
    lines = [
        "# CrowdTensorD Release Evidence",
        "",
        f"Status: `{status['status']}`",
        f"Project: `{project.get('name', '')}` `{project.get('version', '')}`",
        f"Commit: `{git.get('commit', '')}`",
        f"Branch: `{git.get('branch', '')}`",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Checks",
        "",
    ]
    for name, check in payload["checks"].items():
        failed = ", ".join(check["failed"]) if check["failed"] else "none"
        lines.append(f"- `{name}`: {markdown_status(check['ok'])}, total={check['total']}, failed={failed}")
    lines.extend(["", "## Acceptance Reports", ""])
    for name, report in payload["reports"].items():
        state = "missing" if not report["present"] else markdown_status(report["ok"])
        suffix = ""
        if report.get("skipped"):
            suffix = f", skipped={report.get('skip_reason', '')}"
        lines.append(f"- `{name}`: {state}, checks={report['checks_total']}{suffix}")
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
    if status["blocking_reasons"]:
        lines.extend(["", "## Blocking Reasons", ""])
        lines.extend([f"- {reason}" for reason in status["blocking_reasons"]])
    lines.append("")
    return "\n".join(lines)


def write_markdown(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CrowdTensorD release evidence report.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--browser-report", default="")
    parser.add_argument("--remote-report", default="")
    parser.add_argument("--json-out", default="dist/release-evidence.json")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--strict-optional", action="store_true")
    parser.add_argument("--allow-missing-optional", action="store_true", help="accepted for explicit local draft runs")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--git-dir", default="")
    parser.add_argument("--work-tree", default="")
    parser.add_argument("--security-host", default="127.0.0.1")
    parser.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    parser.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    parser.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    parser.add_argument("--miner-token-registry", default=os.environ.get("CROWDTENSOR_MINER_TOKEN_REGISTRY", ""))
    parser.add_argument("--cors-origin", action="append", dest="cors_origins", default=[])
    parser.add_argument("--security-strict", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    evidence = build_evidence(args)
    write_json(evidence, args.json_out)
    write_markdown(evidence, args.markdown_out)
    print(json.dumps(evidence, sort_keys=True))
    raise SystemExit(0 if evidence["release_status"]["ready"] else 1)


if __name__ == "__main__":
    main()
