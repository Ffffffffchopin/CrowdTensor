#!/usr/bin/env python3
"""Build a maintainer-facing release readiness report for CrowdTensorD."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import release_evidence_pack  # noqa: E402
import release_gate  # noqa: E402
import security_preflight  # noqa: E402


SCHEMA = "release_readiness_v1"
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
    "demo-manifest-token",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload), encoding="utf-8")


def relative_artifact_path(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def artifact_entry(
    *,
    output_dir: Path,
    path: Path,
    kind: str,
    schema: str = "",
    ok: bool | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "kind": kind,
        "path": relative_artifact_path(path, output_dir),
        "present": path.is_file(),
    }
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    cwd: Path,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        completed = runner(
            command,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {"name": name, "ok": False, "returncode": None, "error": "timeout"}, {}

    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    if not step["ok"]:
        if completed.stderr:
            step["stderr_tail"] = completed.stderr[-1000:]
        if completed.stdout and not payload:
            step["stdout_tail"] = completed.stdout[-1000:]
    return step, payload


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


def safe_remote_origin(value: str) -> str:
    if "@" not in value:
        return value
    prefix, suffix = value.rsplit("@", 1)
    if "://" in prefix:
        scheme = prefix.split("://", 1)[0]
        return f"{scheme}://<redacted>@{suffix}"
    return f"<redacted>@{suffix}"


def safe_git_info(git_info: dict[str, Any]) -> dict[str, Any]:
    status = [str(line) for line in git_info.get("status") or []]
    return {
        "available": bool(git_info.get("available")),
        "commit": git_info.get("commit", ""),
        "branch": git_info.get("branch", ""),
        "remote_origin": safe_remote_origin(str(git_info.get("remote_origin", ""))),
        "dirty": bool(git_info.get("dirty")),
        "status_count": len(status),
        "status": status[:100],
        "error": git_info.get("error", ""),
    }


def demo_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    summaries = manifest.get("summaries") if isinstance(manifest.get("summaries"), dict) else {}
    remote = summaries.get("remote_compute_evidence") if isinstance(summaries.get("remote_compute_evidence"), dict) else {}
    external = summaries.get("external_llm_evidence") if isinstance(summaries.get("external_llm_evidence"), dict) else {}
    support = summaries.get("support_bundle") if isinstance(summaries.get("support_bundle"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    external_skipped = bool(external.get("skipped"))
    return {
        "ok": bool(manifest.get("ok")),
        "schema": manifest.get("schema", ""),
        "mode": manifest.get("mode", ""),
        "artifact_count": len(artifacts),
        "remote_compute_evidence_ok": remote.get("ok"),
        "external_llm_evidence_ok": True if external_skipped else external.get("ok"),
        "external_llm_evidence_skipped": external_skipped,
        "support_release_gate_ok": support.get("release_gate_ok"),
    }


def run_demo_manifest(
    args: argparse.Namespace,
    *,
    root: Path,
    output_dir: Path,
    runner: Runner,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.skip_demo_manifest:
        return {"name": "demo_manifest", "ok": False, "skipped": True, "reason": "operator_skipped"}, {}
    manifest_dir = output_dir / "demo-manifest"
    command = [
        sys.executable,
        str(root / "scripts" / "demo_manifest_pack.py"),
        "--output-dir",
        str(manifest_dir),
        "--host",
        args.host,
        "--port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
    ]
    if args.skip_external_llm_evidence:
        command.append("--skip-external-llm-evidence")
    step, payload = run_json_step(
        "demo_manifest",
        command,
        runner=runner,
        cwd=root,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    return step, payload


def report_warning_codes(reports: dict[str, dict[str, Any]]) -> list[str]:
    codes: list[str] = []
    for name, report in reports.items():
        if not report.get("present"):
            codes.append(f"{name}_report_missing")
        elif report.get("ok") is not True:
            codes.append(f"{name}_report_failed")
    return codes


def build_status(
    *,
    git_info: dict[str, Any],
    release_gate_report: dict[str, Any],
    security_report: dict[str, Any],
    demo_step: dict[str, Any],
    demo_manifest: dict[str, Any],
    acceptance_reports: dict[str, dict[str, Any]],
    allow_dirty: bool,
) -> dict[str, Any]:
    blocking: list[str] = []
    diagnosis: list[str] = []

    if not git_info.get("available"):
        blocking.append(f"git metadata unavailable: {git_info.get('error', '')}")
        diagnosis.append("git_unavailable")
    if git_info.get("dirty") and not allow_dirty:
        blocking.append("git worktree is dirty")
        diagnosis.append("git_dirty")
    if release_gate_report.get("ok") is not True:
        failed = release_evidence_pack.summarize_checks(release_gate_report)["failed"]
        blocking.append("release gate failed" + (f": {', '.join(failed)}" if failed else ""))
        diagnosis.append("release_gate_failed")
    if security_report.get("ok") is not True:
        blocking.append("security preflight failed")
        diagnosis.append("security_preflight_failed")
    if demo_step.get("skipped"):
        blocking.append("demo manifest was skipped")
        diagnosis.append("demo_manifest_skipped")
    elif demo_manifest.get("ok") is not True:
        blocking.append("demo manifest failed")
        diagnosis.append("demo_manifest_failed")
    for name, report in acceptance_reports.items():
        if report.get("present") and report.get("ok") is not True:
            blocking.append(f"{name} acceptance report is not ok")
            diagnosis.append(f"{name}_report_failed")

    warnings = report_warning_codes(acceptance_reports)
    if not blocking:
        diagnosis.append("release_ready")
    diagnosis.extend(warnings)
    return {
        "ready": not blocking,
        "status": "ready" if not blocking else "blocked",
        "blocking_reasons": blocking,
        "warnings": warnings,
        "diagnosis_codes": sorted(set(diagnosis)),
        "allow_dirty": bool(allow_dirty),
    }


def secret_leak_fragments(payload: dict[str, Any]) -> list[str]:
    encoded = json.dumps(payload, sort_keys=True)
    return [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]


def build_readiness(
    args: argparse.Namespace,
    *,
    runner: Runner = subprocess.run,
    release_gate_report: dict[str, Any] | None = None,
    security_report: dict[str, Any] | None = None,
    git_info: dict[str, Any] | None = None,
    demo_manifest: dict[str, Any] | None = None,
    demo_step: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    release_gate_report = release_gate_report if release_gate_report is not None else release_gate.run_release_gate(root)
    security_report = security_report if security_report is not None else security_preflight.run_preflight(security_args(args))
    git_info = git_info if git_info is not None else release_evidence_pack.collect_git_info(
        root,
        git_dir=args.git_dir,
        work_tree=args.work_tree,
    )
    if demo_manifest is None or demo_step is None:
        demo_step, demo_manifest = run_demo_manifest(args, root=root, output_dir=output_dir, runner=runner)
    demo_manifest = demo_manifest or {}
    demo_step = demo_step or {"name": "demo_manifest", "ok": False, "error": "missing demo step"}

    acceptance_reports = {
        "runtime": release_evidence_pack.load_acceptance_report(args.runtime_report, name="runtime", required=False),
        "browser": release_evidence_pack.load_acceptance_report(args.browser_report, name="browser", required=False),
        "remote": release_evidence_pack.load_acceptance_report(args.remote_report, name="remote", required=False),
    }
    status = build_status(
        git_info=git_info,
        release_gate_report=release_gate_report,
        security_report=security_report,
        demo_step=demo_step,
        demo_manifest=demo_manifest,
        acceptance_reports=acceptance_reports,
        allow_dirty=bool(args.allow_dirty),
    )

    readiness_json = output_dir / "release_readiness.json"
    readiness_md = output_dir / "release_readiness.md"
    manifest_dir = output_dir / "demo-manifest"
    artifacts = {
        "release_readiness_json": {
            "kind": "release_readiness",
            "path": relative_artifact_path(readiness_json, output_dir),
            "present": False,
            "schema": SCHEMA,
        },
        "release_readiness_markdown": {
            "kind": "release_readiness_markdown",
            "path": relative_artifact_path(readiness_md, output_dir),
            "present": False,
        },
        "demo_manifest_json": artifact_entry(
            output_dir=output_dir,
            path=manifest_dir / "demo_manifest.json",
            kind="demo_manifest",
            schema=str(demo_manifest.get("schema") or "demo_manifest_v1"),
            ok=demo_manifest.get("ok") if demo_manifest else None,
        ),
        "demo_manifest_markdown": artifact_entry(
            output_dir=output_dir,
            path=manifest_dir / "demo_manifest.md",
            kind="demo_manifest_markdown",
        ),
    }
    payload = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(status["ready"]),
        "root": str(root),
        "output_dir": str(output_dir),
        "project": release_evidence_pack.read_pyproject(root),
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "git": safe_git_info(git_info),
        "checks": {
            "release_gate": release_evidence_pack.summarize_checks(release_gate_report),
            "security_preflight": release_evidence_pack.summarize_checks(security_report, label_key="id"),
        },
        "reports": {
            "demo_manifest": {
                "ok": bool(demo_step.get("ok")),
                "skipped": bool(demo_step.get("skipped")),
                "step": demo_step,
                "summary": demo_manifest_summary(demo_manifest),
            },
            "acceptance": acceptance_reports,
        },
        "release_status": status,
        "artifacts": artifacts,
        "limitations": [
            "Release readiness is an Alpha maintainer check, not a production network claim",
            "Missing runtime/browser/remote acceptance reports are warnings unless provided and failing",
            "No GPU pooling, WebGPU model shards, libp2p discovery, NAT traversal, or incentives are claimed",
        ],
        "recommended_next_commands": [
            "python3 -m unittest discover -s tests -v",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
            "python3 scripts/release_evidence_pack.py --runtime-report /tmp/crowdtensor_acceptance.json --json-out dist/release-evidence.json --markdown-out dist/release-evidence.md",
        ],
    }
    leaks = secret_leak_fragments(payload)
    if leaks:
        payload["ok"] = False
        payload["release_status"]["ready"] = False
        payload["release_status"]["status"] = "blocked"
        payload["release_status"]["blocking_reasons"].append("release readiness report leaked secret-like material")
        payload["release_status"]["diagnosis_codes"] = sorted(set(
            list(payload["release_status"].get("diagnosis_codes") or []) + ["secret_redaction_failed"]
        ))
        payload["safety_error"] = f"secret-like fragments: {', '.join(leaks)}"

    write_json(payload, readiness_json)
    write_markdown(payload, readiness_md)
    payload["artifacts"]["release_readiness_json"]["present"] = readiness_json.is_file()
    payload["artifacts"]["release_readiness_markdown"]["present"] = readiness_md.is_file()
    write_json(payload, readiness_json)
    write_markdown(payload, readiness_md)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    status = payload.get("release_status") or {}
    git = payload.get("git") or {}
    checks = payload.get("checks") or {}
    demo = ((payload.get("reports") or {}).get("demo_manifest") or {}).get("summary") or {}
    acceptance = ((payload.get("reports") or {}).get("acceptance") or {})
    lines = [
        "# CrowdTensor Release Readiness",
        "",
        f"Status: `{status.get('status', '')}`",
        f"Schema: `{payload.get('schema', '')}`",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Branch: `{git.get('branch', '')}`",
        f"Commit: `{git.get('commit', '')}`",
        f"Dirty: `{git.get('dirty')}` status_count=`{git.get('status_count')}`",
        "",
        "## Checks",
        "",
    ]
    for name, check in checks.items():
        failed = ", ".join(check.get("failed") or []) or "none"
        lines.append(f"- `{name}`: ok=`{check.get('ok')}` total=`{check.get('total')}` failed=`{failed}`")
    lines.extend([
        "",
        "## Demo Manifest",
        "",
        f"- OK: `{demo.get('ok')}`",
        f"- Schema: `{demo.get('schema')}`",
        f"- Mode: `{demo.get('mode')}`",
        f"- Artifacts: `{demo.get('artifact_count')}`",
        "",
        "## Acceptance Reports",
        "",
    ])
    for name, report in acceptance.items():
        state = "missing" if not report.get("present") else report.get("ok")
        lines.append(f"- `{name}`: `{state}` checks=`{report.get('checks_total')}`")
    if status.get("blocking_reasons"):
        lines.extend(["", "## Blocking Reasons", ""])
        lines.extend([f"- {reason}" for reason in status.get("blocking_reasons") or []])
    if status.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in status.get("warnings") or []])
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CrowdTensorD release readiness report.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output-dir", default="dist/release-readiness")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8924)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--external-llm-request-count", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-demo-manifest", action="store_true")
    parser.add_argument("--skip-external-llm-evidence", action="store_true")
    parser.add_argument("--runtime-report", default="")
    parser.add_argument("--browser-report", default="")
    parser.add_argument("--remote-report", default="")
    parser.add_argument("--git-dir", default="")
    parser.add_argument("--work-tree", default="")
    parser.add_argument("--security-host", default="127.0.0.1")
    parser.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    parser.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    parser.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    parser.add_argument("--miner-token-registry", default=os.environ.get("CROWDTENSOR_MINER_TOKEN_REGISTRY", ""))
    parser.add_argument("--cors-origin", action="append", dest="cors_origins", default=[])
    parser.add_argument("--security-strict", action="store_true")
    args = parser.parse_args(argv)
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.external_llm_request_count < 1:
        raise SystemExit("--external-llm-request-count must be at least 1")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be at least 1")
    return args


def main() -> None:
    try:
        payload = build_readiness(parse_args())
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(0 if payload.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
