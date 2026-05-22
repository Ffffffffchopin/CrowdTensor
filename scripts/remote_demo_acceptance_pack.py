#!/usr/bin/env python3
"""Validate a running two-machine remote Miner demo and collect safe artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402


ACCEPTANCE_SCHEMA = "remote_demo_acceptance_v1"
WORKLOAD_TYPE = "model_bundle_infer"
ROUTE_NAME = "remote_python_model_bundle_infer"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    observer_token: str = "",
    admin_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
    request = Request(f"{base_url.rstrip('/')}{path}", headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def admin_results_url(miner_id: str, limit: int) -> str:
    query = urlencode({
        "status": "accepted",
        "miner_id": miner_id,
        "workload_type": WORKLOAD_TYPE,
        "limit": limit,
    })
    return f"/admin/results?{query}"


def completed_task_for(state: dict[str, Any], miner_id: str) -> dict[str, Any] | None:
    completed = [
        task for task in state.get("tasks", [])
        if (
            task.get("status") == "completed"
            and task.get("workload_type") == WORKLOAD_TYPE
            and task.get("miner_id") == miner_id
        )
    ]
    return completed[-1] if completed else None


def latest_result(results: dict[str, Any]) -> dict[str, Any] | None:
    rows = results.get("results") if isinstance(results, dict) else []
    if not isinstance(rows, list) or not rows:
        return None
    return rows[0]


def miner_profile(state: dict[str, Any], miner_id: str) -> dict[str, Any]:
    return (state.get("miner_profiles") or {}).get(miner_id) or {}


def readiness_summary(
    *,
    state: dict[str, Any],
    results: dict[str, Any],
    miner_id: str,
    request_count: int,
) -> dict[str, Any]:
    profile = miner_profile(state, miner_id)
    capabilities = profile.get("last_capabilities") or {}
    workloads = capabilities.get("supported_workloads") or []
    task = completed_task_for(state, miner_id)
    row = latest_result(results)
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    matched: list[str] = []
    missing: list[str] = []
    if profile.get("runtime") == "python-cli":
        matched.append("runtime:python-cli")
    else:
        missing.append("runtime:python-cli")
    if profile.get("backend") == "cpu":
        matched.append("backend:cpu")
    else:
        missing.append("backend:cpu")
    if WORKLOAD_TYPE in workloads:
        matched.append(f"workload:{WORKLOAD_TYPE}")
    else:
        missing.append(f"workload:{WORKLOAD_TYPE}")
    if task and row:
        matched.append("accepted_result")
    else:
        missing.append("accepted_result")
    if validation.get("code") == "ok":
        matched.append("validation:ok")
    else:
        missing.append("validation:ok")
    if int(validation.get("request_count") or 0) == int(request_count):
        matched.append("request_count")
    else:
        missing.append("request_count")
    return {
        "ready": not missing,
        "route": ROUTE_NAME,
        "miner_id": miner_id,
        "matched_capabilities": matched,
        "missing_capabilities": missing,
        "task_id": (task or {}).get("task_id"),
        "accepted_results": state.get("accepted_results"),
        "task_counts": state.get("task_counts", {}),
        "profile": {
            "runtime": profile.get("runtime"),
            "backend": profile.get("backend"),
            "accepted": profile.get("accepted"),
            "rejected": profile.get("rejected"),
            "supported_workloads": list(workloads),
        },
        "inference": {
            "ok": validation.get("code") == "ok",
            "request_count": validation.get("request_count"),
            "request_trace_count": validation.get("request_trace_count"),
            "accuracy": validation.get("accuracy"),
        },
    }


def collect_status(args: argparse.Namespace) -> dict[str, Any]:
    health = request_json("GET", args.coordinator_url, "/health", timeout=args.http_timeout)
    ready = request_json("GET", args.coordinator_url, "/ready", timeout=args.http_timeout)
    state = request_json(
        "GET",
        args.coordinator_url,
        "/state",
        observer_token=args.observer_token,
        timeout=args.http_timeout,
    )
    results = request_json(
        "GET",
        args.coordinator_url,
        admin_results_url(args.miner_id, args.admin_results_limit),
        admin_token=args.admin_token,
        timeout=args.http_timeout,
    )
    return {
        "health": health,
        "ready": ready,
        "state": state,
        "results": results,
        "summary": readiness_summary(
            state=state,
            results=results,
            miner_id=args.miner_id,
            request_count=args.request_count,
        ),
    }


def wait_for_remote_result(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.timeout_seconds
    attempts = 0
    last_status: dict[str, Any] = {}
    errors: list[str] = []
    while time.monotonic() <= deadline:
        attempts += 1
        try:
            last_status = collect_status(args)
            if last_status.get("summary", {}).get("ready") is True:
                return {
                    "ok": True,
                    "attempts": attempts,
                    "elapsed_seconds": round(args.timeout_seconds - max(0.0, deadline - time.monotonic()), 3),
                    "status": last_status,
                    "errors": errors[-5:],
                }
        except Exception as exc:
            errors.append(str(exc))
        time.sleep(args.poll_interval)
    return {
        "ok": False,
        "attempts": attempts,
        "elapsed_seconds": args.timeout_seconds,
        "status": last_status,
        "errors": errors[-5:],
    }


def run_json_command(command: list[str], *, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "returncode": completed.returncode, "error": "command emitted no JSON"}
    payload = json.loads(lines[-1])
    return {"ok": bool(payload.get("ok", True)), "payload": payload}


def collect_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "remote_compute_evidence.json"
    evidence_md = output_dir / "remote_compute_evidence.md"
    support_json = output_dir / "support_bundle.json"
    support_md = output_dir / "support_bundle.md"
    evidence_result = run_json_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "remote_compute_evidence_pack.py"),
            "--mode",
            "collect",
            "--coordinator-url",
            args.coordinator_url,
            "--miner-id",
            args.miner_id,
            "--request-count",
            str(args.request_count),
            "--observer-token",
            args.observer_token,
            "--admin-token",
            args.admin_token,
            "--json-out",
            str(evidence_json),
            "--markdown-out",
            str(evidence_md),
        ],
        timeout=args.artifact_timeout,
    )
    support_result = run_json_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "support_bundle.py"),
            "--coordinator",
            args.coordinator_url,
            "--observer-token",
            args.observer_token,
            "--admin-token",
            args.admin_token,
            "--remote-report",
            str(evidence_json),
            "--json-out",
            str(support_json),
            "--markdown-out",
            str(support_md),
        ],
        timeout=args.artifact_timeout,
    )
    return {
        "evidence": {
            "ok": bool(evidence_result.get("ok")),
            "path": str(evidence_json),
            "markdown_path": str(evidence_md),
            "summary": summarize_evidence(evidence_result.get("payload") or {}),
            "error": evidence_result if not evidence_result.get("ok") else None,
        },
        "support_bundle": {
            "ok": bool(support_result.get("ok")),
            "path": str(support_json),
            "markdown_path": str(support_md),
            "summary": summarize_support_bundle(support_result.get("payload") or {}),
            "error": support_result if not support_result.get("ok") else None,
        },
    }


def summarize_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route_decision") or {}
    summary = payload.get("inference_summary") or {}
    safety = payload.get("safety") or {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "route": route.get("name"),
        "route_confidence": route.get("confidence"),
        "request_count": summary.get("request_count"),
        "requests_per_second": summary.get("requests_per_second"),
        "read_only": safety.get("read_only"),
        "redaction_ok": safety.get("redaction_ok"),
    }


def summarize_support_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    online = payload.get("online") or {}
    return {
        "generated_at": payload.get("generated_at"),
        "project": payload.get("project", {}).get("name"),
        "online_enabled": online.get("enabled"),
        "health_ok": (online.get("health") or {}).get("ok"),
        "state_ok": (online.get("state") or {}).get("ok"),
        "admin_results_ok": (online.get("admin_results") or {}).get("ok"),
    }


def failure_digest(wait: dict[str, Any]) -> dict[str, Any]:
    status = wait.get("status") or {}
    summary = status.get("summary") or {}
    health = status.get("health") or {}
    ready = status.get("ready") or {}
    return {
        "health_ok": health.get("ok"),
        "ready_ok": ready.get("ok"),
        "summary": summary,
        "errors": wait.get("errors", []),
    }


def build_report(
    *,
    args: argparse.Namespace,
    wait: dict[str, Any],
    artifacts: dict[str, Any] | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    ok = bool(
        wait.get("ok")
        and artifacts
        and artifacts.get("evidence", {}).get("ok")
        and artifacts.get("support_bundle", {}).get("ok")
    )
    evidence_summary = (artifacts or {}).get("evidence", {}).get("summary", {})
    support_summary = (artifacts or {}).get("support_bundle", {}).get("summary", {})
    report = {
        "schema": ACCEPTANCE_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "ok": ok,
        "coordinator_url": args.coordinator_url,
        "miner_id": args.miner_id,
        "workload_type": WORKLOAD_TYPE,
        "route": ROUTE_NAME,
        "wait_summary": {
            "ok": wait.get("ok"),
            "attempts": wait.get("attempts"),
            "elapsed_seconds": wait.get("elapsed_seconds"),
            "status": (wait.get("status") or {}).get("summary", {}),
            "failure": None if wait.get("ok") else failure_digest(wait),
        },
        "evidence_summary": evidence_summary,
        "support_bundle_summary": support_summary,
        "artifacts": {
            "evidence_json": (artifacts or {}).get("evidence", {}).get("path"),
            "evidence_markdown": (artifacts or {}).get("evidence", {}).get("markdown_path"),
            "support_bundle_json": (artifacts or {}).get("support_bundle", {}).get("path"),
            "support_bundle_markdown": (artifacts or {}).get("support_bundle", {}).get("markdown_path"),
        },
        "safety": {
            "redacted": True,
            "raw_tokens_in_report": False,
            "raw_state_dump_in_report": False,
            "requires_tls_or_vpn": True,
        },
        "recommended_next_commands": [
            "python3 scripts/remote_demo_runbook_pack.py --coordinator-url https://YOUR_COORDINATOR_HOST",
            "python3 scripts/remote_compute_evidence_pack.py --mode collect --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id REMOTE_MINER",
            "python3 scripts/support_bundle.py --coordinator https://YOUR_COORDINATOR_HOST --json-out /tmp/crowdtensor_support_bundle.json",
        ],
        "limitations": [
            "Validates a controlled two-machine demo; does not start production services",
            "CPU-only model_bundle_infer acceptance; not GPU pooling or production LLM serving",
            "No P2P discovery, NAT traversal, decentralized identity, or incentives are claimed",
        ],
    }
    return support_bundle.sanitize(report)


def render_markdown(payload: dict[str, Any]) -> str:
    wait = payload.get("wait_summary") or {}
    evidence = payload.get("evidence_summary") or {}
    support = payload.get("support_bundle_summary") or {}
    artifacts = payload.get("artifacts") or {}
    lines = [
        "# CrowdTensor Remote Demo Acceptance",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Coordinator: `{payload.get('coordinator_url', '')}`",
        f"Miner: `{payload.get('miner_id', '')}`",
        f"Route: `{payload.get('route', '')}`",
        "",
        "## Wait Summary",
        "",
        f"- Ready: `{wait.get('ok')}`",
        f"- Attempts: `{wait.get('attempts')}`",
        f"- Elapsed seconds: `{wait.get('elapsed_seconds')}`",
        "",
        "## Evidence",
        "",
        f"- Schema: `{evidence.get('schema')}`",
        f"- OK: `{evidence.get('ok')}`",
        f"- Request count: `{evidence.get('request_count')}`",
        f"- Read-only: `{evidence.get('read_only')}`",
        f"- Redaction OK: `{evidence.get('redaction_ok')}`",
        "",
        "## Support Bundle",
        "",
        f"- Online enabled: `{support.get('online_enabled')}`",
        f"- Health OK: `{support.get('health_ok')}`",
        f"- State OK: `{support.get('state_ok')}`",
        f"- Admin results OK: `{support.get('admin_results_ok')}`",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    if wait.get("failure"):
        lines.extend(["", "## Failure", "", f"```json\n{json.dumps(wait.get('failure'), indent=2, sort_keys=True)}\n```"])
    lines.extend(["", "## Limitations", ""])
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


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    wait = wait_for_remote_result(args)
    artifacts = collect_artifacts(args) if wait.get("ok") else None
    return build_report(args=args, wait=wait, artifacts=artifacts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a running CrowdTensorD two-machine remote demo.")
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--miner-id", required=True)
    parser.add_argument("--observer-token", required=True)
    parser.add_argument("--admin-token", required=True)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--artifact-timeout", type=float, default=60.0)
    parser.add_argument("--admin-results-limit", type=int, default=10)
    parser.add_argument("--output-dir", default="dist/remote-demo-acceptance")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.timeout_seconds < 0:
        raise SystemExit("--timeout-seconds must be non-negative")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")
    args.coordinator_url = args.coordinator_url.rstrip("/")
    output_dir = Path(args.output_dir)
    if not args.json_out:
        args.json_out = str(output_dir / "remote_demo_acceptance.json")
    if not args.markdown_out:
        args.markdown_out = str(output_dir / "remote_demo_acceptance.md")
    return args


def main() -> None:
    try:
        args = parse_args()
        payload = run_acceptance(args)
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
