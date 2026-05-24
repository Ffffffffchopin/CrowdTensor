#!/usr/bin/env python3
"""Build safe remote evidence for an operator-owned external_llm_infer Miner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
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

from create_miner_invite import create_invite  # noqa: E402
import support_bundle  # noqa: E402


EVIDENCE_SCHEMA = "remote_external_llm_evidence_v1"
OBSERVABILITY_SCHEMA = "remote_external_llm_observability_v1"
WORKLOAD_TYPE = "external_llm_infer"
ROUTE_NAME = "remote_python_external_llm_infer"
DEFAULT_MINER_ID = "remote-external-llm-miner"
DEFAULT_OBSERVER_TOKEN = "remote-external-llm-observer"
DEFAULT_ADMIN_TOKEN = "remote-external-llm-admin"
DEFAULT_INVITE_TOKEN = "remote-external-llm-token"
SECRET_FRAGMENTS = (
    '"external_llm_result"',
    '"external_llm_results"',
    '"output_text"',
    "CROWDTENSOR_LLM_RUNTIME_API_KEY",
    "CROWDTENSOR_MINER_TOKEN",
    '"lease_token": "',
    '"idempotency_key"',
    "Bearer ",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    observer_token: str = "",
    admin_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
        headers.setdefault("content-type", "application/json")
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def wait_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"coordinator exited early with code {proc.returncode}")
        try:
            if request_json("GET", base_url, "/health", timeout=2.0).get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"coordinator did not become healthy: {last_error}")


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def start_coordinator(args: argparse.Namespace, state_dir: Path, registry_path: Path) -> subprocess.Popen:
    command = [
        sys.executable,
        str(ROOT / "coordinator.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--state-dir",
        str(state_dir),
        "--lease-seconds",
        str(args.lease_seconds),
        "--inner-steps",
        str(args.request_count),
        "--backlog",
        "0",
        "--task-lane",
        f"python-cli:cpu:0:{WORKLOAD_TYPE}",
        "--miner-token-registry",
        str(registry_path),
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_health(args.base_url, proc, args.startup_timeout)
    return proc


def parse_miner_summary(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "accepted_tasks" in payload:
            return payload
    raise RuntimeError(f"missing miner summary JSON in stdout:\n{stdout}")


def adapter_args(args: argparse.Namespace) -> list[str]:
    if args.mock:
        return ["--enable-mock-llm-runtime"]
    if args.llm_runtime_cmd and args.llm_runtime_url:
        raise RuntimeError("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
    if args.llm_runtime_cmd:
        return ["--llm-runtime-cmd", args.llm_runtime_cmd]
    if args.llm_runtime_url:
        values = ["--llm-runtime-url", args.llm_runtime_url]
        if args.llm_runtime_api_key:
            values.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
        return values
    return ["--enable-mock-llm-runtime"]


def adapter_kind(args: argparse.Namespace) -> str:
    if args.mock or (not args.llm_runtime_cmd and not args.llm_runtime_url):
        return "mock"
    if args.llm_runtime_cmd:
        return "command"
    return "http_openai_chat"


def run_invited_miner(args: argparse.Namespace, invite: dict[str, Any]) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_MINER_TOKEN"] = invite["env"]["CROWDTENSOR_MINER_TOKEN"]
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        invite["miner_id"],
        "--once",
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        "0.2",
        "--llm-runtime-model-id",
        args.llm_runtime_model_id,
        "--llm-runtime-timeout",
        str(args.llm_runtime_timeout),
        *adapter_args(args),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.miner_timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "invited external LLM miner failed\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    return parse_miner_summary(completed.stdout)


def create_external_llm_session(args: argparse.Namespace) -> dict[str, Any]:
    return request_json(
        "POST",
        args.base_url,
        "/admin/inference-sessions",
        payload={"request_count": args.request_count, "workload_type": WORKLOAD_TYPE},
        admin_token=args.admin_token,
    )


def admin_results(
    base_url: str,
    *,
    admin_token: str,
    miner_id: str,
    workload_type: str,
    limit: int = 10,
    task_id: str = "",
) -> dict[str, Any]:
    query_params = {
        "status": "accepted",
        "workload_type": workload_type,
        "limit": limit,
    }
    if task_id:
        query_params["task_id"] = task_id
    else:
        query_params["miner_id"] = miner_id
    return request_json("GET", base_url, f"/admin/results?{urlencode(query_params)}", admin_token=admin_token)


def completed_task_for(state: dict[str, Any], miner_id: str, *, task_id: str = "") -> dict[str, Any] | None:
    completed = [
        task for task in state.get("tasks", [])
        if (
            task.get("status") == "completed"
            and task.get("workload_type") == WORKLOAD_TYPE
            and (task.get("task_id") == task_id if task_id else task.get("miner_id") == miner_id)
        )
    ]
    return completed[-1] if completed else None


def latest_ledger_row(ledger: dict[str, Any]) -> dict[str, Any] | None:
    rows = ledger.get("results") if isinstance(ledger, dict) else []
    if not isinstance(rows, list) or not rows:
        return None
    return rows[0]


def safe_profile(profile: dict[str, Any]) -> dict[str, Any]:
    capabilities = profile.get("last_capabilities") or {}
    hardware = capabilities.get("hardware_profile") or {}
    llm_runtime = capabilities.get("external_llm_runtime") or {}
    return {
        "runtime": profile.get("runtime"),
        "backend": profile.get("backend"),
        "accepted": profile.get("accepted"),
        "rejected": profile.get("rejected"),
        "hardware_profile": {
            "os": hardware.get("os"),
            "platform": hardware.get("platform"),
            "machine": hardware.get("machine"),
            "cpu_count": hardware.get("cpu_count"),
            "python_version": hardware.get("python_version"),
        },
        "supported_workloads": list(capabilities.get("supported_workloads") or []),
        "external_llm_runtime": {
            "adapter_kind": llm_runtime.get("adapter_kind"),
            "model_id": llm_runtime.get("model_id"),
        },
        "last_runtime_status": profile.get("last_runtime_status") or {},
    }


def route_decision(*, profile: dict[str, Any], has_task: bool, validation_ok: bool) -> dict[str, Any]:
    capabilities = profile.get("last_capabilities") or {}
    supported = set(capabilities.get("supported_workloads") or [])
    llm_runtime = capabilities.get("external_llm_runtime") or {}
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
    if WORKLOAD_TYPE in supported:
        matched.append(f"workload:{WORKLOAD_TYPE}")
    else:
        missing.append(f"workload:{WORKLOAD_TYPE}")
    if llm_runtime.get("adapter_kind"):
        matched.append("external_llm_runtime")
    else:
        missing.append("external_llm_runtime")
    if has_task:
        matched.append("accepted_result")
    else:
        missing.append("accepted_result")
    if validation_ok:
        matched.append("validation:ok")
    else:
        missing.append("validation:ok")
    return {
        "name": ROUTE_NAME,
        "target": "remote_python_miner",
        "workload": WORKLOAD_TYPE,
        "status": "available" if not missing else "blocked",
        "usable_now": not missing,
        "confidence": "ready" if not missing else "blocked",
        "reason": (
            "remote Python Miner completed read-only external_llm_infer"
            if not missing
            else "remote Python Miner has not completed a compatible external_llm_infer task"
        ),
        "matched_capabilities": matched,
        "missing_capabilities": missing,
    }


def read_only_ok(state: dict[str, Any], row: dict[str, Any] | None) -> bool:
    bundle = (state.get("model") or {}).get("model_bundle") or {}
    return (
        bundle.get("version") == 0
        and bundle.get("optimizer_step") == 0
        and (state.get("model") or {}).get("global_step") == 0
        and state.get("model_updates") == 0
        and not (row or {}).get("model_updated")
        and not (row or {}).get("model_bundle_updated")
    )


def redaction_ok(payload: Any, *, blocked_values: list[str] | None = None) -> bool:
    encoded = json.dumps(payload, sort_keys=True)
    if any(fragment in encoded for fragment in SECRET_FRAGMENTS):
        return False
    for value in blocked_values or []:
        if value and value in encoded:
            return False
    return True


def build_inference_summary(
    *,
    task: dict[str, Any] | None,
    row: dict[str, Any] | None,
    request_count: int,
) -> dict[str, Any]:
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    metrics = (row or {}).get("session_metrics") or (task or {}).get("metrics") or {}
    return {
        "present": bool(task and row),
        "ok": validation.get("code") == "ok",
        "workload_type": WORKLOAD_TYPE if task or row else None,
        "request_count": validation.get("request_count"),
        "expected_request_count": request_count,
        "completion_count": validation.get("completion_count") or metrics.get("completion_count"),
        "output_chars": validation.get("output_chars") or metrics.get("output_chars"),
        "adapter_kind": validation.get("adapter_kind") or metrics.get("adapter_kind"),
        "model_id": validation.get("model_id") or metrics.get("model_id"),
        "elapsed_ms": metrics.get("elapsed_ms"),
        "requests_per_second": metrics.get("requests_per_second"),
    }


def build_observability_summary(
    *,
    mode: str,
    route: dict[str, Any],
    profile: dict[str, Any],
    state: dict[str, Any],
    ledger: dict[str, Any],
    summary: dict[str, Any],
    safety: dict[str, Any],
    miner_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": OBSERVABILITY_SCHEMA,
        "mode": mode,
        "route": {
            "name": route.get("name"),
            "confidence": route.get("confidence"),
            "usable_now": route.get("usable_now"),
            "matched_capabilities": list(route.get("matched_capabilities") or []),
            "missing_capabilities": list(route.get("missing_capabilities") or []),
        },
        "miner": {
            "runtime": profile.get("runtime"),
            "backend": profile.get("backend"),
            "accepted": profile.get("accepted"),
            "rejected": profile.get("rejected"),
            "summary_accepted_tasks": (miner_summary or {}).get("accepted_tasks"),
            "summary_rejected_tasks": (miner_summary or {}).get("rejected_tasks"),
        },
        "work_queue": {
            "task_counts": state.get("task_counts", {}),
            "accepted_results": state.get("accepted_results"),
            "rejected_results": state.get("rejected_results"),
            "ledger_rows": len(ledger.get("results") or []),
        },
        "inference": {
            "ok": summary.get("ok"),
            "request_count": summary.get("request_count"),
            "expected_request_count": summary.get("expected_request_count"),
            "completion_count": summary.get("completion_count"),
            "output_chars": summary.get("output_chars"),
            "adapter_kind": summary.get("adapter_kind"),
            "elapsed_ms": summary.get("elapsed_ms"),
            "requests_per_second": summary.get("requests_per_second"),
        },
        "safety": {
            "read_only": safety.get("read_only"),
            "redaction_ok": safety.get("redaction_ok"),
            "raw_payloads_exposed": safety.get("raw_payloads_exposed"),
        },
    }


def build_evidence(
    *,
    mode: str,
    base_url: str,
    miner_id: str,
    state: dict[str, Any],
    ledger: dict[str, Any],
    miner_summary: dict[str, Any] | None,
    request_count: int,
    adapter: str,
    llm_runtime_model_id: str,
    invite: dict[str, Any] | None = None,
    registry_path: Path | None = None,
    task_id: str = "",
    blocked_values: list[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    task = completed_task_for(state, miner_id, task_id=task_id)
    row = latest_ledger_row(ledger)
    profile = (state.get("miner_profiles") or {}).get(miner_id) or {}
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    summary = build_inference_summary(task=task, row=row, request_count=request_count)
    route = route_decision(profile=profile, has_task=bool(task and row), validation_ok=summary.get("ok") is True)
    registry_hashed = bool(invite and str(invite.get("token_hash", "")).startswith("sha256:"))
    if mode == "collect":
        registry_hashed = None
    read_only = read_only_ok(state, row)
    redacted = redaction_ok(
        {
            "state_tasks": state.get("tasks", []),
            "ledger": ledger,
            "profile": safe_profile(profile),
            "summary": summary,
        },
        blocked_values=blocked_values,
    )
    safety = {
        "read_only": read_only,
        "redaction_ok": redacted,
        "raw_payloads_exposed": not redacted,
        "registry_hashed": registry_hashed,
        "runtime_url_redacted": True,
        "api_credential_redacted": True,
    }
    observability = build_observability_summary(
        mode=mode,
        route=route,
        profile=profile,
        state=state,
        ledger=ledger,
        summary=summary,
        safety=safety,
        miner_summary=miner_summary,
    )
    request_ok = int(summary.get("request_count") or 0) == int(request_count)
    completion_ok = int(summary.get("completion_count") or 0) == int(request_count)
    output_ok = int(summary.get("output_chars") or 0) > 0
    ok = bool(route.get("usable_now") and request_ok and completion_ok and output_ok and read_only and redacted)
    codes: list[str] = []
    if ok:
        codes.append("remote_external_llm_ready")
    else:
        if not task or not row:
            codes.append("remote_external_llm_result_missing")
        if summary.get("ok") is not True and (task or row):
            codes.append("remote_external_llm_validation_failed")
        if not request_ok or not completion_ok:
            codes.append("remote_external_llm_request_count_mismatch")
        if not output_ok:
            codes.append("remote_external_llm_output_missing")
        if not read_only:
            codes.append("remote_external_llm_read_only_failed")
        if not redacted:
            codes.append("remote_external_llm_redaction_failed")
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "mode": mode,
        "ok": ok,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "coordinator": {
            "url": base_url,
            "task_counts": state.get("task_counts", {}),
            "accepted_results": state.get("accepted_results"),
            "rejected_results": state.get("rejected_results"),
        },
        "miner": {
            "miner_id": miner_id,
            "profile": safe_profile(profile),
            "summary": miner_summary or {},
        },
        "invite": {
            "present": bool(invite),
            "registry": str(registry_path) if registry_path else "",
            "token_hash_prefix": str((invite or {}).get("token_hash", "")).split(":", 1)[0] if invite else "",
            "registry_hashed": registry_hashed,
        },
        "workload": {
            "type": WORKLOAD_TYPE,
            "request_count": request_count,
            "task_id": (task or {}).get("task_id"),
            "expected_task_id": task_id,
        },
        "adapter": {
            "kind": summary.get("adapter_kind") or adapter,
            "model_id": summary.get("model_id") or llm_runtime_model_id,
            "operator_owned_runtime": (summary.get("adapter_kind") or adapter) != "mock",
        },
        "route_decision": route,
        "inference_summary": summary,
        "observability_summary": observability,
        "ledger_summary": {
            "rows": len(ledger.get("results") or []),
            "event_index": (row or {}).get("event_index"),
            "status": (row or {}).get("status"),
            "model_updated": bool((row or {}).get("model_updated")),
            "model_bundle_updated": bool((row or {}).get("model_bundle_updated")),
        },
        "safety": safety,
        "diagnosis_codes": sorted(set(codes)),
        "recommended_next_commands": [
            "crowdtensor remote-demo verify --workload external-llm --mock --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --json",
            "python3 scripts/remote_external_llm_evidence_pack.py --mode collect --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id REMOTE_MINER",
            "python3 scripts/support_bundle.py --coordinator https://YOUR_COORDINATOR_HOST --json-out /tmp/crowdtensor_support_bundle.json",
        ],
        "limitations": [
            "Controlled remote external_llm_infer evidence; not public arbitrary prompt serving",
            "Uses fixed claim-time prompts and operator-owned runtime adapters",
            "No production Swarm Inference, P2P/NAT traversal, GPU pooling, WebGPU shards, or incentives are claimed",
        ],
    }
    return support_bundle.sanitize(evidence)


def run_local_loopback(args: argparse.Namespace) -> dict[str, Any]:
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_remote_external_llm_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = state_dir / "miner_registry.json"
    coordinator = None
    try:
        invite = create_invite(
            registry_path=registry_path,
            miner_id=args.miner_id,
            coordinator_url=args.base_url,
            label="remote external LLM evidence",
            token=args.invite_token,
            replace=True,
        )
        coordinator = start_coordinator(args, state_dir, registry_path)
        session = create_external_llm_session(args)
        task_id = str(session.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"admin external LLM session did not return task_id: {session}")
        miner_summary = run_invited_miner(args, invite)
        state = request_json("GET", args.base_url, "/state", observer_token=args.observer_token)
        ledger = admin_results(
            args.base_url,
            admin_token=args.admin_token,
            miner_id=args.miner_id,
            workload_type=WORKLOAD_TYPE,
            task_id=task_id,
        )
        return build_evidence(
            mode="local-loopback",
            base_url=args.base_url,
            miner_id=args.miner_id,
            state=state,
            ledger=ledger,
            miner_summary=miner_summary,
            request_count=args.request_count,
            adapter=adapter_kind(args),
            llm_runtime_model_id=args.llm_runtime_model_id,
            invite=invite,
            registry_path=registry_path,
            task_id=task_id,
            blocked_values=[args.llm_runtime_url, args.llm_runtime_api_key],
        )
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    state = request_json("GET", args.coordinator_url, "/state", observer_token=args.observer_token)
    ledger = admin_results(
        args.coordinator_url,
        admin_token=args.admin_token,
        miner_id=args.miner_id,
        workload_type=WORKLOAD_TYPE,
        task_id=args.task_id,
    )
    return build_evidence(
        mode="collect",
        base_url=args.coordinator_url,
        miner_id=args.miner_id,
        state=state,
        ledger=ledger,
        miner_summary=None,
        request_count=args.request_count,
        adapter=adapter_kind(args),
        llm_runtime_model_id=args.llm_runtime_model_id,
        task_id=args.task_id,
        blocked_values=[args.llm_runtime_url, args.llm_runtime_api_key],
    )


def render_markdown(payload: dict[str, Any]) -> str:
    route = payload.get("route_decision") or {}
    summary = payload.get("inference_summary") or {}
    adapter = payload.get("adapter") or {}
    observability = payload.get("observability_summary") or {}
    observed_queue = observability.get("work_queue") or {}
    observed_safety = observability.get("safety") or {}
    safety = payload.get("safety") or {}
    lines = [
        "# CrowdTensor Remote External LLM Evidence",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Mode: `{payload.get('mode', '')}`",
        f"OK: `{payload.get('ok')}`",
        "",
        "## Route",
        "",
        f"- Route: `{route.get('name', '')}`",
        f"- Workload: `{route.get('workload', '')}`",
        f"- Confidence: `{route.get('confidence', '')}`",
        f"- Matched capabilities: `{', '.join(route.get('matched_capabilities') or [])}`",
        f"- Missing capabilities: `{', '.join(route.get('missing_capabilities') or [])}`",
        "",
        "## Adapter",
        "",
        f"- Kind: `{adapter.get('kind', '')}`",
        f"- Model ID: `{adapter.get('model_id', '')}`",
        f"- Operator-owned runtime: `{adapter.get('operator_owned_runtime')}`",
        "",
        "## Inference",
        "",
        f"- Requests: `{summary.get('request_count')}`",
        f"- Completions: `{summary.get('completion_count')}`",
        f"- Output chars: `{summary.get('output_chars')}`",
        f"- Requests/sec: `{summary.get('requests_per_second')}`",
        f"- Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Observability",
        "",
        f"- Schema: `{observability.get('schema')}`",
        f"- Accepted results: `{observed_queue.get('accepted_results')}`",
        f"- Rejected results: `{observed_queue.get('rejected_results')}`",
        f"- Ledger rows: `{observed_queue.get('ledger_rows')}`",
        f"- Read-only: `{observed_safety.get('read_only')}`",
        f"- Redaction OK: `{observed_safety.get('redaction_ok')}`",
        "",
        "## Safety",
        "",
        f"- Read-only: `{safety.get('read_only')}`",
        f"- Redaction OK: `{safety.get('redaction_ok')}`",
        f"- Runtime URL redacted: `{safety.get('runtime_url_redacted')}`",
        f"- API credential redacted: `{safety.get('api_credential_redacted')}`",
        f"- Raw payloads exposed: `{safety.get('raw_payloads_exposed')}`",
        "",
        "## Limitations",
        "",
    ]
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
    parser = argparse.ArgumentParser(description="Build a safe remote external_llm_infer evidence report.")
    parser.add_argument("--mode", choices=["local-loopback", "collect"], default="local-loopback")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8922)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--miner-id", default=DEFAULT_MINER_ID)
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--invite-token", default=DEFAULT_INVITE_TOKEN)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--mock", action="store_true", help="use deterministic mock external LLM runtime")
    parser.add_argument("--llm-runtime-cmd", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_CMD", ""))
    parser.add_argument("--llm-runtime-url", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_URL", ""))
    parser.add_argument("--llm-runtime-api-key", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_API_KEY", ""))
    parser.add_argument("--llm-runtime-model-id", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_MODEL_ID", "external-llm-runtime"))
    parser.add_argument("--llm-runtime-timeout", type=float, default=float(os.environ.get("CROWDTENSOR_LLM_RUNTIME_TIMEOUT", "30.0")))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.llm_runtime_cmd and args.llm_runtime_url:
        raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
    if args.llm_runtime_timeout <= 0:
        raise SystemExit("--llm-runtime-timeout must be positive")
    args.base_url = f"http://{args.host}:{args.port}"
    if args.mode == "collect" and not args.coordinator_url:
        raise SystemExit("--coordinator-url is required in collect mode")
    if args.mode == "collect":
        args.coordinator_url = args.coordinator_url.rstrip("/")
    return args


def main() -> None:
    try:
        args = parse_args()
        payload = run_collect(args) if args.mode == "collect" else run_local_loopback(args)
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
