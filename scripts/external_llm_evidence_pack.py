#!/usr/bin/env python3
"""Build a safe, shareable external LLM runtime evidence report."""

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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, observer_headers  # noqa: E402
import support_bundle  # noqa: E402


EVIDENCE_SCHEMA = "external_llm_evidence_v1"
WORKLOAD_TYPE = "external_llm_infer"
ADMIN_HEADER = "x-crowdtensor-admin-token"
DEFAULT_REPORT = "/tmp/crowdtensor_external_llm_evidence.json"
DEFAULT_MARKDOWN = "/tmp/crowdtensor_external_llm_evidence.md"
SUCCESS_CODE = "external_llm_evidence_ready"
SECRET_FRAGMENTS = {
    "external_llm_result",
    "external_llm_results",
    "output_text",
    "CROWDTENSOR_LLM_RUNTIME_API_KEY",
    "Bearer ",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    admin_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    if admin_token:
        headers["content-type"] = "application/json"
        headers[ADMIN_HEADER] = admin_token
    elif method == "GET":
        headers.update(observer_headers(json_content=payload is not None))
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def wait_health(base_url: str, proc: subprocess.Popen[str], timeout: float) -> None:
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


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def start_coordinator(args: argparse.Namespace, state_dir: Path) -> subprocess.Popen[str]:
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
        f"python-cli:cpu:1:{WORKLOAD_TYPE}",
    ]
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    env = coordinator_env()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_health(args.base_url, proc, args.startup_timeout)
    return proc


def resolve_adapter(args: argparse.Namespace) -> str:
    if args.mock:
        return "mock"
    if args.llm_runtime_cmd and args.llm_runtime_url:
        raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
    if args.llm_runtime_cmd:
        return "command"
    if args.llm_runtime_url:
        return "http_openai_chat"
    return "mock"


def run_miner(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        args.miner_id,
        "--once",
        "--llm-runtime-model-id",
        args.llm_runtime_model_id,
        "--llm-runtime-timeout",
        str(args.llm_runtime_timeout),
    ]
    if args.adapter_kind == "mock":
        command.append("--enable-mock-llm-runtime")
    elif args.adapter_kind == "command":
        command.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    elif args.adapter_kind == "http_openai_chat":
        command.extend(["--llm-runtime-url", args.llm_runtime_url])
        if args.llm_runtime_api_key:
            command.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    else:
        raise RuntimeError(f"unsupported external LLM adapter: {args.adapter_kind}")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        capture_output=True,
        timeout=args.miner_timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "miner_cli.py failed\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    for line in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("miner_cli.py emitted no JSON summary")


def admin_results(args: argparse.Namespace, **params: str | int) -> dict[str, Any]:
    query = urlencode({key: value for key, value in params.items() if value != ""})
    path = "/admin/results" + (f"?{query}" if query else "")
    return request_json("GET", args.base_url, path, admin_token=args.admin_token)


def latest_external_task(state: dict[str, Any]) -> dict[str, Any]:
    completed = [
        task for task in state.get("tasks", [])
        if isinstance(task, dict)
        and task.get("status") == "completed"
        and task.get("workload_type") == WORKLOAD_TYPE
    ]
    return completed[-1] if completed else {}


def latest_ledger_row(ledger: dict[str, Any]) -> dict[str, Any]:
    rows = ledger.get("results") if isinstance(ledger, dict) else []
    if isinstance(rows, list) and rows:
        return rows[0] if isinstance(rows[0], dict) else {}
    return {}


def redaction_ok(payload: Any, *, blocked_values: list[str] | None = None) -> bool:
    encoded = json.dumps(payload, sort_keys=True)
    if any(fragment in encoded for fragment in SECRET_FRAGMENTS):
        return False
    for value in blocked_values or []:
        if value and value in encoded:
            return False
    return True


def build_evidence(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    ledger: dict[str, Any],
    miner_summary: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    task = latest_external_task(state)
    row = latest_ledger_row(ledger)
    validation = (row.get("validation") or task.get("validation") or {}) if (row or task) else {}
    metrics = row.get("session_metrics") or task.get("metrics") or {}
    profile = (state.get("miner_profiles") or {}).get(args.miner_id) or {}
    capabilities = profile.get("last_capabilities") or {}
    runtime_capability = capabilities.get("external_llm_runtime") or {}
    read_only = bool(
        state.get("model_updates") == 0
        and (state.get("model") or {}).get("global_step") == 0
        and not row.get("model_updated")
        and not row.get("model_bundle_updated")
    )
    request_count = int(validation.get("request_count") or metrics.get("request_count") or 0)
    completion_count = int(validation.get("completion_count") or metrics.get("completion_count") or 0)
    output_chars = int(validation.get("output_chars") or metrics.get("output_chars") or 0)
    adapter_kind = str(validation.get("adapter_kind") or runtime_capability.get("adapter_kind") or args.adapter_kind)
    report = {
        "schema": EVIDENCE_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "ok": False,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "route": {
            "name": "local_external_llm_infer",
            "runtime": "python-cli",
            "backend": "cpu",
            "workload": WORKLOAD_TYPE,
        },
        "adapter": {
            "kind": adapter_kind,
            "model_id": validation.get("model_id") or runtime_capability.get("model_id") or args.llm_runtime_model_id,
            "operator_owned_runtime": adapter_kind != "mock",
        },
        "task": {
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "validation_code": validation.get("code"),
        },
        "summary": {
            "request_count": request_count,
            "completion_count": completion_count,
            "output_chars": output_chars,
            "elapsed_ms": metrics.get("elapsed_ms"),
            "requests_per_second": metrics.get("requests_per_second"),
        },
        "miner": {
            "miner_id": args.miner_id,
            "accepted_tasks": miner_summary.get("accepted_tasks"),
            "rejected_tasks": miner_summary.get("rejected_tasks"),
            "request_retries": miner_summary.get("request_retries"),
            "advertised_workload": WORKLOAD_TYPE in (capabilities.get("supported_workloads") or []),
            "adapter_kind": runtime_capability.get("adapter_kind"),
            "model_id": runtime_capability.get("model_id"),
        },
        "ledger": {
            "accepted_rows": len(ledger.get("results") or []),
            "row_status": row.get("status"),
            "model_updated": bool(row.get("model_updated")),
            "model_bundle_updated": bool(row.get("model_bundle_updated")),
        },
        "safety": {
            "read_only": read_only,
            "redaction_ok": False,
            "raw_payloads_exposed": True,
            "runtime_url_redacted": True,
            "api_credential_redacted": True,
        },
        "diagnosis_codes": [],
        "limitations": [
            "Local external_llm_infer evidence; not production LLM serving",
            "Uses fixed claim-time prompts; not an arbitrary public prompt API",
            "No GPU pooling, WebGPU model shards, P2P routing, or token incentives are claimed",
        ],
        "recommended_next_commands": [
            "crowdtensor llm-infer --mock --json",
            "python3 scripts/external_llm_evidence_check.py --port 8919",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
        ],
    }
    safe = redaction_ok(report, blocked_values=[args.llm_runtime_url, args.llm_runtime_api_key])
    report["safety"]["redaction_ok"] = safe
    report["safety"]["raw_payloads_exposed"] = not safe
    ok = bool(
        task
        and row
        and validation.get("code") == "ok"
        and request_count == args.request_count
        and completion_count == args.request_count
        and output_chars > 0
        and read_only
        and safe
    )
    report["ok"] = ok
    codes: list[str] = []
    if ok:
        codes.append(SUCCESS_CODE)
    else:
        if not task or not row:
            codes.append("external_llm_result_missing")
        if validation.get("code") not in {"ok", None}:
            codes.append("external_llm_validation_failed")
        if request_count != args.request_count or completion_count != args.request_count:
            codes.append("external_llm_request_count_mismatch")
        if output_chars <= 0:
            codes.append("external_llm_output_missing")
        if not read_only:
            codes.append("external_llm_read_only_failed")
        if not safe:
            codes.append("external_llm_redaction_failed")
    report["diagnosis_codes"] = sorted(set(codes))
    return support_bundle.sanitize(report)


def run_loopback(args: argparse.Namespace) -> dict[str, Any]:
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_external_llm_evidence_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    coordinator: subprocess.Popen[str] | None = None
    try:
        coordinator = start_coordinator(args, state_dir)
        miner_summary = run_miner(args)
        state = request_json("GET", args.base_url, "/state")
        ledger = admin_results(args, status="accepted", workload_type=WORKLOAD_TYPE)
        return build_evidence(args=args, state=state, ledger=ledger, miner_summary=miner_summary)
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


def render_markdown(payload: dict[str, Any]) -> str:
    adapter = payload.get("adapter") or {}
    summary = payload.get("summary") or {}
    safety = payload.get("safety") or {}
    lines = [
        "# CrowdTensor External LLM Evidence",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        "",
        "## Runtime",
        "",
        f"- Adapter kind: `{adapter.get('kind', '')}`",
        f"- Model ID: `{adapter.get('model_id', '')}`",
        f"- Workload: `{(payload.get('route') or {}).get('workload', '')}`",
        "",
        "## Summary",
        "",
        f"- Requests: `{summary.get('request_count')}`",
        f"- Completions: `{summary.get('completion_count')}`",
        f"- Output chars: `{summary.get('output_chars')}`",
        f"- Requests/sec: `{summary.get('requests_per_second')}`",
        f"- Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Safety",
        "",
        f"- Read-only: `{safety.get('read_only')}`",
        f"- Redaction OK: `{safety.get('redaction_ok')}`",
        f"- Runtime URL redacted: `{safety.get('runtime_url_redacted')}`",
        f"- API credential redacted: `{safety.get('api_credential_redacted')}`",
        "",
        "## Limitations",
        "",
    ]
    for item in payload.get("limitations") or []:
        lines.append(f"- {item}")
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
    parser = argparse.ArgumentParser(description="Build a safe external_llm_infer evidence report.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8919)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--miner-id", default="external-llm-evidence-miner")
    parser.add_argument("--admin-token", default="local-admin")
    parser.add_argument("--mock", action="store_true", help="force deterministic mock external LLM runtime")
    parser.add_argument("--llm-runtime-cmd", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_CMD", ""))
    parser.add_argument("--llm-runtime-url", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_URL", ""))
    parser.add_argument("--llm-runtime-api-key", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_API_KEY", ""))
    parser.add_argument("--llm-runtime-model-id", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_MODEL_ID", "external-llm-runtime"))
    parser.add_argument("--llm-runtime-timeout", type=float, default=float(os.environ.get("CROWDTENSOR_LLM_RUNTIME_TIMEOUT", "30.0")))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args(argv)
    activate_miner_token(args)
    activate_observer_token(args)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.llm_runtime_timeout <= 0:
        raise SystemExit("--llm-runtime-timeout must be positive")
    args.adapter_kind = resolve_adapter(args)
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    try:
        args = parse_args()
        payload = run_loopback(args)
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
