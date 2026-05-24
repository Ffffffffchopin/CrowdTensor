#!/usr/bin/env python3
"""High-level two-machine home-compute remote Miner demo wrapper."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
from crowdtensor.auth import hash_token  # noqa: E402
from create_miner_invite import create_invite  # noqa: E402


SCHEMA = "remote_home_compute_demo_v1"
MODEL_BUNDLE_WORKLOAD_TYPE = "model_bundle_infer"
MODEL_BUNDLE_ROUTE_NAME = "remote_python_model_bundle_infer"
EXTERNAL_LLM_WORKLOAD_TYPE = "external_llm_infer"
EXTERNAL_LLM_ROUTE_NAME = "remote_python_external_llm_infer"
MODEL_BUNDLE_KIND = "model-bundle"
EXTERNAL_LLM_KIND = "external-llm"
EXTERNAL_LLM_RUNBOOK_SCHEMA = "remote_external_llm_runbook_v1"
EXTERNAL_LLM_ACCEPTANCE_SCHEMA = "remote_external_llm_acceptance_v1"
EXTERNAL_LLM_OBSERVABILITY_SCHEMA = "remote_external_llm_observability_v1"
DOCTOR_SCHEMA = "remote_home_compute_doctor_v1"
COLLECT_SCHEMA = "remote_home_compute_collect_v1"
CLEANUP_SCHEMA = "remote_home_compute_cleanup_v1"
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "CROWDTENSOR_LLM_RUNTIME_API_KEY",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_result",
    "external_llm_results",
    "output_text",
    "Bearer ",
)
REMOTE_DEMO_GENERATED_FILENAMES = (
    "remote_home_compute_demo.json",
    "remote_home_compute_demo.md",
    "remote_home_compute_doctor.json",
    "remote_home_compute_doctor.md",
    "remote_home_compute_collect.json",
    "remote_home_compute_collect.md",
    "remote_home_compute_cleanup.json",
    "remote_home_compute_cleanup.md",
    "remote_demo_acceptance.json",
    "remote_demo_acceptance.md",
    "remote_compute_evidence.json",
    "remote_compute_evidence.md",
    "remote_external_llm_acceptance.json",
    "remote_external_llm_acceptance.md",
    "remote_external_llm_evidence.json",
    "remote_external_llm_evidence.md",
    "support_bundle.json",
    "support_bundle.md",
)
REMOTE_DEMO_PRIVATE_FILENAMES = (
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def workload_type_for(kind: str) -> str:
    return EXTERNAL_LLM_WORKLOAD_TYPE if kind == EXTERNAL_LLM_KIND else MODEL_BUNDLE_WORKLOAD_TYPE


def route_name_for(kind: str) -> str:
    return EXTERNAL_LLM_ROUTE_NAME if kind == EXTERNAL_LLM_KIND else MODEL_BUNDLE_ROUTE_NAME


def scenario_schema_for(kind: str) -> str:
    return "external_llm_fixed_prompt_session_v1" if kind == EXTERNAL_LLM_KIND else "model_bundle_inference_scenario_v1"


def quote_env(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_private_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"export {key}={quote_env(value)}" for key, value in sorted(values.items()) if value]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def token_or_generated(value: str) -> str:
    return value or secrets.token_urlsafe(32)


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


def parse_private_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    import shlex

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str, secret_values: list[str] | None = None) -> str:
    redacted = text
    for value in secret_values or []:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    return value


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    secret_values: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
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
    if payload:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:], secret_values)
    return step, payload


def request_json_observed(
    endpoint: str,
    method: str,
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    observer_token: str = "",
    admin_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            payload = json.loads(raw) if raw else {}
            return {
                "endpoint": endpoint,
                "path": path,
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "json": payload,
            }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "endpoint": endpoint,
            "path": path,
            "ok": False,
            "status": exc.code,
            "error": "http_error",
            "detail": body[:300],
        }
    except json.JSONDecodeError as exc:
        return {
            "endpoint": endpoint,
            "path": path,
            "ok": False,
            "status": None,
            "error": "invalid_json",
            "detail": str(exc)[:300],
        }
    except (OSError, URLError) as exc:
        return {
            "endpoint": endpoint,
            "path": path,
            "ok": False,
            "status": None,
            "error": type(exc).__name__,
            "detail": str(exc)[:300],
        }


def safe_observation(response: dict[str, Any]) -> dict[str, Any]:
    return {
        key: response.get(key)
        for key in ["endpoint", "path", "ok", "status", "error", "detail"]
        if response.get(key) is not None
    }


def response_payload(response: dict[str, Any]) -> dict[str, Any]:
    payload = response.get("json")
    return payload if isinstance(payload, dict) else {}


def admin_results_url(*, workload_type: str, miner_id: str, limit: int = 10, task_id: str = "") -> str:
    query_params = {"status": "accepted", "workload_type": workload_type, "limit": limit}
    if task_id:
        query_params["task_id"] = task_id
    else:
        query_params["miner_id"] = miner_id
    return f"/admin/results?{urlencode(query_params)}"


def completed_task_for(state: dict[str, Any], miner_id: str, *, workload_type: str, task_id: str = "") -> dict[str, Any] | None:
    completed = [
        task for task in state.get("tasks", [])
        if (
            isinstance(task, dict)
            and task.get("status") == "completed"
            and task.get("workload_type") == workload_type
            and (task.get("task_id") == task_id if task_id else task.get("miner_id") == miner_id)
        )
    ]
    return completed[-1] if completed else None


def latest_result(results: dict[str, Any]) -> dict[str, Any] | None:
    rows = results.get("results") if isinstance(results, dict) else []
    if not isinstance(rows, list) or not rows:
        return None
    return rows[0] if isinstance(rows[0], dict) else None


def miner_profile(state: dict[str, Any], miner_id: str) -> dict[str, Any]:
    return (state.get("miner_profiles") or {}).get(miner_id) or {}


def model_bundle_status_summary(
    *,
    state: dict[str, Any],
    results: dict[str, Any],
    miner_id: str,
    request_count: int,
    scenario_id: str,
    task_id: str = "",
) -> dict[str, Any]:
    profile = miner_profile(state, miner_id)
    capabilities = profile.get("last_capabilities") or {}
    workloads = capabilities.get("supported_workloads") or []
    task = completed_task_for(state, miner_id, workload_type=MODEL_BUNDLE_WORKLOAD_TYPE, task_id=task_id)
    row = latest_result(results)
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    metrics = (row or {}).get("session_metrics") or (task or {}).get("metrics") or {}
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
    if MODEL_BUNDLE_WORKLOAD_TYPE in workloads:
        matched.append(f"workload:{MODEL_BUNDLE_WORKLOAD_TYPE}")
    else:
        missing.append(f"workload:{MODEL_BUNDLE_WORKLOAD_TYPE}")
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
    actual_scenario_id = str(validation.get("scenario_id") or "")
    if actual_scenario_id == scenario_id:
        matched.append("scenario_id")
    else:
        missing.append("scenario_id")
    return {
        "ready": not missing,
        "route": MODEL_BUNDLE_ROUTE_NAME,
        "miner_id": miner_id,
        "expected_task_id": task_id,
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
            "scenario_id": actual_scenario_id,
            "accuracy": validation.get("accuracy"),
            "request_trace_count": validation.get("request_trace_count"),
            "requests_per_second": metrics.get("requests_per_second"),
        },
    }


def external_llm_summary(
    *,
    state: dict[str, Any],
    results: dict[str, Any],
    miner_id: str,
    request_count: int,
    task_id: str = "",
) -> dict[str, Any]:
    profile = miner_profile(state, miner_id)
    capabilities = profile.get("last_capabilities") or {}
    workloads = capabilities.get("supported_workloads") or []
    llm_runtime = capabilities.get("external_llm_runtime") or {}
    task = completed_task_for(state, miner_id, workload_type=EXTERNAL_LLM_WORKLOAD_TYPE, task_id=task_id)
    row = latest_result(results)
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    metrics = (row or {}).get("session_metrics") or (task or {}).get("metrics") or {}
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
    if EXTERNAL_LLM_WORKLOAD_TYPE in workloads:
        matched.append(f"workload:{EXTERNAL_LLM_WORKLOAD_TYPE}")
    else:
        missing.append(f"workload:{EXTERNAL_LLM_WORKLOAD_TYPE}")
    if llm_runtime.get("adapter_kind"):
        matched.append("external_llm_runtime")
    else:
        missing.append("external_llm_runtime")
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
    if int(validation.get("completion_count") or metrics.get("completion_count") or 0) == int(request_count):
        matched.append("completion_count")
    else:
        missing.append("completion_count")
    return {
        "ready": not missing,
        "route": EXTERNAL_LLM_ROUTE_NAME,
        "miner_id": miner_id,
        "expected_task_id": task_id,
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
            "external_llm_runtime": {
                "adapter_kind": llm_runtime.get("adapter_kind"),
                "model_id": llm_runtime.get("model_id"),
            },
        },
        "inference": {
            "ok": validation.get("code") == "ok",
            "request_count": validation.get("request_count"),
            "completion_count": validation.get("completion_count") or metrics.get("completion_count"),
            "output_chars": validation.get("output_chars") or metrics.get("output_chars"),
            "adapter_kind": validation.get("adapter_kind") or metrics.get("adapter_kind") or llm_runtime.get("adapter_kind"),
            "model_id": validation.get("model_id") or metrics.get("model_id") or llm_runtime.get("model_id"),
            "requests_per_second": metrics.get("requests_per_second"),
        },
    }


def collect_external_llm_status(args: argparse.Namespace) -> dict[str, Any]:
    task_id = getattr(args, "session_task_id", "") or getattr(args, "task_id", "") or ""
    health_response = request_json_observed("health", "GET", args.coordinator_url, "/health", timeout=args.http_timeout)
    ready_response = request_json_observed("ready", "GET", args.coordinator_url, "/ready", timeout=args.http_timeout)
    state_response = request_json_observed(
        "state",
        "GET",
        args.coordinator_url,
        "/state",
        observer_token=args.observer_token,
        timeout=args.http_timeout,
    )
    results_response = request_json_observed(
        "admin_results",
        "GET",
        args.coordinator_url,
        admin_results_url(
            workload_type=EXTERNAL_LLM_WORKLOAD_TYPE,
            miner_id=args.miner_id,
            limit=args.admin_results_limit,
            task_id=task_id,
        ),
        admin_token=args.admin_token,
        timeout=args.http_timeout,
    )
    state = response_payload(state_response)
    results = response_payload(results_response)
    return {
        "health": response_payload(health_response) if health_response.get("ok") else {"ok": False},
        "ready": response_payload(ready_response) if ready_response.get("ok") else {"ok": False},
        "state": state,
        "results": results,
        "observations": {
            "health": {key: health_response.get(key) for key in ["endpoint", "path", "ok", "status", "error", "detail"] if health_response.get(key) is not None},
            "ready": {key: ready_response.get(key) for key in ["endpoint", "path", "ok", "status", "error", "detail"] if ready_response.get(key) is not None},
            "state": {key: state_response.get(key) for key in ["endpoint", "path", "ok", "status", "error", "detail"] if state_response.get(key) is not None},
            "admin_results": {key: results_response.get(key) for key in ["endpoint", "path", "ok", "status", "error", "detail"] if results_response.get(key) is not None},
        },
        "summary": external_llm_summary(
            state=state,
            results=results,
            miner_id=args.miner_id,
            request_count=args.request_count,
            task_id=task_id,
        ),
    }


def collect_model_bundle_status(args: argparse.Namespace) -> dict[str, Any]:
    task_id = getattr(args, "session_task_id", "") or getattr(args, "task_id", "") or ""
    health_response = request_json_observed("health", "GET", args.coordinator_url, "/health", timeout=args.http_timeout)
    ready_response = request_json_observed("ready", "GET", args.coordinator_url, "/ready", timeout=args.http_timeout)
    state_response = request_json_observed(
        "state",
        "GET",
        args.coordinator_url,
        "/state",
        observer_token=args.observer_token,
        timeout=args.http_timeout,
    )
    results_response = request_json_observed(
        "admin_results",
        "GET",
        args.coordinator_url,
        admin_results_url(
            workload_type=MODEL_BUNDLE_WORKLOAD_TYPE,
            miner_id=args.miner_id,
            limit=args.admin_results_limit,
            task_id=task_id,
        ),
        admin_token=args.admin_token,
        timeout=args.http_timeout,
    )
    state = response_payload(state_response)
    results = response_payload(results_response)
    return {
        "health": response_payload(health_response) if health_response.get("ok") else {"ok": False},
        "ready": response_payload(ready_response) if ready_response.get("ok") else {"ok": False},
        "state": state,
        "results": results,
        "observations": {
            "health": safe_observation(health_response),
            "ready": safe_observation(ready_response),
            "state": safe_observation(state_response),
            "admin_results": safe_observation(results_response),
        },
        "summary": model_bundle_status_summary(
            state=state,
            results=results,
            miner_id=args.miner_id,
            request_count=args.request_count,
            scenario_id=args.scenario_id,
            task_id=task_id,
        ),
    }


def create_external_llm_session(args: argparse.Namespace) -> dict[str, Any]:
    response = request_json_observed(
        "admin_inference_session",
        "POST",
        args.coordinator_url,
        "/admin/inference-sessions",
        payload={"request_count": args.request_count, "workload_type": EXTERNAL_LLM_WORKLOAD_TYPE},
        admin_token=args.admin_token,
        timeout=args.http_timeout,
    )
    if not response.get("ok"):
        return {"ok": False, "observation": {key: response.get(key) for key in ["endpoint", "path", "ok", "status", "error", "detail"] if response.get(key) is not None}}
    session = response_payload(response)
    if (
        session.get("schema") != "inference_session_request_v1"
        or session.get("workload_type") != EXTERNAL_LLM_WORKLOAD_TYPE
        or not session.get("task_id")
        or int(session.get("request_count") or 0) != int(args.request_count)
    ):
        return {
            "ok": False,
            "observation": {key: response.get(key) for key in ["endpoint", "path", "ok", "status", "error", "detail"] if response.get(key) is not None},
            "session": support_bundle.sanitize(session),
            "error": "invalid_session_response",
        }
    return {
        "ok": True,
        "observation": {key: response.get(key) for key in ["endpoint", "path", "ok", "status"] if response.get(key) is not None},
        "session": {
            "created": True,
            "schema": session.get("schema"),
            "task_id": session.get("task_id"),
            "request_count": session.get("request_count"),
            "workload_type": session.get("workload_type"),
            "status": session.get("status"),
            "result_query": session.get("result_query"),
        },
    }


def collect_status_for_workload(args: argparse.Namespace) -> dict[str, Any]:
    if args.workload == EXTERNAL_LLM_KIND:
        return collect_external_llm_status(args)
    return collect_model_bundle_status(args)


def remote_demo_doctor_summary(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    operator_env_path = output_dir / "operator.private.env"
    miner_env_path = output_dir / "miner.private.env"
    registry_path = output_dir / "miner_registry.json"
    runbook_path = output_dir / ("remote_external_llm_runbook.json" if args.workload == EXTERNAL_LLM_KIND else "remote_demo_runbook.json")
    operator_env = parse_private_env(operator_env_path)
    miner_env = parse_private_env(miner_env_path)
    runbook = load_json(runbook_path)
    status = collect_status_for_workload(args) if args.coordinator_url else {}
    observations = status.get("observations") or {}
    ready_payload = status.get("ready") if isinstance(status.get("ready"), dict) else {}
    configured_lanes = ready_payload.get("task_lanes") if isinstance(ready_payload, dict) else []
    lane_visible = any(
        isinstance(lane, dict) and lane.get("workload_type") == workload_type_for(args.workload)
        for lane in configured_lanes or []
    )
    env_checks = {
        "operator_env_present": operator_env_path.is_file(),
        "miner_env_present": miner_env_path.is_file(),
        "registry_present": registry_path.is_file(),
        "runbook_present": runbook_path.is_file(),
        "observer_token_present": bool(operator_env.get("CROWDTENSOR_OBSERVER_TOKEN") or args.observer_token),
        "admin_token_present": bool(operator_env.get("CROWDTENSOR_ADMIN_TOKEN") or args.admin_token),
        "miner_token_present": bool(miner_env.get("CROWDTENSOR_MINER_TOKEN")),
        "registry_hashed": False,
    }
    registry = load_json(registry_path)
    miners = registry.get("miners") if isinstance(registry.get("miners"), list) else []
    env_checks["registry_hashed"] = any(
        isinstance(entry, dict)
        and entry.get("miner_id") == args.miner_id
        and str(entry.get("token", "")).startswith("sha256:")
        for entry in miners
    )
    diagnosis: list[str] = []
    if not env_checks["operator_env_present"]:
        diagnosis.append("operator_env_missing")
    if not env_checks["miner_env_present"]:
        diagnosis.append("miner_env_missing")
    if not env_checks["registry_present"]:
        diagnosis.append("registry_missing")
    if not env_checks["registry_hashed"]:
        diagnosis.append("registry_hash_missing")
    if not env_checks["runbook_present"]:
        diagnosis.append("runbook_missing")
    if not env_checks["observer_token_present"]:
        diagnosis.append("observer_token_missing")
    if not env_checks["admin_token_present"]:
        diagnosis.append("admin_token_missing")
    if observations and (observations.get("health") or {}).get("ok") is False:
        diagnosis.append("coordinator_unreachable")
    if observations and (observations.get("state") or {}).get("status") in {401, 403}:
        diagnosis.append("observer_auth_failed")
    if observations and (observations.get("admin_results") or {}).get("status") in {401, 403}:
        diagnosis.append("admin_auth_failed")
    if observations and not lane_visible:
        diagnosis.append("task_lane_missing")
    summary = status.get("summary") or {}
    if status and args.require_result and summary.get("ready") is not True:
        diagnosis.append("accepted_result_missing")
    ok = not diagnosis
    if ok:
        diagnosis = ["remote_home_compute_doctor_ready"]
    report = support_bundle.sanitize(redact_values({
        "schema": DOCTOR_SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "workload_kind": args.workload,
        "coordinator_url": args.coordinator_url.rstrip("/") if args.coordinator_url else "",
        "miner_id": args.miner_id,
        "output_dir": str(output_dir),
        "diagnosis_codes": sorted(set(diagnosis)),
        "environment": env_checks,
        "runbook_summary": summarize_external_llm_runbook(runbook) if args.workload == EXTERNAL_LLM_KIND else summarize_runbook(runbook),
        "connectivity": {
            "observations": observations,
            "lane_visible": lane_visible if observations else None,
            "status_ready": summary.get("ready"),
            "matched_capabilities": summary.get("matched_capabilities") or [],
            "missing_capabilities": summary.get("missing_capabilities") or [],
        },
        "operator_action": operator_actions_for_diagnosis(diagnosis),
        "safety": {
            "redacted": True,
            "private_env_files_checked_not_exported": True,
            "registry_hash_required": True,
            "requires_tls_or_vpn": True,
            "not_production": True,
        },
        "limitations": [
            "Preflight and connectivity diagnosis only; does not start production services",
            "Controlled two-machine CPU demo; not P2P routing, GPU pooling, or public prompt serving",
        ],
    }, [args.observer_token, args.admin_token]))
    return report


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def operator_actions_for_diagnosis(codes: list[str]) -> list[str]:
    actions = {
        "operator_env_missing": "Run crowdtensor remote-demo prepare and keep operator.private.env on the operator host.",
        "miner_env_missing": "Run crowdtensor remote-demo prepare and copy only miner.private.env to the Miner host.",
        "registry_missing": "Start the Coordinator with the generated miner_registry.json.",
        "registry_hash_missing": "Regenerate the remote-demo output; the registry must contain hashed Miner tokens only.",
        "runbook_missing": "Run crowdtensor remote-demo prepare before doctor, verify, or collect.",
        "observer_token_missing": "Source operator.private.env or pass --observer-token explicitly.",
        "admin_token_missing": "Source operator.private.env or pass --admin-token explicitly.",
        "coordinator_unreachable": "Check Coordinator process, port, firewall, TLS, VPN, or tunnel reachability.",
        "observer_auth_failed": "Verify CROWDTENSOR_OBSERVER_TOKEN matches the Coordinator configuration.",
        "admin_auth_failed": "Verify CROWDTENSOR_ADMIN_TOKEN matches the Coordinator configuration.",
        "task_lane_missing": "Start the Coordinator with the task lane shown in the generated runbook.",
        "accepted_result_missing": "Start the Miner and rerun verify, or rerun collect after a task is accepted.",
        "artifact_collection_failed": "Rerun collect after checking observer/admin tokens and output directory permissions.",
        "remote_home_compute_doctor_ready": "Proceed to crowdtensor remote-demo verify or collect.",
        "remote_home_compute_collect_ready": "Share the generated JSON/Markdown evidence and support bundle.",
        "remote_home_compute_cleanup_ready": "Review deleted_files before removing private env files in future runs.",
    }
    return [actions[code] for code in codes if code in actions]


def summarize_runbook(payload: dict[str, Any]) -> dict[str, Any]:
    demo = payload.get("demo") if isinstance(payload.get("demo"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "coordinator_url": demo.get("coordinator_url"),
        "workload_type": demo.get("workload_type"),
        "route": demo.get("route"),
        "request_count": demo.get("request_count"),
        "scenario_id": demo.get("scenario_id"),
        "scenario_schema": demo.get("scenario_schema"),
        "registry_hashed": safety.get("registry_hashed"),
        "public_artifact_redacted": safety.get("public_artifact_redacted"),
    }


def summarize_external_llm_runbook(payload: dict[str, Any]) -> dict[str, Any]:
    demo = payload.get("demo") if isinstance(payload.get("demo"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    adapter = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "coordinator_url": demo.get("coordinator_url"),
        "workload_type": demo.get("workload_type"),
        "route": demo.get("route"),
        "request_count": demo.get("request_count"),
        "scenario_id": demo.get("scenario_id"),
        "scenario_schema": demo.get("scenario_schema"),
        "adapter_kind": adapter.get("kind"),
        "registry_hashed": safety.get("registry_hashed"),
        "public_artifact_redacted": safety.get("public_artifact_redacted"),
    }


def summarize_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.get("session_request") if isinstance(payload.get("session_request"), dict) else {}
    evidence = payload.get("evidence_summary") if isinstance(payload.get("evidence_summary"), dict) else {}
    observability = payload.get("observability_summary") if isinstance(payload.get("observability_summary"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "diagnosis_codes": payload.get("diagnosis_codes") or [],
        "session_created": session.get("created"),
        "task_id": session.get("task_id"),
        "scenario_id": (payload.get("scenario") or {}).get("scenario_id"),
        "request_count": session.get("request_count"),
        "evidence_schema": evidence.get("schema"),
        "evidence_ok": evidence.get("ok"),
        "observability_schema": observability.get("schema"),
        "accepted_results": (observability.get("work_queue") or {}).get("accepted_results"),
        "requests_per_second": (observability.get("inference") or {}).get("requests_per_second"),
    }


def summarize_external_llm_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.get("session_request") if isinstance(payload.get("session_request"), dict) else {}
    evidence = payload.get("evidence_summary") if isinstance(payload.get("evidence_summary"), dict) else {}
    observability = payload.get("observability_summary") if isinstance(payload.get("observability_summary"), dict) else {}
    inference = evidence.get("inference") if isinstance(evidence.get("inference"), dict) else {}
    adapter = evidence.get("adapter") if isinstance(evidence.get("adapter"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "diagnosis_codes": payload.get("diagnosis_codes") or [],
        "session_created": session.get("created"),
        "task_id": session.get("task_id"),
        "scenario_id": (payload.get("scenario") or {}).get("scenario_id"),
        "request_count": session.get("request_count"),
        "evidence_schema": evidence.get("schema"),
        "evidence_ok": evidence.get("ok"),
        "observability_schema": observability.get("schema"),
        "accepted_results": (observability.get("work_queue") or {}).get("accepted_results"),
        "requests_per_second": inference.get("requests_per_second"),
        "completion_count": inference.get("completion_count"),
        "output_chars": inference.get("output_chars"),
        "adapter_kind": adapter.get("kind"),
    }


def diagnosis_codes(mode: str, *, step_ok: bool, payload: dict[str, Any]) -> list[str]:
    workload_kind = str(payload.get("workload_kind") or "")
    if mode == "prepare" and workload_kind == EXTERNAL_LLM_KIND:
        return ["remote_external_llm_prepare_ready"] if step_ok and payload.get("ok") else ["remote_external_llm_prepare_failed"]
    if mode == "prepare":
        return ["remote_home_compute_prepare_ready"] if step_ok and payload.get("ok") else ["remote_home_compute_prepare_failed"]
    codes = [str(code) for code in payload.get("diagnosis_codes") or [] if isinstance(code, str)]
    if workload_kind == EXTERNAL_LLM_KIND and payload.get("ok") is True and "remote_external_llm_ready" not in codes:
        codes.append("remote_external_llm_ready")
    if payload.get("ok") is True and "remote_home_compute_ready" not in codes:
        codes.append("remote_home_compute_ready")
    if not codes:
        codes.append("remote_home_compute_failed")
    return sorted(set(codes))


def render_markdown(payload: dict[str, Any]) -> str:
    demo = payload.get("demo") or {}
    runbook = payload.get("runbook_summary") or {}
    acceptance = payload.get("acceptance_summary") or {}
    lines = [
        "# CrowdTensor Remote Home-Compute Demo",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Mode: `{payload.get('mode')}`",
        f"Workload kind: `{demo.get('workload_kind')}`",
        f"Route: `{demo.get('route')}`",
        f"Workload: `{demo.get('workload_type')}`",
        f"Scenario: `{demo.get('scenario_id')}`",
        "",
        "## Summary",
        "",
        f"- Runbook OK: `{runbook.get('ok')}`",
        f"- Acceptance OK: `{acceptance.get('ok')}`",
        f"- Session created: `{acceptance.get('session_created')}`",
        f"- Task ID: `{acceptance.get('task_id')}`",
        f"- Evidence OK: `{acceptance.get('evidence_ok')}`",
        f"- Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in sorted((payload.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def render_doctor_markdown(payload: dict[str, Any]) -> str:
    environment = payload.get("environment") or {}
    connectivity = payload.get("connectivity") or {}
    lines = [
        "# CrowdTensor Remote Demo Doctor",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Workload: `{payload.get('workload_kind')}`",
        f"Coordinator: `{payload.get('coordinator_url', '')}`",
        f"Miner: `{payload.get('miner_id', '')}`",
        f"Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Local Files",
        "",
    ]
    for key, value in sorted(environment.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Connectivity", ""])
    lines.append(f"- Lane visible: `{connectivity.get('lane_visible')}`")
    lines.append(f"- Status ready: `{connectivity.get('status_ready')}`")
    lines.append(f"- Matched capabilities: `{', '.join(connectivity.get('matched_capabilities') or [])}`")
    lines.append(f"- Missing capabilities: `{', '.join(connectivity.get('missing_capabilities') or [])}`")
    lines.extend(["", "## Operator Action", ""])
    for action in payload.get("operator_action") or []:
        lines.append(f"- {action}")
    if not payload.get("operator_action"):
        lines.append("- No action required.")
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def render_collect_markdown(payload: dict[str, Any]) -> str:
    evidence = payload.get("evidence_summary") or {}
    support = payload.get("support_bundle_summary") or {}
    status = payload.get("status_summary") or {}
    lines = [
        "# CrowdTensor Remote Demo Collection",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Workload: `{payload.get('workload_kind')}`",
        f"Coordinator: `{payload.get('coordinator_url', '')}`",
        f"Miner: `{payload.get('miner_id', '')}`",
        f"Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Evidence",
        "",
        f"- Schema: `{evidence.get('schema')}`",
        f"- OK: `{evidence.get('ok')}`",
        f"- Mode: `{evidence.get('mode')}`",
        f"- Route: `{evidence.get('route')}`",
        f"- Requests/sec: `{evidence.get('requests_per_second')}`",
        "",
        "## Runtime Status",
        "",
        f"- Ready: `{status.get('ready')}`",
        f"- Task ID: `{status.get('task_id')}`",
        f"- Accepted results: `{status.get('accepted_results')}`",
        "",
        "## Support Bundle",
        "",
        f"- Schema: `{support.get('schema')}`",
        f"- OK: `{support.get('ok')}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in sorted((payload.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def render_cleanup_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Remote Demo Cleanup",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Mode: `{payload.get('mode')}`",
        f"Output dir: `{payload.get('output_dir')}`",
        f"Deleted bytes: `{payload.get('deleted_bytes')}`",
        f"Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Candidates",
        "",
    ]
    for item in payload.get("candidates") or []:
        lines.append(f"- `{item.get('action')}` `{item.get('kind')}` `{item.get('path')}`")
    lines.append("")
    return "\n".join(lines)


def build_prepare(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.workload == EXTERNAL_LLM_KIND:
        return build_external_llm_prepare(args, output_dir=output_dir)
    runbook_json = output_dir / "remote_demo_runbook.json"
    runbook_md = output_dir / "remote_demo_runbook.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_demo_runbook_pack.py"),
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--json-out",
        str(runbook_json),
        "--markdown-out",
        str(runbook_md),
    ]
    if args.replace:
        command.append("--replace")
    step, runbook = run_json_step("remote_demo_prepare", command, runner=runner, timeout_seconds=args.timeout_seconds)
    step["ok"] = bool(step.get("ok") and runbook.get("ok"))
    return build_report(
        args=args,
        mode="prepare",
        step=step,
        runbook=runbook,
        acceptance={},
        output_dir=output_dir,
        write_outputs=True,
    )


def build_doctor(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = remote_demo_doctor_summary(args, output_dir)
    json_path = output_dir / "remote_home_compute_doctor.json"
    md_path = output_dir / "remote_home_compute_doctor.md"
    payload["artifacts"] = {
        "remote_home_compute_doctor_json": artifact_entry(json_path, output_dir, kind="remote_home_compute_doctor", schema=DOCTOR_SCHEMA, ok=payload.get("ok")),
        "remote_home_compute_doctor_markdown": artifact_entry(md_path, output_dir, kind="remote_home_compute_doctor_markdown"),
        "operator_private_env": artifact_entry(output_dir / "operator.private.env", output_dir, kind="private_env"),
        "miner_private_env": artifact_entry(output_dir / "miner.private.env", output_dir, kind="private_env"),
        "miner_registry": artifact_entry(output_dir / "miner_registry.json", output_dir, kind="miner_registry"),
    }
    write_json(payload, str(json_path))
    md_path.write_text(render_doctor_markdown(payload), encoding="utf-8")
    payload["artifacts"]["remote_home_compute_doctor_json"] = artifact_entry(json_path, output_dir, kind="remote_home_compute_doctor", schema=DOCTOR_SCHEMA, ok=payload.get("ok"))
    payload["artifacts"]["remote_home_compute_doctor_markdown"] = artifact_entry(md_path, output_dir, kind="remote_home_compute_doctor_markdown")
    write_json(payload, str(json_path))
    return payload


def build_verify(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.workload == EXTERNAL_LLM_KIND:
        return build_external_llm_verify(args, output_dir=output_dir, runner=runner)
    acceptance_json = output_dir / "remote_demo_acceptance.json"
    acceptance_md = output_dir / "remote_demo_acceptance.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_demo_acceptance_pack.py"),
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--output-dir",
        str(output_dir),
        "--json-out",
        str(acceptance_json),
        "--markdown-out",
        str(acceptance_md),
    ]
    if args.create_session:
        command.append("--create-session")
    step, acceptance = run_json_step(
        "remote_demo_verify",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=[args.observer_token, args.admin_token],
    )
    step["ok"] = bool(step.get("ok") and acceptance.get("ok"))
    runbook = load_json(output_dir / "remote_demo_runbook.json")
    return build_report(
        args=args,
        mode="verify",
        step=step,
        runbook=runbook,
        acceptance=acceptance,
        output_dir=output_dir,
        secret_values=[args.observer_token, args.admin_token],
        write_outputs=True,
    )


def external_llm_adapter_kind(args: argparse.Namespace) -> str:
    if args.mock or (not args.llm_runtime_cmd and not args.llm_runtime_url):
        return "mock"
    if args.llm_runtime_cmd:
        return "command"
    return "http_openai_chat"


def build_external_llm_commands(args: argparse.Namespace, *, observer_hash: str, admin_hash: str) -> dict[str, str]:
    output_dir = Path(args.output_dir).resolve()
    coordinator_url = args.coordinator_url.rstrip("/")
    registry = output_dir / "miner_registry.json"
    state_dir = "state"
    lane = f"python-cli:cpu:1:{EXTERNAL_LLM_WORKLOAD_TYPE}"
    miner_parts = [
        ". ./miner.private.env &&",
        "crowdtensor-miner",
        f"--coordinator {coordinator_url}",
        f"--miner-id {args.miner_id}",
        "--max-tasks 1",
        "--compute-seconds 0.2",
        "--heartbeat-interval 0.1",
        "--max-request-attempts 5",
        f"--llm-runtime-model-id {args.llm_runtime_model_id}",
        f"--llm-runtime-timeout {args.llm_runtime_timeout}",
    ]
    if args.mock or (not args.llm_runtime_cmd and not args.llm_runtime_url):
        miner_parts.append("--enable-mock-llm-runtime")
    elif args.llm_runtime_cmd:
        miner_parts.append("--llm-runtime-cmd \"$CROWDTENSOR_LLM_RUNTIME_CMD\"")
    else:
        miner_parts.append("--llm-runtime-url \"$CROWDTENSOR_LLM_RUNTIME_URL\"")
        miner_parts.append("--llm-runtime-api-key \"$CROWDTENSOR_LLM_RUNTIME_API_KEY\"")
    return {
        "security_preflight": (
            "python3 scripts/security_preflight.py "
            "--host 0.0.0.0 "
            f"--miner-token-registry {registry} "
            f"--observer-token {observer_hash} "
            f"--admin-token {admin_hash} "
            "--strict --json"
        ),
        "start_coordinator": (
            "crowdtensord "
            "--host 0.0.0.0 "
            "--port 8787 "
            f"--state-dir {state_dir} "
            "--lease-seconds 15 "
            f"--inner-steps {args.request_count} "
            "--backlog 0 "
            f"--task-lane {lane} "
            f"--miner-token-registry {registry} "
            f"--observer-token {observer_hash} "
            f"--admin-token {admin_hash}"
        ),
        "start_miner": " ".join(miner_parts),
        "verify_remote_external_llm": (
            ". ./operator.private.env && "
            "crowdtensor remote-demo verify "
            "--workload external-llm "
            f"--coordinator-url {coordinator_url} "
            f"--miner-id {args.miner_id} "
            "--observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" "
            "--admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" "
            f"--request-count {args.request_count} "
            "--json"
        ),
        "collect_remote_external_llm_evidence": (
            ". ./operator.private.env && "
            "python3 scripts/remote_external_llm_evidence_pack.py "
            "--mode collect "
            f"--coordinator-url {coordinator_url} "
            f"--miner-id {args.miner_id} "
            f"--request-count {args.request_count} "
            "--observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" "
            "--admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" "
            "--json-out /tmp/crowdtensor_remote_external_llm_evidence.json "
            "--markdown-out /tmp/crowdtensor_remote_external_llm_evidence.md"
        ),
    }


def build_external_llm_runbook(
    *,
    args: argparse.Namespace,
    invite: dict[str, Any],
    observer_hash: str,
    admin_hash: str,
    operator_env_path: Path,
    miner_env_path: Path,
) -> dict[str, Any]:
    commands = build_external_llm_commands(args, observer_hash=observer_hash, admin_hash=admin_hash)
    return support_bundle.sanitize({
        "schema": EXTERNAL_LLM_RUNBOOK_SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "workload_kind": EXTERNAL_LLM_KIND,
        "demo": {
            "kind": "controlled_two_machine_remote_external_llm",
            "coordinator_url": args.coordinator_url.rstrip("/"),
            "workload_type": EXTERNAL_LLM_WORKLOAD_TYPE,
            "route": EXTERNAL_LLM_ROUTE_NAME,
            "request_count": args.request_count,
            "scenario_schema": scenario_schema_for(EXTERNAL_LLM_KIND),
            "scenario_id": "fixed-claim-time-prompts",
            "scenario_description": "Fixed external_llm_infer prompt set; not arbitrary prompt serving",
        },
        "adapter": {
            "kind": external_llm_adapter_kind(args),
            "model_id": args.llm_runtime_model_id,
            "operator_owned_runtime": external_llm_adapter_kind(args) != "mock",
        },
        "files": {
            "registry": str(Path(args.output_dir).resolve() / "miner_registry.json"),
            "operator_private_env": str(operator_env_path),
            "miner_private_env": str(miner_env_path),
        },
        "tokens": {
            "miner_registry_hashed": str(invite.get("token_hash", "")).startswith("sha256:"),
            "observer_hash_prefix": observer_hash.split(":", 1)[0],
            "admin_hash_prefix": admin_hash.split(":", 1)[0],
        },
        "commands": commands,
        "operator_steps": [
            "Copy only miner.private.env to the remote Miner host.",
            "Start the Coordinator with the external_llm_infer task lane.",
            "Start the remote Miner with mock, command, or OpenAI-compatible local runtime adapter.",
            "Run crowdtensor remote-demo verify --workload external-llm to create and verify a read-only session.",
            "Share only the generated JSON/Markdown evidence, not private env files.",
        ],
        "safety": {
            "public_artifact_redacted": True,
            "private_env_files": True,
            "registry_hashed": str(invite.get("token_hash", "")).startswith("sha256:"),
            "requires_tls_or_vpn": True,
            "raw_tokens_in_public_report": False,
            "raw_prompts_in_public_report": False,
            "raw_outputs_in_public_report": False,
        },
        "limitations": [
            "Controlled two-machine external_llm_infer demo; not production Swarm Inference",
            "Uses fixed claim-time prompts; not public arbitrary prompt serving",
            "No P2P/NAT traversal, GPU pooling, WebGPU model shards, training, or incentives are claimed",
        ],
    })


def build_external_llm_prepare(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    registry_path = output_dir / "miner_registry.json"
    operator_env_path = output_dir / "operator.private.env"
    miner_env_path = output_dir / "miner.private.env"
    runbook_json = output_dir / "remote_external_llm_runbook.json"
    runbook_md = output_dir / "remote_external_llm_runbook.md"
    miner_token = token_or_generated(args.miner_token)
    observer_token = token_or_generated(args.observer_token)
    admin_token = token_or_generated(args.admin_token)
    observer_hash = hash_token(observer_token)
    admin_hash = hash_token(admin_token)
    invite = create_invite(
        registry_path=registry_path,
        miner_id=args.miner_id,
        coordinator_url=args.coordinator_url,
        label="remote external LLM demo miner",
        token=miner_token,
        replace=args.replace,
    )
    operator_env = {
        "CROWDTENSOR_ADMIN_TOKEN": admin_token,
        "CROWDTENSOR_OBSERVER_TOKEN": observer_token,
    }
    if args.llm_runtime_url:
        operator_env["CROWDTENSOR_LLM_RUNTIME_URL"] = args.llm_runtime_url
    if args.llm_runtime_api_key:
        operator_env["CROWDTENSOR_LLM_RUNTIME_API_KEY"] = args.llm_runtime_api_key
    if args.llm_runtime_cmd:
        operator_env["CROWDTENSOR_LLM_RUNTIME_CMD"] = args.llm_runtime_cmd
    write_private_env(operator_env_path, operator_env)
    write_private_env(miner_env_path, {
        "CROWDTENSOR_MINER_TOKEN": miner_token,
        "CROWDTENSOR_LLM_RUNTIME_CMD": args.llm_runtime_cmd,
        "CROWDTENSOR_LLM_RUNTIME_URL": args.llm_runtime_url,
        "CROWDTENSOR_LLM_RUNTIME_API_KEY": args.llm_runtime_api_key,
    })
    runbook = build_external_llm_runbook(
        args=args,
        invite=invite,
        observer_hash=observer_hash,
        admin_hash=admin_hash,
        operator_env_path=operator_env_path,
        miner_env_path=miner_env_path,
    )
    write_json(runbook, str(runbook_json))
    write_markdown(runbook, str(runbook_md))
    step = {"name": "remote_external_llm_prepare", "ok": bool(runbook.get("ok")), "payload_schema": runbook.get("schema"), "payload_ok": runbook.get("ok")}
    return build_report(
        args=args,
        mode="prepare",
        step=step,
        runbook=runbook,
        acceptance={},
        output_dir=output_dir,
        secret_values=[observer_token, admin_token, miner_token, args.llm_runtime_url, args.llm_runtime_api_key],
        write_outputs=True,
    )


def wait_for_external_llm_result(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.remote_timeout_seconds
    attempts = 0
    errors: list[str] = []
    session_create: dict[str, Any] | None = None
    if args.create_session:
        session_create = create_external_llm_session(args)
        if not session_create.get("ok"):
            return {"ok": False, "attempts": 0, "elapsed_seconds": 0.0, "status": {}, "errors": [], "session_create": session_create}
        args.session_task_id = str((session_create.get("session") or {}).get("task_id") or "")
    last_status: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        attempts += 1
        try:
            last_status = collect_external_llm_status(args)
            if last_status.get("summary", {}).get("ready") is True:
                return {
                    "ok": True,
                    "attempts": attempts,
                    "elapsed_seconds": round(args.remote_timeout_seconds - max(0.0, deadline - time.monotonic()), 3),
                    "status": last_status,
                    "errors": errors[-5:],
                    "session_create": session_create,
                }
        except Exception as exc:
            errors.append(str(exc))
        time.sleep(args.poll_interval)
    return {
        "ok": False,
        "attempts": attempts,
        "elapsed_seconds": args.remote_timeout_seconds,
        "status": last_status,
        "errors": errors[-5:],
        "session_create": session_create,
    }


def run_json_command(command: list[str], *, timeout: float, secret_values: list[str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": completed.returncode,
            "stdout_tail": redact_text(completed.stdout[-2000:], secret_values),
            "stderr_tail": redact_text(completed.stderr[-2000:], secret_values),
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "returncode": completed.returncode, "error": "command emitted no JSON"}
    payload = json.loads(lines[-1])
    return {"ok": bool(payload.get("ok", True)), "payload": payload}


def summarize_external_llm_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route_decision") or {}
    inference = payload.get("inference_summary") or {}
    safety = payload.get("safety") or {}
    observability = payload.get("observability_summary") or {}
    adapter = payload.get("adapter") or {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "route": route.get("name"),
        "route_confidence": route.get("confidence"),
        "route_usable_now": route.get("usable_now"),
        "request_count": inference.get("request_count"),
        "completion_count": inference.get("completion_count"),
        "output_chars": inference.get("output_chars"),
        "requests_per_second": inference.get("requests_per_second"),
        "adapter": {"kind": adapter.get("kind"), "model_id": adapter.get("model_id")},
        "inference": inference,
        "read_only": safety.get("read_only"),
        "redaction_ok": safety.get("redaction_ok"),
        "observability_schema": observability.get("schema"),
    }


def summarize_model_bundle_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route_decision") or {}
    inference = payload.get("inference_summary") or {}
    safety = payload.get("safety") or {}
    observability = payload.get("observability_summary") or {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "route": route.get("name"),
        "route_confidence": route.get("confidence"),
        "route_usable_now": route.get("usable_now"),
        "request_count": inference.get("request_count"),
        "scenario_id": inference.get("scenario_id"),
        "scenario_matches": inference.get("scenario_matches"),
        "accuracy": inference.get("accuracy"),
        "request_trace_count": inference.get("request_trace_count"),
        "requests_per_second": inference.get("requests_per_second"),
        "inference": inference,
        "read_only": safety.get("read_only"),
        "redaction_ok": safety.get("redaction_ok"),
        "observability_schema": observability.get("schema"),
    }


def collect_external_llm_artifacts(args: argparse.Namespace, output_dir: Path, *, secret_values: list[str]) -> dict[str, Any]:
    evidence_json = output_dir / "remote_external_llm_evidence.json"
    evidence_md = output_dir / "remote_external_llm_evidence.md"
    support_json = output_dir / "support_bundle.json"
    support_md = output_dir / "support_bundle.md"
    evidence_command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_external_llm_evidence_pack.py"),
        "--mode",
        "collect",
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--task-id",
        getattr(args, "session_task_id", ""),
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
        "--llm-runtime-model-id",
        args.llm_runtime_model_id,
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.mock:
        evidence_command.append("--mock")
    if args.llm_runtime_cmd:
        evidence_command.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    if args.llm_runtime_url:
        evidence_command.extend(["--llm-runtime-url", args.llm_runtime_url])
    if args.llm_runtime_api_key:
        evidence_command.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    evidence_result = run_json_command(evidence_command, timeout=args.artifact_timeout, secret_values=secret_values)
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
        secret_values=secret_values,
    )
    return {
        "evidence": {
            "ok": bool(evidence_result.get("ok")),
            "path": str(evidence_json),
            "markdown_path": str(evidence_md),
            "summary": summarize_external_llm_evidence(evidence_result.get("payload") or {}),
            "error": evidence_result if not evidence_result.get("ok") else None,
        },
        "support_bundle": {
            "ok": bool(support_result.get("ok")),
            "path": str(support_json),
            "markdown_path": str(support_md),
            "summary": {
                "schema": (support_result.get("payload") or {}).get("schema"),
                "ok": (support_result.get("payload") or {}).get("ok"),
            },
            "error": support_result if not support_result.get("ok") else None,
        },
    }


def collect_model_bundle_artifacts(args: argparse.Namespace, output_dir: Path, *, secret_values: list[str]) -> dict[str, Any]:
    evidence_json = output_dir / "remote_compute_evidence.json"
    evidence_md = output_dir / "remote_compute_evidence.md"
    support_json = output_dir / "support_bundle.json"
    support_md = output_dir / "support_bundle.md"
    evidence_command = [
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
        "--scenario-id",
        args.scenario_id,
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    evidence_result = run_json_command(evidence_command, timeout=args.artifact_timeout, secret_values=secret_values)
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
        secret_values=secret_values,
    )
    return {
        "evidence": {
            "ok": bool(evidence_result.get("ok")),
            "path": str(evidence_json),
            "markdown_path": str(evidence_md),
            "summary": summarize_model_bundle_evidence(evidence_result.get("payload") or {}),
            "error": evidence_result if not evidence_result.get("ok") else None,
        },
        "support_bundle": {
            "ok": bool(support_result.get("ok")),
            "path": str(support_json),
            "markdown_path": str(support_md),
            "summary": {
                "schema": (support_result.get("payload") or {}).get("schema"),
                "ok": (support_result.get("payload") or {}).get("ok"),
            },
            "error": support_result if not support_result.get("ok") else None,
        },
    }


def build_collect(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    secret_values = [args.observer_token, args.admin_token, args.llm_runtime_url, args.llm_runtime_api_key]
    status = collect_status_for_workload(args)
    if args.workload == EXTERNAL_LLM_KIND:
        artifacts = collect_external_llm_artifacts(args, output_dir, secret_values=secret_values)
    else:
        artifacts = collect_model_bundle_artifacts(args, output_dir, secret_values=secret_values)
    evidence_summary = artifacts.get("evidence", {}).get("summary", {})
    support_summary = artifacts.get("support_bundle", {}).get("summary", {})
    status_summary = status.get("summary") or {}
    ok = bool(artifacts.get("evidence", {}).get("ok") and artifacts.get("support_bundle", {}).get("ok"))
    diagnosis = ["remote_home_compute_collect_ready"] if ok else ["artifact_collection_failed"]
    report = support_bundle.sanitize(redact_values({
        "schema": COLLECT_SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "workload_kind": args.workload,
        "coordinator_url": args.coordinator_url,
        "miner_id": args.miner_id,
        "output_dir": str(output_dir),
        "diagnosis_codes": diagnosis,
        "status_summary": {
            "ready": status_summary.get("ready"),
            "task_id": status_summary.get("task_id"),
            "accepted_results": status_summary.get("accepted_results"),
            "matched_capabilities": status_summary.get("matched_capabilities") or [],
            "missing_capabilities": status_summary.get("missing_capabilities") or [],
        },
        "evidence_summary": evidence_summary,
        "support_bundle_summary": support_summary,
        "artifacts": {
            "remote_home_compute_collect_json": artifact_entry(output_dir / "remote_home_compute_collect.json", output_dir, kind="remote_home_compute_collect", schema=COLLECT_SCHEMA, ok=ok),
            "remote_home_compute_collect_markdown": artifact_entry(output_dir / "remote_home_compute_collect.md", output_dir, kind="remote_home_compute_collect_markdown"),
            "remote_compute_evidence_json": artifact_entry(output_dir / "remote_compute_evidence.json", output_dir, kind="remote_compute_evidence", schema="remote_compute_evidence_v1"),
            "remote_external_llm_evidence_json": artifact_entry(output_dir / "remote_external_llm_evidence.json", output_dir, kind="remote_external_llm_evidence", schema="remote_external_llm_evidence_v1"),
            "support_bundle_json": artifact_entry(output_dir / "support_bundle.json", output_dir, kind="support_bundle", schema="support_bundle_v1"),
            "support_bundle_markdown": artifact_entry(output_dir / "support_bundle.md", output_dir, kind="support_bundle_markdown"),
        },
        "operator_action": operator_actions_for_diagnosis(diagnosis),
        "safety": {
            "redacted": True,
            "raw_tokens_in_report": False,
            "raw_outputs_in_report": False,
            "raw_state_dump_in_report": False,
            "not_production": True,
        },
        "limitations": [
            "Collection reads an already running controlled remote demo; it does not start services",
            "Evidence is fixed-scenario/fixed-prompt runtime proof, not production Swarm Inference",
            "No P2P/NAT traversal, GPU pooling, WebGPU shards, training, or incentives are claimed",
        ],
    }, secret_values))
    json_path = output_dir / "remote_home_compute_collect.json"
    md_path = output_dir / "remote_home_compute_collect.md"
    write_json(report, str(json_path))
    md_path.write_text(render_collect_markdown(report), encoding="utf-8")
    report["artifacts"]["remote_home_compute_collect_json"] = artifact_entry(json_path, output_dir, kind="remote_home_compute_collect", schema=COLLECT_SCHEMA, ok=report.get("ok"))
    report["artifacts"]["remote_home_compute_collect_markdown"] = artifact_entry(md_path, output_dir, kind="remote_home_compute_collect_markdown")
    write_json(report, str(json_path))
    return report


def cleanup_candidate(path: Path, output_dir: Path, *, include_private: bool) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    is_private = path.name in REMOTE_DEMO_PRIVATE_FILENAMES
    eligible = path.exists() and path.is_file() and (include_private or not is_private)
    return {
        "path": relative,
        "absolute_path": str(path),
        "kind": "private" if is_private else "generated",
        "present": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "eligible": eligible,
        "skip_reason": "private_requires_include_private" if is_private and not include_private else "",
    }


def build_clean(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    allowed = [output_dir / name for name in REMOTE_DEMO_GENERATED_FILENAMES]
    if args.include_private:
        allowed.extend(output_dir / name for name in REMOTE_DEMO_PRIVATE_FILENAMES)
    candidates = [cleanup_candidate(path, output_dir, include_private=args.include_private) for path in allowed if path.exists()]
    apply = bool(args.apply)
    deleted_bytes = 0
    errors: list[str] = []
    for candidate in candidates:
        candidate["action"] = "skipped"
        if not candidate.get("eligible"):
            continue
        if not apply:
            candidate["action"] = "dry_run"
            continue
        path = Path(str(candidate["absolute_path"]))
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError("refusing to delete non-regular file")
            path.unlink()
        except OSError as exc:
            candidate["action"] = "error"
            candidate["error"] = str(exc)
            errors.append(str(candidate["path"]))
            continue
        candidate["action"] = "deleted"
        deleted_bytes += int(candidate.get("bytes") or 0)
    if args.remove_empty_dir and apply and output_dir.exists():
        try:
            output_dir.rmdir()
        except OSError:
            pass
    ok = not errors
    diagnosis = ["remote_home_compute_cleanup_ready"] if ok else ["remote_home_compute_cleanup_failed"]
    report = support_bundle.sanitize({
        "schema": CLEANUP_SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": "apply" if apply else "dry_run",
        "output_dir": str(output_dir),
        "include_private": bool(args.include_private),
        "remove_empty_dir": bool(args.remove_empty_dir),
        "candidate_count": len(candidates),
        "deleted_bytes": deleted_bytes,
        "errors": errors,
        "diagnosis_codes": diagnosis,
        "operator_action": operator_actions_for_diagnosis(diagnosis),
        "candidates": candidates,
        "safety": {
            "dry_run_default": True,
            "private_files_require_include_private": True,
            "only_known_remote_demo_files": True,
            "does_not_delete_state_or_source": True,
        },
    })
    json_path = output_dir / "remote_home_compute_cleanup.json"
    md_path = output_dir / "remote_home_compute_cleanup.md"
    if output_dir.exists():
        write_json(report, str(json_path))
        md_path.write_text(render_cleanup_markdown(report), encoding="utf-8")
    return report


def build_external_llm_acceptance_report(
    *,
    args: argparse.Namespace,
    wait: dict[str, Any],
    artifacts: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    status = wait.get("status") or {}
    summary = status.get("summary") or {}
    observations = status.get("observations") or {}
    session = (wait.get("session_create") or {}).get("session") or {}
    evidence_summary = (artifacts or {}).get("evidence", {}).get("summary", {})
    support_summary = (artifacts or {}).get("support_bundle", {}).get("summary", {})
    inference = summary.get("inference") or {}
    diagnosis: list[str] = []
    if (wait.get("session_create") or {}).get("ok") is False:
        diagnosis.append("session_create_failed")
    if not status or (observations.get("health") or {}).get("ok") is False:
        diagnosis.append("coordinator_unreachable")
    if (observations.get("state") or {}).get("status") in {401, 403}:
        diagnosis.append("observer_auth_failed")
    if (observations.get("admin_results") or {}).get("status") in {401, 403}:
        diagnosis.append("admin_auth_failed")
    if "external_llm_runtime" in (summary.get("missing_capabilities") or []):
        diagnosis.append("external_llm_runtime_missing")
    if "accepted_result" in (summary.get("missing_capabilities") or []):
        diagnosis.append("no_accepted_result")
    if "validation:ok" in (summary.get("missing_capabilities") or []):
        diagnosis.append("validation_failed")
    if "request_count" in (summary.get("missing_capabilities") or []) or "completion_count" in (summary.get("missing_capabilities") or []):
        diagnosis.append("request_count_mismatch")
    if wait.get("ok") and artifacts and (not artifacts.get("evidence", {}).get("ok") or not artifacts.get("support_bundle", {}).get("ok")):
        diagnosis.append("artifact_collection_failed")
    ok = bool(wait.get("ok") and artifacts and artifacts.get("evidence", {}).get("ok") and artifacts.get("support_bundle", {}).get("ok"))
    if ok:
        diagnosis = ["remote_external_llm_ready"]
    elif not diagnosis:
        diagnosis = ["remote_external_llm_failed"]
    observability = {
        "schema": EXTERNAL_LLM_OBSERVABILITY_SCHEMA,
        "route": EXTERNAL_LLM_ROUTE_NAME,
        "miner_id": args.miner_id,
        "availability": {
            "health_ok": (observations.get("health") or {}).get("ok"),
            "ready_ok": (observations.get("ready") or {}).get("ok"),
            "state_ok": (observations.get("state") or {}).get("ok"),
            "admin_results_ok": (observations.get("admin_results") or {}).get("ok"),
            "acceptance_ready": summary.get("ready"),
            "attempts": wait.get("attempts"),
            "elapsed_seconds": wait.get("elapsed_seconds"),
        },
        "work_queue": {
            "task_counts": summary.get("task_counts", {}),
            "accepted_results": summary.get("accepted_results"),
            "task_id": summary.get("task_id"),
            "expected_task_id": summary.get("expected_task_id"),
        },
        "miner": summary.get("profile", {}),
        "inference": inference,
        "artifacts": {
            "evidence_ok": bool((artifacts or {}).get("evidence", {}).get("ok")),
            "support_bundle_ok": bool((artifacts or {}).get("support_bundle", {}).get("ok")),
            "evidence_observability_schema": evidence_summary.get("observability_schema"),
        },
        "diagnosis_codes": diagnosis,
    }
    report = support_bundle.sanitize({
        "schema": EXTERNAL_LLM_ACCEPTANCE_SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "workload_kind": EXTERNAL_LLM_KIND,
        "coordinator_url": args.coordinator_url,
        "miner_id": args.miner_id,
        "workload_type": EXTERNAL_LLM_WORKLOAD_TYPE,
        "route": EXTERNAL_LLM_ROUTE_NAME,
        "scenario": {
            "scenario_schema": scenario_schema_for(EXTERNAL_LLM_KIND),
            "scenario_id": "fixed-claim-time-prompts",
            "scenario_description": "Fixed external_llm_infer prompt set; not arbitrary prompt serving",
        },
        "session_request": session,
        "wait_summary": {
            "ok": wait.get("ok"),
            "attempts": wait.get("attempts"),
            "elapsed_seconds": wait.get("elapsed_seconds"),
            "status": summary,
            "failure": None if wait.get("ok") else {"summary": summary, "errors": wait.get("errors", [])[-5:]},
        },
        "diagnosis_codes": diagnosis,
        "evidence_summary": evidence_summary,
        "support_bundle_summary": support_summary,
        "observability_summary": observability,
        "artifacts": {
            "evidence_json": (artifacts or {}).get("evidence", {}).get("path"),
            "evidence_markdown": (artifacts or {}).get("evidence", {}).get("markdown_path"),
            "support_bundle_json": (artifacts or {}).get("support_bundle", {}).get("path"),
            "support_bundle_markdown": (artifacts or {}).get("support_bundle", {}).get("markdown_path"),
        },
        "safety": {
            "redacted": True,
            "raw_tokens_in_report": False,
            "raw_prompts_in_report": False,
            "raw_outputs_in_report": False,
            "raw_state_dump_in_report": False,
            "requires_tls_or_vpn": True,
        },
        "limitations": [
            "Validates a controlled two-machine external_llm_infer demo; does not start production services",
            "Uses fixed claim-time prompts and operator-owned runtime adapters; not public arbitrary prompt serving",
            "No P2P discovery, NAT traversal, GPU pooling, WebGPU shards, training, or incentives are claimed",
        ],
    })
    write_json(report, str(output_dir / "remote_external_llm_acceptance.json"))
    (output_dir / "remote_external_llm_acceptance.md").write_text(
        render_external_llm_acceptance_markdown(report),
        encoding="utf-8",
    )
    return report


def render_external_llm_acceptance_markdown(payload: dict[str, Any]) -> str:
    wait = payload.get("wait_summary") or {}
    session = payload.get("session_request") or {}
    evidence = payload.get("evidence_summary") or {}
    inference = evidence.get("inference") if isinstance(evidence.get("inference"), dict) else {}
    adapter = evidence.get("adapter") if isinstance(evidence.get("adapter"), dict) else {}
    observability = payload.get("observability_summary") or {}
    safety = payload.get("safety") or {}
    artifacts = payload.get("artifacts") or {}
    lines = [
        "# CrowdTensor Remote External LLM Acceptance",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Coordinator: `{payload.get('coordinator_url', '')}`",
        f"Miner: `{payload.get('miner_id', '')}`",
        f"Route: `{payload.get('route', '')}`",
        "",
        "## Session",
        "",
        f"- Created: `{session.get('created')}`",
        f"- Schema: `{session.get('schema')}`",
        f"- Task ID: `{session.get('task_id')}`",
        f"- Request count: `{session.get('request_count')}`",
        f"- Workload: `{session.get('workload_type')}`",
        "",
        "## Wait Summary",
        "",
        f"- Ready: `{wait.get('ok')}`",
        f"- Attempts: `{wait.get('attempts')}`",
        f"- Elapsed seconds: `{wait.get('elapsed_seconds')}`",
        f"- Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Evidence",
        "",
        f"- Schema: `{evidence.get('schema')}`",
        f"- OK: `{evidence.get('ok')}`",
        f"- Adapter kind: `{adapter.get('kind')}`",
        f"- Model ID: `{adapter.get('model_id')}`",
        f"- Requests: `{inference.get('request_count')}`",
        f"- Completions: `{inference.get('completion_count')}`",
        f"- Output chars: `{inference.get('output_chars')}`",
        f"- Requests/sec: `{inference.get('requests_per_second')}`",
        "",
        "## Observability",
        "",
        f"- Schema: `{observability.get('schema')}`",
        f"- Acceptance ready: `{(observability.get('availability') or {}).get('acceptance_ready')}`",
        f"- Accepted results: `{(observability.get('work_queue') or {}).get('accepted_results')}`",
        f"- Evidence OK: `{(observability.get('artifacts') or {}).get('evidence_ok')}`",
        f"- Support bundle OK: `{(observability.get('artifacts') or {}).get('support_bundle_ok')}`",
        "",
        "## Safety",
        "",
        f"- Redacted: `{safety.get('redacted')}`",
        f"- Raw tokens in report: `{safety.get('raw_tokens_in_report')}`",
        f"- Raw prompts in report: `{safety.get('raw_prompts_in_report')}`",
        f"- Raw outputs in report: `{safety.get('raw_outputs_in_report')}`",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def build_external_llm_verify(args: argparse.Namespace, *, output_dir: Path, runner: Runner = subprocess.run) -> dict[str, Any]:
    secret_values = [args.observer_token, args.admin_token, args.llm_runtime_url, args.llm_runtime_api_key]
    wait = wait_for_external_llm_result(args)
    artifacts = collect_external_llm_artifacts(args, output_dir, secret_values=secret_values) if wait.get("ok") else None
    acceptance = build_external_llm_acceptance_report(args=args, wait=wait, artifacts=artifacts, output_dir=output_dir)
    step = {
        "name": "remote_external_llm_verify",
        "ok": bool(acceptance.get("ok")),
        "payload_schema": acceptance.get("schema"),
        "payload_ok": acceptance.get("ok"),
    }
    runbook = load_json(output_dir / "remote_external_llm_runbook.json")
    return build_report(
        args=args,
        mode="verify",
        step=step,
        runbook=runbook,
        acceptance=acceptance,
        output_dir=output_dir,
        secret_values=secret_values,
        write_outputs=True,
    )


def build_report(
    *,
    args: argparse.Namespace,
    mode: str,
    step: dict[str, Any],
    runbook: dict[str, Any],
    acceptance: dict[str, Any],
    output_dir: Path,
    write_outputs: bool,
    secret_values: list[str] | None = None,
) -> dict[str, Any]:
    ok = bool(step.get("ok") and (runbook.get("ok") if mode == "prepare" else acceptance.get("ok")))
    workload_kind = args.workload
    workload_type = workload_type_for(workload_kind)
    route_name = route_name_for(workload_kind)
    summary_json = output_dir / "remote_home_compute_demo.json"
    summary_md = output_dir / "remote_home_compute_demo.md"
    artifacts = {
        "remote_home_compute_demo_json": artifact_entry(summary_json, output_dir, kind="remote_home_compute_demo", schema=SCHEMA, ok=ok),
        "remote_home_compute_demo_markdown": artifact_entry(summary_md, output_dir, kind="remote_home_compute_demo_markdown"),
        "remote_demo_runbook_json": artifact_entry(output_dir / "remote_demo_runbook.json", output_dir, kind="remote_demo_runbook", schema="remote_demo_runbook_v1"),
        "remote_demo_runbook_markdown": artifact_entry(output_dir / "remote_demo_runbook.md", output_dir, kind="remote_demo_runbook_markdown"),
        "operator_private_env": artifact_entry(output_dir / "operator.private.env", output_dir, kind="private_env"),
        "miner_private_env": artifact_entry(output_dir / "miner.private.env", output_dir, kind="private_env"),
        "remote_demo_acceptance_json": artifact_entry(output_dir / "remote_demo_acceptance.json", output_dir, kind="remote_demo_acceptance", schema="remote_demo_acceptance_v1"),
        "remote_demo_acceptance_markdown": artifact_entry(output_dir / "remote_demo_acceptance.md", output_dir, kind="remote_demo_acceptance_markdown"),
        "remote_compute_evidence_json": artifact_entry(output_dir / "remote_compute_evidence.json", output_dir, kind="remote_compute_evidence", schema="remote_compute_evidence_v1"),
        "remote_external_llm_runbook_json": artifact_entry(output_dir / "remote_external_llm_runbook.json", output_dir, kind="remote_external_llm_runbook", schema=EXTERNAL_LLM_RUNBOOK_SCHEMA),
        "remote_external_llm_runbook_markdown": artifact_entry(output_dir / "remote_external_llm_runbook.md", output_dir, kind="remote_external_llm_runbook_markdown"),
        "remote_external_llm_acceptance_json": artifact_entry(output_dir / "remote_external_llm_acceptance.json", output_dir, kind="remote_external_llm_acceptance", schema=EXTERNAL_LLM_ACCEPTANCE_SCHEMA),
        "remote_external_llm_acceptance_markdown": artifact_entry(output_dir / "remote_external_llm_acceptance.md", output_dir, kind="remote_external_llm_acceptance_markdown"),
        "remote_external_llm_evidence_json": artifact_entry(output_dir / "remote_external_llm_evidence.json", output_dir, kind="remote_external_llm_evidence", schema="remote_external_llm_evidence_v1"),
        "support_bundle_json": artifact_entry(output_dir / "support_bundle.json", output_dir, kind="support_bundle", schema="support_bundle_v1"),
    }
    runbook_summary = summarize_external_llm_runbook(runbook) if workload_kind == EXTERNAL_LLM_KIND else summarize_runbook(runbook)
    acceptance_summary = summarize_external_llm_acceptance(acceptance) if workload_kind == EXTERNAL_LLM_KIND else summarize_acceptance(acceptance)
    scenario = acceptance.get("scenario") or (runbook.get("demo") or {})
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": mode,
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "miner_id": args.miner_id,
        "demo": {
            "workload_kind": workload_kind,
            "workload_type": workload_type,
            "route": route_name,
            "request_count": args.request_count,
            "scenario_id": scenario.get("scenario_id") or ("fixed-claim-time-prompts" if workload_kind == EXTERNAL_LLM_KIND else args.scenario_id),
            "scenario_schema": scenario.get("scenario_schema") or scenario_schema_for(workload_kind),
            "adapter_kind": runbook_summary.get("adapter_kind") or acceptance_summary.get("adapter_kind"),
        },
        "step": step,
        "runbook_summary": runbook_summary,
        "acceptance_summary": acceptance_summary,
        "diagnosis_codes": diagnosis_codes(mode, step_ok=bool(step.get("ok")), payload=acceptance if mode == "verify" else runbook),
        "artifacts": artifacts,
        "safety": {
            "public_artifact_redacted": True,
            "private_env_files": ["operator.private.env", "miner.private.env"],
            "summary_excludes_plaintext_tokens": True,
            "summary_excludes_raw_inference_payloads": True,
            "summary_excludes_raw_external_llm_payloads": True,
            "raw_state_dump_in_report": False,
            "read_only_workload": workload_type,
            "requires_tls_or_vpn": True,
            "not_production": True,
        },
        "limitations": [
            "Controlled two-machine CPU demo; not production Swarm Inference",
            "Requires operator-provided TLS, VPN, tunnel, or reachable private network for real two-machine use",
            "Does not implement P2P/NAT traversal, GPU pooling, WebGPU model shards, arbitrary prompt serving, training, or incentives",
        ],
        "recommended_next_commands": [
            f"crowdtensor remote-demo prepare --workload {workload_kind} --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --json",
            "source dist/remote-home-compute/operator.private.env",
            f"crowdtensor remote-demo verify --workload {workload_kind} --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --json",
        ],
    }
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    allowed_fragments = {
        "CROWDTENSOR_OBSERVER_TOKEN",
        "CROWDTENSOR_ADMIN_TOKEN",
        "CROWDTENSOR_MINER_TOKEN",
    }
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded and fragment not in allowed_fragments]
    leaks.extend(secret for secret in secret_values or [] if secret and secret in encoded)
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"sensitive_output_detected"})
        report["safety_error"] = "remote home-compute demo report contained secret-like fragments"
    if write_outputs:
        write_json(report, str(summary_json))
        write_markdown(report, str(summary_md))
        report["artifacts"]["remote_home_compute_demo_json"] = artifact_entry(summary_json, output_dir, kind="remote_home_compute_demo", schema=SCHEMA, ok=report.get("ok"))
        report["artifacts"]["remote_home_compute_demo_markdown"] = artifact_entry(summary_md, output_dir, kind="remote_home_compute_demo_markdown")
        write_json(report, str(summary_json))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or verify the CrowdTensor remote home-compute demo.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common: dict[str, Any] = {"coordinator_url": "http://127.0.0.1:8787", "miner_id": "remote-linux-1"}
    prepare = subparsers.add_parser("prepare", help="Create the remote home-compute runbook and private env files.")
    prepare.add_argument("--workload", choices=[MODEL_BUNDLE_KIND, EXTERNAL_LLM_KIND], default=MODEL_BUNDLE_KIND)
    prepare.add_argument("--coordinator-url", default=common["coordinator_url"])
    prepare.add_argument("--miner-id", default=common["miner_id"])
    prepare.add_argument("--output-dir", default="dist/remote-home-compute")
    prepare.add_argument("--request-count", type=int, default=4)
    prepare.add_argument("--scenario-id", default="route-baseline")
    prepare.add_argument("--timeout-seconds", type=float, default=180.0)
    prepare.add_argument("--replace", action="store_true")
    prepare.add_argument("--mock", action="store_true", help="use deterministic mock external LLM runtime for --workload external-llm")
    prepare.add_argument("--llm-runtime-cmd", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_CMD", ""))
    prepare.add_argument("--llm-runtime-url", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_URL", ""))
    prepare.add_argument("--llm-runtime-api-key", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_API_KEY", ""))
    prepare.add_argument("--llm-runtime-model-id", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_MODEL_ID", "external-llm-runtime"))
    prepare.add_argument("--llm-runtime-timeout", type=float, default=float(os.environ.get("CROWDTENSOR_LLM_RUNTIME_TIMEOUT", "30.0")))
    prepare.add_argument("--miner-token", default="")
    prepare.add_argument("--observer-token", default="")
    prepare.add_argument("--admin-token", default="")
    prepare.add_argument("--json", action="store_true")
    verify = subparsers.add_parser("verify", help="Create and verify a read-only remote home-compute session.")
    verify.add_argument("--workload", choices=[MODEL_BUNDLE_KIND, EXTERNAL_LLM_KIND], default=MODEL_BUNDLE_KIND)
    verify.add_argument("--coordinator-url", required=True)
    verify.add_argument("--miner-id", required=True)
    verify.add_argument("--observer-token", required=True)
    verify.add_argument("--admin-token", required=True)
    verify.add_argument("--output-dir", default="dist/remote-home-compute")
    verify.add_argument("--request-count", type=int, default=4)
    verify.add_argument("--scenario-id", default="route-baseline")
    verify.add_argument("--timeout-seconds", type=float, default=180.0)
    verify.add_argument("--remote-timeout-seconds", type=float, default=120.0)
    verify.add_argument("--poll-interval", type=float, default=2.0)
    verify.add_argument("--http-timeout", type=float, default=5.0)
    verify.add_argument("--artifact-timeout", type=float, default=60.0)
    verify.add_argument("--admin-results-limit", type=int, default=10)
    verify.add_argument("--create-session", dest="create_session", action="store_true", default=True)
    verify.add_argument("--no-create-session", dest="create_session", action="store_false")
    verify.add_argument("--mock", action="store_true", help="use deterministic mock external LLM runtime for --workload external-llm")
    verify.add_argument("--llm-runtime-cmd", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_CMD", ""))
    verify.add_argument("--llm-runtime-url", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_URL", ""))
    verify.add_argument("--llm-runtime-api-key", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_API_KEY", ""))
    verify.add_argument("--llm-runtime-model-id", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_MODEL_ID", "external-llm-runtime"))
    verify.add_argument("--llm-runtime-timeout", type=float, default=float(os.environ.get("CROWDTENSOR_LLM_RUNTIME_TIMEOUT", "30.0")))
    verify.add_argument("--json", action="store_true")
    doctor = subparsers.add_parser("doctor", help="Check remote-demo files, tokens, Coordinator reachability, and route readiness.")
    doctor.add_argument("--workload", choices=[MODEL_BUNDLE_KIND, EXTERNAL_LLM_KIND], default=MODEL_BUNDLE_KIND)
    doctor.add_argument("--coordinator-url", default=common["coordinator_url"])
    doctor.add_argument("--miner-id", default=common["miner_id"])
    doctor.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    doctor.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    doctor.add_argument("--output-dir", default="dist/remote-home-compute")
    doctor.add_argument("--request-count", type=int, default=4)
    doctor.add_argument("--scenario-id", default="route-baseline")
    doctor.add_argument("--http-timeout", type=float, default=5.0)
    doctor.add_argument("--admin-results-limit", type=int, default=10)
    doctor.add_argument("--require-result", action="store_true")
    doctor.add_argument("--json", action="store_true")
    collect = subparsers.add_parser("collect", help="Collect evidence and Support Bundle from an already running remote-demo.")
    collect.add_argument("--workload", choices=[MODEL_BUNDLE_KIND, EXTERNAL_LLM_KIND], default=MODEL_BUNDLE_KIND)
    collect.add_argument("--coordinator-url", required=True)
    collect.add_argument("--miner-id", required=True)
    collect.add_argument("--observer-token", required=True)
    collect.add_argument("--admin-token", required=True)
    collect.add_argument("--output-dir", default="dist/remote-home-compute")
    collect.add_argument("--request-count", type=int, default=4)
    collect.add_argument("--scenario-id", default="route-baseline")
    collect.add_argument("--task-id", default="")
    collect.add_argument("--http-timeout", type=float, default=5.0)
    collect.add_argument("--artifact-timeout", type=float, default=60.0)
    collect.add_argument("--admin-results-limit", type=int, default=10)
    collect.add_argument("--mock", action="store_true", help="use deterministic mock external LLM runtime metadata for --workload external-llm")
    collect.add_argument("--llm-runtime-cmd", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_CMD", ""))
    collect.add_argument("--llm-runtime-url", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_URL", ""))
    collect.add_argument("--llm-runtime-api-key", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_API_KEY", ""))
    collect.add_argument("--llm-runtime-model-id", default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_MODEL_ID", "external-llm-runtime"))
    collect.add_argument("--llm-runtime-timeout", type=float, default=float(os.environ.get("CROWDTENSOR_LLM_RUNTIME_TIMEOUT", "30.0")))
    collect.add_argument("--json", action="store_true")
    clean = subparsers.add_parser("clean", help="Dry-run or delete known remote-demo generated artifacts.")
    clean.add_argument("--output-dir", default="dist/remote-home-compute")
    clean.add_argument("--apply", action="store_true")
    clean.add_argument("--include-private", action="store_true", help="also delete operator.private.env, miner.private.env, and miner_registry.json")
    clean.add_argument("--remove-empty-dir", action="store_true")
    clean.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(args, "request_count") and args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if hasattr(args, "timeout_seconds") and args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if getattr(args, "remote_timeout_seconds", 0) < 0:
        raise SystemExit("--remote-timeout-seconds must be non-negative")
    if getattr(args, "poll_interval", 1) <= 0:
        raise SystemExit("--poll-interval must be positive")
    if getattr(args, "http_timeout", 1) <= 0:
        raise SystemExit("--http-timeout must be positive")
    if getattr(args, "artifact_timeout", 1) <= 0:
        raise SystemExit("--artifact-timeout must be positive")
    if getattr(args, "admin_results_limit", 1) < 1:
        raise SystemExit("--admin-results-limit must be at least 1")
    if getattr(args, "llm_runtime_cmd", "") and getattr(args, "llm_runtime_url", ""):
        raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
    if getattr(args, "llm_runtime_timeout", 1) <= 0:
        raise SystemExit("--llm-runtime-timeout must be positive")
    if hasattr(args, "coordinator_url"):
        args.coordinator_url = args.coordinator_url.rstrip("/")
    args.session_task_id = ""
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        payload = build_prepare(args)
    elif args.mode == "verify":
        payload = build_verify(args)
    elif args.mode == "doctor":
        payload = build_doctor(args)
    elif args.mode == "collect":
        payload = build_collect(args)
    elif args.mode == "clean":
        payload = build_clean(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"CrowdTensor remote home-compute demo {args.mode}")
        print(f"  ok: {payload.get('ok')}")
        print(f"  schema: {payload.get('schema')}")
        print(f"  output: {payload.get('output_dir')}")
        print(f"  diagnosis: {', '.join(payload.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if payload.get("ok") else 1)


if __name__ == "__main__":
    main()
