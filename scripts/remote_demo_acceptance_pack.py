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
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from crowdtensor.model_bundle import (  # noqa: E402
    DEFAULT_INFERENCE_SCENARIO_ID,
    inference_scenario_summary,
    normalize_inference_scenario_id,
)
import support_bundle  # noqa: E402


ACCEPTANCE_SCHEMA = "remote_demo_acceptance_v1"
WORKLOAD_TYPE = "model_bundle_infer"
ROUTE_NAME = "remote_python_model_bundle_infer"
DIAGNOSIS_ORDER = [
    "session_create_failed",
    "coordinator_unreachable",
    "observer_auth_failed",
    "admin_auth_failed",
    "miner_not_seen",
    "task_lane_missing",
    "workload_not_advertised",
    "no_accepted_result",
    "validation_failed",
    "request_count_mismatch",
    "artifact_collection_failed",
    "acceptance_ready",
]
DIAGNOSIS_LIBRARY = {
    "acceptance_ready": {
        "severity": "info",
        "summary": "Remote demo acceptance completed.",
        "next_steps": [
            "Share the generated acceptance report, remote compute evidence, and support bundle.",
            "Use the same command against the real two-machine Coordinator and Miner before public claims.",
        ],
    },
    "coordinator_unreachable": {
        "severity": "error",
        "summary": "The Coordinator health endpoint is not reachable.",
        "next_steps": [
            "Verify the Coordinator process is running and listening on the expected host and port.",
            "Check firewall, tunnel, TLS, and VPN settings before rerunning the acceptance pack.",
        ],
    },
    "observer_auth_failed": {
        "severity": "error",
        "summary": "The observer token was rejected by the Coordinator state endpoint.",
        "next_steps": [
            "Regenerate or recopy the operator private environment file.",
            "Pass the correct CROWDTENSOR_OBSERVER_TOKEN value without exposing it in logs.",
        ],
    },
    "admin_auth_failed": {
        "severity": "error",
        "summary": "The admin token was rejected by the Coordinator result ledger endpoint.",
        "next_steps": [
            "Regenerate or recopy the operator private environment file.",
            "Pass the correct CROWDTENSOR_ADMIN_TOKEN value without exposing it in logs.",
        ],
    },
    "session_create_failed": {
        "severity": "error",
        "summary": "The admin-created read-only inference session could not be queued.",
        "next_steps": [
            "Confirm the Coordinator supports POST /admin/inference-sessions.",
            "Verify the admin token and rerun with the same --request-count value.",
        ],
    },
    "miner_not_seen": {
        "severity": "warning",
        "summary": "The Coordinator is reachable but the selected Miner profile is absent.",
        "next_steps": [
            "Start the remote Miner with the exact miner id used by the acceptance command.",
            "Check the Miner invite, registry hash, network path, and claim logs.",
        ],
    },
    "task_lane_missing": {
        "severity": "warning",
        "summary": "No visible python-cli/cpu model_bundle_infer task lane is configured.",
        "next_steps": [
            "Start the Coordinator with --task-lane python-cli:cpu:1:model_bundle_infer.",
            "Confirm /ready shows a model_bundle_infer lane before starting the remote Miner.",
        ],
    },
    "workload_not_advertised": {
        "severity": "warning",
        "summary": "The selected Miner does not advertise model_bundle_infer support.",
        "next_steps": [
            "Run the Miner with model bundle inference enabled.",
            "Confirm the Miner profile lists model_bundle_infer in supported_workloads.",
        ],
    },
    "no_accepted_result": {
        "severity": "warning",
        "summary": "No accepted model_bundle_infer result exists for the selected Miner yet.",
        "next_steps": [
            "Keep the acceptance pack running until the Miner claims, computes, and submits a result.",
            "Inspect Miner logs for claim failures, lease expiry, or validation rejection.",
        ],
    },
    "validation_failed": {
        "severity": "error",
        "summary": "The accepted result failed model_bundle_infer validation.",
        "next_steps": [
            "Inspect the validation code and model bundle inference trace in the evidence pack.",
            "Rerun the remote demo after fixing the Miner runtime or bundle configuration.",
        ],
    },
    "request_count_mismatch": {
        "severity": "error",
        "summary": "The accepted result used a different request count than the acceptance command.",
        "next_steps": [
            "Rerun the acceptance pack and evidence collection with the same --request-count value.",
            "Check whether stale ledger rows from an earlier run are being selected.",
        ],
    },
    "artifact_collection_failed": {
        "severity": "error",
        "summary": "Remote result acceptance passed, but evidence or support bundle collection failed.",
        "next_steps": [
            "Inspect the acceptance report artifact summary and rerun the evidence pack manually.",
            "Collect a support bundle after verifying observer and admin access.",
        ],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_json(
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
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def http_detail(body: str) -> str:
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:300]
    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, (dict, list)):
        return json.dumps(support_bundle.sanitize(detail), sort_keys=True)[:300]
    return str(detail)[:300]


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
            "detail": http_detail(body),
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


def observation_digest(response: dict[str, Any]) -> dict[str, Any]:
    digest = {
        "endpoint": response.get("endpoint"),
        "path": response.get("path"),
        "ok": bool(response.get("ok")),
        "status": response.get("status"),
    }
    for key in ["error", "detail"]:
        if response.get(key):
            digest[key] = response.get(key)
    return digest


def response_payload(response: dict[str, Any]) -> dict[str, Any]:
    payload = response.get("json")
    return payload if isinstance(payload, dict) else {}


def response_public_payload(response: dict[str, Any]) -> dict[str, Any]:
    payload = response_payload(response)
    return payload if response.get("ok") and payload else observation_digest(response)


def admin_results_url(miner_id: str, limit: int, *, task_id: str = "") -> str:
    query_params = {
        "status": "accepted",
        "workload_type": WORKLOAD_TYPE,
        "limit": limit,
    }
    if task_id:
        query_params["task_id"] = task_id
    else:
        query_params["miner_id"] = miner_id
    query = urlencode(query_params)
    return f"/admin/results?{query}"


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


def latest_result(results: dict[str, Any]) -> dict[str, Any] | None:
    rows = results.get("results") if isinstance(results, dict) else []
    if not isinstance(rows, list) or not rows:
        return None
    return rows[0]


def miner_profile(state: dict[str, Any], miner_id: str) -> dict[str, Any]:
    return (state.get("miner_profiles") or {}).get(miner_id) or {}


def model_bundle_lane_visible(ready: dict[str, Any], state: dict[str, Any]) -> bool:
    lanes: list[dict[str, Any]] = []
    for source in [ready, state]:
        source_lanes = source.get("task_lanes") if isinstance(source, dict) else []
        if isinstance(source_lanes, list):
            lanes.extend(lane for lane in source_lanes if isinstance(lane, dict))
    for lane in lanes:
        if (
            lane.get("runtime") == "python-cli"
            and lane.get("backend") == "cpu"
            and lane.get("workload_type") == WORKLOAD_TYPE
        ):
            return True
    counts = state.get("task_counts_by_lane") if isinstance(state, dict) else {}
    if isinstance(counts, dict):
        for lane_key in counts:
            if (
                "python-cli" in str(lane_key)
                and "cpu" in str(lane_key)
                and WORKLOAD_TYPE in str(lane_key)
            ):
                return True
    return False


def readiness_summary(
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
    task = completed_task_for(state, miner_id, task_id=task_id)
    row = latest_result(results)
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    scenario = inference_scenario_summary(scenario_id)
    expected_scenario_id = scenario.get("scenario_id")
    actual_scenario_id = str(validation.get("scenario_id") or "")
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
    if expected_scenario_id and actual_scenario_id == expected_scenario_id:
        matched.append("scenario_id")
    else:
        missing.append("scenario_id")
    return {
        "ready": not missing,
        "route": ROUTE_NAME,
        "miner_id": miner_id,
        "scenario": scenario,
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
            "scenario_schema": validation.get("scenario_schema") or scenario.get("scenario_schema"),
            "scenario_id": actual_scenario_id or expected_scenario_id,
            "scenario_description": validation.get("scenario_description") or scenario.get("scenario_description"),
            "scenario_request_count": validation.get("scenario_request_count") or scenario.get("scenario_request_count"),
            "expected_scenario_id": expected_scenario_id,
            "scenario_matches": bool(expected_scenario_id and actual_scenario_id == expected_scenario_id),
            "request_trace_count": validation.get("request_trace_count"),
            "accuracy": validation.get("accuracy"),
        },
    }


def collect_status(args: argparse.Namespace) -> dict[str, Any]:
    task_id = getattr(args, "session_task_id", "") or ""
    health_response = request_json_observed(
        "health",
        "GET",
        args.coordinator_url,
        "/health",
        timeout=args.http_timeout,
    )
    ready_response = request_json_observed(
        "ready",
        "GET",
        args.coordinator_url,
        "/ready",
        timeout=args.http_timeout,
    )
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
        admin_results_url(args.miner_id, args.admin_results_limit, task_id=task_id),
        admin_token=args.admin_token,
        timeout=args.http_timeout,
    )
    health = response_public_payload(health_response)
    ready = response_public_payload(ready_response)
    state = response_payload(state_response)
    results = response_payload(results_response)
    return {
        "health": health,
        "ready": ready,
        "state": state,
        "results": results,
        "observations": {
            "health": observation_digest(health_response),
            "ready": observation_digest(ready_response),
            "state": observation_digest(state_response),
            "admin_results": observation_digest(results_response),
        },
        "summary": readiness_summary(
            state=state,
            results=results,
            miner_id=args.miner_id,
            request_count=args.request_count,
            scenario_id=args.scenario_id,
            task_id=task_id,
        ),
    }


def safe_session_request(session: dict[str, Any] | None, *, create_session: bool) -> dict[str, Any]:
    session = session or {}
    return {
        "created": bool(create_session and (session.get("accepted") is True or session.get("created") is True)),
        "schema": session.get("schema"),
        "task_id": session.get("task_id"),
        "request_count": session.get("request_count"),
        "scenario_schema": session.get("scenario_schema"),
        "scenario_id": session.get("scenario_id"),
        "scenario_description": session.get("scenario_description"),
        "scenario_request_count": session.get("scenario_request_count"),
        "workload_type": session.get("workload_type"),
        "status": session.get("status"),
        "result_query": session.get("result_query"),
    }


def create_inference_session(args: argparse.Namespace) -> dict[str, Any]:
    response = request_json_observed(
        "admin_inference_session",
        "POST",
        args.coordinator_url,
        "/admin/inference-sessions",
        payload={"request_count": args.request_count, "scenario_id": args.scenario_id},
        admin_token=args.admin_token,
        timeout=args.http_timeout,
    )
    if not response.get("ok"):
        return {"ok": False, "observation": observation_digest(response)}
    session = response_payload(response)
    if (
        session.get("schema") != "inference_session_request_v1"
        or session.get("workload_type") != WORKLOAD_TYPE
        or not session.get("task_id")
        or int(session.get("request_count") or 0) != int(args.request_count)
        or session.get("scenario_id") != args.scenario_id
    ):
        return {
            "ok": False,
            "observation": observation_digest(response),
            "session": support_bundle.sanitize(session),
            "error": "invalid_session_response",
        }
    return {
        "ok": True,
        "observation": observation_digest(response),
        "session": safe_session_request(session, create_session=True),
    }


def wait_for_remote_result(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.timeout_seconds
    attempts = 0
    last_status: dict[str, Any] = {}
    errors: list[str] = []
    session_create: dict[str, Any] | None = None
    if getattr(args, "create_session", False):
        session_create = create_inference_session(args)
        if not session_create.get("ok"):
            return {
                "ok": False,
                "attempts": 0,
                "elapsed_seconds": 0.0,
                "status": {},
                "errors": errors[-5:],
                "session_create": session_create,
            }
        args.session_task_id = str((session_create.get("session") or {}).get("task_id") or "")
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
                    "session_create": session_create,
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
        "session_create": session_create,
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
    observability = payload.get("observability_summary") or {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "route": route.get("name"),
        "route_confidence": route.get("confidence"),
        "route_usable_now": route.get("usable_now"),
        "request_count": summary.get("request_count"),
        "scenario_schema": summary.get("scenario_schema"),
        "scenario_id": summary.get("scenario_id"),
        "scenario_matches": summary.get("scenario_matches"),
        "request_trace_count": summary.get("request_trace_count"),
        "accuracy": summary.get("accuracy"),
        "requests_per_second": summary.get("requests_per_second"),
        "read_only": safety.get("read_only"),
        "redaction_ok": safety.get("redaction_ok"),
        "observability_schema": observability.get("schema"),
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


def endpoint_observations(status: dict[str, Any]) -> dict[str, Any]:
    observations = status.get("observations")
    if isinstance(observations, dict) and observations:
        return observations
    observed: dict[str, Any] = {}
    for key in ["health", "ready"]:
        payload = status.get(key)
        if isinstance(payload, dict):
            observed[key] = {
                "endpoint": key,
                "ok": payload.get("ok"),
                "status": payload.get("status"),
            }
    return observed


def status_observed_digest(status: dict[str, Any]) -> dict[str, Any]:
    summary = status.get("summary") or {}
    return {
        "endpoints": endpoint_observations(status),
        "ready": summary.get("ready"),
        "matched_capabilities": summary.get("matched_capabilities", []),
        "missing_capabilities": summary.get("missing_capabilities", []),
        "task_id": summary.get("task_id"),
        "expected_task_id": summary.get("expected_task_id"),
        "accepted_results": summary.get("accepted_results"),
        "task_counts": summary.get("task_counts", {}),
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
    status: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> None:
    if any(item.get("primary_code") == code for item in diagnoses):
        return
    diagnoses.append(diagnosis_payload(code, details=details, observed=status_observed_digest(status)))


def sorted_diagnoses(diagnoses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {code: index for index, code in enumerate(DIAGNOSIS_ORDER)}
    return sorted(diagnoses, key=lambda item: order.get(str(item.get("primary_code")), len(order)))


def artifact_failure_details(artifacts: dict[str, Any] | None) -> dict[str, Any]:
    artifacts = artifacts or {}
    evidence = artifacts.get("evidence") or {}
    support = artifacts.get("support_bundle") or {}
    return {
        "evidence_ok": bool(evidence.get("ok")),
        "support_bundle_ok": bool(support.get("ok")),
        "evidence_path": evidence.get("path"),
        "support_bundle_path": support.get("path"),
    }


def build_observability_summary(
    *,
    args: argparse.Namespace,
    wait: dict[str, Any],
    artifacts: dict[str, Any] | None,
    diagnosis_codes: list[str],
) -> dict[str, Any]:
    status = wait.get("status") or {}
    summary = status.get("summary") or {}
    artifacts = artifacts or {}
    evidence = artifacts.get("evidence") or {}
    support = artifacts.get("support_bundle") or {}
    evidence_summary = evidence.get("summary") or {}
    support_summary = support.get("summary") or {}
    observations = endpoint_observations(status)
    return {
        "schema": "remote_demo_observability_v1",
        "route": ROUTE_NAME,
        "miner_id": args.miner_id,
        "session_request": safe_session_request((wait.get("session_create") or {}).get("session"), create_session=bool(getattr(args, "create_session", False))),
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
        "inference": {
            "ok": (summary.get("inference") or {}).get("ok"),
            "request_count": (summary.get("inference") or {}).get("request_count"),
            "scenario_schema": (summary.get("inference") or {}).get("scenario_schema"),
            "scenario_id": (summary.get("inference") or {}).get("scenario_id"),
            "expected_scenario_id": (summary.get("inference") or {}).get("expected_scenario_id"),
            "scenario_matches": (summary.get("inference") or {}).get("scenario_matches"),
            "request_trace_count": (summary.get("inference") or {}).get("request_trace_count"),
            "accuracy": (summary.get("inference") or {}).get("accuracy"),
            "requests_per_second": evidence_summary.get("requests_per_second"),
        },
        "artifacts": {
            "evidence_ok": bool(evidence.get("ok")),
            "support_bundle_ok": bool(support.get("ok")),
            "evidence_path": evidence.get("path"),
            "support_bundle_path": support.get("path"),
            "evidence_observability_schema": evidence_summary.get("observability_schema"),
            "support_online_enabled": support_summary.get("online_enabled"),
        },
        "diagnosis_codes": list(diagnosis_codes),
    }


def diagnose_acceptance(
    *,
    args: argparse.Namespace,
    wait: dict[str, Any],
    artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    status = wait.get("status") or {}
    session_create = wait.get("session_create") or {}
    observations = endpoint_observations(status)
    health_observation = observations.get("health") or {}
    state_observation = observations.get("state") or {}
    results_observation = observations.get("admin_results") or {}
    state = status.get("state") if isinstance(status.get("state"), dict) else {}
    ready = status.get("ready") if isinstance(status.get("ready"), dict) else {}
    results = status.get("results") if isinstance(status.get("results"), dict) else {}
    task_id = str((session_create.get("session") or {}).get("task_id") or getattr(args, "session_task_id", "") or "")
    task = completed_task_for(state, args.miner_id, task_id=task_id)
    row = latest_result(results)
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    profile = miner_profile(state, args.miner_id)
    capabilities = profile.get("last_capabilities") or {}
    workloads = capabilities.get("supported_workloads") or []
    diagnoses: list[dict[str, Any]] = []

    if getattr(args, "create_session", False) and session_create.get("ok") is not True:
        append_diagnosis(
            diagnoses,
            "session_create_failed",
            status=status,
            details=support_bundle.sanitize({
                "session_create": session_create,
                "request_count": args.request_count,
                "scenario_id": args.scenario_id,
            }),
        )

    if (
        not status
        or (health_observation and health_observation.get("ok") is False and health_observation.get("status") is None)
    ):
        append_diagnosis(
            diagnoses,
            "coordinator_unreachable",
            status=status,
            details={
                "health": health_observation,
                "recent_errors": wait.get("errors", [])[-3:],
            },
        )

    if state_observation.get("status") in {401, 403}:
        append_diagnosis(
            diagnoses,
            "observer_auth_failed",
            status=status,
            details={
                "state_status": state_observation.get("status"),
                "state_detail": state_observation.get("detail", ""),
            },
        )

    if results_observation.get("status") in {401, 403}:
        append_diagnosis(
            diagnoses,
            "admin_auth_failed",
            status=status,
            details={
                "admin_results_status": results_observation.get("status"),
                "admin_results_detail": results_observation.get("detail", ""),
            },
        )

    state_available = bool(state) and state_observation.get("ok", True) is not False
    ready_available = bool(ready) and observations.get("ready", {}).get("ok", True) is not False
    if state_available and not profile:
        known_miners = sorted((state.get("miner_profiles") or {}).keys())
        append_diagnosis(
            diagnoses,
            "miner_not_seen",
            status=status,
            details={
                "miner_id": args.miner_id,
                "known_miners": known_miners[:10],
                "known_miner_count": len(known_miners),
            },
        )

    if (state_available or ready_available) and not task and not row and not model_bundle_lane_visible(ready, state):
        append_diagnosis(
            diagnoses,
            "task_lane_missing",
            status=status,
            details={
                "expected_lane": f"python-cli:cpu:1:{WORKLOAD_TYPE}",
                "ready_task_lanes": ready.get("task_lanes", []),
                "state_task_lanes": state.get("task_lanes", []),
            },
        )

    if profile and WORKLOAD_TYPE not in workloads:
        append_diagnosis(
            diagnoses,
            "workload_not_advertised",
            status=status,
            details={
                "miner_id": args.miner_id,
                "supported_workloads": list(workloads),
            },
        )

    if profile and not (task and row):
        ledger_rows = results.get("results") if isinstance(results.get("results"), list) else []
        append_diagnosis(
            diagnoses,
            "no_accepted_result",
            status=status,
            details={
                "miner_id": args.miner_id,
                "task_id": task_id,
                "attempts": wait.get("attempts"),
                "task_counts": state.get("task_counts", {}),
                "accepted_results": state.get("accepted_results"),
                "ledger_row_count": len(ledger_rows),
            },
        )

    if (task or row) and validation.get("code") != "ok":
        append_diagnosis(
            diagnoses,
            "validation_failed",
            status=status,
            details={"validation": support_bundle.sanitize(validation)},
        )

    if validation and int(validation.get("request_count") or 0) != int(args.request_count):
        append_diagnosis(
            diagnoses,
            "request_count_mismatch",
            status=status,
            details={
                "expected_request_count": int(args.request_count),
                "actual_request_count": validation.get("request_count"),
                "validation": support_bundle.sanitize(validation),
            },
        )

    if validation and validation.get("scenario_id") != args.scenario_id:
        append_diagnosis(
            diagnoses,
            "validation_failed",
            status=status,
            details={
                "expected_scenario_id": args.scenario_id,
                "actual_scenario_id": validation.get("scenario_id"),
                "validation": support_bundle.sanitize(validation),
            },
        )

    if wait.get("ok") and artifacts and (
        not artifacts.get("evidence", {}).get("ok")
        or not artifacts.get("support_bundle", {}).get("ok")
    ):
        append_diagnosis(
            diagnoses,
            "artifact_collection_failed",
            status=status,
            details=artifact_failure_details(artifacts),
        )

    if not diagnoses:
        append_diagnosis(
            diagnoses,
            "acceptance_ready",
            status=status,
            details={
                "miner_id": args.miner_id,
                "workload_type": WORKLOAD_TYPE,
                "route": ROUTE_NAME,
                "scenario_id": args.scenario_id,
            },
        )

    diagnoses = sorted_diagnoses(diagnoses)
    return {
        "primary": diagnoses[0],
        "all": diagnoses,
        "codes": [str(item.get("primary_code")) for item in diagnoses],
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
    diagnosis = diagnose_acceptance(args=args, wait=wait, artifacts=artifacts)
    session_request = safe_session_request((wait.get("session_create") or {}).get("session"), create_session=bool(getattr(args, "create_session", False)))
    observability = build_observability_summary(
        args=args,
        wait=wait,
        artifacts=artifacts,
        diagnosis_codes=diagnosis["codes"],
    )
    scenario = inference_scenario_summary(args.scenario_id)
    report = {
        "schema": ACCEPTANCE_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "ok": ok,
        "coordinator_url": args.coordinator_url,
        "miner_id": args.miner_id,
        "workload_type": WORKLOAD_TYPE,
        "route": ROUTE_NAME,
        "scenario": scenario,
        "session_request": session_request,
        "wait_summary": {
            "ok": wait.get("ok"),
            "attempts": wait.get("attempts"),
            "elapsed_seconds": wait.get("elapsed_seconds"),
            "status": (wait.get("status") or {}).get("summary", {}),
            "failure": None if wait.get("ok") else failure_digest(wait),
        },
        "diagnosis": diagnosis["primary"],
        "diagnosis_codes": diagnosis["codes"],
        "diagnoses": diagnosis["all"],
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
    session = payload.get("session_request") or {}
    diagnosis = payload.get("diagnosis") or {}
    evidence = payload.get("evidence_summary") or {}
    support = payload.get("support_bundle_summary") or {}
    observability = payload.get("observability_summary") or {}
    availability = observability.get("availability") or {}
    observed_queue = observability.get("work_queue") or {}
    observed_miner = observability.get("miner") or {}
    observed_inference = observability.get("inference") or {}
    observed_artifacts = observability.get("artifacts") or {}
    artifacts = payload.get("artifacts") or {}
    lines = [
        "# CrowdTensor Remote Demo Acceptance",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Coordinator: `{payload.get('coordinator_url', '')}`",
        f"Miner: `{payload.get('miner_id', '')}`",
        f"Route: `{payload.get('route', '')}`",
        f"Scenario: `{(payload.get('scenario') or {}).get('scenario_id', '')}`",
        "",
        "## Wait Summary",
        "",
        f"- Ready: `{wait.get('ok')}`",
        f"- Attempts: `{wait.get('attempts')}`",
        f"- Elapsed seconds: `{wait.get('elapsed_seconds')}`",
        "",
        "## Session Request",
        "",
        f"- Created: `{session.get('created')}`",
        f"- Schema: `{session.get('schema')}`",
        f"- Task ID: `{session.get('task_id')}`",
        f"- Request count: `{session.get('request_count')}`",
        f"- Scenario: `{session.get('scenario_id')}`",
        f"- Scenario schema: `{session.get('scenario_schema')}`",
        f"- Workload: `{session.get('workload_type')}`",
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
        "## Evidence",
        "",
        f"- Schema: `{evidence.get('schema')}`",
        f"- OK: `{evidence.get('ok')}`",
        f"- Request count: `{evidence.get('request_count')}`",
        f"- Scenario: `{evidence.get('scenario_id')}`",
        f"- Scenario matches: `{evidence.get('scenario_matches')}`",
        f"- Read-only: `{evidence.get('read_only')}`",
        f"- Redaction OK: `{evidence.get('redaction_ok')}`",
        "",
        "## Observability",
        "",
        f"- Schema: `{observability.get('schema')}`",
        f"- Health OK: `{availability.get('health_ok')}`",
        f"- State OK: `{availability.get('state_ok')}`",
        f"- Admin results OK: `{availability.get('admin_results_ok')}`",
        f"- Accepted results: `{observed_queue.get('accepted_results')}`",
        f"- Miner runtime: `{observed_miner.get('runtime')}`",
        f"- Miner backend: `{observed_miner.get('backend')}`",
        f"- Inference requests: `{observed_inference.get('request_count')}`",
        f"- Inference scenario: `{observed_inference.get('scenario_id')}`",
        f"- Scenario matches: `{observed_inference.get('scenario_matches')}`",
        f"- Request trace count: `{observed_inference.get('request_trace_count')}`",
        f"- Requests/sec: `{observed_inference.get('requests_per_second')}`",
        f"- Evidence artifact OK: `{observed_artifacts.get('evidence_ok')}`",
        f"- Support bundle OK: `{observed_artifacts.get('support_bundle_ok')}`",
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
    ])
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
    parser.add_argument("--scenario-id", default=DEFAULT_INFERENCE_SCENARIO_ID)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--artifact-timeout", type=float, default=60.0)
    parser.add_argument("--admin-results-limit", type=int, default=10)
    parser.add_argument("--create-session", action="store_true")
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
    args.scenario_id = normalize_inference_scenario_id(args.scenario_id) or DEFAULT_INFERENCE_SCENARIO_ID
    args.coordinator_url = args.coordinator_url.rstrip("/")
    args.session_task_id = ""
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
