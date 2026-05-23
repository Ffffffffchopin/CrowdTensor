#!/usr/bin/env python3
"""Create and wait for a read-only CrowdTensor inference session."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ADMIN_HEADER = "x-crowdtensor-admin-token"
CLIENT_SCHEMA = "inference_session_client_v1"
SESSION_SCHEMA = "inference_session_request_v1"
WORKLOAD_TYPE = "model_bundle_infer"

DIAGNOSIS_ORDER = [
    "coordinator_unreachable",
    "admin_auth_failed",
    "session_create_failed",
    "session_timeout",
    "validation_failed",
    "request_count_mismatch",
    "session_client_ready",
]
DIAGNOSIS_LIBRARY = {
    "session_client_ready": {
        "severity": "info",
        "summary": "Read-only inference session completed through the admin session API.",
        "next_steps": [
            "Share the JSON output when reporting a successful controlled inference session.",
            "Use remote_demo_acceptance_pack.py when you need the full remote evidence bundle.",
        ],
    },
    "coordinator_unreachable": {
        "severity": "error",
        "summary": "The Coordinator could not be reached.",
        "next_steps": [
            "Verify --coordinator-url and that the Coordinator process is running.",
            "Check local firewall, tunnel, VPN, or reverse proxy settings before retrying.",
        ],
    },
    "admin_auth_failed": {
        "severity": "error",
        "summary": "The Coordinator rejected the admin token.",
        "next_steps": [
            "Pass the same admin token used to start the Coordinator.",
            "Do not share the raw token in logs or issue reports.",
        ],
    },
    "session_create_failed": {
        "severity": "error",
        "summary": "The read-only inference session could not be created.",
        "next_steps": [
            "Confirm the Coordinator exposes POST /admin/inference-sessions.",
            "Retry with --request-count between 1 and 8.",
        ],
    },
    "session_timeout": {
        "severity": "warning",
        "summary": "No accepted result appeared for the created session task before timeout.",
        "next_steps": [
            "Start a Python Miner that supports model_bundle_infer.",
            "Inspect Coordinator task counts and Miner logs for claim or validation failures.",
        ],
    },
    "validation_failed": {
        "severity": "error",
        "summary": "The accepted session result failed validation.",
        "next_steps": [
            "Inspect the safe validation code and reason in the report.",
            "Rerun with a clean state directory if stale task state is suspected.",
        ],
    },
    "request_count_mismatch": {
        "severity": "error",
        "summary": "The session result used a different request count than requested.",
        "next_steps": [
            "Confirm the client and Coordinator agree on --request-count.",
            "Use the task_id-bound admin result query to avoid stale results.",
        ],
    },
}
SECRET_FRAGMENTS = (
    ADMIN_HEADER,
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_result",
    "inference_results",
    "Bearer ",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def diagnosis_payload(code: str, *, observed: dict[str, Any] | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    template = DIAGNOSIS_LIBRARY[code]
    return {
        "primary_code": code,
        "severity": template["severity"],
        "summary": template["summary"],
        "details": details or {},
        "next_steps": list(template["next_steps"]),
        "observed": observed or {},
    }


def sorted_diagnoses(diagnoses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {code: index for index, code in enumerate(DIAGNOSIS_ORDER)}
    return sorted(diagnoses, key=lambda item: order.get(str(item.get("primary_code")), len(order)))


def observe_request(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    admin_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if admin_token:
        headers[ADMIN_HEADER] = admin_token
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return {
                "ok": True,
                "status": response.status,
                "path": path,
                "json": json.loads(raw) if raw else {},
            }
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "path": path, "detail": detail[:300]}
    except (URLError, OSError) as exc:
        return {"ok": False, "status": None, "path": path, "error": type(exc).__name__, "detail": str(exc)[:300]}


def observation_digest(observation: dict[str, Any]) -> dict[str, Any]:
    digest = {
        "ok": bool(observation.get("ok")),
        "status": observation.get("status"),
        "path": observation.get("path"),
    }
    for key in ["error", "detail"]:
        if observation.get(key):
            digest[key] = observation[key]
    return digest


def observation_failure_code(observation: dict[str, Any], *, default: str) -> str:
    if observation.get("status") in {401, 403}:
        return "admin_auth_failed"
    if observation.get("status") is None and observation.get("ok") is False:
        return "coordinator_unreachable"
    return default


def response_payload(observation: dict[str, Any]) -> dict[str, Any]:
    payload = observation.get("json")
    return payload if isinstance(payload, dict) else {}


def admin_results_path(task_id: str, limit: int) -> str:
    query = urlencode({
        "status": "accepted",
        "workload_type": WORKLOAD_TYPE,
        "task_id": task_id,
        "limit": limit,
    })
    return f"/admin/results?{query}"


def safe_validation(validation: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "accepted",
        "code",
        "reason",
        "bundle_id",
        "base_bundle_version",
        "artifact_hash",
        "predicted_token_id",
        "predicted_token",
        "target_token_id",
        "target_token",
        "correct",
        "request_count",
        "correct_count",
        "accuracy",
        "request_trace",
        "request_trace_count",
        "request_trace_truncated",
        "scenario_schema",
        "scenario_id",
        "scenario_description",
        "scenario_request_count",
    ]
    return {field: validation.get(field) for field in fields if field in validation}


def safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = ["elapsed_ms", "request_count", "correct_count", "accuracy", "requests_per_second"]
    return {field: metrics.get(field) for field in fields if field in metrics}


def safe_session(session: dict[str, Any] | None) -> dict[str, Any]:
    session = session or {}
    return {
        "created": bool(session.get("accepted") is True or session.get("created") is True),
        "schema": session.get("schema"),
        "task_id": session.get("task_id"),
        "status": session.get("status"),
        "workload_type": session.get("workload_type"),
        "request_count": session.get("request_count"),
        "scenario_id": session.get("scenario_id"),
        "result_query": session.get("result_query"),
    }


def safe_result_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "present": True,
        "event_index": row.get("event_index"),
        "task_id": row.get("task_id"),
        "status": row.get("status"),
        "accepted": row.get("accepted"),
        "miner_id": row.get("miner_id"),
        "workload_type": row.get("workload_type"),
        "attempt": row.get("attempt"),
        "read_only": not bool(row.get("model_updated")) and not bool(row.get("model_bundle_updated")),
        "validation": safe_validation(row.get("validation") or {}),
        "session_metrics": safe_metrics(row.get("session_metrics") or {}),
    }


def leak_fragments(payload: dict[str, Any]) -> list[str]:
    encoded = json.dumps(payload, sort_keys=True)
    return [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]


def diagnoses_for(
    *,
    session: dict[str, Any],
    row: dict[str, Any] | None,
    expected_request_count: int,
    failure_code: str = "",
    failure_details: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    diagnoses: list[dict[str, Any]] = []
    observed = {
        "task_id": session.get("task_id"),
        "expected_request_count": expected_request_count,
        "row_present": bool(row),
    }
    if failure_code:
        diagnoses.append(diagnosis_payload(failure_code, observed=observed, details=failure_details))
    elif not session.get("task_id"):
        diagnoses.append(diagnosis_payload("session_create_failed", observed=observed))
    elif not row:
        diagnoses.append(diagnosis_payload("session_timeout", observed=observed))
    else:
        validation = row.get("validation") or {}
        if validation.get("code") != "ok":
            diagnoses.append(
                diagnosis_payload(
                    "validation_failed",
                    observed=observed,
                    details=safe_validation(validation),
                )
            )
        if int(validation.get("request_count") or 0) != int(expected_request_count):
            diagnoses.append(
                diagnosis_payload(
                    "request_count_mismatch",
                    observed=observed,
                    details={
                        "expected_request_count": int(expected_request_count),
                        "actual_request_count": validation.get("request_count"),
                    },
                )
            )
    if not diagnoses:
        diagnoses.append(
            diagnosis_payload(
                "session_client_ready",
                observed={**observed, "row_present": True},
                details={"workload_type": WORKLOAD_TYPE},
            )
        )
    return sorted_diagnoses(diagnoses)


def build_report(
    *,
    session: dict[str, Any] | None,
    row: dict[str, Any] | None,
    expected_request_count: int,
    attempts: int,
    elapsed_seconds: float,
    observations: dict[str, Any] | None = None,
    failure_code: str = "",
    failure_details: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    session_summary = safe_session(session)
    row_summary = safe_result_row(row)
    diagnoses = diagnoses_for(
        session=session_summary,
        row=row_summary if row_summary else None,
        expected_request_count=expected_request_count,
        failure_code=failure_code,
        failure_details=failure_details,
    )
    primary = diagnoses[0]
    report = {
        "schema": CLIENT_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "ok": primary.get("primary_code") == "session_client_ready",
        "workload_type": WORKLOAD_TYPE,
        "session": session_summary,
        "result": row_summary,
        "request_count": expected_request_count,
        "scenario_id": session_summary.get("scenario_id"),
        "attempts": attempts,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "diagnosis": primary,
        "diagnoses": diagnoses,
        "diagnosis_codes": [item["primary_code"] for item in diagnoses],
        "observations": observations or {},
        "limitations": [
            "Read-only model_bundle_infer session client; not production LLM serving",
            "Uses Coordinator admin API; no P2P discovery, NAT traversal, GPU pooling, or arbitrary prompts are claimed",
        ],
    }
    leaks = leak_fragments(report)
    if leaks:
        report["ok"] = False
        report["safety_error"] = f"client report leaked secret-like fragments: {', '.join(leaks)}"
    return report


def create_session(args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
    observation = observe_request(
        "POST",
        args.coordinator_url,
        "/admin/inference-sessions",
        {"request_count": args.request_count, "scenario_id": args.scenario_id},
        admin_token=args.admin_token,
        timeout=args.http_timeout,
    )
    if not observation.get("ok"):
        return None, observation, observation_failure_code(observation, default="session_create_failed")
    session = response_payload(observation)
    if (
        session.get("schema") != SESSION_SCHEMA
        or session.get("workload_type") != WORKLOAD_TYPE
        or not session.get("task_id")
        or int(session.get("request_count") or 0) != int(args.request_count)
    ):
        return session, observation, "session_create_failed"
    return session, observation, ""


def wait_for_result(args: argparse.Namespace, task_id: str) -> tuple[dict[str, Any] | None, int, dict[str, Any], str]:
    deadline = time.monotonic() + args.timeout_seconds
    attempts = 0
    last_observation: dict[str, Any] = {}
    path = admin_results_path(task_id, args.admin_results_limit)
    while time.monotonic() <= deadline:
        attempts += 1
        observation = observe_request(
            "GET",
            args.coordinator_url,
            path,
            admin_token=args.admin_token,
            timeout=args.http_timeout,
        )
        last_observation = observation
        if not observation.get("ok"):
            code = observation_failure_code(observation, default="")
            if code:
                return None, attempts, observation, code
        else:
            rows = response_payload(observation).get("results") or []
            if isinstance(rows, list) and rows:
                return rows[0], attempts, observation, ""
        time.sleep(args.poll_interval)
    return None, attempts, last_observation, ""


def run_client(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    observations: dict[str, Any] = {}
    session, create_observation, create_failure = create_session(args)
    observations["session_create"] = observation_digest(create_observation)
    if create_failure:
        return build_report(
            session=session,
            row=None,
            expected_request_count=args.request_count,
            attempts=0,
            elapsed_seconds=time.monotonic() - started,
            observations=observations,
            failure_code=create_failure,
            failure_details=observation_digest(create_observation),
        )
    assert session is not None
    row, attempts, result_observation, result_failure = wait_for_result(args, str(session["task_id"]))
    observations["admin_results"] = observation_digest(result_observation)
    return build_report(
        session=session,
        row=row,
        expected_request_count=args.request_count,
        attempts=attempts,
        elapsed_seconds=time.monotonic() - started,
        observations=observations,
        failure_code=result_failure,
        failure_details=observation_digest(result_observation) if result_failure else None,
    )


def print_human_report(report: dict[str, Any]) -> None:
    session = report.get("session") or {}
    result = report.get("result") or {}
    validation = result.get("validation") or {}
    metrics = result.get("session_metrics") or {}
    diagnosis = report.get("diagnosis") or {}
    print("CrowdTensor inference session client")
    print(f"  ok: {report.get('ok')}")
    print(f"  diagnosis: {diagnosis.get('primary_code')} severity={diagnosis.get('severity')}")
    print(f"  task: {session.get('task_id')} status={session.get('status')} workload={session.get('workload_type')}")
    if result:
        print(
            "  result: "
            f"miner={result.get('miner_id')} "
            f"requests={validation.get('request_count')} "
            f"accuracy={validation.get('accuracy')} "
            f"requests_per_second={metrics.get('requests_per_second')}"
        )
        print(f"  safety: read_only={result.get('read_only')}")
    print("  next:")
    for step in diagnosis.get("next_steps") or []:
        print(f"    - {step}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and wait for a read-only CrowdTensor inference session.")
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--admin-token", required=True)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--admin-results-limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")
    if args.http_timeout <= 0:
        raise SystemExit("--http-timeout must be positive")
    args.coordinator_url = args.coordinator_url.rstrip("/")
    return args


def main() -> None:
    report = run_client(parse_args())
    if report.get("schema") != CLIENT_SCHEMA:
        raise SystemExit("internal error: invalid client report schema")
    if "--json" in sys.argv:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human_report(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
