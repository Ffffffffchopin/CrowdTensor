#!/usr/bin/env python3
"""CI-safe check for the local multi-Miner scenario sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_REPORT = "/tmp/crowdtensor_multi_miner_sweep.json"
DEFAULT_REQUEUE_REPORT = "/tmp/crowdtensor_multi_miner_requeue.json"
EXECUTION_SEQUENTIAL = "sequential"
EXECUTION_CONCURRENT = "concurrent"
EXECUTION_MODES = {EXECUTION_SEQUENTIAL, EXECUTION_CONCURRENT}
FAILURE_NONE = "none"
FAILURE_KILL_AFTER_CLAIM = "kill-after-claim"
FAILURE_MODES = {FAILURE_NONE, FAILURE_KILL_AFTER_CLAIM}


def assert_no_secret_field(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "lease_token" and child not in {"", None, "<redacted>"}:
                raise SystemExit(f"multi-Miner sweep leaked secret-like material: {key}")
            if key in {"idempotency_key", "result_idempotency_key_hash", "result_lease_token_hash"}:
                raise SystemExit(f"multi-Miner sweep leaked secret-like material: {key}")
            assert_no_secret_field(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_secret_field(child)


def validate_report(
    report: dict[str, Any],
    *,
    expected_scenarios: list[str],
    request_count: int,
    execution_mode: str,
    failure_mode: str,
) -> None:
    if report.get("schema") != "multi_miner_scenario_sweep_v1" or report.get("ok") is not True:
        raise SystemExit(f"unexpected multi-Miner sweep report: {json.dumps(report, sort_keys=True)}")
    if report.get("execution_mode") != execution_mode:
        raise SystemExit(f"execution mode mismatch: {report.get('execution_mode')} expected={execution_mode}")
    if report.get("failure_mode", FAILURE_NONE) != failure_mode:
        raise SystemExit(f"failure mode mismatch: {report.get('failure_mode')} expected={failure_mode}")
    sessions = report.get("sessions") or []
    if len(sessions) != len(expected_scenarios):
        raise SystemExit(f"expected {len(expected_scenarios)} sessions, got {len(sessions)}")
    observed_scenarios = [session.get("scenario_id") for session in sessions]
    if observed_scenarios != expected_scenarios:
        raise SystemExit(f"scenario order mismatch: {observed_scenarios} expected={expected_scenarios}")
    for session in sessions:
        if session.get("ok") is not True:
            raise SystemExit(f"session failed: {session}")
        if session.get("workload_type") != "model_bundle_infer":
            raise SystemExit(f"unexpected workload: {session}")
        if session.get("scenario_matches") is not True:
            raise SystemExit(f"scenario mismatch: {session}")
        if int(session.get("request_count") or 0) != request_count:
            raise SystemExit(f"request_count mismatch: {session}")
        if float(session.get("requests_per_second") or 0.0) <= 0.0:
            raise SystemExit(f"invalid throughput: {session}")
    distribution = report.get("distribution") or {}
    if distribution.get("all_expected_miners_seen") is not True:
        raise SystemExit(f"miner distribution failed: {distribution}")
    if int(distribution.get("distinct_miner_count") or 0) != len(expected_scenarios):
        raise SystemExit(f"distinct miner count mismatch: {distribution}")
    lease_summary = report.get("lease_summary") or {}
    if lease_summary.get("all_task_ids_unique") is not True:
        raise SystemExit(f"task ids were not unique: {lease_summary}")
    if lease_summary.get("one_result_per_task") is not True:
        raise SystemExit(f"expected one accepted ledger row per task: {lease_summary}")
    if lease_summary.get("no_queued_or_leased_remaining") is not True:
        raise SystemExit(f"queued or leased tasks remained after sweep: {lease_summary}")
    if int(lease_summary.get("accepted_ledger_rows") or 0) != len(expected_scenarios):
        raise SystemExit(f"accepted ledger row count mismatch: {lease_summary}")
    process_summary = report.get("process_summary") or {}
    if process_summary.get("all_processes_ok") is not True:
        raise SystemExit(f"miner process summary failed: {process_summary}")
    expected_started = len(expected_scenarios) + (1 if failure_mode == FAILURE_KILL_AFTER_CLAIM else 0)
    if execution_mode == EXECUTION_CONCURRENT:
        if int(process_summary.get("started") or 0) != expected_started:
            raise SystemExit(f"concurrent process start count mismatch: {process_summary}")
        if int(process_summary.get("ok") or 0) != len(expected_scenarios):
            raise SystemExit(f"concurrent process ok count mismatch: {process_summary}")
    requeue_summary = report.get("requeue_summary") or {}
    if failure_mode == FAILURE_KILL_AFTER_CLAIM:
        if requeue_summary.get("enabled") is not True:
            raise SystemExit(f"missing requeue summary: {requeue_summary}")
        if requeue_summary.get("lease_expired") is not True:
            raise SystemExit(f"lease requeue was not observed: {requeue_summary}")
        if requeue_summary.get("rescued_result") is not True:
            raise SystemExit(f"rescued result missing: {requeue_summary}")
        if requeue_summary.get("victim_result_accepted") is True:
            raise SystemExit(f"victim unexpectedly accepted result: {requeue_summary}")
        if not requeue_summary.get("requeued_task_id"):
            raise SystemExit(f"missing requeued task id: {requeue_summary}")
    elif requeue_summary.get("enabled"):
        raise SystemExit(f"unexpected requeue summary in failure_mode=none: {requeue_summary}")
    safety = report.get("safety") or {}
    if not safety.get("read_only") or not safety.get("redaction_ok") or not safety.get("registry_hashed"):
        raise SystemExit(f"unsafe multi-Miner sweep report: {safety}")
    observability = report.get("observability_summary") or {}
    if observability.get("schema") != "multi_miner_scenario_sweep_observability_v1":
        raise SystemExit(f"missing sweep observability summary: {observability}")
    expected_code = (
        "multi_miner_requeue_ready"
        if failure_mode == FAILURE_KILL_AFTER_CLAIM
        else "multi_miner_concurrent_ready"
        if execution_mode == EXECUTION_CONCURRENT
        else "multi_miner_sweep_ready"
    )
    if expected_code not in report.get("diagnosis_codes", []):
        raise SystemExit(f"missing success diagnosis code: {report.get('diagnosis_codes')}")
    encoded = json.dumps(report, sort_keys=True)
    for fragment in [
        "CROWDTENSOR_MINER_TOKEN",
        "multi-miner-sweep-token",
        "inference_result",
        "inference_results",
    ]:
        if fragment in encoded:
            raise SystemExit(f"multi-Miner sweep output leaked secret-like material: {fragment}")
    assert_no_secret_field(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local multi-Miner scenario sweep check.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8916)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-ids", default="route-baseline,gradient-safety,mixed-prompts")
    parser.add_argument("--execution-mode", choices=sorted(EXECUTION_MODES), default=EXECUTION_CONCURRENT)
    parser.add_argument("--failure-mode", choices=sorted(FAILURE_MODES), default=FAILURE_NONE)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--sweep-timeout", type=float, default=120.0)
    parser.add_argument("--json-out", default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.failure_mode == FAILURE_KILL_AFTER_CLAIM and args.json_out == DEFAULT_REPORT:
        args.json_out = DEFAULT_REQUEUE_REPORT
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = args.state_dir
    if not state_dir:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_multi_miner_sweep_check_")
        state_dir = temp_dir.name
    command = [
        sys.executable,
        str(ROOT / "scripts" / "multi_miner_scenario_sweep.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--state-dir",
        state_dir,
        "--request-count",
        str(args.request_count),
        "--scenario-ids",
        args.scenario_ids,
        "--execution-mode",
        args.execution_mode,
        "--failure-mode",
        args.failure_mode,
        "--startup-timeout",
        str(args.startup_timeout),
        "--miner-timeout",
        str(args.miner_timeout),
        "--json-out",
        args.json_out,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=args.sweep_timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "multi_miner_scenario_sweep.py failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("multi_miner_scenario_sweep.py emitted no JSON")
        report = json.loads(lines[-1])
        expected_scenarios = [item.strip() for item in args.scenario_ids.split(",") if item.strip()]
        validate_report(
            report,
            expected_scenarios=expected_scenarios,
            request_count=args.request_count,
            execution_mode=args.execution_mode,
            failure_mode=args.failure_mode,
        )
        report_path = Path(args.json_out)
        if not report_path.is_file():
            raise SystemExit(f"sweep did not write report: {report_path}")
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        validate_report(
            persisted,
            expected_scenarios=expected_scenarios,
            request_count=args.request_count,
            execution_mode=args.execution_mode,
            failure_mode=args.failure_mode,
        )
        print(json.dumps({
            "ok": True,
            "schema": report["schema"],
            "execution_mode": args.execution_mode,
            "failure_mode": args.failure_mode,
            "route": "local_multi_miner_model_bundle_infer",
            "workload_type": "model_bundle_infer",
            "request_count": args.request_count,
            "scenario_ids": expected_scenarios,
            "distinct_miner_count": (report.get("distribution") or {}).get("distinct_miner_count"),
            "diagnosis_codes": report.get("diagnosis_codes", []),
            "report": str(report_path),
        }, sort_keys=True))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
