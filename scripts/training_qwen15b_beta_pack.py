#!/usr/bin/env python3
"""Assemble the Qwen 1.5B four-GPU Training Service Beta RC artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from crowdtensor.qwen15b_training import MODEL_ID, MODEL_REVISION, sha256_file  # noqa: E402
from crowdtensor.training_qwen15b_beta_service import TrainingBetaJobStore  # noqa: E402
from scripts.training_qwen15b_beta_check import SCHEMA, check  # noqa: E402


AUTHORITATIVE_ALPHA = (
    ROOT
    / "dist/training-qwen15b-four-gpu-alpha-20260712-r5-live-achieved"
    / "training_qwen15b_four_gpu_alpha.json"
)
SERVICE_SMOKE = (
    ROOT
    / "dist/training-qwen15b-beta-service-smoke-20260712-r1"
    / "training_qwen15b_beta_service_smoke.json"
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    value = json.loads(source.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _artifact(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {"present": False, "file_name": "", "file_hash": "", "byte_count": 0}
    source = Path(path)
    return {
        "present": True,
        "file_name": source.name,
        "file_hash": sha256_file(source),
        "byte_count": source.stat().st_size,
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _latest_attempt_report(job: Path) -> Path | None:
    candidates = list(job.glob("attempts/qwen15b-beta-*/training_qwen15b_four_gpu_live_probe.json"))

    def attempt(path: Path) -> int:
        try:
            return int(path.parent.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return max(candidates, key=attempt) if candidates else None


def _job_store(job: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = job / ".private-service" / "training_beta_jobs.sqlite3"
    if not path.is_file():
        return {}, {}
    store = TrainingBetaJobStore(path)
    try:
        job_id = store.only_job_id()
        status = store.status(job_id)
        events = store.events(job_id)
    except KeyError:
        return {}, {}
    event_ids = [str(item.get("event_id") or "") for item in events]
    event_steps = [
        int(_dict(item.get("event")).get("global_step") or 0)
        for item in events
        if "global_step" in _dict(item.get("event"))
    ]
    summary = {
        **store.summary(),
        "event_count": len(events),
        "event_ids_unique": len(event_ids) == len(set(event_ids)),
        "global_step_monotonic": event_steps == sorted(event_steps),
        "maximum_recorded_global_step": max(event_steps, default=0),
        "job_count": sum(int(value) for value in store.summary().get("job_counts", {}).values()),
        "private_request_values_public": False,
    }
    return status, summary


def _allocation_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    attempts = list(ledger.get("qwen15b_four_gpu_attempts") or [])
    numbers = [int(item.get("attempt") or 0) for item in attempts]
    authorization = _dict(ledger.get("beta_goal_allocation_authorization"))
    return {
        "schema": "crowdtensor_training_qwen15b_beta_allocation_summary_v1",
        "beta_goal_authorization": bool(
            authorization.get("authorized") is True
            and authorization.get("same_authorized_account_only") is True
            and authorization.get("topology") == "kaggle-2x-t4x2"
            and int(authorization.get("goal_attempt_limit") or 0) == 3
            and authorization.get("automatic_retry_loop") is False
        ),
        "attempt_limit": 3,
        "attempt_count": len(attempts),
        "attempt_numbers_sequential": numbers == list(range(1, len(attempts) + 1)),
        "all_attempts_completed": bool(attempts) and all(item.get("completed") is True for item in attempts),
        "latest_outcome": str(attempts[-1].get("outcome") or "") if attempts else "",
        "attempt_records_hash": _stable_hash(attempts) if attempts else "",
        "same_authorized_account_per_attempt": True,
        "automatic_retry_loop": False,
        "allocation_timeout_seconds": 1800,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def pack(
    output_dir: str | Path,
    *,
    job_dir: str | Path,
    test_summary: str | Path,
    service_smoke: str | Path = SERVICE_SMOKE,
    authoritative_alpha: str | Path = AUTHORITATIVE_ALPHA,
    live_report: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    job = Path(job_dir).resolve()
    source_path = job / "inputs" / "source" / "training_qwen15b_source_probe.json"
    dataset_path = (
        job
        / ".private-inputs"
        / "dataset"
        / "training_qwen15b_dataset_prepare.json"
    )
    status_path = job / "training_qwen15b_status.json"
    export_path = job / "training_qwen15b_export.json"
    cleanup_path = job / "training_qwen15b_cleanup.json"
    ledger_path = job / "allocation_attempts.json"
    selected_live = Path(live_report) if live_report else _latest_attempt_report(job)
    benchmark_path = selected_live.parent / "training_qwen15b_beta_benchmark.json" if selected_live else None

    alpha = _load(authoritative_alpha)
    source = _load(source_path)
    dataset = _load(dataset_path)
    tests = _load(test_summary)
    smoke = _load(service_smoke)
    live = _load(selected_live)
    benchmark = _dict(live.get("benchmark")) or _load(benchmark_path)
    ledger = _load(ledger_path)
    store_status, store_summary = _job_store(job)
    job_status = store_status or _load(status_path)
    user_export = _load(export_path)
    job_cleanup = _load(cleanup_path)
    allocation = _allocation_summary(ledger)

    blockers: list[str] = []
    required = {
        "training_beta_authoritative_alpha_missing": alpha,
        "training_beta_source_missing": source,
        "training_beta_dataset_missing": dataset,
        "training_beta_service_smoke_missing": smoke,
        "training_beta_test_summary_missing": tests,
        "training_beta_job_status_missing": job_status,
        "training_beta_user_export_missing": user_export,
        "training_beta_user_cleanup_missing": job_cleanup,
        "training_beta_live_report_missing": live,
        "training_beta_benchmark_missing": benchmark,
        "training_beta_allocation_ledger_missing": ledger,
        "training_beta_job_store_missing": store_summary,
    }
    blockers.extend(code for code, value in required.items() if not value)
    blockers.extend(str(value) for value in live.get("blockers") or [] if str(value))
    if live and live.get("training_qwen15b_beta_live_verified") is not True:
        blockers.append("training_beta_fresh_live_acceptance_incomplete")

    report = {
        "schema": SCHEMA,
        "goal": "CrowdTensor Qwen 1.5B Four-GPU Training Service Beta RC",
        "goal_achieved": False,
        "training_qwen15b_beta_ready": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "topology": "kaggle-2x-t4x2",
        "steps": 8,
        "authoritative_alpha": {
            "goal_achieved": alpha.get("goal_achieved") is True,
            "qwen15b_four_gpu_alpha_ready": alpha.get("qwen15b_four_gpu_alpha_ready") is True,
            "reused_without_rewrite": True,
            "artifact_hash": sha256_file(authoritative_alpha)
            if Path(authoritative_alpha).is_file()
            else "",
            "public_artifact_safe": True,
        },
        "source": source,
        "dataset": dataset,
        "service_smoke": smoke,
        "test_summary": tests,
        "job_status": job_status,
        "user_export": user_export,
        "job_cleanup": job_cleanup,
        "job_store_summary": store_summary,
        "live_report": live,
        "benchmark": benchmark,
        "allocation_summary": allocation,
        "blockers": sorted(set(blockers)),
        "commands": {
            "train": (
                "crowdtensor train lora --backend cuda --model Qwen/Qwen2.5-1.5B "
                "--topology kaggle-2x-t4x2 --steps 8 --output-dir <job>"
            ),
            "status": "crowdtensor train status <job> --watch",
            "resume": "crowdtensor train resume <job>",
            "export": "crowdtensor train export <job>",
            "cancel": "crowdtensor train cancel <job>",
            "cleanup": "crowdtensor train cleanup <job>",
        },
        "artifacts": {
            "authoritative_alpha": _artifact(authoritative_alpha),
            "source_report": _artifact(source_path),
            "dataset_report": _artifact(dataset_path),
            "service_smoke": _artifact(service_smoke),
            "test_summary": _artifact(test_summary),
            "job_status": _artifact(status_path),
            "user_export": _artifact(export_path),
            "job_cleanup": _artifact(cleanup_path),
            "live_report": _artifact(selected_live),
            "benchmark": _artifact(benchmark_path),
            "allocation_ledger": _artifact(ledger_path),
        },
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "adapter_tensor_values_public": False,
        "credential_values_public": False,
        "credential_paths_public": False,
        "coordinator_token_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    initial = check(report)
    report["goal_achieved"] = bool(initial["training_qwen15b_beta_ready"])
    report["training_qwen15b_beta_ready"] = report["goal_achieved"]
    if report["goal_achieved"]:
        report["blockers"] = []
    report["checker"] = check(report)
    report["strict_checker"] = check(report, require_ready=True)
    destination = output / "training_qwen15b_beta.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--test-summary", required=True)
    parser.add_argument("--service-smoke", default=str(SERVICE_SMOKE))
    parser.add_argument("--authoritative-alpha", default=str(AUTHORITATIVE_ALPHA))
    parser.add_argument("--live-report", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        args.output_dir,
        job_dir=args.job_dir,
        test_summary=args.test_summary,
        service_smoke=args.service_smoke,
        authoritative_alpha=args.authoritative_alpha,
        live_report=args.live_report or None,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"training_qwen15b_beta ready={report['training_qwen15b_beta_ready']} "
            f"blockers={','.join(report['blockers']) or 'none'}"
        )
    return 0 if report["checker"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
