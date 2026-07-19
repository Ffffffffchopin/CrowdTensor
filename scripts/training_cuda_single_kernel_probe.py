#!/usr/bin/env python3
"""Run one bounded Kaggle T4x2 allocation attempt for the CUDA training gate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training_cuda_kaggle_common import (  # noqa: E402
    authenticated_owner,
    delete_succeeded_or_absent,
    extract_kernel_ref,
    kaggle_env,
    public_safety_errors,
    push_accepted,
    run_command,
    safe_slug,
    status_class,
    utc_now,
)
from scripts.training_cuda_single_kernel_package import build_package  # noqa: E402
from crowdtensor.training_contract import sha256_file  # noqa: E402
from crowdtensor.training_allocation_budget import require_attempt_limit  # noqa: E402


SCHEMA = "crowdtensor_cuda_single_kernel_live_probe_v1"
WORKER_REPORT = "training_cuda_single_kernel_gate.json"
CHECKPOINT_BUNDLE = "training_cuda_single_kernel_checkpoint_bundle.zip"
OUTPUT_PATTERN = r"training_cuda_single_kernel_(gate\.json|checkpoint_bundle\.zip)"
TERMINAL = {"complete", "failed"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def reserve_attempt(ledger_path: Path, *, limit: int) -> int:
    ledger = _load_json(ledger_path)
    limit = require_attempt_limit(ledger, kind="single_kernel", requested_limit=limit)
    attempts = list(ledger.get("single_kernel_attempts") or [])
    if len(attempts) >= int(limit):
        raise RuntimeError("single_kernel_allocation_attempt_limit_reached")
    attempt = len(attempts) + 1
    attempts.append({"attempt": attempt, "started_at": utc_now(), "completed": False})
    ledger.update(
        {
            "schema": "crowdtensor_cuda_training_allocation_attempts_v1",
            "single_kernel_attempts": attempts,
            "single_kernel_attempt_limit": int(limit),
        }
    )
    _write_json(ledger_path, ledger)
    return attempt


def finish_attempt(ledger_path: Path, *, attempt: int, outcome: str) -> None:
    ledger = _load_json(ledger_path)
    attempts = list(ledger.get("single_kernel_attempts") or [])
    for record in attempts:
        if int(record.get("attempt") or 0) == int(attempt):
            record.update({"completed": True, "finished_at": utc_now(), "outcome": str(outcome)})
    ledger["single_kernel_attempts"] = attempts
    _write_json(ledger_path, ledger)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-token-file", required=True)
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument("--attempt-ledger", default="dist/training-cuda-two-node-work/allocation_attempts.json")
    parser.add_argument("--attempt-limit", type=int, default=2)
    parser.add_argument("--allocation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.attempt_limit < 1 or args.attempt_limit > 3:
        parser.error("--attempt-limit must be in [1, 3]")
    if args.allocation_timeout_seconds <= 0 or args.allocation_timeout_seconds > 1800:
        parser.error("--allocation-timeout-seconds must be in (0, 1800]")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(args.attempt_ledger).resolve()
    report_path = output / "training_cuda_single_kernel_live_probe.json"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "evidence_ready": False,
        "single_kernel_t4x2_verified": False,
        "started_at": utc_now(),
        "attempt": 0,
        "allocation_started": False,
        "push_attempted": False,
        "attempt_limit": int(args.attempt_limit),
        "allocation_timeout_seconds": float(args.allocation_timeout_seconds),
        "worker_report": {},
        "status_observations": [],
        "blockers": [],
        "cleanup": {
            "kernel_delete_attempted": False,
            "kernel_deleted": False,
            "private_package_removed": False,
            "checkpoint_preserved": True,
            "private_cleanup_state_removed": True,
        },
        "private_token_input_used": True,
        "token_values_public": False,
        "token_paths_public": False,
        "private_kernel": True,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    accepted_ref = ""
    package_root = output / ".private-single-kernel-package"
    private_cleanup_dir = output / ".private-cleanup"
    cleanup_refs: list[str] = []
    attempt = 0
    outcome = "not_started"
    try:
        with kaggle_env(args.raw_token_file, username_hint=args.raw_token_username) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("authorized_kaggle_account_authentication_failed")
            suffix = str(int(time.time()))[-8:]
            slug = safe_slug(f"ct-cuda-train-single-{suffix}")
            package_report = build_package(
                package_root,
                owner=owner,
                slug=slug,
                total_steps=4,
                interrupt_after_step=2,
            )
            expected_ref = str(package_report["kernel_ref"])
            attempt = reserve_attempt(ledger_path, limit=args.attempt_limit)
            report["attempt"] = attempt
            report["allocation_started"] = True
            report["push_attempted"] = True
            cleanup_refs = [expected_ref]
            _write_json(
                private_cleanup_dir / "active_resources.json",
                {
                    "schema": "crowdtensor_cuda_training_private_cleanup_resources_v1",
                    "provider": "kaggle",
                    "kernel_refs": cleanup_refs,
                    "push_attempted": True,
                    "credentials_embedded": False,
                },
            )
            _write_json(report_path, report)
            push = run_command(
                [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(package_report["package_dir"]),
                    "-t",
                    str(int(args.allocation_timeout_seconds)),
                    "--accelerator",
                    "NvidiaTeslaT4",
                ],
                env=env,
                timeout=float(args.push_timeout_seconds),
            )
            report["push"] = push
            if not push_accepted(push):
                output_text = str(push.get("output_tail") or "").lower()
                blocker = "kaggle_gpu_kernel_push_rejected"
                if any(word in output_text for word in ("quota", "maximum", "limit", "session")):
                    blocker = "kaggle_gpu_quota_or_session_unavailable"
                raise RuntimeError(blocker)
            accepted_ref = extract_kernel_ref(str(push.get("output_tail") or ""), expected_ref)
            report["kernel_ref_hash"] = "sha256:" + __import__("hashlib").sha256(
                accepted_ref.encode("utf-8")
            ).hexdigest()
            deadline = time.monotonic() + float(args.allocation_timeout_seconds)
            terminal_class = "unknown"
            while time.monotonic() < deadline:
                status = run_command(
                    ["kaggle", "kernels", "status", accepted_ref],
                    env=env,
                    timeout=float(args.status_timeout_seconds),
                )
                terminal_class = status_class(str(status.get("output_tail") or ""))
                report["status_observations"].append(
                    {
                        "observed_at": utc_now(),
                        "status_class": terminal_class,
                        "status_command_ok": bool(status.get("ok")),
                    }
                )
                _write_json(report_path, report)
                if terminal_class in TERMINAL:
                    break
                time.sleep(max(5.0, float(args.poll_interval_seconds)))
            if terminal_class not in TERMINAL:
                raise RuntimeError("single_kernel_allocation_wait_timeout")
            stage_output = output / ".private-kaggle-output"
            output_step = run_command(
                [
                    "kaggle",
                    "kernels",
                    "output",
                    accepted_ref,
                    "-p",
                    str(stage_output),
                    "--force",
                    "--file-pattern",
                    OUTPUT_PATTERN,
                ],
                env=env,
                timeout=float(args.output_timeout_seconds),
            )
            report["output"] = output_step
            worker_path = stage_output / WORKER_REPORT
            worker = _load_json(worker_path)
            report["worker_report"] = worker
            worker_bundle = dict(worker.get("checkpoint_bundle") or {})
            bundle_source = stage_output / CHECKPOINT_BUNDLE
            checkpoint_summary = {
                "preserved": False,
                "worker_hash_match": False,
                "file_hash": "",
                "byte_count": 0,
                "file_count": int(worker_bundle.get("file_count") or 0),
                "contains_baseline_and_resumed_checkpoints": worker_bundle.get(
                    "contains_baseline_and_resumed_checkpoints"
                )
                is True,
                "checkpoint_values_public": False,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
            if bundle_source.is_file():
                checkpoint_dir = output / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_destination = checkpoint_dir / CHECKPOINT_BUNDLE
                checkpoint_destination.unlink(missing_ok=True)
                shutil.move(str(bundle_source), checkpoint_destination)
                actual_hash = sha256_file(checkpoint_destination)
                checkpoint_summary.update(
                    {
                        "preserved": True,
                        "worker_hash_match": actual_hash == worker_bundle.get("file_hash"),
                        "file_hash": actual_hash,
                        "byte_count": checkpoint_destination.stat().st_size,
                    }
                )
            report["checkpoint_bundle"] = checkpoint_summary
            report["cleanup"]["checkpoint_preserved"] = bool(
                not worker_bundle.get("present") or checkpoint_summary["preserved"]
            )
            report["single_kernel_t4x2_verified"] = bool(
                terminal_class == "complete"
                and worker.get("ok") is True
                and worker.get("single_kernel_t4x2_verified") is True
                and worker.get("gpu_live_verified") is True
                and int(worker.get("cuda_device_count") or 0) >= 2
                and worker.get("checkpoint_resume_verified") is True
                and worker_bundle.get("present") is True
                and checkpoint_summary["preserved"] is True
                and checkpoint_summary["worker_hash_match"] is True
                and checkpoint_summary["contains_baseline_and_resumed_checkpoints"] is True
            )
            report["ok"] = report["single_kernel_t4x2_verified"]
            if not report["ok"]:
                report["blockers"].append("single_kernel_t4x2_worker_gate_failed")
            outcome = "verified" if report["ok"] else "worker_gate_failed"
            shutil.rmtree(stage_output, ignore_errors=True)
    except Exception as exc:
        code = str(exc)[:160] or type(exc).__name__
        report["blockers"].append(code)
        report["error_class"] = type(exc).__name__
        outcome = code
    finally:
        if cleanup_refs:
            report["cleanup"]["kernel_delete_attempted"] = True
            try:
                with kaggle_env(args.raw_token_file, username_hint=args.raw_token_username) as cleanup_env:
                    delete_steps = [
                        run_command(
                            ["kaggle", "kernels", "delete", ref, "-y"],
                            env=cleanup_env,
                            timeout=float(args.delete_timeout_seconds),
                        )
                        for ref in cleanup_refs
                    ]
                report["cleanup"]["kernel_deleted"] = all(
                    delete_succeeded_or_absent(step) for step in delete_steps
                )
                report["cleanup"]["delete"] = delete_steps[0] if delete_steps else {}
            except Exception:
                report["cleanup"]["kernel_deleted"] = False
        else:
            report["cleanup"]["kernel_deleted"] = True
        shutil.rmtree(package_root, ignore_errors=True)
        shutil.rmtree(output / ".private-kaggle-output", ignore_errors=True)
        if report["cleanup"]["kernel_deleted"]:
            shutil.rmtree(private_cleanup_dir, ignore_errors=True)
        report["cleanup"]["private_cleanup_state_removed"] = not private_cleanup_dir.exists()
        report["cleanup"]["private_package_removed"] = not package_root.exists()
        report["finished_at"] = utc_now()
        report["blockers"] = sorted(set(report.get("blockers") or []))
        safety_errors = public_safety_errors(report)
        report["public_artifact_safe"] = not safety_errors
        if safety_errors:
            report["safety_errors"] = safety_errors
        cleanup_ok = bool(
            report["cleanup"]["kernel_deleted"]
            and report["cleanup"]["private_package_removed"]
            and report["cleanup"]["checkpoint_preserved"]
            and report["cleanup"]["private_cleanup_state_removed"]
        )
        report["ok"] = bool(report.get("ok") and cleanup_ok and report["public_artifact_safe"])
        report["evidence_ready"] = bool(cleanup_ok and report["public_artifact_safe"])
        _write_json(report_path, report)
        if attempt:
            finish_attempt(ledger_path, attempt=attempt, outcome=outcome)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                f"training_cuda_single_kernel_probe ok={report['ok']} "
                f"attempt={report['attempt']} blockers={','.join(report['blockers']) or 'none'}"
            )
    return 0 if report["ok"] else (1 if report["evidence_ready"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
