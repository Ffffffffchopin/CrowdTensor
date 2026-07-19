#!/usr/bin/env python3
"""Run one bounded Kaggle TPU stage-2 diagnostic without consuming a live gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from scripts.training_cuda_kaggle_common import (
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
from scripts.training_heterogeneous_tpu_beta_live_probe import (
    _finish_attempt,
    classify_tpu_push,
    collect_kernel_output_with_retry,
    reserve_acquisition_window,
)
from scripts.training_heterogeneous_tpu_stage_diagnostic_package import (
    REPORT_NAME,
    build_package,
)


SCHEMA = "crowdtensor_heterogeneous_training_tpu_stage_diagnostic_live_v1"
REPORT_FILE = "training_heterogeneous_tpu_stage_diagnostic_live_probe.json"
TERMINAL = {"complete", "failed"}


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _base_report() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "live_probe_performed": False,
        "model_id": "Qwen/Qwen2.5-7B",
        "model_revision": "d149729398750b98c0af14eb82c78cfe92750796",
        "stage_id": 2,
        "layer_start": 14,
        "layer_end": 20,
        "requested_accelerator": "tpuV5e8",
        "diagnostic_only": True,
        "full_training_gate_evidence": False,
        "same_job_three_accelerator_evidence": False,
        "live_gate_ledger_modified": False,
        "queue_observations": [],
        "kernel_output_collection": {},
        "kernel_report": {},
        "blockers": [],
        "cleanup": {
            "remote_kernel_deleted": False,
            "temporary_private_package_removed": False,
            "live_resources_left_running": True,
        },
        "credential_values_public": False,
        "credential_paths_public": False,
        "account_labels_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def _finish_report(path: Path, report: dict[str, Any]) -> None:
    report["blockers"] = sorted(
        {str(item) for item in report.get("blockers") or [] if str(item)}
    )
    report["public_artifact_safe"] = not public_safety_errors(report)
    if not report["public_artifact_safe"]:
        report["blockers"].append("heterogeneous_tpu_stage_diagnostic_public_safety_failed")
        report["blockers"] = sorted(set(report["blockers"]))
    report.pop("content_hash", None)
    report["content_hash"] = _stable_hash(report)
    _write_json(path, report)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / REPORT_FILE
    private_root = output / ".private-runtime"
    report = _base_report()
    _finish_report(report_path, report)
    kernel_ref = ""
    acquisition_attempt = 0
    acquisition_finished = False
    acquisition_outcome = "tpu_stage_diagnostic_failed"
    package_removed = False
    remote_deleted = False
    try:
        with kaggle_env(
            args.tpu_token_file, username_hint=args.tpu_token_username
        ) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("heterogeneous_kaggle_authentication_failed")
            suffix = str(int(time.time()))[-9:]
            package = build_package(
                private_root / "package",
                owner=owner,
                slug=safe_slug(f"ct-tpu-stage2-diagnostic-{suffix}"),
                hf_token=str(os.environ.get(args.hf_token_env) or ""),
            )
            acquisition_attempt, acquisition_remaining = reserve_acquisition_window(
                Path(args.acquisition_ledger).resolve(),
                limit=int(args.acquisition_window_limit),
                reuse_attempt=int(args.reuse_acquisition_window),
                window_seconds=float(args.acquisition_window_seconds),
            )
            report["acquisition"] = {
                "window_attempt": acquisition_attempt,
                "submission_kind": "bounded_tpu_stage_diagnostic",
                "remaining_window_seconds_at_reservation": round(
                    acquisition_remaining, 3
                ),
                "live_gate_consumed": False,
                "public_artifact_safe": True,
            }
            push = run_command(
                [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(package["package_dir"]),
                    "-t",
                    str(int(args.kernel_timeout_seconds)),
                    "--accelerator",
                    "tpuV5e8",
                ],
                env=env,
                timeout=float(args.push_timeout_seconds),
            )
            push_outcome = classify_tpu_push(push)
            report["push_summary"] = {
                "outcome": push_outcome,
                "returncode": push.get("returncode"),
                "timed_out": push.get("timed_out") is True,
                "duration_seconds": float(push.get("duration_seconds") or 0.0),
                "public_artifact_safe": True,
            }
            _finish_report(report_path, report)
            shutil.rmtree(private_root / "package", ignore_errors=True)
            package_removed = not (private_root / "package").exists()
            if not push_accepted(push):
                acquisition_outcome = push_outcome
                raise RuntimeError(push_outcome)
            kernel_ref = extract_kernel_ref(
                str(push.get("output_tail") or ""), str(package["kernel_ref"])
            )
            report["live_probe_performed"] = True
            report["kernel_ref_hash"] = _stable_hash(kernel_ref)
            queue_deadline = time.monotonic() + float(acquisition_remaining)
            running_observed = False
            while time.monotonic() < queue_deadline:
                status = run_command(
                    ["kaggle", "kernels", "status", kernel_ref],
                    env=env,
                    timeout=float(args.status_timeout_seconds),
                )
                state = status_class(str(status.get("output_tail") or ""))
                report["queue_observations"] = [
                    *(report.get("queue_observations") or []),
                    {"observed_at": utc_now(), "state": state},
                ][-1440:]
                _finish_report(report_path, report)
                if state == "running":
                    running_observed = True
                    acquisition_outcome = "tpu_running"
                    _finish_attempt(
                        Path(args.acquisition_ledger).resolve(),
                        attempt=acquisition_attempt,
                        outcome=acquisition_outcome,
                    )
                    acquisition_finished = True
                    break
                if state in TERMINAL:
                    acquisition_outcome = "tpu_terminal_before_running"
                    raise RuntimeError(acquisition_outcome)
                time.sleep(max(10.0, float(args.poll_interval_seconds)))
            if not running_observed:
                acquisition_outcome = "kaggle_tpu_queue_window_exhausted"
                raise RuntimeError(acquisition_outcome)

            runtime_deadline = time.monotonic() + float(args.kernel_timeout_seconds)
            terminal_state = ""
            while time.monotonic() < runtime_deadline:
                status = run_command(
                    ["kaggle", "kernels", "status", kernel_ref],
                    env=env,
                    timeout=float(args.status_timeout_seconds),
                )
                state = status_class(str(status.get("output_tail") or ""))
                report["runtime_last_observation"] = {
                    "observed_at": utc_now(),
                    "state": state,
                }
                _finish_report(report_path, report)
                if state in TERMINAL:
                    terminal_state = state
                    break
                time.sleep(max(10.0, float(args.poll_interval_seconds)))
            if not terminal_state:
                report["blockers"].append(
                    "heterogeneous_tpu_stage_diagnostic_runtime_timeout"
                )
            report["terminal_state"] = terminal_state
            kernel, collection = collect_kernel_output_with_retry(
                ref=kernel_ref,
                env=env,
                destination=private_root / "output",
                filename=REPORT_NAME,
                file_pattern=re.escape(REPORT_NAME),
                timeout_seconds=float(args.output_timeout_seconds),
                poll_interval_seconds=max(5.0, float(args.poll_interval_seconds)),
            )
            report["kernel_output_collection"] = collection
            if kernel:
                report["kernel_report"] = kernel
                report["blockers"].extend(kernel.get("blockers") or [])
                if kernel.get("ok") is not True:
                    report["blockers"].append(
                        "heterogeneous_tpu_stage_diagnostic_kernel_incomplete"
                    )
            else:
                report["blockers"].append(
                    "heterogeneous_tpu_stage_diagnostic_output_unavailable"
                )
    except BaseException as exc:
        code = str(exc)
        if not code.startswith(("heterogeneous_", "kaggle_", "tpu_")):
            code = "heterogeneous_tpu_stage_diagnostic_failed:" + type(exc).__name__
        report["blockers"].append(code)
    finally:
        if acquisition_attempt and not acquisition_finished:
            try:
                _finish_attempt(
                    Path(args.acquisition_ledger).resolve(),
                    attempt=acquisition_attempt,
                    outcome=acquisition_outcome,
                )
            except Exception:
                pass
        if kernel_ref:
            try:
                with kaggle_env(
                    args.tpu_token_file, username_hint=args.tpu_token_username
                ) as cleanup_env:
                    delete = run_command(
                        ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                        env=cleanup_env,
                        timeout=float(args.delete_timeout_seconds),
                    )
                    remote_deleted = delete_succeeded_or_absent(delete)
            except Exception:
                remote_deleted = False
        shutil.rmtree(private_root, ignore_errors=True)
        package_removed = package_removed or not private_root.exists()
        report["cleanup"] = {
            "remote_kernel_deleted": bool(remote_deleted or not kernel_ref),
            "temporary_private_package_removed": bool(package_removed),
            "live_resources_left_running": not bool(
                (remote_deleted or not kernel_ref) and package_removed
            ),
        }
        report["ok"] = bool(
            (report.get("kernel_report") or {}).get("ok") is True
            and not report.get("blockers")
            and report["cleanup"]["live_resources_left_running"] is False
        )
        _finish_report(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tpu-token-file", required=True)
    parser.add_argument("--tpu-token-username", required=True)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--acquisition-ledger",
        default="dist/training-heterogeneous-tpu-beta-work/acquisitions.json",
    )
    parser.add_argument("--acquisition-window-limit", type=int, default=3)
    parser.add_argument("--reuse-acquisition-window", type=int, default=3)
    parser.add_argument("--acquisition-window-seconds", type=float, default=43200.0)
    parser.add_argument("--kernel-timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "training_heterogeneous_tpu_stage_diagnostic_live_probe "
            f"ok={report['ok']} blockers={','.join(report['blockers']) or 'none'}"
        )
    return 0 if report["ok"] else (
        1 if not report["cleanup"]["live_resources_left_running"] else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
