#!/usr/bin/env python3
"""Bounded Kaggle CPU kernel concurrency probe.

This script creates private no-op CPU script kernels, submits them through the
configured Kaggle CLI account, polls their statuses briefly, and deletes them.
It is intentionally capped and cleanup-first; it is a quota observation helper,
not a load test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_cpu_concurrency_probe_v1"
STATUS_RE = re.compile(r'has status "([^"]+)"')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:63].strip("-") or "ct-cpu-probe"


def run_command(command: list[str], *, timeout: float) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "duration_seconds": round(time.time() - started, 3),
            "command": redact_command(command),
            "output_tail": redact_output(proc.stdout)[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return {
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.time() - started, 3),
            "command": redact_command(command),
            "timed_out": True,
            "output_tail": redact_output(output)[-4000:],
        }


def redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    for item in command:
        if "KAGGLE_KEY" in item or "token" in item.lower():
            redacted.append("<redacted>")
        else:
            redacted.append(item)
    return redacted


def redact_output(output: str) -> str:
    output = re.sub(r"(?i)(kaggle[_-]?key|api[_-]?key|token)[=:]\S+", r"\1=<redacted>", output)
    return output


def load_kaggle_owner() -> str:
    code = (
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api=KaggleApi(); api.authenticate()\n"
        "print(api.config_values.get('username') or api.config_values.get('user') or '')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    return safe_slug(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_kernel_package(
    kernel_dir: Path,
    *,
    owner: str,
    slug: str,
    index: int,
    sleep_seconds: float,
) -> str:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    marker = f"ct_cpu_concurrency_probe_{index}"
    (kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import time",
                "from datetime import datetime, timezone",
                f"marker = {marker!r}",
                "started = datetime.now(timezone.utc).isoformat()",
                "print(json.dumps({'marker': marker, 'started': started, 'pid': os.getpid()}, sort_keys=True), flush=True)",
                f"time.sleep({float(sleep_seconds)!r})",
                "finished = datetime.now(timezone.utc).isoformat()",
                "print(json.dumps({'marker': marker, 'finished': finished}, sort_keys=True), flush=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        kernel_dir / "kernel-metadata.json",
        {
            "id": f"{owner}/{slug}",
            "title": slug.replace("-", " ").title(),
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "false",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    return f"{owner}/{slug}"


def parse_status(output: str) -> str:
    match = STATUS_RE.search(output)
    if match:
        return match.group(1)
    normalized = output.strip().splitlines()[-1] if output.strip() else ""
    return normalized[:160]


def push_accepted(step: dict[str, Any]) -> bool:
    output = str(step.get("output_tail") or "")
    return bool(step.get("ok")) and "Kernel version" in output and "successfully pushed" in output


def status_class(status: str) -> str:
    upper = status.upper()
    if "RUNNING" in upper:
        return "running"
    if "COMPLETE" in upper or "SUCCESS" in upper:
        return "complete"
    if "FAIL" in upper or "ERROR" in upper or "CANCEL" in upper:
        return "failed"
    if "QUEUE" in upper or "PENDING" in upper or "PREPAR" in upper or "INITIAL" in upper:
        return "queued"
    return "unknown"


def poll_statuses(refs: list[str], *, polls: int, interval: float, timeout: float) -> tuple[list[dict[str, Any]], int]:
    observations: list[dict[str, Any]] = []
    max_running = 0
    for poll_index in range(polls):
        statuses: dict[str, str] = {}
        classes: dict[str, str] = {}
        for ref in refs:
            step = run_command(["kaggle", "kernels", "status", ref], timeout=timeout)
            status = parse_status(str(step.get("output_tail") or ""))
            klass = status_class(status)
            statuses[ref] = status
            classes[ref] = klass
        running_count = sum(1 for value in classes.values() if value == "running")
        max_running = max(max_running, running_count)
        observations.append(
            {
                "poll_index": poll_index,
                "observed_at": utc_now(),
                "running_count": running_count,
                "queued_count": sum(1 for value in classes.values() if value == "queued"),
                "complete_count": sum(1 for value in classes.values() if value == "complete"),
                "failed_count": sum(1 for value in classes.values() if value == "failed"),
                "unknown_count": sum(1 for value in classes.values() if value == "unknown"),
                "statuses": statuses,
            }
        )
        if poll_index + 1 < polls:
            time.sleep(interval)
    return observations, max_running


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dist/kaggle-cpu-concurrency-probe")
    parser.add_argument("--owner", default="")
    parser.add_argument("--max-kernels", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=360.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=600)
    parser.add_argument("--push-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--polls", type=int, default=8)
    parser.add_argument("--poll-interval", type=float, default=20.0)
    parser.add_argument("--slug-prefix", default="ct-cpu-conc")
    parser.add_argument("--keep-kernels", action="store_true")
    args = parser.parse_args()

    started = utc_now()
    output_dir = Path(args.output_dir)
    packages_dir = output_dir / "private-kaggle-cpu-kernels"
    if packages_dir.exists():
        shutil.rmtree(packages_dir)
    packages_dir.mkdir(parents=True, exist_ok=True)

    owner = safe_slug(args.owner) if args.owner else load_kaggle_owner()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "started_at": started,
        "finished_at": "",
        "owner": owner,
        "requested_max_kernels": int(args.max_kernels),
        "sleep_seconds": float(args.sleep_seconds),
        "kernel_timeout_seconds": int(args.kernel_timeout_seconds),
        "submitted_kernel_refs": [],
        "rejected_kernel_refs": [],
        "push_steps": [],
        "status_observations": [],
        "max_observed_running_count": 0,
        "accepted_submission_count": 0,
        "cleanup": {"attempted": False, "deleted_refs": [], "failed_delete_refs": []},
        "diagnosis_codes": [],
        "safety": {
            "cpu_only": True,
            "private_kernels": True,
            "internet_disabled": True,
            "bounded_probe": True,
            "not_load_test": True,
            "credentials_redacted": True,
        },
    }
    if not owner:
        report["diagnosis_codes"] = ["kaggle_owner_missing"]
        report["finished_at"] = utc_now()
        write_json(output_dir / "kaggle_cpu_concurrency_probe.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    suffix = str(int(time.time()))[-8:]
    refs: list[str] = []
    try:
        for index in range(max(0, int(args.max_kernels))):
            slug = safe_slug(f"{args.slug_prefix}-{suffix}-{index}")
            kernel_dir = packages_dir / f"kernel-{index}"
            ref = write_kernel_package(
                kernel_dir,
                owner=owner,
                slug=slug,
                index=index,
                sleep_seconds=float(args.sleep_seconds),
            )
            push = run_command(
                [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(kernel_dir),
                    "-t",
                    str(int(args.kernel_timeout_seconds)),
                ],
                timeout=float(args.push_timeout_seconds),
            )
            push["kernel_ref"] = ref
            push["accepted"] = push_accepted(push)
            report["push_steps"].append(push)
            if not push["accepted"]:
                report["rejected_kernel_refs"].append(ref)
                report["diagnosis_codes"].append("kaggle_kernel_push_rejected")
                break
            refs.append(ref)
            report["submitted_kernel_refs"] = refs
            report["accepted_submission_count"] = len(refs)
            write_json(output_dir / "kaggle_cpu_concurrency_probe.json", report)

        if refs:
            observations, max_running = poll_statuses(
                refs,
                polls=max(1, int(args.polls)),
                interval=max(0.0, float(args.poll_interval)),
                timeout=float(args.status_timeout_seconds),
            )
            report["status_observations"] = observations
            report["max_observed_running_count"] = max_running
            final_classes = {
                ref: status_class(observations[-1]["statuses"].get(ref, "")) for ref in refs
            } if observations else {}
            report["final_status_classes"] = final_classes
            if max_running >= len(refs) and len(refs) == int(args.max_kernels):
                report["diagnosis_codes"].append("probe_cap_reached_without_rejection")
            elif max_running > 0:
                report["diagnosis_codes"].append("running_concurrency_observed")
            else:
                report["diagnosis_codes"].append("no_running_status_observed")
        else:
            report["diagnosis_codes"].append("no_kernels_submitted")
    finally:
        if refs and not args.keep_kernels:
            report["cleanup"]["attempted"] = True
            for ref in refs:
                step = run_command(
                    ["kaggle", "kernels", "delete", "-y", ref],
                    timeout=float(args.delete_timeout_seconds),
                )
                if step.get("ok"):
                    report["cleanup"]["deleted_refs"].append(ref)
                else:
                    report["cleanup"]["failed_delete_refs"].append({"ref": ref, "step": step})
        shutil.rmtree(packages_dir, ignore_errors=True)
        report["private_kernel_payloads_removed"] = not packages_dir.exists()
        report["finished_at"] = utc_now()
        report["ok"] = bool(report["accepted_submission_count"] > 0 and not report["cleanup"]["failed_delete_refs"])
        write_json(output_dir / "kaggle_cpu_concurrency_probe.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
