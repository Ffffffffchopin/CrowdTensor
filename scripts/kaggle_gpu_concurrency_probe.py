#!/usr/bin/env python3
"""Bounded Kaggle T4x2 GPU concurrency probe.

The probe submits private Kaggle script kernels under the configured Kaggle
account, verifies each accepted GPU kernel sees two CUDA devices, observes
whether two kernels overlap, downloads only public-safe worker summaries, and
then deletes the temporary kernels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_gpu_concurrency_probe_v1"
WORKER_REPORT_NAME = "kaggle_gpu_concurrency_worker_report.json"
STATUS_RE = re.compile(r'has status "([^"]+)"')
CODE_URL_RE = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^?\s]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:63].strip("-") or "ct-gpu-concurrency"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def redact_output(output: str) -> str:
    redacted = str(output or "")
    redacted = re.sub(r"(?i)(kaggle[_-]?key|api[_-]?key|token|cookie|oauth)[=:]\S+", r"\1=<redacted>", redacted)
    redacted = re.sub(r"(?i)(bearer\s+)[a-z0-9._=-]+", r"\1<redacted>", redacted)
    return redacted


def redact_command(command: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in command:
        value = str(item)
        if "token" in value.lower() or "cookie" in value.lower() or "KAGGLE_KEY" in value:
            cleaned.append("<redacted>")
        else:
            cleaned.append(value)
    return cleaned


def run_command(command: list[str], *, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = redact_output(proc.stdout or "")
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_public": redact_command(command),
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        output = redact_output(output)
        return {
            "ok": False,
            "returncode": None,
            "timed_out": True,
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_public": redact_command(command),
            "output_tail": output[-4000:],
        }


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


def extract_kernel_ref(text: str) -> str:
    match = CODE_URL_RE.search(text or "")
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return ""


def push_accepted(step: dict[str, Any]) -> bool:
    output = str(step.get("output_tail") or "")
    return bool(step.get("ok")) and "Kernel version" in output and "successfully pushed" in output


def parse_status(output: str) -> str:
    match = STATUS_RE.search(output or "")
    if match:
        return match.group(1)
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    return (lines[-1] if lines else "")[:160]


def status_class(status: str) -> str:
    upper = str(status or "").upper()
    if "RUNNING" in upper:
        return "running"
    if "COMPLETE" in upper or "SUCCESS" in upper:
        return "complete"
    if "FAIL" in upper or "ERROR" in upper or "CANCEL" in upper:
        return "failed"
    if "QUEUE" in upper or "PENDING" in upper or "PREPAR" in upper or "INITIAL" in upper:
        return "queued"
    return "unknown"


def diagnose_push_rejection(output: str) -> list[str]:
    text = str(output or "").lower()
    codes: list[str] = ["kaggle_gpu_kernel_push_rejected"]
    if any(fragment in text for fragment in ["maximum", "limit", "too many", "session", "quota"]):
        codes.append("kaggle_gpu_quota_or_session_limit")
    if "gpu" in text and any(fragment in text for fragment in ["unavailable", "not available", "resource"]):
        codes.append("kaggle_gpu_resource_unavailable")
    if "forbidden" in text or "permission" in text:
        codes.append("kaggle_gpu_permission_denied")
    return sorted(set(codes))


def parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def worker_reports_overlap(worker_reports: list[dict[str, Any]]) -> bool:
    intervals: list[tuple[datetime, datetime]] = []
    for report in worker_reports:
        start = parse_iso(report.get("started_at"))
        finish = parse_iso(report.get("finished_at"))
        if start is None or finish is None or finish <= start:
            continue
        intervals.append((start, finish))
    if len(intervals) < 2:
        return False
    latest_start = max(item[0] for item in intervals)
    earliest_finish = min(item[1] for item in intervals)
    return latest_start < earliest_finish


def render_kernel(index: int, *, hold_seconds: float) -> str:
    return "\n".join(
        [
            "import hashlib",
            "import json",
            "import os",
            "import time",
            "from datetime import datetime, timezone",
            "",
            f"INDEX = {int(index)!r}",
            f"HOLD_SECONDS = {float(hold_seconds)!r}",
            f"REPORT_NAME = {WORKER_REPORT_NAME!r}",
            "",
            "def sha16(value):",
            "    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:16]",
            "",
            "def utc_now():",
            "    return datetime.now(timezone.utc).isoformat()",
            "",
            "report = {",
            "    'schema': 'kaggle_gpu_concurrency_worker_report_v1',",
            "    'ok': False,",
            "    'worker_index': INDEX,",
            "    'started_at': utc_now(),",
            "    'finished_at': '',",
            "    'hold_seconds': HOLD_SECONDS,",
            "    'public_artifact_safe': True,",
            "    'raw_gpu_names_public': False,",
            "}",
            "try:",
            "    import torch",
            "    report['torch_version'] = getattr(torch, '__version__', '')",
            "    report['cuda_available'] = bool(torch.cuda.is_available())",
            "    report['cuda_device_count'] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0",
            "    report['gpu_name_hashes'] = []",
            "    report['total_memory_mb'] = []",
            "    report['matmul_hashes'] = []",
            "    for device_index in range(int(report['cuda_device_count'])):",
            "        props = torch.cuda.get_device_properties(device_index)",
            "        report['gpu_name_hashes'].append(sha16(getattr(props, 'name', 'unknown')))",
            "        report['total_memory_mb'].append(int(getattr(props, 'total_memory', 0) // (1024 * 1024)))",
            "        with torch.cuda.device(device_index):",
            "            torch.manual_seed(1234 + INDEX + device_index)",
            "            a = torch.ones((256, 256), device=f'cuda:{device_index}', dtype=torch.float16)",
            "            b = torch.eye(256, device=f'cuda:{device_index}', dtype=torch.float16)",
            "            c = a @ b",
            "            torch.cuda.synchronize(device_index)",
            "            report['matmul_hashes'].append(sha16(float(c.sum().detach().cpu().item())))",
            "    report['ok'] = bool(report['cuda_available'] and int(report['cuda_device_count']) >= 2)",
            "except Exception as exc:",
            "    report['error_type'] = type(exc).__name__",
            "    report['error_public'] = str(exc)[-300:]",
            "",
            "time.sleep(max(0.0, float(HOLD_SECONDS)))",
            "report['finished_at'] = utc_now()",
            "path = os.path.join('/kaggle/working', REPORT_NAME)",
            "with open(path, 'w', encoding='utf-8') as handle:",
            "    json.dump(report, handle, indent=2, sort_keys=True)",
            "print(json.dumps({",
            "    'schema': report['schema'],",
            "    'ok': report['ok'],",
            "    'worker_index': INDEX,",
            "    'cuda_device_count': report.get('cuda_device_count', 0),",
            "    'public_artifact_safe': True,",
            "}, sort_keys=True), flush=True)",
            "",
        ]
    )


def write_kernel_package(
    kernel_dir: Path,
    *,
    owner: str,
    slug: str,
    index: int,
    accelerator: str,
    hold_seconds: float,
) -> str:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(render_kernel(index, hold_seconds=hold_seconds), encoding="utf-8")
    write_json(
        kernel_dir / "kernel-metadata.json",
        {
            "id": f"{owner}/{slug}",
            "title": slug.replace("-", " ").title(),
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "false",
            "machine_shape": accelerator,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    return f"{owner}/{slug}"


def poll_statuses(
    refs: list[str],
    *,
    polls: int,
    interval: float,
    timeout: float,
) -> tuple[list[dict[str, Any]], int]:
    observations: list[dict[str, Any]] = []
    max_running = 0
    for poll_index in range(max(1, int(polls))):
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
        observation = {
            "poll_index": poll_index,
            "observed_at": utc_now(),
            "running_count": running_count,
            "queued_count": sum(1 for value in classes.values() if value == "queued"),
            "complete_count": sum(1 for value in classes.values() if value == "complete"),
            "failed_count": sum(1 for value in classes.values() if value == "failed"),
            "unknown_count": sum(1 for value in classes.values() if value == "unknown"),
            "statuses": statuses,
        }
        observations.append(observation)
        print(
            f"[{observation['observed_at']}] gpu concurrency poll={poll_index} running={running_count} statuses={classes}",
            flush=True,
        )
        if running_count >= len(refs):
            break
        if poll_index + 1 < max(1, int(polls)):
            time.sleep(max(0.0, float(interval)))
    return observations, max_running


def wait_terminal(ref: str, *, timeout: float, interval: float, status_timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    last: dict[str, Any] = {}
    while time.monotonic() - started <= timeout:
        attempts += 1
        step = run_command(["kaggle", "kernels", "status", ref], timeout=status_timeout)
        status = parse_status(str(step.get("output_tail") or ""))
        klass = status_class(status)
        last = {
            "kernel_ref": ref,
            "attempts": attempts,
            "status": status,
            "status_class": klass,
            "duration_seconds": round(time.monotonic() - started, 3),
            "ok": bool(step.get("ok") and klass == "complete"),
        }
        print(f"[{utc_now()}] wait terminal kernel={ref} attempt={attempts} status_class={klass}", flush=True)
        if klass in {"complete", "failed"}:
            return last
        time.sleep(max(5.0, float(interval)))
    last.update(
        {
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout_waiting_for_terminal_status",
        }
    )
    return last


def load_worker_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def iter_string_values(payload: Any) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for value in payload.values():
            values.extend(iter_string_values(value))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(iter_string_values(value))
    elif isinstance(payload, str):
        values.append(payload)
    return values


def public_redaction_errors(payload: Any) -> list[str]:
    lowered = "\n".join(iter_string_values(payload)).lower()
    patterns = [
        "kaggle_key",
        "api_key",
        "oauth",
        "cookie",
        "bearer ",
        "proxyurl",
        "proxy_url",
        "runtimeproxy",
        "activation_payload",
        "hidden_states",
        "logits",
        "past_key_values",
    ]
    return [pattern for pattern in patterns if pattern in lowered]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="dist/kaggle-gpu-concurrency-probe")
    parser.add_argument("--owner", default="")
    parser.add_argument("--kernel-count", type=int, default=2)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--hold-seconds", type=float, default=180.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=600)
    parser.add_argument("--push-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--terminal-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--polls", type=int, default=18)
    parser.add_argument("--poll-interval", type=float, default=20.0)
    parser.add_argument("--slug-prefix", default="ct-gpu-conc")
    parser.add_argument("--keep-kernels", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    packages_dir = output_dir / "private-kaggle-gpu-kernels"
    if packages_dir.exists():
        shutil.rmtree(packages_dir)
    packages_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    owner = safe_slug(args.owner) if args.owner else load_kaggle_owner()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_ready": False,
        "ok": False,
        "simultaneous_t4x2_verified": False,
        "started_at": utc_now(),
        "finished_at": "",
        "owner": owner,
        "requested_kernel_count": int(args.kernel_count),
        "accelerator": args.accelerator,
        "hold_seconds": float(args.hold_seconds),
        "submitted_kernel_refs": [],
        "rejected_kernel_refs": [],
        "push_steps": [],
        "status_observations": [],
        "terminal_steps": [],
        "output_steps": [],
        "worker_reports": [],
        "max_observed_running_count": 0,
        "accepted_submission_count": 0,
        "worker_runtime_overlap_verified": False,
        "cleanup": {"attempted": False, "deleted_refs": [], "failed_delete_refs": []},
        "blockers": [],
        "diagnosis_codes": [],
        "private_kernel_payloads_removed": False,
        "public_artifact_safe": True,
        "safety": {
            "private_kernels": True,
            "internet_disabled": True,
            "bounded_probe": True,
            "raw_gpu_names_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "runtime_proxy_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
        },
    }

    refs: list[str] = []
    try:
        if not owner:
            report["blockers"].append("kaggle_owner_missing")
            return_code = 2
        elif int(args.kernel_count) < 2:
            report["blockers"].append("kernel_count_below_two")
            return_code = 2
        else:
            suffix = str(int(time.time()))[-8:]
            packages: list[dict[str, Any]] = []
            for index in range(int(args.kernel_count)):
                slug = safe_slug(f"{args.slug_prefix}-{suffix}-{index}")
                kernel_dir = packages_dir / f"kernel-{index}"
                ref = write_kernel_package(
                    kernel_dir,
                    owner=owner,
                    slug=slug,
                    index=index,
                    accelerator=str(args.accelerator),
                    hold_seconds=float(args.hold_seconds),
                )
                packages.append({"index": index, "ref": ref, "kernel_dir": kernel_dir})

            def push_package(package: dict[str, Any]) -> dict[str, Any]:
                command = [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(package["kernel_dir"]),
                    "-t",
                    str(int(args.kernel_timeout_seconds)),
                    "--accelerator",
                    str(args.accelerator),
                ]
                step = run_command(command, timeout=float(args.push_timeout_seconds))
                step["kernel_ref"] = package["ref"]
                actual = extract_kernel_ref(str(step.get("output_tail") or ""))
                if actual:
                    step["actual_kernel_ref"] = actual
                    step["kernel_ref"] = actual
                step["accepted"] = push_accepted(step)
                step["package_index"] = package["index"]
                return step

            with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.kernel_count)) as pool:
                futures = [pool.submit(push_package, package) for package in packages]
                for future in concurrent.futures.as_completed(futures):
                    push = future.result()
                    report["push_steps"].append(push)
                    if push.get("accepted"):
                        refs.append(str(push.get("kernel_ref") or ""))
                    else:
                        ref = str(push.get("kernel_ref") or "")
                        if ref:
                            report["rejected_kernel_refs"].append(ref)
                        report["blockers"].extend(diagnose_push_rejection(str(push.get("output_tail") or "")))
                    report["submitted_kernel_refs"] = refs
                    report["accepted_submission_count"] = len(refs)
                    write_json(output_dir / "kaggle_gpu_concurrency_probe.json", report)

            report["push_steps"] = sorted(report["push_steps"], key=lambda item: int(item.get("package_index") or 0))
            if len(refs) == int(args.kernel_count):
                observations, max_running = poll_statuses(
                    refs,
                    polls=int(args.polls),
                    interval=float(args.poll_interval),
                    timeout=float(args.status_timeout_seconds),
                )
                report["status_observations"] = observations
                report["max_observed_running_count"] = int(max_running)
                for ref in refs:
                    terminal = wait_terminal(
                        ref,
                        timeout=float(args.terminal_timeout_seconds),
                        interval=float(args.poll_interval),
                        status_timeout=float(args.status_timeout_seconds),
                    )
                    report["terminal_steps"].append(terminal)
                    stage_output = output_dir / "kaggle-output" / safe_slug(ref.replace("/", "-"))
                    output_step = run_command(
                        [
                            "kaggle",
                            "kernels",
                            "output",
                            ref,
                            "-p",
                            str(stage_output),
                            "--force",
                            "--file-pattern",
                            WORKER_REPORT_NAME,
                        ],
                        timeout=float(args.output_timeout_seconds),
                    )
                    output_step["kernel_ref"] = ref
                    report["output_steps"].append(output_step)
                    worker_report = load_worker_report(stage_output / WORKER_REPORT_NAME)
                    if worker_report:
                        worker_report["kernel_ref"] = ref
                        report["worker_reports"].append(worker_report)
            elif refs:
                report["diagnosis_codes"].append("partial_gpu_kernel_acceptance")
            else:
                report["diagnosis_codes"].append("no_gpu_kernels_accepted")

            worker_reports = [item for item in report["worker_reports"] if isinstance(item, dict)]
            report["worker_runtime_overlap_verified"] = worker_reports_overlap(worker_reports)
            worker_ok = [
                item
                for item in worker_reports
                if item.get("ok") is True
                and item.get("public_artifact_safe") is True
                and int(item.get("cuda_device_count") or 0) >= 2
            ]
            report["simultaneous_t4x2_verified"] = bool(
                len(worker_ok) >= int(args.kernel_count)
                and (
                    int(report["max_observed_running_count"] or 0) >= int(args.kernel_count)
                    or report["worker_runtime_overlap_verified"] is True
                )
            )
            report["ok"] = bool(report["simultaneous_t4x2_verified"])
            if report["ok"]:
                report["diagnosis_codes"].append("two_kaggle_t4x2_kernels_verified")
            elif not report["blockers"]:
                report["blockers"].append("two_kaggle_t4x2_concurrency_not_verified")
            return_code = 0 if report["ok"] else 1
    finally:
        if refs and not args.keep_kernels:
            report["cleanup"]["attempted"] = True
            for ref in refs:
                step = run_command(
                    ["kaggle", "kernels", "delete", ref, "-y"],
                    timeout=float(args.delete_timeout_seconds),
                )
                if step.get("ok"):
                    report["cleanup"]["deleted_refs"].append(ref)
                else:
                    report["cleanup"]["failed_delete_refs"].append({"ref": ref, "step": step})
        shutil.rmtree(packages_dir, ignore_errors=True)
        report["private_kernel_payloads_removed"] = not packages_dir.exists()
        report["finished_at"] = utc_now()
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []))
        report["blockers"] = sorted(set(report.get("blockers") or []))
        report["public_artifact_safe"] = not public_redaction_errors(report)
        report["evidence_ready"] = bool(
            report["public_artifact_safe"]
            and report["private_kernel_payloads_removed"]
            and not report["cleanup"].get("failed_delete_refs")
        )
        write_json(output_dir / "kaggle_gpu_concurrency_probe.json", report)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(
                "kaggle_gpu_concurrency_probe: "
                f"ok={report['ok']} evidence_ready={report['evidence_ready']} "
                f"accepted={report['accepted_submission_count']} max_running={report['max_observed_running_count']} "
                f"blockers={','.join(report['blockers']) or 'none'}"
            )
    return return_code if report["evidence_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
