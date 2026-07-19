#!/usr/bin/env python3
"""Poll the retained Kaggle GLM 5.2 AWQ TPU stage-smoke notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_awq_tpu_stage_smoke_check as smoke_check  # noqa: E402
from scripts import kaggle_gpu_token_weekly_quota_probe as token_probe  # noqa: E402


SCHEMA = "glm52_kaggle_tpu_awq_stage_smoke_watch_v1"
DEFAULT_REF = "tpuowner/ct-glm52-awq-tpu-stage-smoke-0704-r1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-tpu-awq-stage-smoke-watch"
OUTPUT_REPORT_NAME = "glm52_awq_tpu_stage_smoke.json"
STATUS_RE = re.compile(r'has status "([^"]+)"')
READY_STALE_BLOCKERS = {
    "glm52_awq_tpu_stage_smoke_not_ready",
    "glm52_awq_tpu_stage_smoke_scheduler_queued",
    "glm52_awq_tpu_stage_smoke_kernel_failed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_status(output: str) -> str:
    match = STATUS_RE.search(output or "")
    if match:
        return match.group(1)
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    return lines[-1][:160] if lines else "UNKNOWN"


def status_class(status: str) -> str:
    upper = str(status or "").upper()
    if "COMPLETE" in upper or "SUCCESS" in upper:
        return "complete"
    if "FAIL" in upper or "ERROR" in upper or "CANCEL" in upper:
        return "failed"
    if "RUNNING" in upper:
        return "running"
    if "QUEUE" in upper or "PENDING" in upper or "PREPAR" in upper or "INITIAL" in upper:
        return "queued"
    return "unknown"


def build_env(args: argparse.Namespace) -> tuple[dict[str, str], tempfile.TemporaryDirectory[str] | None]:
    if not args.token_section:
        return os.environ.copy(), None
    sections = {section["label"]: section for section in token_probe.parse_token_sections(Path(args.token_file))}
    if args.token_section not in sections:
        raise SystemExit(f"token section not found: {args.token_section}")
    temp_dir = tempfile.TemporaryDirectory(prefix="ct_glm52_stage_smoke_watch_")
    env = token_probe.clean_env(sections[args.token_section]["env"], config_dir=Path(temp_dir.name))
    return env, temp_dir


def run_command(command: list[str], *, env: dict[str, str], timeout: float) -> dict[str, Any]:
    return token_probe.run_command(command, env=env, timeout=timeout)


def append_observation(watch: dict[str, Any], *, step: dict[str, Any], status: str, started: float) -> None:
    observations = watch.setdefault("observations", [])
    elapsed = float(observations[-1].get("elapsed_seconds") or 0.0) if observations and isinstance(observations[-1], dict) else 0.0
    observations.append(
        {
            "attempt": len(observations) + 1,
            "observed_at": utc_now(),
            "elapsed_seconds": round(elapsed + max(1.0, time.monotonic() - started), 1),
            "ok": step.get("ok") is True,
            "status": status,
            "status_class": status_class(status),
            "output_tail": str(step.get("output_tail") or "")[-500:],
        }
    )


def public_output_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else "",
        "filename": path.name,
    }


def update_from_output(watch: dict[str, Any], output_path: Path) -> None:
    output_report = load_json(output_path)
    errors = smoke_check.validate_report(output_report, require_ready=True) if output_report else ["stage_smoke_output_missing"]
    ready = bool(output_report and not errors and smoke_check._ready(output_report))
    watch["notebook_output_verified"] = ready
    watch["tpu_runtime_ready"] = ready
    watch["stage_runtime_adapter_smoke_ready"] = ready
    watch["stage_smoke_output"] = public_output_entry(output_path)
    watch["stage_smoke_check"] = {
        "schema": smoke_check.SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "stage_runtime_adapter_smoke_ready": ready,
        "source_schema": output_report.get("schema") if output_report else "",
    }
    if ready:
        watch["stage_smoke_summary"] = {
            "model_repo": output_report.get("model_repo"),
            "base_model_id": output_report.get("base_model_id"),
            "quantization": output_report.get("quantization"),
            "stage_id": output_report.get("stage_id"),
            "stage_count": output_report.get("stage_count"),
            "stage_layer_range": output_report.get("stage_layer_range"),
            "jax_tpu_device_count": output_report.get("jax_tpu_device_count") or output_report.get("tpu_device_count"),
            "assigned_weight_key_count": output_report.get("assigned_weight_key_count"),
            "present_stage_key_count": output_report.get("present_stage_key_count"),
            "missing_stage_key_count": output_report.get("missing_stage_key_count"),
            "weight_tensor_values_public": output_report.get("weight_tensor_values_public") is True,
        }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    watch_path = Path(args.watch_report) if args.watch_report else output_dir / "glm52_kaggle_tpu_awq_stage_smoke_watch.json"
    watch = load_json(watch_path)
    if not watch:
        watch = {"schema": SCHEMA, "observations": []}
    env, temp_dir = build_env(args)
    ref = str(args.ref)
    try:
        last_status = str(watch.get("last_status") or "")
        for poll_index in range(max(1, int(args.status_polls))):
            started = time.monotonic()
            status_step = run_command(
                ["kaggle", "kernels", "status", ref],
                env=env,
                timeout=float(args.status_timeout_seconds),
            )
            last_status = parse_status(str(status_step.get("output_tail") or ""))
            append_observation(watch, step=status_step, status=last_status, started=started)
            cls = status_class(last_status)
            if cls == "complete":
                output_dir_runtime = output_dir / "notebook-output"
                output_step = run_command(
                    [
                        "kaggle",
                        "kernels",
                        "output",
                        ref,
                        "-p",
                        str(output_dir_runtime),
                        "--force",
                        "--file-pattern",
                        OUTPUT_REPORT_NAME,
                    ],
                    env=env,
                    timeout=float(args.output_timeout_seconds),
                )
                watch["output_download_step"] = {
                    "ok": output_step.get("ok") is True,
                    "returncode": output_step.get("returncode"),
                    "duration_seconds": output_step.get("duration_seconds"),
                    "command_public": output_step.get("command_public"),
                    "output_tail": str(output_step.get("output_tail") or "")[-1000:],
                }
                update_from_output(watch, output_dir_runtime / OUTPUT_REPORT_NAME)
                break
            if cls == "failed":
                watch["notebook_output_verified"] = False
                watch["tpu_runtime_ready"] = False
                watch["stage_runtime_adapter_smoke_ready"] = False
                watch["blockers"] = sorted(set([*(_list(watch.get("blockers"))), "glm52_awq_tpu_stage_smoke_kernel_failed"]))
                break
            if poll_index + 1 < int(args.status_polls):
                time.sleep(max(0.0, float(args.status_poll_interval_seconds)))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    ready = watch.get("stage_runtime_adapter_smoke_ready") is True
    queued = status_class(str(watch.get("last_status") or last_status)) == "queued"
    blockers = set(str(item) for item in _list(watch.get("blockers")) if item)
    if ready:
        blockers.difference_update(READY_STALE_BLOCKERS)
    elif not queued:
        blockers.discard("glm52_awq_tpu_stage_smoke_scheduler_queued")
    if not ready:
        blockers.add("glm52_awq_tpu_stage_smoke_not_ready")
    if queued and not ready:
        blockers.add("glm52_awq_tpu_stage_smoke_scheduler_queued")
    watch.update(
        {
            "schema": SCHEMA,
            "ref": ref,
            "last_status": last_status,
            "updated_at": utc_now(),
            "notebook_output_verified": ready,
            "tpu_runtime_ready": watch.get("tpu_runtime_ready") is True,
            "stage_runtime_adapter_smoke_ready": ready,
            "blockers": sorted(blockers),
            "public_artifact_safe": True,
            "credentials_public": False,
            "signed_output_url_public": False,
            "safety": {
                "public_artifact_safe": True,
                "credentials_public": False,
                "cookies_public": False,
                "signed_url_public": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "activation_public": False,
                "hidden_state_public": False,
                "logits_public": False,
                "kv_cache_public": False,
                "weight_tensor_values_public": False,
                "safetensors_header_payload_public": False,
            },
        }
    )
    leaks = smoke_check.public_redaction_errors(watch)
    if leaks:
        watch["public_artifact_safe"] = False
        watch["safety"]["public_artifact_safe"] = False
        watch["blockers"].append("public_redaction_scan_failed")
        watch["redaction_errors"] = leaks
    write_json(output_dir / "glm52_kaggle_tpu_awq_stage_smoke_watch.json", watch)
    return watch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--watch-report", default="")
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--token-file", default="~/.config/crowdtensor/kaggle-tokens.md")
    parser.add_argument("--token-section", default="tpuowner")
    parser.add_argument("--status-polls", type=int, default=1)
    parser.add_argument("--status-poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.status_polls < 1 or args.status_polls > 10000:
        raise SystemExit("--status-polls must be between 1 and 10000")
    if args.status_poll_interval_seconds < 0:
        raise SystemExit("--status-poll-interval-seconds must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'glm52_kaggle_tpu_awq_stage_smoke_watch.json'}")
        print(f"Status: {report.get('last_status')}")
        print(f"Stage smoke ready: {report.get('stage_runtime_adapter_smoke_ready')}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
