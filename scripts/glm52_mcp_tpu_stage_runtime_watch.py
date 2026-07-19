#!/usr/bin/env python3
"""Watch a retained MCP/save-notebook GLM 5.2 TPU stage runtime notebook."""

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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_kaggle_stage_runtime_check as stage_check  # noqa: E402
from scripts import kaggle_gpu_token_weekly_quota_probe as token_probe  # noqa: E402


SCHEMA = "glm52_mcp_tpu_stage_runtime_watch_v1"
DEFAULT_REF = "tpuowner/ct-glm52-tpu-value-op-r1"
DEFAULT_OUTPUT_DIR = "dist/glm52-mcp-tpu-stage-runtime-watch"
OUTPUT_REPORT_NAME = "glm52_kaggle_stage_runtime_report.json"
STATUS_RE = re.compile(r'has status "([^"]+)"')
READY_STALE_BLOCKERS = {
    "glm52_mcp_tpu_stage_runtime_not_ready",
    "glm52_mcp_tpu_stage_runtime_scheduler_queued",
    "glm52_mcp_tpu_stage_runtime_output_missing",
    "glm52_mcp_tpu_stage_runtime_check_failed",
}
Runner = Callable[[list[str], dict[str, str], float], dict[str, Any]]


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


def parse_status(output: str) -> str:
    match = STATUS_RE.search(output or "")
    if match:
        return match.group(1)
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    return lines[-1][:160] if lines else "UNKNOWN"


def public_output_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "filename": path.name,
        "present": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def build_env(args: argparse.Namespace) -> tuple[dict[str, str], tempfile.TemporaryDirectory[str] | None]:
    if not args.token_section:
        return os.environ.copy(), None
    sections = {section["label"]: section for section in token_probe.parse_token_sections(Path(args.token_file))}
    if args.token_section not in sections:
        raise SystemExit(f"token section not found: {args.token_section}")
    temp_dir = tempfile.TemporaryDirectory(prefix="ct_glm52_mcp_tpu_watch_")
    env = token_probe.clean_env(sections[args.token_section]["env"], config_dir=Path(temp_dir.name))
    return env, temp_dir


def run_command(command: list[str], env: dict[str, str], timeout: float) -> dict[str, Any]:
    return token_probe.run_command(command, env=env, timeout=timeout)


def append_observation(watch: dict[str, Any], *, step: dict[str, Any], status: str) -> None:
    observations = watch.setdefault("observations", [])
    observations.append(
        {
            "attempt": len(observations) + 1,
            "observed_at": utc_now(),
            "ok": step.get("ok") is True,
            "status": status,
            "status_class": status_class(status),
            "duration_seconds": step.get("duration_seconds"),
            "output_tail": str(step.get("output_tail") or "")[-500:],
        }
    )


def update_from_output(watch: dict[str, Any], output_path: Path) -> None:
    stage_report = load_json(output_path)
    errors = stage_check.validate_report(stage_report, require_verified=True) if stage_report else ["stage_runtime_output_missing"]
    ready = bool(stage_report and not errors and stage_check.stage_runtime_verified(stage_report))
    watch["stage_runtime_report_verified"] = ready
    watch["tpu_stage_runtime_ready"] = ready
    watch["stage_runtime_report"] = public_output_entry(output_path)
    watch["stage_runtime_check"] = {
        "schema": stage_check.SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "stage_runtime_verified": stage_check.stage_runtime_verified(stage_report) if stage_report else False,
        "provider": stage_check.provider(stage_report) if stage_report else "",
        "stage_id": stage_report.get("stage_id", -1) if stage_report else -1,
    }
    if ready:
        watch["stage_runtime_summary"] = {
            "model_id": stage_check.model_id(stage_report),
            "provider": stage_check.provider(stage_report),
            "stage_id": stage_report.get("stage_id"),
            "stage_layer_range": stage_report.get("stage_layer_range"),
            "coordinator_request_id_hash": stage_check.coordinator_request_hash(stage_report),
            "stage_execution_verified": stage_report.get("stage_execution_verified") is True,
            "stage_decode_verified": stage_report.get("stage_decode_verified") is True,
            "stage_runtime_kind": stage_report.get("stage_runtime_kind"),
            "provider_runtime_verified": stage_report.get("provider_runtime_verified") is True,
            "provider_device_count": stage_report.get("provider_device_count"),
            "weight_value_byte_count": stage_report.get("weight_value_byte_count"),
            "weight_tensor_values_public": stage_report.get("weight_tensor_values_public") is True,
        }


def build_report(args: argparse.Namespace, *, runner: Runner = run_command) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    watch_path = Path(args.watch_report) if args.watch_report else output_dir / "glm52_mcp_tpu_stage_runtime_watch.json"
    watch = load_json(watch_path)
    if not watch:
        watch = {"schema": SCHEMA, "observations": [], "started_at": utc_now()}
    env, temp_dir = build_env(args)
    ref = str(args.ref)
    last_status = str(watch.get("last_status") or "")
    try:
        for poll_index in range(max(1, int(args.status_polls))):
            status_step = runner(
                ["kaggle", "kernels", "status", ref],
                env,
                float(args.status_timeout_seconds),
            )
            last_status = parse_status(str(status_step.get("output_tail") or ""))
            append_observation(watch, step=status_step, status=last_status)
            cls = status_class(last_status)
            if cls == "complete":
                output_runtime_dir = output_dir / "notebook-output"
                output_step = runner(
                    [
                        "kaggle",
                        "kernels",
                        "output",
                        ref,
                        "-p",
                        str(output_runtime_dir),
                        "--force",
                        "--file-pattern",
                        OUTPUT_REPORT_NAME,
                    ],
                    env,
                    float(args.output_timeout_seconds),
                )
                watch["output_download_step"] = {
                    "ok": output_step.get("ok") is True,
                    "returncode": output_step.get("returncode"),
                    "duration_seconds": output_step.get("duration_seconds"),
                    "command_public": output_step.get("command_public"),
                    "output_tail": str(output_step.get("output_tail") or "")[-1000:],
                }
                update_from_output(watch, output_runtime_dir / OUTPUT_REPORT_NAME)
                break
            if cls == "failed":
                watch["stage_runtime_report_verified"] = False
                watch["tpu_stage_runtime_ready"] = False
                watch["blockers"] = sorted(set([*(_list(watch.get("blockers"))), "glm52_mcp_tpu_stage_runtime_kernel_failed"]))
                break
            if poll_index + 1 < int(args.status_polls):
                time.sleep(max(0.0, float(args.status_poll_interval_seconds)))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    ready = watch.get("stage_runtime_report_verified") is True
    cls = status_class(last_status)
    blockers = set(str(item) for item in _list(watch.get("blockers")) if item)
    if ready:
        blockers.difference_update(READY_STALE_BLOCKERS)
    else:
        blockers.add("glm52_mcp_tpu_stage_runtime_not_ready")
        if cls == "queued":
            blockers.add("glm52_mcp_tpu_stage_runtime_scheduler_queued")
        elif cls == "complete":
            if not Path(output_dir / "notebook-output" / OUTPUT_REPORT_NAME).is_file():
                blockers.add("glm52_mcp_tpu_stage_runtime_output_missing")
            else:
                blockers.add("glm52_mcp_tpu_stage_runtime_check_failed")
    watch.update(
        {
            "schema": SCHEMA,
            "ref": ref,
            "last_status": last_status,
            "last_status_class": cls,
            "updated_at": utc_now(),
            "stage_runtime_report_verified": ready,
            "tpu_stage_runtime_ready": watch.get("tpu_stage_runtime_ready") is True,
            "same_request_decode_verified": False,
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
    write_json(watch_path, watch)
    return watch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--watch-report", default="")
    parser.add_argument("--token-file", default="~/.config/crowdtensor/kaggle-tokens.md")
    parser.add_argument("--token-section", default="tpuowner")
    parser.add_argument("--status-polls", type=int, default=1)
    parser.add_argument("--status-poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "glm52_mcp_tpu_stage_runtime_watch: "
            f"status={report.get('last_status')} ready={report.get('stage_runtime_report_verified')}"
        )
    return 0 if report.get("public_artifact_safe") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
