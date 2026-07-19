#!/usr/bin/env python3
"""Validate Kaggle Web TPU execution-channel probe artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_web_tpu_execution_channel_probe as probe  # noqa: E402


SCHEMA = "kaggle_web_tpu_execution_channel_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    for field in [
        "web_tpu_execution_channel_ready",
        "small_jax_cell_ready",
        "tiny_qwen_like_cell_ready",
        "tpu_runtime_attached",
        "tpu_device_count",
        "failure_stage",
        "blocker_codes",
        "cleanup_status",
        "public_artifact_safe",
    ]:
        if field not in report:
            errors.append(f"required_field_missing:{field}")

    cells = _dict(report.get("cells"))
    small = _dict(cells.get("small_jax"))
    tiny = _dict(cells.get("tiny_qwen_like"))
    if small.get("schema") != "kaggle_web_tpu_execution_channel_cell_summary_v1":
        errors.append("small_jax_cell_schema_mismatch")
    if tiny.get("schema") != "kaggle_web_tpu_execution_channel_cell_summary_v1":
        errors.append("tiny_qwen_cell_schema_mismatch")

    if report.get("web_tpu_execution_channel_ready") is True:
        if report.get("ok") is not True:
            errors.append("channel_ready_report_not_ok")
        if report.get("small_jax_cell_ready") is not True:
            errors.append("channel_ready_without_small_jax")
        if report.get("tiny_qwen_like_cell_ready") is not True:
            errors.append("channel_ready_without_tiny_qwen")
        if int(report.get("tpu_device_count") or 0) < 1:
            errors.append("channel_ready_without_tpu_device")
        if report.get("failure_stage"):
            errors.append("channel_ready_has_failure_stage")
        if _list(report.get("blocker_codes")):
            errors.append("channel_ready_has_blockers")
        if not str(tiny.get("stage_output_hash") or "").startswith("sha256:"):
            errors.append("tiny_qwen_ready_without_output_hash")
    else:
        if report.get("ok") is True:
            errors.append("channel_not_ready_report_ok")
        if not _list(report.get("blocker_codes")):
            errors.append("channel_not_ready_without_blocker")
        if not str(report.get("failure_stage") or "").strip():
            errors.append("channel_not_ready_without_failure_stage")

    safety = _dict(report.get("safety"))
    for flag in [
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "weight_tensor_values_public",
        "credentials_public",
        "cookies_public",
        "jupyter_proxy_token_public",
        "private_runtime_state_public",
    ]:
        if safety.get(flag) is not False:
            errors.append(f"safety_flag_mismatch:{flag}")
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_mismatch")

    cleanup = _dict(report.get("cleanup_status"))
    if cleanup.get("temporary_kaggle_kernels_deleted") is not True:
        errors.append("cleanup_kernel_deleted_missing")
    if cleanup.get("temporary_private_packages_removed") is not True:
        errors.append("cleanup_private_packages_removed_missing")
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_left_running")
    if cleanup.get("cookie_file_public") is not False:
        errors.append("cleanup_cookie_public_flag_missing")
    if cleanup.get("storage_state_file_public") is not False:
        errors.append("cleanup_storage_state_public_flag_missing")

    artifacts = _dict(report.get("artifacts"))
    if _dict(artifacts.get("summary_json")).get("present") is not True:
        errors.append("artifact_missing:summary_json")
    return errors


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report = load_json(Path(args.report))
        report_path = args.report
    else:
        report = probe.build_report(
            probe.parse_args(["--output-dir", args.output_dir]),
            small_jax_report={
                "schema": probe.CELL_SCHEMA,
                "cell_kind": "small_jax",
                "ok": False,
                "blockers": ["check_requires_report_or_live_probe"],
                "diagnosis_codes": ["check_requires_report_or_live_probe"],
                "public_artifact_safe": True,
            },
            tiny_qwen_report={
                "schema": probe.CELL_SCHEMA,
                "cell_kind": "tiny_qwen_like",
                "ok": False,
                "blockers": ["check_requires_report_or_live_probe"],
                "diagnosis_codes": ["check_requires_report_or_live_probe"],
                "public_artifact_safe": True,
            },
            output_dir=Path(args.output_dir),
        )
        report_path = str(Path(args.output_dir) / "kaggle_web_tpu_execution_channel_probe.json")
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": report_path,
        "web_tpu_execution_channel_ready": report.get("web_tpu_execution_channel_ready") is True,
        "small_jax_cell_ready": report.get("small_jax_cell_ready") is True,
        "tiny_qwen_like_cell_ready": report.get("tiny_qwen_like_cell_ready") is True,
        "tpu_device_count": report.get("tpu_device_count"),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Kaggle Web TPU execution-channel probe artifact.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=probe.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Check ok: {result['ok']}")
        print(f"Report: {result['report_path']}")
        if result["errors"]:
            print("Errors: " + ", ".join(result["errors"]))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
