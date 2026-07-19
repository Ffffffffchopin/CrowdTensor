#!/usr/bin/env python3
"""Run a single GLM 5.2 Kaggle stage against a seeded Coordinator task.

This is a diagnostic probe for failed middle stages in the full same-request
live run. It does not claim GLM 5.2 deployment RC success: the Coordinator is
seeded with a synthetic private activation for the previous stage, then exactly
one target stage is pushed to Kaggle and asked to submit to the Coordinator.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import glm52_kaggle_coordinator_decode_bridge_probe as bridge  # noqa: E402
from scripts import glm52_kaggle_same_request_live_probe as live_probe  # noqa: E402
from scripts import glm52_kaggle_same_request_probe as same_request_probe  # noqa: E402
from scripts import glm52_kaggle_stage_worker_push_probe as push_probe  # noqa: E402


SCHEMA = "glm52_kaggle_seeded_stage_diagnostic_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-seeded-stage-diagnostic"
Runner = Callable[..., subprocess.CompletedProcess[str]]
SENSITIVE_FRAGMENTS = bridge.SENSITIVE_FRAGMENTS + push_probe.SENSITIVE_FRAGMENTS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def parse_hidden_shape(value: str) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return [3, 6144]
    if text.startswith("["):
        loaded = json.loads(text)
        if isinstance(loaded, list):
            shape = [_int(item) for item in loaded]
        else:
            shape = []
    else:
        shape = [_int(item) for item in text.split(",") if str(item).strip()]
    if len(shape) != 2 or any(item <= 0 for item in shape):
        raise SystemExit("--seed-hidden-shape must contain two positive dimensions")
    return shape


def dtype_size(dtype: str) -> int:
    normalized = str(dtype or "").lower()
    if normalized in {"float16", "bfloat16"}:
        return 2
    if normalized == "float32":
        return 4
    raise SystemExit("--seed-hidden-dtype must be float16, bfloat16, or float32")


def stage_specs_from_package(package_report: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    packages = _list(package_report.get("packages"))
    for item in packages:
        if not isinstance(item, dict):
            continue
        specs.append(
            {
                "stage_id": _int(item.get("stage_id")),
                "stage_count": _int(item.get("stage_count"), len(packages)),
                "provider": str(item.get("provider") or ""),
                "stage_layer_range": _list(item.get("stage_layer_range")),
                "compatible_weight_repo": str(item.get("compatible_weight_repo") or same_request_probe.COMPATIBLE_WEIGHT_REPO),
            }
        )
    return bridge.normalize_stage_specs(specs)


def window_for_target(stage_specs: list[dict[str, Any]], target_stage_id: int) -> list[dict[str, Any]]:
    for index, spec in enumerate(stage_specs):
        if _int(spec.get("stage_id"), -1) != int(target_stage_id):
            continue
        if index <= 0:
            raise SystemExit("target_stage_requires_previous_stage")
        if index + 1 >= len(stage_specs):
            raise SystemExit("target_stage_requires_next_stage_for_nonfinal_diagnostic")
        return [stage_specs[index - 1], spec, stage_specs[index + 1]]
    raise SystemExit("target_stage_not_found_in_package")


def seeded_activation(*, shape: list[int], dtype: str, label: str) -> dict[str, Any]:
    byte_count = int(shape[0]) * int(shape[1]) * dtype_size(dtype)
    raw = bytes(byte_count)
    activation_hash = sha_bytes(
        json.dumps({"label": label, "shape": shape, "dtype": dtype}, sort_keys=True).encode("utf-8") + raw
    )
    return {
        "schema": "glm52_private_stage_activation_v1",
        "activation_hash": activation_hash,
        "hidden_shape": list(shape),
        "hidden_dtype": str(dtype),
        "hidden_b64": base64.b64encode(raw).decode("ascii"),
        "activation_public": False,
    }


def seed_previous_stage(state: bridge.Glm52CoordinatorState, previous_spec: dict[str, Any], activation: dict[str, Any]) -> dict[str, Any]:
    previous_stage_id = _int(previous_spec.get("stage_id"))
    claimed = state.claim(miner_id="glm52-seeded-diagnostic-previous-stage", stage_id=previous_stage_id)
    task = _dict(claimed.get("task"))
    if not task:
        return {"ok": False, "reason": "previous_stage_task_not_claimed", "claim": claimed}
    payload = {
        "task_id": str(task.get("task_id") or ""),
        "stage_id": previous_stage_id,
        "generation_step": _int(task.get("generation_step")),
        "public_artifact_safe": True,
        "stage_decode_verified": True,
        "stage_output_hash": sha_json({"seeded_stage_output": previous_stage_id, "activation": activation.get("activation_hash")}),
        "output_hash": sha_json({"seeded_output": previous_stage_id}),
        "weight_value_sha256": sha_json({"seeded_weight": previous_stage_id}),
        "weight_value_byte_count": 16,
        "provider_runtime_verified": True,
        "provider_device_count": 1,
        "stage_decode_report_hash": sha_json({"seeded_stage_decode_report": previous_stage_id}),
        "duration_seconds": 0.0,
        "kv_cache": {
            "schema": "glm52_stage_local_cache_summary_v1",
            "stage_id": previous_stage_id,
            "ready": True,
            "cache_tensors_public": False,
            "past_key_values_public": False,
        },
        "activation": activation,
        "activation_hash": str(activation.get("activation_hash") or ""),
    }
    submitted = state.submit(payload)
    return {
        "ok": submitted.get("accepted") is True,
        "previous_stage_id": previous_stage_id,
        "previous_stage_layer_range": _list(previous_spec.get("stage_layer_range")),
        "seed_activation_hash": str(activation.get("activation_hash") or ""),
        "seed_hidden_shape": _list(activation.get("hidden_shape")),
        "seed_hidden_dtype": str(activation.get("hidden_dtype") or ""),
        "claim_ok": claimed.get("ok") is True,
        "submit_accepted": submitted.get("accepted") is True,
        "submit_ready": submitted.get("ready") is True,
        "submit_reason": str(submitted.get("reason") or ""),
        "public_artifact_safe": True,
    }


def push_args_for_target(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    target_spec: dict[str, Any],
    coordinator_url: str,
    token_path: Path,
) -> argparse.Namespace:
    return push_probe.parse_args(
        [
            "--mode",
            "live",
            "--output-dir",
            str(output_dir),
            "--stage-worker-package-report",
            str(args.stage_worker_package_report),
            "--providers",
            str(target_spec.get("provider") or ""),
            "--stage-ids",
            str(_int(target_spec.get("stage_id"))),
            "--wait-seconds",
            str(args.wait_seconds),
            "--poll-interval-seconds",
            str(args.poll_interval_seconds),
            "--command-timeout-seconds",
            str(args.command_timeout_seconds),
            "--kernel-timeout-seconds",
            str(args.kernel_timeout_seconds),
            "--token-file",
            str(args.token_file),
            "--token-section",
            str(args.token_section),
            "--raw-token-file",
            str(args.raw_token_file),
            "--raw-token-username",
            str(args.raw_token_username),
            "--gpu-accelerator",
            str(args.gpu_accelerator),
            "--tpu-accelerator",
            str(args.tpu_accelerator),
            "--coordinator-url",
            coordinator_url,
            "--coordinator-token-file",
            str(token_path),
            "--coordinator-task-timeout-seconds",
            str(args.coordinator_task_timeout_seconds),
            "--coordinator-poll-interval-seconds",
            str(args.coordinator_worker_poll_interval_seconds),
        ]
    )


def first_push(push_report: dict[str, Any]) -> dict[str, Any]:
    pushes = [item for item in _list(push_report.get("pushes")) if isinstance(item, dict)]
    return pushes[0] if pushes else {}


def selected_stage_diagnostics(stage_report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "ok",
        "provider",
        "stage_id",
        "stage_layer_range",
        "stage_decode_verified",
        "stage_execution_verified",
        "stage_full_decode_verified",
        "stage_runtime_adapter_verified",
        "same_request_route_verified",
        "stage_output_hash",
        "blockers",
        "coordinator_stage_decode_verified",
        "coordinator_stage_tasks_accepted",
        "coordinator_stage_last_submit_accepted",
        "coordinator_stage_last_submit_ready",
        "coordinator_stage_last_full_prefix_layer_range",
        "coordinator_stage_last_full_prefix_adapter_verified",
        "coordinator_stage_last_full_prefix_blocker",
        "coordinator_stage_last_full_prefix_probe_exit_code",
        "coordinator_stage_last_full_prefix_probe_ready",
        "coordinator_stage_last_full_prefix_input_activation_consumed",
        "coordinator_stage_last_full_prefix_input_activation_hash",
        "coordinator_stage_last_full_prefix_output_activation_private_ready",
        "coordinator_stage_last_full_prefix_output_activation_hash",
        "coordinator_stage_last_full_prefix_stdout_hash",
        "coordinator_stage_last_full_prefix_stderr_hash",
        "coordinator_stage_last_full_prefix_probe_blockers",
        "coordinator_stage_last_full_prefix_probe_errors",
    ]
    return {key: stage_report.get(key) for key in keys if key in stage_report}


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    package_report = load_json(args.stage_worker_package_report)
    stage_specs = stage_specs_from_package(package_report)
    window = window_for_target(stage_specs, int(args.stage_id))
    previous_spec, target_spec, next_spec = window
    request_hash = str(
        args.coordinator_request_id_hash
        or package_report.get("coordinator_request_id_hash")
        or bridge.sha_json({"glm52_seeded_stage_diagnostic": int(args.stage_id)})
    )
    state = bridge.Glm52CoordinatorState(
        stage_specs=window,
        coordinator_request_id_hash=request_hash,
        max_new_tokens=1,
    )
    activation = seeded_activation(
        shape=parse_hidden_shape(args.seed_hidden_shape),
        dtype=str(args.seed_hidden_dtype),
        label=f"stage{_int(previous_spec.get('stage_id'))}-to-stage{_int(target_spec.get('stage_id'))}",
    )
    seed_report = seed_previous_stage(state, previous_spec, activation)
    token = secrets.token_urlsafe(32)
    server = bridge.Glm52CoordinatorServer(
        host=str(args.coordinator_bind_host),
        port=int(args.coordinator_port),
        token=token,
        state=state,
    )
    server.start()
    coordinator_url = str(args.coordinator_public_url or "").strip()
    if not coordinator_url:
        coordinator_url = f"http://{args.coordinator_public_host}:{server.port}"
    token_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="glm52-seeded-diagnostic-token-",
        delete=False,
    )
    token_path = Path(token_file.name)
    push_report: dict[str, Any] = {}
    try:
        token_file.write(token)
        token_file.close()
        if seed_report.get("ok") is True:
            push_output_dir = output_dir / "stage-worker-push"
            push_args = push_args_for_target(
                args,
                output_dir=push_output_dir,
                target_spec=target_spec,
                coordinator_url=coordinator_url,
                token_path=token_path,
            )
            push_report = push_probe.build_report(push_args, runner=runner)
            write_json(push_output_dir / "glm52_kaggle_stage_worker_push_probe.json", push_report)
        else:
            push_report = {
                "schema": push_probe.SCHEMA,
                "generated_at": utc_now(),
                "mode": "live",
                "ok": False,
                "live_run_performed": False,
                "stage_runtime_reports_collected": 0,
                "stage_runtime_reports_verified": 0,
                "pushes": [],
                "blockers": ["glm52_seed_previous_stage_failed"],
                "public_artifact_safe": True,
            }
    finally:
        try:
            token_file.close()
        except Exception:
            pass
        try:
            token_path.unlink()
        except OSError:
            pass
        server.stop()

    status = state.public_status()
    push = first_push(push_report)
    stage_report_path = str(push.get("stage_report_path") or "")
    stage_report = load_json(stage_report_path)
    target_stage_id = _int(target_spec.get("stage_id"))
    coordinator_completed_target = bool(
        _int(status.get("stage_task_counts", {}).get(f"stage{target_stage_id}")) >= 1
    )
    target_push_verified = live_probe.stage_push_verified(push_report)
    target_stage_diagnostic_verified = bool(
        target_push_verified
        and coordinator_completed_target
        and stage_report.get("same_request_route_verified") is True
    )
    blockers = set(str(item) for item in _list(push_report.get("blockers")) if item)
    if seed_report.get("ok") is not True:
        blockers.add("glm52_seed_previous_stage_failed")
    if not target_push_verified:
        blockers.add("glm52_seeded_target_stage_push_not_verified")
    if not coordinator_completed_target:
        blockers.add("glm52_seeded_target_stage_coordinator_submit_missing")
    if stage_report and stage_report.get("same_request_route_verified") is not True:
        blockers.add("glm52_seeded_target_stage_same_request_route_not_verified")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": "seeded_stage_diagnostic",
        "model_id": same_request_probe.MODEL_ID,
        "compatible_weight_repo": same_request_probe.COMPATIBLE_WEIGHT_REPO,
        "target_stage_id": target_stage_id,
        "target_provider": str(target_spec.get("provider") or ""),
        "target_stage_layer_range": _list(target_spec.get("stage_layer_range")),
        "previous_stage_id": _int(previous_spec.get("stage_id")),
        "next_stage_id": _int(next_spec.get("stage_id")),
        "coordinator_public_url_present": bool(coordinator_url),
        "coordinator_url_public": False,
        "coordinator_token_public": False,
        "coordinator_request_id_hash": request_hash,
        "seed_report": seed_report,
        "coordinator_status": status,
        "push_report_path": str(output_dir / "stage-worker-push" / "glm52_kaggle_stage_worker_push_probe.json"),
        "target_stage_report_path": stage_report_path,
        "target_stage_report_present": bool(stage_report),
        "target_stage_report_diagnostics": selected_stage_diagnostics(stage_report),
        "target_stage_push_verified": target_push_verified,
        "target_stage_coordinator_completed": coordinator_completed_target,
        "target_stage_diagnostic_verified": target_stage_diagnostic_verified,
        "same_request_decode_verified": False,
        "generated_token_count": 0,
        "blockers": [] if target_stage_diagnostic_verified else sorted(blockers),
        "completion_boundary": {
            "seeded_diagnostic_is_not_deployment_rc": True,
            "synthetic_previous_activation_used": True,
            "requires_full_cpu_gpu_tpu_same_request_rc_for_goal": True,
        },
        "safety": same_request_probe.safety_flags(),
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["target_stage_diagnostic_verified"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-worker-package-report", required=True)
    parser.add_argument("--stage-id", type=int, default=22)
    parser.add_argument("--seed-hidden-shape", default="3,6144")
    parser.add_argument("--seed-hidden-dtype", default="float16")
    parser.add_argument("--coordinator-request-id-hash", default="")
    parser.add_argument("--coordinator-bind-host", default="0.0.0.0")
    parser.add_argument("--coordinator-port", type=int, default=0)
    parser.add_argument("--coordinator-public-host", default=live_probe.DEFAULT_PUBLIC_HOST)
    parser.add_argument("--coordinator-public-url", default="")
    parser.add_argument("--coordinator-task-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--coordinator-worker-poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--wait-seconds", type=float, default=7200.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--command-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=9000)
    parser.add_argument("--token-file", default="~/.config/crowdtensor/kaggle-tokens.md")
    parser.add_argument("--token-section", default="cpuowner")
    parser.add_argument("--raw-token-file", default="")
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument("--gpu-accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--tpu-accelerator", default="tpuV5e8")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_kaggle_seeded_stage_diagnostic_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Target stage diagnostic verified: {report.get('target_stage_diagnostic_verified')}")
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
