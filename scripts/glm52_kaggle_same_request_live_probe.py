#!/usr/bin/env python3
"""Run the GLM 5.2 Kaggle CPU/GPU/TPU same-request live Coordinator attempt."""

from __future__ import annotations

import argparse
import concurrent.futures
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
from scripts import glm52_kaggle_same_request_check as same_request_check  # noqa: E402
from scripts import glm52_kaggle_same_request_probe as same_request_probe  # noqa: E402
from scripts import glm52_kaggle_stage_worker_push_probe as push_probe  # noqa: E402


SCHEMA = "glm52_kaggle_same_request_live_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-same-request-live"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
Runner = Callable[..., subprocess.CompletedProcess[str]]
REQUIRED_PROVIDERS = tuple(same_request_probe.REQUIRED_PROVIDERS)
SENSITIVE_FRAGMENTS = bridge.SENSITIVE_FRAGMENTS + (
    "CT_GLM52_COORDINATOR_TOKEN",
    "X-CrowdTensor-GLM52-Token",
)


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


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def stage_specs_from_package(package_report: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for item in _list(package_report.get("packages")):
        if not isinstance(item, dict):
            continue
        grouped_specs = [spec for spec in _list(item.get("stage_specs")) if isinstance(spec, dict)]
        if grouped_specs:
            for spec in grouped_specs:
                specs.append(
                    {
                        "stage_id": _int(spec.get("stage_id")),
                        "stage_count": _int(spec.get("stage_count"), _int(item.get("stage_count"), len(grouped_specs))),
                        "provider": str(spec.get("provider") or item.get("provider") or ""),
                        "stage_layer_range": _list(spec.get("stage_layer_range")),
                        "compatible_weight_repo": str(
                            spec.get("compatible_weight_repo")
                            or item.get("compatible_weight_repo")
                            or same_request_probe.COMPATIBLE_WEIGHT_REPO
                        ),
                    }
                )
            continue
        specs.append(
            {
                "stage_id": _int(item.get("stage_id")),
                "stage_count": _int(item.get("stage_count"), len(_list(package_report.get("packages")))),
                "provider": str(item.get("provider") or ""),
                "stage_layer_range": _list(item.get("stage_layer_range")),
                "compatible_weight_repo": str(item.get("compatible_weight_repo") or same_request_probe.COMPATIBLE_WEIGHT_REPO),
            }
        )
    return bridge.normalize_stage_specs(specs)


def push_specs_from_package(package_report: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [item for item in _list(package_report.get("packages")) if isinstance(item, dict)]

    def layer_start(item: dict[str, Any]) -> int:
        layer_range = _list(item.get("stage_layer_range"))
        return _int(layer_range[0], _int(item.get("stage_id"))) if len(layer_range) == 2 else _int(item.get("stage_id"))

    return sorted(specs, key=lambda item: (layer_start(item), _int(item.get("stage_id"))))


def provider_list(stage_specs: list[dict[str, Any]]) -> list[str]:
    providers = []
    for spec in stage_specs:
        provider = str(spec.get("provider") or "")
        if provider and provider not in providers:
            providers.append(provider)
    return providers


def stage_id_list(stage_specs: list[dict[str, Any]]) -> list[int]:
    return [_int(spec.get("stage_id")) for spec in stage_specs]


def stage_ids_for_push_spec(spec: dict[str, Any]) -> list[int]:
    ids = [
        _int(item, -1)
        for item in _list(spec.get("stage_ids"))
        if _int(item, -1) >= 0
    ]
    stage_id = _int(spec.get("stage_id"), -1)
    if not ids and stage_id >= 0:
        ids = [stage_id]
    return sorted(set(ids))


def parse_mapping(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_item in str(value or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key:
            mapping[key] = raw_value
    return mapping


def token_config_for_provider(provider: str, args: argparse.Namespace) -> dict[str, str]:
    token_file_map = parse_mapping(getattr(args, "provider_token_file_map", ""))
    token_section_map = parse_mapping(getattr(args, "provider_token_section_map", ""))
    raw_token_file_map = parse_mapping(getattr(args, "provider_raw_token_file_map", ""))
    raw_token_username_map = parse_mapping(getattr(args, "provider_raw_token_username_map", ""))
    return {
        "token_file": token_file_map.get(provider, str(args.token_file)),
        "token_section": token_section_map.get(provider, str(args.token_section)),
        "raw_token_file": raw_token_file_map.get(provider, str(args.raw_token_file)),
        "raw_token_username": raw_token_username_map.get(provider, str(args.raw_token_username)),
    }


def _bool_attr(args: argparse.Namespace, name: str) -> bool:
    return bool(getattr(args, name, False))


def push_args_for_stage(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    stage_worker_package_report: str,
    stage_spec: dict[str, Any],
    coordinator_url: str,
    token_path: Path,
) -> argparse.Namespace:
    provider = str(stage_spec.get("provider") or "")
    stage_ids = stage_ids_for_push_spec(stage_spec)
    stage_id_text = ",".join(str(item) for item in stage_ids)
    token_config = token_config_for_provider(provider, args)
    per_stage_limit = int(args.coordinator_stage_task_limit or args.max_new_tokens)
    package_stage_task_limit = max(1, per_stage_limit) * max(1, len(stage_ids))
    push_args = push_probe.parse_args(
        [
            "--mode",
            "live",
            "--output-dir",
            str(output_dir),
            "--stage-worker-package-report",
            stage_worker_package_report,
            "--providers",
            provider,
            "--stage-ids",
            stage_id_text,
            "--wait-seconds",
            str(args.wait_seconds),
            "--poll-interval-seconds",
            str(args.poll_interval_seconds),
            "--command-timeout-seconds",
            str(args.command_timeout_seconds),
            "--kernel-timeout-seconds",
            str(args.kernel_timeout_seconds),
            "--token-file",
            str(token_config.get("token_file") or ""),
            "--token-section",
            str(token_config.get("token_section") or ""),
            "--raw-token-file",
            str(token_config.get("raw_token_file") or ""),
            "--raw-token-username",
            str(token_config.get("raw_token_username") or ""),
            "--hf-token-env",
            str(getattr(args, "hf_token_env", "") or ""),
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
            "--coordinator-stage-task-limit",
            str(package_stage_task_limit),
            "--full-prefix-prefill-length",
            str(args.full_prefix_prefill_length),
            "--full-prefix-dsa-mask-topk",
            str(args.full_prefix_dsa_mask_topk),
            "--full-prefix-executed-expert-count",
            str(args.full_prefix_executed_expert_count),
            "--full-prefix-top-k",
            str(args.full_prefix_top_k),
            "--full-prefix-row-block-size",
            str(args.full_prefix_row_block_size),
            "--full-prefix-max-tensor-bytes",
            str(args.full_prefix_max_tensor_bytes),
            "--full-prefix-max-block-bytes",
            str(args.full_prefix_max_block_bytes),
            "--cpu-group-stage-attempt-seconds",
            str(args.cpu_group_stage_attempt_seconds),
            "--cpu-group-stage-poll-seconds",
            str(args.cpu_group_stage_poll_seconds),
        ]
    )
    if provider == "kaggle_jax_tpu" and _bool_attr(args, "retain_nonterminal_tpu"):
        push_args.retain_nonterminal_tpu = True
    if provider == "kaggle_cuda" and _bool_attr(args, "retain_nonterminal_gpu"):
        push_args.retain_nonterminal_gpu = True
    if provider == "kaggle_cpu" and _bool_attr(args, "retain_nonterminal_cpu"):
        push_args.retain_nonterminal_cpu = True
    return push_args


def stage_push_verified(report: dict[str, Any]) -> bool:
    pushes = [item for item in _list(report.get("pushes")) if isinstance(item, dict)]
    if report.get("live_run_performed") is not True or not pushes:
        return False
    for push in pushes:
        if push.get("stage_runtime_verified") is not True:
            continue
        stage_report = load_json(str(push.get("stage_report_path") or ""))
        if stage_report.get("stage_decode_verified") is True:
            return True
    return False


def stage_push_decode_blockers(report: dict[str, Any]) -> list[str]:
    blockers: set[str] = set()
    for push in [item for item in _list(report.get("pushes")) if isinstance(item, dict)]:
        provider = str(push.get("provider") or "missing")
        stage_id = _int(push.get("stage_id"), -1)
        stage_report = load_json(str(push.get("stage_report_path") or ""))
        if push.get("stage_runtime_verified") is True and stage_report.get("stage_decode_verified") is not True:
            blockers.add(f"glm52_stage_worker_stage_decode_not_verified:{provider}:stage{stage_id}")
    return sorted(blockers)


def stage_push_exception_report(stage_spec: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    provider = str(stage_spec.get("provider") or "")
    stage_id = _int(stage_spec.get("stage_id"))
    stage_ids = stage_ids_for_push_spec(stage_spec)
    return {
        "schema": push_probe.SCHEMA,
        "generated_at": utc_now(),
        "mode": "live",
        "ok": False,
        "glm52_stage_worker_push_probe_ready": True,
        "live_run_performed": False,
        "stage_runtime_reports_collected": 0,
        "stage_runtime_reports_verified": 0,
        "same_request_route_verified": False,
        "stage_runtime_adapter_verified": False,
        "pushes": [
            {
                "schema": "glm52_kaggle_stage_worker_push_entry_v1",
                "provider": provider,
                "stage_id": stage_id,
                "stage_ids": stage_ids,
                "pushed": False,
                "push_error_blocker": f"stage_push_exception:{type(exc).__name__}",
                "terminal_status": "",
                "output_collected": False,
                "stage_report_path": "",
                "stage_report_present": False,
                "stage_runtime_verified": False,
                "cleanup_performed": False,
                "public_artifact_safe": True,
            }
        ],
        "blockers": [
            "glm52_stage_worker_push_exception",
            f"glm52_stage_worker_push_failed:{provider or 'missing'}",
        ],
        "completion_boundary": {
            "preflight_is_not_runtime_success": True,
            "push_required": True,
            "terminal_kernel_output_required": True,
            "stage_runtime_check_required": True,
            "same_request_probe_required": True,
        },
        "public_artifact_safe": True,
        "safety": same_request_probe.safety_flags(),
}


def stage_ids_from_push(push: dict[str, Any]) -> list[int]:
    ids = [
        _int(item, -1)
        for item in _list(push.get("stage_ids"))
        if _int(item, -1) >= 0
    ]
    stage_id = _int(push.get("stage_id"), -1)
    if not ids and stage_id >= 0:
        ids = [stage_id]
    return sorted(set(ids))


def verified_stage_ids_from_stage_report(report: dict[str, Any]) -> list[int]:
    if report.get("stage_decode_verified") is not True:
        return []
    ids = [
        _int(item, -1)
        for item in _list(report.get("stage_ids_verified") or report.get("stage_ids"))
        if _int(item, -1) >= 0
    ]
    stage_id = _int(report.get("stage_id"), -1)
    if not ids and stage_id >= 0:
        ids = [stage_id]
    return sorted(set(ids))


def merge_push_reports(
    reports: list[dict[str, Any]],
    *,
    output_dir: Path,
    stage_specs: list[dict[str, Any]],
    not_attempted_stage_ids: list[int],
) -> dict[str, Any]:
    pushes: list[dict[str, Any]] = []
    blockers: set[str] = set()
    collected = 0
    verified = 0
    for report in reports:
        pushes.extend(item for item in _list(report.get("pushes")) if isinstance(item, dict))
        blockers.update(str(item) for item in _list(report.get("blockers")) if item)
        collected += _int(report.get("stage_runtime_reports_collected"))
        verified += _int(report.get("stage_runtime_reports_verified"))
        if report.get("ok") is False:
            blockers.add("glm52_stage_worker_push_subreport_not_ok")
    required_stage_ids = set(stage_id_list(stage_specs))
    verified_stage_ids: set[int] = set()
    for push in pushes:
        if push.get("stage_runtime_verified") is not True:
            continue
        stage_report = load_json(str(push.get("stage_report_path") or ""))
        report_stage_ids = verified_stage_ids_from_stage_report(stage_report)
        verified_stage_ids.update(report_stage_ids or stage_ids_from_push(push))
    providers = {str(push.get("provider") or "") for push in pushes if push.get("stage_runtime_verified") is True}
    for provider in REQUIRED_PROVIDERS:
        if provider not in providers:
            blockers.add(f"glm52_stage_worker_push_provider_missing:{provider}")
    missing_verified = sorted(required_stage_ids - verified_stage_ids)
    if missing_verified:
        blockers.add("glm52_stage_worker_live_reports_missing")
        blockers.add("glm52_stage_worker_live_reports_not_verified")
    if not_attempted_stage_ids:
        blockers.add("glm52_stage_worker_push_stopped_before_all_stages")
    return {
        "schema": push_probe.SCHEMA,
        "generated_at": utc_now(),
        "mode": "live",
        "ok": True,
        "glm52_stage_worker_push_probe_ready": True,
        "live_run_performed": bool(reports),
        "stage_runtime_reports_collected": collected,
        "stage_runtime_reports_verified": verified,
        "same_request_route_verified": False,
        "stage_runtime_adapter_verified": False,
        "pushes": pushes,
        "attempted_stage_ids": sorted({stage_id for push in pushes for stage_id in stage_ids_from_push(push)}),
        "not_attempted_stage_ids": list(not_attempted_stage_ids),
        "missing_verified_stage_ids": missing_verified,
        "stage_worker_subreport_count": len(reports),
        "stage_worker_subreport_dir": str(output_dir),
        "blockers": sorted(blockers),
        "completion_boundary": {
            "preflight_is_not_runtime_success": True,
            "push_required": True,
            "terminal_kernel_output_required": True,
            "stage_runtime_check_required": True,
            "same_request_probe_required": True,
        },
        "public_artifact_safe": True,
        "safety": same_request_probe.safety_flags(),
    }


def cleanup_from_push(push_report: dict[str, Any]) -> dict[str, Any]:
    pushes = [item for item in _list(push_report.get("pushes")) if isinstance(item, dict)]
    accepted_pushes = [item for item in pushes if item.get("pushed") is True]
    cleaned = bool(pushes) and all(item.get("cleanup_performed") is True for item in accepted_pushes)
    retained = [
        {
            "provider": str(item.get("provider") or ""),
            "stage_id": _int(item.get("stage_id")),
            "terminal_status": str(item.get("terminal_status") or ""),
        }
        for item in accepted_pushes
        if item.get("cleanup_performed") is not True
    ]
    blockers = []
    if not cleaned:
        blockers.append("glm52_same_request_live_cleanup_incomplete")
    return {
        "schema": "glm52_kaggle_same_request_live_cleanup_v1",
        "temporary_kaggle_kernels_deleted": cleaned,
        "temporary_private_packages_removed": cleaned,
        "live_resources_left_running": False if cleaned else bool(retained),
        "retained_or_uncleaned_kernels": retained,
        "blockers": blockers,
        "public_artifact_safe": True,
    }


def collected_stage_reports(push_report: dict[str, Any]) -> list[dict[str, Any]]:
    reports = []
    for item in _list(push_report.get("pushes")):
        if not isinstance(item, dict):
            continue
        path = str(item.get("stage_report_path") or "")
        report = load_json(path)
        if report:
            reports.append(report)
    return reports


def build_same_request_report(
    output_dir: Path,
    *,
    stage_reports: list[dict[str, Any]],
    coordinator_report: dict[str, Any],
    cleanup_report: dict[str, Any],
) -> dict[str, Any]:
    args = same_request_probe.parse_args(["--mode", "assemble", "--output-dir", str(output_dir)])
    report = same_request_probe.build_report(
        args,
        stage_reports=stage_reports,
        coordinator_report=coordinator_report,
        cleanup_report=cleanup_report,
    )
    write_json(output_dir / "glm52_kaggle_same_request_probe.json", report)
    return report


def run_live(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    package_report = load_json(args.stage_worker_package_report)
    stage_specs = stage_specs_from_package(package_report)
    push_specs = push_specs_from_package(package_report)
    blockers: set[str] = set()
    if not stage_specs:
        blockers.add("glm52_live_stage_specs_missing")
        stage_specs = bridge.default_stage_specs()
    if not push_specs:
        blockers.add("glm52_live_push_specs_missing")
        push_specs = stage_specs
    request_hash = str(
        args.coordinator_request_id_hash
        or package_report.get("coordinator_request_id_hash")
        or bridge.sha_json({"glm52_live_request": utc_now()})
    )
    token = secrets.token_urlsafe(32)
    state = bridge.Glm52CoordinatorState(
        stage_specs=stage_specs,
        coordinator_request_id_hash=request_hash,
        max_new_tokens=max(1, int(args.max_new_tokens)),
    )
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
        prefix="glm52-coordinator-token-",
        delete=False,
    )
    token_path = Path(token_file.name)
    try:
        token_file.write(token)
        token_file.close()
        push_output_dir = output_dir / "stage-worker-push"
        subreports: list[dict[str, Any]] = []
        not_attempted_stage_ids: list[int] = []
        concurrent_pushes = bool(args.concurrent_stage_pushes or int(args.max_new_tokens) > 1)

        def run_stage_push(spec: dict[str, Any]) -> dict[str, Any]:
            stage_ids = stage_ids_for_push_spec(spec)
            stage_id = stage_ids[0] if stage_ids else _int(spec.get("stage_id"))
            stage_label = str(stage_id) if len(stage_ids) <= 1 else f"{stage_ids[0]}-{stage_ids[-1]}"
            stage_dir = push_output_dir / f"stage-{stage_label}-{str(spec.get('provider') or 'missing')}"
            push_args = push_args_for_stage(
                args,
                output_dir=stage_dir,
                stage_worker_package_report=str(args.stage_worker_package_report),
                stage_spec=spec,
                coordinator_url=coordinator_url,
                token_path=token_path,
            )
            completed_before = _int(state.public_status().get("completed_task_count"))
            try:
                stage_report = push_probe.build_report(push_args, runner=runner)
            except Exception as exc:  # pragma: no cover - defensive live artifact path.
                stage_report = stage_push_exception_report(spec, exc)
            stage_blockers = set(stage_push_decode_blockers(stage_report))
            completed_after = _int(state.public_status().get("completed_task_count"))
            if completed_after <= completed_before:
                provider = str(spec.get("provider") or "missing")
                stage_blockers.add(f"glm52_stage_worker_coordinator_submit_missing:{provider}:stage{stage_id}")
            if stage_blockers:
                stage_report["blockers"] = sorted(set(_list(stage_report.get("blockers"))) | stage_blockers)
            write_json(stage_dir / "glm52_kaggle_stage_worker_push_probe.json", stage_report)
            return stage_report

        if concurrent_pushes:
            max_workers = max(1, int(args.stage_push_parallelism or len(push_specs)))
            future_map: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for spec in push_specs:
                    future_map[executor.submit(run_stage_push, spec)] = spec
                for future in concurrent.futures.as_completed(future_map):
                    spec = future_map[future]
                    try:
                        subreports.append(future.result())
                    except Exception as exc:  # pragma: no cover - defensive live artifact path.
                        stage_report = stage_push_exception_report(spec, exc)
                        stage_ids = stage_ids_for_push_spec(spec)
                        stage_id = stage_ids[0] if stage_ids else _int(spec.get("stage_id"))
                        stage_label = str(stage_id) if len(stage_ids) <= 1 else f"{stage_ids[0]}-{stage_ids[-1]}"
                        stage_dir = push_output_dir / f"stage-{stage_label}-{str(spec.get('provider') or 'missing')}"
                        write_json(stage_dir / "glm52_kaggle_stage_worker_push_probe.json", stage_report)
                        subreports.append(stage_report)
        else:
            stop_after_failure = False
            for spec in push_specs:
                stage_ids = stage_ids_for_push_spec(spec)
                stage_id = stage_ids[0] if stage_ids else _int(spec.get("stage_id"))
                if stop_after_failure:
                    not_attempted_stage_ids.extend(stage_ids or [stage_id])
                    continue
                stage_report = run_stage_push(spec)
                subreports.append(stage_report)
                stage_ok = bool(stage_push_verified(stage_report))
                if not stage_ok and not args.continue_after_stage_failure:
                    stop_after_failure = True
        push_report = merge_push_reports(
            subreports,
            output_dir=push_output_dir,
            stage_specs=stage_specs,
            not_attempted_stage_ids=not_attempted_stage_ids,
        )
        write_json(push_output_dir / "glm52_kaggle_stage_worker_push_probe.json", push_report)
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

    coordinator_report = state.coordinator_report()
    cleanup_report = cleanup_from_push(push_report)
    worker_stage_reports = collected_stage_reports(push_report)
    coordinator_stage_reports = state.same_request_stage_reports()
    same_request_dir = output_dir / "same-request"
    same_request = build_same_request_report(
        same_request_dir,
        stage_reports=coordinator_stage_reports,
        coordinator_report=coordinator_report,
        cleanup_report=cleanup_report,
    )
    same_errors = same_request_check.validate_report(same_request, require_verified=True)
    blockers.update(str(item) for item in _list(push_report.get("blockers")) if item)
    blockers.update(str(item) for item in _list(same_request.get("blockers")) if item)
    blockers.update(str(item) for item in _list(cleanup_report.get("blockers")) if item)
    if same_errors:
        blockers.add("glm52_same_request_live_check_not_verified")
    status = state.public_status()
    required_stage_ids = set(stage_id_list(stage_specs))
    coordinator_stage_ids = {
        _int(item.get("stage_id"), -1)
        for item in coordinator_stage_reports
        if _int(item.get("stage_id"), -1) >= 0
    }
    worker_verified_stage_ids = {
        stage_id
        for item in worker_stage_reports
        for stage_id in verified_stage_ids_from_stage_report(item)
    }
    expected_task_count = len(stage_specs) * max(1, int(args.max_new_tokens))
    worker_verified_task_count = 0
    for item in worker_stage_reports:
        if item.get("stage_decode_verified") is not True:
            continue
        coordinator_decode = _dict(item.get("coordinator_decode"))
        worker_verified_task_count += max(
            1,
            _int(
                item.get("coordinator_stage_tasks_accepted")
                or coordinator_decode.get("coordinator_stage_tasks_accepted")
            ),
        )
    full_stage_count_verified = bool(
        len(coordinator_stage_reports) == expected_task_count
        and coordinator_stage_ids == required_stage_ids
        and worker_verified_stage_ids == required_stage_ids
        and worker_verified_task_count >= expected_task_count
        and _int(status.get("completed_task_count")) >= expected_task_count
        and _int(status.get("generated_token_count")) >= max(1, int(args.max_new_tokens))
    )
    if not full_stage_count_verified:
        blockers.add("glm52_live_full_stage_count_not_verified")
    verified = bool(
        same_request.get("same_request_decode_verified") is True
        and not same_errors
        and full_stage_count_verified
    )
    worker_collected_stage_ids = {
        stage_id
        for item in worker_stage_reports
        for stage_id in (
            [
                _int(candidate, -1)
                for candidate in _list(item.get("stage_ids_verified") or item.get("stage_ids"))
                if _int(candidate, -1) >= 0
            ]
            or ([_int(item.get("stage_id"), -1)] if _int(item.get("stage_id"), -1) >= 0 else [])
        )
    }
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": "live",
        "model_id": same_request_probe.MODEL_ID,
        "compatible_weight_repo": same_request_probe.COMPATIBLE_WEIGHT_REPO,
        "target_generated_token_count": max(1, int(args.max_new_tokens)),
        "runtime_tuning": {
            "full_prefix_prefill_length": int(args.full_prefix_prefill_length),
            "full_prefix_dsa_mask_topk": int(args.full_prefix_dsa_mask_topk),
            "full_prefix_executed_expert_count": int(args.full_prefix_executed_expert_count),
            "full_prefix_top_k": int(args.full_prefix_top_k),
            "full_prefix_row_block_size": int(args.full_prefix_row_block_size),
            "full_prefix_max_tensor_bytes": int(args.full_prefix_max_tensor_bytes),
            "full_prefix_max_block_bytes": int(args.full_prefix_max_block_bytes),
            "cpu_group_stage_attempt_seconds": float(args.cpu_group_stage_attempt_seconds),
            "cpu_group_stage_poll_seconds": float(args.cpu_group_stage_poll_seconds),
        },
        "expected_stage_task_count": expected_task_count,
        "coordinator_public_url_present": bool(coordinator_url),
        "coordinator_url_public": False,
        "coordinator_token_public": False,
        "coordinator_request_id_hash": request_hash,
        "stage_count": len(stage_specs),
        "stage_order": status.get("stage_order"),
        "push_report_path": str(push_output_dir / "glm52_kaggle_stage_worker_push_probe.json"),
        "coordinator_report_path": str(output_dir / "glm52_kaggle_coordinator_report.json"),
        "cleanup_report_path": str(output_dir / "glm52_kaggle_cleanup_report.json"),
        "same_request_report_path": str(same_request_dir / "glm52_kaggle_same_request_probe.json"),
        "same_request_decode_verified": verified,
        "generated_token_count": _int(same_request.get("generated_token_count")),
        "generated_token_hashes": _list(status.get("generated_token_hashes")),
        "accepted_providers": _list(same_request.get("accepted_providers")),
        "stage_runtime_reports_collected": len(worker_collected_stage_ids),
        "stage_runtime_reports_verified": len(worker_verified_stage_ids),
        "stage_worker_package_reports_collected": _int(push_report.get("stage_runtime_reports_collected")),
        "stage_worker_package_reports_verified": _int(push_report.get("stage_runtime_reports_verified")),
        "coordinator_stage_reports_collected": len(coordinator_stage_reports),
        "worker_stage_decode_reports_collected": len(worker_collected_stage_ids),
        "worker_stage_decode_task_count": worker_verified_task_count,
        "full_stage_count_verified": full_stage_count_verified,
        "coordinator_status": status,
        "cleanup_status": cleanup_report,
        "same_request_check": {
            "ok": not same_errors,
            "error_count": len(same_errors),
            "errors": same_errors,
        },
        "blockers": [] if verified else sorted(blockers),
        "completion_boundary": {
            "requires_verified_same_request_report": True,
            "requires_generated_token_hash": True,
            "requires_cleanup_evidence": True,
            "requires_all_three_kaggle_provider_families": True,
        },
        "safety": same_request_probe.safety_flags(),
        "public_artifact_safe": True,
    }
    write_json(output_dir / "glm52_kaggle_coordinator_report.json", coordinator_report)
    write_json(output_dir / "glm52_kaggle_cleanup_report.json", cleanup_report)
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["same_request_decode_verified"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-worker-package-report", required=True)
    parser.add_argument("--coordinator-request-id-hash", default="")
    parser.add_argument("--coordinator-bind-host", default="0.0.0.0")
    parser.add_argument("--coordinator-port", type=int, default=0)
    parser.add_argument("--coordinator-public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--coordinator-public-url", default="")
    parser.add_argument("--coordinator-task-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--coordinator-worker-poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--coordinator-stage-task-limit", type=int, default=0)
    parser.add_argument("--full-prefix-prefill-length", type=int, default=0)
    parser.add_argument("--full-prefix-dsa-mask-topk", type=int, default=0)
    parser.add_argument("--full-prefix-executed-expert-count", type=int, default=0)
    parser.add_argument("--full-prefix-top-k", type=int, default=0)
    parser.add_argument("--full-prefix-row-block-size", type=int, default=0)
    parser.add_argument("--full-prefix-max-tensor-bytes", type=int, default=0)
    parser.add_argument("--full-prefix-max-block-bytes", type=int, default=0)
    parser.add_argument("--cpu-group-stage-attempt-seconds", type=float, default=0.0)
    parser.add_argument("--cpu-group-stage-poll-seconds", type=float, default=0.0)
    parser.add_argument("--concurrent-stage-pushes", action="store_true")
    parser.add_argument("--stage-push-parallelism", type=int, default=0)
    parser.add_argument("--wait-seconds", type=float, default=7200.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--command-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=9000)
    parser.add_argument("--token-file", default="~/.config/crowdtensor/kaggle-tokens.md")
    parser.add_argument("--token-section", default="cpuowner")
    parser.add_argument("--raw-token-file", default="")
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument("--hf-token-env", default="HF_TOKEN,HUGGING_FACE_HUB_TOKEN")
    parser.add_argument("--provider-token-file-map", default="")
    parser.add_argument("--provider-token-section-map", default="")
    parser.add_argument("--provider-raw-token-file-map", default="")
    parser.add_argument("--provider-raw-token-username-map", default="")
    parser.add_argument("--gpu-accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--tpu-accelerator", default="tpuV5e8")
    parser.add_argument("--retain-nonterminal-tpu", action="store_true")
    parser.add_argument("--retain-nonterminal-gpu", action="store_true")
    parser.add_argument("--retain-nonterminal-cpu", action="store_true")
    parser.add_argument("--continue-after-stage-failure", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1")
    if args.coordinator_stage_task_limit < 0:
        raise SystemExit("--coordinator-stage-task-limit must be non-negative")
    for name in [
        "full_prefix_prefill_length",
        "full_prefix_dsa_mask_topk",
        "full_prefix_executed_expert_count",
        "full_prefix_top_k",
        "full_prefix_row_block_size",
        "full_prefix_max_tensor_bytes",
        "full_prefix_max_block_bytes",
    ]:
        if int(getattr(args, name)) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    for name in ["cpu_group_stage_attempt_seconds", "cpu_group_stage_poll_seconds"]:
        if float(getattr(args, name)) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.stage_push_parallelism < 0:
        raise SystemExit("--stage-push-parallelism must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = run_live(args)
    path = output_dir / "glm52_kaggle_same_request_live_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Same-request verified: {report.get('same_request_decode_verified')}")
        print(f"Generated tokens: {report.get('generated_token_count')}")
    return 0 if report.get("same_request_decode_verified") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
