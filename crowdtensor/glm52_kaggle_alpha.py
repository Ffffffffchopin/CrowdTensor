"""GLM 5.2 Kaggle CPU/GPU/TPU Alpha service helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from scripts import glm52_kaggle_same_request_live_probe as live_probe


SCHEMA = "glm52_kaggle_alpha_service_v1"
STATUS_SCHEMA = "glm52_kaggle_alpha_status_v1"
GENERATE_SCHEMA = "glm52_kaggle_alpha_generate_response_v1"
CLEANUP_SCHEMA = "glm52_kaggle_alpha_cleanup_v1"
RESUME_PRIVATE_INPUTS_SCHEMA = "glm52_kaggle_alpha_resume_private_inputs_v1"
MODEL_ID = live_probe.same_request_probe.MODEL_ID
COMPATIBLE_WEIGHT_REPO = live_probe.same_request_probe.COMPATIBLE_WEIGHT_REPO
SUPPORTED_MODEL_REQUESTS = (COMPATIBLE_WEIGHT_REPO, MODEL_ID)
REQUIRED_ACCELERATORS = ("cpu", "gpu", "tpu")
SUPPORTED_ACCELERATORS = REQUIRED_ACCELERATORS
KAGGLE_RUNTIME_BLOCKER_CLASSES = (
    "kaggle_kernel_push_timeout",
    "kaggle_kernel_push_http_429",
    "kaggle_kernel_push_empty_response",
    "kaggle_kernel_status_timeout",
    "kaggle_kernel_wait_timeout",
    "kaggle_kernel_output_timeout",
    "kaggle_kernel_output_http_429",
    "kaggle_kernel_output_empty_response",
    "kaggle_kernel_output_stage_report_missing",
    "kaggle_kernel_terminal_error",
    "kaggle_kernel_terminal_cancelled",
    "kaggle_kernel_delete_timeout",
)
DEFAULT_STAGE_WORKER_PACKAGE_REPORT = (
    "dist/glm52-kaggle-stage-worker-package-20260707-r209-r5-hf-fetch-retries/"
    "glm52_kaggle_stage_worker_package.json"
)
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-alpha-service"
SENSITIVE_FRAGMENTS = live_probe.SENSITIVE_FRAGMENTS + (
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"raw_generated_text":',
    '"generated_token_ids":',
)
Runner = Callable[..., subprocess.CompletedProcess[str]]
GenerateFn = Callable[["AlphaConfig", dict[str, Any]], dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_utc_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def safety_flags() -> dict[str, bool]:
    return {
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
    }


def normalize_model_request(value: Any) -> str:
    return str(value or COMPATIBLE_WEIGHT_REPO).strip() or COMPATIBLE_WEIGHT_REPO


def model_request_supported(value: Any) -> bool:
    return normalize_model_request(value) in SUPPORTED_MODEL_REQUESTS


def normalize_accelerator_request(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = REQUIRED_ACCELERATORS
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_item in raw_items:
        item = str(raw_item or "").strip().lower()
        if item and item not in seen:
            normalized.append(item)
            seen.add(item)
    if not normalized:
        normalized = list(REQUIRED_ACCELERATORS)
    ordered = [item for item in REQUIRED_ACCELERATORS if item in seen]
    ordered.extend(item for item in normalized if item not in REQUIRED_ACCELERATORS)
    return ordered


def accelerator_request_status(value: Any) -> dict[str, Any]:
    requested = normalize_accelerator_request(value)
    requested_set = set(requested)
    missing = [item for item in REQUIRED_ACCELERATORS if item not in requested_set]
    unsupported = [item for item in requested if item not in SUPPORTED_ACCELERATORS]
    return {
        "requested": requested,
        "required": list(REQUIRED_ACCELERATORS),
        "missing_required": missing,
        "unsupported": unsupported,
        "all_required_present": not missing,
        "supported": not unsupported,
        "complete": not missing and not unsupported,
    }


def hf_token_env_names(value: Any) -> list[str]:
    raw_items = str(value or "HF_TOKEN,HUGGING_FACE_HUB_TOKEN").split(",")
    names: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        name = str(raw_item or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def hf_token_env_status(value: Any) -> dict[str, Any]:
    names = hf_token_env_names(value)
    configured = [name for name in names if os.environ.get(name)]
    return {
        "hf_token_env_supported": True,
        "hf_token_env_count": len(names),
        "hf_token_env_name_hashes": [sha_text(name) for name in names],
        "hf_token_env_configured": bool(configured),
        "hf_token_env_configured_count": len(configured),
        "hf_token_public": False,
    }


def resume_private_inputs_status(hf_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RESUME_PRIVATE_INPUTS_SCHEMA,
        "required_for_live_resume": True,
        "resume_command_omits_private_credentials": True,
        "kaggle_credentials_required": True,
        "kaggle_credential_values_public": False,
        "kaggle_token_file_paths_public": False,
        "kaggle_token_section_names_public": False,
        "provider_specific_credentials_supported": True,
        "supported_kaggle_private_input_methods": [
            "section_token_file",
            "raw_token_file",
            "provider_token_section_map",
            "provider_raw_token_file_map",
        ],
        "hf_private_access_supported": True,
        "hf_env_names_public": False,
        "hf_env_values_public": False,
        "hf_env_name_hashes": list(hf_status.get("hf_token_env_name_hashes") or []),
        "hf_env_configured": hf_status.get("hf_token_env_configured") is True,
        "hf_env_configured_count": _int(hf_status.get("hf_token_env_configured_count")),
        "public_artifact_safe": True,
    }


@dataclass
class AlphaConfig:
    output_dir: Path
    requested_model: str = COMPATIBLE_WEIGHT_REPO
    accelerators: tuple[str, ...] = REQUIRED_ACCELERATORS
    hf_token_env: str = "HF_TOKEN,HUGGING_FACE_HUB_TOKEN"
    stage_worker_package_report: str = DEFAULT_STAGE_WORKER_PACKAGE_REPORT
    token_file: str = "~/.config/crowdtensor/kaggle-tokens.md"
    token_section: str = "cpuowner"
    raw_token_file: str = ""
    raw_token_username: str = ""
    provider_token_file_map: str = ""
    provider_token_section_map: str = ""
    provider_raw_token_file_map: str = ""
    provider_raw_token_username_map: str = ""
    coordinator_bind_host: str = "0.0.0.0"
    coordinator_public_host: str = live_probe.DEFAULT_PUBLIC_HOST
    coordinator_public_url: str = ""
    gpu_accelerator: str = "NvidiaTeslaT4"
    tpu_accelerator: str = "tpuV5e8"
    wait_seconds: float = 7200.0
    poll_interval_seconds: float = 60.0
    command_timeout_seconds: float = 180.0
    kernel_timeout_seconds: int = 9000
    coordinator_task_timeout_seconds: float = 7200.0
    coordinator_worker_poll_interval_seconds: float = 5.0
    stage_push_parallelism: int = 0
    full_prefix_prefill_length: int = 0
    full_prefix_dsa_mask_topk: int = 0
    full_prefix_executed_expert_count: int = 0
    full_prefix_top_k: int = 0
    full_prefix_row_block_size: int = 0
    full_prefix_max_tensor_bytes: int = 0
    full_prefix_max_block_bytes: int = 0
    cpu_group_stage_attempt_seconds: float = 0.0
    cpu_group_stage_poll_seconds: float = 0.0
    default_max_new_tokens: int = 8
    max_new_tokens_limit: int = 16


class AlphaState:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.path = output_dir / "glm52_kaggle_alpha_status.json"
        self.lock = threading.RLock()
        self.payload: dict[str, Any] = {
            "schema": STATUS_SCHEMA,
            "generated_at": utc_now(),
            "ok": True,
            "model_id": MODEL_ID,
            "compatible_weight_repo": COMPATIBLE_WEIGHT_REPO,
            "phase": "initialized",
            "phases": ["initialized"],
            "latest_request": {},
            "latest_live_report_path": "",
            "cleanup_status": {},
            "public_artifact_safe": True,
            "safety": safety_flags(),
        }
        self.payload.update(initial_status_from_artifacts(output_dir))
        self.persist()

    def update(self, phase: str, **values: Any) -> dict[str, Any]:
        with self.lock:
            phases = list(self.payload.get("phases") or [])
            phases.append(str(phase))
            self.payload.update(values)
            self.payload["phase"] = str(phase)
            self.payload["phases"] = phases[-100:]
            self.payload["generated_at"] = utc_now()
            self.persist()
            return dict(self.payload)

    def persist(self) -> None:
        write_json(self.path, self.payload)

    def public_status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.payload)


def build_live_args(config: AlphaConfig, *, request_output_dir: Path, max_new_tokens: int) -> argparse.Namespace:
    argv = [
        "--output-dir",
        str(request_output_dir),
        "--stage-worker-package-report",
        str(config.stage_worker_package_report),
        "--coordinator-bind-host",
        str(config.coordinator_bind_host),
        "--coordinator-public-host",
        str(config.coordinator_public_host),
        "--coordinator-public-url",
        str(config.coordinator_public_url),
        "--coordinator-task-timeout-seconds",
        str(config.coordinator_task_timeout_seconds),
        "--coordinator-worker-poll-interval-seconds",
        str(config.coordinator_worker_poll_interval_seconds),
        "--wait-seconds",
        str(config.wait_seconds),
        "--poll-interval-seconds",
        str(config.poll_interval_seconds),
        "--command-timeout-seconds",
        str(config.command_timeout_seconds),
        "--kernel-timeout-seconds",
        str(config.kernel_timeout_seconds),
        "--token-file",
        str(config.token_file),
        "--token-section",
        str(config.token_section),
        "--raw-token-file",
        str(config.raw_token_file),
        "--raw-token-username",
        str(config.raw_token_username),
        "--hf-token-env",
        str(config.hf_token_env),
        "--provider-token-file-map",
        str(config.provider_token_file_map),
        "--provider-token-section-map",
        str(config.provider_token_section_map),
        "--provider-raw-token-file-map",
        str(config.provider_raw_token_file_map),
        "--provider-raw-token-username-map",
        str(config.provider_raw_token_username_map),
        "--gpu-accelerator",
        str(config.gpu_accelerator),
        "--tpu-accelerator",
        str(config.tpu_accelerator),
        "--max-new-tokens",
        str(max_new_tokens),
        "--full-prefix-prefill-length",
        str(config.full_prefix_prefill_length),
        "--full-prefix-dsa-mask-topk",
        str(config.full_prefix_dsa_mask_topk),
        "--full-prefix-executed-expert-count",
        str(config.full_prefix_executed_expert_count),
        "--full-prefix-top-k",
        str(config.full_prefix_top_k),
        "--full-prefix-row-block-size",
        str(config.full_prefix_row_block_size),
        "--full-prefix-max-tensor-bytes",
        str(config.full_prefix_max_tensor_bytes),
        "--full-prefix-max-block-bytes",
        str(config.full_prefix_max_block_bytes),
        "--cpu-group-stage-attempt-seconds",
        str(config.cpu_group_stage_attempt_seconds),
        "--cpu-group-stage-poll-seconds",
        str(config.cpu_group_stage_poll_seconds),
    ]
    if int(config.stage_push_parallelism or 0) > 0:
        argv.extend(["--stage-push-parallelism", str(config.stage_push_parallelism)])
    return live_probe.parse_args(argv)


def request_timeout_seconds(config: AlphaConfig, request: dict[str, Any]) -> float:
    raw_timeout = request.get("timeout_seconds", request.get("timeout"))
    requested = _float(raw_timeout, 0.0)
    if requested <= 0:
        return float(config.wait_seconds)
    return max(1.0, min(float(requested), float(config.wait_seconds)))


def validate_generate_request(config: AlphaConfig, payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return "prompt_required"
    if "max_new_tokens" in payload:
        try:
            max_new_tokens = int(payload.get("max_new_tokens"))
        except (TypeError, ValueError):
            return "max_new_tokens_invalid"
        if max_new_tokens < 1:
            return "max_new_tokens_below_minimum"
        if max_new_tokens > int(config.max_new_tokens_limit):
            return "max_new_tokens_above_limit"
    return ""


def generate_validation_error_response(error: str) -> dict[str, Any]:
    return {
        "schema": GENERATE_SCHEMA,
        "ok": False,
        "error": "invalid_generate_request",
        "validation_error": str(error),
        "blockers": [f"glm52_alpha_generate_request_{error}"],
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "generated_token_count": 0,
        "generated_token_hashes": [],
        "same_request_decode_verified": False,
        "accepted_providers": [],
        "public_artifact_safe": True,
        "safety": safety_flags(),
    }


def current_gpu_quota_blocker(config: AlphaConfig) -> dict[str, Any]:
    alpha_report = load_json(config.output_dir / "glm52_kaggle_alpha.json")
    quota_report = load_json(config.output_dir / "gpu-quota-preflight" / "kaggle_gpu_token_weekly_quota_probe.json")
    quota_summary = _dict(alpha_report.get("gpu_quota_summary"))
    quota_blocked = quota_summary.get("all_auth_ok_accounts_gpu_quota_exhausted") is True
    next_refresh = str(quota_summary.get("next_quota_refresh_time") or "")
    if not quota_blocked and quota_report:
        summary = _dict(quota_report.get("summary"))
        auth_ok = _int(summary.get("auth_ok_count"))
        accepted = _int(summary.get("gpu_submission_accepted_count"))
        exhausted = _int(summary.get("weekly_gpu_quota_exhausted_count"))
        quota_blocked = bool(auth_ok > 0 and accepted == 0 and exhausted >= auth_ok)
        refresh_times: list[str] = []
        for account in [item for item in _list(quota_report.get("accounts")) if isinstance(item, dict)]:
            quota = _dict(account.get("accelerator_quota"))
            refresh = str(quota.get("quota_refresh_time") or "")
            if refresh:
                refresh_times.append(refresh)
        next_refresh = sorted(set(refresh_times))[0] if refresh_times else ""
    refresh_time = _parse_utc_datetime(next_refresh)
    if not quota_blocked:
        return {}
    if refresh_time is not None and refresh_time <= datetime.now(timezone.utc):
        return {}
    cleanup = cleanup_status_from_alpha_report(alpha_report) or cleanup_status_from_quota_preflight(quota_report)
    blocker_report = _dict(alpha_report.get("blocker_report"))
    resume_private_inputs = _dict(blocker_report.get("resume_private_inputs")) or _dict(alpha_report.get("resume_private_inputs"))
    return {
        "blocked": True,
        "blocker": "kaggle_gpu_quota_unavailable",
        "next_quota_refresh_time": next_refresh,
        "next_resume_command": str(blocker_report.get("next_resume_command") or ""),
        "next_resume_command_redacts_credentials": blocker_report.get("next_resume_command_redacts_credentials") is not False,
        "resume_private_inputs": resume_private_inputs,
        "cleanup_status": cleanup,
        "phase_status": _dict(alpha_report.get("phase_status")),
        "source": "alpha_report" if alpha_report else ("gpu_quota_preflight" if quota_report else ""),
        "public_artifact_safe": True,
    }


def initial_status_from_artifacts(output_dir: Path) -> dict[str, Any]:
    alpha_report = load_json(output_dir / "glm52_kaggle_alpha.json")
    if not alpha_report:
        return {}
    config = AlphaConfig(output_dir=output_dir)
    quota_blocker = current_gpu_quota_blocker(config)
    blocker_report = _dict(alpha_report.get("blocker_report"))
    resume_private_inputs = (
        _dict(quota_blocker.get("resume_private_inputs"))
        or _dict(blocker_report.get("resume_private_inputs"))
        or _dict(alpha_report.get("resume_private_inputs"))
    )
    cleanup = _dict(quota_blocker.get("cleanup_status")) or cleanup_status_from_alpha_report(alpha_report)
    ready = alpha_report.get("glm52_kaggle_alpha_ready") is True
    phase = "alpha_ready" if ready else "alpha_blocked"
    if quota_blocker:
        phase = "blocked_gpu_quota"
    phases = ["initialized", phase]
    return {
        "phase": phase,
        "phases": phases,
        "alpha_report_present": True,
        "glm52_kaggle_alpha_ready": ready,
        "generated_token_count": _int(alpha_report.get("generated_token_count")),
        "same_request_multitoken_verified": alpha_report.get("same_request_multitoken_verified") is True,
        "accepted_providers": _list(alpha_report.get("accepted_providers")),
        "blockers": _list(alpha_report.get("blockers")),
        "external_resource_blockers": _dict(blocker_report.get("external_resource_blockers")),
        "next_resume_command": str(blocker_report.get("next_resume_command") or ""),
        "next_resume_command_redacts_credentials": blocker_report.get("next_resume_command_redacts_credentials") is not False,
        "resume_private_inputs": resume_private_inputs,
        "phase_status": _dict(alpha_report.get("phase_status")),
        "cleanup_status": cleanup,
        "public_artifact_safe": True,
    }


def generate_with_live_probe(config: AlphaConfig, request: dict[str, Any], *, runner: Runner = subprocess.run) -> dict[str, Any]:
    prompt = str(request.get("prompt") or "")
    max_new_tokens = max(1, min(_int(request.get("max_new_tokens"), config.default_max_new_tokens), config.max_new_tokens_limit))
    timeout_seconds = request_timeout_seconds(config, request)
    request_config = replace(
        config,
        wait_seconds=timeout_seconds,
        coordinator_task_timeout_seconds=min(float(config.coordinator_task_timeout_seconds), timeout_seconds),
    )
    request_hash = sha_text(
        json.dumps(
            {
                "prompt_hash": sha_text(prompt),
                "max_new_tokens": max_new_tokens,
                "timeout_seconds": timeout_seconds,
                "model_id": MODEL_ID,
                "created_at": utc_now(),
            },
            sort_keys=True,
        )
    )
    request_dir = config.output_dir / "requests" / request_hash.replace(":", "-")
    quota_blocker = current_gpu_quota_blocker(config)
    if quota_blocker:
        response = {
            "schema": GENERATE_SCHEMA,
            "ok": False,
            "generated_at": utc_now(),
            "model_id": MODEL_ID,
            "compatible_weight_repo": COMPATIBLE_WEIGHT_REPO,
            "request_id_hash": request_hash,
            "prompt_hash": sha_text(prompt),
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "target_generated_token_count": max_new_tokens,
            "timeout_seconds": timeout_seconds,
            "generated_token_count": 0,
            "generated_token_hashes": [],
            "same_request_decode_verified": False,
            "accepted_providers": [],
            "coordinator_request_id_hash": "",
            "live_report_path": "",
            "cleanup_status": _dict(quota_blocker.get("cleanup_status")),
            "duration_seconds": 0.0,
            "blockers": ["kaggle_gpu_quota_unavailable", "glm52_alpha_request_blocked_by_current_gpu_quota_preflight"],
            "external_resource_blockers": {
                "kaggle_gpu_quota_unavailable": True,
                "next_quota_refresh_time": str(quota_blocker.get("next_quota_refresh_time") or ""),
            },
            "next_resume_command": str(quota_blocker.get("next_resume_command") or ""),
            "next_resume_command_redacts_credentials": quota_blocker.get("next_resume_command_redacts_credentials") is True,
            "resume_private_inputs": _dict(quota_blocker.get("resume_private_inputs")),
            "phase_status": _dict(quota_blocker.get("phase_status")),
            "public_artifact_safe": True,
            "safety": safety_flags(),
        }
        leaks = public_redaction_errors(response)
        if leaks:
            response["public_artifact_safe"] = False
            response["blockers"] = sorted(set(_list(response.get("blockers")) + ["public_redaction_scan_failed"]))
        write_json(request_dir / "glm52_kaggle_alpha_generate_response.json", response)
        return response
    live_args = build_live_args(request_config, request_output_dir=request_dir, max_new_tokens=max_new_tokens)
    started = time.monotonic()
    report = live_probe.run_live(live_args, runner=runner)
    duration = round(time.monotonic() - started, 3)
    live_report_path = request_dir / "glm52_kaggle_same_request_live_probe.json"
    write_json(live_report_path, report)
    response = {
        "schema": GENERATE_SCHEMA,
        "ok": report.get("same_request_decode_verified") is True,
        "generated_at": utc_now(),
        "model_id": MODEL_ID,
        "compatible_weight_repo": COMPATIBLE_WEIGHT_REPO,
        "request_id_hash": request_hash,
        "prompt_hash": sha_text(prompt),
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "target_generated_token_count": max_new_tokens,
        "timeout_seconds": timeout_seconds,
        "generated_token_count": _int(report.get("generated_token_count")),
        "generated_token_hashes": _list(report.get("generated_token_hashes")),
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "accepted_providers": _list(report.get("accepted_providers")),
        "coordinator_request_id_hash": str(report.get("coordinator_request_id_hash") or ""),
        "live_report_path": str(live_report_path),
        "cleanup_status": _dict(report.get("cleanup_status")),
        "duration_seconds": duration,
        "blockers": _list(report.get("blockers")),
        "public_artifact_safe": True,
        "safety": safety_flags(),
    }
    leaks = public_redaction_errors(response)
    if leaks:
        response["ok"] = False
        response["public_artifact_safe"] = False
        response["blockers"] = sorted(set(_list(response.get("blockers")) + ["public_redaction_scan_failed"]))
    write_json(request_dir / "glm52_kaggle_alpha_generate_response.json", response)
    return response


def build_service_report(config: AlphaConfig, *, host: str, port: int, run: bool) -> dict[str, Any]:
    requested_model = normalize_model_request(config.requested_model)
    model_supported = model_request_supported(requested_model)
    accelerator_status = accelerator_request_status(config.accelerators)
    hf_status = hf_token_env_status(config.hf_token_env)
    resume_private_inputs = resume_private_inputs_status(hf_status)
    service_config_ready = bool(model_supported and accelerator_status["complete"])
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": service_config_ready,
        "mode": "serve",
        "model_id": MODEL_ID,
        "compatible_weight_repo": COMPATIBLE_WEIGHT_REPO,
        "requested_model": requested_model,
        "model_request_supported": model_supported,
        "model_request": {
            "requested": requested_model,
            "model_id": MODEL_ID,
            "compatible_weight_repo": COMPATIBLE_WEIGHT_REPO,
            "supported": model_supported,
            "non_glm_fallback": False,
        },
        "service_api_ready": service_config_ready,
        "cli_generate_command_available": True,
        "cli_generate_artifact_recovery_supported": True,
        "cli_serve_default_matches_deploy": True,
        "cli_status_default_matches_deploy": True,
        "cli_cleanup_default_matches_deploy": True,
        "status_loads_existing_alpha_artifacts": True,
        "status_exposes_resume_private_inputs": True,
        "routes": {
            "health": "GET /health",
            "status": "GET /status",
            "generate": "POST /generate",
            "cleanup": "POST /cleanup",
        },
        "cleanup_route_ready": True,
        "generate_validates_request_schema": True,
        "generate_routes_to_same_request_live_probe": True,
        "generate_uses_current_gpu_quota_blocker": True,
        "kaggle_runtime_blocker_classification_ready": True,
        "kaggle_runtime_blocker_classes": list(KAGGLE_RUNTIME_BLOCKER_CLASSES),
        "generate_request_fields": ["prompt", "max_new_tokens", "timeout", "timeout_seconds"],
        "default_max_new_tokens": int(config.default_max_new_tokens),
        "max_new_tokens_limit": int(config.max_new_tokens_limit),
        "accelerators": list(accelerator_status["requested"]),
        "required_accelerators": list(REQUIRED_ACCELERATORS),
        "accelerator_request_complete": bool(accelerator_status["complete"]),
        "accelerator_request": accelerator_status,
        **hf_status,
        "resume_private_inputs": resume_private_inputs,
        "gpu_accelerator": str(config.gpu_accelerator),
        "tpu_accelerator": str(config.tpu_accelerator),
        "stage_push_parallelism": int(config.stage_push_parallelism),
        "wait_seconds": float(config.wait_seconds),
        "poll_interval_seconds": float(config.poll_interval_seconds),
        "command_timeout_seconds": float(config.command_timeout_seconds),
        "kernel_timeout_seconds": int(config.kernel_timeout_seconds),
        "coordinator_task_timeout_seconds": float(config.coordinator_task_timeout_seconds),
        "coordinator_worker_poll_interval_seconds": float(config.coordinator_worker_poll_interval_seconds),
        "runtime_tuning": {
            "full_prefix_prefill_length": int(config.full_prefix_prefill_length),
            "full_prefix_dsa_mask_topk": int(config.full_prefix_dsa_mask_topk),
            "full_prefix_executed_expert_count": int(config.full_prefix_executed_expert_count),
            "full_prefix_top_k": int(config.full_prefix_top_k),
            "full_prefix_row_block_size": int(config.full_prefix_row_block_size),
            "full_prefix_max_tensor_bytes": int(config.full_prefix_max_tensor_bytes),
            "full_prefix_max_block_bytes": int(config.full_prefix_max_block_bytes),
            "cpu_group_stage_attempt_seconds": float(config.cpu_group_stage_attempt_seconds),
            "cpu_group_stage_poll_seconds": float(config.cpu_group_stage_poll_seconds),
        },
        "stage_worker_package_report": str(config.stage_worker_package_report),
        "output_dir": str(config.output_dir),
        "server": {"host": host, "port": int(port), "run": bool(run)},
        "credentials_public": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "public_artifact_safe": True,
        "safety": safety_flags(),
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
    return report


class AlphaHTTPServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        config: AlphaConfig,
        generate_fn: GenerateFn | None = None,
    ) -> None:
        state = AlphaState(config.output_dir)
        generate = generate_fn or (lambda cfg, payload: generate_with_live_probe(cfg, payload))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> tuple[dict[str, Any], str]:
                try:
                    size = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return {}, "invalid_content_length"
                if size <= 0:
                    return {}, "empty_json_body"
                try:
                    loaded = json.loads(self.rfile.read(size).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return {}, "malformed_json"
                if not isinstance(loaded, dict):
                    return {}, "json_body_not_object"
                return loaded, ""

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    self._send(200, {"schema": SCHEMA, "ok": True, "model_id": MODEL_ID, "public_artifact_safe": True})
                    return
                if path == "/status":
                    self._send(200, state.public_status())
                    return
                self._send(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/cleanup":
                    report = build_cleanup_report(config.output_dir)
                    state.update(
                        "cleanup_completed" if report.get("ok") else "cleanup_blocked",
                        cleanup_status=_dict(report.get("cleanup_status")),
                        blockers=_list(report.get("blockers")),
                    )
                    self._send(200 if report.get("ok") else 503, report)
                    return
                if path != "/generate":
                    self._send(404, {"ok": False, "error": "not_found"})
                    return
                payload, parse_error = self._read_json()
                validation_error = parse_error or validate_generate_request(config, payload)
                if validation_error:
                    response = generate_validation_error_response(validation_error)
                    state.update(
                        "generate_request_invalid",
                        latest_request={
                            "validation_error": validation_error,
                            "prompt_hash": sha_text(payload.get("prompt") or "") if isinstance(payload, dict) else "",
                        },
                        blockers=_list(response.get("blockers")),
                    )
                    self._send(400, response)
                    return
                state.update(
                    "generate_received",
                    latest_request={
                        "prompt_hash": sha_text(payload.get("prompt") or ""),
                        "max_new_tokens": _int(payload.get("max_new_tokens"), config.default_max_new_tokens),
                        "timeout_seconds": request_timeout_seconds(config, payload),
                    },
                )
                try:
                    state.update("same_request_live_started")
                    response = generate(config, payload)
                    state.update(
                        "decode_completed" if response.get("ok") else "decode_blocked",
                        latest_request={
                            "request_id_hash": response.get("request_id_hash"),
                            "prompt_hash": response.get("prompt_hash"),
                            "target_generated_token_count": response.get("target_generated_token_count"),
                            "timeout_seconds": response.get("timeout_seconds"),
                            "generated_token_count": response.get("generated_token_count"),
                            "same_request_decode_verified": response.get("same_request_decode_verified") is True,
                        },
                        latest_live_report_path=str(response.get("live_report_path") or ""),
                        cleanup_status=_dict(response.get("cleanup_status")),
                        blockers=_list(response.get("blockers")),
                        external_resource_blockers=_dict(response.get("external_resource_blockers")),
                        next_resume_command=str(response.get("next_resume_command") or ""),
                        phase_status=_dict(response.get("phase_status")),
                    )
                    self._send(200 if response.get("ok") else 503, response)
                except Exception as exc:  # pragma: no cover - live failure path.
                    state.update("generate_failed", latest_error_type=type(exc).__name__, latest_error_hash=sha_text(str(exc)))
                    self._send(
                        500,
                        {
                            "schema": GENERATE_SCHEMA,
                            "ok": False,
                            "error": type(exc).__name__,
                            "error_hash": sha_text(str(exc)),
                            "public_artifact_safe": True,
                            "safety": safety_flags(),
                        },
                    )

        self.state = state
        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def cleanup_status_verified(cleanup: dict[str, Any]) -> bool:
    return bool(
        cleanup.get("temporary_kaggle_kernels_deleted") is True
        and cleanup.get("temporary_private_packages_removed") is True
        and cleanup.get("live_resources_left_running") is False
    )


def quota_preflight_cleanup_verified(report: dict[str, Any]) -> bool:
    if not report:
        return False
    if report.get("private_kernel_payloads_removed") is not True:
        return False
    for account in [item for item in _list(report.get("accounts")) if isinstance(item, dict)]:
        cleanup = _dict(account.get("cleanup"))
        if account.get("push_accepted") is True and cleanup.get("deleted") is not True:
            return False
    return True


def cleanup_status_from_quota_preflight(report: dict[str, Any]) -> dict[str, Any]:
    if not quota_preflight_cleanup_verified(report):
        return {}
    summary = _dict(report.get("summary"))
    return {
        "temporary_kaggle_kernels_deleted": True,
        "temporary_private_packages_removed": True,
        "live_resources_left_running": False,
        "cleanup_mode": "gpu_quota_preflight_skipped_live",
        "gpu_quota_preflight_cleanup_verified": True,
        "gpu_submission_accepted_count": _int(summary.get("gpu_submission_accepted_count")),
        "weekly_gpu_quota_exhausted_count": _int(summary.get("weekly_gpu_quota_exhausted_count")),
        "auth_ok_count": _int(summary.get("auth_ok_count")),
        "public_artifact_safe": True,
    }


def cleanup_status_from_alpha_report(report: dict[str, Any]) -> dict[str, Any]:
    if not report or report.get("cleanup_verified") is not True:
        return {}
    live_summary = _dict(report.get("live_summary"))
    live_cleanup = _dict(live_summary.get("cleanup_status"))
    if cleanup_status_verified(live_cleanup):
        return dict(live_cleanup)
    quota_summary = _dict(report.get("gpu_quota_summary"))
    if quota_summary.get("cleanup_verified") is True and quota_summary.get("all_auth_ok_accounts_gpu_quota_exhausted") is True:
        return {
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "cleanup_mode": "gpu_quota_preflight_skipped_live",
            "gpu_quota_preflight_cleanup_verified": True,
            "gpu_submission_accepted_count": _int(quota_summary.get("gpu_submission_accepted_count")),
            "weekly_gpu_quota_exhausted_count": _int(quota_summary.get("weekly_gpu_quota_exhausted_count")),
            "auth_ok_count": _int(quota_summary.get("auth_ok_count")),
            "public_artifact_safe": True,
        }
    return {}


def build_cleanup_report(output_dir: Path) -> dict[str, Any]:
    status = load_json(output_dir / "glm52_kaggle_alpha_status.json")
    alpha_report = load_json(output_dir / "glm52_kaggle_alpha.json")
    cli_summary = load_json(output_dir / "glm52_kaggle_alpha_cli_summary.json")
    live_report = load_json(output_dir / "live" / "glm52_kaggle_same_request_live_probe.json")
    quota_report = load_json(output_dir / "gpu-quota-preflight" / "kaggle_gpu_token_weekly_quota_probe.json")
    cleanup = _dict(status.get("cleanup_status"))
    source = "service_status" if cleanup_status_verified(cleanup) else ""
    if not source:
        live_cleanup = _dict(live_report.get("cleanup_status"))
        if cleanup_status_verified(live_cleanup):
            cleanup = live_cleanup
            source = "live_report"
    if not source:
        alpha_cleanup = cleanup_status_from_alpha_report(alpha_report)
        if cleanup_status_verified(alpha_cleanup):
            cleanup = alpha_cleanup
            source = "alpha_report"
    if not source and cli_summary.get("live_skipped_by_gpu_quota_preflight") is True:
        quota_cleanup = cleanup_status_from_quota_preflight(quota_report)
        if cleanup_status_verified(quota_cleanup):
            cleanup = quota_cleanup
            source = "gpu_quota_preflight"
    ok = cleanup_status_verified(cleanup)
    report = {
        "schema": CLEANUP_SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "cleanup_status": cleanup,
        "cleanup_evidence_source": source or "missing",
        "status_report_present": bool(status),
        "alpha_report_present": bool(alpha_report),
        "cli_summary_present": bool(cli_summary),
        "live_report_present": bool(live_report),
        "gpu_quota_preflight_report_present": bool(quota_report),
        "live_skipped_by_gpu_quota_preflight": cli_summary.get("live_skipped_by_gpu_quota_preflight") is True,
        "alpha_cleanup_verified": alpha_report.get("cleanup_verified") is True,
        "quota_preflight_cleanup_verified": quota_preflight_cleanup_verified(quota_report),
        "temporary_kaggle_kernels_deleted": cleanup.get("temporary_kaggle_kernels_deleted") is True,
        "temporary_private_packages_removed": cleanup.get("temporary_private_packages_removed") is True,
        "live_resources_left_running": cleanup.get("live_resources_left_running") if isinstance(cleanup.get("live_resources_left_running"), bool) else None,
        "blockers": [] if ok else ["glm52_alpha_cleanup_proof_missing_or_incomplete"],
        "public_artifact_safe": True,
        "safety": safety_flags(),
    }
    write_json(output_dir / "glm52_kaggle_alpha_cleanup.json", report)
    return report
