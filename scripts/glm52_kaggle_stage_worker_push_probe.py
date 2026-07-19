#!/usr/bin/env python3
"""Preflight or run bounded Kaggle pushes for GLM 5.2 stage workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import glm52_kaggle_stage_runtime_check as stage_runtime_check


SCHEMA = "glm52_kaggle_stage_worker_push_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-stage-worker-push-probe"
REQUIRED_PROVIDERS = ["kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"]
TERMINAL_STATUSES = {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED"}
NONTERMINAL_STATUSES = {"RUNNING", "QUEUED", "PENDING", "UNKNOWN"}
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_API_TOKEN",
    "CT_GLM52_COORDINATOR_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
)
PRIVATE_RUNTIME_ENV_FILENAME = "ct_glm52_private_runtime_env.json"
PRIVATE_RUNTIME_ENV_INLINE_SENTINEL = "CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE = {}\n"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def shell_command(command: list[Any]) -> str:
    return shlex.join([str(part) for part in command])


def accelerator_for_provider(provider: str, args: argparse.Namespace) -> str:
    if provider == "kaggle_cuda":
        return str(getattr(args, "gpu_accelerator", "") or "")
    if provider == "kaggle_jax_tpu":
        return str(getattr(args, "tpu_accelerator", "") or "")
    return ""


def push_command_for_package(package_dir: str, provider: str, args: argparse.Namespace) -> list[str]:
    command = ["kaggle", "kernels", "push", "-p", package_dir]
    kernel_timeout_seconds = _int(getattr(args, "kernel_timeout_seconds", 0), 0)
    if kernel_timeout_seconds > 0:
        command.extend(["-t", str(kernel_timeout_seconds)])
    accelerator = accelerator_for_provider(provider, args)
    if accelerator:
        command.extend(["--accelerator", accelerator])
    return command


def safe_tail(text: str, limit: int = 1600) -> str:
    redacted = str(text or "")[-limit:]
    for fragment in SENSITIVE_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def parse_token_sections(path: Path) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    label = ""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines() + ["# END"]:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if label and values:
                sections.append({"label": label, "env": dict(values)})
            label = line.lstrip("#").strip()
            values = {}
            continue
        if line.startswith("export ") and "=" in line:
            key, raw_value = line[len("export ") :].split("=", 1)
            key = key.strip()
            value = raw_value.strip().strip("'\"")
            if key:
                values[key] = value
    return [item for item in sections if item.get("label") != "END"]


def parse_raw_token_file(path: Path, *, username_hint: str = "") -> dict[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("kaggle_raw_token_file_empty")
    values: dict[str, str] = {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        username = loaded.get("username") or loaded.get("KAGGLE_USERNAME")
        key = loaded.get("key") or loaded.get("KAGGLE_KEY") or loaded.get("api_key")
        api_token = loaded.get("api_token") or loaded.get("KAGGLE_API_TOKEN") or loaded.get("token")
        if username and key:
            env = {"KAGGLE_USERNAME": str(username), "KAGGLE_KEY": str(key)}
            if api_token:
                env["KAGGLE_API_TOKEN"] = str(api_token)
            return env
        if api_token:
            env = {"KAGGLE_API_TOKEN": str(api_token)}
            if username or username_hint:
                env["KAGGLE_USERNAME"] = str(username or username_hint)
            return env
    sections = parse_token_sections(path)
    if sections:
        normalized_hint = re.sub(r"[^a-z0-9]+", "", username_hint.lower())

        def matches_hint(section: dict[str, Any]) -> bool:
            env = section.get("env") or {}
            candidates = [section.get("label"), env.get("KAGGLE_USERNAME")]
            return any(
                normalized_hint
                and re.sub(r"[^a-z0-9]+", "", str(candidate or "").lower())
                == normalized_hint
                for candidate in candidates
            )

        if len(sections) == 1:
            selected = sections[0]
        elif username_hint:
            matched = [section for section in sections if matches_hint(section)]
            if len(matched) != 1:
                raise RuntimeError("kaggle_raw_token_section_not_found")
            selected = matched[0]
        else:
            raise RuntimeError("kaggle_raw_token_section_hint_required")
        selected_values = {
            str(key): str(value)
            for key, value in dict(selected.get("env") or {}).items()
            if key in {"KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"}
            and value
        }
        if selected_values.get("KAGGLE_API_TOKEN"):
            if username_hint and not selected_values.get("KAGGLE_USERNAME"):
                selected_values["KAGGLE_USERNAME"] = str(username_hint)
            return selected_values
        if {"KAGGLE_USERNAME", "KAGGLE_KEY"}.issubset(selected_values):
            return selected_values
        raise RuntimeError("kaggle_raw_token_section_invalid")
    for part in shlex.split(text):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip("'\"")
        if key in {"KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"} and value:
            values[key] = value
    if {"KAGGLE_USERNAME", "KAGGLE_KEY"}.issubset(values):
        env = {"KAGGLE_USERNAME": values["KAGGLE_USERNAME"], "KAGGLE_KEY": values["KAGGLE_KEY"]}
        if values.get("KAGGLE_API_TOKEN"):
            env["KAGGLE_API_TOKEN"] = values["KAGGLE_API_TOKEN"]
        return env
    if values.get("KAGGLE_API_TOKEN"):
        env = {"KAGGLE_API_TOKEN": values["KAGGLE_API_TOKEN"]}
        if values.get("KAGGLE_USERNAME") or username_hint:
            env["KAGGLE_USERNAME"] = values.get("KAGGLE_USERNAME") or str(username_hint)
        return env
    compact = re.split(r"[\s,:]+", text)
    compact = [item for item in compact if item]
    if len(compact) >= 2:
        return {"KAGGLE_USERNAME": compact[0], "KAGGLE_KEY": compact[1]}
    if len(compact) == 1 and username_hint:
        return {"KAGGLE_USERNAME": str(username_hint), "KAGGLE_KEY": compact[0], "KAGGLE_API_TOKEN": compact[0]}
    raise RuntimeError("kaggle_raw_token_file_format_unrecognized")


def isolated_kaggle_env(token_env: dict[str, str], config_dir: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("KAGGLE_") and key != "MY_KAGGLE_TOKEN"
    }
    env.update({str(key): str(value) for key, value in token_env.items()})
    env["KAGGLE_CONFIG_DIR"] = config_dir
    return env


def kaggle_env_for_token_section(args: argparse.Namespace) -> tuple[dict[str, str] | None, tempfile.TemporaryDirectory[str] | None]:
    raw_token_file = str(getattr(args, "raw_token_file", "") or "").strip()
    if raw_token_file:
        config_dir = tempfile.TemporaryDirectory(prefix="glm52-kaggle-stage-worker-config-")
        env = isolated_kaggle_env(
            parse_raw_token_file(
                Path(raw_token_file).expanduser(),
                username_hint=str(getattr(args, "raw_token_username", "") or "").strip(),
            ),
            config_dir.name,
        )
        return env, config_dir
    section_name = str(getattr(args, "token_section", "") or "").strip()
    if not section_name:
        return None, None
    token_file = Path(str(getattr(args, "token_file", "") or "")).expanduser()
    sections = parse_token_sections(token_file)
    normalized = {str(item.get("label") or "").strip(): item for item in sections}
    section = normalized.get(section_name)
    if not section:
        raise RuntimeError("kaggle_token_section_not_found")
    config_dir = tempfile.TemporaryDirectory(prefix="glm52-kaggle-stage-worker-config-")
    env = isolated_kaggle_env(_dict(section.get("env")), config_dir.name)
    return env, config_dir


def coordinator_private_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    coordinator_url = str(getattr(args, "coordinator_url", "") or "").strip()
    token_file = str(getattr(args, "coordinator_token_file", "") or "").strip()
    if not coordinator_url or not token_file:
        payload: dict[str, str] = {}
    else:
        token_path = Path(token_file).expanduser()
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        payload = {}
        if token:
            payload = {
                "CT_GLM52_COORDINATOR_URL": coordinator_url,
                "CT_GLM52_COORDINATOR_TOKEN": token,
            }
    hf_token = hf_token_from_env(getattr(args, "hf_token_env", ""))
    if hf_token:
        payload.setdefault("HF_TOKEN", hf_token)
        payload.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    if not payload:
        return {}
    task_timeout = float(getattr(args, "coordinator_task_timeout_seconds", 0.0) or 0.0)
    if task_timeout > 0:
        payload["CT_GLM52_COORDINATOR_TASK_TIMEOUT_SECONDS"] = str(task_timeout)
    poll_interval = float(getattr(args, "coordinator_poll_interval_seconds", 0.0) or 0.0)
    if poll_interval > 0:
        payload["CT_GLM52_COORDINATOR_POLL_INTERVAL_SECONDS"] = str(poll_interval)
    stage_task_limit = int(getattr(args, "coordinator_stage_task_limit", 0) or 0)
    if stage_task_limit > 0:
        payload["CT_GLM52_COORDINATOR_STAGE_TASK_LIMIT"] = str(stage_task_limit)
    runtime_tuning = {
        "CT_GLM52_FULL_PREFIX_PREFILL_LENGTH": int(getattr(args, "full_prefix_prefill_length", 0) or 0),
        "CT_GLM52_FULL_PREFIX_DSA_MASK_TOPK": int(getattr(args, "full_prefix_dsa_mask_topk", 0) or 0),
        "CT_GLM52_FULL_PREFIX_EXECUTED_EXPERT_COUNT": int(getattr(args, "full_prefix_executed_expert_count", 0) or 0),
        "CT_GLM52_FULL_PREFIX_TOP_K": int(getattr(args, "full_prefix_top_k", 0) or 0),
        "CT_GLM52_FULL_PREFIX_ROW_BLOCK_SIZE": int(getattr(args, "full_prefix_row_block_size", 0) or 0),
        "CT_GLM52_FULL_PREFIX_MAX_BLOCK_BYTES": int(getattr(args, "full_prefix_max_block_bytes", 0) or 0),
    }
    for key, value in runtime_tuning.items():
        if value > 0:
            payload[key] = str(value)
    max_tensor_bytes = int(getattr(args, "full_prefix_max_tensor_bytes", 0) or 0)
    if max_tensor_bytes > 0:
        payload["CT_GLM52_FULL_PREFIX_MAX_TENSOR_BYTES"] = str(max_tensor_bytes)
    cpu_group_attempt = float(getattr(args, "cpu_group_stage_attempt_seconds", 0.0) or 0.0)
    if cpu_group_attempt > 0:
        payload["CT_GLM52_CPU_GROUP_STAGE_ATTEMPT_SECONDS"] = str(cpu_group_attempt)
    cpu_group_poll = float(getattr(args, "cpu_group_stage_poll_seconds", 0.0) or 0.0)
    if cpu_group_poll > 0:
        payload["CT_GLM52_CPU_GROUP_STAGE_POLL_SECONDS"] = str(cpu_group_poll)
    return payload


def hf_token_env_names(value: Any) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw_item in str(value or "").split(","):
        name = str(raw_item or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def hf_token_from_env(value: Any) -> str:
    for name in hf_token_env_names(value):
        token = os.environ.get(name, "")
        if token:
            return str(token)
    return ""


def hf_token_public_summary(args: argparse.Namespace) -> dict[str, Any]:
    names = hf_token_env_names(getattr(args, "hf_token_env", ""))
    configured_count = sum(1 for name in names if os.environ.get(name))
    return {
        "hf_token_env_supported": True,
        "hf_token_env_count": len(names),
        "hf_token_env_name_hashes": ["sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest() for name in names],
        "hf_token_env_configured": configured_count > 0,
        "hf_token_env_configured_count": configured_count,
        "hf_token_private_runtime_env_uploaded": bool(hf_token_from_env(getattr(args, "hf_token_env", ""))),
        "hf_token_public": False,
    }


def write_private_runtime_env_file(package_dir: str, args: argparse.Namespace) -> dict[str, Any]:
    payload = coordinator_private_runtime_env(args)
    hf_summary = hf_token_public_summary(args)
    if not payload:
        return {
            "coordinator_private_runtime_env_uploaded": False,
            "coordinator_private_runtime_env_inlined": False,
            "coordinator_private_runtime_env_key_count": 0,
            "coordinator_url_public": False,
            "coordinator_token_public": False,
            **hf_summary,
            "private_runtime_env_local_removed": None,
            "private_runtime_env_kernel_restored": None,
        }
    path = Path(package_dir) / PRIVATE_RUNTIME_ENV_FILENAME
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {
        "coordinator_private_runtime_env_uploaded": True,
        "coordinator_private_runtime_env_inlined": False,
        "coordinator_private_runtime_env_key_count": len(payload),
        "coordinator_url_public": False,
        "coordinator_token_public": False,
        **hf_summary,
        "private_runtime_env_filename": PRIVATE_RUNTIME_ENV_FILENAME,
        "private_runtime_env_local_removed": False,
        "private_runtime_env_kernel_restored": None,
    }
    kernel_path = Path(package_dir) / "kernel.py"
    if kernel_path.is_file():
        try:
            original = kernel_path.read_text(encoding="utf-8")
            if PRIVATE_RUNTIME_ENV_INLINE_SENTINEL in original:
                replacement = "CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE = " + json.dumps(payload, sort_keys=True) + "\n"
                kernel_path.write_text(original.replace(PRIVATE_RUNTIME_ENV_INLINE_SENTINEL, replacement, 1), encoding="utf-8")
                summary["coordinator_private_runtime_env_inlined"] = True
                summary["private_runtime_env_kernel_restored"] = False
                summary["_private_kernel_path"] = str(kernel_path)
                summary["_private_kernel_original_text"] = original
        except OSError:
            summary["coordinator_private_runtime_env_inlined"] = False
            summary["private_runtime_env_kernel_restored"] = False
    return summary


def remove_private_runtime_env_file(package_dir: str, summary: dict[str, Any]) -> dict[str, Any]:
    original = summary.pop("_private_kernel_original_text", None)
    kernel_path_text = summary.pop("_private_kernel_path", "")
    if original is not None and kernel_path_text:
        kernel_path = Path(str(kernel_path_text))
        try:
            kernel_path.write_text(str(original), encoding="utf-8")
            summary["private_runtime_env_kernel_restored"] = True
        except OSError:
            summary["private_runtime_env_kernel_restored"] = False
    if summary.get("coordinator_private_runtime_env_uploaded") is not True:
        return summary
    path = Path(package_dir) / PRIVATE_RUNTIME_ENV_FILENAME
    try:
        if path.is_file():
            path.unlink()
        summary["private_runtime_env_local_removed"] = not path.exists()
    except OSError:
        summary["private_runtime_env_local_removed"] = False
    return summary


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def extract_status(text: str) -> str:
    upper = str(text or "").upper()
    for status in ["COMPLETE", "RUNNING", "QUEUED", "PENDING", "ERROR", "FAILED", "CANCELLED", "CANCELED"]:
        if status in upper:
            return status
    return "UNKNOWN"


def run_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "error": "timeout",
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_line": shell_command(command),
        }
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": safe_tail(completed.stdout or ""),
        "stderr_tail": safe_tail(completed.stderr or ""),
        "command_line": shell_command(command),
    }


def step_output_text(step: dict[str, Any]) -> str:
    return f"{step.get('stdout_tail') or ''}\n{step.get('stderr_tail') or ''}"


def kaggle_step_blockers(provider: str, step: dict[str, Any], kind: str) -> list[str]:
    if not step:
        return []
    provider_key = provider or "missing"
    kind_key = kind.replace("_", "-").replace("-", "_")
    if step.get("error") == "timeout":
        return [f"kaggle_kernel_{kind_key}_timeout:{provider_key}"]
    if step.get("ok") is True:
        return []
    text = step_output_text(step).lower()
    blockers: list[str] = []
    if "429" in text or "too many requests" in text or "rate limit" in text:
        blockers.append(f"kaggle_kernel_{kind_key}_http_429:{provider_key}")
    if not text.strip():
        blockers.append(f"kaggle_kernel_{kind_key}_empty_response:{provider_key}")
    if not blockers:
        blockers.append(f"kaggle_kernel_{kind_key}_command_failed:{provider_key}")
    return blockers


def kaggle_push_error_blocker(provider: str, push_step: dict[str, Any]) -> str:
    text = step_output_text(push_step).lower()
    if "notebook not found" in text:
        return "kaggle_kernel_notebook_not_found"
    if "maximum batch gpu session count" in text:
        return "kaggle_gpu_batch_session_limit_reached"
    if "quota" in text and "gpu" in text:
        return "kaggle_gpu_quota_or_session_rejected"
    if "429" in text or "too many requests" in text or "rate limit" in text:
        return f"kaggle_kernel_push_http_429:{provider or 'missing'}"
    if "accelerator" in text and "unavailable" in text:
        return "kaggle_accelerator_unavailable"
    if push_step.get("error") == "timeout":
        return f"kaggle_kernel_push_timeout:{provider or 'missing'}"
    if push_step.get("ok") is not True:
        if not text.strip():
            return f"kaggle_kernel_push_empty_response:{provider or 'missing'}"
        return f"glm52_stage_worker_push_command_failed:{provider or 'missing'}"
    if "kernel push error" not in text:
        return ""
    return "kaggle_kernel_push_error"


def terminal_status_blocker(provider: str, status: str) -> str:
    normalized = str(status or "").strip().upper()
    provider_key = provider or "missing"
    if normalized in {"ERROR", "FAILED"}:
        return f"kaggle_kernel_terminal_error:{provider_key}"
    if normalized in {"CANCELLED", "CANCELED"}:
        return f"kaggle_kernel_terminal_cancelled:{provider_key}"
    return ""


def stage_report_path(output_dir: Path) -> Path:
    direct = output_dir / "glm52_kaggle_stage_runtime_report.json"
    if direct.is_file():
        return direct
    matches = sorted(output_dir.rglob("glm52_kaggle_stage_runtime_report.json")) if output_dir.exists() else []
    return matches[0] if matches else direct


def check_stage_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "ok": False,
            "error_count": 1,
            "errors": ["stage_report_missing"],
            "stage_runtime_verified": False,
            "provider": "",
            "stage_id": -1,
        }
    try:
        report = load_json(path)
        errors = stage_runtime_check.validate_report(report, require_verified=True)
    except (OSError, json.JSONDecodeError, SystemExit) as exc:
        return {
            "ok": False,
            "error_count": 1,
            "errors": [f"stage_report_read_failed:{type(exc).__name__}"],
            "stage_runtime_verified": False,
            "provider": "",
            "stage_id": -1,
        }
    return {
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "stage_runtime_verified": stage_runtime_check.stage_runtime_verified(report),
        "provider": stage_runtime_check.provider(report),
        "stage_id": _int(report.get("stage_id"), -1),
        "stage_ids": [
            _int(item, -1)
            for item in _list(report.get("stage_ids_verified") or report.get("stage_ids"))
            if _int(item, -1) >= 0
        ],
    }


def stage_ids_for_entry(entry: dict[str, Any]) -> list[int]:
    ids = [
        _int(item, -1)
        for item in _list(entry.get("stage_ids"))
        if _int(item, -1) >= 0
    ]
    stage_id = _int(entry.get("stage_id"), -1)
    if not ids and stage_id >= 0:
        ids = [stage_id]
    return sorted(set(ids))


def package_entries(package_report: dict[str, Any], providers: list[str], stage_ids: set[int] | None = None) -> list[dict[str, Any]]:
    entries = [item for item in _list(package_report.get("packages")) if isinstance(item, dict)]
    selected = set(providers or REQUIRED_PROVIDERS)
    filtered = [entry for entry in entries if str(entry.get("provider") or "") in selected]
    if stage_ids:
        filtered = [entry for entry in filtered if set(stage_ids_for_entry(entry)) & stage_ids]
    return filtered


def build_preflight(args: argparse.Namespace, package_report: dict[str, Any]) -> dict[str, Any]:
    entries = package_entries(package_report, args.providers, args.stage_ids)
    pushes = []
    blockers = {
        "glm52_stage_worker_push_not_started",
        "glm52_stage_worker_live_reports_missing",
        "glm52_same_request_not_started",
    }
    providers = {str(entry.get("provider") or "") for entry in entries}
    for provider in REQUIRED_PROVIDERS:
        if provider not in providers:
            blockers.add(f"glm52_stage_worker_push_provider_missing:{provider}")
    for entry in entries:
        package_dir = str(entry.get("package_dir") or "")
        provider = str(entry.get("provider") or "")
        pushes.append({
            "schema": "glm52_kaggle_stage_worker_push_entry_v1",
            "provider": provider,
            "stage_id": _int(entry.get("stage_id")),
            "stage_ids": stage_ids_for_entry(entry),
            "package_dir": package_dir,
            "push_command": shell_command(push_command_for_package(package_dir, provider, args)),
            "requested_accelerator": accelerator_for_provider(provider, args),
            "status_command": "kaggle kernels status <kernel-ref>",
            "output_command": "kaggle kernels output <kernel-ref> -p <output-dir> --force --file-pattern glm52_kaggle_stage_runtime_report.json",
            "cleanup_command": "kaggle kernels delete <kernel-ref> -y",
            "pushed": False,
            "terminal_status": "",
            "output_collected": False,
            "stage_report_path": "",
            "stage_report_present": False,
            "stage_report_check": {
                "ok": False,
                "error_count": 0,
                "errors": [],
                "stage_runtime_verified": False,
                "provider": "",
                "stage_id": -1,
            },
            "stage_runtime_verified": False,
            "cleanup_performed": False,
            "public_artifact_safe": True,
        })
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "mode": "preflight",
        "ok": True,
        "glm52_stage_worker_push_probe_ready": True,
        "live_run_performed": False,
        "stage_runtime_reports_collected": 0,
        "stage_runtime_reports_verified": 0,
        "same_request_route_verified": False,
        "stage_runtime_adapter_verified": False,
        "pushes": pushes,
        "blockers": sorted(blockers),
        "completion_boundary": {
            "preflight_is_not_runtime_success": True,
            "push_required": True,
            "terminal_kernel_output_required": True,
            "stage_runtime_check_required": True,
            "same_request_probe_required": True,
        },
        "public_artifact_safe": True,
        "safety": {
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


def _stage_report_provider(path: Path) -> str:
    report = load_json(path)
    return stage_runtime_check.provider(report)


def _stage_report_id(path: Path, default: int) -> int:
    report = load_json(path)
    return _int(report.get("stage_id"), default)


def _watch_stage_report_path(watch_report: dict[str, Any]) -> Path | None:
    stage_report = _dict(watch_report.get("stage_runtime_report"))
    raw_path = str(stage_report.get("path") or "")
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_file() else None


def _stage_report_key(path: Path) -> tuple[str, int]:
    report = load_json(path)
    return stage_runtime_check.provider(report), _int(report.get("stage_id"), -1)


def imported_stage_paths(args: argparse.Namespace, tpu_watch_report: dict[str, Any]) -> dict[tuple[str, int], Path]:
    paths: dict[tuple[str, int], Path] = {}
    for raw_path in getattr(args, "import_stage_report", []) or []:
        path = Path(str(raw_path))
        if not path.is_file():
            continue
        provider, stage_id = _stage_report_key(path)
        if provider and stage_id >= 0:
            paths[(provider, stage_id)] = path
    if tpu_watch_report.get("stage_runtime_report_verified") is True:
        tpu_path = _watch_stage_report_path(tpu_watch_report)
        if tpu_path is not None:
            provider, stage_id = _stage_report_key(tpu_path)
            if provider and stage_id >= 0:
                paths[(provider, stage_id)] = tpu_path
    return paths


def build_import(args: argparse.Namespace, package_report: dict[str, Any]) -> dict[str, Any]:
    entries = package_entries(package_report, args.providers, args.stage_ids)
    tpu_watch_report = load_json(getattr(args, "tpu_watch_report", ""))
    stage_paths = imported_stage_paths(args, tpu_watch_report)
    pushes: list[dict[str, Any]] = []
    blockers: set[str] = set(str(item) for item in _list(tpu_watch_report.get("blockers")) if item)
    collected = 0
    verified = 0
    for entry in entries:
        package_dir = str(entry.get("package_dir") or "")
        provider = str(entry.get("provider") or "")
        stage_id = _int(entry.get("stage_id"), REQUIRED_PROVIDERS.index(provider) if provider in REQUIRED_PROVIDERS else -1)
        report_path = stage_paths.get((provider, stage_id))
        report_present = bool(report_path and report_path.is_file())
        report_check = check_stage_report(report_path) if report_path is not None else {
            "ok": False,
            "error_count": 1,
            "errors": ["stage_report_missing"],
            "stage_runtime_verified": False,
            "provider": "",
            "stage_id": -1,
        }
        if report_present:
            collected += 1
        if report_check.get("ok") is True and report_check.get("stage_runtime_verified") is True:
            verified += 1
            terminal_status = "IMPORTED"
        else:
            blockers.add(f"glm52_stage_worker_stage_report_not_verified:{provider or 'missing'}")
            terminal_status = "MISSING"
        if provider == "kaggle_jax_tpu":
            terminal_status = str(tpu_watch_report.get("last_status") or terminal_status)
            if tpu_watch_report:
                blockers.add("kaggle_mcp_tpu_notebook_scheduler_queued") if "QUEUED" in terminal_status.upper() else None
            direct_tpu_report_verified = (
                report_check.get("ok") is True
                and report_check.get("stage_runtime_verified") is True
            )
            if not direct_tpu_report_verified and tpu_watch_report.get("stage_runtime_report_verified") is not True:
                blockers.add("glm52_mcp_tpu_stage_runtime_not_ready")
        pushes.append({
            "schema": "glm52_kaggle_stage_worker_push_entry_v1",
            "provider": provider,
            "stage_id": _stage_report_id(report_path, stage_id) if report_path is not None else stage_id,
            "stage_ids": stage_ids_for_entry(entry),
            "package_dir": package_dir,
            "kernel_ref": str(tpu_watch_report.get("ref") or "") if provider == "kaggle_jax_tpu" else "",
            "requested_accelerator": accelerator_for_provider(provider, args),
            "pushed": False,
            "imported_stage_report": report_present,
            "mcp_watch_report_path": str(getattr(args, "tpu_watch_report", "") or "") if provider == "kaggle_jax_tpu" else "",
            "mcp_watch_observation_count": len(_list(tpu_watch_report.get("observations"))) if provider == "kaggle_jax_tpu" else 0,
            "terminal_status": terminal_status,
            "output_collected": report_present,
            "stage_report_path": str(report_path) if report_path is not None else "",
            "stage_report_present": report_present,
            "stage_report_check": report_check,
            "stage_runtime_verified": report_check.get("ok") is True and report_check.get("stage_runtime_verified") is True,
            "cleanup_performed": bool(report_present and getattr(args, "import_cleanup_verified", False)),
            "public_artifact_safe": True,
        })
    providers = {push["provider"] for push in pushes}
    for provider in REQUIRED_PROVIDERS:
        if provider not in providers:
            blockers.add(f"glm52_stage_worker_push_provider_missing:{provider}")
    if collected < len(args.providers):
        blockers.add("glm52_stage_worker_live_reports_missing")
    if verified < len(args.providers):
        blockers.add("glm52_stage_worker_live_reports_not_verified")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "mode": "import",
        "ok": True,
        "glm52_stage_worker_push_probe_ready": True,
        "live_run_performed": collected > 0,
        "stage_runtime_reports_collected": collected,
        "stage_runtime_reports_verified": verified,
        "same_request_route_verified": False,
        "stage_runtime_adapter_verified": False,
        "pushes": pushes,
        "supporting_artifacts": {
            "import_stage_reports": [str(path) for path in getattr(args, "import_stage_report", []) or []],
            "mcp_tpu_watch": str(getattr(args, "tpu_watch_report", "") or ""),
        },
        "blockers": sorted(blockers),
        "completion_boundary": {
            "preflight_is_not_runtime_success": True,
            "push_required": True,
            "terminal_kernel_output_required": True,
            "stage_runtime_check_required": True,
            "same_request_probe_required": True,
        },
        "public_artifact_safe": True,
        "safety": {
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


def _kernel_ref_from_metadata(package_dir: str) -> str:
    metadata = load_json(Path(package_dir) / "kernel-metadata.json")
    return str(metadata.get("id") or "")


def _kernel_ref_for_entry(entry: dict[str, Any], package_dir: str) -> str:
    kernel_ref = str(entry.get("kernel_ref") or "").strip()
    if kernel_ref:
        return kernel_ref
    return _kernel_ref_from_metadata(package_dir)


def run_live(
    args: argparse.Namespace,
    package_report: dict[str, Any],
    *,
    runner: Runner = subprocess.run,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    entries = package_entries(package_report, args.providers, args.stage_ids)
    pushes: list[dict[str, Any]] = []
    collected = 0
    verified = 0
    blockers: set[str] = set()
    for entry in entries:
        provider = str(entry.get("provider") or "")
        stage_id = _int(entry.get("stage_id"))
        package_dir = str(entry.get("package_dir") or "")
        kernel_ref = _kernel_ref_for_entry(entry, package_dir)
        push_command = push_command_for_package(package_dir, provider, args)
        private_runtime_env = write_private_runtime_env_file(package_dir, args)
        try:
            push_step = run_step(
                "kaggle_kernel_push",
                push_command,
                runner=runner,
                timeout_seconds=args.command_timeout_seconds,
                env=env,
            )
        finally:
            private_runtime_env = remove_private_runtime_env_file(package_dir, private_runtime_env)
        status_step: dict[str, Any] = {}
        output_step: dict[str, Any] = {}
        cleanup_step: dict[str, Any] = {}
        terminal_status = ""
        report_path: Path | None = None
        report_present = False
        report_check = {
            "ok": False,
            "error_count": 0,
            "errors": [],
            "stage_runtime_verified": False,
            "provider": "",
            "stage_id": -1,
        }
        if push_step.get("ok") and kernel_ref:
            push_error_blocker = kaggle_push_error_blocker(provider, push_step)
        else:
            push_error_blocker = kaggle_push_error_blocker(provider, push_step)
        push_accepted = bool(kernel_ref and not push_error_blocker)
        if push_accepted:
            deadline = time.monotonic() + args.wait_seconds
            first_poll = True
            while first_poll or time.monotonic() <= deadline:
                first_poll = False
                status_step = run_step(
                    "kaggle_kernel_status",
                    ["kaggle", "kernels", "status", kernel_ref],
                    runner=runner,
                    timeout_seconds=args.command_timeout_seconds,
                    env=env,
                )
                terminal_status = extract_status(f"{status_step.get('stdout_tail') or ''}\n{status_step.get('stderr_tail') or ''}")
                if terminal_status in TERMINAL_STATUSES:
                    break
                time.sleep(max(0.1, args.poll_interval_seconds))
            blockers.update(kaggle_step_blockers(provider, status_step, "status"))
            if terminal_status in NONTERMINAL_STATUSES:
                blockers.add(f"kaggle_kernel_wait_timeout:{provider or 'missing'}")
            terminal_blocker = terminal_status_blocker(provider, terminal_status)
            if terminal_blocker:
                blockers.add(terminal_blocker)
            if terminal_status == "COMPLETE":
                output_dir = Path(args.output_dir) / "notebook-output" / f"stage-{stage_id}-{provider}"
                output_step = run_step(
                    "kaggle_kernel_output",
                    [
                        "kaggle",
                        "kernels",
                        "output",
                        kernel_ref,
                        "-p",
                        str(output_dir),
                        "--force",
                        "--file-pattern",
                        "glm52_kaggle_stage_runtime_report.json",
                    ],
                    runner=runner,
                    timeout_seconds=args.command_timeout_seconds,
                    env=env,
                )
                blockers.update(kaggle_step_blockers(provider, output_step, "output"))
                if output_step.get("ok"):
                    report_path = stage_report_path(output_dir)
                    report_present = report_path.is_file()
                    report_check = check_stage_report(report_path)
                    if report_present:
                        collected += 1
                    if report_check.get("ok") is True and report_check.get("stage_runtime_verified") is True:
                        verified += 1
                    else:
                        blockers.add(f"glm52_stage_worker_stage_report_not_verified:{provider or 'missing'}")
                    if not report_present:
                        blockers.add(f"kaggle_kernel_output_stage_report_missing:{provider or 'missing'}")
            retain_nonterminal_tpu = bool(
                provider == "kaggle_jax_tpu"
                and args.retain_nonterminal_tpu
                and terminal_status in NONTERMINAL_STATUSES
            )
            retain_nonterminal_gpu = bool(
                provider == "kaggle_cuda"
                and args.retain_nonterminal_gpu
                and terminal_status in NONTERMINAL_STATUSES
            )
            retain_nonterminal_cpu = bool(
                provider == "kaggle_cpu"
                and args.retain_nonterminal_cpu
                and terminal_status in NONTERMINAL_STATUSES
            )
            if retain_nonterminal_tpu or retain_nonterminal_gpu or retain_nonterminal_cpu:
                retain_reason = (
                    "retain_nonterminal_tpu"
                    if retain_nonterminal_tpu
                    else "retain_nonterminal_gpu"
                    if retain_nonterminal_gpu
                    else "retain_nonterminal_cpu"
                )
                retain_blocker = (
                    "kaggle_tpu_kernel_retained_for_queue"
                    if retain_nonterminal_tpu
                    else "kaggle_gpu_kernel_retained_for_queue_or_run"
                    if retain_nonterminal_gpu
                    else "kaggle_cpu_kernel_retained_for_run"
                )
                blockers.add(
                    retain_blocker
                )
                cleanup_step = {
                    "name": "kaggle_kernel_delete",
                    "ok": False,
                    "skipped": True,
                    "reason": retain_reason,
                    "command_line": shell_command(["kaggle", "kernels", "delete", kernel_ref, "-y"]),
                }
            else:
                cleanup_step = run_step(
                    "kaggle_kernel_delete",
                    ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                    runner=runner,
                    timeout_seconds=args.command_timeout_seconds,
                    env=env,
                )
                blockers.update(kaggle_step_blockers(provider, cleanup_step, "delete"))
        else:
            blockers.add(f"glm52_stage_worker_push_failed:{provider or 'missing'}")
            if push_error_blocker:
                blockers.add(push_error_blocker)
        pushes.append({
            "schema": "glm52_kaggle_stage_worker_push_entry_v1",
            "provider": provider,
            "stage_id": stage_id,
            "stage_ids": stage_ids_for_entry(entry),
            "package_dir": package_dir,
            "kernel_ref": kernel_ref,
            "requested_accelerator": accelerator_for_provider(provider, args),
            "pushed": push_accepted,
            "push_error_blocker": push_error_blocker,
            "terminal_status": terminal_status,
            "output_collected": output_step.get("ok") is True,
            "stage_report_path": str(report_path) if report_path is not None else "",
            "stage_report_present": report_present,
            "stage_report_check": report_check,
            "stage_runtime_verified": report_check.get("ok") is True and report_check.get("stage_runtime_verified") is True,
            "cleanup_performed": cleanup_step.get("ok") is True,
            **private_runtime_env,
            "steps": [step for step in [push_step, status_step, output_step, cleanup_step] if step],
            "public_artifact_safe": True,
        })
    if collected < len(entries):
        blockers.add("glm52_stage_worker_live_reports_missing")
    if verified < len(entries):
        blockers.add("glm52_stage_worker_live_reports_not_verified")
    if not entries:
        blockers.add("glm52_stage_worker_push_entries_missing")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "mode": "live",
        "ok": True,
        "glm52_stage_worker_push_probe_ready": True,
        "live_run_performed": True,
        "stage_runtime_reports_collected": collected,
        "stage_runtime_reports_verified": verified,
        "same_request_route_verified": False,
        "stage_runtime_adapter_verified": False,
        "pushes": pushes,
        "blockers": sorted(blockers),
        "completion_boundary": {
            "preflight_is_not_runtime_success": True,
            "push_required": True,
            "terminal_kernel_output_required": True,
            "stage_runtime_check_required": True,
            "same_request_probe_required": True,
        },
        "public_artifact_safe": True,
        "safety": {
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


def _retain_blocker_for_provider(provider: str) -> str:
    if provider == "kaggle_jax_tpu":
        return "kaggle_tpu_kernel_retained_for_queue"
    if provider == "kaggle_cuda":
        return "kaggle_gpu_kernel_retained_for_queue_or_run"
    if provider == "kaggle_cpu":
        return "kaggle_cpu_kernel_retained_for_run"
    return "kaggle_kernel_retained_for_nonterminal_status"


def _retain_reason_for_provider(provider: str) -> str:
    if provider == "kaggle_jax_tpu":
        return "retain_nonterminal_tpu"
    if provider == "kaggle_cuda":
        return "retain_nonterminal_gpu"
    if provider == "kaggle_cpu":
        return "retain_nonterminal_cpu"
    return "retain_nonterminal_kernel"


def run_collect(
    args: argparse.Namespace,
    package_report: dict[str, Any],
    *,
    runner: Runner = subprocess.run,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    entries = package_entries(package_report, args.providers, args.stage_ids)
    pushes: list[dict[str, Any]] = []
    collected = 0
    verified = 0
    blockers: set[str] = set()
    for entry in entries:
        provider = str(entry.get("provider") or "")
        stage_id = _int(entry.get("stage_id"))
        package_dir = str(entry.get("package_dir") or "")
        kernel_ref = _kernel_ref_for_entry(entry, package_dir)
        status_step: dict[str, Any] = {}
        output_step: dict[str, Any] = {}
        cleanup_step: dict[str, Any] = {}
        terminal_status = ""
        report_path: Path | None = None
        report_present = False
        report_check = {
            "ok": False,
            "error_count": 0,
            "errors": [],
            "stage_runtime_verified": False,
            "provider": "",
            "stage_id": -1,
        }
        if kernel_ref:
            deadline = time.monotonic() + args.wait_seconds
            first_poll = True
            while first_poll or time.monotonic() <= deadline:
                first_poll = False
                status_step = run_step(
                    "kaggle_kernel_status",
                    ["kaggle", "kernels", "status", kernel_ref],
                    runner=runner,
                    timeout_seconds=args.command_timeout_seconds,
                    env=env,
                )
                terminal_status = extract_status(f"{status_step.get('stdout_tail') or ''}\n{status_step.get('stderr_tail') or ''}")
                if terminal_status in TERMINAL_STATUSES:
                    break
                time.sleep(max(0.1, args.poll_interval_seconds))
            blockers.update(kaggle_step_blockers(provider, status_step, "status"))
            if terminal_status in NONTERMINAL_STATUSES:
                blockers.add(f"kaggle_kernel_wait_timeout:{provider or 'missing'}")
            terminal_blocker = terminal_status_blocker(provider, terminal_status)
            if terminal_blocker:
                blockers.add(terminal_blocker)
            if terminal_status == "COMPLETE":
                output_dir = Path(args.output_dir) / "notebook-output" / f"stage-{stage_id}-{provider}"
                output_step = run_step(
                    "kaggle_kernel_output",
                    [
                        "kaggle",
                        "kernels",
                        "output",
                        kernel_ref,
                        "-p",
                        str(output_dir),
                        "--force",
                        "--file-pattern",
                        "glm52_kaggle_stage_runtime_report.json",
                    ],
                    runner=runner,
                    timeout_seconds=args.command_timeout_seconds,
                    env=env,
                )
                blockers.update(kaggle_step_blockers(provider, output_step, "output"))
                if output_step.get("ok"):
                    report_path = stage_report_path(output_dir)
                    report_present = report_path.is_file()
                    report_check = check_stage_report(report_path)
                    if report_present:
                        collected += 1
                    if report_check.get("ok") is True and report_check.get("stage_runtime_verified") is True:
                        verified += 1
                    else:
                        blockers.add(f"glm52_stage_worker_stage_report_not_verified:{provider or 'missing'}")
                    if not report_present:
                        blockers.add(f"kaggle_kernel_output_stage_report_missing:{provider or 'missing'}")
                cleanup_step = run_step(
                    "kaggle_kernel_delete",
                    ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                    runner=runner,
                    timeout_seconds=args.command_timeout_seconds,
                    env=env,
                )
                blockers.update(kaggle_step_blockers(provider, cleanup_step, "delete"))
            elif terminal_status in NONTERMINAL_STATUSES:
                blockers.add(_retain_blocker_for_provider(provider))
                cleanup_step = {
                    "name": "kaggle_kernel_delete",
                    "ok": False,
                    "skipped": True,
                    "reason": _retain_reason_for_provider(provider),
                    "command_line": shell_command(["kaggle", "kernels", "delete", kernel_ref, "-y"]),
                }
            elif terminal_status:
                blockers.add(f"kaggle_kernel_terminal_without_stage_report:{provider or 'missing'}")
        else:
            blockers.add(f"glm52_stage_worker_kernel_ref_missing:{provider or 'missing'}")
        pushes.append({
            "schema": "glm52_kaggle_stage_worker_push_entry_v1",
            "provider": provider,
            "stage_id": stage_id,
            "stage_ids": stage_ids_for_entry(entry),
            "package_dir": package_dir,
            "kernel_ref": kernel_ref,
            "requested_accelerator": accelerator_for_provider(provider, args),
            "pushed": False,
            "existing_kernel_observed": bool(kernel_ref),
            "push_error_blocker": "",
            "terminal_status": terminal_status,
            "output_collected": output_step.get("ok") is True,
            "stage_report_path": str(report_path) if report_path is not None else "",
            "stage_report_present": report_present,
            "stage_report_check": report_check,
            "stage_runtime_verified": report_check.get("ok") is True and report_check.get("stage_runtime_verified") is True,
            "cleanup_performed": cleanup_step.get("ok") is True,
            "steps": [step for step in [status_step, output_step, cleanup_step] if step],
            "public_artifact_safe": True,
        })
    if collected < len(entries):
        blockers.add("glm52_stage_worker_live_reports_missing")
    if verified < len(entries):
        blockers.add("glm52_stage_worker_live_reports_not_verified")
    if not entries:
        blockers.add("glm52_stage_worker_push_entries_missing")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "mode": "collect",
        "ok": True,
        "glm52_stage_worker_push_probe_ready": True,
        "live_run_performed": True,
        "stage_runtime_reports_collected": collected,
        "stage_runtime_reports_verified": verified,
        "same_request_route_verified": False,
        "stage_runtime_adapter_verified": False,
        "pushes": pushes,
        "blockers": sorted(blockers),
        "completion_boundary": {
            "preflight_is_not_runtime_success": True,
            "push_required": True,
            "terminal_kernel_output_required": True,
            "stage_runtime_check_required": True,
            "same_request_probe_required": True,
        },
        "public_artifact_safe": True,
        "safety": {
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


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    package_report = load_json(args.stage_worker_package_report)
    config_dir: tempfile.TemporaryDirectory[str] | None = None
    kaggle_env: dict[str, str] | None = None
    try:
        if args.mode == "import":
            report = build_import(args, package_report)
            leaks = public_redaction_errors(report)
            if leaks:
                report["ok"] = False
                report["public_artifact_safe"] = False
                report["safety"]["public_artifact_safe"] = False
                report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
            return report
        try:
            kaggle_env, config_dir = kaggle_env_for_token_section(args)
        except (OSError, RuntimeError, json.JSONDecodeError):
            report = build_preflight(args, package_report)
            report["mode"] = args.mode
            report["blockers"] = sorted(set(_list(report.get("blockers")) + ["kaggle_token_section_unavailable"]))
            return report
        report = (
            build_preflight(args, package_report)
            if args.mode == "preflight"
            else run_collect(args, package_report, runner=runner, env=kaggle_env)
            if args.mode == "collect"
            else run_live(args, package_report, runner=runner, env=kaggle_env)
        )
    finally:
        if config_dir is not None:
            config_dir.cleanup()
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "live", "collect", "import"], default="preflight")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-worker-package-report", required=True)
    parser.add_argument("--providers", default=",".join(REQUIRED_PROVIDERS))
    parser.add_argument("--stage-ids", default="", help="Optional comma-separated stage ids to run from the selected providers.")
    parser.add_argument("--import-stage-report", action="append", default=[])
    parser.add_argument("--tpu-watch-report", default="")
    parser.add_argument("--import-cleanup-verified", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--command-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=0)
    parser.add_argument("--token-file", default="~/.config/crowdtensor/kaggle-tokens.md")
    parser.add_argument("--token-section", default="")
    parser.add_argument("--raw-token-file", default="", help="Optional private Kaggle token file outside section format.")
    parser.add_argument("--raw-token-username", default="", help="Username for raw token files that contain only a Kaggle key.")
    parser.add_argument("--hf-token-env", default="HF_TOKEN,HUGGING_FACE_HUB_TOKEN", help="Comma-separated private env vars to forward as HF token inside uploaded kernels only.")
    parser.add_argument("--coordinator-url", default="", help="Private Coordinator base URL injected into uploaded kernels only.")
    parser.add_argument("--coordinator-token-file", default="", help="Private file containing the Coordinator bearer token for uploaded kernels.")
    parser.add_argument("--coordinator-task-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--coordinator-poll-interval-seconds", type=float, default=0.0)
    parser.add_argument("--coordinator-stage-task-limit", type=int, default=1)
    parser.add_argument("--full-prefix-prefill-length", type=int, default=0)
    parser.add_argument("--full-prefix-dsa-mask-topk", type=int, default=0)
    parser.add_argument("--full-prefix-executed-expert-count", type=int, default=0)
    parser.add_argument("--full-prefix-top-k", type=int, default=0)
    parser.add_argument("--full-prefix-row-block-size", type=int, default=0)
    parser.add_argument("--full-prefix-max-tensor-bytes", type=int, default=0)
    parser.add_argument("--full-prefix-max-block-bytes", type=int, default=0)
    parser.add_argument("--cpu-group-stage-attempt-seconds", type=float, default=0.0)
    parser.add_argument("--cpu-group-stage-poll-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--tpu-accelerator", default="tpuV5e8")
    parser.add_argument("--retain-nonterminal-tpu", action="store_true")
    parser.add_argument("--retain-nonterminal-gpu", action="store_true")
    parser.add_argument("--retain-nonterminal-cpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.kernel_timeout_seconds < 0:
        raise SystemExit("--kernel-timeout-seconds must be non-negative")
    if args.coordinator_stage_task_limit < 1:
        raise SystemExit("--coordinator-stage-task-limit must be at least 1")
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
    args.providers = [item.strip() for item in str(args.providers).split(",") if item.strip()]
    args.stage_ids = {
        int(item.strip())
        for item in str(args.stage_ids or "").split(",")
        if item.strip()
    }
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_kaggle_stage_worker_push_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Mode: {report.get('mode')}")
        print(f"Live run performed: {report.get('live_run_performed')}")
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
