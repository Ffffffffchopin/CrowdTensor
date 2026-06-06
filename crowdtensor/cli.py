"""User-facing CrowdTensor command line entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import support_bundle  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - fallback for unusual packaging layouts
    support_bundle = None

from crowdtensor.p2p_lite import (
    PEER_SCHEMA,
    fetch_peer_catalog,
    post_announce,
    sanitize_peer,
    sign_peer_announcement,
    stable_peer_id,
)
from crowdtensor.real_p2p import (
    DEFAULT_DISCOVERY_BACKEND,
    PROVIDER_RECORD_SCHEMA,
    DISCOVERY_BACKENDS,
    LIBP2P_KAD_BACKEND,
    LIBP2P_KAD_COMPAT_BACKEND,
    build_provider_record,
    fetch_provider_catalog,
    post_provider_record,
    post_route_lookup,
)
from crowdtensor.session_protocol import (
    build_route_decision,
    build_session_request,
    coordinator_payload_for_request,
    peer_backend_compatible,
    peer_model_compatible,
    required_stage_capabilities,
    safe_generation_summary,
    safe_stream_event,
    safe_stream_events,
    stable_hash_payload,
    stable_hash_text,
)


SUMMARY_SCHEMA = "local_proof_summary_v1"
CLEANUP_SCHEMA = "cleanup_report_v1"
REMOTE_RUNBOOK_CLI_SCHEMA = "remote_runbook_cli_v1"
REMOTE_ACCEPTANCE_CLI_SCHEMA = "remote_acceptance_cli_v1"
REMOTE_HOME_DEMO_SCHEMA = "remote_home_compute_demo_v1"
HOME_INFERENCE_CLI_SCHEMA = "home_inference_cli_v1"
LLM_INFERENCE_CLI_SCHEMA = "llm_inference_cli_v1"
CPU_INFERENCE_BETA_CLI_SCHEMA = "cpu_inference_beta_cli_v1"
CPU_INFERENCE_BETA_RC_CLI_SCHEMA = "cpu_inference_beta_rc_cli_v1"
SHARDED_INFERENCE_CLI_SCHEMA = "sharded_inference_cli_v1"
MICRO_LLM_SHARDED_CLI_SCHEMA = "micro_llm_sharded_cli_v1"
MICRO_LLM_ARTIFACT_CLI_SCHEMA = "micro_llm_artifact_cli_v1"
REAL_LLM_SHARDED_CLI_SCHEMA = "real_llm_sharded_cli_v1"
REMOTE_SHARDED_INFERENCE_BETA_CLI_SCHEMA = "remote_sharded_inference_beta_cli_v1"
REMOTE_MICRO_LLM_SHARDED_BETA_CLI_SCHEMA = "remote_micro_llm_sharded_beta_cli_v1"
REMOTE_REAL_LLM_SHARDED_BETA_CLI_SCHEMA = "remote_real_llm_sharded_beta_cli_v1"
KAGGLE_REAL_RUNTIME_SCHEMA = "kaggle_real_runtime_acceptance_v1"
MICRO_LLM_LIVE_RC_CLI_SCHEMA = "micro_llm_live_rc_cli_v1"
REAL_LLM_LIVE_RC_CLI_SCHEMA = "real_llm_live_rc_cli_v1"
REAL_LLM_INTERNET_ALPHA_CLI_SCHEMA = "real_llm_internet_alpha_cli_v1"
REAL_LLM_INTERNET_BETA_CLI_SCHEMA = "real_llm_internet_beta_cli_v1"
SWARM_INFERENCE_BETA_CLI_SCHEMA = "swarm_inference_beta_cli_v1"
PUBLIC_SWARM_INFERENCE_ALPHA_CLI_SCHEMA = "public_swarm_inference_alpha_cli_v1"
PUBLIC_SWARM_INFERENCE_ALPHA_RC_CLI_SCHEMA = "public_swarm_inference_alpha_rc_cli_v1"
PUBLIC_SWARM_INFERENCE_BETA_CLI_SCHEMA = "public_swarm_inference_beta_cli_v1"
PUBLIC_SWARM_INFERENCE_BETA_RC_CLI_SCHEMA = "public_swarm_inference_beta_rc_cli_v1"
PUBLIC_SWARM_PRODUCT_BETA_CLI_SCHEMA = "public_swarm_product_beta_cli_v1"
PUBLIC_SWARM_DEVELOPER_PREVIEW_CLI_SCHEMA = "public_swarm_developer_preview_cli_v1"
PUBLIC_SWARM_LIVE_PREVIEW_RC_CLI_SCHEMA = "public_swarm_live_preview_rc_cli_v1"
PUBLIC_SWARM_OPERATOR_PREVIEW_CLI_SCHEMA = "public_swarm_operator_preview_cli_v1"
PUBLIC_SWARM_TRIAL_CLI_SCHEMA = "public_swarm_trial_cli_v1"
PUBLIC_SWARM_PREVIEW_V04_CLI_SCHEMA = "public_swarm_preview_v04_cli_v1"
P2P_SWARM_INFERENCE_V06_CLI_SCHEMA = "p2p_swarm_inference_v06_cli_v1"
PUBLIC_P2P_SWARM_INFERENCE_V1_RC_CLI_SCHEMA = "public_p2p_swarm_inference_v1_rc_cli_v1"
REAL_P2P_SWARM_INFERENCE_CORE_RC_CLI_SCHEMA = "real_p2p_swarm_inference_core_rc_cli_v1"
PETALS_CLASS_P2P_CANDIDATE_CLI_SCHEMA = "petals_class_p2p_candidate_cli_v1"
PUBLIC_REAL_LLM_SWARM_BETA_CLI_SCHEMA = "public_real_llm_swarm_beta_cli_v1"
USABLE_SWARM_INFERENCE_CLI_SCHEMA = "usable_swarm_inference_cli_v1"
PUBLIC_SWARM_INFERENCE_V2_CLI_SCHEMA = "public_swarm_inference_v2_cli_v1"
PUBLIC_SWARM_GPU_INFERENCE_BETA_CLI_SCHEMA = "public_swarm_gpu_inference_beta_cli_v1"
GPU_SHARDED_GENERATION_BETA_CLI_SCHEMA = "gpu_sharded_generation_beta_cli_v1"
PUBLIC_SWARM_PRODUCT_CLI_SCHEMA = "public_swarm_product_cli_v1"
INFER_CLI_SCHEMA = "crowdtensor_infer_cli_v1"
P2P_LITE_CLI_SCHEMA = "p2p_lite_cli_v1"
P2PD_CLI_SCHEMA = "p2pd_cli_v1"
REAL_P2P_CLI_SCHEMA = "real_p2p_cli_v1"
P2P_DAEMON_CLI_SCHEMA = "p2p_daemon_cli_v1"
DEFAULT_P2P_BOOTSTRAP = "http://127.0.0.1:8788"
DEFAULT_PRODUCT_GENERATE_PROMPT = "CrowdTensor routes a tiny model across two stage miners."
INFER_PROMPT_PLACEHOLDER = "<prompt>"
INFER_BATCH_PROMPTS_PLACEHOLDER = "<prompt-1>,<prompt-2>"
OBSERVER_TOKEN_ENV_PLACEHOLDER = "${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}"
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "sharded_inference_result",
    "activation_results",
    "logits",
    "external_llm_results",
    "output_text",
    "Bearer ",
)
PLACEHOLDER_SECRET_PATTERN = re.compile(r"\b[A-Za-z0-9_.-]+-secret\b")
REMOTE_DEMO_WORKLOADS = ["model-bundle", "external-llm", "sharded-model-bundle", "micro-llm-sharded", "real-llm-sharded"]
CLEANUP_TMP_DIR_PATTERNS = (
    "crowdtensor_local_proof*",
    "crowdtensor_demo_manifest_*",
    "crowdtensor_cli_test_*",
    "crowdtensor_*_test_*",
)
CLEANUP_REPORT_PATTERNS = (
    "crowdtensor_*.json",
    "crowdtensor_*.md",
)
PROTECTED_REPO_PARTS = {".git", ".venv", "venv", "state"}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def request_json_url(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    admin_token: str = "",
    observer_token: str = "",
    timeout: float = 10.0,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def stream_progress_summary(
    events: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    source: str = "admin-session-stream",
    expected_request_count: int = 1,
) -> dict[str, Any]:
    counts = [int(event.get("generated_token_count") or 0) for event in events if isinstance(event, dict)]
    target = max(1, int(max_new_tokens))
    monotonic = counts == sorted(counts) and len(counts) == len(set(counts))
    expected_counts = list(range(1, target + 1))
    complete = bool(expected_counts and all(count in counts for count in expected_counts))
    expected_requests = max(1, int(expected_request_count or 1))
    per_request: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        request_key = str(event.get("request_id") or event.get("prompt_hash") or "request-1")
        entry = per_request.setdefault(
            request_key,
            {
                "request_id": event.get("request_id"),
                "prompt_hash": event.get("prompt_hash"),
                "observed_token_counts": [],
            },
        )
        try:
            generated_count = int(event.get("generated_token_count") or 0)
        except (TypeError, ValueError):
            generated_count = 0
        if generated_count > 0:
            entry["observed_token_counts"].append(generated_count)
        entry.setdefault("first_event_index", index)
        entry["last_event_index"] = index
    per_request_progress: list[dict[str, Any]] = []
    for request_key, entry in sorted(per_request.items(), key=lambda item: int(item[1].get("first_event_index") or 0)):
        request_counts = list(entry.get("observed_token_counts") or [])
        request_monotonic = request_counts == sorted(request_counts) and len(request_counts) == len(set(request_counts))
        request_complete = bool(expected_counts and all(count in request_counts for count in expected_counts))
        per_request_progress.append({
            "request_key": request_key,
            "request_id": entry.get("request_id"),
            "prompt_hash": entry.get("prompt_hash"),
            "event_count": len(request_counts),
            "observed_token_counts": request_counts,
            "max_observed_token_count": max(request_counts) if request_counts else 0,
            "target_token_count": target,
            "monotonic_progress": request_monotonic,
            "stream_progress_complete": request_complete,
        })
    per_request_complete = bool(
        len(per_request_progress) >= expected_requests
        and all(item.get("stream_progress_complete") for item in per_request_progress[:expected_requests])
    )
    per_request_monotonic = bool(
        len(per_request_progress) >= expected_requests
        and all(item.get("monotonic_progress") for item in per_request_progress[:expected_requests])
    )
    observations: list[float] = []
    for event in events:
        if not isinstance(event, dict) or event.get("observed_at") is None:
            continue
        try:
            observations.append(float(event.get("observed_at")))
        except (TypeError, ValueError):
            continue
    latency: dict[str, Any] = {
        "first_event_observed_at": observations[0] if observations else None,
        "last_event_observed_at": observations[-1] if observations else None,
        "event_span_seconds": (
            round(observations[-1] - observations[0], 6)
            if len(observations) >= 2
            else 0.0
        ),
    }
    return {
        "source": source,
        "event_count": len(events),
        "progress_counts": counts,
        "monotonic_counts_ready": monotonic,
        "monotonic_progress": monotonic,
        "all_token_events_ready": complete,
        "stream_progress_complete": complete,
        "expected_request_count": expected_requests,
        "per_request_progress": per_request_progress,
        "per_request_progress_complete": per_request_complete,
        "per_request_monotonic_progress": per_request_monotonic,
        "complete_token_count": max(counts) if counts else 0,
        "observed_token_counts": counts,
        "max_observed_token_count": max(counts) if counts else 0,
        "target_token_count": target,
        "max_new_tokens": target,
        "latency": latency,
    }


def parse_prompt_texts_arg(primary: str = "", batch: str = "") -> list[str]:
    if str(batch or "").strip():
        prompts = [item.strip() for item in str(batch).split(",") if item.strip()]
    else:
        prompts = [str(primary or "").strip()] if str(primary or "").strip() else []
    if not prompts:
        raise ValueError("prompt_text is required")
    if len(prompts) > 4:
        raise ValueError("prompt_texts must contain at most 4 prompts")
    for prompt in prompts:
        if len(prompt) > 256:
            raise ValueError("prompt_text must be at most 256 characters")
    return prompts


def redacted_command(command: list[str], sensitive_flags: set[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for item in command:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        result.append(item)
        if item in sensitive_flags:
            redact_next = True
    return result


def command_line(command: list[str]) -> str:
    def quote_part(part: Any) -> str:
        text = str(part)
        if text == OBSERVER_TOKEN_ENV_PLACEHOLDER:
            return text
        return shlex.quote(text)

    return " ".join(quote_part(part) for part in command)


def command_entry(
    label: str,
    command: list[str],
    *,
    sensitive_flags: set[str] | None = None,
    requires_env: list[str] | None = None,
) -> dict[str, Any]:
    safe_command = redacted_command(command, sensitive_flags or set())
    entry: dict[str, Any] = {
        "label": label,
        "command": safe_command,
        "command_line": command_line(safe_command),
    }
    if requires_env:
        entry["requires_env"] = list(requires_env)
    return entry


def local_coordinator_port_from_url(coordinator_url: str, default: int = 8787) -> int:
    try:
        parsed = urlparse(str(coordinator_url or ""))
        return int(parsed.port or default)
    except (TypeError, ValueError):
        return default


def local_infer_command_line(item: dict[str, Any], report: dict[str, Any]) -> str:
    command = item.get("command") if isinstance(item.get("command"), list) else []
    prompt = str(report.get("local_prompt_text") or "")
    prompt_texts = str(report.get("local_prompt_texts") or "")
    if command:
        rendered = [str(part) for part in command]
        if prompt_texts:
            rendered = [
                prompt_texts if part == INFER_BATCH_PROMPTS_PLACEHOLDER else part
                for part in rendered
                if part != INFER_PROMPT_PLACEHOLDER
            ]
        elif prompt:
            rendered = [prompt if part == INFER_PROMPT_PLACEHOLDER else part for part in rendered]
        return command_line(rendered)
    return str(item.get("command_line") or command_line([str(part) for part in command]))


def local_generate_command_line(item: dict[str, Any], report: dict[str, Any]) -> str:
    command = item.get("command") if isinstance(item.get("command"), list) else []
    if not command:
        return str(item.get("command_line") or "")
    prompt = str(report.get("local_prompt_text") or "")
    prompt_texts = str(report.get("local_prompt_texts") or "")
    rendered = [str(part) for part in command]
    if prompt_texts:
        rendered = [prompt_texts if part == INFER_BATCH_PROMPTS_PLACEHOLDER else part for part in rendered]
    elif prompt:
        rendered = [prompt if part == INFER_PROMPT_PLACEHOLDER else part for part in rendered]
    return command_line(rendered)


def format_p2p_status(p2p: dict[str, Any]) -> str:
    discovery = p2p.get("discovery") if isinstance(p2p.get("discovery"), dict) else {}
    parts = [
        f"enabled={p2p.get('enabled')}",
        f"backend={p2p.get('backend') or 'none'}",
    ]
    if p2p.get("bootstrap"):
        parts.append(f"bootstrap={p2p.get('bootstrap')}")
    if p2p.get("swarm_id") and p2p.get("swarm_id") != "default":
        parts.append(f"swarm_id={p2p.get('swarm_id')}")
    if p2p.get("peer_count") is not None:
        parts.append(f"peers={p2p.get('peer_count')}")
    if discovery:
        parts.append(f"discovery_ok={discovery.get('ok')}")
        if discovery.get("error"):
            parts.append(f"discovery_error={discovery.get('error')}")
    return " ".join(str(part) for part in parts)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize(value: Any) -> Any:
    if support_bundle is not None:
        return support_bundle.sanitize(value)
    return value


def redact_text(text: str, redact_values: list[str] | None = None) -> str:
    redacted = text
    for value in redact_values or []:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    redacted = PLACEHOLDER_SECRET_PATTERN.sub("<redacted>", redacted)
    return redacted


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    return value


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def diagnosis_codes(*payloads: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summary = payload.get("diagnosis_summary")
        if isinstance(summary, dict):
            for code in summary.get("codes") or []:
                if isinstance(code, str):
                    codes.add(code)
    return sorted(codes)


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def has_protected_repo_part(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return any(part in PROTECTED_REPO_PARTS for part in relative.parts)


def path_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            continue
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def age_hours(path: Path, *, now: float | None = None) -> float:
    current = time.time() if now is None else now
    return max(0.0, (current - path.stat().st_mtime) / 3600.0)


def cleanup_candidate(
    path: Path,
    *,
    kind: str,
    reason: str,
    root: Path,
    tmp_root: Path,
    older_than_hours: float,
    include_reports: bool,
    age_gated: bool,
    report_gated: bool = False,
) -> dict[str, Any]:
    candidate = {
        "path": str(path.resolve()),
        "kind": kind,
        "reason": reason,
        "bytes": 0,
        "eligible": False,
        "skip_reason": "",
    }
    try:
        if path.is_symlink():
            candidate["skip_reason"] = "symlink"
            return candidate
        resolved = path.resolve()
        if not (is_relative_to(resolved, root) or is_relative_to(resolved, tmp_root)):
            candidate["skip_reason"] = "outside_allowed_roots"
            return candidate
        if is_relative_to(resolved, root) and has_protected_repo_part(resolved, root):
            candidate["skip_reason"] = "protected_repo_path"
            return candidate
        candidate["bytes"] = path_size(resolved)
        if report_gated and not include_reports:
            candidate["skip_reason"] = "requires_include_reports"
            return candidate
        if age_gated:
            candidate["age_hours"] = round(age_hours(resolved), 3)
            if float(candidate["age_hours"]) < older_than_hours:
                candidate["skip_reason"] = "too_new"
                return candidate
        candidate["eligible"] = True
        return candidate
    except OSError as exc:
        candidate["skip_reason"] = f"stat_failed: {exc}"
        return candidate


def discover_cleanup_candidates(
    *,
    root: Path = ROOT,
    tmp_root: Path = Path("/tmp"),
    older_than_hours: float = 24.0,
    include_reports: bool = False,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate: dict[str, Any]) -> None:
        key = str(candidate.get("path"))
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    for cache_dir in root.rglob("__pycache__"):
        add(cleanup_candidate(
            cache_dir,
            kind="python_cache",
            reason="repo __pycache__ directory",
            root=root,
            tmp_root=tmp_root,
            older_than_hours=older_than_hours,
            include_reports=include_reports,
            age_gated=False,
        ))
    for pyc_file in root.rglob("*.pyc"):
        add(cleanup_candidate(
            pyc_file,
            kind="python_cache",
            reason="repo .pyc file",
            root=root,
            tmp_root=tmp_root,
            older_than_hours=older_than_hours,
            include_reports=include_reports,
            age_gated=False,
        ))
    for pattern in CLEANUP_TMP_DIR_PATTERNS:
        for temp_dir in tmp_root.glob(pattern):
            if temp_dir.is_dir() or temp_dir.is_symlink():
                add(cleanup_candidate(
                    temp_dir,
                    kind="tmp_dir",
                    reason=f"temporary CrowdTensor artifact matching {pattern}",
                    root=root,
                    tmp_root=tmp_root,
                    older_than_hours=older_than_hours,
                    include_reports=include_reports,
                    age_gated=True,
                ))
    for pattern in CLEANUP_REPORT_PATTERNS:
        for report in tmp_root.glob(pattern):
            if report.is_file() or report.is_symlink():
                add(cleanup_candidate(
                    report,
                    kind="report",
                    reason=f"optional CrowdTensor report matching {pattern}",
                    root=root,
                    tmp_root=tmp_root,
                    older_than_hours=older_than_hours,
                    include_reports=include_reports,
                    age_gated=True,
                    report_gated=True,
                ))
    return sorted(candidates, key=lambda item: str(item.get("path")))


def delete_candidate(path: Path) -> None:
    if path.is_symlink():
        raise OSError("refusing to delete symlink")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def build_cleanup_report(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    tmp_root: Path = Path("/tmp"),
) -> dict[str, Any]:
    candidates = discover_cleanup_candidates(
        root=root,
        tmp_root=tmp_root,
        older_than_hours=args.older_than_hours,
        include_reports=args.include_reports,
    )
    apply = bool(args.apply)
    deleted_bytes = 0
    errors: list[str] = []
    for candidate in candidates:
        candidate["action"] = "skipped"
        if not candidate.get("eligible"):
            continue
        if not apply:
            candidate["action"] = "dry_run"
            continue
        try:
            delete_candidate(Path(str(candidate["path"])))
        except OSError as exc:
            candidate["action"] = "error"
            candidate["error"] = str(exc)
            errors.append(str(candidate["path"]))
            continue
        candidate["action"] = "deleted"
        deleted_bytes += int(candidate.get("bytes") or 0)
    return sanitize({
        "schema": CLEANUP_SCHEMA,
        "generated_at": utc_now(),
        "ok": not errors,
        "mode": "apply" if apply else "dry_run",
        "root": str(root.resolve()),
        "tmp_root": str(tmp_root.resolve()),
        "older_than_hours": args.older_than_hours,
        "include_reports": bool(args.include_reports),
        "candidate_count": len(candidates),
        "deleted_bytes": deleted_bytes,
        "errors": errors,
        "candidates": candidates,
        "safety": {
            "dry_run_default": True,
            "reports_require_include_reports": True,
            "allowed_roots": [str(root.resolve()), str(tmp_root.resolve())],
            "protected_repo_parts": sorted(PROTECTED_REPO_PARTS),
        },
    })


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    cwd: Path,
    timeout_seconds: int,
    redact_secrets: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - started, 3)
        return (
            {
                "name": name,
                "ok": False,
                "returncode": None,
                "duration_seconds": elapsed,
                "error": "timeout",
            },
            {},
        )
    elapsed = round(time.monotonic() - started, 3)
    step = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": elapsed,
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    if not step["ok"]:
        if completed.stderr:
            step["stderr_tail"] = sanitize(redact_text(completed.stderr[-1000:], redact_secrets))
        if completed.stdout and not payload:
            step["stdout_tail"] = sanitize(redact_text(completed.stdout[-1000:], redact_secrets))
    return step, payload


def build_local_proof(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    doctor_payload: dict[str, Any] = {}
    if args.skip_doctor:
        steps.append({"name": "doctor", "ok": True, "skipped": True})
    else:
        doctor_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "doctor.py"),
            "--root",
            str(ROOT),
            "--state-dir",
            str(output_dir / "doctor-state"),
            "--port",
            "0",
            "--json",
        ]
        doctor_step, doctor_payload = run_json_step(
            "doctor",
            doctor_cmd,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
        )
        doctor_step["ok"] = bool(doctor_step.get("ok") and doctor_payload.get("ok"))
        if not doctor_step["ok"]:
            errors.append("doctor_failed")
        steps.append(doctor_step)

    matrix_cmd = [sys.executable, str(SCRIPTS_DIR / "runtime_matrix.py"), "--json"]
    matrix_step, matrix_payload = run_json_step(
        "runtime_matrix",
        matrix_cmd,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    matrix_step["ok"] = bool(matrix_step.get("ok") and matrix_payload.get("ok"))
    if not matrix_step["ok"]:
        errors.append("runtime_matrix_blocked")
    steps.append(matrix_step)

    home_payload: dict[str, Any] = {}
    manifest_payload: dict[str, Any] = {}
    if matrix_step["ok"]:
        home_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "home_compute_demo.py"),
            "--port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--json",
        ]
        home_step, home_payload = run_json_step(
            "home_compute_demo",
            home_cmd,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
        )
        home_step["ok"] = bool(home_step.get("ok") and home_payload.get("ok"))
        if not home_step["ok"]:
            errors.append("home_compute_demo_failed")
        steps.append(home_step)

        if home_step["ok"]:
            manifest_cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "demo_manifest_pack.py"),
                "--output-dir",
                str(output_dir),
                "--port",
                str(args.base_port + 1),
                "--request-count",
                str(args.request_count),
            ]
            manifest_step, manifest_payload = run_json_step(
                "demo_manifest",
                manifest_cmd,
                runner=runner,
                cwd=ROOT,
                timeout_seconds=args.timeout_seconds,
            )
            manifest_step["ok"] = bool(manifest_step.get("ok") and manifest_payload.get("ok"))
            if not manifest_step["ok"]:
                errors.append("demo_manifest_failed")
            steps.append(manifest_step)
    else:
        steps.append({"name": "home_compute_demo", "ok": False, "skipped": True, "reason": "runtime_matrix_blocked"})
        steps.append({"name": "demo_manifest", "ok": False, "skipped": True, "reason": "runtime_matrix_blocked"})

    manifest_json = output_dir / "demo_manifest.json"
    manifest_md = output_dir / "demo_manifest.md"
    summary_json = output_dir / "local_proof_summary.json"
    artifacts["demo_manifest_json"] = artifact_entry(
        manifest_json,
        output_dir,
        kind="demo_manifest",
        schema=str(manifest_payload.get("schema") or "demo_manifest_v1"),
        ok=manifest_payload.get("ok") if manifest_payload else None,
    )
    artifacts["demo_manifest_markdown"] = artifact_entry(manifest_md, output_dir, kind="demo_manifest_markdown")
    artifacts["local_proof_summary"] = {
        "kind": "local_proof_summary",
        "path": "local_proof_summary.json",
        "present": True,
        "schema": SUMMARY_SCHEMA,
    }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": utc_now(),
        "ok": not errors and all(bool(step.get("ok")) for step in steps if not step.get("skipped")),
        "output_dir": str(output_dir),
        "request_count": args.request_count,
        "base_port": args.base_port,
        "steps": steps,
        "diagnosis_codes": diagnosis_codes(matrix_payload, home_payload, manifest_payload),
        "artifacts": artifacts,
        "errors": errors,
        "limitations": [
            "CPU-only local proof; not production Swarm Inference",
            "Read-only model_bundle_infer rehearsal; not arbitrary prompt or real LLM serving",
            "No GPU pooling, WebGPU model shards, libp2p discovery, NAT traversal, or incentives are claimed",
        ],
        "recommended_next_commands": [
            "crowdtensor home-infer --json",
            "python3 scripts/demo_manifest_check.py --base-port 8914",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
            "python3 scripts/remote_demo_runbook_pack.py --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --output-dir dist/remote-demo",
        ],
    }
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary["errors"].append("sensitive_output_detected")
        summary["safety_error"] = "local proof summary contained secret-like fragments"

    summary = sanitize(summary)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _remote_summary_artifact(path: Path, output_dir: Path, *, kind: str, schema: str = "") -> dict[str, Any]:
    return artifact_entry(path, output_dir, kind=kind, schema=schema)


def build_home_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "home_compute_evidence.json"
    evidence_md = output_dir / "home_compute_evidence.md"
    summary_json = output_dir / "home_inference_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "home_compute_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.runtime_report:
        command.extend(["--runtime-report", args.runtime_report])
    step, payload = run_json_step(
        "home_compute_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    inference = payload.get("inference_summary") if isinstance(payload.get("inference_summary"), dict) else {}
    route = payload.get("route_decision") if isinstance(payload.get("route_decision"), dict) else {}
    scenario = payload.get("scenario") if isinstance(payload.get("scenario"), dict) else {}
    artifacts = {
        "home_compute_evidence_json": artifact_entry(
            evidence_json,
            output_dir,
            kind="home_compute_evidence",
            schema=str(payload.get("schema") or "home_compute_evidence_v1"),
            ok=payload.get("ok") if payload else None,
        ),
        "home_compute_evidence_markdown": artifact_entry(
            evidence_md,
            output_dir,
            kind="home_compute_evidence_markdown",
        ),
        "home_inference_cli_summary": {
            "kind": "home_inference_cli_summary",
            "path": "home_inference_cli_summary.json",
            "present": True,
            "schema": HOME_INFERENCE_CLI_SCHEMA,
        },
    }
    summary = {
        "schema": HOME_INFERENCE_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "port": args.port,
        "request_count": args.request_count,
        "scenario_id": args.scenario_id,
        "step": step,
        "evidence_schema": payload.get("schema") or "home_compute_evidence_v1",
        "route": {
            "name": route.get("name"),
            "target": route.get("target"),
            "workload": route.get("workload"),
            "confidence": route.get("confidence"),
            "usable_now": route.get("usable_now"),
        },
        "diagnosis_codes": diagnosis_codes(payload),
        "scenario": {
            "scenario_schema": scenario.get("scenario_schema") or inference.get("scenario_schema"),
            "scenario_id": scenario.get("scenario_id") or inference.get("scenario_id"),
            "scenario_description": scenario.get("scenario_description") or inference.get("scenario_description"),
            "scenario_request_count": scenario.get("scenario_request_count") or inference.get("scenario_request_count"),
        },
        "inference": {
            "present": inference.get("present"),
            "ok": inference.get("ok"),
            "workload_type": inference.get("workload_type"),
            "request_count": inference.get("request_count"),
            "request_trace_count": inference.get("request_trace_count"),
            "requests_per_second": inference.get("requests_per_second"),
            "read_only": inference.get("read_only"),
            "redaction_ok": inference.get("redaction_ok"),
        },
        "artifacts": artifacts,
        "safety": {
            "captured_output_redacted": True,
            "summary_excludes_raw_inference_payloads": True,
            "read_only_workload": "model_bundle_infer",
            "not_production": True,
        },
        "limitations": [
            "CPU-only read-only model_bundle_infer proof; not production Swarm Inference",
            "Does not provide arbitrary prompt serving, real LLM serving, GPU pooling, WebGPU shards, or P2P routing",
        ],
        "recommended_next_commands": [
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
            "crowdtensor remote-runbook --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --json",
        ],
    }
    summary = sanitize(summary)
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
        summary["safety_error"] = "home inference summary contained secret-like fragments"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_llm_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "external_llm_evidence.json"
    evidence_md = output_dir / "external_llm_evidence.md"
    summary_json = output_dir / "llm_inference_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "external_llm_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--llm-runtime-model-id",
        args.llm_runtime_model_id,
        "--llm-runtime-timeout",
        str(args.llm_runtime_timeout),
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.mock:
        command.append("--mock")
    if args.llm_runtime_cmd:
        command.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    if args.llm_runtime_url:
        command.extend(["--llm-runtime-url", args.llm_runtime_url])
    if args.llm_runtime_api_key:
        command.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    secret_values = [args.llm_runtime_url, args.llm_runtime_api_key]
    step, payload = run_json_step(
        "external_llm_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    adapter = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else {}
    llm_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    artifacts = {
        "external_llm_evidence_json": artifact_entry(
            evidence_json,
            output_dir,
            kind="external_llm_evidence",
            schema=str(payload.get("schema") or "external_llm_evidence_v1"),
            ok=payload.get("ok") if payload else None,
        ),
        "external_llm_evidence_markdown": artifact_entry(
            evidence_md,
            output_dir,
            kind="external_llm_evidence_markdown",
        ),
        "llm_inference_cli_summary": {
            "kind": "llm_inference_cli_summary",
            "path": "llm_inference_cli_summary.json",
            "present": True,
            "schema": LLM_INFERENCE_CLI_SCHEMA,
        },
    }
    summary = {
        "schema": LLM_INFERENCE_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "port": args.port,
        "request_count": args.request_count,
        "step": step,
        "evidence_schema": payload.get("schema") or "external_llm_evidence_v1",
        "adapter": {
            "kind": adapter.get("kind"),
            "model_id": adapter.get("model_id"),
            "operator_owned_runtime": adapter.get("operator_owned_runtime"),
        },
        "inference": {
            "request_count": llm_summary.get("request_count"),
            "completion_count": llm_summary.get("completion_count"),
            "output_chars": llm_summary.get("output_chars"),
            "requests_per_second": llm_summary.get("requests_per_second"),
        },
        "diagnosis_codes": diagnosis_codes(payload),
        "artifacts": artifacts,
        "safety": {
            "captured_output_redacted": True,
            "summary_excludes_raw_external_llm_payloads": True,
            "runtime_url_redacted": True,
            "api_credential_redacted": True,
            "read_only_workload": "external_llm_infer",
            "not_production": True,
        },
        "limitations": [
            "Local external_llm_infer proof; not production LLM serving",
            "Uses fixed claim-time prompts; not an arbitrary public prompt API",
            "Does not provide GPU pooling, WebGPU shards, P2P routing, or incentives",
        ],
        "recommended_next_commands": [
            "python3 scripts/external_llm_evidence_check.py --port 8919",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
        ],
    }
    summary = sanitize(redact_values(summary, secret_values))
    encoded = json.dumps(summary, sort_keys=True)
    if any(secret and secret in encoded for secret in secret_values):
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_cpu_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "beta-rc":
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "cpu_inference_beta_rc_pack.py"),
            "--output-dir",
            str(output_dir),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--scenario-id",
            args.scenario_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--json",
        ]
        if args.kaggle_real_runtime_report:
            command.extend(["--kaggle-real-runtime-report", args.kaggle_real_runtime_report])
        step, payload = run_json_step(
            "cpu_inference_beta_rc",
            command,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds * 3,
            redact_secrets=[],
        )
        if payload:
            payload = sanitize(payload)
            payload.setdefault("cli_schema", CPU_INFERENCE_BETA_RC_CLI_SCHEMA)
            return payload
        return sanitize({
            "schema": "cpu_inference_beta_rc_v1",
            "cli_schema": CPU_INFERENCE_BETA_RC_CLI_SCHEMA,
            "ok": False,
            "mode": args.mode,
            "output_dir": str(output_dir),
            "step": step,
            "diagnosis_codes": ["beta_rc_blocked"],
            "limitations": [
                "CPU Inference Beta RC evidence only; not production Swarm Inference",
                "Does not provide GPU/TPU workloads, P2P routing, NAT traversal, or arbitrary public prompt serving",
            ],
        })
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "cpu_inference_beta_pack.py"),
        "--mode",
        args.mode,
        "--workload",
        args.workload,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--json",
    ]
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.miner_id:
        command.extend(["--miner-id", args.miner_id])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.mock:
        command.append("--mock")
    if args.llm_runtime_cmd:
        command.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    if args.llm_runtime_url:
        command.extend(["--llm-runtime-url", args.llm_runtime_url])
    if args.llm_runtime_api_key:
        command.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    if args.llm_runtime_model_id:
        command.extend(["--llm-runtime-model-id", args.llm_runtime_model_id])
    if args.llm_runtime_timeout:
        command.extend(["--llm-runtime-timeout", str(args.llm_runtime_timeout)])
    secret_values = [args.observer_token, args.admin_token, args.llm_runtime_url, args.llm_runtime_api_key]
    step, payload = run_json_step(
        "cpu_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = redact_values(payload, secret_values)
        payload.setdefault("cli_schema", CPU_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "cpu_inference_beta_v1",
        "cli_schema": CPU_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["cpu_inference_beta_failed"],
        "limitations": [
            "CPU-only Beta inference proof; not production Swarm Inference",
            "Does not provide GPU pooling, WebGPU shards, P2P routing, NAT traversal, or arbitrary public prompt serving",
        ],
    })


def build_sharded_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "sharded_inference_evidence.json"
    evidence_md = output_dir / "sharded_inference_evidence.md"
    summary_json = output_dir / "sharded_inference_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "sharded_inference_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    step, payload = run_json_step(
        "sharded_inference_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    payload = sanitize(payload) if payload else {}
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    summary = {
        "schema": SHARDED_INFERENCE_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "evidence_schema": payload.get("schema") or "sharded_inference_evidence_v1",
        "failure_mode": args.failure_mode,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "request_count": args.request_count,
        "scenario_id": args.scenario_id,
        "step": step,
        "diagnosis_codes": diagnosis_codes(payload),
        "session": payload.get("session") if isinstance(payload.get("session"), dict) else {},
        "stage_summary": payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {},
        "safety": payload.get("safety") if isinstance(payload.get("safety"), dict) else {},
        "artifacts": {
            "sharded_inference_evidence_json": artifact_entry(
                evidence_json,
                output_dir,
                kind="sharded_inference_evidence",
                schema=str(payload.get("schema") or "sharded_inference_evidence_v1"),
                ok=payload.get("ok") if payload else None,
            ),
            "sharded_inference_evidence_markdown": artifact_entry(
                evidence_md,
                output_dir,
                kind="sharded_inference_evidence_markdown",
            ),
            "sharded_inference_cli_summary": artifact_entry(
                summary_json,
                output_dir,
                kind="sharded_inference_cli_summary",
                schema=SHARDED_INFERENCE_CLI_SCHEMA,
                ok=bool(step.get("ok")),
            ),
        },
        "limitations": [
            "CPU-only fixed two-stage pipeline; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, real LLM sharding, or arbitrary prompt serving",
        ],
    }
    summary = sanitize(summary)
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
        summary["safety_error"] = "sharded inference summary contained secret-like fragments"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"]["sharded_inference_cli_summary"]["present"] = True
    return summary


def build_micro_llm_sharded_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "micro_llm_sharded_evidence.json"
    evidence_md = output_dir / "micro_llm_sharded_evidence.md"
    summary_json = output_dir / "micro_llm_sharded_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "micro_llm_sharded_inference_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if getattr(args, "micro_llm_artifact", ""):
        command.extend(["--micro-llm-artifact", args.micro_llm_artifact])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    step, payload = run_json_step(
        "micro_llm_sharded_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    payload = sanitize(payload) if payload else {}
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    summary = {
        "schema": MICRO_LLM_SHARDED_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "evidence_schema": payload.get("schema") or "micro_llm_sharded_evidence_v1",
        "failure_mode": args.failure_mode,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "request_count": args.request_count,
        "decode_steps": args.decode_steps,
        "micro_llm_artifact": getattr(args, "micro_llm_artifact", ""),
        "step": step,
        "diagnosis_codes": diagnosis_codes(payload),
        "session": payload.get("session") if isinstance(payload.get("session"), dict) else {},
        "stage_summary": payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {},
        "safety": payload.get("safety") if isinstance(payload.get("safety"), dict) else {},
        "artifacts": {
            "micro_llm_sharded_evidence_json": artifact_entry(
                evidence_json,
                output_dir,
                kind="micro_llm_sharded_evidence",
                schema=str(payload.get("schema") or "micro_llm_sharded_evidence_v1"),
                ok=payload.get("ok") if payload else None,
            ),
            "micro_llm_sharded_evidence_markdown": artifact_entry(
                evidence_md,
                output_dir,
                kind="micro_llm_sharded_evidence_markdown",
            ),
            "micro_llm_sharded_cli_summary": artifact_entry(
                summary_json,
                output_dir,
                kind="micro_llm_sharded_cli_summary",
                schema=MICRO_LLM_SHARDED_CLI_SCHEMA,
                ok=bool(step.get("ok")),
            ),
        },
        "limitations": [
            "CPU-only deterministic micro-LLM two-stage pipeline; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    }
    summary = sanitize(summary)
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
        summary["safety_error"] = "micro-LLM sharded inference summary contained secret-like fragments"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"]["micro_llm_sharded_cli_summary"]["present"] = True
    return summary


def build_real_llm_sharded_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "real_llm_sharded_evidence.json"
    evidence_md = output_dir / "real_llm_sharded_evidence.md"
    summary_json = output_dir / "real_llm_sharded_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_llm_sharded_inference_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--hf-model-id",
        str(args.hf_model_id),
        "--real-llm-partition-mode",
        str(getattr(args, "real_llm_partition_mode", "full")),
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    step, payload = run_json_step(
        "real_llm_sharded_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    payload = sanitize(payload) if payload else {}
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    summary = {
        "schema": REAL_LLM_SHARDED_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "evidence_schema": payload.get("schema") or "real_llm_sharded_evidence_v1",
        "failure_mode": args.failure_mode,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "request_count": args.request_count,
        "max_new_tokens": getattr(args, "max_new_tokens", 1),
        "hf_model_id": args.hf_model_id,
        "real_llm_partition_mode": getattr(args, "real_llm_partition_mode", "full"),
        "step": step,
        "diagnosis_codes": diagnosis_codes(payload),
        "session": payload.get("session") if isinstance(payload.get("session"), dict) else {},
        "artifact": payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {},
        "stage_summary": payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {},
        "safety": payload.get("safety") if isinstance(payload.get("safety"), dict) else {},
        "artifacts": {
            "real_llm_sharded_evidence_json": artifact_entry(
                evidence_json,
                output_dir,
                kind="real_llm_sharded_evidence",
                schema=str(payload.get("schema") or "real_llm_sharded_evidence_v1"),
                ok=payload.get("ok") if payload else None,
            ),
            "real_llm_sharded_evidence_markdown": artifact_entry(
                evidence_md,
                output_dir,
                kind="real_llm_sharded_evidence_markdown",
            ),
            "real_llm_sharded_cli_summary": artifact_entry(
                summary_json,
                output_dir,
                kind="real_llm_sharded_cli_summary",
                schema=REAL_LLM_SHARDED_CLI_SCHEMA,
                ok=bool(step.get("ok")),
            ),
        },
        "limitations": [
            "CPU-only tiny Hugging Face GPT two-stage pipeline; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    }
    summary = sanitize(summary)
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
        summary["safety_error"] = "real LLM sharded inference summary contained secret-like fragments"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"]["real_llm_sharded_cli_summary"]["present"] = True
    return summary


def build_micro_llm_artifact(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "micro_llm_artifact_cli_summary.json"
    artifact_json = output_dir / "micro_llm_artifact.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "micro_llm_artifact_pack.py"),
        "--output-dir",
        str(output_dir),
        "--artifact-id",
        args.artifact_id,
        "--version",
        str(args.version),
        "--json-out",
        str(artifact_json),
    ]
    if args.inspect:
        command.append("--inspect")
    step, payload = run_json_step(
        "micro_llm_artifact",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    payload = sanitize(payload) if payload else {}
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    summary = sanitize({
        "schema": MICRO_LLM_ARTIFACT_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "artifact_schema": payload.get("schema") or "micro_llm_artifact_v1",
        "artifact_id": payload.get("artifact_id"),
        "artifact_hash": payload.get("artifact_hash"),
        "artifact_version": payload.get("artifact_version"),
        "manifest_path": payload.get("manifest_path"),
        "step": step,
        "diagnosis_codes": ["micro_llm_artifact_ready"] if step.get("ok") else ["micro_llm_artifact_failed"],
        "artifacts": {
            "micro_llm_artifact_manifest": artifact_entry(output_dir / "manifest.json", output_dir, kind="micro_llm_artifact_manifest", schema="micro_llm_artifact_v1", ok=bool(step.get("ok"))),
            "micro_llm_artifact_json": artifact_entry(artifact_json, output_dir, kind="micro_llm_artifact", schema="micro_llm_artifact_v1", ok=payload.get("ok") if payload else None),
            "micro_llm_artifact_cli_summary": artifact_entry(summary_json, output_dir, kind="micro_llm_artifact_cli_summary", schema=MICRO_LLM_ARTIFACT_CLI_SCHEMA, ok=bool(step.get("ok"))),
        },
        "limitations": [
            "Dependency-free tiny Micro-LLM artifact; not a HF, GGUF, llama.cpp, or large-model artifact",
            "CPU-only/read-only proof boundary when used with micro-llm-shard-infer",
        ],
    })
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"]["micro_llm_artifact_cli_summary"]["present"] = True
    return summary


def build_remote_sharded_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_sharded_inference_beta_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
    ]
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    command.append("--json")
    secret_values = [args.observer_token, args.admin_token]
    step, payload = run_json_step(
        "remote_sharded_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REMOTE_SHARDED_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "remote_sharded_inference_beta_v1",
        "cli_schema": REMOTE_SHARDED_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["remote_sharded_inference_failed"],
        "limitations": [
            "CPU-only two-stage pipeline-sharded inference Beta; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, real LLM sharding, or arbitrary prompt serving",
        ],
    })


def build_remote_micro_llm_sharded_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_micro_llm_sharded_beta_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
    ]
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    command.append("--json")
    secret_values = [args.observer_token, args.admin_token]
    step, payload = run_json_step(
        "remote_micro_llm_sharded_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REMOTE_MICRO_LLM_SHARDED_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "remote_micro_llm_sharded_beta_v1",
        "cli_schema": REMOTE_MICRO_LLM_SHARDED_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["remote_micro_llm_sharded_failed"],
        "limitations": [
            "CPU-only deterministic micro-LLM pipeline-sharded inference Beta; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    })


def build_remote_real_llm_sharded_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        str(args.hf_model_id),
        "--real-llm-partition-mode",
        str(getattr(args, "real_llm_partition_mode", "full")),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    command.append("--json")
    secret_values = [args.observer_token, args.admin_token]
    step, payload = run_json_step(
        "remote_real_llm_sharded_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REMOTE_REAL_LLM_SHARDED_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "remote_real_llm_sharded_beta_v1",
        "cli_schema": REMOTE_REAL_LLM_SHARDED_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["remote_real_llm_sharded_failed"],
        "limitations": [
            "CPU-only tiny Hugging Face GPT pipeline-sharded inference Beta; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    })


def build_micro_llm_live_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "micro_llm_live_rc_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--port",
        str(args.port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--micro-llm-artifact",
        str(args.micro_llm_artifact),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--artifact-timeout",
        str(args.artifact_timeout),
        "--admin-results-limit",
        str(args.admin_results_limit),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--failure-mode",
        getattr(args, "failure_mode", "none"),
        "--victim-compute-seconds",
        str(getattr(args, "victim_compute_seconds", 45.0)),
        "--claim-observe-timeout",
        str(getattr(args, "claim_observe_timeout", 180.0)),
        "--requeue-timeout",
        str(getattr(args, "requeue_timeout", 120.0)),
        "--json",
    ]
    if args.kaggle_output_dir:
        command.extend(["--kaggle-output-dir", args.kaggle_output_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    secret_values = [args.observer_token, args.admin_token]
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    step, payload = run_json_step(
        "micro_llm_live_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", MICRO_LLM_LIVE_RC_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "micro_llm_live_rc_v1",
        "cli_schema": MICRO_LLM_LIVE_RC_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["micro_llm_live_rc_blocked"],
        "limitations": [
            "CPU-only read-only micro-LLM live two-node RC; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, GGUF/llama.cpp serving, large-model sharding, or arbitrary prompt serving",
        ],
    })


def build_real_llm_live_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_llm_live_rc_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--failure-mode",
        getattr(args, "failure_mode", "none"),
        "--victim-compute-seconds",
        str(getattr(args, "victim_compute_seconds", 45.0)),
        "--claim-observe-timeout",
        str(getattr(args, "claim_observe_timeout", 180.0)),
        "--requeue-timeout",
        str(getattr(args, "requeue_timeout", 120.0)),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    secret_values = [args.observer_token, args.admin_token]
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    step, payload = run_json_step(
        "real_llm_live_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REAL_LLM_LIVE_RC_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "real_llm_live_rc_v1",
        "cli_schema": REAL_LLM_LIVE_RC_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["real_llm_live_rc_blocked"],
        "limitations": [
            "CPU-only read-only real small-LLM live two-node RC; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving",
        ],
    })


def build_real_llm_internet_alpha(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_llm_internet_alpha_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    secret_values = [args.observer_token, args.admin_token]
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.skip_requeue:
        command.append("--skip-requeue")
    step, payload = run_json_step(
        "real_llm_internet_alpha",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), 240.0) * 3 + 120.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REAL_LLM_INTERNET_ALPHA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "real_llm_internet_alpha_v1",
        "cli_schema": REAL_LLM_INTERNET_ALPHA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["real_llm_internet_alpha_blocked"],
        "limitations": [
            "CPU-only read-only real small-LLM Internet Alpha; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving",
        ],
    })


def build_real_llm_internet_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_llm_internet_beta_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        args.hf_model_id,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--kaggle-push-timeout-seconds",
        str(args.kaggle_push_timeout_seconds),
        "--kaggle-delete-timeout-seconds",
        str(args.kaggle_delete_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.ready_url:
        command.extend(["--ready-url", args.ready_url])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.kaggle_owner:
        command.extend(["--kaggle-owner", args.kaggle_owner])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if not args.inline_kernel_payload:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    step, payload = run_json_step(
        "real_llm_internet_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 300.0) + 600.0,
    )
    if payload:
        payload = sanitize(redact_values(payload, []))
        payload.setdefault("cli_schema", REAL_LLM_INTERNET_BETA_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "real_llm_internet_beta_v1",
        "cli_schema": REAL_LLM_INTERNET_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["real_llm_internet_beta_blocked"],
        "limitations": [
            "CPU-only read-only real small-LLM Internet Beta; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving",
        ],
    })


def build_swarm_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "swarm_inference_beta_pack.py"),
        args.swarm_action,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    if args.swarm_action == "live":
        command.extend([
            "--public-host",
            args.public_host,
            "--bind-host",
            args.bind_host,
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--request-count",
            str(args.request_count),
            "--hf-model-id",
            args.hf_model_id,
            "--dataset-title",
            args.dataset_title,
            "--kernel-title-prefix",
            args.kernel_title_prefix,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--startup-timeout",
            str(args.startup_timeout),
            "--process-exit-timeout",
            str(args.process_exit_timeout),
            "--poll-interval",
            str(args.poll_interval),
            "--http-timeout",
            str(args.http_timeout),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-status-timeout-seconds",
            str(args.kaggle_status_timeout_seconds),
            "--kaggle-status-poll-interval",
            str(args.kaggle_status_poll_interval),
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--idle-sleep",
            str(args.idle_sleep),
            "--max-request-attempts",
            str(args.max_request_attempts),
            "--failure-mode",
            getattr(args, "failure_mode", "none"),
            "--victim-compute-seconds",
            str(getattr(args, "victim_compute_seconds", 45.0)),
            "--claim-observe-timeout",
            str(getattr(args, "claim_observe_timeout", 180.0)),
            "--requeue-timeout",
            str(getattr(args, "requeue_timeout", 120.0)),
        ])
        if args.ready_url:
            command.extend(["--ready-url", args.ready_url])
        if args.coordinator_url:
            command.extend(["--coordinator-url", args.coordinator_url])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        if args.kaggle_owner:
            command.extend(["--kaggle-owner", args.kaggle_owner])
        if args.dataset_slug:
            command.extend(["--dataset-slug", args.dataset_slug])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        if args.inline_kernel_payload:
            command.append("--inline-kernel-payload")
        else:
            command.append("--no-inline-kernel-payload")
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
        if args.keep_live_private_artifacts:
            command.append("--keep-live-private-artifacts")
    elif args.swarm_action != "clean":
        command.extend([
            "--coordinator-url",
            args.coordinator_url,
            "--port",
            str(args.port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--request-count",
            str(args.request_count),
            "--hf-model-id",
            args.hf_model_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.swarm_action == "prepare":
        command.extend([
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.replace:
            command.append("--replace")
    elif args.swarm_action == "coordinator":
        command.extend(["--lease-seconds", str(args.lease_seconds)])
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.run:
            command.append("--run")
    elif args.swarm_action == "miner":
        command.extend([
            "--stage",
            args.stage,
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.run:
            command.append("--run")
    elif args.swarm_action == "verify":
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.prompt_texts:
            command.extend(["--prompt-texts", args.prompt_texts])
        if args.real_internet_beta_report:
            command.extend(["--real-internet-beta-report", args.real_internet_beta_report])
    elif args.swarm_action == "collect":
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.miner_id:
            command.extend(["--miner-id", args.miner_id])
        command.extend(["--artifact-timeout", str(args.artifact_timeout)])
    elif args.swarm_action == "clean":
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "swarm_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(getattr(args, "timeout_seconds", 60.0)), float(getattr(args, "remote_timeout_seconds", 60.0)), 60.0) + (600.0 if args.swarm_action == "live" else 0.0),
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", SWARM_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "swarm_inference_beta_v1",
        "cli_schema": SWARM_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.swarm_action,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["swarm_inference_beta_failed"],
        "limitations": [
            "CPU-only read-only real tiny-LLM Swarm Inference Beta; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_inference_alpha(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_alpha_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--request-count",
        str(args.request_count),
        "--hf-model-id",
        args.hf_model_id,
        "--failure-mode",
        args.failure_mode,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--kaggle-push-timeout-seconds",
        str(args.kaggle_push_timeout_seconds),
        "--kaggle-delete-timeout-seconds",
        str(args.kaggle_delete_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--victim-compute-seconds",
        str(getattr(args, "victim_compute_seconds", 45.0)),
        "--claim-observe-timeout",
        str(getattr(args, "claim_observe_timeout", 180.0)),
        "--requeue-timeout",
        str(getattr(args, "requeue_timeout", 120.0)),
        "--json",
    ]
    if args.ready_url:
        command.extend(["--ready-url", args.ready_url])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.kaggle_owner:
        command.extend(["--kaggle-owner", args.kaggle_owner])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if args.inline_kernel_payload:
        command.append("--inline-kernel-payload")
    else:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    if args.keep_live_private_artifacts:
        command.append("--keep-live-private-artifacts")
    if args.keep_child_artifacts:
        command.append("--keep-child-artifacts")
    if args.skip_local_requeue:
        command.append("--skip-local-requeue")
    step, payload = run_json_step(
        "public_swarm_inference_alpha",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 300.0) + 1200.0,
    )
    if payload:
        payload = sanitize(redact_values(payload, []))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_ALPHA_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "public_swarm_inference_alpha_v1",
        "cli_schema": PUBLIC_SWARM_INFERENCE_ALPHA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_alpha_failed"],
        "limitations": [
            "CPU-only read-only real tiny-LLM Public Swarm Inference Alpha; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_inference_alpha_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_alpha_rc_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--stage0-report",
        args.stage0_report,
        "--stage1-report",
        args.stage1_report,
        "--summary-report",
        args.summary_report,
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    step, payload = run_json_step(
        "public_swarm_inference_alpha_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=float(args.timeout_seconds) + 60.0,
    )
    if payload:
        payload = sanitize(redact_values(payload, []))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_ALPHA_RC_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "public_swarm_inference_alpha_rc_v1",
        "cli_schema": PUBLIC_SWARM_INFERENCE_ALPHA_RC_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_alpha_rc_failed"],
        "limitations": [
            "CPU-only read-only Public Swarm Inference Alpha RC evidence; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    action = getattr(args, "public_swarm_beta_action", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_beta_pack.py"),
        action,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if action == "product-beta":
        command.extend([
            "--base-port",
            str(args.base_port),
            "--hf-model-id",
            args.hf_model_id,
            "--gpu-report",
            args.gpu_report,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--cpu-request-count",
            str(args.cpu_request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--scenario-id",
            args.scenario_id,
            "--cpu-timeout-seconds",
            str(args.cpu_timeout_seconds),
        ])
    if action in {"prepare", "coordinator", "miner", "verify", "collect", "local-loopback"}:
        command.extend([
            "--coordinator-url",
            args.coordinator_url,
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--hf-model-id",
            args.hf_model_id,
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if action in {"prepare", "coordinator", "miner"}:
        command.extend(["--lease-seconds", str(args.lease_seconds)])
    if action in {"prepare", "miner"}:
        command.extend([
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
    if action in {"prepare", "coordinator", "verify", "collect"}:
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
    if action == "prepare" and args.replace:
        command.append("--replace")
    if action == "coordinator" and args.run:
        command.append("--run")
    if action == "miner":
        command.extend(["--stage", args.stage])
        if args.run:
            command.append("--run")
    if action in {"verify", "local-loopback"} and args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if action == "verify" and args.real_internet_beta_report:
        command.extend(["--real-internet-beta-report", args.real_internet_beta_report])
    if action == "collect":
        if args.miner_id:
            command.extend(["--miner-id", args.miner_id])
        command.extend(["--artifact-timeout", str(args.artifact_timeout)])
    if action == "clean":
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
    if action == "evidence-import":
        command.extend([
            "--alpha-rc-report",
            args.alpha_rc_report,
            "--stage0-report",
            args.stage0_report,
            "--stage1-report",
            args.stage1_report,
            "--summary-report",
            args.summary_report,
        ])
        if args.allow_missing_live_evidence:
            command.append("--allow-missing-live-evidence")
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(getattr(args, "remote_timeout_seconds", 60.0)), 60.0) + 180.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_inference_beta_v1",
        "cli_schema": PUBLIC_SWARM_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": action,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_beta_failed"],
        "limitations": [
            "CPU-only read-only Public Swarm Inference Beta; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_inference_beta_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "public_swarm_beta_rc_mode", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_beta_rc_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--prompt-text",
        args.prompt_text,
        "--scenario-id",
        args.scenario_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if getattr(args, "stream_generation", False):
        command.append("--stream-generation")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_inference_beta_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 300.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_BETA_RC_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_inference_beta_rc_v1",
        "cli_schema": PUBLIC_SWARM_INFERENCE_BETA_RC_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_beta_rc_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm Inference Beta RC; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_product_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "public_swarm_product_beta_mode", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_product_beta_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--prompt-text",
        args.prompt_text,
        "--scenario-id",
        args.scenario_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if getattr(args, "stream_generation", False):
        command.append("--stream-generation")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_product_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 360.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_PRODUCT_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_product_beta_v1",
        "cli_schema": PUBLIC_SWARM_PRODUCT_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_product_beta_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm Product Beta; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    })


def build_public_real_llm_swarm_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "public_real_llm_swarm_beta_mode", "release")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_real_llm_swarm_beta_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--product-report",
        args.product_report,
        "--external-report",
        args.external_report,
        "--p2p-report",
        args.p2p_report,
        "--usable-report",
        args.usable_report,
        "--public-swarm-v2-report",
        args.public_swarm_v2_report,
        "--public-swarm-v2-preview-report",
        args.public_swarm_v2_preview_report,
        "--public-swarm-v2-real-p2p-report",
        args.public_swarm_v2_real_p2p_report,
        "--p2p-runtime-smoke-report",
        args.p2p_runtime_smoke_report,
        "--p2p-external-report",
        args.p2p_external_report,
        "--p2p-requeue-report",
        args.p2p_requeue_report,
        "--p2p-batch-stream-report",
        args.p2p_batch_stream_report,
        "--gpu-report",
        args.gpu_report,
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--p2p-port",
        str(args.p2p_port),
        "--p2p-libp2p-port",
        str(args.p2p_libp2p_port),
        "--public-swarm-v2-p2p-port",
        str(args.public_swarm_v2_p2p_port),
        "--public-swarm-v2-coordinator-port",
        str(args.public_swarm_v2_coordinator_port),
        "--public-swarm-v2-real-p2p-port",
        str(args.public_swarm_v2_real_p2p_port),
        "--public-swarm-v2-real-p2p-coordinator-port",
        str(args.public_swarm_v2_real_p2p_coordinator_port),
        "--public-swarm-v2-real-p2p-libp2p-port",
        str(args.public_swarm_v2_real_p2p_libp2p_port),
        "--public-swarm-v2-real-p2p-discovery-backend",
        args.public_swarm_v2_real_p2p_discovery_backend,
        "--public-swarm-v2-backend",
        args.public_swarm_v2_backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--public-swarm-v2-timeout-seconds",
        str(args.public_swarm_v2_timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if getattr(args, "stream_generation", False):
        command.append("--stream-generation")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    step, payload = run_json_step(
        "public_real_llm_swarm_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 600.0,
    )
    if payload:
        payload = sanitize(redact_values(payload))
        payload.setdefault("cli_schema", PUBLIC_REAL_LLM_SWARM_BETA_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "public_real_llm_swarm_beta_v1",
        "cli_schema": PUBLIC_REAL_LLM_SWARM_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_real_llm_swarm_beta_failed"],
        "limitations": [
            "Coordinator-backed Public Real-LLM Swarm Inference Beta; not production Hivemind/Petals parity",
            "Does not provide Coordinator-free P2P execution, production NAT traversal, economics, anti-Sybil security, or large-model serving",
        ],
    })


def build_usable_swarm_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "usable_swarm_mode", "local")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "usable_swarm_inference_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--p2p-report",
        args.p2p_report,
        "--swarm-id",
        args.swarm_id,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--preview-v04-report",
        args.preview_v04_report,
        "--product-mvp-report",
        args.product_mvp_report,
        "--optional-model-report",
        args.optional_model_report,
        "--json",
    ]
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if getattr(args, "stream_generation", False):
        command.append("--stream-generation")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    step, payload = run_json_step(
        "usable_swarm_inference",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.startup_timeout), 60.0) + 1200.0,
    )
    if payload:
        payload = sanitize(redact_values(payload))
        payload.setdefault("cli_schema", USABLE_SWARM_INFERENCE_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "usable_swarm_inference_v1",
        "cli_schema": USABLE_SWARM_INFERENCE_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["usable_swarm_inference_failed"],
        "limitations": [
            "Coordinator-backed user-facing P2P discovery path for tiny/small real-LLM split inference",
            "Does not provide full Hivemind/Petals production parity, Coordinator-free execution, production NAT traversal, an economic system, or large-model throughput",
        ],
    })


def build_public_swarm_inference_v2(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "public_swarm_v2_mode", "local")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_v2_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--usable-report",
        args.usable_report,
        "--preview-report",
        args.preview_report,
        "--real-p2p-report",
        args.real_p2p_report,
        "--gpu-report",
        args.gpu_report,
        "--fresh-external-attempt-report",
        args.fresh_external_attempt_report,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--real-p2p-port",
        str(args.real_p2p_port),
        "--real-p2p-coordinator-port",
        str(args.real_p2p_coordinator_port),
        "--real-p2p-libp2p-port",
        str(args.real_p2p_libp2p_port),
        "--real-p2p-discovery-backend",
        args.real_p2p_discovery_backend,
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if getattr(args, "stream_generation", False):
        command.append("--stream-generation")
    if getattr(args, "fresh_external_report", False):
        command.append("--fresh-external-report")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    step, payload = run_json_step(
        "public_swarm_inference_v2",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.startup_timeout), 60.0) + 1800.0,
    )
    if payload:
        payload = sanitize(redact_values(payload))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_V2_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "public_swarm_inference_v2",
        "cli_schema": PUBLIC_SWARM_INFERENCE_V2_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_v2_failed"],
        "limitations": [
            "Coordinator-backed public preview; not full Hivemind/Petals production parity",
            "Does not provide Coordinator-free P2P execution, production NAT traversal, economics, anti-Sybil security, or large-model serving",
        ],
    })


def _safe_nested_get(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _infer_generation_from_report(payload: dict[str, Any]) -> dict[str, Any]:
    generation = _safe_nested_get(payload, "readiness", "local_p2p_generate", "generation")
    if generation:
        return generation
    generation = _safe_nested_get(payload, "readiness", "p2p_product_path", "generation")
    if generation:
        return generation
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    return generation if isinstance(generation, dict) else {}


def _infer_route_from_report(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") == "product_swarm_mvp_check_v1":
        stage = payload.get("stage_assignment") if isinstance(payload.get("stage_assignment"), dict) else {}
        ledger = payload.get("ledger") if isinstance(payload.get("ledger"), dict) else {}
        return {
            "route_source": "local-product-loopback",
            "route_ready": bool(payload.get("ok")),
            "distinct_stage_miners": bool(stage.get("distinct_stage_miners")),
            "stage_assignment_valid": bool(stage.get("distinct_stage_miners") or "stage_assignment_valid" in (payload.get("diagnosis_codes") or [])),
            "accepted_rows": ledger.get("accepted_rows"),
        }
    local = _safe_nested_get(payload, "readiness", "local_p2p_generate")
    if local:
        return {
            "route_source": local.get("route_source"),
            "route_ready": bool(local.get("route_ready")),
            "distinct_stage_miners": bool(local.get("distinct_stage_miners")),
            "stage_assignment_valid": bool(local.get("stage_assignment", {}).get("stage_assignment_valid"))
            if isinstance(local.get("stage_assignment"), dict)
            else bool(local.get("distinct_stage_miners")),
            "accepted_rows": local.get("accepted_rows"),
        }
    p2p = _safe_nested_get(payload, "readiness", "p2p_product_path")
    if p2p:
        return {
            "route_source": p2p.get("route_source"),
            "route_ready": bool(p2p.get("route_ready")),
            "distinct_stage_miners": bool(p2p.get("distinct_stage_miners")),
            "stage_assignment_valid": bool(p2p.get("stage_assignment", {}).get("stage_assignment_valid"))
            if isinstance(p2p.get("stage_assignment"), dict)
            else bool(p2p.get("distinct_stage_miners")),
            "accepted_rows": p2p.get("accepted_rows"),
        }
    route = payload.get("route") if isinstance(payload.get("route"), dict) else {}
    return {
        "route_source": route.get("route_source"),
        "route_ready": bool(route.get("usable_now") or route.get("coordinator_url_present")),
        "coordinator_url": route.get("coordinator_url"),
        "missing_capabilities": route.get("missing_capabilities") if isinstance(route.get("missing_capabilities"), list) else [],
        "distinct_stage_miners": False,
        "stage_assignment_valid": False,
        "accepted_rows": None,
    }


def _infer_model_from_report(payload: dict[str, Any], default_model_id: str) -> str:
    local_model = _safe_nested_get(payload, "readiness", "local_p2p_generate", "model")
    for source in [
        payload.get("hf_model_id"),
        _safe_nested_get(payload, "public_swarm_v2").get("hf_model_id"),
        _safe_nested_get(payload, "usable_swarm").get("hf_model_id"),
        _safe_nested_get(payload, "session").get("hf_model_id"),
        _safe_nested_get(payload, "session").get("model_id"),
        local_model.get("observed_hf_model_id"),
        local_model.get("expected_hf_model_id"),
    ]:
        if isinstance(source, str) and source:
            return source
    return default_model_id


def _infer_stream_from_report(payload: dict[str, Any]) -> dict[str, Any]:
    stream = _safe_nested_get(payload, "readiness", "local_p2p_generate", "stream")
    if stream:
        return stream
    stream = _safe_nested_get(payload, "readiness", "p2p_product_path", "stream")
    if stream:
        return stream
    stream = payload.get("stream") if isinstance(payload.get("stream"), dict) else {}
    return stream if isinstance(stream, dict) else {}


def _infer_batch_from_report(payload: dict[str, Any]) -> dict[str, Any]:
    batch = _safe_nested_get(payload, "readiness", "local_p2p_generate", "batch")
    if batch:
        return batch
    batch = _safe_nested_get(payload, "readiness", "p2p_product_path", "batch")
    if batch:
        return batch
    batch = payload.get("batch") if isinstance(payload.get("batch"), dict) else {}
    return batch if isinstance(batch, dict) else {}


def _safe_infer_stream_progress(stream: dict[str, Any]) -> dict[str, Any]:
    progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
    per_request_progress = []
    raw_per_request = progress.get("per_request_progress") if isinstance(progress.get("per_request_progress"), list) else []
    for item in raw_per_request:
        if not isinstance(item, dict):
            continue
        counts = item.get("observed_token_counts") if isinstance(item.get("observed_token_counts"), list) else []
        per_request_progress.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "event_count": _safe_int(item.get("event_count")),
            "observed_token_counts": [_safe_int(count) for count in counts],
            "max_observed_token_count": _safe_int(item.get("max_observed_token_count")),
            "target_token_count": _safe_int(item.get("target_token_count")),
            "monotonic_progress": bool(item.get("monotonic_progress")),
            "stream_progress_complete": bool(item.get("stream_progress_complete")),
        })
    counts = progress.get("observed_token_counts") if isinstance(progress.get("observed_token_counts"), list) else []
    return {
        "stream_progress_complete": bool(progress.get("stream_progress_complete")),
        "all_token_events_ready": bool(progress.get("all_token_events_ready")),
        "monotonic_progress": bool(progress.get("monotonic_progress")),
        "observed_token_counts": [_safe_int(count) for count in counts],
        "max_observed_token_count": _safe_int(progress.get("max_observed_token_count")),
        "target_token_count": _safe_int(progress.get("target_token_count") or progress.get("max_new_tokens")),
        "expected_request_count": _safe_int(progress.get("expected_request_count"), 1),
        "per_request_progress": per_request_progress,
        "per_request_progress_complete": bool(progress.get("per_request_progress_complete")),
        "per_request_monotonic_progress": bool(progress.get("per_request_monotonic_progress")),
    }


def _safe_infer_stream_events(stream: dict[str, Any]) -> list[dict[str, Any]]:
    events = stream.get("events") if isinstance(stream.get("events"), list) else []
    safe_events = []
    for item in events:
        if not isinstance(item, dict):
            continue
        safe_events.append({
            "schema": item.get("schema"),
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "generated_token_count": _safe_int(item.get("generated_token_count")),
            "max_new_tokens": item.get("max_new_tokens"),
            "generation_step": item.get("generation_step"),
            "generated_text_hash": item.get("generated_text_hash"),
            "decoded_tokens_match": item.get("decoded_tokens_match"),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        })
    return safe_events


def _iter_infer_generated_text_candidates(value: Any, *, _seen: set[int] | None = None) -> list[dict[str, Any]]:
    seen = _seen if _seen is not None else set()
    candidates: list[dict[str, Any]] = []
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return candidates
        seen.add(marker)
        if isinstance(value.get("generated_text"), str) and value.get("generated_text"):
            candidates.append(value)
        for item in value.values():
            candidates.extend(_iter_infer_generated_text_candidates(item, _seen=seen))
    elif isinstance(value, list):
        for item in value:
            candidates.extend(_iter_infer_generated_text_candidates(item, _seen=seen))
    return candidates


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_local_output_from_private_state(
    output_dir: Path,
    *,
    prompts: list[str],
    generation: dict[str, Any],
    max_chars: int = 4096,
) -> dict[str, Any]:
    """Extract local display-only text from private task state.

    The public evidence packs intentionally redact generated text.  For the
    human CLI path we can recover it from the local private run directory, but
    the returned value must never be persisted into shareable artifacts.
    """

    expected_hash = str(generation.get("generated_text_hash") or "")
    expected_tokens = _safe_int(generation.get("generated_token_count"))
    prompt_hashes = [stable_hash_text(prompt) for prompt in prompts if prompt]
    prompt_hash_set = set(prompt_hashes)
    best_by_prompt: dict[str, tuple[int, dict[str, Any]]] = {}
    fallback: tuple[int, dict[str, Any]] | None = None
    for path in sorted(output_dir.rglob("tasks.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for candidate in _iter_infer_generated_text_candidates(payload):
                text = candidate.get("generated_text")
                if not isinstance(text, str) or not text:
                    continue
                candidate_prompt_hash = str(candidate.get("prompt_hash") or "")
                if prompt_hash_set and candidate_prompt_hash and candidate_prompt_hash not in prompt_hash_set:
                    continue
                candidate_tokens = _safe_int(candidate.get("generated_token_count"))
                if expected_tokens and candidate_tokens and candidate_tokens < expected_tokens:
                    continue
                candidate_hash = str(candidate.get("generated_text_hash") or stable_hash_text(text))
                if expected_hash and candidate_hash != expected_hash and candidate_tokens < expected_tokens:
                    continue
                score = 0
                if expected_hash and candidate_hash == expected_hash:
                    score += 1000
                if expected_tokens and candidate_tokens >= expected_tokens:
                    score += 100
                score += min(candidate_tokens, 64)
                if candidate.get("decoded_tokens_match") is True or candidate.get("baseline_match") is True:
                    score += 10
                if _safe_int(candidate.get("stage_id"), -1) == 1 or _safe_int(candidate.get("split_index"), -1) == 1:
                    score += 5
                output = {
                    "request_id": candidate.get("request_id"),
                    "prompt_hash": candidate_prompt_hash,
                    "generated_token_count": candidate_tokens,
                    "generated_text": text[:max_chars],
                }
                key = candidate_prompt_hash or str(candidate.get("request_id") or "")
                if key:
                    current = best_by_prompt.get(key)
                    if current is None or score > current[0]:
                        best_by_prompt[key] = (score, output)
                if fallback is None or score > fallback[0]:
                    fallback = (score, output)
    ordered_outputs: list[dict[str, Any]] = []
    for prompt_hash in prompt_hashes:
        if prompt_hash in best_by_prompt:
            ordered_outputs.append(best_by_prompt[prompt_hash][1])
    if not ordered_outputs and fallback is not None:
        ordered_outputs.append(fallback[1])
    if not ordered_outputs:
        return {}
    return {
        "source": "local-private-task-state",
        "outputs": ordered_outputs,
        "generated_text": str(ordered_outputs[0].get("generated_text") or ""),
        "output_count": len(ordered_outputs),
        "display_only": True,
        "public_artifact_safe": False,
        "note": "Shown only in local human output; JSON and saved artifacts keep raw generated text redacted.",
    }


def _strip_infer_local_output_text(summary: dict[str, Any]) -> dict[str, Any]:
    local_output = summary.get("local_output") if isinstance(summary.get("local_output"), dict) else {}
    if not local_output:
        return summary
    local_output["available"] = False
    local_output["generated_text"] = ""
    local_output["display_only"] = False
    outputs = local_output.get("outputs")
    if isinstance(outputs, list):
        for output in outputs:
            if isinstance(output, dict):
                output["generated_text"] = ""
    return summary


def _infer_wait_progress_action(payload: dict[str, Any]) -> str:
    progress = payload.get("wait_progress") if isinstance(payload.get("wait_progress"), dict) else {}
    if not progress:
        return "Increase --timeout-seconds and confirm both stage Miners are running."
    if not progress.get("session_created"):
        return "The session was not created; check --admin-token and Coordinator /admin/inference-sessions."
    last_error = str(progress.get("last_error_type") or "")
    if last_error:
        return f"Coordinator polling reported {last_error}; check token permissions and Coordinator reachability, then rerun with --dry-run."
    if not progress.get("ledger_endpoint_ready"):
        return "The session was created but /admin/results was not reachable; check --admin-token and Coordinator admin API access."
    observed = _safe_int(progress.get("max_observed_token_count"))
    target = _safe_int(progress.get("target_token_count"))
    accepted = _safe_int(progress.get("accepted_rows_seen"))
    expected_requests = _safe_int(progress.get("expected_request_count"), 1)
    observed_requests = _safe_int(progress.get("observed_request_count"))
    if accepted <= 0:
        return "No accepted result rows appeared; confirm both stage Miners are joined, healthy, and advertising the requested backend."
    if expected_requests > 1 and observed_requests < expected_requests:
        return f"Only {observed_requests}/{expected_requests} batch results appeared; keep both stage Miners running and increase --timeout-seconds."
    if observed > 0 and target > 0 and observed < target:
        return f"Generation reached {observed}/{target} tokens before timeout; increase --timeout-seconds or check slow stage Miner logs."
    if progress.get("stream_endpoint_ready") and _safe_int(progress.get("stream_event_count")) > 0:
        return "Stream progress was visible but the final result did not complete; increase --timeout-seconds and inspect stage Miner logs."
    return "Accepted rows were visible but completion was not; increase --timeout-seconds and inspect Coordinator admin results."


def _ready_to_submit_status(
    *,
    submit_ok: bool | None,
    route_ready: bool,
    coordinator_ok: bool | None,
    coordinator_preflight_required: bool | None,
    stage_preflight_ok: bool | None,
    stage_preflight_required: bool,
    source: str,
) -> dict[str, Any]:
    if stage_preflight_required:
        stage_verification = "ready" if stage_preflight_ok else "failed"
    else:
        stage_verification = "skipped" if route_ready and coordinator_ok is not False else "not_checked"
    warning_codes = []
    if not route_ready:
        warning_codes.append("route_not_ready")
    if coordinator_ok is False:
        warning_codes.append("coordinator_not_ready")
    if stage_verification == "failed":
        warning_codes.append("stage_preflight_failed")
    elif stage_verification == "skipped":
        warning_codes.append("stage_preflight_skipped")
    fully_verified = bool(submit_ok and route_ready and coordinator_ok is not False and stage_verification == "ready")
    if submit_ok is True and fully_verified:
        readiness_label = "verified"
        readiness_summary = "Route, Coordinator, and distinct stage Miners are verified."
    elif submit_ok is True:
        readiness_label = "partial"
        readiness_summary = "Request can be submitted, but stage Miner readiness is not fully verified."
    elif submit_ok is False:
        readiness_label = "blocked"
        readiness_summary = "Request is not ready to submit; follow operator_action and rerun preflight."
    else:
        readiness_label = "skipped"
        readiness_summary = "Request shape is valid, but live readiness was skipped."
    return {
        "ok": submit_ok,
        "fully_verified": fully_verified,
        "readiness_label": readiness_label,
        "readiness_summary": readiness_summary,
        "route_ready": route_ready,
        "coordinator_ready": coordinator_ok,
        "coordinator_preflight_required": coordinator_preflight_required,
        "stage_preflight_ok": stage_preflight_ok,
        "stage_preflight_required": stage_preflight_required,
        "stage_verification": stage_verification,
        "warning_codes": warning_codes,
        "source": source,
        "public_artifact_safe": True,
    }


def _ready_to_submit_action(kind: str, ready_to_submit: dict[str, Any]) -> str:
    if ready_to_submit and ready_to_submit.get("ok") is True and not ready_to_submit.get("fully_verified"):
        warnings = set(str(code) for code in (ready_to_submit.get("warning_codes") or []))
        if "stage_preflight_skipped" in warnings:
            return (
                f"{kind} can be submitted, but stage0/stage1 were not fully verified; "
                "rerun --dry-run with --observer-token to check /state before submitting."
            )
        return f"{kind} can be submitted, but readiness was not fully verified; rerun --dry-run with live preflight before submitting."
    if ready_to_submit and ready_to_submit.get("ok") is None:
        return f"{kind} request shape is valid, but live readiness was skipped; rerun --dry-run without --skip-live-preflight before submitting."
    return ""


def _infer_operator_action(args: argparse.Namespace, payload: dict[str, Any], *, ok: bool) -> str:
    if ok:
        if bool(payload.get("dry_run")):
            ready_to_submit = payload.get("ready_to_submit") if isinstance(payload.get("ready_to_submit"), dict) else {}
            action = _ready_to_submit_action("Inference", ready_to_submit)
            if action:
                return action
            return "Rerun without --dry-run to submit the inference request."
        if getattr(args, "infer_mode", "local") == "local" and not getattr(args, "full_evidence", False):
            return "Run with --full-evidence for the broader Public Swarm v2 proof."
        return ""
    codes = set(str(code) for code in (payload.get("diagnosis_codes") or []))
    step = payload.get("step") if isinstance(payload.get("step"), dict) else {}
    detail = " ".join(
        str(value)
        for value in [payload.get("detail"), payload.get("error"), step.get("stderr_tail"), step.get("error")]
        if value
    )
    if "hf_dependencies_missing" in codes or "product_swarm_mvp_hf_runtime_missing" in codes or "transformers" in detail:
        return "Install optional runtime dependencies with: python -m pip install -e '.[hf]'"
    if "p2p_discovery_unreachable" in codes:
        return "Start the P2P discovery daemon with crowdtensor p2pd --run, or pass --coordinator-url for a direct Coordinator route."
    if "generate_route_unavailable" in codes or "coordinator_route_missing" in codes:
        return "Start a Coordinator and two stage Miners, or pass --coordinator-url/--peer-bootstrap for an existing swarm."
    if "coordinator_ready_failed" in codes:
        return "Coordinator route exists but /ready is not reachable; start the Coordinator or fix --coordinator-url, then rerun --dry-run."
    if "stage_preflight_failed" in codes:
        return "Start or rejoin distinct stage0 and stage1 Miners, then rerun --dry-run with --observer-token to verify /state."
    if "admin_token_required" in codes:
        return "Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN."
    if "generation_timeout" in codes:
        return _infer_wait_progress_action(payload)
    if "crowdtensor_infer_source_report_missing" in codes:
        return "Inspect the child report under the output directory, then rerun with --json for machine-readable diagnostics."
    return "Inspect infer_summary.json and the child report under output_dir for the failing check."


def _infer_command_args(
    args: argparse.Namespace,
    *,
    include_prompt: bool = True,
    mode: str | None = None,
    dry_run: bool | None = None,
    full_evidence: bool | None = None,
    include_admin: bool = False,
    include_observer: bool = False,
    coordinator_url_override: str = "",
) -> list[str]:
    command = ["crowdtensor", "infer"]
    resolved_mode = mode or getattr(args, "infer_mode", "local")
    prompt_texts = str(getattr(args, "prompt_texts", "") or "")
    if include_prompt and not prompt_texts:
        command.append(INFER_PROMPT_PLACEHOLDER)
    command.extend(["--mode", resolved_mode])
    output_dir = str(getattr(args, "output_dir", "") or "")
    if output_dir:
        command.extend(["--output-dir", output_dir])
    if prompt_texts:
        command.extend(["--prompt-texts", INFER_BATCH_PROMPTS_PLACEHOLDER])
    try:
        max_new_tokens = int(getattr(args, "max_new_tokens", 8) or 8)
    except (TypeError, ValueError):
        max_new_tokens = 8
    backend = str(getattr(args, "backend", "cpu") or "cpu")
    if backend != "cpu":
        command.extend(["--backend", backend])
    hf_model_id = str(getattr(args, "hf_model_id", "") or "sshleifer/tiny-gpt2")
    if hf_model_id != "sshleifer/tiny-gpt2":
        command.extend(["--hf-model-id", hf_model_id])
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    if bool(getattr(args, "stream", False)):
        command.append("--stream")
    if bool(getattr(args, "include_output", False)):
        command.append("--include-output")
    use_full_evidence = bool(getattr(args, "full_evidence", False)) if full_evidence is None else bool(full_evidence)
    if resolved_mode == "local" and use_full_evidence:
        max_new_tokens = min(max(max_new_tokens, 8), 32)
    elif resolved_mode == "local":
        max_new_tokens = min(max(max_new_tokens, 2), 8)
    elif resolved_mode == "existing":
        max_new_tokens = min(max(max_new_tokens, 2), 32)
    command.extend(["--max-new-tokens", str(max_new_tokens)])
    if use_full_evidence:
        command.append("--full-evidence")
    use_dry_run = bool(getattr(args, "dry_run", False)) if dry_run is None else bool(dry_run)
    if use_dry_run:
        command.append("--dry-run")
    coordinator_url = coordinator_url_override or str(getattr(args, "coordinator_url", "") or "")
    peer_bootstrap = str(getattr(args, "peer_bootstrap", "") or "")
    p2p = bool(getattr(args, "p2p", False) or (resolved_mode == "existing" and peer_bootstrap))
    if coordinator_url:
        command.extend(["--coordinator-url", coordinator_url])
    if peer_bootstrap:
        command.extend(["--peer-bootstrap", peer_bootstrap])
    if p2p:
        command.append("--p2p")
        swarm_id = str(getattr(args, "swarm_id", "") or "default")
        if swarm_id != "default":
            command.extend(["--swarm-id", swarm_id])
        p2p_backend = str(getattr(args, "p2p_backend", "lite") or "lite")
        if p2p_backend != "lite":
            command.extend(["--p2p-backend", p2p_backend])
    if include_admin and str(getattr(args, "admin_token", "") or ""):
        command.extend(["--admin-token", "<redacted>"])
    if include_observer:
        command.extend(["--observer-token", OBSERVER_TOKEN_ENV_PLACEHOLDER])
    timeout_seconds = float(getattr(args, "timeout_seconds", 420.0) or 420.0)
    if timeout_seconds != 420.0:
        command.extend(["--timeout-seconds", str(timeout_seconds)])
    return command


def _infer_next_commands(args: argparse.Namespace, payload: dict[str, Any], *, ok: bool, mode: str) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    codes = set(str(code) for code in (payload.get("diagnosis_codes") or []))
    step = payload.get("step") if isinstance(payload.get("step"), dict) else {}
    detail = " ".join(
        str(value)
        for value in [payload.get("detail"), payload.get("error"), step.get("stderr_tail"), step.get("error")]
        if value
    )
    hf_missing = (
        "hf_dependencies_missing" in codes
        or "product_swarm_mvp_hf_runtime_missing" in codes
        or "transformers" in detail
    )
    if hf_missing:
        commands.append(command_entry(
            "install Hugging Face runtime",
            ["python", "-m", "pip", "install", "-e", ".[hf]"],
        ))
    if mode == "local":
        commands.append(command_entry(
            "run local inference",
            _infer_command_args(args, full_evidence=False),
        ))
        if not bool(getattr(args, "full_evidence", False)):
            commands.append(command_entry(
                "run broader local evidence",
                _infer_command_args(args, full_evidence=True),
            ))
        return commands

    route = payload.get("route") if isinstance(payload.get("route"), dict) else {}
    coordinator_url = str(route.get("coordinator_url") or getattr(args, "coordinator_url", "") or "")
    peer_bootstrap = str(getattr(args, "peer_bootstrap", "") or "")
    use_p2p = bool(getattr(args, "p2p", False) or peer_bootstrap)
    swarm_id = str(getattr(args, "swarm_id", "") or "default")
    if "p2p_discovery_unreachable" in codes:
        p2pd_command = ["crowdtensor", "p2pd", "--port", str(local_coordinator_port_from_url(peer_bootstrap, default=8788))]
        if swarm_id != "default":
            p2pd_command.extend(["--swarm-id", swarm_id])
        p2pd_command.append("--run")
        commands.append(command_entry("start P2P discovery daemon", p2pd_command))
    route_missing = "coordinator_route_missing" in codes
    suggested_coordinator_url = coordinator_url or (
        "http://127.0.0.1:8787" if route_missing and not use_p2p else ""
    )
    dry_run_command = _infer_command_args(
        args,
        mode="existing",
        dry_run=True,
        include_admin=False,
        include_observer=True,
        coordinator_url_override=suggested_coordinator_url,
    )
    submit_command = _infer_command_args(
        args,
        mode="existing",
        dry_run=False,
        include_admin=False,
        include_observer=False,
        coordinator_url_override=suggested_coordinator_url,
    )
    check_command = command_entry(
        "check existing swarm",
        dry_run_command,
        requires_env=["CROWDTENSOR_OBSERVER_TOKEN"],
    )
    submit_command_entry = command_entry(
        "submit inference",
        submit_command,
        requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
    )
    startup_needed = {
        "generate_route_unavailable",
        "coordinator_route_missing",
        "coordinator_ready_failed",
        "stage_preflight_failed",
    }
    needs_startup_commands = bool(not ok and codes.intersection(startup_needed))
    if needs_startup_commands:
        local_port = local_coordinator_port_from_url(suggested_coordinator_url or coordinator_url)
        serve_args = argparse.Namespace(
            profile="gpu-generation" if getattr(args, "backend", "cpu") == "cuda" else "cpu-real-llm",
            bind_host="127.0.0.1",
            public_host="127.0.0.1",
            port=local_port,
            state_dir="state",
            hf_model_id=getattr(args, "hf_model_id", "sshleifer/tiny-gpt2"),
            hf_cache_dir=getattr(args, "hf_cache_dir", ""),
            lease_seconds=15.0,
            p2p=use_p2p,
            p2p_backend=getattr(args, "p2p_backend", "lite"),
            peer_bootstrap=peer_bootstrap or DEFAULT_P2P_BOOTSTRAP,
            swarm_id=swarm_id,
            peer_id="",
            peer_url="",
            i_understand_public_bind=False,
            admin_token=getattr(args, "admin_token", ""),
            miner_token="",
            observer_token=getattr(args, "observer_token", ""),
            peer_secret="",
        )
        commands.extend([
            command_entry("start Coordinator", _product_cli_serve_command(serve_args, include_run=True)),
            command_entry(
                "start stage0 Miner",
                _product_cli_join_command(
                    serve_args,
                    coordinator_url=suggested_coordinator_url or coordinator_url or f"http://127.0.0.1:{local_port}",
                    stage="stage0",
                    miner_id="stage0-miner",
                    include_run=True,
                ),
            ),
            command_entry(
                "start stage1 Miner",
                _product_cli_join_command(
                    serve_args,
                    coordinator_url=suggested_coordinator_url or coordinator_url or f"http://127.0.0.1:{local_port}",
                    stage="stage1",
                    miner_id="stage1-miner",
                    include_run=True,
                ),
            ),
        ])
    commands.append(check_command)
    commands.append(submit_command_entry)
    if "generation_timeout" in codes:
        wait_progress = payload.get("wait_progress") if isinstance(payload.get("wait_progress"), dict) else {}
        retry_command = submit_command[:]
        retry_command.extend(["--timeout-seconds", str(_retry_timeout_seconds(wait_progress))])
        commands.append(command_entry(
            "retry inference with longer timeout",
            retry_command,
            requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
        ))
    return commands


def _infer_ready_to_submit(
    payload: dict[str, Any],
    route: dict[str, Any],
    *,
    ok: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source = payload.get("ready_to_submit") if isinstance(payload.get("ready_to_submit"), dict) else {}
    if not dry_run and not source:
        return {}
    coordinator_ready = payload.get("coordinator_ready") if isinstance(payload.get("coordinator_ready"), dict) else {}
    stage_preflight = payload.get("stage_preflight") if isinstance(payload.get("stage_preflight"), dict) else {}
    stage_checked = bool(stage_preflight.get("checked"))
    stage_ok = bool(stage_preflight.get("ok")) if stage_checked else None
    route_ready = bool(source.get("route_ready")) if "route_ready" in source else bool(route.get("route_ready"))
    coordinator_ok = (
        bool(source.get("coordinator_ready"))
        if source.get("coordinator_ready") is not None
        else (
            bool(coordinator_ready.get("ok"))
            if "ok" in coordinator_ready
            else None
        )
    )
    stage_preflight_ok = source.get("stage_preflight_ok") if source.get("stage_preflight_ok") is not None else stage_ok
    stage_preflight_required = (
        bool(source.get("stage_preflight_required"))
        if "stage_preflight_required" in source
        else stage_checked
    )
    submit_ok = bool(source.get("ok")) if source.get("ok") is not None else (bool(ok) if dry_run else None)
    return _ready_to_submit_status(
        submit_ok=submit_ok,
        route_ready=route_ready,
        coordinator_ok=coordinator_ok,
        coordinator_preflight_required=source.get("coordinator_preflight_required"),
        stage_preflight_ok=stage_preflight_ok,
        stage_preflight_required=stage_preflight_required,
        source=source.get("source") or ("infer-dry-run-preflight" if dry_run else "infer-submit-result"),
    )


def _infer_summary_from_payload(
    args: argparse.Namespace,
    payload: dict[str, Any],
    *,
    mode: str,
    output_dir: Path,
    step: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generation = _infer_generation_from_report(payload)
    route = _infer_route_from_report(payload)
    stream = _infer_stream_from_report(payload)
    batch = _infer_batch_from_report(payload)
    wait_progress = payload.get("wait_progress") if isinstance(payload.get("wait_progress"), dict) else {}
    local_output = payload.get("local_output") if isinstance(payload.get("local_output"), dict) else {}
    allow_local_generated_text = bool(not getattr(args, "json", False))
    generated_text = (
        local_output.get("generated_text")
        if allow_local_generated_text and isinstance(local_output.get("generated_text"), str)
        else ""
    )
    raw_outputs = local_output.get("outputs") if isinstance(local_output.get("outputs"), list) else []
    display_outputs = []
    if allow_local_generated_text:
        for item in raw_outputs:
            if not isinstance(item, dict):
                continue
            display_outputs.append({
                "request_id": item.get("request_id"),
                "prompt_hash": item.get("prompt_hash"),
                "generated_token_count": item.get("generated_token_count"),
                "generated_text": item.get("generated_text") if isinstance(item.get("generated_text"), str) else "",
            })
    if not generated_text and display_outputs:
        first_output = display_outputs[0]
        generated_text = str(first_output.get("generated_text") or "")
    generated_tokens = int(generation.get("generated_token_count") or 0)
    max_new_tokens = int(generation.get("max_new_tokens") or getattr(args, "max_new_tokens", 0) or 0)
    try:
        prompts = parse_prompt_texts_arg(str(getattr(args, "prompt_text", "") or ""), str(getattr(args, "prompt_texts", "") or ""))
    except ValueError:
        prompts = [str(getattr(args, "prompt_text", "") or "")]
    dry_run = bool(payload.get("dry_run"))
    if dry_run:
        ok = bool(payload.get("ok") and route.get("route_ready"))
    else:
        ok = bool(payload.get("ok") and generated_tokens >= max_new_tokens and max_new_tokens > 0)
    codes = set(payload.get("diagnosis_codes") or [])
    if ok:
        if dry_run:
            codes.add("crowdtensor_infer_preflight_ready")
            codes.add("user_friendly_infer_preflight_ready")
        else:
            codes.add("crowdtensor_infer_ready")
            codes.add("user_friendly_infer_ready")
    else:
        codes.add("crowdtensor_infer_blocked")
    ready_to_submit = _infer_ready_to_submit(payload, route, ok=ok, dry_run=dry_run)
    operator_action = _infer_operator_action(args, payload, ok=ok)
    next_commands = _infer_next_commands(args, payload, ok=ok, mode=mode)
    summary = {
        "schema": INFER_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": mode,
        "dry_run": dry_run,
        "output_dir": str(output_dir),
        "prompt": {
            "prompt_hash": stable_hash_payload([stable_hash_text(prompt) for prompt in prompts]),
            "prompt_count": int(batch.get("expected_request_count") or batch.get("request_count") or 1),
            "raw_prompt_public": False,
        },
        "model": {
            "hf_model_id": _infer_model_from_report(payload, str(getattr(args, "hf_model_id", "") or "sshleifer/tiny-gpt2")),
            "backend": str(getattr(args, "backend", "cpu") or "cpu"),
        },
        "generation": {
            "generated_token_count": generated_tokens,
            "max_new_tokens": max_new_tokens,
            "generated_text_hash": generation.get("generated_text_hash"),
            "decoded_tokens_match": generation.get("decoded_tokens_match"),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "route": route,
        "p2p": payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {},
        "coordinator_ready": payload.get("coordinator_ready") if isinstance(payload.get("coordinator_ready"), dict) else {},
        "stage_preflight": payload.get("stage_preflight") if isinstance(payload.get("stage_preflight"), dict) else {},
        "ready_to_submit": ready_to_submit,
        "batch": {
            "enabled": bool(batch.get("enabled")),
            "ready": bool(batch.get("batch_generation_ready")),
            "observed_request_count": batch.get("observed_request_count"),
        },
        "stream": {
            "enabled": bool(stream.get("enabled") or stream.get("requested")),
            "ready": bool(stream.get("stream_generation_ready")),
            "event_count": int(stream.get("event_count") or 0),
            "source": stream.get("source"),
            "progress": _safe_infer_stream_progress(stream),
            "events": _safe_infer_stream_events(stream),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "wait_progress": wait_progress,
        "local_output": {
            "available": bool(generated_text or display_outputs),
            "generated_text": generated_text if generated_text else "",
            "outputs": display_outputs,
            "output_count": len(display_outputs) if display_outputs else (1 if generated_text else 0),
            "source": local_output.get("source") or "",
            "note": local_output.get("note") or "",
            "display_only": bool(generated_text or display_outputs),
            "public_artifact_safe": False,
        },
        "source_report": {
            "schema": payload.get("schema"),
            "mode": payload.get("mode"),
            "ok": payload.get("ok"),
        },
        "operator_action": operator_action,
        "next_commands": next_commands,
        "step": step or {},
        "diagnosis_codes": sorted(codes),
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "read_only_workload": True,
            "not_production": True,
            "coordinator_backed": True,
        },
        "limitations": [
            "User-friendly inference front door for the current small-model swarm path.",
            "Coordinator-backed, read-only, tiny/small-model scoped; not production Hivemind/Petals parity or large-model serving.",
        ],
    }
    artifacts = {
        "infer_summary": {
            "kind": "crowdtensor_infer_summary",
            "path": "infer_summary.json",
            "present": True,
            "schema": INFER_CLI_SCHEMA,
            "ok": ok,
        }
    }
    if mode == "local":
        if payload.get("schema") == "product_swarm_mvp_check_v1":
            artifacts["product_swarm_mvp_report"] = artifact_entry(
                output_dir / "product-swarm-mvp" / "product_swarm_mvp_check.json",
                output_dir,
                kind="product_swarm_mvp_check",
                schema="product_swarm_mvp_check_v1",
                ok=payload.get("ok") if payload else None,
            )
        else:
            artifacts["public_swarm_v2_report"] = artifact_entry(
                output_dir / "public-swarm-v2" / "public_swarm_inference_v2.json",
                output_dir,
                kind="public_swarm_inference_v2",
                schema="public_swarm_inference_v2",
                ok=payload.get("ok") if payload else None,
            )
    summary["artifacts"] = artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    persisted_summary = json.loads(json.dumps(summary))
    _strip_infer_local_output_text(persisted_summary)
    (output_dir / "infer_summary.json").write_text(
        json.dumps(sanitize(redact_values(persisted_summary)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sanitize(redact_values(summary))


def build_infer(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "infer_mode", "local")
    if mode == "existing":
        generate_args = argparse.Namespace(
            prompt_text=args.prompt_text,
            prompt_texts=args.prompt_texts,
            scenario_id=args.scenario_id,
            max_new_tokens=args.max_new_tokens,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            coordinator_url=args.coordinator_url,
            peer_bootstrap=args.peer_bootstrap,
            p2p=bool(args.p2p or args.peer_bootstrap),
            p2p_backend=args.p2p_backend,
            swarm_id=getattr(args, "swarm_id", "default"),
            admin_token=args.admin_token,
            observer_token=getattr(args, "observer_token", ""),
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            http_timeout=args.http_timeout,
            admin_results_limit=args.admin_results_limit,
            dry_run=args.dry_run,
            skip_live_preflight=bool(args.dry_run),
            include_output=args.include_output,
            stream=args.stream,
            json=args.json,
        )
        payload = build_product_generate(generate_args)
        if args.dry_run:
            payload = _attach_infer_existing_preflight(payload, args)
        return _infer_summary_from_payload(args, payload, mode=mode, output_dir=output_dir)
    if not args.full_evidence:
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "product_swarm_mvp_check.py"),
            "--output-dir",
            str(output_dir / "product-swarm-mvp"),
            "--port",
            str(args.coordinator_port),
            "--backend",
            args.backend,
            "--hf-model-id",
            args.hf_model_id,
            "--prompt-text",
            args.prompt_text,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--startup-timeout",
            str(args.startup_timeout),
            "--session-queue-timeout",
            str(args.timeout_seconds),
            "--miner-timeout",
            str(args.timeout_seconds),
            "--generate-timeout",
            str(args.timeout_seconds),
            "--require-hf-runtime",
            "--json",
        ]
        if args.prompt_texts:
            command.extend(["--prompt-texts", args.prompt_texts])
        if args.stream:
            command.append("--stream-generation")
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        step, payload = run_json_step(
            "crowdtensor_infer_local_product_loopback",
            command,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=int(max(float(args.timeout_seconds), float(args.startup_timeout), 60.0) + 900.0),
        )
        if not payload:
            payload = {
                "schema": "product_swarm_mvp_check_v1",
                "ok": False,
                "mode": "local-loopback",
                "diagnosis_codes": ["crowdtensor_infer_source_report_missing"],
            }
        if not args.json:
            try:
                prompts = parse_prompt_texts_arg(str(args.prompt_text or ""), str(args.prompt_texts or ""))
            except ValueError:
                prompts = [str(args.prompt_text or "")]
            local_output = _infer_local_output_from_private_state(
                output_dir / "product-swarm-mvp",
                prompts=prompts,
                generation=_infer_generation_from_report(payload),
            )
            if local_output:
                payload = {**payload, "local_output": local_output}
        return _infer_summary_from_payload(args, payload, mode=mode, output_dir=output_dir, step=step)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_v2_pack.py"),
        "local",
        "--output-dir",
        str(output_dir / "public-swarm-v2"),
        "--usable-report",
        args.usable_report,
        "--preview-report",
        args.preview_report,
        "--real-p2p-report",
        args.real_p2p_report,
        "--gpu-report",
        args.gpu_report,
        "--fresh-external-attempt-report",
        args.fresh_external_attempt_report,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--real-p2p-port",
        str(args.real_p2p_port),
        "--real-p2p-coordinator-port",
        str(args.real_p2p_coordinator_port),
        "--real-p2p-libp2p-port",
        str(args.real_p2p_libp2p_port),
        "--real-p2p-discovery-backend",
        args.real_p2p_discovery_backend,
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if args.stream:
        command.append("--stream-generation")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    step, payload = run_json_step(
        "crowdtensor_infer_local_swarm",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=int(max(float(args.timeout_seconds), float(args.startup_timeout), 60.0) + 1800.0),
    )
    if not payload:
        payload = {
            "schema": "public_swarm_inference_v2",
            "ok": False,
            "mode": "local",
            "diagnosis_codes": ["crowdtensor_infer_source_report_missing"],
        }
    if not args.json:
        try:
            prompts = parse_prompt_texts_arg(str(args.prompt_text or ""), str(args.prompt_texts or ""))
        except ValueError:
            prompts = [str(args.prompt_text or "")]
        local_output = _infer_local_output_from_private_state(
            output_dir / "public-swarm-v2",
            prompts=prompts,
            generation=_infer_generation_from_report(payload),
        )
        if local_output:
            payload = {**payload, "local_output": local_output}
    return _infer_summary_from_payload(args, payload, mode=mode, output_dir=output_dir, step=step)


def build_public_swarm_developer_preview(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "preview_mode", "local")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_developer_preview_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--product-beta-report",
        args.product_beta_report,
        "--prompt-text",
        args.prompt_text,
        "--scenario-id",
        args.scenario_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_developer_preview",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 480.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_DEVELOPER_PREVIEW_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_developer_preview_v1",
        "cli_schema": PUBLIC_SWARM_DEVELOPER_PREVIEW_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_developer_preview_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm Developer Preview; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_live_preview_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "live_preview_mode", "local-smoke")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_live_preview_rc_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--ready-url",
        args.ready_url,
        "--coordinator-url",
        args.coordinator_url,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--developer-preview-report",
        args.developer_preview_report,
        "--alpha-rc-report",
        args.alpha_rc_report,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--failure-mode",
        args.failure_mode,
        "--kaggle-owner",
        args.kaggle_owner,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--kaggle-push-timeout-seconds",
        str(args.kaggle_push_timeout_seconds),
        "--kaggle-delete-timeout-seconds",
        str(args.kaggle_delete_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--victim-compute-seconds",
        str(args.victim_compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--claim-observe-timeout",
        str(args.claim_observe_timeout),
        "--requeue-timeout",
        str(args.requeue_timeout),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if args.inline_kernel_payload:
        command.append("--inline-kernel-payload")
    else:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    if args.keep_live_private_artifacts:
        command.append("--keep-live-private-artifacts")
    if args.keep_child_artifacts:
        command.append("--keep-child-artifacts")
    step, payload = run_json_step(
        "public_swarm_live_preview_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 960.0,
    )
    if payload:
        payload = sanitize(redact_values(payload))
        payload.setdefault("cli_schema", PUBLIC_SWARM_LIVE_PREVIEW_RC_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "public_swarm_live_preview_rc_v1",
        "cli_schema": PUBLIC_SWARM_LIVE_PREVIEW_RC_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_live_preview_rc_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm Live Preview RC; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_operator_preview(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "operator_preview_mode", "local-smoke")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_operator_preview_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--release-base-port",
        str(args.release_base_port),
        "--ready-url",
        args.ready_url,
        "--coordinator-url",
        args.coordinator_url,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--developer-preview-report",
        args.developer_preview_report,
        "--alpha-rc-report",
        args.alpha_rc_report,
        "--live-stage0-report",
        args.live_stage0_report,
        "--live-stage1-report",
        args.live_stage1_report,
        "--release-readiness-report",
        args.release_readiness_report,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--failure-mode",
        args.failure_mode,
        "--kaggle-owner",
        args.kaggle_owner,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--release-timeout-seconds",
        str(args.release_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--kaggle-push-timeout-seconds",
        str(args.kaggle_push_timeout_seconds),
        "--kaggle-delete-timeout-seconds",
        str(args.kaggle_delete_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--victim-compute-seconds",
        str(args.victim_compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--claim-observe-timeout",
        str(args.claim_observe_timeout),
        "--requeue-timeout",
        str(args.requeue_timeout),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if args.inline_kernel_payload:
        command.append("--inline-kernel-payload")
    else:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    if args.keep_live_private_artifacts:
        command.append("--keep-live-private-artifacts")
    if args.keep_child_artifacts:
        command.append("--keep-child-artifacts")
    if args.allow_dirty_release:
        command.append("--allow-dirty-release")
    step, payload = run_json_step(
        "public_swarm_operator_preview",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), float(args.release_timeout_seconds), 60.0) + 1260.0,
    )
    if payload:
        payload = sanitize(redact_values(payload))
        payload.setdefault("cli_schema", PUBLIC_SWARM_OPERATOR_PREVIEW_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "public_swarm_operator_preview_v1",
        "cli_schema": PUBLIC_SWARM_OPERATOR_PREVIEW_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_operator_preview_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm v0.1 Operator Preview; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_trial(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "swarm_trial_mode", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_trial_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--backend",
        args.backend,
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--release-base-port",
        str(args.release_base_port),
        "--ready-url",
        args.ready_url,
        "--coordinator-url",
        args.coordinator_url,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--product-beta-report",
        args.product_beta_report,
        "--operator-preview-report",
        args.operator_preview_report,
        "--developer-preview-report",
        args.developer_preview_report,
        "--alpha-rc-report",
        args.alpha_rc_report,
        "--live-stage0-report",
        args.live_stage0_report,
        "--live-stage1-report",
        args.live_stage1_report,
        "--release-readiness-report",
        args.release_readiness_report,
        "--prompt-text",
        args.prompt_text,
        "--scenario-id",
        args.scenario_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--failure-mode",
        args.failure_mode,
        "--kaggle-owner",
        args.kaggle_owner,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--release-timeout-seconds",
        str(args.release_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--kaggle-push-timeout-seconds",
        str(args.kaggle_push_timeout_seconds),
        "--kaggle-delete-timeout-seconds",
        str(args.kaggle_delete_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--victim-compute-seconds",
        str(args.victim_compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--claim-observe-timeout",
        str(args.claim_observe_timeout),
        "--requeue-timeout",
        str(args.requeue_timeout),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if args.inline_kernel_payload:
        command.append("--inline-kernel-payload")
    else:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    if args.keep_live_private_artifacts:
        command.append("--keep-live-private-artifacts")
    if args.keep_child_artifacts:
        command.append("--keep-child-artifacts")
    if args.allow_dirty_release:
        command.append("--allow-dirty-release")
    step, payload = run_json_step(
        "public_swarm_trial",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), float(args.release_timeout_seconds), float(args.kaggle_status_timeout_seconds), 60.0) + 1800.0,
    )
    if payload:
        payload = redact_values(payload)
        payload.setdefault("cli_schema", PUBLIC_SWARM_TRIAL_CLI_SCHEMA)
        return payload
    return {
        "schema": "public_swarm_trial_v1",
        "cli_schema": PUBLIC_SWARM_TRIAL_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_trial_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm v0.2 Usable Inference Trial; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    }


def build_public_swarm_preview_v04(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "preview_v04_mode", "evidence-import")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_preview_v04_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--backend",
        args.backend,
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--optional-model-id",
        args.optional_model_id,
        "--gpu-report",
        args.gpu_report,
        "--live-stage0-report",
        args.live_stage0_report,
        "--live-stage1-report",
        args.live_stage1_report,
        "--product-mvp-report",
        args.product_mvp_report,
        "--optional-model-report",
        args.optional_model_report,
        "--product-beta-report",
        args.product_beta_report,
        "--prompt-text",
        args.prompt_text,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--kaggle-owner",
        args.kaggle_owner,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--session-queue-timeout",
        str(args.session_queue_timeout),
        "--miner-timeout",
        str(args.miner_timeout),
        "--generate-timeout",
        str(args.generate_timeout),
        "--json",
    ]
    if args.run_optional_model:
        command.append("--run-optional-model")
    if args.require_optional_model_ready:
        command.append("--require-optional-model-ready")
    if args.require_hf_runtime:
        command.append("--require-hf-runtime")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    step, payload = run_json_step(
        "public_swarm_preview_v04",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), float(args.miner_timeout), float(args.generate_timeout), 60.0) + 1500.0,
    )
    if payload:
        payload = redact_values(payload)
        payload.setdefault("cli_schema", PUBLIC_SWARM_PREVIEW_V04_CLI_SCHEMA)
        return payload
    return {
        "schema": "public_swarm_preview_v04_v1",
        "cli_schema": PUBLIC_SWARM_PREVIEW_V04_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_preview_v04_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm Preview v0.4; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    }


def build_p2p_swarm_inference_v06(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "p2p_swarm_v06_mode", "evidence-import")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "p2p_swarm_inference_v06_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--swarm-id",
        args.swarm_id,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--preview-v04-report",
        args.preview_v04_report,
        "--product-mvp-report",
        args.product_mvp_report,
        "--optional-model-report",
        args.optional_model_report,
        "--p2p-discovery-report",
        args.p2p_discovery_report,
        "--json",
    ]
    if getattr(args, "peer_secret", ""):
        command.extend(["--peer-secret", args.peer_secret])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if getattr(args, "stream_generation", False):
        command.append("--stream-generation")
    if getattr(args, "require_signed", False):
        command.append("--require-signed")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.peer_bootstrap:
        command.extend(["--peer-bootstrap", args.peer_bootstrap])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.verify_generate:
        command.append("--verify-generate")
    if mode == "kaggle-auto":
        command.extend([
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-stage-timeout-seconds",
            str(args.kaggle_stage_timeout_seconds),
        ])
        if args.kaggle_owner:
            command.extend(["--kaggle-owner", args.kaggle_owner])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    step, payload = run_json_step(
        "p2p_swarm_inference_v06",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.startup_timeout), 60.0) + (900.0 if mode == "kaggle-auto" else 300.0),
        redact_secrets=[getattr(args, "peer_secret", ""), getattr(args, "admin_token", "")],
    )
    if payload:
        payload = redact_values(payload)
        payload.setdefault("cli_schema", P2P_SWARM_INFERENCE_V06_CLI_SCHEMA)
        return payload
    return {
        "schema": "p2p_swarm_inference_v06_v1",
        "cli_schema": P2P_SWARM_INFERENCE_V06_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["p2p_swarm_inference_v06_failed"],
        "limitations": [
            "P2P discovery/routing prototype with Coordinator result-ledger fallback",
            "Does not provide production NAT traversal, decentralized security, economic system, large-model throughput, or Hivemind/Petals parity",
        ],
    }


def build_public_p2p_swarm_inference_v1_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "public_p2p_v1_rc_mode", "evidence-import")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_p2p_swarm_inference_v1_rc_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--swarm-id",
        args.swarm_id,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--v06-local-report",
        args.v06_local_report,
        "--v06-external-report",
        args.v06_external_report,
        "--v06-kaggle-report",
        args.v06_kaggle_report,
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.peer_secret:
        command.extend(["--peer-secret", args.peer_secret])
    if args.signed_local_report:
        command.extend(["--signed-local-report", args.signed_local_report])
    if mode == "kaggle-auto":
        command.extend([
            "--kaggle-stage-timeout-seconds",
            str(args.kaggle_stage_timeout_seconds),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
        ])
        if args.kaggle_owner:
            command.extend(["--kaggle-owner", args.kaggle_owner])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    step, payload = run_json_step(
        "public_p2p_swarm_inference_v1_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.startup_timeout), 60.0) + (1500.0 if mode == "kaggle-auto" else 900.0),
        redact_secrets=[args.peer_secret],
    )
    if payload:
        payload = redact_values(payload, [args.peer_secret])
        payload.setdefault("cli_schema", PUBLIC_P2P_SWARM_INFERENCE_V1_RC_CLI_SCHEMA)
        return sanitize(payload)
    return sanitize({
        "schema": "public_p2p_swarm_inference_v1_rc_v1",
        "cli_schema": PUBLIC_P2P_SWARM_INFERENCE_V1_RC_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_p2p_swarm_inference_v1_rc_failed"],
        "limitations": [
            "Signed P2P public-preview RC with Coordinator result-ledger fallback",
            "Does not provide production NAT traversal, libp2p DHT, decentralized security, economic system, large-model throughput, or Hivemind/Petals production parity",
        ],
    })


def build_real_p2p_swarm_inference_core_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "real_p2p_rc_mode", "local-smoke")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_p2p_swarm_inference_core_rc_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--swarm-id",
        args.swarm_id,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--libp2p-port",
        str(args.libp2p_port),
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--session-queue-timeout",
        str(args.session_queue_timeout),
        "--miner-timeout",
        str(args.miner_timeout),
        "--generate-timeout",
        str(args.generate_timeout),
        "--http-timeout",
        str(args.http_timeout),
        "--discovery-backend",
        args.discovery_backend,
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if getattr(args, "stream_generation", False):
        command.append("--stream-generation")
    if args.peer_secret:
        command.extend(["--peer-secret", args.peer_secret])
    if mode == "external-existing":
        command.extend(["--peer-bootstrap", args.peer_bootstrap])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.verify_generate:
            command.append("--verify-generate")
    if mode == "evidence-import" and args.real_p2p_report:
        command.extend(["--real-p2p-report", args.real_p2p_report])
    if mode in {"kaggle-auto", "kaggle-connectivity", "kaggle-runtime-smoke"}:
        command.extend([
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-stage-timeout-seconds",
            str(args.kaggle_stage_timeout_seconds),
        ])
        if hasattr(args, "kaggle_status_poll_seconds"):
            command.extend(["--kaggle-status-poll-seconds", str(args.kaggle_status_poll_seconds)])
        command.extend([
            "--failure-mode",
            args.failure_mode,
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--victim-compute-seconds",
            str(args.victim_compute_seconds),
            "--claim-observe-timeout",
            str(args.claim_observe_timeout),
            "--requeue-timeout",
            str(args.requeue_timeout),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.kaggle_owner:
            command.extend(["--kaggle-owner", args.kaggle_owner])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    step, payload = run_json_step(
        "real_p2p_swarm_inference_core_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.generate_timeout), float(getattr(args, "kaggle_stage_timeout_seconds", 60.0)), 60.0) + (1200.0 if mode in {"kaggle-auto", "kaggle-runtime-smoke"} else 300.0),
        redact_secrets=[args.peer_secret, args.admin_token],
    )
    if payload:
        payload = redact_values(payload, [args.peer_secret, args.admin_token])
        payload.setdefault("cli_schema", REAL_P2P_SWARM_INFERENCE_CORE_RC_CLI_SCHEMA)
        return sanitize(payload)
    return sanitize({
        "schema": "real_p2p_swarm_inference_core_rc_v1",
        "cli_schema": REAL_P2P_SWARM_INFERENCE_CORE_RC_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["real_p2p_swarm_inference_core_rc_failed"],
        "limitations": [
            "Real P2P provider-record discovery with Coordinator result-ledger fallback",
            "Does not provide Hivemind/Petals production parity, economics, large-model throughput, or complete anti-Sybil security",
        ],
    })


def build_petals_class_p2p_candidate(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "petals_candidate_mode", "evidence-import")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "petals_class_p2p_candidate_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--local-report",
        args.local_report,
        "--runtime-smoke-report",
        args.runtime_smoke_report,
        "--external-report",
        args.external_report,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--libp2p-port",
        str(args.libp2p_port),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--session-queue-timeout",
        str(args.session_queue_timeout),
        "--miner-timeout",
        str(args.miner_timeout),
        "--generate-timeout",
        str(args.generate_timeout),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if args.requeue_report:
        command.extend(["--requeue-report", args.requeue_report])
    if args.peer_scoring_report:
        command.extend(["--peer-scoring-report", args.peer_scoring_report])
    if args.allow_retained_alpha_without_requeue:
        command.append("--allow-retained-alpha-without-requeue")
    step, payload = run_json_step(
        "petals_class_p2p_candidate",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(
            float(args.timeout_seconds),
            float(args.startup_timeout),
            float(args.generate_timeout),
            float(args.miner_timeout),
            60.0,
        ) + (900.0 if mode == "local-smoke" else 120.0),
    )
    if payload:
        payload = sanitize(redact_values(payload))
        payload.setdefault("cli_schema", PETALS_CLASS_P2P_CANDIDATE_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "petals_class_p2p_candidate_v1",
        "cli_schema": PETALS_CLASS_P2P_CANDIDATE_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["petals_class_p2p_candidate_failed"],
        "limitations": [
            "Petals-class P2P candidate aggregate with Coordinator result-ledger fallback",
            "Does not provide full Hivemind/Petals production parity, complete NAT traversal, economics, anti-Sybil security, or large-model throughput",
        ],
    })


def build_public_swarm_gpu_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    action = getattr(args, "public_swarm_gpu_beta_action", "local-smoke")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_gpu_inference_beta_pack.py"),
        action,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if action in {"prepare", "coordinator", "miner", "verify", "collect", "local-loopback", "local-smoke", "kaggle-package", "kaggle-auto"}:
        command.extend([
            "--coordinator-url",
            args.coordinator_url,
            "--public-host",
            args.public_host,
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--hf-model-id",
            args.hf_model_id,
            "--real-llm-partition-mode",
            getattr(args, "real_llm_partition_mode", "stage-local"),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if action in {"prepare", "coordinator", "miner"}:
        command.extend(["--lease-seconds", str(args.lease_seconds)])
    if action == "kaggle-auto":
        command.extend([
            "--kaggle-owner",
            args.kaggle_owner,
            "--dataset-title",
            args.dataset_title,
            "--kernel-title-prefix",
            args.kernel_title_prefix,
            "--startup-timeout",
            str(args.startup_timeout),
            "--process-exit-timeout",
            str(args.process_exit_timeout),
            "--poll-interval",
            str(args.poll_interval),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-status-timeout-seconds",
            str(args.kaggle_status_timeout_seconds),
            "--kaggle-status-poll-interval",
            str(args.kaggle_status_poll_interval),
        ])
        if args.dataset_slug:
            command.extend(["--dataset-slug", args.dataset_slug])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        if args.inline_kernel_payload:
            command.append("--inline-kernel-payload")
        else:
            command.append("--no-inline-kernel-payload")
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    if action in {"prepare", "miner"}:
        command.extend([
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
    if action == "kaggle-auto":
        command.extend([
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--victim-compute-seconds",
            str(args.victim_compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--idle-sleep",
            str(args.idle_sleep),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
    if action in {"prepare", "coordinator", "verify", "collect"}:
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
    if action == "prepare" and args.replace:
        command.append("--replace")
    if action == "coordinator" and args.run:
        command.append("--run")
    if action == "miner":
        command.extend(["--stage", args.stage])
        if args.run:
            command.append("--run")
    if action in {"verify", "local-loopback"} and args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if action == "collect":
        if args.miner_id:
            command.extend(["--miner-id", args.miner_id])
        command.extend(["--artifact-timeout", str(args.artifact_timeout)])
    if action == "clean":
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
    if action == "evidence-import":
        command.extend(["--gpu-report", args.gpu_report])
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_gpu_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(getattr(args, "remote_timeout_seconds", 60.0)), 60.0) + 240.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_GPU_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_gpu_inference_beta_v1",
        "cli_schema": PUBLIC_SWARM_GPU_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": action,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_gpu_inference_beta_failed"],
        "limitations": [
            "Optional CUDA read-only Public Swarm GPU Inference Beta; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU pooling marketplace, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_gpu_sharded_generation_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "gpu_generate_mode", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "gpu_sharded_generation_beta_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--public-host",
        args.public_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--bind-host",
        args.bind_host,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if mode == "kaggle-auto":
        command.extend([
            "--kaggle-owner",
            args.kaggle_owner,
            "--dataset-title",
            args.dataset_title,
            "--kernel-title-prefix",
            args.kernel_title_prefix,
            "--startup-timeout",
            str(args.startup_timeout),
            "--process-exit-timeout",
            str(args.process_exit_timeout),
            "--poll-interval",
            str(args.poll_interval),
            "--http-timeout",
            str(args.http_timeout),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-status-timeout-seconds",
            str(args.kaggle_status_timeout_seconds),
            "--kaggle-status-poll-interval",
            str(args.kaggle_status_poll_interval),
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--victim-compute-seconds",
            str(args.victim_compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--idle-sleep",
            str(args.idle_sleep),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.dataset_slug:
            command.extend(["--dataset-slug", args.dataset_slug])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        command.append("--inline-kernel-payload" if args.inline_kernel_payload else "--no-inline-kernel-payload")
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    if mode == "evidence-import":
        command.extend(["--gpu-report", args.gpu_report])
    step, payload = run_json_step(
        "gpu_sharded_generation_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(getattr(args, "kaggle_status_timeout_seconds", 60.0))) + 600.0,
    )
    if payload:
        payload.setdefault("cli_schema", GPU_SHARDED_GENERATION_BETA_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "gpu_sharded_generation_beta_v1",
        "cli_schema": GPU_SHARDED_GENERATION_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["gpu_sharded_generation_blocked"],
        "limitations": [
            "Tiny GPT CUDA sharded generation Beta; not production Swarm Inference",
            "Does not provide P2P routing, GPU marketplace, large-model serving, training, or arbitrary public prompt serving",
        ],
    })


def _serve_task_lane(args: argparse.Namespace) -> str:
    if args.profile == "gpu-generation":
        return f"python-cli:cuda:0:real_llm_sharded_infer"
    return f"python-cli:cpu:0:real_llm_sharded_infer"


def _p2p_stage_capabilities(*, backend: str, stage: str) -> list[str]:
    if stage == "both":
        return required_stage_capabilities(backend=backend, stage_mode="split")
    return required_stage_capabilities(backend=backend, stage_mode=stage)


def build_p2p_peer(
    *,
    swarm_id: str,
    peer_id: str,
    role: str,
    coordinator_url: str = "",
    peer_url: str = "",
    backend: str = "",
    stage_role: str = "",
    hf_model_id: str = "",
    ttl_seconds: float = 60.0,
) -> dict[str, Any]:
    capabilities: dict[str, Any] = {
        "runtime": "python-cli",
        "health": "ready",
    }
    if backend:
        capabilities["backend"] = backend
    if hf_model_id:
        capabilities["hf_model_id"] = hf_model_id
    if stage_role:
        capabilities["real_llm_sharded_stage_role"] = stage_role
        capabilities["real_llm_sharded_stage_capabilities"] = _p2p_stage_capabilities(backend=backend or "cpu", stage=stage_role)
    urls = {}
    if coordinator_url:
        urls["coordinator"] = coordinator_url
    if peer_url:
        urls["peer"] = peer_url
    return sanitize_peer({
        "schema": PEER_SCHEMA,
        "swarm_id": swarm_id,
        "peer_id": peer_id,
        "role": role,
        "urls": urls,
        "backend": backend,
        "stage_role": stage_role,
        "capabilities": capabilities,
        "ttl_seconds": ttl_seconds,
    })


def maybe_sign_p2p_peer(peer: dict[str, Any], peer_secret: str = "") -> dict[str, Any]:
    return sanitize_peer(sign_peer_announcement(peer, peer_secret)) if peer_secret else peer


def _p2p_backend(args: argparse.Namespace) -> str:
    return str(getattr(args, "p2p_backend", "lite") or "lite")


def _p2p_route_source(args: argparse.Namespace) -> str:
    if not getattr(args, "p2p", False):
        return "peer-bootstrap" if getattr(args, "peer_bootstrap", "") else "coordinator-url"
    return "real-p2p-discovery" if _p2p_backend(args) == "real" else "p2p-discovery"


def announce_p2p_peer(bootstrap: str, peer: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    payload = post_announce(bootstrap, peer, timeout=timeout)
    return {
        "ok": bool(payload.get("ok")),
        "schema": payload.get("schema") or "p2p_lite_announce_v1",
        "peer_id": peer.get("peer_id"),
        "bootstrap": bootstrap,
    }


def announce_real_p2p_provider(
    bootstrap: str,
    peer: dict[str, Any],
    *,
    timeout: float,
    record_secret: str = "",
) -> dict[str, Any]:
    record = build_provider_record(peer, record_secret)
    payload = post_provider_record(bootstrap, record, timeout=timeout)
    response_record = payload.get("record") if isinstance(payload.get("record"), dict) else {}
    return {
        "ok": bool(payload.get("ok")),
        "schema": payload.get("schema") or "real_p2p_announce_v1",
        "peer_id": peer.get("peer_id"),
        "record_id": response_record.get("record_id") or record.get("record_id"),
        "bootstrap": bootstrap,
        "provider_schema": PROVIDER_RECORD_SCHEMA,
    }


def announce_discovery_peer(
    bootstrap: str,
    peer: dict[str, Any],
    *,
    timeout: float,
    backend: str = "lite",
    peer_secret: str = "",
) -> dict[str, Any]:
    if backend == "real":
        return announce_real_p2p_provider(bootstrap, peer, timeout=timeout, record_secret=peer_secret)
    return announce_p2p_peer(bootstrap, maybe_sign_p2p_peer(peer, peer_secret), timeout=timeout)


class DiscoveryRefreshThread:
    def __init__(
        self,
        *,
        bootstrap: str,
        peer: dict[str, Any],
        timeout: float,
        backend: str,
        peer_secret: str,
        interval_seconds: float,
    ) -> None:
        self.bootstrap = bootstrap
        self.peer = peer
        self.timeout = timeout
        self.backend = backend
        self.peer_secret = peer_secret
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.stop_event = threading.Event()
        self.last_result: dict[str, Any] = {}
        self.refresh_count = 0
        self.failure_count = 0
        self._thread = threading.Thread(target=self._run, name="crowdtensor-discovery-refresh", daemon=True)

    def start(self) -> "DiscoveryRefreshThread":
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self._thread.join(timeout=max(1.0, min(5.0, self.interval_seconds)))
        return {
            "schema": "discovery_refresh_v1",
            "refresh_count": self.refresh_count,
            "failure_count": self.failure_count,
            "last_result": self.last_result,
            "interval_seconds": self.interval_seconds,
        }

    def _announce_once(self) -> dict[str, Any]:
        peer = dict(self.peer)
        peer.pop("last_seen", None)
        peer.pop("expires_at", None)
        peer.pop("peer_signature", None)
        peer.pop("peer_identity", None)
        return announce_discovery_peer(
            self.bootstrap,
            peer,
            timeout=self.timeout,
            backend=self.backend,
            peer_secret=self.peer_secret,
        )

    def _run_once(self) -> dict[str, Any]:
        try:
            self.last_result = self._announce_once()
            if self.last_result.get("ok"):
                self.refresh_count += 1
            else:
                self.failure_count += 1
        except Exception as exc:
            self.failure_count += 1
            self.last_result = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200]}
        return self.last_result

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._run_once()


def maybe_start_discovery_refresh(args: argparse.Namespace, peer: dict[str, Any], *, bootstrap: str) -> DiscoveryRefreshThread | None:
    if not getattr(args, "p2p", False):
        return None
    ttl = max(1.0, float(getattr(args, "ttl_seconds", 60.0)))
    interval = max(1.0, min(30.0, ttl / 3.0))
    return DiscoveryRefreshThread(
        bootstrap=bootstrap,
        peer=peer,
        timeout=float(getattr(args, "http_timeout", 5.0)),
        backend=_p2p_backend(args),
        peer_secret=str(getattr(args, "peer_secret", "")),
        interval_seconds=interval,
    ).start()


def build_serve_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "coordinator.py"),
        "--host",
        args.bind_host,
        "--port",
        str(args.port),
        "--state-dir",
        args.state_dir,
        "--backlog",
        "0",
        "--task-lane",
        _serve_task_lane(args),
        "--admin-token",
        args.admin_token,
        "--real-llm-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        "hf_transformers_cuda" if args.profile == "gpu-generation" else "hf_transformers_cpu",
        "--real-llm-partition-mode",
        "stage-local",
        "--lease-seconds",
        str(args.lease_seconds),
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.miner_token:
        command.extend(["--miner-token", args.miner_token])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    return command


def _product_cli_serve_command(args: argparse.Namespace, *, include_run: bool = True) -> list[str]:
    command = [
        "crowdtensor",
        "serve",
        "--profile",
        args.profile,
        "--bind-host",
        args.bind_host,
        "--public-host",
        args.public_host,
        "--port",
        str(args.port),
    ]
    if args.state_dir != "state":
        command.extend(["--state-dir", args.state_dir])
    if args.hf_model_id != "sshleifer/tiny-gpt2":
        command.extend(["--hf-model-id", args.hf_model_id])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.lease_seconds != 15.0:
        command.extend(["--lease-seconds", str(args.lease_seconds)])
    if args.p2p:
        command.append("--p2p")
        if args.p2p_backend != "lite":
            command.extend(["--p2p-backend", args.p2p_backend])
        command.extend(["--peer-bootstrap", args.peer_bootstrap or DEFAULT_P2P_BOOTSTRAP])
        if args.swarm_id != "default":
            command.extend(["--swarm-id", args.swarm_id])
        if args.peer_id:
            command.extend(["--peer-id", args.peer_id])
        if args.peer_url:
            command.extend(["--peer-url", args.peer_url])
    if args.bind_host in {"0.0.0.0", "::"} and args.i_understand_public_bind:
        command.append("--i-understand-public-bind")
    if include_run:
        command.append("--run")
    return command


def _product_cli_join_command(
    args: argparse.Namespace,
    *,
    coordinator_url: str,
    stage: str | None = None,
    miner_id: str | None = None,
    include_run: bool = True,
) -> list[str]:
    command = ["crowdtensor", "join"]
    stage_value = stage or args.stage
    miner_value = miner_id or args.miner_id
    backend = str(
        getattr(args, "backend", "")
        or ("cuda" if getattr(args, "profile", "") == "gpu-generation" else "cpu")
    )
    if args.p2p:
        command.append("--p2p")
        if args.p2p_backend != "lite":
            command.extend(["--p2p-backend", args.p2p_backend])
        command.extend(["--peer-bootstrap", args.peer_bootstrap or DEFAULT_P2P_BOOTSTRAP])
        if args.swarm_id != "default":
            command.extend(["--swarm-id", args.swarm_id])
        if args.peer_id and stage is None and miner_id is None:
            command.extend(["--peer-id", args.peer_id])
        if args.peer_url and stage is None and miner_id is None:
            command.extend(["--peer-url", args.peer_url])
    else:
        command.extend(["--coordinator-url", coordinator_url])
    command.extend(["--miner-id", miner_value, "--stage", stage_value])
    if backend != "cpu":
        command.extend(["--backend", backend])
    if args.hf_model_id != "sshleifer/tiny-gpt2":
        command.extend(["--hf-model-id", args.hf_model_id])
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if getattr(args, "once", False):
        command.append("--once")
    if getattr(args, "max_tasks", 0) > 0:
        command.extend(["--max-tasks", str(args.max_tasks)])
    if getattr(args, "max_runtime_seconds", 0.0) > 0:
        command.extend(["--max-runtime-seconds", str(args.max_runtime_seconds)])
    if getattr(args, "compute_seconds", 0.0) > 0:
        command.extend(["--compute-seconds", str(args.compute_seconds)])
    if getattr(args, "max_request_attempts", 3) != 3:
        command.extend(["--max-request-attempts", str(args.max_request_attempts)])
    if include_run:
        command.append("--run")
    return command


def _product_cli_generate_command(
    *,
    p2p: bool,
    coordinator_url: str,
    peer_bootstrap: str,
    p2p_backend: str,
    backend: str,
    hf_model_id: str,
    dry_run: bool,
    max_new_tokens: int = 16,
    prompt_text: str = "",
    prompt_texts: str = "",
    swarm_id: str = "default",
    include_observer: bool = False,
    stream: bool = False,
) -> list[str]:
    command = [
        "crowdtensor",
        "generate",
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if p2p:
        command.append("--p2p")
        if swarm_id != "default":
            command.extend(["--swarm-id", swarm_id])
        if p2p_backend != "lite":
            command.extend(["--p2p-backend", p2p_backend])
        command.extend(["--peer-bootstrap", peer_bootstrap or DEFAULT_P2P_BOOTSTRAP])
    else:
        command.extend(["--coordinator-url", coordinator_url])
    if backend != "cpu":
        command.extend(["--backend", backend])
    if hf_model_id != "sshleifer/tiny-gpt2":
        command.extend(["--hf-model-id", hf_model_id])
    if prompt_texts:
        command.extend(["--prompt-texts", prompt_texts])
    elif prompt_text:
        command.extend(["--prompt-text", prompt_text])
    if dry_run:
        command.append("--dry-run")
    if include_observer:
        command.extend(["--observer-token", OBSERVER_TOKEN_ENV_PLACEHOLDER])
    if stream:
        command.append("--stream")
    return command


def _product_env_requirements(args: argparse.Namespace, names: list[str]) -> list[str]:
    env_by_name = {
        "admin": "CROWDTENSOR_ADMIN_TOKEN",
        "miner": "CROWDTENSOR_MINER_TOKEN",
        "observer": "CROWDTENSOR_OBSERVER_TOKEN",
        "peer": "CROWDTENSOR_P2P_PEER_SECRET",
    }
    requirements: list[str] = []
    for name in names:
        env_name = env_by_name[name]
        value = ""
        if name == "admin":
            value = str(getattr(args, "admin_token", "") or "")
            if value == "local-admin":
                value = ""
        elif name == "miner":
            value = str(getattr(args, "miner_token", "") or "")
        elif name == "observer":
            value = str(getattr(args, "observer_token", "") or "")
        elif name == "peer":
            value = str(getattr(args, "peer_secret", "") or "")
        if value and env_name not in requirements:
            requirements.append(env_name)
    return requirements


def _product_serve_next_commands(args: argparse.Namespace, *, coordinator_url: str) -> list[dict[str, Any]]:
    backend = "cuda" if args.profile == "gpu-generation" else "cpu"
    p2p_bootstrap = args.peer_bootstrap or DEFAULT_P2P_BOOTSTRAP
    return [
        command_entry(
            "start Coordinator",
            _product_cli_serve_command(args, include_run=True),
            requires_env=_product_env_requirements(args, ["admin", "miner", "observer", "peer"]),
        ),
        command_entry(
            "start stage0 Miner",
            _product_cli_join_command(
                args,
                coordinator_url=coordinator_url,
                stage="stage0",
                miner_id="stage0-miner",
                include_run=True,
            ),
            requires_env=_product_env_requirements(args, ["miner", "peer"]),
        ),
        command_entry(
            "start stage1 Miner",
            _product_cli_join_command(
                args,
                coordinator_url=coordinator_url,
                stage="stage1",
                miner_id="stage1-miner",
                include_run=True,
            ),
            requires_env=_product_env_requirements(args, ["miner", "peer"]),
        ),
        command_entry(
            "check generation route",
            _product_cli_generate_command(
                p2p=bool(args.p2p),
                coordinator_url=coordinator_url,
                peer_bootstrap=p2p_bootstrap,
                p2p_backend=_p2p_backend(args),
                backend=backend,
                hf_model_id=args.hf_model_id,
                dry_run=True,
                swarm_id=str(getattr(args, "swarm_id", "default") or "default"),
                include_observer=True,
            ),
            requires_env=["CROWDTENSOR_OBSERVER_TOKEN"],
        ),
        command_entry(
            "submit generation",
            _product_cli_generate_command(
                p2p=bool(args.p2p),
                coordinator_url=coordinator_url,
                peer_bootstrap=p2p_bootstrap,
                p2p_backend=_p2p_backend(args),
                backend=backend,
                hf_model_id=args.hf_model_id,
                dry_run=False,
                swarm_id=str(getattr(args, "swarm_id", "default") or "default"),
            ),
            requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
        ),
    ]


def _product_public_bind_next_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    local_command = [
        "crowdtensor",
        "serve",
        "--profile",
        args.profile,
        "--bind-host",
        "127.0.0.1",
        "--public-host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--run",
    ]
    public_command = _product_cli_serve_command(args, include_run=True)
    public_command.append("--i-understand-public-bind")
    return [
        command_entry(
            "start local-only Coordinator",
            local_command,
            requires_env=_product_env_requirements(args, ["admin", "miner", "observer", "peer"]),
        ),
        command_entry(
            "start trusted public Coordinator",
            public_command,
            requires_env=_product_env_requirements(args, ["admin", "miner", "observer", "peer"]),
        ),
    ]


def _product_join_next_commands(args: argparse.Namespace, *, coordinator_url: str) -> list[dict[str, Any]]:
    p2p_bootstrap = args.peer_bootstrap or (DEFAULT_P2P_BOOTSTRAP if args.p2p else "")
    commands = [
        command_entry(
            "start this Miner",
            _product_cli_join_command(args, coordinator_url=coordinator_url, include_run=True),
            requires_env=_product_env_requirements(args, ["miner", "peer"]),
        )
    ]
    if args.stage == "stage0":
        commands.append(command_entry(
            "start stage1 Miner",
            _product_cli_join_command(
                args,
                coordinator_url=coordinator_url,
                stage="stage1",
                miner_id="stage1-miner",
                include_run=True,
            ),
            requires_env=_product_env_requirements(args, ["miner", "peer"]),
        ))
    elif args.stage == "stage1":
        commands.append(command_entry(
            "start stage0 Miner",
            _product_cli_join_command(
                args,
                coordinator_url=coordinator_url,
                stage="stage0",
                miner_id="stage0-miner",
                include_run=True,
            ),
            requires_env=_product_env_requirements(args, ["miner", "peer"]),
        ))
    commands.append(command_entry(
        "check generation route",
        _product_cli_generate_command(
            p2p=bool(args.p2p),
            coordinator_url=coordinator_url,
            peer_bootstrap=p2p_bootstrap,
            p2p_backend=_p2p_backend(args),
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            dry_run=True,
            swarm_id=str(getattr(args, "swarm_id", "default") or "default"),
            include_observer=True,
        ),
        requires_env=["CROWDTENSOR_OBSERVER_TOKEN"],
    ))
    commands.append(command_entry(
        "submit generation",
        _product_cli_generate_command(
            p2p=bool(args.p2p),
            coordinator_url=coordinator_url,
            peer_bootstrap=p2p_bootstrap,
            p2p_backend=_p2p_backend(args),
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            dry_run=False,
            swarm_id=str(getattr(args, "swarm_id", "default") or "default"),
        ),
        requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
    ))
    return commands


def build_product_serve(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    command = build_serve_command(args)
    public_bind = args.bind_host in {"0.0.0.0", "::"}
    coordinator_url = f"http://{args.public_host}:{args.port}"
    if public_bind and not args.i_understand_public_bind:
        safe_command = redacted_command(command, {"--admin-token", "--miner-token", "--observer-token"})
        return sanitize({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "serve",
            "profile": args.profile,
            "command": safe_command,
            "command_line": command_line(safe_command),
            "next_commands": _product_public_bind_next_commands(args),
            "diagnosis_codes": ["public_bind_requires_explicit_ack"],
            "operator_action": "Add --i-understand-public-bind only on a trusted network boundary, or keep --bind-host on 127.0.0.1.",
            "safety": {"public_bind_requires_explicit_ack": True},
        })
    p2p_bootstrap = args.peer_bootstrap or DEFAULT_P2P_BOOTSTRAP
    peer_announce: dict[str, Any] = {}
    if args.p2p:
        p2p_backend = _p2p_backend(args)
        peer = build_p2p_peer(
            swarm_id=args.swarm_id,
            peer_id=args.peer_id or stable_peer_id(f"{args.swarm_id}:coordinator:{args.public_host}:{args.port}"),
            role="coordinator",
            coordinator_url=coordinator_url,
            peer_url=args.peer_url,
            backend="cuda" if args.profile == "gpu-generation" else "cpu",
            hf_model_id=args.hf_model_id,
            ttl_seconds=args.ttl_seconds,
        )
        try:
            peer_announce = announce_discovery_peer(
                p2p_bootstrap,
                peer,
                timeout=args.http_timeout,
                backend=p2p_backend,
                peer_secret=args.peer_secret,
            )
        except Exception as exc:
            peer_announce = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200], "bootstrap": p2p_bootstrap}
    else:
        peer = {}

    safe_command = redacted_command(command, {"--admin-token", "--miner-token", "--observer-token"})
    report = {
        "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "ok": bool(not args.p2p or peer_announce.get("ok")),
        "mode": "serve",
        "profile": args.profile,
        "coordinator_url": coordinator_url,
        "p2p": {
            "enabled": bool(args.p2p),
            "backend": _p2p_backend(args) if args.p2p else "",
            "bootstrap": p2p_bootstrap if args.p2p else "",
            "announce": peer_announce,
        },
        "command": safe_command,
        "command_line": command_line(safe_command),
        "next_commands": _product_serve_next_commands(args, coordinator_url=coordinator_url),
        "printed_only": not args.run,
        "diagnosis_codes": ["serve_command_ready"] + (
            [("real_p2p_coordinator_announce_ready" if _p2p_backend(args) == "real" else "p2p_coordinator_announce_ready")]
            if args.p2p and peer_announce.get("ok")
            else []
        ),
        "safety": {
            "admin_token_from_env_supported": bool(os.environ.get("CROWDTENSOR_ADMIN_TOKEN")),
            "public_bind_explicit": public_bind,
            "not_production": True,
            "p2p_discovery_enabled": bool(args.p2p),
            "coordinator_result_fallback": True,
        },
    }
    if args.p2p and not peer_announce.get("ok"):
        report["diagnosis_codes"] = [
            "real_p2p_coordinator_announce_failed" if _p2p_backend(args) == "real" else "p2p_coordinator_announce_failed"
        ]
    report["operator_action"] = _product_serve_operator_action(report)
    if not args.run:
        return sanitize(report)
    refresh = maybe_start_discovery_refresh(args, peer, bootstrap=p2p_bootstrap) if peer_announce.get("ok") else None
    try:
        completed = runner(command, cwd=str(ROOT), text=True)
    finally:
        if refresh is not None:
            report["p2p"]["refresh"] = refresh.stop()
    report["returncode"] = completed.returncode
    report["ok"] = bool(report.get("ok") and completed.returncode == 0)
    report["operator_action"] = _product_serve_operator_action(report)
    return sanitize(report)


def _product_serve_operator_action(report: dict[str, Any]) -> str:
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []))
    if "public_bind_requires_explicit_ack" in codes:
        return "Add --i-understand-public-bind only on a trusted network boundary, or keep --bind-host on 127.0.0.1."
    if "p2p_coordinator_announce_failed" in codes or "real_p2p_coordinator_announce_failed" in codes:
        return "Start the matching P2P discovery daemon, check --peer-bootstrap/--peer-secret, then rerun serve --p2p."
    if report.get("ok"):
        generate_hint = "crowdtensor generate --p2p --dry-run" if (report.get("p2p") or {}).get("enabled") else f"crowdtensor generate --coordinator-url {report.get('coordinator_url')} --dry-run"
        if report.get("printed_only"):
            return f"Rerun with --run to start the Coordinator, start stage0 and stage1 Miners with crowdtensor join, then run {generate_hint}."
        if (report.get("p2p") or {}).get("enabled"):
            return "Start or keep stage0 and stage1 Miners joined, then run crowdtensor generate --p2p --dry-run before submitting."
        return f"Start stage0 and stage1 Miners with --coordinator-url, then run {generate_hint}."
    if report.get("returncode") not in {None, 0}:
        return "Coordinator process exited; inspect its stderr/logs, fix the runtime error, and rerun serve --run."
    return "Inspect the serve JSON report and Coordinator command for the failing check."


def build_join_command(args: argparse.Namespace, *, coordinator_url: str) -> list[str]:
    backend = "hf_transformers_cuda" if args.backend == "cuda" else "hf_transformers_cpu"
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        coordinator_url,
        "--miner-id",
        args.miner_id,
        "--enable-hf-tiny-gpt-runtime",
        "--real-llm-backend",
        backend,
        "--real-llm-stage-role",
        args.stage,
        "--real-llm-partition-mode",
        "stage-local",
        "--hf-model-id",
        args.hf_model_id,
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.miner_token:
        command.extend(["--miner-token", args.miner_token])
    if args.once:
        command.append("--once")
    if args.max_tasks > 0:
        command.extend(["--max-tasks", str(args.max_tasks)])
    if getattr(args, "max_runtime_seconds", 0.0) > 0:
        command.extend(["--max-runtime-seconds", str(args.max_runtime_seconds)])
    if getattr(args, "compute_seconds", 0.0) > 0:
        command.extend(["--compute-seconds", str(args.compute_seconds)])
    if getattr(args, "max_request_attempts", 3) != 3:
        command.extend(["--max-request-attempts", str(args.max_request_attempts)])
    if getattr(args, "retry_base_sleep", 0.2) != 0.2:
        command.extend(["--retry-base-sleep", str(args.retry_base_sleep)])
    if getattr(args, "retry_max_sleep", 2.0) != 2.0:
        command.extend(["--retry-max-sleep", str(args.retry_max_sleep)])
    if getattr(args, "idle_sleep", 2.0) != 2.0:
        command.extend(["--idle-sleep", str(args.idle_sleep)])
    return command


def resolve_coordinator_from_bootstrap(bootstrap_url: str, *, timeout: float = 5.0) -> tuple[str, list[dict[str, Any]]]:
    payload = fetch_peer_catalog(bootstrap_url, timeout=timeout)
    peers = payload.get("peers") if isinstance(payload, dict) else []
    peer_list = [peer for peer in peers if isinstance(peer, dict)]
    for peer in peer_list:
        if peer.get("role") != "coordinator":
            continue
        urls = peer.get("urls") if isinstance(peer.get("urls"), dict) else {}
        if urls.get("coordinator"):
            return str(urls["coordinator"]), peer_list
    return "", peer_list


def resolve_coordinator_from_discovery(
    bootstrap_url: str,
    *,
    timeout: float = 5.0,
    backend: str = "lite",
    session_request: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if backend == "real":
        payload = fetch_provider_catalog(bootstrap_url, timeout=timeout)
    else:
        payload = fetch_peer_catalog(bootstrap_url, timeout=timeout)
    peers = payload.get("peers") if isinstance(payload, dict) else []
    peer_list = [peer for peer in peers if isinstance(peer, dict)]
    expected_model_id = str((session_request or {}).get("hf_model_id") or "")
    expected_backend = str((session_request or {}).get("backend") or "")
    for peer in peer_list:
        if peer.get("role") != "coordinator":
            continue
        urls = peer.get("urls") if isinstance(peer.get("urls"), dict) else {}
        coordinator = str(urls.get("coordinator") or peer.get("coordinator_url") or "")
        if not coordinator:
            continue
        if expected_model_id and not peer_model_compatible(peer, expected_model_id):
            continue
        if expected_backend and not peer_backend_compatible(peer, expected_backend):
            continue
        return coordinator, peer_list, payload
    return "", peer_list, payload


def build_product_join(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    coordinator_url = args.coordinator_url
    p2p_bootstrap = args.peer_bootstrap or (DEFAULT_P2P_BOOTSTRAP if args.p2p else "")
    p2p_backend = _p2p_backend(args)
    peers: list[dict[str, Any]] = []
    catalog_payload: dict[str, Any] = {}
    discovery_error: dict[str, Any] = {}
    route_probe_request = build_session_request(
        prompt_text="join",
        backend=args.backend,
        hf_model_id=args.hf_model_id,
        stage_mode="split",
        max_new_tokens=1,
        route_source=_p2p_route_source(args),
    )
    if not coordinator_url and p2p_bootstrap:
        try:
            coordinator_url, peers, catalog_payload = resolve_coordinator_from_discovery(
                p2p_bootstrap,
                timeout=args.http_timeout,
                backend=p2p_backend if args.p2p else "lite",
                session_request=route_probe_request,
            )
        except Exception as exc:
            discovery_error = {
                "ok": False,
                "error": type(exc).__name__,
                "detail": str(exc)[:200],
                "bootstrap": p2p_bootstrap,
            }
    if not coordinator_url:
        local_coordinator_url = "http://127.0.0.1:8787"
        p2p_port = local_coordinator_port_from_url(p2p_bootstrap, default=8788)
        serve_args = argparse.Namespace(
            profile="gpu-generation" if args.backend == "cuda" else "cpu-real-llm",
            bind_host="127.0.0.1",
            public_host="127.0.0.1",
            port=8787,
            state_dir="state",
            admin_token="",
            miner_token="",
            observer_token="",
            hf_model_id=args.hf_model_id,
            hf_cache_dir=args.hf_cache_dir,
            lease_seconds=15.0,
            p2p=bool(args.p2p),
            p2p_backend=p2p_backend,
            peer_bootstrap=p2p_bootstrap or DEFAULT_P2P_BOOTSTRAP,
            swarm_id=args.swarm_id,
            peer_id="",
            peer_url="",
            ttl_seconds=args.ttl_seconds,
            peer_secret="",
            http_timeout=args.http_timeout,
            i_understand_public_bind=False,
            run=True,
            json=False,
        )
        p2p_join_args = argparse.Namespace(**vars(args))
        p2p_join_args.p2p = True
        p2p_join_args.peer_bootstrap = p2p_bootstrap or DEFAULT_P2P_BOOTSTRAP
        p2p_join_args.coordinator_url = ""
        report = {
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "join",
            "next_commands": [
                command_entry("start P2P discovery daemon", ["crowdtensor", "p2pd", "--port", str(p2p_port), "--run"]),
                command_entry("start local Coordinator", _product_cli_serve_command(serve_args, include_run=True)),
                *_product_join_next_commands(args, coordinator_url=local_coordinator_url),
                *(
                    []
                    if args.p2p
                    else [
                        command_entry(
                            "discover through P2P",
                            _product_cli_join_command(p2p_join_args, coordinator_url="", include_run=True),
                        )
                    ]
                ),
            ],
            "p2p": {
                "enabled": bool(args.p2p or p2p_bootstrap),
                "backend": p2p_backend if args.p2p else "lite",
                "bootstrap": p2p_bootstrap,
                "discovery": discovery_error,
            },
            "diagnosis_codes": (["p2p_discovery_unreachable"] if discovery_error else []) + ["coordinator_route_missing"],
        }
        report["operator_action"] = _product_join_operator_action(report)
        return sanitize(report)
    peer_announce: dict[str, Any] = {}
    if args.p2p:
        peer = build_p2p_peer(
            swarm_id=args.swarm_id,
            peer_id=args.peer_id or args.miner_id,
            role="miner",
            peer_url=args.peer_url,
            backend=args.backend,
            stage_role=args.stage,
            hf_model_id=args.hf_model_id,
            ttl_seconds=args.ttl_seconds,
        )
        try:
            peer_announce = announce_discovery_peer(
                p2p_bootstrap,
                peer,
                timeout=args.http_timeout,
                backend=p2p_backend,
                peer_secret=args.peer_secret,
            )
        except Exception as exc:
            peer_announce = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200], "bootstrap": p2p_bootstrap}
    else:
        peer = {}
    command = build_join_command(args, coordinator_url=coordinator_url)
    ready = bool(not args.p2p or peer_announce.get("ok"))
    safe_command = redacted_command(command, {"--miner-token", "--peer-secret"})
    report = {
        "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "ok": ready,
        "mode": "join",
        "coordinator_url": coordinator_url,
        "peer_bootstrap_used": bool(p2p_bootstrap),
        "peer_count": len(peers),
        "p2p": {
            "enabled": bool(args.p2p),
            "backend": p2p_backend if args.p2p else "",
            "bootstrap": p2p_bootstrap if args.p2p else "",
            "announce": peer_announce,
            "stage_capabilities": _p2p_stage_capabilities(backend=args.backend, stage=args.stage),
            "catalog_schema": catalog_payload.get("schema") if catalog_payload else "",
        },
        "command": safe_command,
        "command_line": command_line(safe_command),
        "next_commands": _product_join_next_commands(args, coordinator_url=coordinator_url),
        "printed_only": not args.run,
        "diagnosis_codes": ["join_command_ready"] + (
            [("real_p2p_stage_miner_announce_ready" if p2p_backend == "real" else "p2p_stage_miner_announce_ready")]
            if args.p2p and peer_announce.get("ok")
            else []
        ),
        "safety": {
            "tokens_redacted_in_report": True,
            "not_production": True,
            "p2p_discovery_enabled": bool(args.p2p),
            "coordinator_result_fallback": True,
        },
    }
    if args.p2p and not peer_announce.get("ok"):
        report["diagnosis_codes"] = [
            "real_p2p_stage_miner_announce_failed" if p2p_backend == "real" else "p2p_stage_miner_announce_failed"
        ]
        report["operator_action"] = _product_join_operator_action(report)
        if not args.run:
            return sanitize(redact_values(report, [args.miner_token, args.peer_secret]))
    report["operator_action"] = _product_join_operator_action(report)
    if args.run:
        refresh = maybe_start_discovery_refresh(args, peer, bootstrap=p2p_bootstrap) if peer_announce.get("ok") else None
        try:
            completed = runner(command, cwd=str(ROOT), text=True)
        finally:
            if refresh is not None:
                report["p2p"]["refresh"] = refresh.stop()
        report["returncode"] = completed.returncode
        report["ok"] = bool(ready and completed.returncode == 0)
        report["operator_action"] = _product_join_operator_action(report)
    return sanitize(redact_values(report, [args.miner_token, args.peer_secret]))


def _product_join_operator_action(report: dict[str, Any]) -> str:
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []))
    if "p2p_discovery_unreachable" in codes:
        return "Start the P2P discovery daemon with crowdtensor p2pd --run, or use a direct --coordinator-url."
    if "coordinator_route_missing" in codes:
        return "Start the Coordinator with crowdtensor serve, or pass --coordinator-url/--peer-bootstrap for discovery."
    if "p2p_stage_miner_announce_failed" in codes or "real_p2p_stage_miner_announce_failed" in codes:
        return "Check the P2P discovery daemon and --peer-secret, then rerun join --p2p so this stage is visible to generate --dry-run."
    if report.get("ok"):
        stage_caps = ((report.get("p2p") or {}).get("stage_capabilities") or [])
        stage_text = "/".join(stage_caps) if stage_caps else "the selected stage"
        generate_hint = "crowdtensor generate --p2p --dry-run" if (report.get("p2p") or {}).get("enabled") else f"crowdtensor generate --coordinator-url {report.get('coordinator_url')} --dry-run"
        if report.get("printed_only"):
            return f"Rerun with --run to start {stage_text}; keep one stage0 and one stage1 Miner running, then run {generate_hint}."
        return f"Keep this Miner running; start the other stage if needed, then run {generate_hint}."
    if report.get("returncode") not in {None, 0}:
        return "Miner process exited; inspect its stderr/logs, fix the runtime error, and rerun join --run."
    return "Inspect the join JSON report and Miner command for the failing check."


def _product_generate_local_output_from_validation(validation: dict[str, Any], *, max_chars: int = 4096) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    raw_results = validation.get("inference_results") if isinstance(validation.get("inference_results"), list) else []
    if not raw_results and isinstance(validation.get("inference_result"), dict):
        raw_results = [validation["inference_result"]]
    if not raw_results and isinstance(validation.get("generated_text"), str) and validation.get("generated_text"):
        raw_results = [validation]
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        text = item.get("generated_text")
        if not isinstance(text, str) or not text:
            continue
        key = str(item.get("request_id") or item.get("prompt_hash") or len(outputs))
        if key in seen:
            continue
        seen.add(key)
        outputs.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "generated_token_count": item.get("generated_token_count"),
            "generated_text": text[:max_chars],
        })
    if not outputs:
        return {}
    return {
        "generated_text": str(outputs[0].get("generated_text") or ""),
        "outputs": outputs,
        "output_count": len(outputs),
        "display_only": True,
        "public_artifact_safe": False,
    }


def _empty_generate_wait_progress(*, timeout_seconds: float, poll_interval: float) -> dict[str, Any]:
    return {
        "poll_count": 0,
        "timeout_seconds": float(timeout_seconds),
        "poll_interval": float(poll_interval),
        "session_created": False,
        "ledger_endpoint_ready": False,
        "stream_endpoint_ready": False,
        "accepted_rows_seen": 0,
        "last_accepted_rows": 0,
        "max_observed_token_count": 0,
        "target_token_count": 0,
        "observed_request_count": 0,
        "expected_request_count": 0,
        "batch_generation_ready": False,
        "stream_event_count": 0,
        "last_error_type": "",
        "last_error_detail": "",
        "completion_observed": False,
        "public_artifact_safe": True,
    }


def _update_generate_wait_progress(
    progress: dict[str, Any],
    *,
    generation: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
    stream_events: list[dict[str, Any]],
    expected_request_count: int,
) -> None:
    progress["accepted_rows_seen"] = max(int(progress.get("accepted_rows_seen") or 0), len(ledger_rows))
    progress["last_accepted_rows"] = len(ledger_rows)
    progress["stream_event_count"] = len(stream_events)
    progress["target_token_count"] = int(generation.get("max_new_tokens") or progress.get("target_token_count") or 0)
    progress["max_observed_token_count"] = max(
        int(progress.get("max_observed_token_count") or 0),
        int(generation.get("generated_token_count") or 0),
    )
    progress["observed_request_count"] = max(
        int(progress.get("observed_request_count") or 0),
        int(generation.get("observed_request_count") or generation.get("request_count") or 0),
    )
    progress["expected_request_count"] = max(
        int(progress.get("expected_request_count") or 0),
        int(generation.get("expected_request_count") or expected_request_count or 0),
    )
    progress["batch_generation_ready"] = bool(progress.get("batch_generation_ready") or generation.get("batch_generation_ready"))
    progress["completion_observed"] = bool(progress.get("completion_observed") or generation.get("multi_token_generation_ready"))


def _safe_generate_error(exc: Exception) -> tuple[str, str]:
    detail = str(exc)[:200]
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if body:
            detail = body[:200]
    return type(exc).__name__, detail


def _retry_timeout_seconds(wait_progress: dict[str, Any]) -> int:
    current = _safe_int(wait_progress.get("timeout_seconds"), 120)
    return min(max(current * 2, current + 60, 120), 1800)


def _safe_stream_payload_event(event: dict[str, Any], *, max_new_tokens: int) -> dict[str, Any]:
    try:
        generated_count = int(event.get("generated_token_count") or 0)
    except (TypeError, ValueError):
        generated_count = 0
    return {
        "schema": event.get("schema") or "session_stream_event_v1",
        "request_id": event.get("request_id"),
        "prompt_hash": event.get("prompt_hash"),
        "generated_token_count": generated_count,
        "max_new_tokens": event.get("max_new_tokens") or max_new_tokens,
        "generation_step": event.get("generation_step"),
        "generated_text_hash": event.get("generated_text_hash"),
        "decoded_tokens_match": event.get("decoded_tokens_match"),
        "observed_at": event.get("observed_at"),
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def _product_generate_operator_action(report: dict[str, Any]) -> str:
    if report.get("ok"):
        if bool(report.get("dry_run")):
            ready_to_submit = report.get("ready_to_submit") if isinstance(report.get("ready_to_submit"), dict) else {}
            action = _ready_to_submit_action("Generation", ready_to_submit)
            if action:
                return action
            if ready_to_submit and ready_to_submit.get("ok") is False:
                stage_preflight = report.get("stage_preflight") if isinstance(report.get("stage_preflight"), dict) else {}
                if stage_preflight.get("checked") and not stage_preflight.get("ok"):
                    return "Dry-run request shape is valid, but stage0/stage1 Miners are not both visible; start or rejoin stage Miners before submitting."
                coordinator_ready = report.get("coordinator_ready") if isinstance(report.get("coordinator_ready"), dict) else {}
                if coordinator_ready and coordinator_ready.get("ok") is False:
                    return "Dry-run request shape is valid, but Coordinator /ready is not reachable; start the Coordinator before submitting."
            return "Rerun without --dry-run to submit the generation request."
        return ""
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []))
    detail = " ".join(str(report.get(key) or "") for key in ["detail", "error"])
    if "hf_dependencies_missing" in codes or "transformers" in detail:
        return "Install optional runtime dependencies with: python -m pip install -e '.[hf]'"
    if "p2p_discovery_unreachable" in codes:
        return "Start the P2P discovery daemon with crowdtensor p2pd --run, or pass --coordinator-url for a direct Coordinator route."
    if "coordinator_ready_failed" in codes:
        return "Coordinator route exists but /ready failed; start or restart the Coordinator and retry generate --dry-run."
    if "stage_preflight_failed" in codes:
        return "Start or rejoin distinct stage0 and stage1 Miners, then rerun generate --dry-run with --observer-token when using /state."
    if "generate_route_unavailable" in codes or "coordinator_route_missing" in codes:
        return "Start a Coordinator and distinct stage0/stage1 Miners, or pass --peer-bootstrap for discovery."
    if "admin_token_required" in codes:
        return "Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN."
    if "session_create_failed" in codes:
        return "Check Coordinator reachability, --admin-token, and /admin/inference-sessions; rerun with --dry-run first."
    if "generation_timeout" in codes:
        return _infer_wait_progress_action(report)
    return "Inspect the generated JSON report and Coordinator admin results for the failing check."


def _product_generate_next_commands(report: dict[str, Any]) -> list[dict[str, Any]]:
    route = report.get("route") if isinstance(report.get("route"), dict) else {}
    p2p = report.get("p2p") if isinstance(report.get("p2p"), dict) else {}
    session_request = report.get("session_request") if isinstance(report.get("session_request"), dict) else {}
    p2p_enabled = bool(p2p.get("enabled"))
    p2p_backend = str(p2p.get("backend") or "lite")
    swarm_id = str(p2p.get("swarm_id") or "default")
    peer_bootstrap = str(p2p.get("bootstrap") or (DEFAULT_P2P_BOOTSTRAP if p2p_enabled else ""))
    coordinator_url = str(route.get("coordinator_url") or "")
    backend = str(session_request.get("backend") or route.get("backend") or "cpu")
    hf_model_id = str(session_request.get("hf_model_id") or route.get("hf_model_id") or "sshleifer/tiny-gpt2")
    batch = session_request.get("batch") if isinstance(session_request.get("batch"), dict) else {}
    try:
        request_count = int(batch.get("request_count") or session_request.get("request_count") or 1)
    except (TypeError, ValueError):
        request_count = 1
    prompt_placeholder = "" if request_count != 1 else INFER_PROMPT_PLACEHOLDER
    prompt_texts_placeholder = INFER_BATCH_PROMPTS_PLACEHOLDER if request_count > 1 else ""
    try:
        max_new_tokens = int(session_request.get("max_new_tokens") or report.get("max_new_tokens") or 16)
    except (TypeError, ValueError):
        max_new_tokens = 16
    stream_report = report.get("stream") if isinstance(report.get("stream"), dict) else {}
    stream_requested = bool(stream_report.get("enabled") or stream_report.get("requested"))
    commands: list[dict[str, Any]] = []
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []))
    detail = " ".join(str(report.get(key) or "") for key in ["detail", "error"])
    if "hf_dependencies_missing" in codes or "transformers" in detail:
        commands.append(command_entry(
            "install Hugging Face runtime",
            ["python", "-m", "pip", "install", "-e", ".[hf]"],
        ))
    if "p2p_discovery_unreachable" in codes:
        p2pd_command = ["crowdtensor", "p2pd", "--port", str(local_coordinator_port_from_url(peer_bootstrap, default=8788))]
        if swarm_id != "default":
            p2pd_command.extend(["--swarm-id", swarm_id])
        p2pd_command.append("--run")
        commands.append(command_entry("start P2P discovery daemon", p2pd_command))
    route_missing = "coordinator_route_missing" in codes
    suggested_coordinator_url = coordinator_url or (
        "http://127.0.0.1:8787" if route_missing and not p2p_enabled else ""
    )
    route_command = _product_cli_generate_command(
        p2p=p2p_enabled,
        coordinator_url=suggested_coordinator_url or coordinator_url,
        peer_bootstrap=peer_bootstrap,
        p2p_backend=p2p_backend,
        backend=backend,
        hf_model_id=hf_model_id,
        dry_run=True,
        max_new_tokens=max_new_tokens,
        prompt_text=prompt_placeholder,
        prompt_texts=prompt_texts_placeholder,
        swarm_id=swarm_id,
        include_observer=True,
        stream=stream_requested,
    )
    check_command = (
        command_entry("check generation route", route_command, requires_env=["CROWDTENSOR_OBSERVER_TOKEN"])
        if p2p_enabled or suggested_coordinator_url or coordinator_url
        else None
    )
    submit_command = (
        command_entry(
            "submit generation",
            _product_cli_generate_command(
                p2p=p2p_enabled,
                coordinator_url=suggested_coordinator_url or coordinator_url,
                peer_bootstrap=peer_bootstrap,
                p2p_backend=p2p_backend,
                backend=backend,
                hf_model_id=hf_model_id,
                dry_run=False,
                max_new_tokens=max_new_tokens,
                prompt_text=prompt_placeholder,
                prompt_texts=prompt_texts_placeholder,
                swarm_id=swarm_id,
                stream=stream_requested,
            ),
            requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
        )
        if p2p_enabled or suggested_coordinator_url or coordinator_url
        else None
    )
    if (
        "generate_route_unavailable" in codes
        or "coordinator_route_missing" in codes
        or "coordinator_ready_failed" in codes
        or "stage_preflight_failed" in codes
    ):
        local_port = local_coordinator_port_from_url(suggested_coordinator_url or coordinator_url)
        serve_args = argparse.Namespace(
            profile="gpu-generation" if backend == "cuda" else "cpu-real-llm",
            bind_host="127.0.0.1",
            public_host="127.0.0.1",
            port=local_port,
            state_dir="state",
            hf_model_id=hf_model_id,
            hf_cache_dir="",
            lease_seconds=15.0,
            p2p=p2p_enabled,
            p2p_backend=p2p_backend,
            peer_bootstrap=peer_bootstrap or DEFAULT_P2P_BOOTSTRAP,
            swarm_id=swarm_id,
            peer_id="",
            peer_url="",
            i_understand_public_bind=False,
            admin_token="",
            miner_token="",
            observer_token="",
            peer_secret="",
        )
        commands.extend([
            command_entry("start Coordinator", _product_cli_serve_command(serve_args, include_run=True)),
            command_entry("start stage0 Miner", _product_cli_join_command(
                serve_args,
                coordinator_url=suggested_coordinator_url or coordinator_url or f"http://127.0.0.1:{local_port}",
                stage="stage0",
                miner_id="stage0-miner",
                include_run=True,
            )),
            command_entry("start stage1 Miner", _product_cli_join_command(
                serve_args,
                coordinator_url=suggested_coordinator_url or coordinator_url or f"http://127.0.0.1:{local_port}",
                stage="stage1",
                miner_id="stage1-miner",
                include_run=True,
            )),
        ])
    if check_command:
        commands.append(check_command)
    if submit_command:
        commands.append(submit_command)
    if "generation_timeout" in codes and (p2p_enabled or suggested_coordinator_url or coordinator_url):
        retry_command = (
            submit_command["command"][:]
            if isinstance(submit_command, dict) and isinstance(submit_command.get("command"), list)
            else _product_cli_generate_command(
                p2p=p2p_enabled,
                coordinator_url=suggested_coordinator_url or coordinator_url,
                peer_bootstrap=peer_bootstrap,
                p2p_backend=p2p_backend,
                backend=backend,
                hf_model_id=hf_model_id,
                dry_run=False,
                max_new_tokens=max_new_tokens,
                prompt_text=prompt_placeholder,
                prompt_texts=prompt_texts_placeholder,
                swarm_id=swarm_id,
                stream=stream_requested,
            )
        )
        wait_progress = report.get("wait_progress") if isinstance(report.get("wait_progress"), dict) else {}
        retry_command.extend(["--timeout-seconds", str(_retry_timeout_seconds(wait_progress))])
        commands.append(command_entry(
            "retry generation with longer timeout",
            retry_command,
            requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
        ))
    return commands


def _finalize_product_generate_report(report: dict[str, Any], *, admin_token: str = "") -> dict[str, Any]:
    report.setdefault("operator_action", _product_generate_operator_action(report))
    report.setdefault("next_commands", _product_generate_next_commands(report))
    return sanitize(redact_values(report, [admin_token]))


def _should_live_check_generate_coordinator(args: argparse.Namespace, coordinator_url: str) -> bool:
    del coordinator_url
    return bool(getattr(args, "coordinator_url", ""))


def _product_generate_dry_run_preflight(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    route: dict[str, Any],
    effective_coordinator_url: str,
) -> dict[str, Any]:
    route_ready = bool(route.get("usable_now") if getattr(args, "p2p", False) else route.get("coordinator_url_present"))
    skip_live_preflight = bool(getattr(args, "skip_live_preflight", False))
    should_live_check = bool(not skip_live_preflight and _should_live_check_generate_coordinator(args, effective_coordinator_url))
    coordinator_ready = (
        _coordinator_ready_preflight(effective_coordinator_url, timeout=float(getattr(args, "http_timeout", 30.0)))
        if should_live_check
        else {
            "ok": None,
            "checked": False,
            "reason": (
                "live_preflight_skipped"
                if skip_live_preflight
                else ("not_checked_for_discovered_remote_coordinator" if effective_coordinator_url else "coordinator_url_missing")
            ),
            "public_artifact_safe": True,
        }
    )
    backend = str(getattr(args, "backend", "cpu") or "cpu")
    p2p_route = bool(getattr(args, "p2p", False) or getattr(args, "peer_bootstrap", ""))
    if skip_live_preflight:
        stage_preflight = {
            "checked": False,
            "ok": None,
            "reason": "live_preflight_skipped",
            "source": "not-checked",
            "required_capabilities": required_stage_capabilities(backend=backend, stage_mode="split"),
            "public_artifact_safe": True,
        }
    elif p2p_route:
        stage_preflight = _route_stage_preflight(route, backend=backend)
    elif should_live_check and effective_coordinator_url and coordinator_ready.get("ok"):
        stage_preflight = _coordinator_stage_preflight(
            effective_coordinator_url,
            observer_token=str(getattr(args, "observer_token", "") or ""),
            backend=backend,
            timeout=float(getattr(args, "http_timeout", 30.0)),
        )
    else:
        stage_preflight = {
            "checked": False,
            "ok": None,
            "reason": "coordinator_not_ready" if effective_coordinator_url else "route_not_ready",
            "source": "not-checked",
            "required_capabilities": required_stage_capabilities(backend=backend, stage_mode="split"),
            "public_artifact_safe": True,
        }
    stage_required = bool(stage_preflight.get("checked"))
    stage_ok = bool(stage_preflight.get("ok")) if stage_required else None
    coordinator_ok = bool(coordinator_ready.get("ok")) if should_live_check else None
    live_ready = None if skip_live_preflight else bool(route_ready and (coordinator_ok is not False) and (stage_ok is not False))
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []))
    if should_live_check:
        codes.add("coordinator_ready_preflight_ready" if coordinator_ready.get("ok") else "coordinator_ready_failed")
    else:
        codes.add("coordinator_ready_preflight_skipped")
    if stage_preflight.get("checked") and stage_preflight.get("ok"):
        codes.add("stage_preflight_ready")
    elif stage_preflight.get("checked"):
        codes.add("stage_preflight_failed")
    else:
        codes.add("stage_preflight_skipped")
    ready_to_submit = _ready_to_submit_status(
        submit_ok=live_ready,
        route_ready=route_ready,
        coordinator_ok=coordinator_ok,
        coordinator_preflight_required=should_live_check,
        stage_preflight_ok=stage_ok,
        stage_preflight_required=stage_required,
        source="dry-run-preflight",
    )
    report.update({
        "ok": bool(report.get("ok") and (live_ready is not False)),
        "coordinator_ready": coordinator_ready,
        "stage_preflight": stage_preflight,
        "ready_to_submit": ready_to_submit,
        "diagnosis_codes": sorted(codes),
    })
    return report


def _coordinator_ready_preflight(base_url: str, *, timeout: float) -> dict[str, Any]:
    if not base_url:
        return {"ok": False, "error": "coordinator_url_missing"}
    try:
        payload = request_json_url("GET", base_url, "/ready", admin_token="", timeout=timeout)
    except Exception as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "detail": str(exc)[:200],
        }
    return {
        "ok": bool(payload),
        "schema": payload.get("schema") if isinstance(payload, dict) else "",
        "service": payload.get("service") if isinstance(payload, dict) else "",
        "protocol": payload.get("protocol") if isinstance(payload, dict) else "",
        "auth": payload.get("auth") if isinstance(payload, dict) and isinstance(payload.get("auth"), dict) else {},
        "task_lanes": payload.get("task_lanes") if isinstance(payload, dict) and isinstance(payload.get("task_lanes"), list) else [],
    }


def _stage_preflight_capabilities(capabilities: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in [
        "real_llm_sharded_stage_capabilities",
        "capabilities",
        "supported_capabilities",
        "supported_workloads",
    ]:
        raw = capabilities.get(key)
        if isinstance(raw, list):
            values.update(str(item) for item in raw if str(item))
    return values


def _stage_preflight_from_state(state: dict[str, Any], *, backend: str) -> dict[str, Any]:
    required = required_stage_capabilities(backend=backend, stage_mode="split")
    profiles = state.get("miner_profiles") if isinstance(state.get("miner_profiles"), dict) else {}
    matched: dict[str, str] = {}
    miner_capabilities: list[dict[str, Any]] = []
    for miner_id, profile in sorted(profiles.items()):
        if not isinstance(profile, dict):
            continue
        capabilities = profile.get("last_capabilities") if isinstance(profile.get("last_capabilities"), dict) else {}
        advertised = _stage_preflight_capabilities(capabilities)
        matched_for_miner = sorted(capability for capability in required if capability in advertised)
        if not matched_for_miner:
            continue
        miner_capabilities.append({
            "miner_id": str(miner_id),
            "matched_capabilities": matched_for_miner,
            "backend": capabilities.get("backend") or profile.get("backend") or "",
            "runtime": capabilities.get("runtime") or profile.get("runtime") or "",
        })
        for capability in matched_for_miner:
            matched.setdefault(capability, str(miner_id))
    missing = [capability for capability in required if capability not in matched]
    distinct_stage_miners = len({miner_id for miner_id in matched.values() if miner_id}) >= len(required)
    ok = bool(not missing and distinct_stage_miners)
    return {
        "checked": True,
        "ok": ok,
        "source": "coordinator-state",
        "required_capabilities": required,
        "matched_capabilities": matched,
        "missing_capabilities": missing,
        "miner_count": len(profiles),
        "matched_miner_count": len(miner_capabilities),
        "matched_miners": miner_capabilities,
        "distinct_stage_miners": distinct_stage_miners,
        "public_artifact_safe": True,
    }


def _coordinator_stage_preflight(
    base_url: str,
    *,
    observer_token: str,
    backend: str,
    timeout: float,
) -> dict[str, Any]:
    if not base_url:
        return {"checked": False, "ok": False, "reason": "coordinator_url_missing", "source": "not-checked"}
    if not observer_token:
        return {
            "checked": False,
            "ok": None,
            "reason": "observer_token_missing",
            "source": "not-checked",
            "required_capabilities": required_stage_capabilities(backend=backend, stage_mode="split"),
            "public_artifact_safe": True,
        }
    try:
        payload = request_json_url("GET", base_url, "/state", observer_token=observer_token, timeout=timeout)
    except Exception as exc:
        return {
            "checked": True,
            "ok": False,
            "source": "coordinator-state",
            "error": type(exc).__name__,
            "detail": str(exc)[:200],
            "required_capabilities": required_stage_capabilities(backend=backend, stage_mode="split"),
            "public_artifact_safe": True,
        }
    return _stage_preflight_from_state(payload if isinstance(payload, dict) else {}, backend=backend)


def _route_stage_preflight(route: dict[str, Any], *, backend: str) -> dict[str, Any]:
    required = [str(item) for item in (route.get("required_capabilities") or []) if str(item)]
    if not required:
        required = required_stage_capabilities(backend=backend, stage_mode="split")
    matched = route.get("matched_capabilities") if isinstance(route.get("matched_capabilities"), dict) else {}
    safe_matched = {
        str(capability): str(miner_id)
        for capability, miner_id in matched.items()
        if str(capability) in set(required) and str(miner_id)
    }
    if not safe_matched:
        matched_peers = [str(peer_id) for peer_id in (route.get("matched_peers") or []) if str(peer_id)]
        if len(matched_peers) >= len(required):
            safe_matched = {
                capability: matched_peers[index]
                for index, capability in enumerate(required)
            }
    missing = [capability for capability in required if capability not in safe_matched]
    distinct_stage_miners = len({miner_id for miner_id in safe_matched.values() if miner_id}) >= len(required)
    ok = bool(not missing and distinct_stage_miners)
    return {
        "checked": True,
        "ok": ok,
        "source": "p2p-route",
        "required_capabilities": required,
        "matched_capabilities": safe_matched,
        "missing_capabilities": missing,
        "miner_count": None,
        "matched_miner_count": len({miner_id for miner_id in safe_matched.values() if miner_id}),
        "matched_miners": [
            {"miner_id": miner_id, "matched_capabilities": sorted([
                capability for capability, matched_miner in safe_matched.items() if matched_miner == miner_id
            ])}
            for miner_id in sorted({miner_id for miner_id in safe_matched.values() if miner_id})
        ],
        "distinct_stage_miners": distinct_stage_miners,
        "public_artifact_safe": True,
    }


def _attach_infer_existing_preflight(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    route = payload.get("route") if isinstance(payload.get("route"), dict) else {}
    route_ready = bool(route.get("usable_now") or route.get("coordinator_url_present"))
    coordinator_url = str(route.get("coordinator_url") or getattr(args, "coordinator_url", "") or "")
    timeout = float(getattr(args, "http_timeout", 30.0))
    coordinator_ready = _coordinator_ready_preflight(coordinator_url, timeout=timeout) if route_ready else {}
    p2p_route = bool(getattr(args, "p2p", False) or getattr(args, "peer_bootstrap", ""))
    backend = str(getattr(args, "backend", "cpu") or "cpu")
    if p2p_route and route_ready:
        stage_preflight = _route_stage_preflight(route, backend=backend)
    elif route_ready and coordinator_ready.get("ok"):
        stage_preflight = _coordinator_stage_preflight(
            coordinator_url,
            observer_token=str(getattr(args, "observer_token", "") or ""),
            backend=backend,
            timeout=timeout,
        )
    else:
        stage_preflight = {
            "checked": False,
            "ok": None,
            "reason": "coordinator_not_ready" if route_ready else "route_not_ready",
            "source": "not-checked",
            "public_artifact_safe": True,
        }
    codes = list(payload.get("diagnosis_codes") or [])
    if route_ready and coordinator_ready.get("ok"):
        codes.append("coordinator_ready_preflight_ready")
    elif route_ready:
        codes.append("coordinator_ready_failed")
    if stage_preflight.get("checked") and stage_preflight.get("ok"):
        codes.append("stage_preflight_ready")
    elif stage_preflight.get("checked"):
        codes.append("stage_preflight_failed")
    else:
        codes.append("stage_preflight_skipped")
    stage_required = bool(stage_preflight.get("checked"))
    stage_ok = bool(stage_preflight.get("ok")) if stage_required else True
    ready_to_submit = _ready_to_submit_status(
        submit_ok=bool(route_ready and coordinator_ready.get("ok") and stage_ok),
        route_ready=route_ready,
        coordinator_ok=bool(coordinator_ready.get("ok")) if route_ready else None,
        coordinator_preflight_required=route_ready,
        stage_preflight_ok=bool(stage_preflight.get("ok")) if stage_required else None,
        stage_preflight_required=stage_required,
        source="infer-existing-preflight",
    )
    merged = {
        **payload,
        "coordinator_ready": coordinator_ready,
        "stage_preflight": stage_preflight,
        "ready_to_submit": ready_to_submit,
        "diagnosis_codes": sorted(set(str(code) for code in codes)),
    }
    merged["ok"] = bool(payload.get("ok") and route_ready and coordinator_ready.get("ok") and stage_ok)
    return merged


def build_product_generate(args: argparse.Namespace) -> dict[str, Any]:
    prompt_texts = parse_prompt_texts_arg(args.prompt_text, getattr(args, "prompt_texts", ""))
    prompt_text = prompt_texts[0]
    batch_enabled = len(prompt_texts) > 1
    p2p_backend = _p2p_backend(args)
    stream_enabled = bool(getattr(args, "stream", False))
    session_request = build_session_request(
        prompt_text=prompt_text,
        prompt_texts=prompt_texts,
        backend=args.backend,
        hf_model_id=args.hf_model_id,
        stage_mode="split",
        max_new_tokens=args.max_new_tokens,
        scenario_id=args.scenario_id,
        route_source=_p2p_route_source(args),
    )
    coordinator_url = args.coordinator_url
    p2p_bootstrap = args.peer_bootstrap or (DEFAULT_P2P_BOOTSTRAP if args.p2p else "")
    peers: list[dict[str, Any]] = []
    catalog_payload: dict[str, Any] = {}
    route_lookup_payload: dict[str, Any] = {}
    discovery_error: dict[str, Any] = {}
    if p2p_bootstrap:
        try:
            resolved_url, peers, catalog_payload = resolve_coordinator_from_discovery(
                p2p_bootstrap,
                timeout=args.http_timeout,
                backend=p2p_backend if args.p2p else "lite",
                session_request=session_request,
            )
            coordinator_url = coordinator_url or resolved_url
        except Exception as exc:
            discovery_error = {
                "ok": False,
                "error": type(exc).__name__,
                "detail": str(exc)[:200],
                "bootstrap": p2p_bootstrap,
            }
        if args.p2p and p2p_backend == "real":
            try:
                route_lookup_payload = post_route_lookup(
                    p2p_bootstrap,
                    session_request,
                    coordinator_url=coordinator_url,
                    timeout=args.http_timeout,
                )
            except Exception as exc:
                route_lookup_payload = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200]}
    route = build_route_decision(session_request, coordinator_url=coordinator_url, peer_catalog=peers)
    if route_lookup_payload.get("route"):
        route = route_lookup_payload["route"]
    effective_coordinator_url = str(route.get("coordinator_url") or coordinator_url or "")
    route_ready_code = "real_p2p_generate_route_ready" if p2p_backend == "real" else "p2p_generate_route_ready"
    if args.dry_run:
        report = {
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": bool(route.get("usable_now") if args.p2p else route.get("coordinator_url_present")),
            "mode": "generate",
            "dry_run": True,
            "session_request": session_request,
            "batch": session_request.get("batch"),
            "route": route,
            "p2p": {
                "enabled": bool(args.p2p),
                "backend": p2p_backend if args.p2p else "",
                "bootstrap": p2p_bootstrap if args.p2p else "",
                "swarm_id": str(getattr(args, "swarm_id", "default") or "default") if args.p2p else "",
                "peer_count": len(peers),
                "catalog_schema": catalog_payload.get("schema") if catalog_payload else "",
                "route_lookup_schema": route_lookup_payload.get("schema") if route_lookup_payload else "",
                "discovery": discovery_error,
            },
            "diagnosis_codes": (
                ["p2p_discovery_unreachable", "coordinator_route_missing"]
                if args.p2p and discovery_error
                else [
                    (
                        route_ready_code
                        if route.get("usable_now")
                        else ("coordinator_route_missing" if not route.get("coordinator_url_present") else "stage_capability_missing")
                    )
                    if args.p2p
                    else ("generate_dry_run_ready" if route.get("coordinator_url_present") else "coordinator_route_missing")
                ]
            ),
        }
        report = _product_generate_dry_run_preflight(
            report,
            args,
            route=route,
            effective_coordinator_url=effective_coordinator_url,
        )
        return _finalize_product_generate_report(report)
    if args.p2p and not route.get("usable_now"):
        return _finalize_product_generate_report({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "generate",
            "session_request": session_request,
            "batch": session_request.get("batch"),
            "route": route,
            "p2p": {
                "enabled": True,
                "backend": p2p_backend,
                "bootstrap": p2p_bootstrap,
                "swarm_id": str(getattr(args, "swarm_id", "default") or "default"),
                "peer_count": len(peers),
                "catalog_schema": catalog_payload.get("schema") if catalog_payload else "",
                "route_lookup_schema": route_lookup_payload.get("schema") if route_lookup_payload else "",
                "discovery": discovery_error,
            },
            "diagnosis_codes": (
                ["p2p_discovery_unreachable", "coordinator_route_missing"]
                if discovery_error
                else ["generate_route_unavailable"]
            ),
        })
    if not effective_coordinator_url:
        return _finalize_product_generate_report({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "generate",
            "session_request": session_request,
            "batch": session_request.get("batch"),
            "route": route,
            "diagnosis_codes": ["coordinator_route_missing"],
        })
    if not args.admin_token:
        return _finalize_product_generate_report({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "generate",
            "session_request": session_request,
            "batch": session_request.get("batch"),
            "route": route,
            "diagnosis_codes": ["admin_token_required"],
        })
    private_payload = coordinator_payload_for_request(session_request, prompt_text=prompt_text, prompt_texts=prompt_texts)
    session_create_timeout = max(float(args.http_timeout), min(float(args.timeout_seconds), 30.0))
    try:
        session = request_json_url(
            "POST",
            effective_coordinator_url,
            "/admin/inference-sessions",
            private_payload,
            admin_token=args.admin_token,
            timeout=session_create_timeout,
        )
    except Exception as exc:
        detail = str(exc)[:240]
        diagnosis = ["session_create_failed"]
        if isinstance(exc, HTTPError):
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            if body:
                detail = body[:240]
            if "requires optional Hugging Face dependencies" in body or "transformers" in body:
                diagnosis.append("hf_dependencies_missing")
                detail = "real_llm_sharded_infer requires optional Hugging Face dependencies; install with python -m pip install -e '.[hf]'"
        elif "requires optional Hugging Face dependencies" in str(exc) or "transformers" in str(exc):
            diagnosis.append("hf_dependencies_missing")
            detail = "real_llm_sharded_infer requires optional Hugging Face dependencies; install with python -m pip install -e '.[hf]'"
        return _finalize_product_generate_report({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "generate",
            "session_request": session_request,
            "batch": session_request.get("batch"),
            "route": route,
            "diagnosis_codes": diagnosis,
            "error": type(exc).__name__,
            "detail": detail,
        }, admin_token=args.admin_token)
    result_row: dict[str, Any] | None = None
    stream_events: list[dict[str, Any]] = []
    stream_seen_keys: set[tuple[str, int]] = set()
    stream_source = "disabled" if not stream_enabled else "admin-session-stream"
    stream_endpoint_ready = False
    last_ledger_rows: list[dict[str, Any]] = []
    wait_progress = _empty_generate_wait_progress(timeout_seconds=args.timeout_seconds, poll_interval=args.poll_interval)
    wait_progress.update({
        "session_created": bool(session.get("session_id")),
        "target_token_count": int(args.max_new_tokens),
        "expected_request_count": len(prompt_texts),
    })
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() <= deadline:
        wait_progress["poll_count"] = int(wait_progress.get("poll_count") or 0) + 1
        session_id = str(session.get("session_id") or "")
        stream_payload: dict[str, Any] = {}
        stream_endpoint_ready_this_poll = False
        if stream_enabled and session_id:
            stream_query = urlencode({
                "session_id": session_id,
                "workload_type": "real_llm_sharded_infer",
                "limit": args.admin_results_limit,
                "max_new_tokens": args.max_new_tokens,
            })
            try:
                stream_payload = request_json_url(
                    "GET",
                    effective_coordinator_url,
                    f"/admin/session-stream?{stream_query}",
                    admin_token=args.admin_token,
                    timeout=args.http_timeout,
                )
            except HTTPError as exc:
                wait_progress["last_error_type"], wait_progress["last_error_detail"] = _safe_generate_error(exc)
                if exc.code == 404:
                    stream_source = "admin-results-ledger-fallback"
                else:
                    stream_payload = {}
            except Exception as exc:
                wait_progress["last_error_type"], wait_progress["last_error_detail"] = _safe_generate_error(exc)
                stream_payload = {}
            else:
                stream_endpoint_ready_this_poll = bool(stream_payload.get("schema") == "admin_session_stream_v1")
                if stream_endpoint_ready_this_poll:
                    stream_endpoint_ready = True
                    wait_progress["stream_endpoint_ready"] = True
                    stream_source = "admin-session-stream"
                    payload_events = stream_payload.get("events") if isinstance(stream_payload, dict) else []
                    if isinstance(payload_events, list):
                        for event in payload_events:
                            if not isinstance(event, dict):
                                continue
                            safe_event = _safe_stream_payload_event(event, max_new_tokens=args.max_new_tokens)
                            generated_count = int(safe_event.get("generated_token_count") or 0)
                            request_key = str(safe_event.get("request_id") or safe_event.get("prompt_hash") or "")
                            stream_key = (request_key, generated_count)
                            if generated_count > 0 and stream_key not in stream_seen_keys:
                                stream_events.append(safe_event)
                                stream_seen_keys.add(stream_key)
                                if not args.json:
                                    print(
                                        "stream "
                                        f"{safe_event.get('generated_token_count')}/{safe_event.get('max_new_tokens')} "
                                        f"hash={safe_event.get('generated_text_hash')}",
                                        flush=True,
                                    )
        query = f"/admin/results?status=accepted&workload_type=real_llm_sharded_infer&limit={args.admin_results_limit}"
        if session_id:
            query += f"&session_id={session_id}"
        try:
            ledger = request_json_url("GET", effective_coordinator_url, query, admin_token=args.admin_token, timeout=args.http_timeout)
        except Exception as exc:
            wait_progress["last_error_type"], wait_progress["last_error_detail"] = _safe_generate_error(exc)
            ledger = {}
        else:
            wait_progress["ledger_endpoint_ready"] = True
        rows = ledger.get("results") if isinstance(ledger, dict) else []
        if isinstance(rows, list):
            ledger_rows = [row for row in rows if isinstance(row, dict)]
            last_ledger_rows = ledger_rows
            best_generation = safe_generation_summary(ledger_rows[0], max_new_tokens=args.max_new_tokens) if ledger_rows else {}
            for row in ledger_rows[1:]:
                candidate = safe_generation_summary(row, max_new_tokens=args.max_new_tokens)
                if int(candidate.get("generated_token_count") or 0) > int(best_generation.get("generated_token_count") or 0):
                    best_generation = candidate
            _update_generate_wait_progress(
                wait_progress,
                generation=best_generation,
                ledger_rows=ledger_rows,
                stream_events=stream_events,
                expected_request_count=len(prompt_texts),
            )
            if stream_enabled and not stream_endpoint_ready_this_poll:
                def _stream_sort_key(row: dict[str, Any]) -> tuple[int, int]:
                    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
                    return (
                        int(validation.get("generated_token_count") or 0),
                        int(row.get("event_index") or 0),
                    )

                for row in sorted(ledger_rows, key=_stream_sort_key):
                    for event in safe_stream_events(
                        row,
                        max_new_tokens=args.max_new_tokens,
                        observed_at=row.get("terminal_at"),
                    ):
                        generated_count = int(event.get("generated_token_count") or 0)
                        request_key = str(event.get("request_id") or event.get("prompt_hash") or "")
                        stream_key = (request_key, generated_count)
                        if generated_count <= 0 or stream_key in stream_seen_keys:
                            continue
                        stream_events.append(event)
                        stream_seen_keys.add(stream_key)
                        stream_source = (
                            "admin-session-stream-with-ledger-fallback"
                            if stream_endpoint_ready
                            else "admin-results-ledger-fallback"
                        )
                        if not args.json:
                            print(
                                "stream "
                                f"{event.get('generated_token_count')}/{event.get('max_new_tokens')} "
                                f"hash={event.get('generated_text_hash')}",
                                flush=True,
                            )
            for row in ledger_rows:
                row_generation = safe_generation_summary(row, max_new_tokens=args.max_new_tokens)
                if (
                    row_generation.get("multi_token_generation_ready")
                    and int(row_generation.get("request_count") or 1) >= len(prompt_texts)
                    and (not batch_enabled or row_generation.get("batch_generation_ready"))
                ):
                    result_row = row
                    break
        if result_row:
            break
        time.sleep(args.poll_interval)
    generation = safe_generation_summary(result_row or {}, max_new_tokens=args.max_new_tokens)
    batch_ready = bool(
        result_row
        and generation.get("request_count") == len(prompt_texts)
        and (not batch_enabled or generation.get("batch_generation_ready"))
    )
    ok = bool(result_row and generation.get("multi_token_generation_ready") and batch_ready)
    stream_progress = stream_progress_summary(
        stream_events,
        max_new_tokens=args.max_new_tokens,
        source=stream_source,
        expected_request_count=len(prompt_texts),
    )
    stream_ready = bool(
        stream_enabled
        and ok
        and (
            (
                stream_progress.get("stream_progress_complete")
                and stream_progress.get("monotonic_progress")
            )
            if not batch_enabled
            else (
                stream_progress.get("per_request_progress_complete")
                and stream_progress.get("per_request_monotonic_progress")
            )
        )
    )
    _update_generate_wait_progress(
        wait_progress,
        generation=generation,
        ledger_rows=last_ledger_rows,
        stream_events=stream_events,
        expected_request_count=len(prompt_texts),
    )
    report = {
        "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "ok": ok,
        "mode": "generate",
        "dry_run": False,
        "session_request": session_request,
        "batch": {
            "enabled": batch_enabled,
            "request_count": len(prompt_texts),
            "expected_request_count": len(prompt_texts),
            "observed_request_count": int(generation.get("observed_request_count") or generation.get("request_count") or 0),
            "prompt_hashes": session_request.get("prompt_hashes") or [],
            "prompt_char_counts": session_request.get("prompt_char_counts") or [],
            "batch_generation_ready": batch_ready,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "route": route,
        "p2p": {
            "enabled": bool(args.p2p),
            "backend": p2p_backend if args.p2p else "",
            "bootstrap": p2p_bootstrap if args.p2p else "",
            "swarm_id": str(getattr(args, "swarm_id", "default") or "default") if args.p2p else "",
            "peer_count": len(peers),
            "catalog_schema": catalog_payload.get("schema") if catalog_payload else "",
            "route_lookup_schema": route_lookup_payload.get("schema") if route_lookup_payload else "",
        },
        "session": {
            "schema": session.get("schema"),
            "session_id": session.get("session_id"),
            "workload_type": session.get("workload_type"),
            "max_new_tokens": session.get("max_new_tokens"),
            "backend": session.get("backend"),
            "hf_model_id": session.get("model_id") or args.hf_model_id,
        },
        "wait_progress": wait_progress,
        "generation": generation,
        "stream": {
            "enabled": stream_enabled,
            "event_count": len(stream_events),
            "events": stream_events,
            "source": stream_source,
            "endpoint_ready": stream_endpoint_ready,
            "stream_generation_ready": stream_ready,
            "progress": stream_progress,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "diagnosis_codes": (
            ["public_swarm_generate_ready"]
            + (["public_swarm_generate_batch_ready"] if batch_ready and batch_enabled else [])
            + (["public_swarm_generate_stream_ready"] if stream_ready else [])
            + (["public_swarm_generate_stream_endpoint_ready"] if stream_ready and stream_endpoint_ready else [])
            + ([route_ready_code] if args.p2p and route.get("usable_now") else [])
            if ok
            else ["generation_timeout"]
        ),
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "not_production": True,
            "p2p_discovery_enabled": bool(args.p2p),
            "coordinator_result_fallback": True,
        },
    }
    if (args.include_output or not args.json) and result_row and not args.json:
        validation = result_row.get("validation") if isinstance(result_row.get("validation"), dict) else {}
        local_output = _product_generate_local_output_from_validation(validation)
        if local_output:
            report["local_output"] = local_output
        report["local_output_note"] = "Raw generated text is shown only in local human output; JSON and public artifacts expose hashes only."
    elif args.include_output:
        report["local_output_note"] = "Raw generated text is suppressed in JSON/public output; rerun without --json for local display."
    return _finalize_product_generate_report(report, admin_token=args.admin_token)


def print_product_generate(report: dict[str, Any]) -> None:
    print("CrowdTensor generate")
    print(f"  ok: {report.get('ok')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    session = report.get("session") if isinstance(report.get("session"), dict) else {}
    if session:
        print(
            "  session: "
            f"{session.get('session_id')} "
            f"workload={session.get('workload_type')} "
            f"backend={session.get('backend')}"
        )
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    if generation:
        print(
            "  generation: "
            f"{generation.get('generated_token_count')}/{generation.get('max_new_tokens')} "
            f"hash={generation.get('generated_text_hash')}"
        )
    batch = report.get("batch") if isinstance(report.get("batch"), dict) else {}
    if batch.get("enabled"):
        print(
            "  batch: "
            f"requests={batch.get('request_count')} "
            f"ready={batch.get('batch_generation_ready')}"
        )
    stream = report.get("stream") if isinstance(report.get("stream"), dict) else {}
    if stream.get("enabled"):
        progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
        print(
            "  stream_events: "
            f"{stream.get('event_count')} "
            f"source={stream.get('source')} "
            f"complete={progress.get('stream_progress_complete')}"
        )
    wait_progress = report.get("wait_progress") if isinstance(report.get("wait_progress"), dict) else {}
    if wait_progress:
        print(
            "  wait: "
            f"polls={wait_progress.get('poll_count')} "
            f"accepted_rows={wait_progress.get('accepted_rows_seen')} "
            f"tokens={wait_progress.get('max_observed_token_count')}/{wait_progress.get('target_token_count')} "
            f"ledger={wait_progress.get('ledger_endpoint_ready')} "
            f"stream={wait_progress.get('stream_endpoint_ready')}"
        )
    local_output = report.get("local_output") if isinstance(report.get("local_output"), dict) else {}
    outputs = local_output.get("outputs") if isinstance(local_output.get("outputs"), list) else []
    if len(outputs) <= 1 and local_output.get("generated_text"):
        print(f"  output: {local_output.get('generated_text')}")
    elif outputs:
        for index, item in enumerate(outputs, start=1):
            if isinstance(item, dict) and item.get("generated_text"):
                print(f"  output[{index}]: {item.get('generated_text')}")
    route = report.get("route") if isinstance(report.get("route"), dict) else {}
    if route:
        missing = route.get("missing_capabilities") or []
        print(
            "  route: "
            f"source={route.get('route_source')} "
            f"coordinator={route.get('coordinator_url_present')} "
            f"missing={','.join(missing) if missing else 'none'}"
        )
    p2p = report.get("p2p") if isinstance(report.get("p2p"), dict) else {}
    if p2p:
        print(f"  p2p: {format_p2p_status(p2p)}")
    coordinator_ready = report.get("coordinator_ready") if isinstance(report.get("coordinator_ready"), dict) else {}
    if coordinator_ready:
        print(
            "  coordinator_ready: "
            f"{coordinator_ready.get('ok')} "
            f"service={coordinator_ready.get('service')} "
            f"protocol={coordinator_ready.get('protocol')}"
        )
    stage_preflight = report.get("stage_preflight") if isinstance(report.get("stage_preflight"), dict) else {}
    if stage_preflight:
        missing = stage_preflight.get("missing_capabilities") if isinstance(stage_preflight.get("missing_capabilities"), list) else []
        print(
            "  stage_preflight: "
            f"checked={stage_preflight.get('checked')} "
            f"ok={stage_preflight.get('ok')} "
            f"matched_miners={stage_preflight.get('matched_miner_count')} "
            f"missing={','.join(str(item) for item in missing) if missing else 'none'}"
        )
    ready_to_submit = report.get("ready_to_submit") if isinstance(report.get("ready_to_submit"), dict) else {}
    if ready_to_submit:
        print(
            "  ready_to_submit: "
            f"{ready_to_submit.get('ok')} "
            f"label={ready_to_submit.get('readiness_label')} "
            f"fully_verified={ready_to_submit.get('fully_verified')} "
            f"route={ready_to_submit.get('route_ready')} "
            f"coordinator={ready_to_submit.get('coordinator_ready')} "
            f"stage={ready_to_submit.get('stage_preflight_ok')} "
            f"stage_verification={ready_to_submit.get('stage_verification')}"
        )
        if ready_to_submit.get("readiness_summary"):
            print(f"  readiness: {ready_to_submit.get('readiness_summary')}")
    if report.get("local_output_note"):
        print(f"  note: {report.get('local_output_note')}")
    if report.get("operator_action"):
        print(f"  action: {report.get('operator_action')}")
    for index, item in enumerate(report.get("next_commands") or [], start=1):
        if isinstance(item, dict) and item.get("command_line"):
            requires_env = item.get("requires_env") if isinstance(item.get("requires_env"), list) else []
            suffix = f"  # requires {', '.join(str(name) for name in requires_env)}" if requires_env else ""
            print(f"  next[{index}] {item.get('label')}: {local_generate_command_line(item, report)}{suffix}")


def print_product_serve(report: dict[str, Any]) -> None:
    print("CrowdTensor serve")
    print(f"  ok: {report.get('ok')}")
    print(f"  profile: {report.get('profile')}")
    print(f"  coordinator_url: {report.get('coordinator_url')}")
    p2p = report.get("p2p") if isinstance(report.get("p2p"), dict) else {}
    if p2p:
        announce = p2p.get("announce") if isinstance(p2p.get("announce"), dict) else {}
        print(
            "  p2p: "
            f"enabled={p2p.get('enabled')} "
            f"backend={p2p.get('backend')} "
            f"announced={announce.get('ok')}"
        )
    if report.get("printed_only"):
        print(f"  command: {report.get('command_line') or command_line(report.get('command') or [])}")
    if report.get("returncode") is not None:
        print(f"  returncode: {report.get('returncode')}")
    if report.get("operator_action"):
        print(f"  action: {report.get('operator_action')}")
    for index, item in enumerate(report.get("next_commands") or [], start=1):
        if isinstance(item, dict) and item.get("command_line"):
            requires_env = item.get("requires_env") if isinstance(item.get("requires_env"), list) else []
            suffix = f"  # requires {', '.join(str(name) for name in requires_env)}" if requires_env else ""
            print(f"  next[{index}] {item.get('label')}: {item.get('command_line')}{suffix}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")


def print_product_join(report: dict[str, Any]) -> None:
    print("CrowdTensor join")
    print(f"  ok: {report.get('ok')}")
    print(f"  coordinator_url: {report.get('coordinator_url')}")
    p2p = report.get("p2p") if isinstance(report.get("p2p"), dict) else {}
    if p2p:
        announce = p2p.get("announce") if isinstance(p2p.get("announce"), dict) else {}
        caps = p2p.get("stage_capabilities") if isinstance(p2p.get("stage_capabilities"), list) else []
        print(
            "  p2p: "
            f"enabled={p2p.get('enabled')} "
            f"backend={p2p.get('backend')} "
            f"announced={announce.get('ok')} "
            f"stage_caps={','.join(str(item) for item in caps) if caps else 'none'}"
        )
    if report.get("printed_only"):
        print(f"  command: {report.get('command_line') or command_line(report.get('command') or [])}")
    if report.get("returncode") is not None:
        print(f"  returncode: {report.get('returncode')}")
    if report.get("operator_action"):
        print(f"  action: {report.get('operator_action')}")
    for index, item in enumerate(report.get("next_commands") or [], start=1):
        if isinstance(item, dict) and item.get("command_line"):
            requires_env = item.get("requires_env") if isinstance(item.get("requires_env"), list) else []
            suffix = f"  # requires {', '.join(str(name) for name in requires_env)}" if requires_env else ""
            print(f"  next[{index}] {item.get('label')}: {item.get('command_line')}{suffix}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")


def print_infer(report: dict[str, Any]) -> None:
    print("CrowdTensor infer")
    print(f"  ok: {report.get('ok')}")
    print(f"  mode: {report.get('mode')}")
    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    print(f"  model: {model.get('hf_model_id')} backend={model.get('backend')}")
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    print(
        "  generation: "
        f"{generation.get('generated_token_count')}/{generation.get('max_new_tokens')} "
        f"hash={generation.get('generated_text_hash')}"
    )
    route = report.get("route") if isinstance(report.get("route"), dict) else {}
    print(
        "  route: "
        f"source={route.get('route_source')} "
        f"ready={route.get('route_ready')} "
        f"distinct_stage_miners={route.get('distinct_stage_miners')}"
    )
    p2p = report.get("p2p") if isinstance(report.get("p2p"), dict) else {}
    if p2p:
        print(f"  p2p: {format_p2p_status(p2p)}")
    coordinator_ready = report.get("coordinator_ready") if isinstance(report.get("coordinator_ready"), dict) else {}
    if coordinator_ready:
        print(
            "  coordinator_ready: "
            f"{coordinator_ready.get('ok')} "
            f"service={coordinator_ready.get('service')} "
            f"protocol={coordinator_ready.get('protocol')}"
        )
    stage_preflight = report.get("stage_preflight") if isinstance(report.get("stage_preflight"), dict) else {}
    if stage_preflight:
        missing = stage_preflight.get("missing_capabilities") if isinstance(stage_preflight.get("missing_capabilities"), list) else []
        print(
            "  stage_preflight: "
            f"checked={stage_preflight.get('checked')} "
            f"ok={stage_preflight.get('ok')} "
            f"matched_miners={stage_preflight.get('matched_miner_count')} "
            f"missing={','.join(str(item) for item in missing) if missing else 'none'}"
        )
    ready_to_submit = report.get("ready_to_submit") if isinstance(report.get("ready_to_submit"), dict) else {}
    if ready_to_submit:
        print(
            "  ready_to_submit: "
            f"{ready_to_submit.get('ok')} "
            f"label={ready_to_submit.get('readiness_label')} "
            f"fully_verified={ready_to_submit.get('fully_verified')} "
            f"route={ready_to_submit.get('route_ready')} "
            f"coordinator={ready_to_submit.get('coordinator_ready')} "
            f"stage={ready_to_submit.get('stage_preflight_ok')} "
            f"stage_verification={ready_to_submit.get('stage_verification')}"
        )
        if ready_to_submit.get("readiness_summary"):
            print(f"  readiness: {ready_to_submit.get('readiness_summary')}")
    stream = report.get("stream") if isinstance(report.get("stream"), dict) else {}
    if stream.get("enabled"):
        print(f"  stream: ready={stream.get('ready')} events={stream.get('event_count')} source={stream.get('source')}")
        progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
        per_request = progress.get("per_request_progress") if isinstance(progress.get("per_request_progress"), list) else []
        if len(per_request) > 1:
            for index, item in enumerate(per_request, start=1):
                if not isinstance(item, dict):
                    continue
                counts = item.get("observed_token_counts") or []
                print(
                    f"  stream[{index}]: "
                    f"counts={counts} "
                    f"complete={item.get('stream_progress_complete')}"
                )
        elif progress:
            print(
                "  stream_progress: "
                f"counts={progress.get('observed_token_counts') or []} "
                f"complete={progress.get('stream_progress_complete')}"
            )
    wait_progress = report.get("wait_progress") if isinstance(report.get("wait_progress"), dict) else {}
    if wait_progress:
        print(
            "  wait: "
            f"polls={wait_progress.get('poll_count')} "
            f"accepted_rows={wait_progress.get('accepted_rows_seen')} "
            f"tokens={wait_progress.get('max_observed_token_count')}/{wait_progress.get('target_token_count')} "
            f"ledger={wait_progress.get('ledger_endpoint_ready')} "
            f"stream={wait_progress.get('stream_endpoint_ready')}"
        )
    local_output = report.get("local_output") if isinstance(report.get("local_output"), dict) else {}
    outputs = local_output.get("outputs") if isinstance(local_output.get("outputs"), list) else []
    if len(outputs) <= 1 and local_output.get("generated_text"):
        print(f"  output: {local_output.get('generated_text')}")
    if len(outputs) > 1:
        for index, item in enumerate(outputs, start=1):
            if isinstance(item, dict) and item.get("generated_text"):
                print(f"  output[{index}]: {item.get('generated_text')}")
    if local_output.get("note"):
        print(f"  note: {local_output.get('note')}")
    print(f"  output_dir: {report.get('output_dir')}")
    if report.get("operator_action"):
        print(f"  action: {report.get('operator_action')}")
    for index, item in enumerate(report.get("next_commands") or [], start=1):
        if isinstance(item, dict) and item.get("command_line"):
            requires_env = item.get("requires_env") if isinstance(item.get("requires_env"), list) else []
            suffix = f"  # requires {', '.join(str(name) for name in requires_env)}" if requires_env else ""
            print(f"  next[{index}] {item.get('label')}: {local_infer_command_line(item, report)}{suffix}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")


def build_peer_cli(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    action = args.peer_action
    if action == "check":
        command = [sys.executable, str(SCRIPTS_DIR / "p2p_lite_discovery_check.py"), "--json"]
        step, payload = run_json_step("p2p_lite_discovery_check", command, runner=runner, cwd=ROOT, timeout_seconds=args.timeout_seconds)
        return payload or sanitize({
            "schema": P2P_LITE_CLI_SCHEMA,
            "ok": False,
            "mode": "check",
            "step": step,
            "diagnosis_codes": ["p2p_lite_check_failed"],
        })
    if action == "daemon":
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "p2p_lite_daemon.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--swarm-id",
            args.swarm_id,
            "--role",
            args.role,
            "--ttl-seconds",
            str(args.ttl_seconds),
        ]
        if args.peer_id:
            command.extend(["--peer-id", args.peer_id])
        if args.peer_url:
            command.extend(["--peer-url", args.peer_url])
        if args.coordinator_url:
            command.extend(["--coordinator-url", args.coordinator_url])
        if args.backend:
            command.extend(["--backend", args.backend])
        if args.stage_role:
            command.extend(["--stage-role", args.stage_role])
        for capability in args.stage_capability or []:
            command.extend(["--stage-capability", capability])
        for bootstrap in args.bootstrap or []:
            command.extend(["--bootstrap", bootstrap])
        if args.peer_secret:
            command.extend(["--peer-secret", args.peer_secret])
        if args.require_signed:
            command.append("--require-signed")
        if args.signature_max_age_seconds != 3600.0:
            command.extend(["--signature-max-age-seconds", str(args.signature_max_age_seconds)])
        if args.print_peer:
            command.append("--print-peer")
        if not args.run:
            return sanitize(redact_values({
                "schema": P2P_LITE_CLI_SCHEMA,
                "ok": True,
                "mode": "daemon",
                "command": redacted_command(command, {"--peer-secret"}),
                "printed_only": True,
                "diagnosis_codes": ["p2p_lite_daemon_command_ready"],
            }, [args.peer_secret]))
        completed = runner(command, cwd=str(ROOT), text=True)
        return sanitize({
            "schema": P2P_LITE_CLI_SCHEMA,
            "ok": completed.returncode == 0,
            "mode": "daemon",
            "returncode": completed.returncode,
            "diagnosis_codes": ["p2p_lite_daemon_exited"],
        })
    if action == "resolve":
        session_request = build_session_request(
            prompt_text=args.prompt_text,
            backend=args.backend,
            stage_mode="split",
            max_new_tokens=args.max_new_tokens,
            route_source="peer-bootstrap",
        )
        coordinator_url = ""
        peers: list[dict[str, Any]] = []
        if args.bootstrap:
            coordinator_url, peers = resolve_coordinator_from_bootstrap(args.bootstrap, timeout=args.http_timeout)
        route = build_route_decision(session_request, coordinator_url=coordinator_url, peer_catalog=peers)
        return sanitize({
            "schema": P2P_LITE_CLI_SCHEMA,
            "ok": bool(route.get("usable_now")),
            "mode": "resolve",
            "session_request": session_request,
            "route": route,
            "peer_count": len(peers),
            "diagnosis_codes": list(route.get("diagnosis_codes") or []),
        })
    if action == "announce":
        peer = sanitize_peer({
            "schema": PEER_SCHEMA,
            "swarm_id": args.swarm_id,
            "peer_id": args.peer_id,
            "role": args.role,
            "urls": {"coordinator": args.coordinator_url, "peer": args.peer_url},
            "backend": args.backend,
            "stage_role": args.stage_role,
            "capabilities": {
                "runtime": "python-cli",
                "backend": args.backend,
                "real_llm_sharded_stage_role": args.stage_role,
                "real_llm_sharded_stage_capabilities": list(args.stage_capability or []),
            },
            "ttl_seconds": args.ttl_seconds,
        })
        peer = maybe_sign_p2p_peer(peer, args.peer_secret)
        payload = post_announce(args.bootstrap, peer, timeout=args.http_timeout)
        return sanitize(redact_values({
            "schema": P2P_LITE_CLI_SCHEMA,
            "ok": bool(payload.get("ok")),
            "mode": "announce",
            "peer_id": peer.get("peer_id"),
            "identity_verified": bool(((payload.get("peer") or {}) if isinstance(payload.get("peer"), dict) else {}).get("identity_verified")),
            "diagnosis_codes": ["p2p_lite_announce_ready" if payload.get("ok") else "p2p_lite_announce_failed"],
        }, [args.peer_secret]))
    raise SystemExit(f"unknown peer action: {action}")


def build_p2pd_cli(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "p2p_lite_daemon.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--swarm-id",
        args.swarm_id,
        "--role",
        args.role,
        "--ttl-seconds",
        str(args.ttl_seconds),
    ]
    if args.peer_id:
        command.extend(["--peer-id", args.peer_id])
    if args.peer_url:
        command.extend(["--peer-url", args.peer_url])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.backend:
        command.extend(["--backend", args.backend])
    if args.stage_role:
        command.extend(["--stage-role", args.stage_role])
    for capability in args.stage_capability or []:
        command.extend(["--stage-capability", capability])
    for bootstrap in args.bootstrap or []:
        command.extend(["--bootstrap", bootstrap])
    if args.peer_secret:
        command.extend(["--peer-secret", args.peer_secret])
    if args.require_signed:
        command.append("--require-signed")
    if args.signature_max_age_seconds != 3600.0:
        command.extend(["--signature-max-age-seconds", str(args.signature_max_age_seconds)])
    if args.print_peer:
        command.append("--print-peer")
    report = {
        "schema": P2PD_CLI_SCHEMA,
        "ok": True,
        "mode": "p2pd",
        "command": redacted_command(command, {"--peer-secret"}),
        "printed_only": not args.run,
        "diagnosis_codes": ["p2pd_command_ready"] + (["p2p_signed_announce_required"] if args.require_signed else []),
        "safety": {
            "tokens_gossiped": False,
            "raw_prompts_gossiped": False,
            "activations_gossiped": False,
            "peer_secret_gossiped": False,
            "signed_announcement_required": bool(args.require_signed),
            "coordinator_result_fallback": True,
            "not_production": True,
            "not_dht": True,
            "not_nat_traversal": True,
        },
    }
    if not args.run:
        return sanitize(redact_values(report, [args.peer_secret]))
    completed = runner(command, cwd=str(ROOT), text=True)
    report["returncode"] = completed.returncode
    report["ok"] = completed.returncode == 0
    report["diagnosis_codes"] = ["p2pd_exited"]
    return sanitize(redact_values(report, [args.peer_secret]))


def build_p2p_daemon_cli(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_p2p_daemon.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--swarm-id",
        args.swarm_id,
        "--role",
        args.role,
        "--ttl-seconds",
        str(args.ttl_seconds),
        "--discovery-backend",
        args.discovery_backend,
    ]
    if args.public_host:
        command.extend(["--public-host", args.public_host])
    if args.node_id:
        command.extend(["--node-id", args.node_id])
    if args.peer_url:
        command.extend(["--peer-url", args.peer_url])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.backend:
        command.extend(["--backend", args.backend])
    if args.stage_role:
        command.extend(["--stage-role", args.stage_role])
    for capability in args.stage_capability or []:
        command.extend(["--stage-capability", capability])
    for bootstrap in args.bootstrap or []:
        command.extend(["--bootstrap", bootstrap])
    if args.record_secret:
        command.extend(["--record-secret", args.record_secret])
    if args.require_signed:
        command.append("--require-signed")
    if args.signature_max_age_seconds != 3600.0:
        command.extend(["--signature-max-age-seconds", str(args.signature_max_age_seconds)])
    if getattr(args, "libp2p_host", "127.0.0.1") != "127.0.0.1":
        command.extend(["--libp2p-host", args.libp2p_host])
    if getattr(args, "libp2p_port", 0):
        command.extend(["--libp2p-port", str(args.libp2p_port)])
    if getattr(args, "libp2p_public_host", ""):
        command.extend(["--libp2p-public-host", args.libp2p_public_host])
    if getattr(args, "peer_key_file", ""):
        command.extend(["--peer-key-file", args.peer_key_file])
    if getattr(args, "kad_protocol", "/crowdtensor/kad/1.0.0") != "/crowdtensor/kad/1.0.0":
        command.extend(["--kad-protocol", args.kad_protocol])
    if args.print_record:
        command.append("--print-record")
    libp2p_backend = args.discovery_backend in {LIBP2P_KAD_BACKEND, LIBP2P_KAD_COMPAT_BACKEND}
    report = {
        "schema": P2P_DAEMON_CLI_SCHEMA,
        "ok": True,
        "mode": "p2p-daemon",
        "command": redacted_command(command, {"--record-secret"}),
        "printed_only": not args.run,
        "diagnosis_codes": ["p2p_daemon_command_ready", "real_p2p_provider_store_ready", "replaceable_discovery_backend_ready"]
        + (["libp2p_discovery_backend_ready", "p2p_peer_identity_ready", "p2p_provider_dht_ready"] if libp2p_backend else [])
        + (["real_p2p_signed_provider_record_required"] if args.require_signed else []),
        "safety": {
            "tokens_gossiped": False,
            "raw_prompts_gossiped": False,
            "activations_gossiped": False,
            "peer_secret_gossiped": False,
            "signed_provider_record_required": bool(args.require_signed),
            "coordinator_result_fallback": True,
            "not_production": True,
            "replaceable_discovery_backend": True,
            "discovery_backend": args.discovery_backend,
            "libp2p_runtime_ready": libp2p_backend,
            "dht_runtime_ready": libp2p_backend,
            "provider_record_transport": "libp2p-stream" if libp2p_backend else "http-provider-store",
            "nat_traversal_ready": False,
            "relay_ready": False,
            "hivemind_petals_parity": False,
        },
    }
    if not args.run:
        return sanitize(redact_values(report, [args.record_secret]))
    completed = runner(command, cwd=str(ROOT), text=True)
    report["returncode"] = completed.returncode
    report["ok"] = completed.returncode == 0
    report["diagnosis_codes"] = ["p2p_daemon_exited"]
    return sanitize(redact_values(report, [args.record_secret]))


def build_public_swarm_product_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_product_rc_pack.py"),
        "--output-dir",
        args.output_dir,
        "--gpu-report",
        args.gpu_report,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    step, payload = run_json_step(
        "public_swarm_product_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds + 60,
    )
    if payload:
        return payload
    return sanitize({
        "schema": "public_swarm_product_rc_v1",
        "cli_schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "ok": False,
        "mode": "rc",
        "step": step,
        "diagnosis_codes": ["public_swarm_product_rc_failed"],
    })


def build_release_ready(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "release_readiness_pack.py"),
        "--output-dir",
        str(output_dir),
        "--host",
        args.host,
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    if args.skip_external_llm_evidence:
        command.append("--skip-external-llm-evidence")
    if args.runtime_report:
        command.extend(["--runtime-report", args.runtime_report])
    if args.browser_report:
        command.extend(["--browser-report", args.browser_report])
    if args.remote_report:
        command.extend(["--remote-report", args.remote_report])
    step, payload = run_json_step(
        "release_readiness",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    if payload:
        return sanitize(payload)
    return sanitize({
        "schema": "release_readiness_v1",
        "ok": False,
        "release_status": {
            "ready": False,
            "status": "blocked",
            "blocking_reasons": [step.get("error") or "release readiness command failed"],
            "diagnosis_codes": ["release_readiness_failed"],
        },
        "step": step,
    })


def build_remote_runbook(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_demo_runbook_pack.py"),
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
    ]
    if args.replace:
        command.append("--replace")
    step, payload = run_json_step(
        "remote_demo_runbook",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))

    runbook_json = output_dir / "remote_demo_runbook.json"
    runbook_md = output_dir / "remote_demo_runbook.md"
    operator_env = output_dir / "operator.private.env"
    miner_env = output_dir / "miner.private.env"
    summary_json = output_dir / "remote_runbook_cli_summary.json"
    artifacts = {
        "remote_demo_runbook_json": _remote_summary_artifact(
            runbook_json,
            output_dir,
            kind="remote_demo_runbook",
            schema=str(payload.get("schema") or "remote_demo_runbook_v1"),
        ),
        "remote_demo_runbook_markdown": _remote_summary_artifact(
            runbook_md,
            output_dir,
            kind="remote_demo_runbook_markdown",
        ),
        "operator_private_env": _remote_summary_artifact(operator_env, output_dir, kind="private_env"),
        "miner_private_env": _remote_summary_artifact(miner_env, output_dir, kind="private_env"),
        "remote_runbook_cli_summary": {
            "kind": "remote_runbook_cli_summary",
            "path": "remote_runbook_cli_summary.json",
            "present": True,
            "schema": REMOTE_RUNBOOK_CLI_SCHEMA,
        },
    }
    demo = payload.get("demo") if isinstance(payload.get("demo"), dict) else {}
    scenario = {
        "scenario_schema": demo.get("scenario_schema"),
        "scenario_id": demo.get("scenario_id"),
        "scenario_description": demo.get("scenario_description"),
        "scenario_request_count": demo.get("scenario_request_count"),
    }
    summary = {
        "schema": REMOTE_RUNBOOK_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "miner_id": args.miner_id,
        "request_count": args.request_count,
        "scenario": scenario,
        "step": step,
        "runbook_schema": payload.get("schema") or "remote_demo_runbook_v1",
        "workload_type": demo.get("workload_type") or "model_bundle_infer",
        "artifacts": artifacts,
        "safety": {
            "private_env_files": ["operator.private.env", "miner.private.env"],
            "public_artifacts_exclude_plaintext_tokens": True,
            "captured_output_redacted": True,
            "not_production": True,
        },
        "limitations": [
            "Controlled two-machine demo runbook; not production Swarm Inference",
            "Requires operator-provided TLS, VPN, or private networking for non-local use",
            "Does not implement P2P/NAT traversal, GPU pooling, WebGPU model shards, or incentives",
        ],
        "recommended_next_commands": [
            f"source {output_dir / 'operator.private.env'}",
            (
                "crowdtensor remote-acceptance --coordinator-url https://YOUR_COORDINATOR_HOST "
                "--miner-id remote-linux-1 --observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" "
                "--admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --create-session "
                f"--scenario-id {args.scenario_id} --json"
            ),
        ],
    }
    summary = sanitize(redact_values(summary))
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded and fragment not in {
        "CROWDTENSOR_OBSERVER_TOKEN",
        "CROWDTENSOR_ADMIN_TOKEN",
    }]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_remote_acceptance(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    secret_values = [args.observer_token, args.admin_token]
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_demo_acceptance_pack.py"),
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--output-dir",
        str(output_dir),
    ]
    if args.create_session:
        command.append("--create-session")
    step, payload = run_json_step(
        "remote_demo_acceptance",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))

    acceptance_json = output_dir / "remote_demo_acceptance.json"
    acceptance_md = output_dir / "remote_demo_acceptance.md"
    summary_json = output_dir / "remote_acceptance_cli_summary.json"
    artifacts = {
        "remote_demo_acceptance_json": _remote_summary_artifact(
            acceptance_json,
            output_dir,
            kind="remote_demo_acceptance",
            schema=str(payload.get("schema") or "remote_demo_acceptance_v1"),
        ),
        "remote_demo_acceptance_markdown": _remote_summary_artifact(
            acceptance_md,
            output_dir,
            kind="remote_demo_acceptance_markdown",
        ),
        "remote_acceptance_cli_summary": {
            "kind": "remote_acceptance_cli_summary",
            "path": "remote_acceptance_cli_summary.json",
            "present": True,
            "schema": REMOTE_ACCEPTANCE_CLI_SCHEMA,
        },
    }
    summary = {
        "schema": REMOTE_ACCEPTANCE_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "miner_id": args.miner_id,
        "request_count": args.request_count,
        "scenario": payload.get("scenario") or {},
        "create_session": bool(args.create_session),
        "step": step,
        "acceptance_schema": payload.get("schema") or "remote_demo_acceptance_v1",
        "diagnosis_codes": diagnosis_codes(payload),
        "artifacts": artifacts,
        "safety": {
            "captured_output_redacted": True,
            "summary_excludes_plaintext_tokens": True,
            "read_only_workload": "model_bundle_infer",
            "not_production": True,
        },
        "limitations": [
            "Controlled two-machine acceptance wrapper; not production Swarm Inference",
            "Requires a running Coordinator and remote Miner already configured by the operator",
            "Does not implement P2P/NAT traversal, GPU pooling, WebGPU model shards, or incentives",
        ],
    }
    summary = sanitize(redact_values(summary, secret_values))
    encoded = json.dumps(summary, sort_keys=True)
    if any(secret and secret in encoded for secret in secret_values):
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_remote_demo(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.remote_demo_action == "kaggle-real":
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "kaggle_real_runtime_acceptance_pack.py"),
            args.kaggle_real_action,
            "--public-host",
            args.public_host,
            "--port",
            str(args.port),
            "--miner-id",
            args.miner_id,
            "--workload",
            args.workload,
            "--output-dir",
            str(output_dir),
            "--request-count",
            str(args.request_count),
            "--scenario-id",
            args.scenario_id,
            "--decode-steps",
            str(args.decode_steps),
            "--stage-mode",
            args.stage_mode,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ]
        if getattr(args, "micro_llm_artifact", ""):
            command.extend(["--micro-llm-artifact", args.micro_llm_artifact])
        if getattr(args, "prompt_texts", ""):
            command.extend(["--prompt-texts", args.prompt_texts])
        if args.require_distinct_stage_miners:
            command.append("--require-distinct-stage-miners")
        if args.coordinator_url:
            command.extend(["--coordinator-url", args.coordinator_url])
        secret_values: list[str] = []
        if args.kaggle_real_action == "prepare":
            command.extend([
                "--bind-host",
                args.bind_host,
                "--backlog",
                str(args.backlog),
                "--lease-seconds",
                str(args.lease_seconds),
            ])
            for flag, value in [
                ("--miner-token", args.miner_token),
                ("--observer-token", args.observer_token),
                ("--admin-token", args.admin_token),
            ]:
                if value:
                    command.extend([flag, value])
                    secret_values.append(value)
            if args.replace:
                command.append("--replace")
        elif args.kaggle_real_action == "verify":
            for flag, value in [("--observer-token", args.observer_token), ("--admin-token", args.admin_token)]:
                if value:
                    command.extend([flag, value])
                    secret_values.append(value)
            command.extend([
                "--remote-timeout-seconds",
                str(args.remote_timeout_seconds),
                "--poll-interval",
                str(args.poll_interval),
                "--http-timeout",
                str(args.http_timeout),
                "--artifact-timeout",
                str(args.artifact_timeout),
                "--admin-results-limit",
                str(args.admin_results_limit),
            ])
            if args.require_existing_result:
                command.append("--require-existing-result")
            if args.collect_on_failure:
                command.append("--collect-on-failure")
        elif args.kaggle_real_action == "collect":
            for flag, value in [("--observer-token", args.observer_token), ("--admin-token", args.admin_token)]:
                if value:
                    command.extend([flag, value])
                    secret_values.append(value)
            command.extend([
                "--http-timeout",
                str(args.http_timeout),
                "--artifact-timeout",
                str(args.artifact_timeout),
                "--admin-results-limit",
                str(args.admin_results_limit),
            ])
            if args.task_id:
                command.extend(["--task-id", args.task_id])
        else:
            raise SystemExit(f"unknown kaggle-real action: {args.kaggle_real_action}")
        step, payload = run_json_step(
            "kaggle_real_runtime_acceptance",
            command,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
            redact_secrets=secret_values,
        )
        if not payload:
            payload = {
                "schema": KAGGLE_REAL_RUNTIME_SCHEMA,
                "generated_at": utc_now(),
                "ok": False,
                "mode": args.kaggle_real_action,
                "output_dir": str(output_dir),
                "coordinator_url": args.coordinator_url or f"http://{args.public_host}:{args.port}",
                "miner_id": args.miner_id,
                "step": step,
                "diagnosis_codes": ["kaggle_runtime_blocked"],
            }
        payload = redact_values(payload, secret_values)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    if args.remote_demo_action == "clean":
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "remote_home_compute_demo_pack.py"),
            "clean",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
        step, payload = run_json_step(
            "remote_home_compute_demo",
            command,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
        )
        if not payload:
            payload = {
                "schema": REMOTE_HOME_DEMO_SCHEMA,
                "generated_at": utc_now(),
                "ok": False,
                "mode": args.remote_demo_action,
                "output_dir": str(output_dir),
                "step": step,
                "diagnosis_codes": ["remote_home_compute_failed"],
            }
        return sanitize(payload)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_home_compute_demo_pack.py"),
        args.remote_demo_action,
        "--workload",
        args.workload,
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--json",
    ]
    if hasattr(args, "decode_steps"):
        command.extend(["--decode-steps", str(args.decode_steps)])
    if hasattr(args, "stage_role"):
        command.extend(["--stage-role", args.stage_role])
    if hasattr(args, "stage_mode"):
        command.extend(["--stage-mode", args.stage_mode])
    if getattr(args, "require_distinct_stage_miners", False):
        command.append("--require-distinct-stage-miners")
    if args.workload == "micro-llm-sharded":
        if getattr(args, "micro_llm_artifact", ""):
            command.extend(["--micro-llm-artifact", args.micro_llm_artifact])
        if getattr(args, "prompt_texts", ""):
            command.extend(["--prompt-texts", args.prompt_texts])
    if args.workload == "real-llm-sharded":
        if getattr(args, "hf_model_id", ""):
            command.extend(["--hf-model-id", args.hf_model_id])
        if getattr(args, "hf_cache_dir", ""):
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        if getattr(args, "prompt_texts", ""):
            command.extend(["--prompt-texts", args.prompt_texts])
    if hasattr(args, "target"):
        command.extend(["--target", args.target])
    secret_values: list[str] = []
    if args.remote_demo_action in {"prepare", "verify"} and hasattr(args, "timeout_seconds"):
        command.extend(["--timeout-seconds", str(args.timeout_seconds)])
    if hasattr(args, "mock") and getattr(args, "mock", False):
        command.append("--mock")
    if hasattr(args, "llm_runtime_cmd") and getattr(args, "llm_runtime_cmd", ""):
        command.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    if hasattr(args, "llm_runtime_url") and getattr(args, "llm_runtime_url", ""):
        command.extend(["--llm-runtime-url", args.llm_runtime_url])
    if hasattr(args, "llm_runtime_api_key") and getattr(args, "llm_runtime_api_key", ""):
        command.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    if hasattr(args, "llm_runtime_model_id") and getattr(args, "llm_runtime_model_id", ""):
        command.extend(["--llm-runtime-model-id", args.llm_runtime_model_id])
    if hasattr(args, "llm_runtime_timeout") and getattr(args, "llm_runtime_timeout", None) is not None:
        command.extend(["--llm-runtime-timeout", str(args.llm_runtime_timeout)])
    if args.remote_demo_action == "prepare":
        if args.replace:
            command.append("--replace")
    elif args.remote_demo_action == "verify":
        secret_values = [args.observer_token, args.admin_token, args.llm_runtime_url, args.llm_runtime_api_key]
        command.extend([
            "--observer-token",
            args.observer_token,
            "--admin-token",
            args.admin_token,
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--poll-interval",
            str(args.poll_interval),
            "--http-timeout",
            str(args.http_timeout),
            "--artifact-timeout",
            str(args.artifact_timeout),
            "--admin-results-limit",
            str(args.admin_results_limit),
        ])
        if args.create_session:
            command.append("--create-session")
        else:
            command.append("--no-create-session")
    elif args.remote_demo_action == "doctor":
        secret_values = [args.observer_token, args.admin_token]
        command.extend([
            "--observer-token",
            args.observer_token,
            "--admin-token",
            args.admin_token,
            "--http-timeout",
            str(args.http_timeout),
            "--admin-results-limit",
            str(args.admin_results_limit),
        ])
        if args.require_result:
            command.append("--require-result")
    elif args.remote_demo_action == "collect":
        secret_values = [args.observer_token, args.admin_token, args.llm_runtime_url, args.llm_runtime_api_key]
        command.extend([
            "--observer-token",
            args.observer_token,
            "--admin-token",
            args.admin_token,
            "--http-timeout",
            str(args.http_timeout),
            "--artifact-timeout",
            str(args.artifact_timeout),
            "--admin-results-limit",
            str(args.admin_results_limit),
        ])
        if args.task_id:
            command.extend(["--task-id", args.task_id])
    else:
        raise SystemExit(f"unknown remote-demo action: {args.remote_demo_action}")
    step, payload = run_json_step(
        "remote_home_compute_demo",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if not payload:
        payload = {
            "schema": REMOTE_HOME_DEMO_SCHEMA,
            "generated_at": utc_now(),
            "ok": False,
            "mode": args.remote_demo_action,
            "output_dir": str(output_dir),
            "coordinator_url": args.coordinator_url.rstrip("/"),
            "miner_id": args.miner_id,
            "step": step,
            "diagnosis_codes": ["remote_home_compute_failed"],
        }
    payload = sanitize(redact_values(payload, secret_values))
    encoded = json.dumps(payload, sort_keys=True)
    if any(secret and secret in encoded for secret in secret_values):
        payload["ok"] = False
        payload.setdefault("diagnosis_codes", [])
        if "sensitive_output_detected" not in payload["diagnosis_codes"]:
            payload["diagnosis_codes"].append("sensitive_output_detected")
    return payload


def print_local_proof(summary: dict[str, Any]) -> None:
    print("CrowdTensor local proof")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    for step in summary.get("steps") or []:
        state = "skipped" if step.get("skipped") else step.get("ok")
        print(f"  step {step.get('name')}: {state}")
    artifacts = summary.get("artifacts") or {}
    manifest = artifacts.get("demo_manifest_json") or {}
    print(f"  demo manifest: {manifest.get('path')} present={manifest.get('present')}")


def print_cleanup_report(report: dict[str, Any]) -> None:
    print("CrowdTensor artifact cleanup")
    print(f"  ok: {report.get('ok')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  candidates: {report.get('candidate_count')}")
    print(f"  deleted_bytes: {report.get('deleted_bytes')}")
    for candidate in report.get("candidates") or []:
        print(
            "  "
            f"{candidate.get('action')} "
            f"{candidate.get('kind')} "
            f"{candidate.get('bytes')}B "
            f"{candidate.get('path')}"
        )


def print_home_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor home inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    route = summary.get("route") or {}
    print(f"  route: {route.get('name')} target={route.get('target')} confidence={route.get('confidence')}")
    scenario = summary.get("scenario") or {}
    if scenario.get("scenario_id"):
        print(f"  scenario: {scenario.get('scenario_id')} ({scenario.get('scenario_schema')})")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    inference = summary.get("inference") or {}
    print(
        "  inference: "
        f"present={inference.get('present')} "
        f"requests={inference.get('request_count')} "
        f"trace={inference.get('request_trace_count')} "
        f"rps={inference.get('requests_per_second')}"
    )
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_llm_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor LLM inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    adapter = summary.get("adapter") or {}
    print(f"  adapter: {adapter.get('kind')} model={adapter.get('model_id')}")
    inference = summary.get("inference") or {}
    print(
        "  inference: "
        f"requests={inference.get('request_count')} "
        f"completions={inference.get('completion_count')} "
        f"chars={inference.get('output_chars')} "
        f"rps={inference.get('requests_per_second')}"
    )
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_cpu_inference_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor CPU inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_sharded_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor sharded inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  failure_mode: {summary.get('failure_mode')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    session = summary.get("session") or {}
    print(f"  session: {session.get('session_id')} stages={session.get('stage_count')}")
    stage = summary.get("stage_summary") or {}
    stage0 = stage.get("stage_0") or {}
    stage1 = stage.get("stage_1") or {}
    print(f"  stage0: task={stage0.get('task_id')} miner={stage0.get('miner_id')} activations={stage0.get('activation_count')}")
    print(f"  stage1: task={stage1.get('task_id')} miner={stage1.get('miner_id')} baseline_match={stage1.get('baseline_match')}")
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_micro_llm_sharded_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor micro-LLM sharded inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  failure_mode: {summary.get('failure_mode')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    session = summary.get("session") or {}
    print(f"  session: {session.get('session_id')} stages={session.get('stage_count')} decode_steps={session.get('decode_steps')}")
    stage = summary.get("stage_summary") or {}
    stage0 = stage.get("stage_0") or {}
    stage1 = stage.get("stage_1") or {}
    print(f"  stage0: task={stage0.get('task_id')} miner={stage0.get('miner_id')} activations={stage0.get('activation_count')}")
    print(
        "  stage1: "
        f"task={stage1.get('task_id')} "
        f"miner={stage1.get('miner_id')} "
        f"baseline_match={stage1.get('baseline_match')} "
        f"decoded_tokens_match={stage1.get('decoded_tokens_match')}"
    )
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_real_llm_sharded_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor real tiny-LLM sharded inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  failure_mode: {summary.get('failure_mode')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    session = summary.get("session") or {}
    print(f"  session: {session.get('session_id')} stages={session.get('stage_count')} model={session.get('model_id')}")
    stage = summary.get("stage_summary") or {}
    stage0 = stage.get("stage_0") or {}
    stage1 = stage.get("stage_1") or {}
    print(f"  stage0: task={stage0.get('task_id')} miner={stage0.get('miner_id')} activations={stage0.get('activation_count')}")
    print(
        "  stage1: "
        f"task={stage1.get('task_id')} "
        f"miner={stage1.get('miner_id')} "
        f"baseline_match={stage1.get('baseline_match')} "
        f"decoded_tokens_match={stage1.get('decoded_tokens_match')}"
    )
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_micro_llm_artifact(summary: dict[str, Any]) -> None:
    print("CrowdTensor micro-LLM artifact")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  artifact: {summary.get('artifact_id')} {summary.get('artifact_hash')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_remote_sharded_inference_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor remote sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_remote_micro_llm_sharded_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor remote micro-LLM sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  decode_steps: {report.get('decode_steps')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_remote_real_llm_sharded_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor remote real tiny-LLM sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  hf_model_id: {report.get('hf_model_id')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_micro_llm_live_rc(report: dict[str, Any]) -> None:
    print("CrowdTensor micro-LLM live two-node RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    print(f"  local stand-in: {runtime.get('local_generated_stage_upload_standins')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_real_llm_live_rc(report: dict[str, Any]) -> None:
    print("CrowdTensor real small-LLM live two-node RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    print(f"  local stand-in: {runtime.get('local_generated_stage_upload_standins')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_real_llm_internet_alpha(report: dict[str, Any]) -> None:
    print("CrowdTensor real Internet Swarm Inference Alpha")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    print(f"  local stand-in: {runtime.get('local_generated_stage_upload_standins')}")
    print(f"  package only: {runtime.get('package_only')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    print(f"  stage requeue: {runtime.get('stage_requeue_verified')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_real_llm_internet_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor real Internet Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    lifecycle = report.get("kaggle_lifecycle") or {}
    print(f"  kaggle auto: {runtime.get('kaggle_auto')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    print(f"  kernels deleted: {lifecycle.get('kernels_deleted')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_swarm_inference_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    if report.get("command_text"):
        print(f"  command: {report.get('command_text')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_alpha(report: dict[str, Any]) -> None:
    session = report.get("session") if isinstance(report.get("session"), dict) else {}
    print("CrowdTensor Public Swarm Inference Alpha")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  model: {session.get('model_id')}")
    print(f"  external runtime: {session.get('live_external_runtime_verified')}")
    print(f"  local requeue: {session.get('local_stage_requeue_verified')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_alpha_rc(report: dict[str, Any]) -> None:
    rc = report.get("release_candidate") if isinstance(report.get("release_candidate"), dict) else {}
    print("CrowdTensor Public Swarm Inference Alpha RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {rc.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_beta(report: dict[str, Any]) -> None:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    print("CrowdTensor Public Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_beta_rc(report: dict[str, Any]) -> None:
    rc = report.get("rc") if isinstance(report.get("rc"), dict) else {}
    print("CrowdTensor Public Swarm Inference Beta RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {rc.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_product_beta(report: dict[str, Any]) -> None:
    beta = report.get("product_beta") if isinstance(report.get("product_beta"), dict) else {}
    print("CrowdTensor Public Swarm Product Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_real_llm_swarm_beta(report: dict[str, Any]) -> None:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    print("CrowdTensor Public Real-LLM Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  cpu_default_ready: {beta.get('cpu_default_ready')}")
    print(f"  external_two_stage_ready: {beta.get('external_two_stage_ready')}")
    print(f"  external_stage_requeue_ready: {beta.get('external_stage_requeue_ready')}")
    print(f"  p2p_ready_product_beta: {beta.get('p2p_ready_product_beta')}")
    print(f"  cuda_optional_fail_closed_ready: {beta.get('cuda_optional_fail_closed_ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_usable_swarm_inference(report: dict[str, Any]) -> None:
    usable = report.get("usable_swarm") if isinstance(report.get("usable_swarm"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    p2p = readiness.get("p2p_product_path") if isinstance(readiness.get("p2p_product_path"), dict) else {}
    print("CrowdTensor Usable Swarm Inference v1")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {usable.get('ready')}")
    print(f"  p2p route ready: {p2p.get('route_ready')}")
    print(f"  real generate ready: {p2p.get('real_generate_ready')}")
    print(f"  generated tokens: {p2p.get('generated_token_count')}/{p2p.get('max_new_tokens')}")
    print(f"  distinct stage miners: {p2p.get('distinct_stage_miners')}")
    print(f"  stage rescue ready: {p2p.get('stage_rescue_ready') and p2p.get('real_stage_rescue_ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_v2(report: dict[str, Any]) -> None:
    v2 = report.get("public_swarm_v2") if isinstance(report.get("public_swarm_v2"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    local = readiness.get("local_p2p_generate") if isinstance(readiness.get("local_p2p_generate"), dict) else {}
    external = readiness.get("external_validation") if isinstance(readiness.get("external_validation"), dict) else {}
    p2p = readiness.get("p2p_route_hardening") if isinstance(readiness.get("p2p_route_hardening"), dict) else {}
    cuda = readiness.get("cuda_optional") if isinstance(readiness.get("cuda_optional"), dict) else {}
    perf = readiness.get("performance") if isinstance(readiness.get("performance"), dict) else {}
    print("CrowdTensor Public Swarm Inference v2")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {v2.get('ready')}")
    print(f"  local tokens: {local.get('generated_token_count')}/{local.get('max_new_tokens')}")
    print(f"  local accepted rows: {local.get('accepted_rows')} ready={local.get('accepted_rows_ready')}")
    print(f"  kv cache ready: {local.get('kv_cache_ready')}")
    print(f"  batch ready: {local.get('batch_ready')}")
    print(f"  stream ready: {local.get('stream_ready')}")
    print(f"  p2p route: {p2p.get('preferred_route')} ready={p2p.get('ready')}")
    print(f"  external ready: {external.get('ready')} tokens={external.get('generated_token_count')}/{external.get('max_new_tokens')} accepted_rows={external.get('accepted_rows')} rows_ready={external.get('accepted_rows_ready')}")
    print(f"  model match: local={((local.get('model') or {}).get('compatible') if isinstance(local.get('model'), dict) else None)} external={((external.get('model') or {}).get('compatible') if isinstance(external.get('model'), dict) else None)} p2p={((p2p.get('model') or {}).get('compatible') if isinstance(p2p.get('model'), dict) else None)}")
    print(f"  cuda fail-closed: {cuda.get('fail_closed_ready')}")
    print(f"  performance: latency={perf.get('stage_latency_ready')} throughput={perf.get('throughput_summary_ready')} memory={perf.get('memory_or_vram_summary_ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_developer_preview(report: dict[str, Any]) -> None:
    preview = report.get("developer_preview") if isinstance(report.get("developer_preview"), dict) else {}
    print("CrowdTensor Public Swarm Developer Preview")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {preview.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_live_preview_rc(report: dict[str, Any]) -> None:
    preview = report.get("live_preview") if isinstance(report.get("live_preview"), dict) else {}
    print("CrowdTensor Public Swarm Live Preview RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {preview.get('ready')}")
    print(f"  external_runtime_verified: {preview.get('external_runtime_verified')}")
    print(f"  fresh_live_kaggle_run: {preview.get('fresh_live_kaggle_run')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_operator_preview(report: dict[str, Any]) -> None:
    preview = report.get("operator_preview") if isinstance(report.get("operator_preview"), dict) else {}
    print("CrowdTensor Public Swarm v0.1 Operator Preview")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {preview.get('ready')}")
    print(f"  serve_join_generate_ready: {preview.get('serve_join_generate_ready')}")
    print(f"  cpu_fallback_ready: {preview.get('cpu_fallback_ready')}")
    print(f"  live_preview_ready: {preview.get('live_preview_ready')}")
    print(f"  external_runtime_verified: {preview.get('external_runtime_verified')}")
    print(f"  external_runtime_blocked: {preview.get('external_runtime_blocked')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_trial(report: dict[str, Any]) -> None:
    trial = report.get("trial") if isinstance(report.get("trial"), dict) else {}
    print("CrowdTensor Public Swarm v0.2 Usable Inference Trial")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {trial.get('ready')}")
    print(f"  serve_join_generate_trial_ready: {trial.get('serve_join_generate_trial_ready')}")
    print(f"  degraded_cpu_fallback_ready: {trial.get('degraded_cpu_fallback_ready')}")
    print(f"  gpu_generation_ready: {trial.get('gpu_generation_ready')}")
    print(f"  external_runtime_verified: {trial.get('external_runtime_verified')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_preview_v04(report: dict[str, Any]) -> None:
    preview = report.get("preview") if isinstance(report.get("preview"), dict) else {}
    print("CrowdTensor Public Swarm Inference Preview v0.4")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {preview.get('ready')}")
    print(f"  external_two_stage_generation_ready: {preview.get('external_two_stage_generation_ready')}")
    print(f"  external_stage_requeue_ready: {preview.get('external_stage_requeue_ready')}")
    print(f"  stage_latency_ready: {preview.get('stage_latency_ready')}")
    print(f"  throughput_summary_ready: {preview.get('throughput_summary_ready')}")
    print(f"  memory_or_vram_summary_ready: {preview.get('memory_or_vram_summary_ready')}")
    print(f"  optional_model_ready: {preview.get('optional_model_ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_gpu_inference_beta(report: dict[str, Any]) -> None:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    print("CrowdTensor Public Swarm GPU Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  backend: {beta.get('backend')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_gpu_sharded_generation_beta(report: dict[str, Any]) -> None:
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    gpu = report.get("gpu") if isinstance(report.get("gpu"), dict) else {}
    print("CrowdTensor GPU sharded generation Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  backend: {gpu.get('backend')}")
    print(f"  model: {gpu.get('model_id')}")
    print(f"  generated_tokens: {generation.get('generated_token_count')}/{generation.get('max_new_tokens')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_release_ready(report: dict[str, Any]) -> None:
    status = report.get("release_status") or {}
    git = report.get("git") or {}
    print("CrowdTensor release readiness")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  status: {status.get('status')}")
    print(f"  branch: {git.get('branch')} commit={git.get('commit')}")
    print(f"  dirty: {git.get('dirty')} status_count={git.get('status_count')}")
    print(f"  diagnosis: {', '.join(status.get('diagnosis_codes') or [])}")
    for reason in status.get("blocking_reasons") or []:
        print(f"  blocker: {reason}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_remote_cli_report(report: dict[str, Any], *, title: str) -> None:
    print(title)
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  miner: {report.get('miner_id')}")
    codes = report.get("diagnosis_codes") or []
    if codes:
        print(f"  diagnosis: {', '.join(codes)}")
    step = report.get("step") or {}
    if step:
        print(f"  step {step.get('name')}: {step.get('ok')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="crowdtensor",
        description="CrowdTensor user-facing command line tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("local-proof", help="Run the CPU-only local proof and collect safe artifacts.")
    local.add_argument("--output-dir", default="dist/local-proof")
    local.add_argument("--base-port", type=int, default=8914)
    local.add_argument("--request-count", type=int, default=4)
    local.add_argument("--timeout-seconds", type=int, default=180)
    local.add_argument("--skip-doctor", action="store_true")
    local.add_argument("--json", action="store_true")
    clean = subparsers.add_parser("clean-artifacts", help="Safely clean generated CrowdTensor caches and temp artifacts.")
    clean.add_argument("--apply", action="store_true", help="delete eligible artifacts; default is dry-run")
    clean.add_argument("--dry-run", action="store_true", help="show candidates without deleting; this is the default")
    clean.add_argument("--include-reports", action="store_true", help="allow deletion of /tmp/crowdtensor_*.json/md reports")
    clean.add_argument("--older-than-hours", type=float, default=24.0)
    clean.add_argument("--json", action="store_true")

    infer = subparsers.add_parser(
        "infer",
        help="Run a user-friendly CrowdTensor swarm inference request.",
        description=(
            "Run the shortest user-facing CrowdTensor inference path.\n\n"
            "Default mode starts the fast local product loopback proof, runs a tiny real GPT split\n"
            "across stage0/stage1 workers, prints local display-only output in human mode, and\n"
            "writes a redacted infer_summary.json. Use --mode existing to target an already\n"
            "running Coordinator or P2P-discovered swarm. Reports include action and next[...] lines\n"
            "with copyable follow-up commands. ready_to_submit labels mean: verified is ready\n"
            "after route, Coordinator, and stage Miner checks; partial can submit but still needs\n"
            "observer-token stage verification; blocked needs the printed operator_action;\n"
            "skipped is request-shape only.\n\n"
            "Boundaries: Coordinator-backed, read-only, tiny/small-model scoped; not production\n"
            "Hivemind/Petals parity, not large-model serving, and not a permissionless P2P network."
        ),
        epilog=(
            "examples:\n"
            "  crowdtensor infer \"CrowdTensor routes small models across home compute\"\n"
            "  crowdtensor infer \"your prompt\" --max-new-tokens 8 --stream\n"
            "  crowdtensor infer \"your prompt\" --mode existing --coordinator-url http://127.0.0.1:8787 --dry-run\n"
            "  CROWDTENSOR_ADMIN_TOKEN=... crowdtensor infer \"your prompt\" --mode existing --coordinator-url http://127.0.0.1:8787\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    infer.add_argument("prompt_text", nargs="?", default="CrowdTensor routes small models across home compute")
    infer.add_argument("--mode", dest="infer_mode", choices=["local", "existing"], default="local")
    infer.add_argument("--output-dir", default="dist/infer")
    infer.add_argument("--prompt-texts", default="", help="comma-separated bounded batch of up to 4 prompts")
    infer.add_argument("--max-new-tokens", type=int, default=8)
    infer.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    infer.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    infer.add_argument("--hf-cache-dir", default="")
    infer.add_argument("--stream", action="store_true", help="request safe stream-progress evidence")
    infer.add_argument("--include-output", action="store_true", help="request local raw output from generate; JSON and saved artifacts still suppress it")
    infer.add_argument("--dry-run", action="store_true", help="check route/session readiness without submitting an inference task")
    infer.add_argument("--full-evidence", action="store_true", help="use the full local Public Swarm v2 gate instead of the faster product loopback path")
    infer.add_argument("--coordinator-url", default="")
    infer.add_argument("--peer-bootstrap", default="")
    infer.add_argument("--p2p", action="store_true")
    infer.add_argument("--p2p-backend", choices=["lite", "real"], default="lite")
    infer.add_argument("--swarm-id", default="default")
    infer.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    infer.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    infer.add_argument("--scenario-id", default="crowdtensor-infer")
    infer.add_argument("--public-host", default="127.0.0.1")
    infer.add_argument("--p2p-port", type=int, default=9788)
    infer.add_argument("--coordinator-port", type=int, default=9789)
    infer.add_argument("--real-p2p-port", type=int, default=9790)
    infer.add_argument("--real-p2p-coordinator-port", type=int, default=9791)
    infer.add_argument("--real-p2p-libp2p-port", type=int, default=0)
    infer.add_argument("--real-p2p-discovery-backend", choices=sorted(DISCOVERY_BACKENDS), default="http-provider-store")
    infer.add_argument("--usable-report", default="dist/usable-swarm-inference-v1/usable_swarm_inference.json")
    infer.add_argument("--preview-report", default="dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json")
    infer.add_argument(
        "--real-p2p-report",
        default="dist/goal-final-infer-real-p2p-core-fresh-16tok-import-strict-20260601/real_p2p_swarm_inference_core_rc.json",
    )
    infer.add_argument("--gpu-report", default="dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json")
    infer.add_argument("--fresh-external-attempt-report", default="")
    infer.add_argument("--startup-timeout", type=float, default=45.0)
    infer.add_argument("--timeout-seconds", type=float, default=420.0)
    infer.add_argument("--poll-interval", type=float, default=1.0)
    infer.add_argument("--http-timeout", type=float, default=30.0)
    infer.add_argument("--admin-results-limit", type=int, default=50)
    infer.add_argument("--json", action="store_true")

    serve = subparsers.add_parser(
        "serve",
        help="Print or run a product-facing Coordinator command.",
        description=(
            "Start or print the Coordinator used by the product inference flow. The Coordinator\n"
            "leases read-only generation work to stage Miners; keep it running, then start one\n"
            "stage0 and one stage1 Miner with crowdtensor join before running generate --dry-run."
        ),
        epilog=(
            "examples:\n"
            "  crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run\n"
            "  crowdtensor serve --profile cpu-real-llm --port 8787 --json\n"
            "  crowdtensor serve --p2p --peer-bootstrap http://127.0.0.1:8788 --run\n"
            "\n"
            "Boundary: local/private Coordinator by default; use explicit network controls for remote hosts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    serve.add_argument("--profile", choices=["cpu-real-llm", "gpu-generation"], default="cpu-real-llm")
    serve.add_argument("--bind-host", default="127.0.0.1")
    serve.add_argument("--public-host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--state-dir", default="state")
    serve.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", "local-admin"))
    serve.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    serve.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    serve.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    serve.add_argument("--hf-cache-dir", default="")
    serve.add_argument("--lease-seconds", type=float, default=15.0)
    serve.add_argument("--p2p", action="store_true", help="announce this Coordinator to a p2pd bootstrap before printing/running")
    serve.add_argument("--p2p-backend", choices=["lite", "real"], default="lite")
    serve.add_argument("--peer-bootstrap", default=DEFAULT_P2P_BOOTSTRAP)
    serve.add_argument("--swarm-id", default="default")
    serve.add_argument("--peer-id", default="")
    serve.add_argument("--peer-url", default="")
    serve.add_argument("--ttl-seconds", type=float, default=60.0)
    serve.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    serve.add_argument("--http-timeout", type=float, default=5.0)
    serve.add_argument("--i-understand-public-bind", action="store_true")
    serve.add_argument("--run", action="store_true")
    serve.add_argument("--json", action="store_true")

    join = subparsers.add_parser(
        "join",
        help="Print or run a product-facing Miner command.",
        description=(
            "Start or print a product Miner for the bounded generation flow. Use distinct stage0\n"
            "and stage1 Miners so generate --dry-run can confirm both halves of the tiny model\n"
            "route before a token-backed generation request is submitted."
        ),
        epilog=(
            "examples:\n"
            "  crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run\n"
            "  crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage1-miner --stage stage1 --run\n"
            "  crowdtensor join --p2p --peer-bootstrap http://127.0.0.1:8788 --miner-id stage0-miner --stage stage0 --run\n"
            "\n"
            "Boundary: CPU tiny-model route by default; not large-model serving or permissionless P2P."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    join.add_argument("--coordinator-url", default="")
    join.add_argument("--peer-bootstrap", default="")
    join.add_argument("--p2p", action="store_true", help="discover Coordinator and announce stage capability through p2pd")
    join.add_argument("--p2p-backend", choices=["lite", "real"], default="lite")
    join.add_argument("--swarm-id", default="default")
    join.add_argument("--peer-id", default="")
    join.add_argument("--peer-url", default="")
    join.add_argument("--ttl-seconds", type=float, default=60.0)
    join.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    join.add_argument("--miner-id", default="public-swarm-miner")
    join.add_argument("--stage", choices=["stage0", "stage1", "both"], default="both")
    join.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    join.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    join.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    join.add_argument("--hf-cache-dir", default="")
    join.add_argument("--once", action="store_true")
    join.add_argument("--max-tasks", type=int, default=0)
    join.add_argument("--max-runtime-seconds", type=float, default=0.0)
    join.add_argument("--compute-seconds", type=float, default=0.0)
    join.add_argument("--max-request-attempts", type=int, default=3)
    join.add_argument("--retry-base-sleep", type=float, default=0.2)
    join.add_argument("--retry-max-sleep", type=float, default=2.0)
    join.add_argument("--idle-sleep", type=float, default=2.0)
    join.add_argument("--http-timeout", type=float, default=5.0)
    join.add_argument("--run", action="store_true")
    join.add_argument("--json", action="store_true")

    generate = subparsers.add_parser(
        "generate",
        help="Create a bounded public product generation session.",
        description=(
            "Create a bounded CrowdTensor generation request against an existing Coordinator\n"
            "or P2P-discovered product swarm. Reports include action and next[...] lines with\n"
            "copyable follow-up commands; missing routes return startup guidance instead of a\n"
            "bare parser error. ready_to_submit labels mean: verified is ready after route,\n"
            "Coordinator, and stage Miner checks; partial can submit but still needs observer-token\n"
            "stage verification; blocked needs the printed operator_action; skipped is request-shape only.\n\n"
            "Boundaries: Coordinator-backed, read-only, tiny/small-model scoped; not production\n"
            "Hivemind/Petals parity, not large-model serving, and not arbitrary public prompt serving."
        ),
        epilog=(
            "examples:\n"
            "  crowdtensor generate \"your prompt\"\n"
            "  crowdtensor generate \"your prompt\" --coordinator-url http://127.0.0.1:8787 --dry-run\n"
            "  CROWDTENSOR_ADMIN_TOKEN=... crowdtensor generate \"your prompt\" --coordinator-url http://127.0.0.1:8787\n"
            "  crowdtensor generate --prompt-texts \"first prompt,second prompt\" --coordinator-url http://127.0.0.1:8787 --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    generate.add_argument("prompt_text_arg", nargs="?", default="", help="optional prompt text, equivalent to --prompt-text")
    generate.add_argument("--prompt-text", "--prompt", dest="prompt_text", default=DEFAULT_PRODUCT_GENERATE_PROMPT)
    generate.add_argument("--prompt-texts", default="", help="comma-separated bounded batch of up to 4 prompts")
    generate.add_argument("--scenario-id", default="public-swarm-product-rc")
    generate.add_argument("--max-new-tokens", type=int, default=16)
    generate.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    generate.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    generate.add_argument("--coordinator-url", default="")
    generate.add_argument("--peer-bootstrap", default="")
    generate.add_argument("--p2p", action="store_true", help="resolve Coordinator and stage peers through p2pd")
    generate.add_argument("--p2p-backend", choices=["lite", "real"], default="lite")
    generate.add_argument("--swarm-id", default="default", help="label for P2P-discovered swarms; accepted for parity with p2pd/serve/join")
    generate.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    generate.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    generate.add_argument("--timeout-seconds", type=float, default=120.0)
    generate.add_argument("--poll-interval", type=float, default=1.0)
    generate.add_argument("--http-timeout", type=float, default=30.0)
    generate.add_argument("--admin-results-limit", type=int, default=50)
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--skip-live-preflight", action="store_true", help="dry-run only: skip Coordinator /ready and /state checks for CI-safe protocol/package checks")
    generate.add_argument("--include-output", action="store_true")
    generate.add_argument("--stream", action="store_true", help="emit safe per-token progress while waiting for the final result")
    generate.add_argument("--json", action="store_true")

    p2pd = subparsers.add_parser("p2pd", help="Print or run the P2P discovery daemon command.")
    p2pd.add_argument("--host", default="127.0.0.1")
    p2pd.add_argument("--port", type=int, default=8788)
    p2pd.add_argument("--swarm-id", default="default")
    p2pd.add_argument("--peer-id", default="")
    p2pd.add_argument("--role", choices=["coordinator", "miner", "observer"], default="observer")
    p2pd.add_argument("--peer-url", default="")
    p2pd.add_argument("--coordinator-url", default="")
    p2pd.add_argument("--backend", choices=["", "cpu", "cuda"], default="")
    p2pd.add_argument("--stage-role", choices=["", "stage0", "stage1", "both"], default="")
    p2pd.add_argument("--stage-capability", action="append", default=[])
    p2pd.add_argument("--bootstrap", action="append", default=[])
    p2pd.add_argument("--ttl-seconds", type=float, default=60.0)
    p2pd.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    p2pd.add_argument("--require-signed", action="store_true")
    p2pd.add_argument("--signature-max-age-seconds", type=float, default=3600.0)
    p2pd.add_argument("--print-peer", action="store_true")
    p2pd.add_argument("--run", action="store_true")
    p2pd.add_argument("--json", action="store_true")

    p2p_daemon = subparsers.add_parser("p2p-daemon", help="Print or run the real-P2P provider discovery daemon command.")
    p2p_daemon.add_argument("--host", default="127.0.0.1")
    p2p_daemon.add_argument("--port", type=int, default=8888)
    p2p_daemon.add_argument("--public-host", default="")
    p2p_daemon.add_argument("--swarm-id", default="default")
    p2p_daemon.add_argument("--node-id", default="")
    p2p_daemon.add_argument("--role", choices=["coordinator", "miner", "observer"], default="observer")
    p2p_daemon.add_argument("--peer-url", default="")
    p2p_daemon.add_argument("--coordinator-url", default="")
    p2p_daemon.add_argument("--backend", choices=["", "cpu", "cuda"], default="")
    p2p_daemon.add_argument("--stage-role", choices=["", "stage0", "stage1", "both"], default="")
    p2p_daemon.add_argument("--stage-capability", action="append", default=[])
    p2p_daemon.add_argument("--bootstrap", action="append", default=[])
    p2p_daemon.add_argument("--ttl-seconds", type=float, default=60.0)
    p2p_daemon.add_argument("--record-secret", "--peer-secret", dest="record_secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    p2p_daemon.add_argument("--require-signed", action="store_true")
    p2p_daemon.add_argument("--signature-max-age-seconds", type=float, default=3600.0)
    p2p_daemon.add_argument("--discovery-backend", default=DEFAULT_DISCOVERY_BACKEND)
    p2p_daemon.add_argument("--libp2p-host", default="127.0.0.1")
    p2p_daemon.add_argument("--libp2p-port", type=int, default=0)
    p2p_daemon.add_argument("--libp2p-public-host", default="")
    p2p_daemon.add_argument("--peer-key-file", default="")
    p2p_daemon.add_argument("--kad-protocol", default="/crowdtensor/kad/1.0.0")
    p2p_daemon.add_argument("--print-record", action="store_true")
    p2p_daemon.add_argument("--run", action="store_true")
    p2p_daemon.add_argument("--json", action="store_true")

    peer = subparsers.add_parser("peer", help="Run or query the P2P-lite discovery layer.")
    peer_subparsers = peer.add_subparsers(dest="peer_action", required=True)
    peer_check = peer_subparsers.add_parser("check", help="Run the P2P-lite discovery check.")
    peer_check.add_argument("--timeout-seconds", type=int, default=60)
    peer_check.add_argument("--json", action="store_true")

    peer_daemon = peer_subparsers.add_parser("daemon", help="Print or run the P2P-lite daemon command.")
    peer_daemon.add_argument("--host", default="127.0.0.1")
    peer_daemon.add_argument("--port", type=int, default=8788)
    peer_daemon.add_argument("--swarm-id", default="default")
    peer_daemon.add_argument("--peer-id", default="")
    peer_daemon.add_argument("--role", choices=["coordinator", "miner", "observer"], default="observer")
    peer_daemon.add_argument("--peer-url", default="")
    peer_daemon.add_argument("--coordinator-url", default="")
    peer_daemon.add_argument("--backend", choices=["", "cpu", "cuda"], default="")
    peer_daemon.add_argument("--stage-role", choices=["", "stage0", "stage1", "both"], default="")
    peer_daemon.add_argument("--stage-capability", action="append", default=[])
    peer_daemon.add_argument("--bootstrap", action="append", default=[])
    peer_daemon.add_argument("--ttl-seconds", type=float, default=60.0)
    peer_daemon.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    peer_daemon.add_argument("--require-signed", action="store_true")
    peer_daemon.add_argument("--signature-max-age-seconds", type=float, default=3600.0)
    peer_daemon.add_argument("--print-peer", action="store_true")
    peer_daemon.add_argument("--run", action="store_true")
    peer_daemon.add_argument("--json", action="store_true")

    peer_resolve = peer_subparsers.add_parser("resolve", help="Resolve a Coordinator route through a P2P-lite bootstrap.")
    peer_resolve.add_argument("--bootstrap", required=True)
    peer_resolve.add_argument("--prompt-text", default="CrowdTensor route probe")
    peer_resolve.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    peer_resolve.add_argument("--max-new-tokens", type=int, default=16)
    peer_resolve.add_argument("--http-timeout", type=float, default=5.0)
    peer_resolve.add_argument("--json", action="store_true")

    peer_announce = peer_subparsers.add_parser("announce", help="Announce one peer to a P2P-lite bootstrap.")
    peer_announce.add_argument("--bootstrap", required=True)
    peer_announce.add_argument("--swarm-id", default="default")
    peer_announce.add_argument("--peer-id", required=True)
    peer_announce.add_argument("--role", choices=["coordinator", "miner", "observer"], default="observer")
    peer_announce.add_argument("--peer-url", default="")
    peer_announce.add_argument("--coordinator-url", default="")
    peer_announce.add_argument("--backend", choices=["", "cpu", "cuda"], default="")
    peer_announce.add_argument("--stage-role", choices=["", "stage0", "stage1", "both"], default="")
    peer_announce.add_argument("--stage-capability", action="append", default=[])
    peer_announce.add_argument("--ttl-seconds", type=float, default=60.0)
    peer_announce.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    peer_announce.add_argument("--http-timeout", type=float, default=5.0)
    peer_announce.add_argument("--json", action="store_true")

    product_rc = subparsers.add_parser(
        "public-swarm-product-rc",
        help="Build the Coordinator product surface + session protocol + P2P-lite RC artifact.",
    )
    product_rc.add_argument("--output-dir", default="dist/public-swarm-product-rc")
    product_rc.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    product_rc.add_argument("--max-new-tokens", type=int, default=16)
    product_rc.add_argument("--timeout-seconds", type=int, default=120)
    product_rc.add_argument("--json", action="store_true")

    home = subparsers.add_parser(
        "home-infer",
        help="Run the CPU-only read-only home inference proof and collect safe artifacts.",
    )
    home.add_argument("--output-dir", default="dist/home-infer")
    home.add_argument("--port", type=int, default=8909)
    home.add_argument("--request-count", type=int, default=4)
    home.add_argument("--scenario-id", default="route-baseline")
    home.add_argument("--timeout-seconds", type=int, default=180)
    home.add_argument("--runtime-report", default="")
    home.add_argument("--json", action="store_true")
    llm = subparsers.add_parser(
        "llm-infer",
        help="Run a local external_llm_infer proof against mock or operator-owned LLM runtime.",
    )
    llm.add_argument("--output-dir", default="dist/llm-infer")
    llm.add_argument("--port", type=int, default=8919)
    llm.add_argument("--request-count", type=int, default=3)
    llm.add_argument("--timeout-seconds", type=int, default=180)
    llm.add_argument("--mock", action="store_true", help="use the deterministic built-in mock runtime")
    llm.add_argument("--llm-runtime-cmd", default="")
    llm.add_argument("--llm-runtime-url", default="")
    llm.add_argument("--llm-runtime-api-key", default="")
    llm.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    llm.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    llm.add_argument("--json", action="store_true")
    cpu_infer = subparsers.add_parser(
        "cpu-infer",
        help="Run the CPU-only inference Beta aggregate proof.",
    )
    cpu_infer.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing", "beta-rc"], default="local")
    cpu_infer.add_argument("--output-dir", default="dist/cpu-infer")
    cpu_infer.add_argument("--base-port", type=int, default=8970)
    cpu_infer.add_argument("--request-count", type=int, default=4)
    cpu_infer.add_argument("--external-llm-request-count", type=int, default=3)
    cpu_infer.add_argument("--scenario-id", default="route-baseline")
    cpu_infer.add_argument("--workload", choices=["model-bundle", "external-llm", "all"], default="all")
    cpu_infer.add_argument("--coordinator-url", default="")
    cpu_infer.add_argument("--miner-id", default="remote-linux-1")
    cpu_infer.add_argument("--observer-token", default="")
    cpu_infer.add_argument("--admin-token", default="")
    cpu_infer.add_argument("--timeout-seconds", type=int, default=240)
    cpu_infer.add_argument("--remote-timeout-seconds", type=float, default=60.0)
    cpu_infer.add_argument("--kaggle-real-runtime-report", default="")
    cpu_infer.add_argument("--poll-interval", type=float, default=1.0)
    cpu_infer.add_argument("--mock", action="store_true")
    cpu_infer.add_argument("--llm-runtime-cmd", default="")
    cpu_infer.add_argument("--llm-runtime-url", default="")
    cpu_infer.add_argument("--llm-runtime-api-key", default="")
    cpu_infer.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    cpu_infer.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    cpu_infer.add_argument("--json", action="store_true")
    shard = subparsers.add_parser(
        "shard-infer",
        help="Run the CPU-only two-stage pipeline-sharded inference Alpha proof.",
    )
    shard.add_argument("--output-dir", default="dist/shard-infer")
    shard.add_argument("--port", type=int, default=9820)
    shard.add_argument("--request-count", type=int, default=4)
    shard.add_argument("--scenario-id", default="route-baseline")
    shard.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    shard.add_argument("--stage-mode", choices=["both", "split"], default="both")
    shard.add_argument("--require-distinct-stage-miners", action="store_true")
    shard.add_argument("--timeout-seconds", type=int, default=120)
    shard.add_argument("--json", action="store_true")
    micro_shard = subparsers.add_parser(
        "micro-llm-shard-infer",
        help="Run the CPU-only deterministic micro-LLM pipeline-sharded inference Alpha proof.",
    )
    micro_shard.add_argument("--output-dir", default="dist/micro-llm-shard-infer")
    micro_shard.add_argument("--port", type=int, default=9860)
    micro_shard.add_argument("--request-count", type=int, default=4)
    micro_shard.add_argument("--decode-steps", type=int, default=4)
    micro_shard.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    micro_shard.add_argument("--stage-mode", choices=["both", "split"], default="both")
    micro_shard.add_argument("--require-distinct-stage-miners", action="store_true")
    micro_shard.add_argument("--micro-llm-artifact", default="")
    micro_shard.add_argument("--prompt-texts", default="arn,ten")
    micro_shard.add_argument("--timeout-seconds", type=int, default=150)
    micro_shard.add_argument("--json", action="store_true")
    real_shard = subparsers.add_parser(
        "real-llm-shard-infer",
        help="Run the optional CPU-only tiny Hugging Face LLM pipeline-sharded inference Alpha proof.",
    )
    real_shard.add_argument("--output-dir", default="dist/real-llm-shard-infer")
    real_shard.add_argument("--port", type=int, default=9880)
    real_shard.add_argument("--request-count", type=int, default=1)
    real_shard.add_argument("--max-new-tokens", type=int, default=1)
    real_shard.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_shard.add_argument("--hf-cache-dir", default="")
    real_shard.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    real_shard.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    real_shard.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    real_shard.add_argument("--stage-mode", choices=["both", "split"], default="both")
    real_shard.add_argument("--require-distinct-stage-miners", action="store_true")
    real_shard.add_argument("--timeout-seconds", type=int, default=240)
    real_shard.add_argument("--json", action="store_true")
    micro_artifact = subparsers.add_parser(
        "micro-llm-artifact",
        help="Build or inspect the dependency-free file-backed Micro-LLM artifact.",
    )
    micro_artifact.add_argument("--output-dir", default="dist/micro-llm-artifact")
    micro_artifact.add_argument("--artifact-id", default="crowdtensor-micro-llm-alpha")
    micro_artifact.add_argument("--version", type=int, default=1)
    micro_artifact.add_argument("--inspect", action="store_true")
    micro_artifact.add_argument("--timeout-seconds", type=int, default=60)
    micro_artifact.add_argument("--json", action="store_true")
    shard_beta = subparsers.add_parser(
        "shard-infer-beta",
        help="Run the CPU-only remote pipeline-sharded inference Beta proof.",
    )
    shard_beta.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing"], default="remote-loopback")
    shard_beta.add_argument("--output-dir", default="dist/remote-sharded-inference")
    shard_beta.add_argument("--base-port", type=int, default=9830)
    shard_beta.add_argument("--request-count", type=int, default=4)
    shard_beta.add_argument("--scenario-id", default="route-baseline")
    shard_beta.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    shard_beta.add_argument("--stage-mode", choices=["both", "split"], default="both")
    shard_beta.add_argument("--require-distinct-stage-miners", action="store_true")
    shard_beta.add_argument("--coordinator-url", default="")
    shard_beta.add_argument("--observer-token", default="")
    shard_beta.add_argument("--admin-token", default="")
    shard_beta.add_argument("--timeout-seconds", type=int, default=180)
    shard_beta.add_argument("--remote-timeout-seconds", type=float, default=90.0)
    shard_beta.add_argument("--json", action="store_true")
    micro_shard_beta = subparsers.add_parser(
        "micro-llm-shard-infer-beta",
        help="Run the CPU-only remote micro-LLM pipeline-sharded inference Beta proof.",
    )
    micro_shard_beta.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing"], default="remote-loopback")
    micro_shard_beta.add_argument("--output-dir", default="dist/remote-micro-llm-sharded-inference")
    micro_shard_beta.add_argument("--base-port", type=int, default=9870)
    micro_shard_beta.add_argument("--request-count", type=int, default=4)
    micro_shard_beta.add_argument("--decode-steps", type=int, default=4)
    micro_shard_beta.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    micro_shard_beta.add_argument("--stage-mode", choices=["both", "split"], default="both")
    micro_shard_beta.add_argument("--require-distinct-stage-miners", action="store_true")
    micro_shard_beta.add_argument("--coordinator-url", default="")
    micro_shard_beta.add_argument("--observer-token", default="")
    micro_shard_beta.add_argument("--admin-token", default="")
    micro_shard_beta.add_argument("--timeout-seconds", type=int, default=180)
    micro_shard_beta.add_argument("--remote-timeout-seconds", type=float, default=90.0)
    micro_shard_beta.add_argument("--json", action="store_true")
    real_shard_beta = subparsers.add_parser(
        "real-llm-shard-infer-beta",
        help="Run the optional CPU-only remote tiny Hugging Face LLM pipeline-sharded inference Beta proof.",
    )
    real_shard_beta.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing"], default="remote-loopback")
    real_shard_beta.add_argument("--output-dir", default="dist/remote-real-llm-sharded-inference")
    real_shard_beta.add_argument("--base-port", type=int, default=9890)
    real_shard_beta.add_argument("--request-count", type=int, default=1)
    real_shard_beta.add_argument("--max-new-tokens", type=int, default=1)
    real_shard_beta.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_shard_beta.add_argument("--hf-cache-dir", default="")
    real_shard_beta.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    real_shard_beta.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    real_shard_beta.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    real_shard_beta.add_argument("--stage-mode", choices=["both", "split"], default="split")
    real_shard_beta.add_argument("--require-distinct-stage-miners", action="store_true")
    real_shard_beta.add_argument("--coordinator-url", default="")
    real_shard_beta.add_argument("--observer-token", default="")
    real_shard_beta.add_argument("--admin-token", default="")
    real_shard_beta.add_argument("--timeout-seconds", type=int, default=300)
    real_shard_beta.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    real_shard_beta.add_argument("--json", action="store_true")
    micro_live_rc = subparsers.add_parser(
        "micro-llm-live-rc",
        help="Run the stage-aware micro-LLM live two-node RC proof.",
    )
    micro_live_rc.add_argument("--mode", choices=["local-generated", "external-existing"], default="local-generated")
    micro_live_rc.add_argument("--output-dir", default="dist/micro-llm-live-rc")
    micro_live_rc.add_argument("--kaggle-output-dir", default="")
    micro_live_rc.add_argument("--public-host", default="24.199.118.54")
    micro_live_rc.add_argument("--port", type=int, default=9180)
    micro_live_rc.add_argument("--coordinator-url", default="")
    micro_live_rc.add_argument("--miner-id", default="kaggle-cpu-1")
    micro_live_rc.add_argument("--request-count", type=int, default=2)
    micro_live_rc.add_argument("--decode-steps", type=int, default=3)
    micro_live_rc.add_argument("--micro-llm-artifact", default="")
    micro_live_rc.add_argument("--timeout-seconds", type=int, default=240)
    micro_live_rc.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    micro_live_rc.add_argument("--startup-timeout", type=float, default=20.0)
    micro_live_rc.add_argument("--process-exit-timeout", type=float, default=10.0)
    micro_live_rc.add_argument("--poll-interval", type=float, default=1.0)
    micro_live_rc.add_argument("--http-timeout", type=float, default=5.0)
    micro_live_rc.add_argument("--artifact-timeout", type=float, default=60.0)
    micro_live_rc.add_argument("--admin-results-limit", type=int, default=10)
    micro_live_rc.add_argument("--lease-seconds", type=float, default=15.0)
    micro_live_rc.add_argument("--compute-seconds", type=float, default=0.2)
    micro_live_rc.add_argument("--heartbeat-interval", type=float, default=0.1)
    micro_live_rc.add_argument("--idle-sleep", type=float, default=0.2)
    micro_live_rc.add_argument("--max-request-attempts", type=int, default=20)
    micro_live_rc.add_argument("--observer-token", default="")
    micro_live_rc.add_argument("--admin-token", default="")
    micro_live_rc.add_argument("--json", action="store_true")
    real_live_rc = subparsers.add_parser(
        "real-llm-live-rc",
        help="Run the real small-LLM live two-node RC proof with generated stage upload packages.",
    )
    real_live_rc.add_argument("--mode", choices=["local-generated", "kaggle-generated", "external-existing"], default="local-generated")
    real_live_rc.add_argument("--output-dir", default="dist/real-llm-live-rc")
    real_live_rc.add_argument("--public-host", default="24.199.118.54")
    real_live_rc.add_argument("--bind-host", default="0.0.0.0")
    real_live_rc.add_argument("--port", type=int, default=9184)
    real_live_rc.add_argument("--coordinator-url", default="")
    real_live_rc.add_argument("--miner-id", default="kaggle-real-llm")
    real_live_rc.add_argument("--request-count", type=int, default=1)
    real_live_rc.add_argument("--max-new-tokens", type=int, default=1)
    real_live_rc.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_live_rc.add_argument("--hf-cache-dir", default="")
    real_live_rc.add_argument("--timeout-seconds", type=int, default=300)
    real_live_rc.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    real_live_rc.add_argument("--startup-timeout", type=float, default=30.0)
    real_live_rc.add_argument("--process-exit-timeout", type=float, default=20.0)
    real_live_rc.add_argument("--poll-interval", type=float, default=1.0)
    real_live_rc.add_argument("--http-timeout", type=float, default=30.0)
    real_live_rc.add_argument("--lease-seconds", type=float, default=15.0)
    real_live_rc.add_argument("--compute-seconds", type=float, default=0.2)
    real_live_rc.add_argument("--heartbeat-interval", type=float, default=0.1)
    real_live_rc.add_argument("--idle-sleep", type=float, default=0.5)
    real_live_rc.add_argument("--max-request-attempts", type=int, default=120)
    real_live_rc.add_argument("--observer-token", default="")
    real_live_rc.add_argument("--admin-token", default="")
    real_live_rc.add_argument("--json", action="store_true")
    real_internet_alpha = subparsers.add_parser(
        "real-llm-internet-alpha",
        help="Run the real Internet Swarm Inference Alpha proof with local requeue and external verification modes.",
    )
    real_internet_alpha.add_argument("--mode", choices=["local-generated", "package", "external-existing"], default="local-generated")
    real_internet_alpha.add_argument("--output-dir", default="dist/real-llm-internet-alpha")
    real_internet_alpha.add_argument("--public-host", default="24.199.118.54")
    real_internet_alpha.add_argument("--bind-host", default="0.0.0.0")
    real_internet_alpha.add_argument("--port", type=int, default=9186)
    real_internet_alpha.add_argument("--base-port", type=int, default=9188)
    real_internet_alpha.add_argument("--coordinator-url", default="")
    real_internet_alpha.add_argument("--miner-id", default="internet-real-llm")
    real_internet_alpha.add_argument("--request-count", type=int, default=1)
    real_internet_alpha.add_argument("--max-new-tokens", type=int, default=1)
    real_internet_alpha.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_internet_alpha.add_argument("--hf-cache-dir", default="")
    real_internet_alpha.add_argument("--timeout-seconds", type=int, default=300)
    real_internet_alpha.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    real_internet_alpha.add_argument("--startup-timeout", type=float, default=30.0)
    real_internet_alpha.add_argument("--process-exit-timeout", type=float, default=20.0)
    real_internet_alpha.add_argument("--poll-interval", type=float, default=1.0)
    real_internet_alpha.add_argument("--http-timeout", type=float, default=30.0)
    real_internet_alpha.add_argument("--lease-seconds", type=float, default=15.0)
    real_internet_alpha.add_argument("--compute-seconds", type=float, default=0.2)
    real_internet_alpha.add_argument("--heartbeat-interval", type=float, default=0.1)
    real_internet_alpha.add_argument("--idle-sleep", type=float, default=0.5)
    real_internet_alpha.add_argument("--max-request-attempts", type=int, default=120)
    real_internet_alpha.add_argument("--observer-token", default="")
    real_internet_alpha.add_argument("--admin-token", default="")
    real_internet_alpha.add_argument("--skip-requeue", action="store_true")
    real_internet_alpha.add_argument("--json", action="store_true")
    real_internet_beta = subparsers.add_parser(
        "real-llm-internet-beta",
        help="Run the real Internet Swarm Inference Beta Kaggle automation with cleanup-backed evidence.",
    )
    real_internet_beta.add_argument("--mode", choices=["kaggle-auto"], default="kaggle-auto")
    real_internet_beta.add_argument("--output-dir", default="dist/real-llm-internet-beta-kaggle-auto")
    real_internet_beta.add_argument("--public-host", default="24.199.118.54")
    real_internet_beta.add_argument("--bind-host", default="0.0.0.0")
    real_internet_beta.add_argument("--port", type=int, default=9190)
    real_internet_beta.add_argument("--base-port", type=int, default=9191)
    real_internet_beta.add_argument("--ready-url", default="")
    real_internet_beta.add_argument("--coordinator-url", default="")
    real_internet_beta.add_argument("--miner-id", default="internet-real-llm-beta")
    real_internet_beta.add_argument("--request-count", type=int, default=2)
    real_internet_beta.add_argument("--max-new-tokens", type=int, default=1)
    real_internet_beta.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_internet_beta.add_argument("--hf-cache-dir", default="")
    real_internet_beta.add_argument("--kaggle-owner", default="")
    real_internet_beta.add_argument("--dataset-slug", default="")
    real_internet_beta.add_argument("--dataset-title", default="CrowdTensor Real LLM Internet Beta Package")
    real_internet_beta.add_argument("--kernel-slug-prefix", default="")
    real_internet_beta.add_argument("--kernel-title-prefix", default="CrowdTensor Real LLM Internet Beta Miner")
    real_internet_beta.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    real_internet_beta.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    real_internet_beta.add_argument("--skip-kaggle-cleanup", action="store_true")
    real_internet_beta.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="none")
    real_internet_beta.add_argument("--timeout-seconds", type=float, default=900.0)
    real_internet_beta.add_argument("--remote-timeout-seconds", type=float, default=720.0)
    real_internet_beta.add_argument("--startup-timeout", type=float, default=45.0)
    real_internet_beta.add_argument("--process-exit-timeout", type=float, default=20.0)
    real_internet_beta.add_argument("--poll-interval", type=float, default=1.0)
    real_internet_beta.add_argument("--http-timeout", type=float, default=30.0)
    real_internet_beta.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    real_internet_beta.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    real_internet_beta.add_argument("--kaggle-status-timeout-seconds", type=float, default=900.0)
    real_internet_beta.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    real_internet_beta.add_argument("--lease-seconds", type=float, default=15.0)
    real_internet_beta.add_argument("--compute-seconds", type=float, default=0.2)
    real_internet_beta.add_argument("--victim-compute-seconds", type=float, default=45.0)
    real_internet_beta.add_argument("--heartbeat-interval", type=float, default=0.1)
    real_internet_beta.add_argument("--idle-sleep", type=float, default=0.5)
    real_internet_beta.add_argument("--claim-observe-timeout", type=float, default=180.0)
    real_internet_beta.add_argument("--requeue-timeout", type=float, default=120.0)
    real_internet_beta.add_argument("--max-request-attempts", type=int, default=240)
    real_internet_beta.add_argument("--json", action="store_true")
    swarm = subparsers.add_parser(
        "swarm-infer-beta",
        help="Prepare, run, verify, collect, or clean the user-facing real tiny-LLM Swarm Inference Beta.",
    )
    swarm_subparsers = swarm.add_subparsers(dest="swarm_action", required=True)

    def add_swarm_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", default="dist/swarm-inference-beta")
        target.add_argument("--coordinator-url", default="http://127.0.0.1:9200")
        target.add_argument("--port", type=int, default=9200)
        target.add_argument("--bind-host", default="0.0.0.0")
        target.add_argument("--miner-id-prefix", default="swarm-beta")
        target.add_argument("--request-count", type=int, default=2)
        target.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
        target.add_argument("--hf-cache-dir", default="")
        target.add_argument("--timeout-seconds", type=float, default=360.0)
        target.add_argument("--remote-timeout-seconds", type=float, default=240.0)
        target.add_argument("--http-timeout", type=float, default=30.0)
        target.add_argument("--json", action="store_true")

    swarm_prepare = swarm_subparsers.add_parser("prepare", help="Create operator runbook and stage0/stage1 Miner join packs.")
    add_swarm_common(swarm_prepare)
    swarm_prepare.add_argument("--observer-token", default="")
    swarm_prepare.add_argument("--admin-token", default="")
    swarm_prepare.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_prepare.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_prepare.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_prepare.add_argument("--max-request-attempts", type=int, default=120)
    swarm_prepare.add_argument("--replace", action="store_true")

    swarm_coordinator = swarm_subparsers.add_parser("coordinator", help="Print or run the generated Coordinator command.")
    add_swarm_common(swarm_coordinator)
    swarm_coordinator.add_argument("--observer-token", default="")
    swarm_coordinator.add_argument("--admin-token", default="")
    swarm_coordinator.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_coordinator.add_argument("--run", action="store_true")

    swarm_miner = swarm_subparsers.add_parser("miner", help="Print or run a generated stage Miner command.")
    add_swarm_common(swarm_miner)
    swarm_miner.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    swarm_miner.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_miner.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_miner.add_argument("--max-request-attempts", type=int, default=120)
    swarm_miner.add_argument("--run", action="store_true")

    swarm_verify = swarm_subparsers.add_parser("verify", help="Verify a running two-stage Swarm Inference Beta session.")
    add_swarm_common(swarm_verify)
    swarm_verify.add_argument("--observer-token", default="")
    swarm_verify.add_argument("--admin-token", default="")
    swarm_verify.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    swarm_verify.add_argument("--real-internet-beta-report", default="")

    swarm_collect = swarm_subparsers.add_parser("collect", help="Collect redacted evidence from a running Swarm Inference Beta.")
    add_swarm_common(swarm_collect)
    swarm_collect.add_argument("--observer-token", default="")
    swarm_collect.add_argument("--admin-token", default="")
    swarm_collect.add_argument("--miner-id", default="")
    swarm_collect.add_argument("--artifact-timeout", type=float, default=60.0)

    swarm_live = swarm_subparsers.add_parser("live", help="Run the side-effectful public Kaggle auto proof for Swarm Inference Beta.")
    swarm_live.add_argument("--output-dir", default="dist/swarm-inference-beta-live")
    swarm_live.add_argument("--public-host", default="24.199.118.54")
    swarm_live.add_argument("--bind-host", default="0.0.0.0")
    swarm_live.add_argument("--port", type=int, default=9210)
    swarm_live.add_argument("--base-port", type=int, default=9211)
    swarm_live.add_argument("--ready-url", default="")
    swarm_live.add_argument("--coordinator-url", default="")
    swarm_live.add_argument("--miner-id-prefix", default="swarm-beta-live")
    swarm_live.add_argument("--request-count", type=int, default=2)
    swarm_live.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    swarm_live.add_argument("--hf-cache-dir", default="")
    swarm_live.add_argument("--kaggle-owner", default="")
    swarm_live.add_argument("--dataset-slug", default="")
    swarm_live.add_argument("--dataset-title", default="CrowdTensor Swarm Inference Beta Live")
    swarm_live.add_argument("--kernel-slug-prefix", default="crowdtensor-swarm-inference-beta-live")
    swarm_live.add_argument("--kernel-title-prefix", default="CrowdTensor Swarm Inference Beta Live")
    swarm_live.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    swarm_live.add_argument("--skip-kaggle-cleanup", action="store_true")
    swarm_live.add_argument("--keep-live-private-artifacts", action="store_true")
    swarm_live.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="none")
    swarm_live.add_argument("--timeout-seconds", type=float, default=300.0)
    swarm_live.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    swarm_live.add_argument("--startup-timeout", type=float, default=60.0)
    swarm_live.add_argument("--process-exit-timeout", type=float, default=10.0)
    swarm_live.add_argument("--poll-interval", type=float, default=1.0)
    swarm_live.add_argument("--http-timeout", type=float, default=30.0)
    swarm_live.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    swarm_live.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    swarm_live.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    swarm_live.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    swarm_live.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_live.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_live.add_argument("--victim-compute-seconds", type=float, default=45.0)
    swarm_live.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_live.add_argument("--idle-sleep", type=float, default=0.2)
    swarm_live.add_argument("--claim-observe-timeout", type=float, default=180.0)
    swarm_live.add_argument("--requeue-timeout", type=float, default=120.0)
    swarm_live.add_argument("--max-request-attempts", type=int, default=240)
    swarm_live.add_argument("--json", action="store_true")

    swarm_clean = swarm_subparsers.add_parser("clean", help="Dry-run or delete known Swarm Inference Beta artifacts.")
    swarm_clean.add_argument("--output-dir", default="dist/swarm-inference-beta")
    swarm_clean.add_argument("--timeout-seconds", type=float, default=60.0)
    swarm_clean.add_argument("--apply", action="store_true")
    swarm_clean.add_argument("--include-private", action="store_true")
    swarm_clean.add_argument("--remove-empty-dir", action="store_true")
    swarm_clean.add_argument("--json", action="store_true")
    swarm_session = subparsers.add_parser(
        "swarm-session",
        help="Run the Public Swarm Inference Alpha session wrapper around real tiny-LLM split inference.",
    )
    swarm_session.add_argument("--mode", choices=["local-generated", "live-kaggle"], default="live-kaggle")
    swarm_session.add_argument("--output-dir", default="dist/public-swarm-inference-alpha")
    swarm_session.add_argument("--public-host", default="24.199.118.54")
    swarm_session.add_argument("--bind-host", default="0.0.0.0")
    swarm_session.add_argument("--port", type=int, default=9220)
    swarm_session.add_argument("--base-port", type=int, default=9221)
    swarm_session.add_argument("--ready-url", default="")
    swarm_session.add_argument("--coordinator-url", default="")
    swarm_session.add_argument("--miner-id-prefix", default="public-swarm-alpha")
    swarm_session.add_argument("--request-count", type=int, default=2)
    swarm_session.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    swarm_session.add_argument("--hf-cache-dir", default="")
    swarm_session.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="kill-stage0-after-claim")
    swarm_session.add_argument("--skip-local-requeue", action="store_true")
    swarm_session.add_argument("--kaggle-owner", default="")
    swarm_session.add_argument("--dataset-slug", default="")
    swarm_session.add_argument("--dataset-title", default="CrowdTensor Public Swarm Inference Alpha")
    swarm_session.add_argument("--kernel-slug-prefix", default="crowdtensor-public-swarm-alpha")
    swarm_session.add_argument("--kernel-title-prefix", default="CrowdTensor Public Swarm Inference Alpha")
    swarm_session.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    swarm_session.add_argument("--skip-kaggle-cleanup", action="store_true")
    swarm_session.add_argument("--keep-live-private-artifacts", action="store_true")
    swarm_session.add_argument("--keep-child-artifacts", action="store_true")
    swarm_session.add_argument("--timeout-seconds", type=float, default=300.0)
    swarm_session.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    swarm_session.add_argument("--startup-timeout", type=float, default=60.0)
    swarm_session.add_argument("--process-exit-timeout", type=float, default=10.0)
    swarm_session.add_argument("--poll-interval", type=float, default=1.0)
    swarm_session.add_argument("--http-timeout", type=float, default=30.0)
    swarm_session.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    swarm_session.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    swarm_session.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    swarm_session.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    swarm_session.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_session.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_session.add_argument("--victim-compute-seconds", type=float, default=45.0)
    swarm_session.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_session.add_argument("--idle-sleep", type=float, default=0.2)
    swarm_session.add_argument("--claim-observe-timeout", type=float, default=180.0)
    swarm_session.add_argument("--requeue-timeout", type=float, default=120.0)
    swarm_session.add_argument("--max-request-attempts", type=int, default=240)
    swarm_session.add_argument("--json", action="store_true")
    public_swarm_rc = subparsers.add_parser(
        "public-swarm-alpha-rc",
        help="Build the Public Swarm Inference Alpha release-candidate evidence artifact.",
    )
    public_swarm_rc.add_argument("--mode", choices=["evidence-import", "local-smoke"], default="evidence-import")
    public_swarm_rc.add_argument("--output-dir", default="dist/public-swarm-inference-alpha-rc")
    public_swarm_rc.add_argument(
        "--stage0-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/"
            "public_swarm_inference_alpha.json"
        ),
    )
    public_swarm_rc.add_argument(
        "--stage1-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/"
            "public_swarm_inference_alpha.json"
        ),
    )
    public_swarm_rc.add_argument("--summary-report", default="dist/public-swarm-inference-alpha-live-requeue-summary.json")
    public_swarm_rc.add_argument("--request-count", type=int, default=2)
    public_swarm_rc.add_argument("--timeout-seconds", type=float, default=120.0)
    public_swarm_rc.add_argument("--json", action="store_true")
    public_swarm_beta = subparsers.add_parser(
        "public-swarm-beta",
        help="Prepare, run, verify, collect, or validate Public Swarm Inference Beta.",
    )
    public_beta_subparsers = public_swarm_beta.add_subparsers(dest="public_swarm_beta_action", required=True)

    def add_public_beta_base(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", default="dist/public-swarm-inference-beta")
        target.add_argument("--request-count", type=int, default=1)
        target.add_argument("--timeout-seconds", type=float, default=300.0)
        target.add_argument("--json", action="store_true")

    def add_public_beta_runtime(target: argparse.ArgumentParser) -> None:
        target.add_argument("--coordinator-url", default="http://127.0.0.1:9200")
        target.add_argument("--port", type=int, default=9200)
        target.add_argument("--base-port", type=int, default=9290)
        target.add_argument("--bind-host", default="0.0.0.0")
        target.add_argument("--miner-id-prefix", default="public-swarm-beta")
        target.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
        target.add_argument("--hf-cache-dir", default="")
        target.add_argument("--remote-timeout-seconds", type=float, default=240.0)
        target.add_argument("--http-timeout", type=float, default=30.0)

    def add_public_beta_tokens(target: argparse.ArgumentParser) -> None:
        target.add_argument("--observer-token", default="")
        target.add_argument("--admin-token", default="")

    public_beta_prepare = public_beta_subparsers.add_parser("prepare", help="Create Public Beta stage0/stage1 join packs.")
    add_public_beta_base(public_beta_prepare)
    add_public_beta_runtime(public_beta_prepare)
    add_public_beta_tokens(public_beta_prepare)
    public_beta_prepare.add_argument("--lease-seconds", type=float, default=15.0)
    public_beta_prepare.add_argument("--compute-seconds", type=float, default=0.2)
    public_beta_prepare.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_beta_prepare.add_argument("--max-request-attempts", type=int, default=120)
    public_beta_prepare.add_argument("--replace", action="store_true")

    public_beta_coordinator = public_beta_subparsers.add_parser("coordinator", help="Print or run the Public Beta Coordinator command.")
    add_public_beta_base(public_beta_coordinator)
    add_public_beta_runtime(public_beta_coordinator)
    add_public_beta_tokens(public_beta_coordinator)
    public_beta_coordinator.add_argument("--lease-seconds", type=float, default=15.0)
    public_beta_coordinator.add_argument("--run", action="store_true")

    public_beta_miner = public_beta_subparsers.add_parser("miner", help="Print or run a Public Beta stage Miner command.")
    add_public_beta_base(public_beta_miner)
    add_public_beta_runtime(public_beta_miner)
    public_beta_miner.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    public_beta_miner.add_argument("--compute-seconds", type=float, default=0.2)
    public_beta_miner.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_beta_miner.add_argument("--max-request-attempts", type=int, default=120)
    public_beta_miner.add_argument("--run", action="store_true")

    public_beta_verify = public_beta_subparsers.add_parser("verify", help="Verify a running Public Beta two-stage session.")
    add_public_beta_base(public_beta_verify)
    add_public_beta_runtime(public_beta_verify)
    add_public_beta_tokens(public_beta_verify)
    public_beta_verify.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    public_beta_verify.add_argument("--real-internet-beta-report", default="")

    public_beta_collect = public_beta_subparsers.add_parser("collect", help="Collect redacted Public Beta evidence.")
    add_public_beta_base(public_beta_collect)
    add_public_beta_runtime(public_beta_collect)
    add_public_beta_tokens(public_beta_collect)
    public_beta_collect.add_argument("--miner-id", default="")
    public_beta_collect.add_argument("--artifact-timeout", type=float, default=60.0)

    public_beta_clean = public_beta_subparsers.add_parser("clean", help="Dry-run or delete known Public Beta generated files.")
    add_public_beta_base(public_beta_clean)
    public_beta_clean.add_argument("--apply", action="store_true")
    public_beta_clean.add_argument("--include-private", action="store_true")
    public_beta_clean.add_argument("--remove-empty-dir", action="store_true")

    public_beta_product = public_beta_subparsers.add_parser("product-beta", help="Validate the product-shaped Public Swarm Inference Beta aggregate.")
    add_public_beta_base(public_beta_product)
    public_beta_product.add_argument("--base-port", type=int, default=9290)
    public_beta_product.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_beta_product.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    public_beta_product.add_argument("--max-new-tokens", type=int, default=16)
    public_beta_product.add_argument("--cpu-request-count", type=int, default=2)
    public_beta_product.add_argument("--external-llm-request-count", type=int, default=1)
    public_beta_product.add_argument("--scenario-id", default="route-baseline")
    public_beta_product.add_argument("--cpu-timeout-seconds", type=float, default=180.0)

    public_beta_loopback = public_beta_subparsers.add_parser("local-loopback", help="Run a fresh local two-stage CPU tiny GPT split proof.")
    add_public_beta_base(public_beta_loopback)
    add_public_beta_runtime(public_beta_loopback)
    public_beta_loopback.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")

    public_beta_import = public_beta_subparsers.add_parser("evidence-import", help="Import retained Public Swarm Alpha RC live evidence.")
    add_public_beta_base(public_beta_import)
    public_beta_import.add_argument("--alpha-rc-report", default="dist/public-swarm-inference-alpha-rc/public_swarm_inference_alpha_rc.json")
    public_beta_import.add_argument(
        "--stage0-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/"
            "public_swarm_inference_alpha.json"
        ),
    )
    public_beta_import.add_argument(
        "--stage1-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/"
            "public_swarm_inference_alpha.json"
        ),
    )
    public_beta_import.add_argument("--summary-report", default="dist/public-swarm-inference-alpha-live-requeue-summary.json")
    public_beta_import.add_argument("--allow-missing-live-evidence", action="store_true")

    public_swarm_beta_rc = subparsers.add_parser(
        "public-swarm-beta-rc",
        help="Build the Coordinator-backed Public Swarm Inference Beta RC artifact.",
    )
    public_swarm_beta_rc.add_argument("public_swarm_beta_rc_mode", choices=["local-loopback", "package", "external-existing"])
    public_swarm_beta_rc.add_argument("--output-dir", default="dist/public-swarm-inference-beta-rc")
    public_swarm_beta_rc.add_argument("--base-port", type=int, default=9310)
    public_swarm_beta_rc.add_argument("--port", type=int, default=9310)
    public_swarm_beta_rc.add_argument("--public-host", default="127.0.0.1")
    public_swarm_beta_rc.add_argument("--bind-host", default="127.0.0.1")
    public_swarm_beta_rc.add_argument("--coordinator-url", default="")
    public_swarm_beta_rc.add_argument("--target", choices=["local", "kaggle"], default="local")
    public_swarm_beta_rc.add_argument("--miner-id-prefix", default="public-swarm-beta-rc")
    public_swarm_beta_rc.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_swarm_beta_rc.add_argument("--hf-cache-dir", default="")
    public_swarm_beta_rc.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    public_swarm_beta_rc.add_argument("--prompt-text", default="CrowdTensor public beta RC")
    public_swarm_beta_rc.add_argument("--prompt-texts", default="")
    public_swarm_beta_rc.add_argument("--stream-generation", action="store_true")
    public_swarm_beta_rc.add_argument("--scenario-id", default="route-baseline")
    public_swarm_beta_rc.add_argument("--request-count", type=int, default=1)
    public_swarm_beta_rc.add_argument("--max-new-tokens", type=int, default=2)
    public_swarm_beta_rc.add_argument("--cpu-request-count", type=int, default=1)
    public_swarm_beta_rc.add_argument("--external-llm-request-count", type=int, default=1)
    public_swarm_beta_rc.add_argument("--observer-token", default="")
    public_swarm_beta_rc.add_argument("--admin-token", default="")
    public_swarm_beta_rc.add_argument("--timeout-seconds", type=float, default=300.0)
    public_swarm_beta_rc.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    public_swarm_beta_rc.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    public_swarm_beta_rc.add_argument("--startup-timeout", type=float, default=45.0)
    public_swarm_beta_rc.add_argument("--process-exit-timeout", type=float, default=20.0)
    public_swarm_beta_rc.add_argument("--poll-interval", type=float, default=1.0)
    public_swarm_beta_rc.add_argument("--http-timeout", type=float, default=10.0)
    public_swarm_beta_rc.add_argument("--json", action="store_true")

    public_swarm_product_beta = subparsers.add_parser(
        "public-swarm-product-beta",
        help="Build the user-facing Public Swarm Product Beta artifact.",
    )
    public_swarm_product_beta.add_argument("public_swarm_product_beta_mode", choices=["local-loopback", "package", "external-existing"])
    public_swarm_product_beta.add_argument("--output-dir", default="dist/public-swarm-product-beta")
    public_swarm_product_beta.add_argument("--base-port", type=int, default=9320)
    public_swarm_product_beta.add_argument("--port", type=int, default=9320)
    public_swarm_product_beta.add_argument("--public-host", default="127.0.0.1")
    public_swarm_product_beta.add_argument("--bind-host", default="127.0.0.1")
    public_swarm_product_beta.add_argument("--coordinator-url", default="")
    public_swarm_product_beta.add_argument("--target", choices=["local", "kaggle"], default="local")
    public_swarm_product_beta.add_argument("--miner-id-prefix", default="public-swarm-product-beta")
    public_swarm_product_beta.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_swarm_product_beta.add_argument("--hf-cache-dir", default="")
    public_swarm_product_beta.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    public_swarm_product_beta.add_argument("--prompt-text", default="CrowdTensor product beta")
    public_swarm_product_beta.add_argument("--prompt-texts", default="")
    public_swarm_product_beta.add_argument("--stream-generation", action="store_true")
    public_swarm_product_beta.add_argument("--scenario-id", default="route-baseline")
    public_swarm_product_beta.add_argument("--request-count", type=int, default=1)
    public_swarm_product_beta.add_argument("--max-new-tokens", type=int, default=2)
    public_swarm_product_beta.add_argument("--cpu-request-count", type=int, default=1)
    public_swarm_product_beta.add_argument("--external-llm-request-count", type=int, default=1)
    public_swarm_product_beta.add_argument("--observer-token", default="")
    public_swarm_product_beta.add_argument("--admin-token", default="")
    public_swarm_product_beta.add_argument("--timeout-seconds", type=float, default=300.0)
    public_swarm_product_beta.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    public_swarm_product_beta.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    public_swarm_product_beta.add_argument("--startup-timeout", type=float, default=45.0)
    public_swarm_product_beta.add_argument("--process-exit-timeout", type=float, default=20.0)
    public_swarm_product_beta.add_argument("--poll-interval", type=float, default=1.0)
    public_swarm_product_beta.add_argument("--http-timeout", type=float, default=10.0)
    public_swarm_product_beta.add_argument("--json", action="store_true")

    public_real_llm_swarm_beta = subparsers.add_parser(
        "public-real-llm-swarm-beta",
        help="Build the top-level Public Real-LLM Swarm Inference Beta v1 artifact.",
    )
    public_real_llm_swarm_beta.add_argument(
        "public_real_llm_swarm_beta_mode",
        choices=["release", "local-smoke", "local-model-variant", "package", "evidence-import"],
        nargs="?",
        default="release",
    )
    public_real_llm_swarm_beta.add_argument("--output-dir", default="dist/public-real-llm-swarm-beta")
    public_real_llm_swarm_beta.add_argument("--product-report", default="dist/public-swarm-product-beta/public_swarm_product_beta.json")
    public_real_llm_swarm_beta.add_argument(
        "--external-report",
        default="dist/goal-final-infer-real-llm-internet-beta-import-16tok-gpu-summary-20260602/real_llm_internet_beta.json",
    )
    public_real_llm_swarm_beta.add_argument(
        "--p2p-report",
        default="dist/goal-final-infer-petals-candidate-16tok-batch-stream-composed-20260602/petals_class_p2p_candidate.json",
    )
    public_real_llm_swarm_beta.add_argument(
        "--usable-report",
        default="dist/goal-final-infer-usable-swarm-16tok-kv-cache-20260601/usable_swarm_inference.json",
    )
    public_real_llm_swarm_beta.add_argument(
        "--public-swarm-v2-report",
        default="dist/public-swarm-inference-v2/public_swarm_inference_v2.json",
    )
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-preview-report", default="dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json")
    public_real_llm_swarm_beta.add_argument(
        "--public-swarm-v2-real-p2p-report",
        default="dist/goal-final-infer-real-p2p-core-fresh-16tok-import-strict-20260601/real_p2p_swarm_inference_core_rc.json",
    )
    public_real_llm_swarm_beta.add_argument(
        "--p2p-runtime-smoke-report",
        default="dist/real-p2p-libp2p-kaggle-runtime-smoke-20260531-r6/real_p2p_swarm_inference_core_rc.json",
    )
    public_real_llm_swarm_beta.add_argument(
        "--p2p-external-report",
        default="dist/goal-final-infer-fresh-real-p2p-kaggle-16tok-20260601/real_p2p_swarm_inference_core_rc.json",
    )
    public_real_llm_swarm_beta.add_argument(
        "--p2p-requeue-report",
        default="dist/petals-p2p-candidate-live-stage0-20260531-r6/real_p2p_swarm_inference_core_rc.json",
    )
    public_real_llm_swarm_beta.add_argument(
        "--p2p-batch-stream-report",
        default="dist/goal-final-infer-public-swarm-v2-batch-stream-16tok-20260602/public_swarm_inference_v2.json",
    )
    public_real_llm_swarm_beta.add_argument(
        "--gpu-report",
        default="dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json",
    )
    public_real_llm_swarm_beta.add_argument("--base-port", type=int, default=9340)
    public_real_llm_swarm_beta.add_argument("--port", type=int, default=9340)
    public_real_llm_swarm_beta.add_argument("--p2p-port", type=int, default=9860)
    public_real_llm_swarm_beta.add_argument("--p2p-libp2p-port", type=int, default=10860)
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-p2p-port", type=int, default=9888)
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-coordinator-port", type=int, default=9889)
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-real-p2p-port", type=int, default=9890)
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-real-p2p-coordinator-port", type=int, default=9891)
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-real-p2p-libp2p-port", type=int, default=0)
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-real-p2p-discovery-backend", choices=sorted(DISCOVERY_BACKENDS), default="http-provider-store")
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-backend", choices=["cpu", "cuda"], default="cpu")
    public_real_llm_swarm_beta.add_argument("--public-host", default="127.0.0.1")
    public_real_llm_swarm_beta.add_argument("--bind-host", default="127.0.0.1")
    public_real_llm_swarm_beta.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_real_llm_swarm_beta.add_argument("--hf-cache-dir", default="")
    public_real_llm_swarm_beta.add_argument("--prompt-text", default="CrowdTensor public real LLM swarm beta")
    public_real_llm_swarm_beta.add_argument("--prompt-texts", default="")
    public_real_llm_swarm_beta.add_argument("--stream-generation", action="store_true")
    public_real_llm_swarm_beta.add_argument("--request-count", type=int, default=1)
    public_real_llm_swarm_beta.add_argument("--max-new-tokens", type=int, default=16)
    public_real_llm_swarm_beta.add_argument("--cpu-request-count", type=int, default=1)
    public_real_llm_swarm_beta.add_argument("--external-llm-request-count", type=int, default=1)
    public_real_llm_swarm_beta.add_argument("--timeout-seconds", type=float, default=300.0)
    public_real_llm_swarm_beta.add_argument("--public-swarm-v2-timeout-seconds", type=float, default=420.0)
    public_real_llm_swarm_beta.add_argument("--remote-timeout-seconds", type=float, default=240.0)
    public_real_llm_swarm_beta.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    public_real_llm_swarm_beta.add_argument("--startup-timeout", type=float, default=45.0)
    public_real_llm_swarm_beta.add_argument("--process-exit-timeout", type=float, default=20.0)
    public_real_llm_swarm_beta.add_argument("--poll-interval", type=float, default=1.0)
    public_real_llm_swarm_beta.add_argument("--http-timeout", type=float, default=10.0)
    public_real_llm_swarm_beta.add_argument("--json", action="store_true")

    usable_swarm = subparsers.add_parser(
        "usable-swarm",
        help="Build the Usable Swarm Inference v1 artifact for the p2pd/serve/join/generate path.",
    )
    usable_swarm.add_argument("usable_swarm_mode", choices=["local", "package", "evidence-import"], nargs="?", default="local")
    usable_swarm.add_argument("--output-dir", default="dist/usable-swarm-inference-v1")
    usable_swarm.add_argument("--p2p-report", default="dist/goal-final-infer-p2p-v06-16tok-kv-cache-20260601/p2p_swarm_inference_v06.json")
    usable_swarm.add_argument("--swarm-id", default="usable-swarm-v1")
    usable_swarm.add_argument("--public-host", default="127.0.0.1")
    usable_swarm.add_argument("--p2p-port", type=int, default=9788)
    usable_swarm.add_argument("--coordinator-port", type=int, default=9789)
    usable_swarm.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    usable_swarm.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    usable_swarm.add_argument("--hf-cache-dir", default="")
    usable_swarm.add_argument("--prompt-text", default="CrowdTensor usable swarm inference")
    usable_swarm.add_argument("--prompt-texts", default="")
    usable_swarm.add_argument("--stream-generation", action="store_true")
    usable_swarm.add_argument("--max-new-tokens", type=int, default=8)
    usable_swarm.add_argument("--startup-timeout", type=float, default=45.0)
    usable_swarm.add_argument("--timeout-seconds", type=float, default=240.0)
    usable_swarm.add_argument("--http-timeout", type=float, default=10.0)
    usable_swarm.add_argument("--preview-v04-report", default="dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json")
    usable_swarm.add_argument("--product-mvp-report", default="dist/public-swarm-preview-v04-distilgpt2-strict/product-mvp/product_swarm_mvp_check.json")
    usable_swarm.add_argument("--optional-model-report", default="dist/public-swarm-preview-v04-distilgpt2-strict/optional-model-mvp/product_swarm_mvp_check.json")
    usable_swarm.add_argument("--json", action="store_true")

    public_swarm_v2 = subparsers.add_parser(
        "public-swarm-v2",
        help="Build the Public Swarm Inference v2 artifact for the 16-token P2P path.",
    )
    public_swarm_v2.add_argument("public_swarm_v2_mode", choices=["local", "local-model-variant", "package", "evidence-import"], nargs="?", default="local")
    public_swarm_v2.add_argument("--output-dir", default="dist/public-swarm-inference-v2")
    public_swarm_v2.add_argument("--usable-report", default="dist/goal-final-infer-usable-swarm-16tok-kv-cache-20260601/usable_swarm_inference.json")
    public_swarm_v2.add_argument("--preview-report", default="dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json")
    public_swarm_v2.add_argument("--real-p2p-report", default="dist/goal-final-infer-real-p2p-core-fresh-16tok-import-strict-20260601/real_p2p_swarm_inference_core_rc.json")
    public_swarm_v2.add_argument("--gpu-report", default="dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json")
    public_swarm_v2.add_argument("--fresh-external-report", action="store_true")
    public_swarm_v2.add_argument("--fresh-external-attempt-report", default="")
    public_swarm_v2.add_argument("--public-host", default="127.0.0.1")
    public_swarm_v2.add_argument("--p2p-port", type=int, default=9888)
    public_swarm_v2.add_argument("--coordinator-port", type=int, default=9889)
    public_swarm_v2.add_argument("--real-p2p-port", type=int, default=9890)
    public_swarm_v2.add_argument("--real-p2p-coordinator-port", type=int, default=9891)
    public_swarm_v2.add_argument("--real-p2p-libp2p-port", type=int, default=0)
    public_swarm_v2.add_argument("--real-p2p-discovery-backend", choices=sorted(DISCOVERY_BACKENDS), default="http-provider-store")
    public_swarm_v2.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    public_swarm_v2.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_swarm_v2.add_argument("--hf-cache-dir", default="")
    public_swarm_v2.add_argument("--prompt-text", default="CrowdTensor Public Swarm Inference v2")
    public_swarm_v2.add_argument("--prompt-texts", default="")
    public_swarm_v2.add_argument("--stream-generation", action="store_true")
    public_swarm_v2.add_argument("--max-new-tokens", type=int, default=16)
    public_swarm_v2.add_argument("--startup-timeout", type=float, default=60.0)
    public_swarm_v2.add_argument("--timeout-seconds", type=float, default=420.0)
    public_swarm_v2.add_argument("--http-timeout", type=float, default=30.0)
    public_swarm_v2.add_argument("--json", action="store_true")

    preview = subparsers.add_parser(
        "preview",
        help="Build the Public Swarm Developer Preview artifact.",
    )
    preview.add_argument("preview_mode", choices=["local", "package", "external-existing", "evidence-import"])
    preview.add_argument("--output-dir", default="dist/public-swarm-developer-preview")
    preview.add_argument("--base-port", type=int, default=9330)
    preview.add_argument("--port", type=int, default=9330)
    preview.add_argument("--public-host", default="127.0.0.1")
    preview.add_argument("--bind-host", default="127.0.0.1")
    preview.add_argument("--coordinator-url", default="")
    preview.add_argument("--target", choices=["local", "kaggle"], default="local")
    preview.add_argument("--miner-id-prefix", default="public-swarm-preview")
    preview.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    preview.add_argument("--hf-cache-dir", default="")
    preview.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    preview.add_argument("--product-beta-report", default="dist/public-swarm-product-beta/public_swarm_product_beta.json")
    preview.add_argument("--prompt-text", default="CrowdTensor developer preview")
    preview.add_argument("--scenario-id", default="route-baseline")
    preview.add_argument("--request-count", type=int, default=1)
    preview.add_argument("--max-new-tokens", type=int, default=2)
    preview.add_argument("--cpu-request-count", type=int, default=1)
    preview.add_argument("--external-llm-request-count", type=int, default=1)
    preview.add_argument("--observer-token", default="")
    preview.add_argument("--admin-token", default="")
    preview.add_argument("--timeout-seconds", type=float, default=300.0)
    preview.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    preview.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    preview.add_argument("--startup-timeout", type=float, default=45.0)
    preview.add_argument("--process-exit-timeout", type=float, default=20.0)
    preview.add_argument("--poll-interval", type=float, default=1.0)
    preview.add_argument("--http-timeout", type=float, default=10.0)
    preview.add_argument("--json", action="store_true")

    live_preview = subparsers.add_parser(
        "live-preview",
        help="Build the Public Swarm Live Preview RC artifact.",
    )
    live_preview.add_argument("live_preview_mode", choices=["local-smoke", "package", "live-kaggle", "evidence-import"])
    live_preview.add_argument("--output-dir", default="dist/public-swarm-live-preview-rc")
    live_preview.add_argument("--public-host", default="24.199.118.54")
    live_preview.add_argument("--bind-host", default="0.0.0.0")
    live_preview.add_argument("--port", type=int, default=9340)
    live_preview.add_argument("--base-port", type=int, default=9341)
    live_preview.add_argument("--ready-url", default="")
    live_preview.add_argument("--coordinator-url", default="")
    live_preview.add_argument("--target", choices=["local", "kaggle"], default="kaggle")
    live_preview.add_argument("--miner-id-prefix", default="public-swarm-live-preview")
    live_preview.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    live_preview.add_argument("--hf-cache-dir", default="")
    live_preview.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    live_preview.add_argument("--developer-preview-report", default="dist/public-swarm-developer-preview/public_swarm_developer_preview.json")
    live_preview.add_argument("--alpha-rc-report", default="dist/public-swarm-inference-alpha-rc/public_swarm_inference_alpha_rc.json")
    live_preview.add_argument("--request-count", type=int, default=2)
    live_preview.add_argument("--max-new-tokens", type=int, default=2)
    live_preview.add_argument("--cpu-request-count", type=int, default=1)
    live_preview.add_argument("--external-llm-request-count", type=int, default=1)
    live_preview.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="kill-stage0-after-claim")
    live_preview.add_argument("--kaggle-owner", default="")
    live_preview.add_argument("--dataset-slug", default="")
    live_preview.add_argument("--dataset-title", default="CrowdTensor Public Swarm Live Preview RC")
    live_preview.add_argument("--kernel-slug-prefix", default="ct-live-preview")
    live_preview.add_argument("--kernel-title-prefix", default="CrowdTensor Public Swarm Live Preview RC")
    live_preview.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    live_preview.add_argument("--skip-kaggle-cleanup", action="store_true")
    live_preview.add_argument("--keep-live-private-artifacts", action="store_true")
    live_preview.add_argument("--keep-child-artifacts", action="store_true")
    live_preview.add_argument("--timeout-seconds", type=float, default=300.0)
    live_preview.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    live_preview.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    live_preview.add_argument("--startup-timeout", type=float, default=60.0)
    live_preview.add_argument("--process-exit-timeout", type=float, default=20.0)
    live_preview.add_argument("--poll-interval", type=float, default=1.0)
    live_preview.add_argument("--http-timeout", type=float, default=30.0)
    live_preview.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    live_preview.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    live_preview.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    live_preview.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    live_preview.add_argument("--lease-seconds", type=float, default=15.0)
    live_preview.add_argument("--compute-seconds", type=float, default=0.2)
    live_preview.add_argument("--victim-compute-seconds", type=float, default=45.0)
    live_preview.add_argument("--heartbeat-interval", type=float, default=0.1)
    live_preview.add_argument("--idle-sleep", type=float, default=0.2)
    live_preview.add_argument("--claim-observe-timeout", type=float, default=180.0)
    live_preview.add_argument("--requeue-timeout", type=float, default=120.0)
    live_preview.add_argument("--max-request-attempts", type=int, default=240)
    live_preview.add_argument("--json", action="store_true")

    operator_preview = subparsers.add_parser(
        "operator-preview",
        help="Build the Public Swarm v0.1 Operator Preview artifact.",
    )
    operator_preview.add_argument("operator_preview_mode", choices=["local-smoke", "package", "live-kaggle", "evidence-import"])
    operator_preview.add_argument("--output-dir", default="dist/public-swarm-operator-preview")
    operator_preview.add_argument("--public-host", default="24.199.118.54")
    operator_preview.add_argument("--bind-host", default="0.0.0.0")
    operator_preview.add_argument("--port", type=int, default=9350)
    operator_preview.add_argument("--base-port", type=int, default=9351)
    operator_preview.add_argument("--release-base-port", type=int, default=9360)
    operator_preview.add_argument("--ready-url", default="")
    operator_preview.add_argument("--coordinator-url", default="")
    operator_preview.add_argument("--target", choices=["local", "kaggle"], default="kaggle")
    operator_preview.add_argument("--miner-id-prefix", default="public-swarm-operator-preview")
    operator_preview.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    operator_preview.add_argument("--hf-cache-dir", default="")
    operator_preview.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    operator_preview.add_argument("--developer-preview-report", default="dist/public-swarm-developer-preview/public_swarm_developer_preview.json")
    operator_preview.add_argument("--alpha-rc-report", default="dist/public-swarm-inference-alpha-rc/public_swarm_inference_alpha_rc.json")
    operator_preview.add_argument(
        "--live-stage0-report",
        default=(
            "dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/"
            "public_swarm_live_preview_rc.json"
        ),
    )
    operator_preview.add_argument(
        "--live-stage1-report",
        default=(
            "dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/"
            "public_swarm_live_preview_rc.json"
        ),
    )
    operator_preview.add_argument("--release-readiness-report", default="dist/release-readiness/release_readiness.json")
    operator_preview.add_argument("--request-count", type=int, default=2)
    operator_preview.add_argument("--max-new-tokens", type=int, default=2)
    operator_preview.add_argument("--cpu-request-count", type=int, default=1)
    operator_preview.add_argument("--external-llm-request-count", type=int, default=1)
    operator_preview.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="kill-stage0-after-claim")
    operator_preview.add_argument("--kaggle-owner", default="")
    operator_preview.add_argument("--dataset-slug", default="")
    operator_preview.add_argument("--dataset-title", default="CrowdTensor Public Swarm v0.1 Operator Preview")
    operator_preview.add_argument("--kernel-slug-prefix", default="ct-operator-preview")
    operator_preview.add_argument("--kernel-title-prefix", default="CrowdTensor Operator Preview")
    operator_preview.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    operator_preview.add_argument("--skip-kaggle-cleanup", action="store_true")
    operator_preview.add_argument("--keep-live-private-artifacts", action="store_true")
    operator_preview.add_argument("--keep-child-artifacts", action="store_true")
    operator_preview.add_argument("--allow-dirty-release", action="store_true")
    operator_preview.add_argument("--timeout-seconds", type=float, default=300.0)
    operator_preview.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    operator_preview.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    operator_preview.add_argument("--release-timeout-seconds", type=float, default=240.0)
    operator_preview.add_argument("--startup-timeout", type=float, default=60.0)
    operator_preview.add_argument("--process-exit-timeout", type=float, default=20.0)
    operator_preview.add_argument("--poll-interval", type=float, default=1.0)
    operator_preview.add_argument("--http-timeout", type=float, default=30.0)
    operator_preview.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    operator_preview.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    operator_preview.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    operator_preview.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    operator_preview.add_argument("--lease-seconds", type=float, default=15.0)
    operator_preview.add_argument("--compute-seconds", type=float, default=0.2)
    operator_preview.add_argument("--victim-compute-seconds", type=float, default=45.0)
    operator_preview.add_argument("--heartbeat-interval", type=float, default=0.1)
    operator_preview.add_argument("--idle-sleep", type=float, default=0.2)
    operator_preview.add_argument("--claim-observe-timeout", type=float, default=180.0)
    operator_preview.add_argument("--requeue-timeout", type=float, default=120.0)
    operator_preview.add_argument("--max-request-attempts", type=int, default=240)
    operator_preview.add_argument("--json", action="store_true")

    swarm_trial = subparsers.add_parser(
        "swarm-trial",
        help="Build the Public Swarm v0.2 ordinary-user usable inference trial.",
    )
    swarm_trial.add_argument("swarm_trial_mode", choices=["local-loopback", "package", "live-kaggle", "evidence-import"])
    swarm_trial.add_argument("--output-dir", default="dist/public-swarm-trial")
    swarm_trial.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    swarm_trial.add_argument("--public-host", default="24.199.118.54")
    swarm_trial.add_argument("--bind-host", default="0.0.0.0")
    swarm_trial.add_argument("--port", type=int, default=9400)
    swarm_trial.add_argument("--base-port", type=int, default=9401)
    swarm_trial.add_argument("--release-base-port", type=int, default=9410)
    swarm_trial.add_argument("--ready-url", default="")
    swarm_trial.add_argument("--coordinator-url", default="")
    swarm_trial.add_argument("--target", choices=["local", "kaggle"], default="kaggle")
    swarm_trial.add_argument("--miner-id-prefix", default="public-swarm-trial")
    swarm_trial.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    swarm_trial.add_argument("--hf-cache-dir", default="")
    swarm_trial.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    swarm_trial.add_argument("--product-beta-report", default="dist/public-swarm-product-beta/public_swarm_product_beta.json")
    swarm_trial.add_argument("--operator-preview-report", default="dist/public-swarm-operator-preview/public_swarm_operator_preview.json")
    swarm_trial.add_argument("--developer-preview-report", default="dist/public-swarm-developer-preview/public_swarm_developer_preview.json")
    swarm_trial.add_argument("--alpha-rc-report", default="dist/public-swarm-inference-alpha-rc/public_swarm_inference_alpha_rc.json")
    swarm_trial.add_argument(
        "--live-stage0-report",
        default=(
            "dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/"
            "public_swarm_live_preview_rc.json"
        ),
    )
    swarm_trial.add_argument(
        "--live-stage1-report",
        default=(
            "dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/"
            "public_swarm_live_preview_rc.json"
        ),
    )
    swarm_trial.add_argument("--release-readiness-report", default="dist/release-readiness/release_readiness.json")
    swarm_trial.add_argument("--prompt-text", default="CrowdTensor swarm trial")
    swarm_trial.add_argument("--scenario-id", default="route-baseline")
    swarm_trial.add_argument("--request-count", type=int, default=1)
    swarm_trial.add_argument("--max-new-tokens", type=int, default=2)
    swarm_trial.add_argument("--cpu-request-count", type=int, default=1)
    swarm_trial.add_argument("--external-llm-request-count", type=int, default=1)
    swarm_trial.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="kill-stage0-after-claim")
    swarm_trial.add_argument("--kaggle-owner", default="")
    swarm_trial.add_argument("--dataset-slug", default="")
    swarm_trial.add_argument("--dataset-title", default="CrowdTensor Public Swarm v0.2 Trial")
    swarm_trial.add_argument("--kernel-slug-prefix", default="ct-swarm-trial")
    swarm_trial.add_argument("--kernel-title-prefix", default="CrowdTensor Swarm Trial")
    swarm_trial.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    swarm_trial.add_argument("--skip-kaggle-cleanup", action="store_true")
    swarm_trial.add_argument("--keep-live-private-artifacts", action="store_true")
    swarm_trial.add_argument("--keep-child-artifacts", action="store_true")
    swarm_trial.add_argument("--allow-dirty-release", action="store_true")
    swarm_trial.add_argument("--timeout-seconds", type=float, default=300.0)
    swarm_trial.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    swarm_trial.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    swarm_trial.add_argument("--release-timeout-seconds", type=float, default=240.0)
    swarm_trial.add_argument("--startup-timeout", type=float, default=60.0)
    swarm_trial.add_argument("--process-exit-timeout", type=float, default=20.0)
    swarm_trial.add_argument("--poll-interval", type=float, default=1.0)
    swarm_trial.add_argument("--http-timeout", type=float, default=30.0)
    swarm_trial.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    swarm_trial.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    swarm_trial.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    swarm_trial.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    swarm_trial.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_trial.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_trial.add_argument("--victim-compute-seconds", type=float, default=45.0)
    swarm_trial.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_trial.add_argument("--idle-sleep", type=float, default=0.2)
    swarm_trial.add_argument("--claim-observe-timeout", type=float, default=180.0)
    swarm_trial.add_argument("--requeue-timeout", type=float, default=120.0)
    swarm_trial.add_argument("--max-request-attempts", type=int, default=240)
    swarm_trial.add_argument("--json", action="store_true")

    preview_v04 = subparsers.add_parser(
        "preview-v04",
        help="Build the Public Swarm Inference Preview v0.4 artifact.",
    )
    preview_v04.add_argument("preview_v04_mode", choices=["local-smoke", "package", "evidence-import"])
    preview_v04.add_argument("--output-dir", default="dist/public-swarm-preview-v04")
    preview_v04.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    preview_v04.add_argument("--public-host", default="24.199.118.54")
    preview_v04.add_argument("--bind-host", default="0.0.0.0")
    preview_v04.add_argument("--port", type=int, default=9440)
    preview_v04.add_argument("--base-port", type=int, default=9441)
    preview_v04.add_argument("--target", choices=["local", "kaggle"], default="kaggle")
    preview_v04.add_argument("--miner-id-prefix", default="public-swarm-preview-v04")
    preview_v04.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    preview_v04.add_argument("--optional-model-id", choices=["distilgpt2", "gpt2"], default="distilgpt2")
    preview_v04.add_argument("--run-optional-model", action="store_true")
    preview_v04.add_argument("--require-optional-model-ready", action="store_true")
    preview_v04.add_argument("--require-hf-runtime", action="store_true")
    preview_v04.add_argument("--hf-cache-dir", default="")
    preview_v04.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    preview_v04.add_argument(
        "--live-stage0-report",
        default=(
            "dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/"
            "public_swarm_live_preview_rc.json"
        ),
    )
    preview_v04.add_argument(
        "--live-stage1-report",
        default=(
            "dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/"
            "public_swarm_live_preview_rc.json"
        ),
    )
    preview_v04.add_argument("--product-mvp-report", default="dist/product-swarm-mvp/product_swarm_mvp_check.json")
    preview_v04.add_argument("--optional-model-report", default="")
    preview_v04.add_argument("--product-beta-report", default="dist/public-swarm-product-beta/public_swarm_product_beta.json")
    preview_v04.add_argument("--prompt-text", default="CrowdTensor preview v0.4")
    preview_v04.add_argument("--request-count", type=int, default=1)
    preview_v04.add_argument("--max-new-tokens", type=int, default=2)
    preview_v04.add_argument("--kaggle-owner", default="")
    preview_v04.add_argument("--timeout-seconds", type=float, default=300.0)
    preview_v04.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    preview_v04.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    preview_v04.add_argument("--startup-timeout", type=float, default=60.0)
    preview_v04.add_argument("--session-queue-timeout", type=float, default=45.0)
    preview_v04.add_argument("--miner-timeout", type=float, default=240.0)
    preview_v04.add_argument("--generate-timeout", type=float, default=240.0)
    preview_v04.add_argument("--json", action="store_true")

    p2p_v06 = subparsers.add_parser(
        "p2p-swarm-v06",
        help="Build the P2P Swarm Inference v0.6 prototype evidence artifact.",
    )
    p2p_v06.add_argument("p2p_swarm_v06_mode", choices=["local-smoke", "package", "evidence-import", "external-existing", "kaggle-auto"])
    p2p_v06.add_argument("--output-dir", default="dist/p2p-swarm-inference-v06")
    p2p_v06.add_argument("--swarm-id", default="p2p-v06")
    p2p_v06.add_argument("--public-host", default="24.199.118.54")
    p2p_v06.add_argument("--p2p-port", type=int, default=9560)
    p2p_v06.add_argument("--coordinator-port", type=int, default=9561)
    p2p_v06.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    p2p_v06.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    p2p_v06.add_argument("--hf-cache-dir", default="")
    p2p_v06.add_argument("--prompt-text", default="CrowdTensor P2P v0.6")
    p2p_v06.add_argument("--prompt-texts", default="")
    p2p_v06.add_argument("--stream-generation", action="store_true")
    p2p_v06.add_argument("--max-new-tokens", type=int, default=2)
    p2p_v06.add_argument("--startup-timeout", type=float, default=30.0)
    p2p_v06.add_argument("--timeout-seconds", type=float, default=90.0)
    p2p_v06.add_argument("--http-timeout", type=float, default=5.0)
    p2p_v06.add_argument("--preview-v04-report", default="dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json")
    p2p_v06.add_argument("--product-mvp-report", default="dist/public-swarm-preview-v04-distilgpt2-strict/product-mvp/product_swarm_mvp_check.json")
    p2p_v06.add_argument("--optional-model-report", default="dist/public-swarm-preview-v04-distilgpt2-strict/optional-model-mvp/product_swarm_mvp_check.json")
    p2p_v06.add_argument("--p2p-discovery-report", default="")
    p2p_v06.add_argument("--peer-bootstrap", default="")
    p2p_v06.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    p2p_v06.add_argument("--require-signed", action="store_true")
    p2p_v06.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    p2p_v06.add_argument("--verify-generate", action="store_true")
    p2p_v06.add_argument("--kaggle-owner", default=os.environ.get("KAGGLE_USERNAME", ""))
    p2p_v06.add_argument("--kernel-slug-prefix", default="")
    p2p_v06.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    p2p_v06.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    p2p_v06.add_argument("--kaggle-stage-timeout-seconds", type=float, default=600.0)
    p2p_v06.add_argument("--skip-kaggle-cleanup", action="store_true")
    p2p_v06.add_argument("--json", action="store_true")

    public_p2p_v1_rc = subparsers.add_parser(
        "public-p2p-v1-rc",
        help="Build the signed Public P2P Swarm Inference v1.0 RC artifact.",
    )
    public_p2p_v1_rc.add_argument("public_p2p_v1_rc_mode", choices=["local-smoke", "package", "evidence-import", "kaggle-auto"])
    public_p2p_v1_rc.add_argument("--output-dir", default="dist/public-p2p-swarm-inference-v1-rc")
    public_p2p_v1_rc.add_argument("--swarm-id", default="public-p2p-v1-rc")
    public_p2p_v1_rc.add_argument("--public-host", default="24.199.118.54")
    public_p2p_v1_rc.add_argument("--p2p-port", type=int, default=9660)
    public_p2p_v1_rc.add_argument("--coordinator-port", type=int, default=9661)
    public_p2p_v1_rc.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    public_p2p_v1_rc.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_p2p_v1_rc.add_argument("--hf-cache-dir", default="")
    public_p2p_v1_rc.add_argument("--prompt-text", default="CrowdTensor public P2P v1 RC")
    public_p2p_v1_rc.add_argument("--max-new-tokens", type=int, default=2)
    public_p2p_v1_rc.add_argument("--startup-timeout", type=float, default=30.0)
    public_p2p_v1_rc.add_argument("--timeout-seconds", type=float, default=120.0)
    public_p2p_v1_rc.add_argument("--http-timeout", type=float, default=10.0)
    public_p2p_v1_rc.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    public_p2p_v1_rc.add_argument("--signed-local-report", default="")
    public_p2p_v1_rc.add_argument("--v06-local-report", default="dist/p2p-swarm-inference-v06-local-smoke-refresh2/p2p_swarm_inference_v06.json")
    public_p2p_v1_rc.add_argument("--v06-external-report", default="dist/p2p-swarm-inference-v06-kaggle-auto-final/p2p_swarm_inference_v06.json")
    public_p2p_v1_rc.add_argument("--v06-kaggle-report", default="dist/p2p-swarm-inference-v06-kaggle-auto-final/kaggle-auto/p2p_v06_kaggle_auto.json")
    public_p2p_v1_rc.add_argument("--kaggle-owner", default=os.environ.get("KAGGLE_USERNAME", ""))
    public_p2p_v1_rc.add_argument("--kernel-slug-prefix", default="")
    public_p2p_v1_rc.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    public_p2p_v1_rc.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    public_p2p_v1_rc.add_argument("--kaggle-stage-timeout-seconds", type=float, default=600.0)
    public_p2p_v1_rc.add_argument("--skip-kaggle-cleanup", action="store_true")
    public_p2p_v1_rc.add_argument("--json", action="store_true")

    real_p2p_rc = subparsers.add_parser(
        "real-p2p-rc",
        help="Build the Real P2P provider-core Swarm Inference RC artifact.",
    )
    real_p2p_rc.add_argument("real_p2p_rc_mode", choices=["local-smoke", "package", "external-existing", "evidence-import", "kaggle-auto", "kaggle-connectivity", "kaggle-runtime-smoke"])
    real_p2p_rc.add_argument("--output-dir", default="dist/real-p2p-swarm-inference-core-rc")
    real_p2p_rc.add_argument("--swarm-id", default="real-p2p-core-rc")
    real_p2p_rc.add_argument("--public-host", default="24.199.118.54")
    real_p2p_rc.add_argument("--p2p-port", type=int, default=9760)
    real_p2p_rc.add_argument("--coordinator-port", type=int, default=9761)
    real_p2p_rc.add_argument("--libp2p-port", type=int, default=0)
    real_p2p_rc.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    real_p2p_rc.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_p2p_rc.add_argument("--hf-cache-dir", default="")
    real_p2p_rc.add_argument("--prompt-text", default="CrowdTensor real P2P core RC")
    real_p2p_rc.add_argument("--prompt-texts", default="")
    real_p2p_rc.add_argument("--stream-generation", action="store_true")
    real_p2p_rc.add_argument("--max-new-tokens", type=int, default=2)
    real_p2p_rc.add_argument("--startup-timeout", type=float, default=45.0)
    real_p2p_rc.add_argument("--timeout-seconds", type=float, default=120.0)
    real_p2p_rc.add_argument("--session-queue-timeout", type=float, default=45.0)
    real_p2p_rc.add_argument("--miner-timeout", type=float, default=180.0)
    real_p2p_rc.add_argument("--generate-timeout", type=float, default=180.0)
    real_p2p_rc.add_argument("--http-timeout", type=float, default=10.0)
    real_p2p_rc.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    real_p2p_rc.add_argument("--peer-bootstrap", default="")
    real_p2p_rc.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    real_p2p_rc.add_argument("--verify-generate", action="store_true")
    real_p2p_rc.add_argument("--discovery-backend", choices=sorted(DISCOVERY_BACKENDS), default="http-provider-store")
    real_p2p_rc.add_argument("--real-p2p-report", default="")
    real_p2p_rc.add_argument("--kaggle-owner", default=os.environ.get("KAGGLE_USERNAME", ""))
    real_p2p_rc.add_argument("--kernel-slug-prefix", default="")
    real_p2p_rc.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    real_p2p_rc.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    real_p2p_rc.add_argument("--kaggle-stage-timeout-seconds", type=float, default=600.0)
    real_p2p_rc.add_argument("--kaggle-status-poll-seconds", type=float, default=15.0)
    real_p2p_rc.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="none")
    real_p2p_rc.add_argument("--lease-seconds", type=float, default=15.0)
    real_p2p_rc.add_argument("--compute-seconds", type=float, default=0.2)
    real_p2p_rc.add_argument("--victim-compute-seconds", type=float, default=45.0)
    real_p2p_rc.add_argument("--claim-observe-timeout", type=float, default=180.0)
    real_p2p_rc.add_argument("--requeue-timeout", type=float, default=120.0)
    real_p2p_rc.add_argument("--max-request-attempts", type=int, default=240)
    real_p2p_rc.add_argument("--skip-kaggle-cleanup", action="store_true")
    real_p2p_rc.add_argument("--json", action="store_true")

    petals_candidate = subparsers.add_parser(
        "petals-candidate",
        help="Build the Petals-class real-P2P inference candidate artifact.",
    )
    petals_candidate.add_argument("petals_candidate_mode", choices=["local-smoke", "package", "evidence-import"])
    petals_candidate.add_argument("--output-dir", default="dist/petals-class-p2p-candidate")
    petals_candidate.add_argument("--local-report", default="dist/real-p2p-libp2p-local-smoke-ready/real_p2p_swarm_inference_core_rc.json")
    petals_candidate.add_argument("--runtime-smoke-report", default="dist/real-p2p-libp2p-kaggle-runtime-smoke-20260531-r6/real_p2p_swarm_inference_core_rc.json")
    petals_candidate.add_argument("--external-report", default="dist/real-p2p-libp2p-kaggle-auto-20260531-r4/real_p2p_swarm_inference_core_rc.json")
    petals_candidate.add_argument("--requeue-report", default="")
    petals_candidate.add_argument("--peer-scoring-report", default="")
    petals_candidate.add_argument("--allow-retained-alpha-without-requeue", action="store_true")
    petals_candidate.add_argument("--public-host", default="24.199.118.54")
    petals_candidate.add_argument("--p2p-port", type=int, default=9860)
    petals_candidate.add_argument("--coordinator-port", type=int, default=9861)
    petals_candidate.add_argument("--libp2p-port", type=int, default=10860)
    petals_candidate.add_argument("--max-new-tokens", type=int, default=8)
    petals_candidate.add_argument("--timeout-seconds", type=float, default=300.0)
    petals_candidate.add_argument("--startup-timeout", type=float, default=45.0)
    petals_candidate.add_argument("--session-queue-timeout", type=float, default=45.0)
    petals_candidate.add_argument("--miner-timeout", type=float, default=240.0)
    petals_candidate.add_argument("--generate-timeout", type=float, default=240.0)
    petals_candidate.add_argument("--http-timeout", type=float, default=10.0)
    petals_candidate.add_argument("--json", action="store_true")

    public_swarm_gpu_beta = subparsers.add_parser(
        "public-swarm-gpu-beta",
        help="Prepare, smoke-check, or validate optional CUDA Public Swarm Inference Beta.",
    )
    public_gpu_subparsers = public_swarm_gpu_beta.add_subparsers(dest="public_swarm_gpu_beta_action", required=True)

    def add_public_gpu_base(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", default="dist/public-swarm-gpu-inference-beta")
        target.add_argument("--request-count", type=int, default=1)
        target.add_argument("--timeout-seconds", type=float, default=300.0)
        target.add_argument("--json", action="store_true")

    def add_public_gpu_runtime(target: argparse.ArgumentParser) -> None:
        target.add_argument("--coordinator-url", default="http://127.0.0.1:9300")
        target.add_argument("--public-host", default="24.199.118.54")
        target.add_argument("--port", type=int, default=9320)
        target.add_argument("--base-port", type=int, default=9321)
        target.add_argument("--bind-host", default="0.0.0.0")
        target.add_argument("--miner-id-prefix", default="public-swarm-gpu-beta")
        target.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
        target.add_argument("--hf-cache-dir", default="")
        target.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="stage-local")
        target.add_argument("--remote-timeout-seconds", type=float, default=240.0)
        target.add_argument("--http-timeout", type=float, default=30.0)

    def add_public_gpu_tokens(target: argparse.ArgumentParser) -> None:
        target.add_argument("--observer-token", default="")
        target.add_argument("--admin-token", default="")

    public_gpu_smoke = public_gpu_subparsers.add_parser("local-smoke", help="Run CI-safe CUDA availability diagnostics without claiming GPU readiness.")
    add_public_gpu_base(public_gpu_smoke)
    add_public_gpu_runtime(public_gpu_smoke)

    public_gpu_loopback = public_gpu_subparsers.add_parser("local-loopback", help="Run a local CUDA tiny GPT split proof when CUDA is available.")
    add_public_gpu_base(public_gpu_loopback)
    add_public_gpu_runtime(public_gpu_loopback)
    public_gpu_loopback.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")

    public_gpu_package = public_gpu_subparsers.add_parser("kaggle-package", help="Generate a private Kaggle GPU runbook/template package.")
    add_public_gpu_base(public_gpu_package)
    add_public_gpu_runtime(public_gpu_package)

    public_gpu_auto = public_gpu_subparsers.add_parser("kaggle-auto", help="Run the side-effectful private Kaggle GPU two-stage proof.")
    add_public_gpu_base(public_gpu_auto)
    add_public_gpu_runtime(public_gpu_auto)
    public_gpu_auto.add_argument("--kaggle-owner", default="")
    public_gpu_auto.add_argument("--dataset-slug", default="")
    public_gpu_auto.add_argument("--dataset-title", default="CrowdTensor Public Swarm GPU Beta Package")
    public_gpu_auto.add_argument("--kernel-slug-prefix", default="")
    public_gpu_auto.add_argument("--kernel-title-prefix", default="CrowdTensor Public Swarm GPU Beta Miner")
    public_gpu_auto.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    public_gpu_auto.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    public_gpu_auto.add_argument("--skip-kaggle-cleanup", action="store_true")
    public_gpu_auto.add_argument("--startup-timeout", type=float, default=45.0)
    public_gpu_auto.add_argument("--process-exit-timeout", type=float, default=20.0)
    public_gpu_auto.add_argument("--poll-interval", type=float, default=1.0)
    public_gpu_auto.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    public_gpu_auto.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    public_gpu_auto.add_argument("--kaggle-status-timeout-seconds", type=float, default=900.0)
    public_gpu_auto.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    public_gpu_auto.add_argument("--lease-seconds", type=float, default=15.0)
    public_gpu_auto.add_argument("--compute-seconds", type=float, default=0.2)
    public_gpu_auto.add_argument("--victim-compute-seconds", type=float, default=45.0)
    public_gpu_auto.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_gpu_auto.add_argument("--idle-sleep", type=float, default=0.5)
    public_gpu_auto.add_argument("--max-request-attempts", type=int, default=120)

    public_gpu_import = public_gpu_subparsers.add_parser("evidence-import", help="Import retained Public Swarm GPU Beta evidence.")
    add_public_gpu_base(public_gpu_import)
    public_gpu_import.add_argument("--gpu-report", required=True)

    public_gpu_prepare = public_gpu_subparsers.add_parser("prepare", help="Create operator workflow artifacts using the existing Public Beta shape.")
    add_public_gpu_base(public_gpu_prepare)
    add_public_gpu_runtime(public_gpu_prepare)
    add_public_gpu_tokens(public_gpu_prepare)
    public_gpu_prepare.add_argument("--lease-seconds", type=float, default=15.0)
    public_gpu_prepare.add_argument("--compute-seconds", type=float, default=0.2)
    public_gpu_prepare.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_gpu_prepare.add_argument("--max-request-attempts", type=int, default=120)
    public_gpu_prepare.add_argument("--replace", action="store_true")

    public_gpu_coordinator = public_gpu_subparsers.add_parser("coordinator", help="Print or run the generated Coordinator command.")
    add_public_gpu_base(public_gpu_coordinator)
    add_public_gpu_runtime(public_gpu_coordinator)
    add_public_gpu_tokens(public_gpu_coordinator)
    public_gpu_coordinator.add_argument("--lease-seconds", type=float, default=15.0)
    public_gpu_coordinator.add_argument("--run", action="store_true")

    public_gpu_miner = public_gpu_subparsers.add_parser("miner", help="Print or run a generated stage Miner command.")
    add_public_gpu_base(public_gpu_miner)
    add_public_gpu_runtime(public_gpu_miner)
    public_gpu_miner.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    public_gpu_miner.add_argument("--compute-seconds", type=float, default=0.2)
    public_gpu_miner.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_gpu_miner.add_argument("--max-request-attempts", type=int, default=120)
    public_gpu_miner.add_argument("--run", action="store_true")

    public_gpu_verify = public_gpu_subparsers.add_parser("verify", help="Verify a running operator workflow.")
    add_public_gpu_base(public_gpu_verify)
    add_public_gpu_runtime(public_gpu_verify)
    add_public_gpu_tokens(public_gpu_verify)
    public_gpu_verify.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")

    public_gpu_collect = public_gpu_subparsers.add_parser("collect", help="Collect redacted Public GPU Beta evidence.")
    add_public_gpu_base(public_gpu_collect)
    add_public_gpu_runtime(public_gpu_collect)
    add_public_gpu_tokens(public_gpu_collect)
    public_gpu_collect.add_argument("--miner-id", default="")
    public_gpu_collect.add_argument("--artifact-timeout", type=float, default=60.0)

    public_gpu_clean = public_gpu_subparsers.add_parser("clean", help="Dry-run or delete known generated files.")
    add_public_gpu_base(public_gpu_clean)
    public_gpu_clean.add_argument("--apply", action="store_true")
    public_gpu_clean.add_argument("--include-private", action="store_true")
    public_gpu_clean.add_argument("--remove-empty-dir", action="store_true")

    gpu_generate = subparsers.add_parser(
        "gpu-generate",
        help="Run or import the optional CUDA multi-machine sharded generation Beta.",
    )
    gpu_generate_subparsers = gpu_generate.add_subparsers(dest="gpu_generate_mode", required=True)

    def add_gpu_generate_base(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", default="dist/gpu-sharded-generation-beta")
        target.add_argument("--request-count", type=int, default=1)
        target.add_argument("--max-new-tokens", type=int, default=16)
        target.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
        target.add_argument("--hf-cache-dir", default="")
        target.add_argument("--real-llm-partition-mode", choices=["stage-local", "stage_local"], default="stage-local")
        target.add_argument("--timeout-seconds", type=float, default=900.0)
        target.add_argument("--remote-timeout-seconds", type=float, default=900.0)
        target.add_argument("--json", action="store_true")

    def add_gpu_generate_runtime(target: argparse.ArgumentParser) -> None:
        target.add_argument("--public-host", default="24.199.118.54")
        target.add_argument("--port", type=int, default=9340)
        target.add_argument("--base-port", type=int, default=9341)
        target.add_argument("--bind-host", default="0.0.0.0")
        target.add_argument("--miner-id-prefix", default="gpu-sharded-generation-beta")
        target.add_argument("--http-timeout", type=float, default=30.0)

    gpu_generate_loopback = gpu_generate_subparsers.add_parser("local-loopback", help="Run a local CUDA split multi-token generation proof.")
    add_gpu_generate_base(gpu_generate_loopback)
    add_gpu_generate_runtime(gpu_generate_loopback)

    gpu_generate_auto = gpu_generate_subparsers.add_parser("kaggle-auto", help="Run the side-effectful private Kaggle GPU multi-token proof.")
    add_gpu_generate_base(gpu_generate_auto)
    add_gpu_generate_runtime(gpu_generate_auto)
    gpu_generate_auto.add_argument("--kaggle-owner", default="")
    gpu_generate_auto.add_argument("--dataset-slug", default="")
    gpu_generate_auto.add_argument("--dataset-title", default="CrowdTensor GPU Sharded Generation Beta Package")
    gpu_generate_auto.add_argument("--kernel-slug-prefix", default="")
    gpu_generate_auto.add_argument("--kernel-title-prefix", default="CrowdTensor GPU Sharded Generation Beta Miner")
    gpu_generate_auto.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    gpu_generate_auto.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    gpu_generate_auto.add_argument("--skip-kaggle-cleanup", action="store_true")
    gpu_generate_auto.add_argument("--startup-timeout", type=float, default=45.0)
    gpu_generate_auto.add_argument("--process-exit-timeout", type=float, default=20.0)
    gpu_generate_auto.add_argument("--poll-interval", type=float, default=1.0)
    gpu_generate_auto.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    gpu_generate_auto.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    gpu_generate_auto.add_argument("--kaggle-status-timeout-seconds", type=float, default=1200.0)
    gpu_generate_auto.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    gpu_generate_auto.add_argument("--lease-seconds", type=float, default=15.0)
    gpu_generate_auto.add_argument("--compute-seconds", type=float, default=0.2)
    gpu_generate_auto.add_argument("--victim-compute-seconds", type=float, default=45.0)
    gpu_generate_auto.add_argument("--heartbeat-interval", type=float, default=0.1)
    gpu_generate_auto.add_argument("--idle-sleep", type=float, default=0.5)
    gpu_generate_auto.add_argument("--max-request-attempts", type=int, default=240)

    gpu_generate_import = gpu_generate_subparsers.add_parser("evidence-import", help="Import retained GPU sharded generation evidence.")
    add_gpu_generate_base(gpu_generate_import)
    add_gpu_generate_runtime(gpu_generate_import)
    gpu_generate_import.add_argument("--gpu-report", required=True)

    release_ready = subparsers.add_parser(
        "release-ready",
        help="Build the Alpha maintainer release readiness report.",
    )
    release_ready.add_argument("--output-dir", default="dist/release-readiness")
    release_ready.add_argument("--host", default="127.0.0.1")
    release_ready.add_argument("--base-port", type=int, default=8924)
    release_ready.add_argument("--request-count", type=int, default=4)
    release_ready.add_argument("--external-llm-request-count", type=int, default=3)
    release_ready.add_argument("--timeout-seconds", type=int, default=180)
    release_ready.add_argument("--allow-dirty", action="store_true")
    release_ready.add_argument("--skip-external-llm-evidence", action="store_true")
    release_ready.add_argument("--runtime-report", default="")
    release_ready.add_argument("--browser-report", default="")
    release_ready.add_argument("--remote-report", default="")
    release_ready.add_argument("--json", action="store_true")
    runbook = subparsers.add_parser(
        "remote-runbook",
        help="Build a safe controlled two-machine remote demo runbook through the user CLI.",
    )
    runbook.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    runbook.add_argument("--miner-id", default="remote-linux-1")
    runbook.add_argument("--output-dir", default="dist/remote-demo")
    runbook.add_argument("--request-count", type=int, default=4)
    runbook.add_argument("--scenario-id", default="route-baseline")
    runbook.add_argument("--timeout-seconds", type=int, default=180)
    runbook.add_argument("--replace", action="store_true", help="replace an existing Miner entry in the generated registry")
    runbook.add_argument("--json", action="store_true")
    remote = subparsers.add_parser(
        "remote-acceptance",
        help="Validate a running controlled two-machine remote demo through the user CLI.",
    )
    remote.add_argument("--coordinator-url", required=True)
    remote.add_argument("--miner-id", required=True)
    remote.add_argument("--observer-token", required=True)
    remote.add_argument("--admin-token", required=True)
    remote.add_argument("--output-dir", default="dist/remote-demo-acceptance")
    remote.add_argument("--request-count", type=int, default=4)
    remote.add_argument("--scenario-id", default="route-baseline")
    remote.add_argument("--timeout-seconds", type=int, default=180)
    remote.add_argument("--remote-timeout-seconds", type=float, default=120.0)
    remote.add_argument("--poll-interval", type=float, default=2.0)
    remote.add_argument("--create-session", dest="create_session", action="store_true", default=True)
    remote.add_argument("--no-create-session", dest="create_session", action="store_false")
    remote.add_argument("--json", action="store_true")
    remote_demo = subparsers.add_parser(
        "remote-demo",
        help="Prepare or verify the high-level two-machine home-compute remote Miner demo.",
    )
    remote_demo_subparsers = remote_demo.add_subparsers(dest="remote_demo_action", required=True)
    remote_demo_prepare = remote_demo_subparsers.add_parser(
        "prepare",
        help="Create the recommended remote home-compute runbook and private env files.",
    )
    remote_demo_prepare.add_argument("--workload", choices=REMOTE_DEMO_WORKLOADS, default="model-bundle")
    remote_demo_prepare.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    remote_demo_prepare.add_argument("--target", choices=["generic", "kaggle"], default="generic")
    remote_demo_prepare.add_argument("--miner-id", default="remote-linux-1")
    remote_demo_prepare.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_prepare.add_argument("--request-count", type=int, default=4)
    remote_demo_prepare.add_argument("--scenario-id", default="route-baseline")
    remote_demo_prepare.add_argument("--decode-steps", type=int, default=4)
    remote_demo_prepare.add_argument("--stage-role", choices=["stage0", "stage1", "both"], default="both")
    remote_demo_prepare.add_argument("--micro-llm-artifact", default="")
    remote_demo_prepare.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_prepare.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    remote_demo_prepare.add_argument("--hf-cache-dir", default="")
    remote_demo_prepare.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_prepare.add_argument("--replace", action="store_true")
    remote_demo_prepare.add_argument("--mock", action="store_true")
    remote_demo_prepare.add_argument("--llm-runtime-cmd", default="")
    remote_demo_prepare.add_argument("--llm-runtime-url", default="")
    remote_demo_prepare.add_argument("--llm-runtime-api-key", default="")
    remote_demo_prepare.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    remote_demo_prepare.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    remote_demo_prepare.add_argument("--json", action="store_true")
    remote_demo_doctor = remote_demo_subparsers.add_parser(
        "doctor",
        help="Check remote-demo files, token presence, Coordinator reachability, and route readiness.",
    )
    remote_demo_doctor.add_argument("--workload", choices=REMOTE_DEMO_WORKLOADS, default="model-bundle")
    remote_demo_doctor.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    remote_demo_doctor.add_argument("--miner-id", default="remote-linux-1")
    remote_demo_doctor.add_argument("--observer-token", default="")
    remote_demo_doctor.add_argument("--admin-token", default="")
    remote_demo_doctor.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_doctor.add_argument("--request-count", type=int, default=4)
    remote_demo_doctor.add_argument("--scenario-id", default="route-baseline")
    remote_demo_doctor.add_argument("--decode-steps", type=int, default=4)
    remote_demo_doctor.add_argument("--stage-mode", choices=["both", "split"], default="both")
    remote_demo_doctor.add_argument("--require-distinct-stage-miners", action="store_true")
    remote_demo_doctor.add_argument("--micro-llm-artifact", default="")
    remote_demo_doctor.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_doctor.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    remote_demo_doctor.add_argument("--hf-cache-dir", default="")
    remote_demo_doctor.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_doctor.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_doctor.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_doctor.add_argument("--require-result", action="store_true")
    remote_demo_doctor.add_argument("--json", action="store_true")
    remote_demo_verify = remote_demo_subparsers.add_parser(
        "verify",
        help="Create and verify a read-only remote home-compute session.",
    )
    remote_demo_verify.add_argument("--workload", choices=REMOTE_DEMO_WORKLOADS, default="model-bundle")
    remote_demo_verify.add_argument("--coordinator-url", required=True)
    remote_demo_verify.add_argument("--miner-id", required=True)
    remote_demo_verify.add_argument("--observer-token", required=True)
    remote_demo_verify.add_argument("--admin-token", required=True)
    remote_demo_verify.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_verify.add_argument("--request-count", type=int, default=4)
    remote_demo_verify.add_argument("--scenario-id", default="route-baseline")
    remote_demo_verify.add_argument("--decode-steps", type=int, default=4)
    remote_demo_verify.add_argument("--stage-mode", choices=["both", "split"], default="both")
    remote_demo_verify.add_argument("--require-distinct-stage-miners", action="store_true")
    remote_demo_verify.add_argument("--micro-llm-artifact", default="")
    remote_demo_verify.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_verify.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    remote_demo_verify.add_argument("--hf-cache-dir", default="")
    remote_demo_verify.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_verify.add_argument("--remote-timeout-seconds", type=float, default=120.0)
    remote_demo_verify.add_argument("--poll-interval", type=float, default=2.0)
    remote_demo_verify.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_verify.add_argument("--artifact-timeout", type=float, default=60.0)
    remote_demo_verify.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_verify.add_argument("--create-session", dest="create_session", action="store_true", default=True)
    remote_demo_verify.add_argument("--no-create-session", dest="create_session", action="store_false")
    remote_demo_verify.add_argument("--mock", action="store_true")
    remote_demo_verify.add_argument("--llm-runtime-cmd", default="")
    remote_demo_verify.add_argument("--llm-runtime-url", default="")
    remote_demo_verify.add_argument("--llm-runtime-api-key", default="")
    remote_demo_verify.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    remote_demo_verify.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    remote_demo_verify.add_argument("--json", action="store_true")
    remote_demo_collect = remote_demo_subparsers.add_parser(
        "collect",
        help="Collect evidence and Support Bundle from an already running remote-demo.",
    )
    remote_demo_collect.add_argument("--workload", choices=REMOTE_DEMO_WORKLOADS, default="model-bundle")
    remote_demo_collect.add_argument("--coordinator-url", required=True)
    remote_demo_collect.add_argument("--miner-id", required=True)
    remote_demo_collect.add_argument("--observer-token", required=True)
    remote_demo_collect.add_argument("--admin-token", required=True)
    remote_demo_collect.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_collect.add_argument("--request-count", type=int, default=4)
    remote_demo_collect.add_argument("--scenario-id", default="route-baseline")
    remote_demo_collect.add_argument("--decode-steps", type=int, default=4)
    remote_demo_collect.add_argument("--stage-mode", choices=["both", "split"], default="both")
    remote_demo_collect.add_argument("--require-distinct-stage-miners", action="store_true")
    remote_demo_collect.add_argument("--micro-llm-artifact", default="")
    remote_demo_collect.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_collect.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    remote_demo_collect.add_argument("--hf-cache-dir", default="")
    remote_demo_collect.add_argument("--task-id", default="")
    remote_demo_collect.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_collect.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_collect.add_argument("--artifact-timeout", type=float, default=60.0)
    remote_demo_collect.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_collect.add_argument("--mock", action="store_true")
    remote_demo_collect.add_argument("--llm-runtime-cmd", default="")
    remote_demo_collect.add_argument("--llm-runtime-url", default="")
    remote_demo_collect.add_argument("--llm-runtime-api-key", default="")
    remote_demo_collect.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    remote_demo_collect.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    remote_demo_collect.add_argument("--json", action="store_true")
    remote_demo_clean = remote_demo_subparsers.add_parser(
        "clean",
        help="Dry-run or delete known files generated by remote-demo.",
    )
    remote_demo_clean.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_clean.add_argument("--timeout-seconds", type=int, default=60)
    remote_demo_clean.add_argument("--apply", action="store_true")
    remote_demo_clean.add_argument("--include-private", action="store_true")
    remote_demo_clean.add_argument("--remove-empty-dir", action="store_true")
    remote_demo_clean.add_argument("--json", action="store_true")
    remote_demo_kaggle_real = remote_demo_subparsers.add_parser(
        "kaggle-real",
        help="Prepare, verify, or collect the real Kaggle CPU Miner runtime acceptance.",
    )
    remote_demo_kaggle_real.add_argument("--action", dest="kaggle_real_action", choices=["prepare", "verify", "collect"], default="prepare")
    remote_demo_kaggle_real.add_argument("--public-host", default="24.199.118.54")
    remote_demo_kaggle_real.add_argument("--port", type=int, default=9180)
    remote_demo_kaggle_real.add_argument("--coordinator-url", default="")
    remote_demo_kaggle_real.add_argument("--miner-id", default="kaggle-cpu-1")
    remote_demo_kaggle_real.add_argument("--workload", choices=["model-bundle", "micro-llm-sharded"], default="model-bundle")
    remote_demo_kaggle_real.add_argument("--output-dir", default="dist/kaggle-real-runtime")
    remote_demo_kaggle_real.add_argument("--request-count", type=int, default=2)
    remote_demo_kaggle_real.add_argument("--scenario-id", default="route-baseline")
    remote_demo_kaggle_real.add_argument("--decode-steps", type=int, default=4)
    remote_demo_kaggle_real.add_argument("--stage-mode", choices=["both", "split"], default="both")
    remote_demo_kaggle_real.add_argument("--require-distinct-stage-miners", action="store_true")
    remote_demo_kaggle_real.add_argument("--micro-llm-artifact", default="")
    remote_demo_kaggle_real.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_kaggle_real.add_argument("--timeout-seconds", type=int, default=240)
    remote_demo_kaggle_real.add_argument("--bind-host", default="0.0.0.0")
    remote_demo_kaggle_real.add_argument("--backlog", type=int, default=1)
    remote_demo_kaggle_real.add_argument("--lease-seconds", type=float, default=15.0)
    remote_demo_kaggle_real.add_argument("--miner-token", default="")
    remote_demo_kaggle_real.add_argument("--observer-token", default="")
    remote_demo_kaggle_real.add_argument("--admin-token", default="")
    remote_demo_kaggle_real.add_argument("--replace", action="store_true")
    remote_demo_kaggle_real.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    remote_demo_kaggle_real.add_argument("--poll-interval", type=float, default=2.0)
    remote_demo_kaggle_real.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_kaggle_real.add_argument("--artifact-timeout", type=float, default=60.0)
    remote_demo_kaggle_real.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_kaggle_real.add_argument("--require-existing-result", action="store_true")
    remote_demo_kaggle_real.add_argument("--collect-on-failure", action="store_true")
    remote_demo_kaggle_real.add_argument("--task-id", default="")
    remote_demo_kaggle_real.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"local-proof", "infer", "serve", "join", "generate", "p2pd", "p2p-daemon", "home-infer", "llm-infer", "cpu-infer", "shard-infer", "micro-llm-shard-infer", "real-llm-shard-infer", "micro-llm-artifact", "shard-infer-beta", "micro-llm-shard-infer-beta", "real-llm-shard-infer-beta", "micro-llm-live-rc", "real-llm-live-rc", "real-llm-internet-alpha", "real-llm-internet-beta", "swarm-session", "public-swarm-alpha-rc", "public-swarm-beta", "public-swarm-beta-rc", "public-swarm-product-beta", "public-real-llm-swarm-beta", "usable-swarm", "preview", "live-preview", "operator-preview", "swarm-trial", "public-swarm-gpu-beta", "gpu-generate", "real-p2p-rc", "petals-candidate", "release-ready", "remote-runbook", "remote-acceptance"} or (
        args.command == "remote-demo" and hasattr(args, "request_count")
    ):
        if hasattr(args, "request_count") and args.request_count < 1:
            raise SystemExit("--request-count must be at least 1")
        if hasattr(args, "timeout_seconds") and args.timeout_seconds < 1:
            raise SystemExit("--timeout-seconds must be at least 1")
    if args.command == "infer":
        try:
            parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.infer_mode == "local" and args.full_evidence and (args.max_new_tokens < 8 or args.max_new_tokens > 32):
            raise SystemExit("infer --mode local --full-evidence requires --max-new-tokens between 8 and 32")
        if args.infer_mode == "local" and not args.full_evidence and (args.max_new_tokens < 2 or args.max_new_tokens > 8):
            raise SystemExit("infer --mode local requires --max-new-tokens between 2 and 8, or use --full-evidence for 8..32")
        if args.infer_mode == "existing" and (args.max_new_tokens < 2 or args.max_new_tokens > 32):
            raise SystemExit("infer --mode existing requires --max-new-tokens between 2 and 32")
        if args.p2p_port < 1 or args.coordinator_port < 1 or args.real_p2p_port < 1 or args.real_p2p_coordinator_port < 1:
            raise SystemExit("infer ports must be positive")
        if args.real_p2p_libp2p_port < 0:
            raise SystemExit("--real-p2p-libp2p-port must be non-negative")
        for name in ["startup_timeout", "timeout_seconds", "poll_interval", "http_timeout"]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
        if args.infer_mode != "existing" and args.dry_run:
            raise SystemExit("infer --dry-run is supported for --mode existing route preflight only")
    if args.command == "serve":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.ttl_seconds <= 0:
            raise SystemExit("--ttl-seconds must be positive")
        if args.lease_seconds <= 0:
            raise SystemExit("--lease-seconds must be positive")
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
    if args.command == "join":
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.ttl_seconds <= 0:
            raise SystemExit("--ttl-seconds must be positive")
        if args.max_tasks < 0:
            raise SystemExit("--max-tasks must be non-negative")
        if args.max_runtime_seconds < 0:
            raise SystemExit("--max-runtime-seconds must be non-negative")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.retry_base_sleep < 0 or args.retry_max_sleep < 0:
            raise SystemExit("--retry-base-sleep and --retry-max-sleep must be non-negative")
        if args.idle_sleep <= 0:
            raise SystemExit("--idle-sleep must be positive")
    if args.command == "generate":
        if args.prompt_text_arg:
            args.prompt_text = args.prompt_text_arg
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        try:
            parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
    if args.command == "p2pd":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.ttl_seconds <= 0:
            raise SystemExit("--ttl-seconds must be positive")
    if args.command == "p2p-daemon":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.ttl_seconds <= 0:
            raise SystemExit("--ttl-seconds must be positive")
        if args.signature_max_age_seconds <= 0:
            raise SystemExit("--signature-max-age-seconds must be positive")
        if args.require_signed and not args.record_secret:
            raise SystemExit("--require-signed requires --record-secret")
        if args.discovery_backend not in DISCOVERY_BACKENDS:
            raise SystemExit("--discovery-backend must be one of: " + ", ".join(sorted(DISCOVERY_BACKENDS)))
        if args.libp2p_port < 0:
            raise SystemExit("--libp2p-port must be non-negative")
    if args.command == "peer":
        if args.peer_action == "daemon":
            if args.port < 1:
                raise SystemExit("--port must be positive")
            if args.ttl_seconds <= 0:
                raise SystemExit("--ttl-seconds must be positive")
        if args.peer_action in {"resolve", "announce"} and args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.peer_action == "resolve" and (args.max_new_tokens < 1 or args.max_new_tokens > 32):
            raise SystemExit("--max-new-tokens must be between 1 and 32")
    if args.command == "public-swarm-product-rc":
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.timeout_seconds < 1:
            raise SystemExit("--timeout-seconds must be positive")
    if args.command == "release-ready":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.external_llm_request_count < 1:
            raise SystemExit("--external-llm-request-count must be at least 1")
    if args.command == "real-llm-internet-beta":
        if args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        if args.failure_mode != "none" and args.max_new_tokens != 1:
            raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
        if args.port < 1 or args.base_port < 1:
            raise SystemExit("--port and --base-port must be positive")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "kaggle_push_timeout_seconds",
            "kaggle_delete_timeout_seconds",
            "kaggle_status_timeout_seconds",
            "kaggle_status_poll_interval",
            "lease_seconds",
            "victim_compute_seconds",
            "heartbeat_interval",
            "idle_sleep",
            "claim_observe_timeout",
            "requeue_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.failure_mode != "none":
            if args.victim_compute_seconds <= args.lease_seconds:
                args.victim_compute_seconds = args.lease_seconds + 30.0
            if args.requeue_timeout <= args.lease_seconds:
                args.requeue_timeout = args.lease_seconds + 45.0
    if args.command == "swarm-infer-beta":
        if getattr(args, "timeout_seconds", 1) <= 0:
            raise SystemExit("--timeout-seconds must be positive")
        if args.swarm_action != "clean":
            if args.port < 1:
                raise SystemExit("--port must be positive")
            if args.request_count < 1 or args.request_count > 4:
                raise SystemExit("--request-count must be between 1 and 4")
            for name in ["remote_timeout_seconds", "http_timeout"]:
                if getattr(args, name) <= 0:
                    raise SystemExit(f"--{name.replace('_', '-')} must be positive")
            if hasattr(args, "lease_seconds") and args.lease_seconds <= 0:
                raise SystemExit("--lease-seconds must be positive")
            if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
                raise SystemExit("--compute-seconds must be non-negative")
            if hasattr(args, "heartbeat_interval") and args.heartbeat_interval <= 0:
                raise SystemExit("--heartbeat-interval must be positive")
            if hasattr(args, "victim_compute_seconds") and args.victim_compute_seconds <= 0:
                raise SystemExit("--victim-compute-seconds must be positive")
            if hasattr(args, "claim_observe_timeout") and args.claim_observe_timeout <= 0:
                raise SystemExit("--claim-observe-timeout must be positive")
            if hasattr(args, "requeue_timeout") and args.requeue_timeout <= 0:
                raise SystemExit("--requeue-timeout must be positive")
            if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
                raise SystemExit("--max-request-attempts must be at least 1")
            if hasattr(args, "artifact_timeout") and args.artifact_timeout <= 0:
                raise SystemExit("--artifact-timeout must be positive")
            if getattr(args, "swarm_action", "") == "live" and getattr(args, "failure_mode", "none") != "none":
                if args.victim_compute_seconds <= args.lease_seconds:
                    args.victim_compute_seconds = args.lease_seconds + 30.0
                if args.requeue_timeout <= args.lease_seconds:
                    args.requeue_timeout = args.lease_seconds + 45.0
    if args.command == "swarm-session":
        if args.port < 1 or args.base_port < 1:
            raise SystemExit("--port and --base-port must be positive")
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "kaggle_push_timeout_seconds",
            "kaggle_delete_timeout_seconds",
            "kaggle_status_timeout_seconds",
            "kaggle_status_poll_interval",
            "lease_seconds",
            "victim_compute_seconds",
            "heartbeat_interval",
            "idle_sleep",
            "claim_observe_timeout",
            "requeue_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.failure_mode != "none":
            if args.victim_compute_seconds <= args.lease_seconds:
                args.victim_compute_seconds = args.lease_seconds + 30.0
            if args.requeue_timeout <= args.lease_seconds:
                args.requeue_timeout = args.lease_seconds + 45.0
    if args.command == "public-swarm-alpha-rc":
        if args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
    if args.command == "public-swarm-beta":
        if args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if getattr(args, "base_port", 1) < 1:
            raise SystemExit("--base-port must be positive")
        if getattr(args, "port", 1) < 1:
            raise SystemExit("--port must be positive")
        for name in ["remote_timeout_seconds", "http_timeout", "artifact_timeout", "lease_seconds", "heartbeat_interval"]:
            if hasattr(args, name) and getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if hasattr(args, "max_new_tokens") and (args.max_new_tokens < 2 or args.max_new_tokens > 32):
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if hasattr(args, "cpu_request_count") and (args.cpu_request_count < 1 or args.cpu_request_count > 4):
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if hasattr(args, "external_llm_request_count") and (args.external_llm_request_count < 1 or args.external_llm_request_count > 4):
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if hasattr(args, "cpu_timeout_seconds") and args.cpu_timeout_seconds <= 0:
            raise SystemExit("--cpu-timeout-seconds must be positive")
    if args.command == "public-swarm-beta-rc":
        try:
            parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1:
            raise SystemExit("--base-port and --port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.public_swarm_beta_rc_mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "public-swarm-product-beta":
        try:
            parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1:
            raise SystemExit("--base-port and --port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.public_swarm_product_beta_mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "public-real-llm-swarm-beta":
        try:
            parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if (
            args.base_port < 1
            or args.port < 1
            or args.p2p_port < 1
            or args.public_swarm_v2_p2p_port < 1
            or args.public_swarm_v2_coordinator_port < 1
            or args.public_swarm_v2_real_p2p_port < 1
            or args.public_swarm_v2_real_p2p_coordinator_port < 1
            or args.public_swarm_v2_real_p2p_libp2p_port < 0
        ):
            raise SystemExit("public real LLM beta ports must be positive, except --public-swarm-v2-real-p2p-libp2p-port may be 0")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.command == "usable-swarm":
        try:
            parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.p2p_port < 1 or args.coordinator_port < 1:
            raise SystemExit("--p2p-port and --coordinator-port must be positive")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        for name in ["startup_timeout", "timeout_seconds", "http_timeout"]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.command == "public-swarm-v2":
        try:
            parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.p2p_port < 1 or args.coordinator_port < 1:
            raise SystemExit("--p2p-port and --coordinator-port must be positive")
        if args.max_new_tokens < 8 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 8 and 32")
        for name in ["startup_timeout", "timeout_seconds", "http_timeout"]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.command == "preview":
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1:
            raise SystemExit("--base-port and --port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.preview_mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "live-preview":
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1:
            raise SystemExit("--base-port and --port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "kaggle_push_timeout_seconds",
            "kaggle_delete_timeout_seconds",
            "kaggle_status_timeout_seconds",
            "kaggle_status_poll_interval",
            "lease_seconds",
            "victim_compute_seconds",
            "heartbeat_interval",
            "idle_sleep",
            "claim_observe_timeout",
            "requeue_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.live_preview_mode == "live-kaggle" and not args.kaggle_owner and not os.environ.get("KAGGLE_USERNAME"):
            raise SystemExit("--kaggle-owner or KAGGLE_USERNAME is required for live-kaggle")
    if args.command == "operator-preview":
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1 or args.release_base_port < 1:
            raise SystemExit("--base-port, --port, and --release-base-port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "release_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "kaggle_push_timeout_seconds",
            "kaggle_delete_timeout_seconds",
            "kaggle_status_timeout_seconds",
            "kaggle_status_poll_interval",
            "lease_seconds",
            "victim_compute_seconds",
            "heartbeat_interval",
            "idle_sleep",
            "claim_observe_timeout",
            "requeue_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.operator_preview_mode == "live-kaggle" and not args.kaggle_owner and not os.environ.get("KAGGLE_USERNAME"):
            raise SystemExit("--kaggle-owner or KAGGLE_USERNAME is required for live-kaggle")
    if args.command == "swarm-trial":
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1 or args.release_base_port < 1:
            raise SystemExit("--base-port, --port, and --release-base-port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "release_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "kaggle_push_timeout_seconds",
            "kaggle_delete_timeout_seconds",
            "kaggle_status_timeout_seconds",
            "kaggle_status_poll_interval",
            "lease_seconds",
            "victim_compute_seconds",
            "heartbeat_interval",
            "idle_sleep",
            "claim_observe_timeout",
            "requeue_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.swarm_trial_mode == "live-kaggle" and not args.kaggle_owner and not os.environ.get("KAGGLE_USERNAME"):
            raise SystemExit("--kaggle-owner or KAGGLE_USERNAME is required for live-kaggle")
    if args.command == "preview-v04":
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1:
            raise SystemExit("--base-port and --port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "startup_timeout",
            "session_queue_timeout",
            "miner_timeout",
            "generate_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.require_optional_model_ready and not args.run_optional_model and not args.optional_model_report:
            raise SystemExit("--require-optional-model-ready requires --run-optional-model or --optional-model-report")
    if args.command == "p2p-swarm-v06":
        if args.p2p_port < 1 or args.coordinator_port < 1:
            raise SystemExit("--p2p-port and --coordinator-port must be positive")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        try:
            parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.p2p_swarm_v06_mode == "external-existing" and not args.peer_bootstrap:
            raise SystemExit("external-existing requires --peer-bootstrap")
        if args.require_signed and not args.peer_secret:
            raise SystemExit("--require-signed requires --peer-secret")
        if args.p2p_swarm_v06_mode == "kaggle-auto":
            for name in ["kaggle_push_timeout_seconds", "kaggle_delete_timeout_seconds", "kaggle_stage_timeout_seconds"]:
                if getattr(args, name) <= 0:
                    raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        for name in ["startup_timeout", "timeout_seconds", "http_timeout"]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.command == "public-p2p-v1-rc":
        if args.p2p_port < 1 or args.coordinator_port < 1:
            raise SystemExit("--p2p-port and --coordinator-port must be positive")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.public_p2p_v1_rc_mode == "kaggle-auto" and not args.kaggle_owner:
            raise SystemExit("--kaggle-owner or KAGGLE_USERNAME is required for kaggle-auto")
        for name in ["startup_timeout", "timeout_seconds", "http_timeout", "kaggle_push_timeout_seconds", "kaggle_delete_timeout_seconds", "kaggle_stage_timeout_seconds"]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.command == "real-p2p-rc":
        if args.p2p_port < 1 or args.coordinator_port < 1:
            raise SystemExit("--p2p-port and --coordinator-port must be positive")
        if args.libp2p_port < 0:
            raise SystemExit("--libp2p-port must be non-negative")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.real_p2p_rc_mode == "external-existing" and not args.peer_bootstrap:
            raise SystemExit("external-existing requires --peer-bootstrap")
        if args.real_p2p_rc_mode == "evidence-import" and not args.real_p2p_report:
            raise SystemExit("evidence-import requires --real-p2p-report")
        if (args.prompt_texts or args.stream_generation) and args.real_p2p_rc_mode != "external-existing":
            raise SystemExit("--prompt-texts and --stream-generation are currently supported for real-p2p-rc external-existing only")
        if (args.prompt_texts or args.stream_generation) and not args.verify_generate:
            raise SystemExit("--prompt-texts and --stream-generation require --verify-generate")
        if args.real_p2p_rc_mode in {"kaggle-auto", "kaggle-connectivity", "kaggle-runtime-smoke"} and not args.kaggle_owner and not os.environ.get("KAGGLE_USERNAME"):
            raise SystemExit(f"--kaggle-owner or KAGGLE_USERNAME is required for {args.real_p2p_rc_mode}")
        parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
        for name in [
            "startup_timeout",
            "timeout_seconds",
            "session_queue_timeout",
            "miner_timeout",
            "generate_timeout",
            "http_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.real_p2p_rc_mode in {"kaggle-auto", "kaggle-connectivity", "kaggle-runtime-smoke"}:
            for name in ["kaggle_push_timeout_seconds", "kaggle_delete_timeout_seconds", "kaggle_stage_timeout_seconds", "kaggle_status_poll_seconds"]:
                if getattr(args, name) <= 0:
                    raise SystemExit(f"--{name.replace('_', '-')} must be positive")
            for name in ["lease_seconds", "victim_compute_seconds", "claim_observe_timeout", "requeue_timeout"]:
                if getattr(args, name) <= 0:
                    raise SystemExit(f"--{name.replace('_', '-')} must be positive")
            if args.compute_seconds < 0:
                raise SystemExit("--compute-seconds must be non-negative")
            if args.max_request_attempts < 1:
                raise SystemExit("--max-request-attempts must be at least 1")
    if args.command == "petals-candidate":
        if args.p2p_port < 1 or args.coordinator_port < 1 or args.libp2p_port < 1:
            raise SystemExit("--p2p-port, --coordinator-port, and --libp2p-port must be positive")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        for name in [
            "startup_timeout",
            "timeout_seconds",
            "session_queue_timeout",
            "miner_timeout",
            "generate_timeout",
            "http_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.command == "public-swarm-gpu-beta":
        if args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if getattr(args, "base_port", 1) < 1:
            raise SystemExit("--base-port must be positive")
        if getattr(args, "port", 1) < 1:
            raise SystemExit("--port must be positive")
        for name in ["remote_timeout_seconds", "http_timeout", "artifact_timeout", "lease_seconds", "heartbeat_interval"]:
            if hasattr(args, name) and getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
    if args.command == "gpu-generate":
        if args.request_count != 1:
            raise SystemExit("--request-count must be 1 for gpu-generate Beta")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if getattr(args, "base_port", 1) < 1:
            raise SystemExit("--base-port must be positive")
        if getattr(args, "port", 1) < 1:
            raise SystemExit("--port must be positive")
        for name in [
            "remote_timeout_seconds",
            "http_timeout",
            "lease_seconds",
            "heartbeat_interval",
            "idle_sleep",
        ]:
            if hasattr(args, name) and getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
    if args.command == "llm-infer":
        if args.llm_runtime_cmd and args.llm_runtime_url:
            raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
        if args.llm_runtime_timeout <= 0:
            raise SystemExit("--llm-runtime-timeout must be positive")
    if args.command == "cpu-infer":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.external_llm_request_count < 1:
            raise SystemExit("--external-llm-request-count must be at least 1")
        if args.remote_timeout_seconds < 0:
            raise SystemExit("--remote-timeout-seconds must be non-negative")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
        if args.llm_runtime_cmd and args.llm_runtime_url:
            raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
        if args.llm_runtime_timeout <= 0:
            raise SystemExit("--llm-runtime-timeout must be positive")
        if args.mode == "remote-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "shard-infer":
        if args.port < 1:
            raise SystemExit("--port must be positive")
    if args.command == "micro-llm-shard-infer":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
    if args.command == "real-llm-shard-infer":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.request_count > 4:
            raise SystemExit("--request-count must be at most 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        if args.failure_mode != "none" and args.max_new_tokens != 1:
            raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
    if args.command == "micro-llm-artifact":
        if args.version < 1:
            raise SystemExit("--version must be at least 1")
    if args.command == "shard-infer-beta":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.remote_timeout_seconds <= 0:
            raise SystemExit("--remote-timeout-seconds must be positive")
        if args.mode == "remote-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
            if args.failure_mode != "none":
                raise SystemExit("remote-existing does not orchestrate failure-mode kills; use remote-loopback for requeue proof")
    if args.command == "micro-llm-shard-infer-beta":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
        if args.remote_timeout_seconds <= 0:
            raise SystemExit("--remote-timeout-seconds must be positive")
        if args.mode == "remote-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
            if args.failure_mode != "none":
                raise SystemExit("remote-existing does not orchestrate failure-mode kills; use remote-loopback for requeue proof")
    if args.command == "real-llm-shard-infer-beta":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.request_count > 4:
            raise SystemExit("--request-count must be at most 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        if args.failure_mode != "none" and args.max_new_tokens != 1:
            raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
        if args.remote_timeout_seconds <= 0:
            raise SystemExit("--remote-timeout-seconds must be positive")
        if args.mode == "remote-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
            if args.failure_mode != "none":
                raise SystemExit("remote-existing does not orchestrate failure-mode kills; use remote-loopback for requeue proof")
    if args.command == "micro-llm-live-rc":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "artifact_timeout",
            "lease_seconds",
            "heartbeat_interval",
            "idle_sleep",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
    if args.command == "real-llm-live-rc":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.request_count > 4:
            raise SystemExit("--request-count must be at most 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "lease_seconds",
            "heartbeat_interval",
            "idle_sleep",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "real-llm-internet-alpha":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.request_count > 4:
            raise SystemExit("--request-count must be at most 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "lease_seconds",
            "heartbeat_interval",
            "idle_sleep",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "remote-acceptance":
        if args.remote_timeout_seconds < 0:
            raise SystemExit("--remote-timeout-seconds must be non-negative")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
    if args.command == "remote-demo" and args.remote_demo_action in {"doctor", "verify", "collect"}:
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
    if args.command == "remote-demo" and hasattr(args, "decode_steps"):
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
    if args.command == "remote-demo" and getattr(args, "workload", "") == "real-llm-sharded":
        if getattr(args, "prompt_texts", "") == "arn,ten":
            args.prompt_texts = "CrowdTensor routes home CPU,A miner returns one token"
        if args.remote_demo_action in {"doctor", "verify", "collect"} and getattr(args, "stage_mode", "both") == "both":
            args.stage_mode = "split"
        if args.remote_demo_action in {"doctor", "verify", "collect"} and getattr(args, "stage_mode", "") == "split":
            args.require_distinct_stage_miners = True
    if args.command == "remote-demo" and args.remote_demo_action == "kaggle-real":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.backlog < 1:
            raise SystemExit("--backlog must be at least 1")
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
        if args.workload == "micro-llm-sharded" and args.stage_mode == "split":
            args.require_distinct_stage_miners = True
        if args.lease_seconds <= 0:
            raise SystemExit("--lease-seconds must be positive")
        if args.remote_timeout_seconds < 0:
            raise SystemExit("--remote-timeout-seconds must be non-negative")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.artifact_timeout <= 0:
            raise SystemExit("--artifact-timeout must be positive")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
    if args.command == "remote-demo" and args.remote_demo_action == "verify":
        if args.remote_timeout_seconds < 0:
            raise SystemExit("--remote-timeout-seconds must be non-negative")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
        if args.artifact_timeout <= 0:
            raise SystemExit("--artifact-timeout must be positive")
    if args.command == "remote-demo" and args.remote_demo_action == "collect":
        if args.artifact_timeout <= 0:
            raise SystemExit("--artifact-timeout must be positive")
    if args.command == "remote-demo" and args.remote_demo_action in {"prepare", "verify", "collect"}:
        if args.llm_runtime_cmd and args.llm_runtime_url:
            raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
        if args.llm_runtime_timeout <= 0:
            raise SystemExit("--llm-runtime-timeout must be positive")
    if args.command == "clean-artifacts":
        if args.older_than_hours < 0:
            raise SystemExit("--older-than-hours must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "local-proof":
        summary = build_local_proof(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_local_proof(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "clean-artifacts":
        report = build_cleanup_report(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print_cleanup_report(report)
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "infer":
        report = build_infer(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            local_report = dict(report)
            local_report["local_prompt_text"] = str(getattr(args, "prompt_text", "") or "")
            local_report["local_prompt_texts"] = str(getattr(args, "prompt_texts", "") or "")
            print_infer(local_report)
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "serve":
        report = build_product_serve(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print_product_serve(report)
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "join":
        report = build_product_join(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print_product_join(report)
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "generate":
        report = build_product_generate(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            local_report = dict(report)
            local_report["local_prompt_text"] = str(getattr(args, "prompt_text", "") or "")
            local_report["local_prompt_texts"] = str(getattr(args, "prompt_texts", "") or "")
            print_product_generate(local_report)
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "p2pd":
        report = build_p2pd_cli(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            if report.get("command"):
                print("\n".join(report.get("command") or []))
            else:
                print(f"CrowdTensor p2pd ok={report.get('ok')} diagnosis={','.join(report.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "p2p-daemon":
        report = build_p2p_daemon_cli(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            if report.get("command"):
                print("\n".join(report.get("command") or []))
            else:
                print(f"CrowdTensor p2p-daemon ok={report.get('ok')} diagnosis={','.join(report.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "peer":
        report = build_peer_cli(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            if report.get("command"):
                print("\n".join(report.get("command") or []))
            else:
                print(f"CrowdTensor peer ok={report.get('ok')} diagnosis={','.join(report.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "public-swarm-product-rc":
        report = build_public_swarm_product_rc(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(f"CrowdTensor Public Swarm Product RC ok={report.get('ok')} diagnosis={','.join(report.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "home-infer":
        summary = build_home_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_home_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "llm-infer":
        summary = build_llm_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_llm_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "cpu-infer":
        summary = build_cpu_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_cpu_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "shard-infer":
        summary = build_sharded_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_sharded_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "micro-llm-shard-infer":
        summary = build_micro_llm_sharded_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_micro_llm_sharded_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-shard-infer":
        summary = build_real_llm_sharded_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_real_llm_sharded_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "micro-llm-artifact":
        summary = build_micro_llm_artifact(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_micro_llm_artifact(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "shard-infer-beta":
        summary = build_remote_sharded_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_sharded_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "micro-llm-shard-infer-beta":
        summary = build_remote_micro_llm_sharded_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_micro_llm_sharded_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-shard-infer-beta":
        summary = build_remote_real_llm_sharded_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_real_llm_sharded_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "micro-llm-live-rc":
        summary = build_micro_llm_live_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_micro_llm_live_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-live-rc":
        summary = build_real_llm_live_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_real_llm_live_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-internet-alpha":
        summary = build_real_llm_internet_alpha(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_real_llm_internet_alpha(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-internet-beta":
        summary = build_real_llm_internet_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_real_llm_internet_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "swarm-infer-beta":
        summary = build_swarm_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_swarm_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "swarm-session":
        summary = build_public_swarm_inference_alpha(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_alpha(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-alpha-rc":
        summary = build_public_swarm_inference_alpha_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_alpha_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-beta":
        summary = build_public_swarm_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-beta-rc":
        summary = build_public_swarm_inference_beta_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_beta_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-product-beta":
        summary = build_public_swarm_product_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_product_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-real-llm-swarm-beta":
        summary = build_public_real_llm_swarm_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_real_llm_swarm_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "usable-swarm":
        summary = build_usable_swarm_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_usable_swarm_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-v2":
        summary = build_public_swarm_inference_v2(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_v2(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "preview":
        summary = build_public_swarm_developer_preview(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_developer_preview(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "live-preview":
        summary = build_public_swarm_live_preview_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_live_preview_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "operator-preview":
        summary = build_public_swarm_operator_preview(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_operator_preview(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "swarm-trial":
        summary = build_public_swarm_trial(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_trial(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "preview-v04":
        summary = build_public_swarm_preview_v04(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_preview_v04(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "p2p-swarm-v06":
        summary = build_p2p_swarm_inference_v06(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(f"CrowdTensor P2P Swarm Inference v0.6 ok={summary.get('ok')} diagnosis={','.join(summary.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-p2p-v1-rc":
        summary = build_public_p2p_swarm_inference_v1_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(f"CrowdTensor Public P2P Swarm Inference v1.0 RC ok={summary.get('ok')} diagnosis={','.join(summary.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-p2p-rc":
        summary = build_real_p2p_swarm_inference_core_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(f"CrowdTensor Real P2P Swarm Inference Core RC ok={summary.get('ok')} diagnosis={','.join(summary.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "petals-candidate":
        summary = build_petals_class_p2p_candidate(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(f"CrowdTensor Petals-Class P2P Candidate ok={summary.get('ok')} diagnosis={','.join(summary.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-gpu-beta":
        summary = build_public_swarm_gpu_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_gpu_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "gpu-generate":
        summary = build_gpu_sharded_generation_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_gpu_sharded_generation_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "release-ready":
        report = build_release_ready(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print_release_ready(report)
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "remote-runbook":
        summary = build_remote_runbook(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_cli_report(summary, title="CrowdTensor remote runbook")
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "remote-acceptance":
        summary = build_remote_acceptance(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_cli_report(summary, title="CrowdTensor remote acceptance")
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "remote-demo":
        summary = build_remote_demo(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_cli_report(summary, title="CrowdTensor remote home-compute demo")
        raise SystemExit(0 if summary.get("ok") else 1)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
