#!/usr/bin/env python3
"""User-facing Swarm Inference Beta wrapper for real tiny-LLM split inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
from create_miner_invite import create_invite  # noqa: E402
from crowdtensor.auth import hash_token  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID, DEFAULT_PROMPTS  # noqa: E402
from product_swarm_mvp_check import parse_prompt_texts_arg, read_prompt_texts_file  # noqa: E402


SCHEMA = "swarm_inference_beta_v1"
CHECK_READY_CODE = "swarm_inference_beta_ready"
WORKLOAD_TYPE = "real_llm_sharded_infer"
ROUTE_NAME = "remote_python_real_llm_sharded_infer"
DEFAULT_OUTPUT_DIR = "dist/swarm-inference-beta"
DEFAULT_LIVE_OUTPUT_DIR = "dist/swarm-inference-beta-live"
DEFAULT_COORDINATOR_URL = "http://127.0.0.1:9200"
DEFAULT_PORT = 9200
DEFAULT_LIVE_PORT = 9210
STAGES = ("stage0", "stage1")
PRIVATE_FILENAMES = {
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
}
LIVE_PRIVATE_RELATIVE_PATHS = (
    "real-internet-beta/alpha-package/package-live-rc/remote-real-llm-runtime/operator.private.env",
    "real-internet-beta/alpha-package/package-live-rc/remote-real-llm-runtime/miner.stage0.private.env",
    "real-internet-beta/alpha-package/package-live-rc/remote-real-llm-runtime/miner.stage1.private.env",
    "real-internet-beta/alpha-package/package-live-rc/remote-real-llm-runtime/miner_registry.json",
    "real-internet-beta/alpha-package/package-live-rc/kaggle-upload-real-llm-stage0/miner.private.env",
    "real-internet-beta/alpha-package/package-live-rc/kaggle-upload-real-llm-stage1/miner.private.env",
    "real-internet-beta/alpha-package/package-live-rc/kaggle-upload-real-llm-stage0-victim/miner.private.env",
    "real-internet-beta/alpha-package/package-live-rc/kaggle-upload-real-llm-stage0-rescue/miner.private.env",
    "real-internet-beta/alpha-package/package-live-rc/kaggle-upload-real-llm-stage1-victim/miner.private.env",
    "real-internet-beta/alpha-package/package-live-rc/kaggle-upload-real-llm-stage1-rescue/miner.private.env",
)
LIVE_TRANSIENT_RELATIVE_DIRS = (
    "real-internet-beta/alpha-package/package-live-rc",
    "real-internet-beta/kaggle-package",
)
KNOWN_GENERATED = (
    "swarm_inference_beta.json",
    "swarm_inference_beta.md",
    "swarm_inference_beta_prepare.json",
    "swarm_inference_beta_prepare.md",
    "swarm_inference_beta_verify.json",
    "swarm_inference_beta_verify.md",
    "swarm_inference_beta_collect.json",
    "swarm_inference_beta_collect.md",
    "swarm_inference_beta_live.json",
    "swarm_inference_beta_live.md",
    "swarm_inference_beta_clean.json",
    "swarm_inference_beta_clean.md",
    "SWARM_INFERENCE_BETA.md",
    "start_coordinator.sh",
    "verify.sh",
    "collect.sh",
)
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "activation_result",
    "hidden_state",
    "input_ids",
    "logits",
    "real_llm_sharded_result",
    "Bearer ",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quote_env(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def write_private_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"export {key}={quote_env(value)}" for key, value in sorted(values.items()) if value]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str, secret_values: list[str] | None = None) -> str:
    redacted = str(text)
    for value in secret_values or []:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Swarm Inference Beta artifacts summarize two-stage tiny GPT split "
            "readiness, live proof cleanup, requeue evidence, hashes/counts, and "
            "support diagnostics only. They do not include answer text."
        ),
    }


def parse_prompt_texts(value: str) -> list[str]:
    prompts = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return prompts or list(DEFAULT_PROMPTS)


def prompt_list_from_args(args: argparse.Namespace) -> list[str]:
    prompt_list = getattr(args, "prompt_texts_list", None)
    if isinstance(prompt_list, list) and prompt_list:
        return [str(prompt) for prompt in prompt_list]
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    if prompt_texts_file:
        return read_prompt_texts_file(prompt_texts_file)
    return parse_prompt_texts_arg("", str(getattr(args, "prompt_texts", "") or ""))


def prompt_scope_summary(args: argparse.Namespace) -> dict[str, Any]:
    action = getattr(args, "action", "")
    prompts = prompt_list_from_args(args) if action == "verify" else []
    inline_prompt_text = bool(prompts)
    source = "prompt-texts-file" if action == "verify" and str(getattr(args, "prompt_texts_file", "") or "") else "prompt-texts" if inline_prompt_text else "none"
    return {
        "source": source,
        "prompt_count": len(prompts),
        "inline_prompt_text": source == "prompt-texts",
        "terminal_next_commands_local_private": source == "prompt-texts",
        "terminal_logs_local_private": source == "prompt-texts",
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": source in {"prompt-texts", "prompt-texts-file"},
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Swarm Inference Beta artifacts record prompt source/count and placeholder "
            "safety only; raw prompt text is excluded from public JSON, Markdown, "
            "support bundles, and runbooks."
        ),
    }


def prompt_secret_values(args: argparse.Namespace) -> list[str]:
    if getattr(args, "action", "") != "verify":
        return []
    return prompt_list_from_args(args)


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "hash-only",
        "saved_markdown_display": "hash-only",
        "json_stdout_display": "hash-only-json",
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Swarm Inference Beta report is shareable operator evidence, not "
            "a local answer transcript; raw prompts, generated text, token ids, "
            "activations, leases, credentials, private env files, and raw runtime "
            "state are excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share swarm_inference_beta*.json/md and support_bundle artifacts; "
            "they contain readiness evidence, cleanup state, hashes, counts, and "
            "diagnostics, not raw prompts or answers."
        ),
    }


def prompt_scope_text(prompt_scope: dict[str, Any]) -> str:
    return (
        f"source={prompt_scope.get('source') or 'unknown'} "
        f"count={prompt_scope.get('prompt_count')} "
        f"inline_prompt_text={bool(prompt_scope.get('inline_prompt_text'))} "
        f"terminal_next_commands_local_private={bool(prompt_scope.get('terminal_next_commands_local_private'))} "
        f"saved_artifacts_prompt_placeholders={bool(prompt_scope.get('saved_artifacts_prompt_placeholders'))} "
        f"prompt_file_path_public={bool(prompt_scope.get('prompt_file_path_public'))} "
        f"raw_prompt_public={bool(prompt_scope.get('raw_prompt_public'))} "
        f"public_artifact_safe={bool(prompt_scope.get('public_artifact_safe'))}"
    )


def _safe_relative_path(output_dir: Path, relative_path: str) -> Path:
    root = output_dir.resolve()
    target = (output_dir / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"refusing to clean path outside output dir: {relative_path}") from exc
    return target


def live_cleanup_paths(output_dir: Path) -> list[Path]:
    paths = [_safe_relative_path(output_dir, rel) for rel in LIVE_PRIVATE_RELATIVE_PATHS]
    paths.extend(_safe_relative_path(output_dir, rel) for rel in LIVE_TRANSIENT_RELATIVE_DIRS)
    return paths


def is_private_artifact_path(path: Path) -> bool:
    return path.name in PRIVATE_FILENAMES or path.name.endswith(".private.env")


def cleanup_live_transients(output_dir: Path, *, apply: bool) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    deleted_bytes = 0
    deleted_count = 0
    errors: list[dict[str, str]] = []
    for path in live_cleanup_paths(output_dir):
        try:
            relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        present = path.exists() or path.is_symlink()
        is_dir = path.is_dir() and not path.is_symlink()
        bytes_count = 0
        if path.is_file() or path.is_symlink():
            try:
                bytes_count = path.stat().st_size
            except OSError:
                bytes_count = 0
        elif is_dir:
            for child in path.rglob("*"):
                if child.is_file() and not child.is_symlink():
                    try:
                        bytes_count += child.stat().st_size
                    except OSError:
                        pass
        candidate: dict[str, Any] = {
            "path": relative,
            "present": present,
            "kind": "private_file" if is_private_artifact_path(path) else "transient_dir",
            "bytes": bytes_count,
            "action": "missing",
        }
        if not present:
            candidates.append(candidate)
            continue
        if not apply:
            candidate["action"] = "dry_run"
            candidates.append(candidate)
            continue
        try:
            if path.is_symlink():
                raise OSError("refusing to delete symlink")
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                raise OSError("refusing to delete non-regular path")
        except OSError as exc:
            candidate["action"] = "error"
            candidate["error"] = str(exc)
            errors.append({"path": relative, "error": str(exc)})
            candidates.append(candidate)
            continue
        candidate["action"] = "deleted"
        deleted_bytes += bytes_count
        deleted_count += 1
        candidates.append(candidate)
    return {
        "schema": "swarm_inference_beta_live_cleanup_v1",
        "ok": not errors,
        "mode": "apply" if apply else "dry_run",
        "candidate_count": len(candidates),
        "deleted_count": deleted_count,
        "deleted_bytes": deleted_bytes,
        "errors": errors,
        "candidates": candidates,
    }


def refresh_artifact_presence(report_path: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    payload = load_json(report_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return payload
    artifact_root = output_dir if output_dir is not None else report_path.parent
    changed = False
    for entry in artifacts.values():
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = artifact_root / path
        actual = path.is_file()
        if entry.get("present") != actual:
            entry["present"] = actual
            changed = True
    if changed:
        write_json(report_path, payload)
    return payload


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for summary in summaries.values():
            if isinstance(summary, dict):
                for code in summary.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    secret_values: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
        }, {}
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    if payload:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:], secret_values)
    return step, payload


def port_from_url(url: str, default: int = DEFAULT_PORT) -> int:
    parsed = urlparse(url)
    if parsed.port:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else default


def stage_miner_id(prefix: str, stage: str) -> str:
    return f"{prefix}-{stage}"


def root_operator_env(output_dir: Path) -> dict[str, str]:
    return parse_env(output_dir / "operator.private.env")


def secret_values_for(output_dir: Path, *extra: str) -> list[str]:
    values: list[str] = [item for item in extra if item]
    for path in [
        output_dir / "operator.private.env",
        output_dir / "stage0" / "miner.private.env",
        output_dir / "stage1" / "miner.private.env",
    ]:
        values.extend(value for value in parse_env(path).values() if value)
    return values


def render_coordinator_script(args: argparse.Namespace, *, observer_hash: str, admin_hash: str) -> str:
    output_dir = Path(args.output_dir).resolve()
    port = port_from_url(args.coordinator_url, args.port)
    registry = output_dir / "miner_registry.json"
    state_dir = output_dir / "state"
    hf_cache = f" \\\n  --hf-cache-dir {quote_env(args.hf_cache_dir)}" if args.hf_cache_dir else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail

exec crowdtensord \\
  --host {quote_env(args.bind_host)} \\
  --port {port} \\
  --state-dir {quote_env(str(state_dir))} \\
  --lease-seconds {args.lease_seconds} \\
  --inner-steps {args.request_count} \\
  --backlog 0 \\
  --task-lane python-cli:cpu:0:{WORKLOAD_TYPE} \\
  --miner-token-registry {quote_env(str(registry))} \\
  --observer-token {quote_env(observer_hash)} \\
  --admin-token {quote_env(admin_hash)} \\
  --real-llm-model-id {quote_env(args.hf_model_id)}{hf_cache}
"""


def render_stage_join_script(args: argparse.Namespace, *, stage: str) -> str:
    output_dir = Path(args.output_dir).resolve()
    stage_dir = output_dir / stage
    cache_arg = f" \\\n  --hf-cache-dir {quote_env(args.hf_cache_dir)}" if args.hf_cache_dir else ""
    env_default = str(stage_dir / "miner.private.env")
    return f"""#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${{CROWDTENSOR_MINER_ENV_FILE:-{env_default}}}"
if [ ! -f "$ENV_FILE" ]; then
  echo "missing Miner env file: $ENV_FILE" >&2
  exit 2
fi

set -a
. "$ENV_FILE"
set +a

exec crowdtensor-miner \\
  --coordinator {quote_env(args.coordinator_url.rstrip('/'))} \\
  --miner-id {quote_env(stage_miner_id(args.miner_id_prefix, stage))} \\
  --max-tasks 1 \\
  --compute-seconds {args.compute_seconds} \\
  --heartbeat-interval {args.heartbeat_interval} \\
  --enable-hf-tiny-gpt-runtime \\
  --hf-model-id {quote_env(args.hf_model_id)}{cache_arg} \\
  --real-llm-stage-role {stage} \\
  --max-request-attempts {args.max_request_attempts}
"""


def render_stage_join_markdown(args: argparse.Namespace, *, stage: str) -> str:
    return "\n".join([
        "# CrowdTensor Swarm Inference Beta Miner Join",
        "",
        f"Stage: `{stage}`",
        f"Miner ID: `{stage_miner_id(args.miner_id_prefix, stage)}`",
        f"Coordinator URL: `{args.coordinator_url.rstrip('/')}`",
        f"Workload: `{WORKLOAD_TYPE}`",
        f"Model: `{args.hf_model_id}`",
        "",
        "## Miner Host Steps",
        "",
        "1. Install CrowdTensor with the optional HF runtime: `python -m pip install -e '.[hf]'`.",
        "2. Copy only this stage directory to the Miner host.",
        "3. Run `bash miner_join.sh` from this directory.",
        "",
        "## Boundaries",
        "",
        "- CPU-only, read-only tiny GPT split inference.",
        "- This is not production Swarm Inference, P2P routing, GPU pooling, or large-model serving.",
        "",
    ])


def render_operator_runbook(args: argparse.Namespace, *, stage_reports: list[dict[str, Any]]) -> str:
    port = port_from_url(args.coordinator_url, args.port)
    return "\n".join([
        "# CrowdTensor Swarm Inference Beta",
        "",
        "This package productizes the current real tiny-LLM stage0/stage1 proof for a controlled two-machine or three-process CPU demo.",
        "",
        "## Coordinator Host",
        "",
        "```bash",
        "bash start_coordinator.sh",
        "```",
        "",
        f"The Coordinator listens on port `{port}` and uses `miner_registry.json` with hashed Miner tokens.",
        "",
        "## Miner Hosts",
        "",
        "Run one stage per host for the distinct-stage proof:",
        "",
        "```bash",
        "cd stage0 && bash miner_join.sh",
        "cd stage1 && bash miner_join.sh",
        "```",
        "",
        "## Operator Verification",
        "",
        "```bash",
        "bash verify.sh",
        "bash collect.sh",
        "```",
        "",
        "## Stage Join Packs",
        "",
        *[
            f"- `{item['stage']}`: miner `{item['miner_id']}`, files under `{item['directory']}`"
            for item in stage_reports
        ],
        "",
        "## Safety Boundary",
        "",
        "- Keep `operator.private.env`, `miner.private.env`, and `miner_registry.json` private.",
        "- Public reports must not contain raw hidden states, logits, activations, prompts, lease material, or token values.",
        "- This is CPU-only, read-only, fixed tiny GPT split inference; it is not P2P/NAT traversal, GPU pooling, training, or large-model serving.",
        "",
    ])


def render_verify_script(args: argparse.Namespace) -> str:
    output_dir = Path(args.output_dir).resolve()
    return f"""#!/usr/bin/env bash
set -euo pipefail

set -a
. {quote_env(str(output_dir / 'operator.private.env'))}
set +a

exec crowdtensor swarm-infer-beta verify \\
  --output-dir {quote_env(str(output_dir))} \\
  --coordinator-url {quote_env(args.coordinator_url.rstrip('/'))} \\
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \\
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \\
  --request-count {args.request_count} \\
  --hf-model-id {quote_env(args.hf_model_id)} \\
  --json
"""


def render_collect_script(args: argparse.Namespace) -> str:
    output_dir = Path(args.output_dir).resolve()
    return f"""#!/usr/bin/env bash
set -euo pipefail

set -a
. {quote_env(str(output_dir / 'operator.private.env'))}
set +a

exec crowdtensor swarm-infer-beta collect \\
  --output-dir {quote_env(str(output_dir))} \\
  --coordinator-url {quote_env(args.coordinator_url.rstrip('/'))} \\
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \\
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \\
  --request-count {args.request_count} \\
  --hf-model-id {quote_env(args.hf_model_id)} \\
  --json
"""


def summarize_external_beta(path: str) -> dict[str, Any]:
    if not path:
        return {"present": False}
    payload = load_json(Path(path))
    if not payload:
        return {"present": False, "path": path, "ok": False}
    runtime = payload.get("runtime_classification") if isinstance(payload.get("runtime_classification"), dict) else {}
    lifecycle = payload.get("kaggle_lifecycle") if isinstance(payload.get("kaggle_lifecycle"), dict) else {}
    return {
        "present": True,
        "path": path,
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
        "runtime_classification": {
            "external_runtime_verified": runtime.get("external_runtime_verified"),
            "kaggle_auto": runtime.get("kaggle_auto"),
        },
        "kaggle_lifecycle": {
            "kernels_deleted": lifecycle.get("kernels_deleted"),
        },
    }


def summarize_live_beta(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime_classification") if isinstance(payload.get("runtime_classification"), dict) else {}
    lifecycle = payload.get("kaggle_lifecycle") if isinstance(payload.get("kaggle_lifecycle"), dict) else {}
    workload = payload.get("workload") if isinstance(payload.get("workload"), dict) else {}
    requeue = payload.get("live_requeue_summary") if isinstance(payload.get("live_requeue_summary"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "coordinator_url": payload.get("coordinator_url"),
        "diagnosis_codes": diagnosis_codes(payload),
        "runtime_classification": {
            "kaggle_auto": runtime.get("kaggle_auto"),
            "external_runtime_verified": runtime.get("external_runtime_verified"),
            "kaggle_notebook_verified": runtime.get("kaggle_notebook_verified"),
            "stage_requeue_verified": runtime.get("stage_requeue_verified"),
        },
        "workload": {
            "workload_type": workload.get("workload_type"),
            "stage_mode": workload.get("stage_mode"),
            "request_count": workload.get("request_count"),
            "hf_model_id": workload.get("hf_model_id"),
            "require_distinct_stage_miners": workload.get("require_distinct_stage_miners"),
        },
        "kaggle_lifecycle": {
            "kernels_deleted": lifecycle.get("kernels_deleted"),
            "cleanup_required": lifecycle.get("cleanup_required"),
            "cleanup_skipped": lifecycle.get("cleanup_skipped"),
            "pushed_ref_count": len(lifecycle.get("pushed_refs") or {}),
            "stage_ref_count": len(lifecycle.get("stage_refs") or {}),
        },
        "live_requeue_summary": {
            "enabled": requeue.get("enabled"),
            "failure_mode": requeue.get("failure_mode"),
            "target_stage": requeue.get("target_stage"),
            "claim_observed": requeue.get("claim_observed"),
            "victim_kernel_deleted": requeue.get("victim_kernel_deleted"),
            "lease_expired": requeue.get("lease_expired"),
            "rescued_result": requeue.get("rescued_result"),
            "victim_result_accepted": requeue.get("victim_result_accepted"),
        },
    }


def finish_report(
    report: dict[str, Any],
    *,
    output_dir: Path,
    json_name: str,
    markdown_name: str,
    secret_values: list[str] | None = None,
    prompt_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope or {
        "source": "none",
        "prompt_count": 0,
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": False,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
    })
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "swarm inference beta report contained secret-like fragments"
    json_path = output_dir / json_name
    md_path = output_dir / markdown_name
    report.setdefault("artifacts", {})
    report["artifacts"]["swarm_inference_beta_json"] = artifact_entry(
        json_path,
        output_dir,
        kind="swarm_inference_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["swarm_inference_beta_markdown"] = artifact_entry(
        md_path,
        output_dir,
        kind="swarm_inference_beta_markdown",
    )
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["swarm_inference_beta_json"]["present"] = True
    report["artifacts"]["swarm_inference_beta_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def build_prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "miner_registry.json"
    observer_token = args.observer_token or os.urandom(24).hex()
    admin_token = args.admin_token or os.urandom(24).hex()
    observer_hash = hash_token(observer_token)
    admin_hash = hash_token(admin_token)
    write_private_env(output_dir / "operator.private.env", {
        "CROWDTENSOR_OBSERVER_TOKEN": observer_token,
        "CROWDTENSOR_ADMIN_TOKEN": admin_token,
    })

    stage_reports: list[dict[str, Any]] = []
    for stage in STAGES:
        miner_token = os.urandom(24).hex()
        miner_id = stage_miner_id(args.miner_id_prefix, stage)
        create_invite(
            registry_path=registry_path,
            miner_id=miner_id,
            coordinator_url=args.coordinator_url,
            label=f"Swarm Inference Beta {stage} Miner",
            token=miner_token,
            replace=args.replace,
        )
        stage_dir = output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        write_private_env(stage_dir / "miner.private.env", {"CROWDTENSOR_MINER_TOKEN": miner_token})
        join_script = stage_dir / "miner_join.sh"
        join_script.write_text(render_stage_join_script(args, stage=stage), encoding="utf-8")
        join_script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        (stage_dir / "MINER_JOIN.md").write_text(render_stage_join_markdown(args, stage=stage), encoding="utf-8")
        stage_reports.append({
            "stage": stage,
            "miner_id": miner_id,
            "directory": stage,
            "join_script": f"{stage}/miner_join.sh",
            "join_runbook": f"{stage}/MINER_JOIN.md",
            "stage_role": stage,
            "ready": True,
        })

    start_script = output_dir / "start_coordinator.sh"
    start_script.write_text(render_coordinator_script(args, observer_hash=observer_hash, admin_hash=admin_hash), encoding="utf-8")
    start_script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    verify_script = output_dir / "verify.sh"
    verify_script.write_text(render_verify_script(args), encoding="utf-8")
    verify_script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    collect_script = output_dir / "collect.sh"
    collect_script.write_text(render_collect_script(args), encoding="utf-8")
    collect_script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    runbook_md = output_dir / "SWARM_INFERENCE_BETA.md"
    runbook_md.write_text(render_operator_runbook(args, stage_reports=stage_reports), encoding="utf-8")

    registry = load_json(registry_path)
    miners = registry.get("miners") if isinstance(registry.get("miners"), list) else []
    registry_hashed = all(str(entry.get("token", "")).startswith("sha256:") for entry in miners if isinstance(entry, dict))
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": registry_hashed and len(stage_reports) == 2,
        "mode": "prepare",
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "request_count": args.request_count,
        "hf_model_id": args.hf_model_id,
        "workload": {
            "workload_type": WORKLOAD_TYPE,
            "route": ROUTE_NAME,
            "stage_mode": "split",
            "require_distinct_stage_miners": True,
            "read_only": True,
        },
        "stage_join_packs": stage_reports,
        "generated_files": {
            "operator_runbook": "SWARM_INFERENCE_BETA.md",
            "start_coordinator": "start_coordinator.sh",
            "verify": "verify.sh",
            "collect": "collect.sh",
        },
        "diagnosis_codes": [
            "swarm_inference_beta_prepare_ready",
            "two_machine_runbook_ready",
            "stage0_join_pack_ready",
            "stage1_join_pack_ready",
            "miner_registry_hashed",
        ] if registry_hashed else ["swarm_inference_beta_prepare_failed"],
        "artifacts": {
            "operator_runbook": artifact_entry(runbook_md, output_dir, kind="swarm_inference_beta_runbook"),
            "start_coordinator": artifact_entry(start_script, output_dir, kind="start_coordinator_script"),
            "stage0_join_script": artifact_entry(output_dir / "stage0" / "miner_join.sh", output_dir, kind="miner_join_script"),
            "stage1_join_script": artifact_entry(output_dir / "stage1" / "miner_join.sh", output_dir, kind="miner_join_script"),
        },
        "safety": {
            "private_env_files_created": True,
            "registry_hashed": registry_hashed,
            "public_artifacts_exclude_plaintext_tokens": True,
            "requires_operator_provided_transport": True,
            "not_production": True,
        },
        "limitations": limitations(),
        "recommended_next_commands": [
            f"bash {start_script}",
            f"bash {output_dir / 'stage0' / 'miner_join.sh'}",
            f"bash {output_dir / 'stage1' / 'miner_join.sh'}",
            f"bash {verify_script}",
        ],
    }
    return finish_report(
        report,
        output_dir=output_dir,
        json_name="swarm_inference_beta_prepare.json",
        markdown_name="swarm_inference_beta_prepare.md",
        secret_values=secret_values_for(output_dir),
        prompt_scope=prompt_scope_summary(args),
    )


def command_report(args: argparse.Namespace, *, action: str, command: list[str]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": action,
        "output_dir": str(output_dir),
        "command": command,
        "command_text": " ".join(shlex.quote(part) for part in command),
        "diagnosis_codes": [f"swarm_inference_beta_{action}_command_ready"],
        "safety": {
            "command_only": not args.run,
            "token_values_in_command": False,
            "not_production": True,
        },
        "limitations": limitations(),
    }
    return finish_report(
        report,
        output_dir=output_dir,
        json_name=f"swarm_inference_beta_{action}.json",
        markdown_name=f"swarm_inference_beta_{action}.md",
        secret_values=secret_values_for(output_dir),
        prompt_scope=prompt_scope_summary(args),
    )


def coordinator_command(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir).resolve()
    env = root_operator_env(output_dir)
    observer = args.observer_token or env.get("CROWDTENSOR_OBSERVER_TOKEN", "")
    admin = args.admin_token or env.get("CROWDTENSOR_ADMIN_TOKEN", "")
    if not observer or not admin:
        raise SystemExit("coordinator requires operator tokens; run prepare or pass --observer-token and --admin-token")
    port = port_from_url(args.coordinator_url, args.port)
    command = [
        "crowdtensord",
        "--host",
        args.bind_host,
        "--port",
        str(port),
        "--state-dir",
        str(output_dir / "state"),
        "--lease-seconds",
        str(args.lease_seconds),
        "--inner-steps",
        str(args.request_count),
        "--backlog",
        "0",
        "--task-lane",
        f"python-cli:cpu:0:{WORKLOAD_TYPE}",
        "--miner-token-registry",
        str(output_dir / "miner_registry.json"),
        "--observer-token",
        hash_token(observer),
        "--admin-token",
        hash_token(admin),
        "--real-llm-model-id",
        args.hf_model_id,
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return command


def miner_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    output_dir = Path(args.output_dir).resolve()
    stage = str(args.stage)
    stage_env = parse_env(output_dir / stage / "miner.private.env")
    if not stage_env.get("CROWDTENSOR_MINER_TOKEN"):
        raise SystemExit(f"miner requires {stage}/miner.private.env; run prepare first")
    command = [
        "crowdtensor-miner",
        "--coordinator",
        args.coordinator_url.rstrip("/"),
        "--miner-id",
        stage_miner_id(args.miner_id_prefix, stage),
        "--max-tasks",
        "1",
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--enable-hf-tiny-gpt-runtime",
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-stage-role",
        stage,
        "--max-request-attempts",
        str(args.max_request_attempts),
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    env = os.environ.copy()
    env.update(stage_env)
    return command, env


def build_process_action(args: argparse.Namespace, *, action: str) -> dict[str, Any]:
    if action == "coordinator":
        command = coordinator_command(args)
        env = None
    else:
        command, env = miner_command(args)
    if not args.run:
        return command_report(args, action=action, command=command)
    completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    output_dir = Path(args.output_dir).resolve()
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": completed.returncode == 0,
        "mode": action,
        "output_dir": str(output_dir),
        "returncode": completed.returncode,
        "diagnosis_codes": [
            f"swarm_inference_beta_{action}_exited"
            if completed.returncode == 0
            else f"swarm_inference_beta_{action}_failed"
        ],
        "safety": {"not_production": True},
        "limitations": limitations(),
    }
    return finish_report(
        report,
        output_dir=output_dir,
        json_name=f"swarm_inference_beta_{action}.json",
        markdown_name=f"swarm_inference_beta_{action}.md",
        secret_values=secret_values_for(output_dir),
        prompt_scope=prompt_scope_summary(args),
    )


def build_verify(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = root_operator_env(output_dir)
    observer = args.observer_token or env.get("CROWDTENSOR_OBSERVER_TOKEN", "")
    admin = args.admin_token or env.get("CROWDTENSOR_ADMIN_TOKEN", "")
    secret_values = secret_values_for(output_dir, observer, admin, *prompt_secret_values(args))
    if not observer or not admin:
        report = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "ok": False,
            "mode": "verify",
            "output_dir": str(output_dir),
            "diagnosis_codes": ["operator_tokens_missing"],
            "operator_action": ["Run prepare, source operator.private.env, or pass --observer-token and --admin-token."],
            "limitations": limitations(),
        }
        return finish_report(
            report,
            output_dir=output_dir,
            json_name="swarm_inference_beta_verify.json",
            markdown_name="swarm_inference_beta_verify.md",
            secret_values=secret_values,
            prompt_scope=prompt_scope_summary(args),
        )

    verify_dir = output_dir / "verify"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        "remote-existing",
        "--output-dir",
        str(verify_dir),
        "--coordinator-url",
        args.coordinator_url.rstrip("/"),
        "--observer-token",
        observer,
        "--admin-token",
        admin,
        "--request-count",
        str(args.request_count),
        "--hf-model-id",
        args.hf_model_id,
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.prompt_texts_file:
        command.extend(["--prompt-texts-file", args.prompt_texts_file])
    elif args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    step, payload = run_json_step(
        "remote_real_llm_sharded_beta_verify",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=secret_values,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    external = summarize_external_beta(args.real_internet_beta_report)
    codes = set(diagnosis_codes(payload))
    if external.get("present"):
        codes.add("external_beta_evidence_imported")
        codes.update(str(code) for code in external.get("diagnosis_codes") or [])
    if step.get("ok"):
        codes.update({
            CHECK_READY_CODE,
            "real_llm_split_route_ready",
            "two_machine_swarm_inference_ready",
        })
    else:
        codes.add("swarm_inference_beta_verify_failed")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "mode": "verify",
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "request_count": args.request_count,
        "hf_model_id": args.hf_model_id,
        "step": step,
        "remote_real_llm_sharded_beta_summary": summarize_remote_real_llm(payload),
        "external_beta_evidence": external,
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "remote_real_llm_sharded_beta_json": artifact_entry(
                verify_dir / "remote_real_llm_sharded_beta.json",
                output_dir,
                kind="remote_real_llm_sharded_beta",
                schema="remote_real_llm_sharded_beta_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "remote_real_llm_sharded_beta_markdown": artifact_entry(
                verify_dir / "remote_real_llm_sharded_beta.md",
                output_dir,
                kind="remote_real_llm_sharded_beta_markdown",
            ),
        },
        "safety": {
            "read_only_workload": WORKLOAD_TYPE,
            "activation_payloads_redacted": True,
            "token_values_redacted": True,
            "not_production": True,
        },
        "limitations": limitations(),
    }
    return finish_report(
        report,
        output_dir=output_dir,
        json_name="swarm_inference_beta_verify.json",
        markdown_name="swarm_inference_beta_verify.md",
        secret_values=secret_values,
        prompt_scope=prompt_scope_summary(args),
    )


def summarize_remote_real_llm(payload: dict[str, Any]) -> dict[str, Any]:
    summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    inner = next((value for value in summaries.values() if isinstance(value, dict)), {})
    session = inner.get("session") if isinstance(inner.get("session"), dict) else {}
    assignment = inner.get("stage_assignment") if isinstance(inner.get("stage_assignment"), dict) else {}
    safety = inner.get("safety") if isinstance(inner.get("safety"), dict) else {}
    artifact = inner.get("artifact") if isinstance(inner.get("artifact"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
        "session": {
            "session_id": session.get("session_id"),
            "stage_count": session.get("stage_count"),
            "request_count": session.get("request_count"),
            "model_id": session.get("model_id"),
            "artifact_hash": session.get("artifact_hash"),
        },
        "artifact": {
            "schema": artifact.get("schema"),
            "backend": artifact.get("backend"),
            "model_id": artifact.get("model_id"),
            "artifact_hash": artifact.get("artifact_hash"),
        },
        "stage_assignment": {
            "stage0_miner_id": assignment.get("stage0_miner_id"),
            "stage1_miner_id": assignment.get("stage1_miner_id"),
            "distinct_stage_miners": assignment.get("distinct_stage_miners"),
            "stage_assignment_valid": assignment.get("stage_assignment_valid"),
        },
        "safety": {
            "read_only": safety.get("read_only"),
            "redaction_ok": safety.get("redaction_ok"),
            "raw_activation_redacted": safety.get("raw_activation_redacted"),
        },
    }


def build_collect(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = root_operator_env(output_dir)
    observer = args.observer_token or env.get("CROWDTENSOR_OBSERVER_TOKEN", "")
    admin = args.admin_token or env.get("CROWDTENSOR_ADMIN_TOKEN", "")
    secret_values = secret_values_for(output_dir, observer, admin)
    if not observer or not admin:
        report = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "ok": False,
            "mode": "collect",
            "output_dir": str(output_dir),
            "diagnosis_codes": ["operator_tokens_missing"],
            "limitations": limitations(),
        }
        return finish_report(
            report,
            output_dir=output_dir,
            json_name="swarm_inference_beta_collect.json",
            markdown_name="swarm_inference_beta_collect.md",
            secret_values=secret_values,
            prompt_scope=prompt_scope_summary(args),
        )
    collect_dir = output_dir / "collect"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_home_compute_demo_pack.py"),
        "collect",
        "--workload",
        "real-llm-sharded",
        "--output-dir",
        str(collect_dir),
        "--coordinator-url",
        args.coordinator_url.rstrip("/"),
        "--miner-id",
        args.miner_id or stage_miner_id(args.miner_id_prefix, "stage1"),
        "--observer-token",
        observer,
        "--admin-token",
        admin,
        "--request-count",
        str(args.request_count),
        "--hf-model-id",
        args.hf_model_id,
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--http-timeout",
        str(args.http_timeout),
        "--artifact-timeout",
        str(args.artifact_timeout),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    step, payload = run_json_step(
        "remote_real_llm_sharded_collect",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=secret_values,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    codes = set(diagnosis_codes(payload))
    if step.get("ok"):
        codes.add("swarm_inference_beta_collect_ready")
    else:
        codes.add("swarm_inference_beta_collect_failed")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "mode": "collect",
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "step": step,
        "collect_summary": {
            "schema": payload.get("schema"),
            "ok": payload.get("ok"),
            "diagnosis_codes": payload.get("diagnosis_codes") or [],
            "status_summary": payload.get("status_summary") or {},
            "evidence_summary": payload.get("evidence_summary") or {},
            "support_bundle_summary": payload.get("support_bundle_summary") or {},
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "remote_home_compute_collect_json": artifact_entry(
                collect_dir / "remote_home_compute_collect.json",
                output_dir,
                kind="remote_home_compute_collect",
                schema="remote_home_compute_collect_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "support_bundle_json": artifact_entry(
                collect_dir / "support_bundle.json",
                output_dir,
                kind="support_bundle",
                schema="support_bundle_v1",
            ),
        },
        "safety": {
            "token_values_redacted": True,
            "raw_state_dump_in_report": False,
            "not_production": True,
        },
        "limitations": limitations(),
    }
    return finish_report(
        report,
        output_dir=output_dir,
        json_name="swarm_inference_beta_collect.json",
        markdown_name="swarm_inference_beta_collect.md",
        secret_values=secret_values,
        prompt_scope=prompt_scope_summary(args),
    )


def build_support_bundle_for_live(args: argparse.Namespace, *, output_dir: Path, live_report: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    support_json = output_dir / "support_bundle.json"
    support_md = output_dir / "support_bundle.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "support_bundle.py"),
        "--remote-report",
        str(live_report),
        "--json-out",
        str(support_json),
        "--markdown-out",
        str(support_md),
    ]
    step, payload = run_json_step(
        "swarm_inference_beta_live_support_bundle",
        command,
        runner=runner,
        timeout_seconds=min(max(float(args.timeout_seconds), 60.0), 180.0),
    )
    step["ok"] = bool(step.get("ok") and payload)
    return step, payload


def build_live(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    live_dir = output_dir / "real-internet-beta"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_llm_internet_beta_pack.py"),
        "--mode",
        "kaggle-auto",
        "--output-dir",
        str(live_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id",
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
        args.failure_mode,
        "--victim-compute-seconds",
        str(args.victim_compute_seconds),
        "--claim-observe-timeout",
        str(args.claim_observe_timeout),
        "--requeue-timeout",
        str(args.requeue_timeout),
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

    step, payload = run_json_step(
        "real_llm_internet_beta_kaggle_auto",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 300.0) + 600.0,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    live_json = live_dir / "real_llm_internet_beta.json"
    live_codes = set(diagnosis_codes(payload))
    required_codes = {
        "real_llm_internet_beta_ready",
        "real_llm_internet_alpha_ready",
        "external_runtime_verified",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "kaggle_kernels_deleted",
        "token_rotation_required",
    }
    runtime = payload.get("runtime_classification") if isinstance(payload.get("runtime_classification"), dict) else {}
    lifecycle = payload.get("kaggle_lifecycle") if isinstance(payload.get("kaggle_lifecycle"), dict) else {}
    requeue_summary = payload.get("live_requeue_summary") if isinstance(payload.get("live_requeue_summary"), dict) else {}
    missing = sorted(required_codes - live_codes)
    if not runtime.get("external_runtime_verified") and "external_runtime_verified" not in missing:
        missing.append("external_runtime_verified")
    if not lifecycle.get("kernels_deleted") and "kaggle_kernels_deleted" not in missing:
        missing.append("kaggle_kernels_deleted")
    if args.skip_kaggle_cleanup and "kaggle_cleanup_skipped" not in missing:
        missing.append("kaggle_cleanup_skipped")
    if args.failure_mode != "none":
        if "external_stage_requeue_ready" not in live_codes:
            missing.append("external_stage_requeue_ready")
        if not runtime.get("stage_requeue_verified"):
            missing.append("stage_requeue_verified")
    cleanup_summary = cleanup_live_transients(output_dir, apply=not args.keep_live_private_artifacts)
    if live_json.is_file():
        payload = refresh_artifact_presence(live_json)

    support_step: dict[str, Any] = {"name": "swarm_inference_beta_live_support_bundle", "ok": False, "skipped": True}
    support_payload: dict[str, Any] = {}
    if live_json.is_file():
        support_step, support_payload = build_support_bundle_for_live(args, output_dir=output_dir, live_report=live_json, runner=runner)

    codes = set(live_codes)
    if support_step.get("ok"):
        codes.add("swarm_inference_beta_support_bundle_ready")
    else:
        codes.add("swarm_inference_beta_support_bundle_missing")
    if cleanup_summary.get("ok") and not args.keep_live_private_artifacts:
        codes.add("swarm_inference_beta_live_private_artifacts_cleaned")
    elif args.keep_live_private_artifacts:
        codes.add("swarm_inference_beta_live_private_artifacts_retained")
    else:
        codes.add("swarm_inference_beta_live_private_cleanup_failed")
    if step.get("ok") and not missing:
        codes.update({
            CHECK_READY_CODE,
            "two_machine_swarm_inference_ready",
            "swarm_inference_beta_live_ready",
        })
    else:
        codes.add("swarm_inference_beta_live_failed")

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok") and not missing),
        "mode": "live",
        "live_mode": "kaggle-auto",
        "output_dir": str(output_dir),
        "public_host": args.public_host,
        "coordinator_url": payload.get("coordinator_url") or args.coordinator_url or f"http://{args.public_host}:{args.port}",
        "port": args.port,
        "base_port": args.base_port,
        "request_count": args.request_count,
        "hf_model_id": args.hf_model_id,
        "failure_mode": args.failure_mode,
        "step": step,
        "support_bundle_step": support_step,
        "live_cleanup_summary": cleanup_summary,
        "missing_required_codes": sorted(set(missing)),
        "real_llm_internet_beta_summary": summarize_live_beta(payload),
        "live_requeue_summary": {
            "enabled": requeue_summary.get("enabled"),
            "failure_mode": requeue_summary.get("failure_mode"),
            "target_stage": requeue_summary.get("target_stage"),
            "claim_observed": requeue_summary.get("claim_observed"),
            "victim_kernel_deleted": requeue_summary.get("victim_kernel_deleted"),
            "lease_expired": requeue_summary.get("lease_expired"),
            "rescued_result": requeue_summary.get("rescued_result"),
            "victim_result_accepted": requeue_summary.get("victim_result_accepted"),
        },
        "support_bundle_summary": {
            "schema": support_payload.get("schema"),
            "present": bool(support_payload),
            "doctor_ok": (support_payload.get("doctor") or {}).get("ok") if isinstance(support_payload.get("doctor"), dict) else None,
            "release_gate_ok": (support_payload.get("release_gate") or {}).get("ok") if isinstance(support_payload.get("release_gate"), dict) else None,
            "remote_report_ok": ((support_payload.get("reports") or {}).get("remote") or {}).get("ok")
            if isinstance(support_payload.get("reports"), dict)
            else None,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "real_llm_internet_beta_json": artifact_entry(
                live_json,
                output_dir,
                kind="real_llm_internet_beta",
                schema="real_llm_internet_beta_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "real_llm_internet_beta_markdown": artifact_entry(
                live_dir / "real_llm_internet_beta.md",
                output_dir,
                kind="real_llm_internet_beta_markdown",
            ),
            "support_bundle_json": artifact_entry(
                output_dir / "support_bundle.json",
                output_dir,
                kind="support_bundle",
                schema="support_bundle_v1",
                ok=bool(support_payload),
            ),
            "support_bundle_markdown": artifact_entry(
                output_dir / "support_bundle.md",
                output_dir,
                kind="support_bundle_markdown",
            ),
        },
        "safety": {
            "read_only_workload": WORKLOAD_TYPE,
            "cpu_only": True,
            "activation_payloads_redacted": True,
            "token_values_redacted": True,
            "kaggle_cleanup_required": True,
            "kaggle_cleanup_ok": bool(lifecycle.get("kernels_deleted")),
            "live_requeue_required": args.failure_mode != "none",
            "live_requeue_verified": bool(runtime.get("stage_requeue_verified")),
            "skip_kaggle_cleanup": bool(args.skip_kaggle_cleanup),
            "token_rotation_required": True,
            "local_private_artifacts_removed": bool(cleanup_summary.get("ok") and not args.keep_live_private_artifacts),
            "raw_runtime_state_removed": bool(cleanup_summary.get("ok") and not args.keep_live_private_artifacts),
            "keep_live_private_artifacts": bool(args.keep_live_private_artifacts),
            "not_production": True,
            "not_p2p": True,
            "not_large_model_serving": True,
        },
        "limitations": limitations(),
        "operator_action": [] if step.get("ok") and not missing else [
            "Check Kaggle authentication, public host reachability, and whether temporary kernels were cleaned up.",
            "Do not publish or commit generated Kaggle kernels/datasets; rotate tokens after temporary public HTTP proofs.",
        ],
    }
    return finish_report(
        report,
        output_dir=output_dir,
        json_name="swarm_inference_beta_live.json",
        markdown_name="swarm_inference_beta_live.md",
        prompt_scope=prompt_scope_summary(args),
    )


def cleanup_candidate(path: Path, output_dir: Path, *, include_private: bool) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    private = path.name in PRIVATE_FILENAMES
    eligible = path.exists() and path.is_file() and (include_private or not private)
    return {
        "path": relative,
        "absolute_path": str(path),
        "kind": "private" if private else "generated",
        "present": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "eligible": eligible,
        "skip_reason": "private_requires_include_private" if private and not include_private else "",
    }


def clean_paths(output_dir: Path, *, include_private: bool) -> list[Path]:
    paths = [output_dir / name for name in KNOWN_GENERATED]
    paths.extend(output_dir / stage / name for stage in STAGES for name in ["miner_join.sh", "MINER_JOIN.md"])
    paths.extend(output_dir / child / name for child in ["verify", "collect"] for name in [
        "remote_real_llm_sharded_beta.json",
        "remote_real_llm_sharded_beta.md",
        "remote_home_compute_collect.json",
        "remote_home_compute_collect.md",
        "support_bundle.json",
        "support_bundle.md",
    ])
    paths.extend(output_dir / name for name in ["support_bundle.json", "support_bundle.md"])
    paths.extend(output_dir / "real-internet-beta" / name for name in [
        "real_llm_internet_beta.json",
        "real_llm_internet_beta.md",
    ])
    if include_private:
        paths.append(output_dir / "operator.private.env")
        paths.append(output_dir / "miner_registry.json")
        paths.extend(output_dir / stage / "miner.private.env" for stage in STAGES)
    return paths


def build_clean(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    candidates = [
        cleanup_candidate(path, output_dir, include_private=args.include_private)
        for path in clean_paths(output_dir, include_private=args.include_private)
        if path.exists()
    ]
    deleted_bytes = 0
    errors: list[str] = []
    for candidate in candidates:
        candidate["action"] = "skipped"
        if not candidate.get("eligible"):
            continue
        if not args.apply:
            candidate["action"] = "dry_run"
            continue
        path = Path(str(candidate["absolute_path"]))
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError("refusing to delete non-regular file")
            path.unlink()
        except OSError as exc:
            candidate["action"] = "error"
            candidate["error"] = str(exc)
            errors.append(str(candidate["path"]))
            continue
        candidate["action"] = "deleted"
        deleted_bytes += int(candidate.get("bytes") or 0)
    if args.remove_empty_dir and args.apply:
        for child in [output_dir / "stage0", output_dir / "stage1", output_dir / "verify", output_dir / "collect", output_dir]:
            try:
                child.rmdir()
            except OSError:
                pass
    ok = not errors
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": "clean",
        "cleanup_mode": "apply" if args.apply else "dry_run",
        "output_dir": str(output_dir),
        "include_private": bool(args.include_private),
        "candidate_count": len(candidates),
        "deleted_bytes": deleted_bytes,
        "errors": errors,
        "diagnosis_codes": ["swarm_inference_beta_cleanup_ready"] if ok else ["swarm_inference_beta_cleanup_failed"],
        "candidates": candidates,
        "safety": {
            "dry_run_default": True,
            "private_files_require_include_private": True,
            "only_known_swarm_inference_beta_files": True,
            "does_not_delete_state_or_source": True,
        },
        "limitations": limitations(),
    }
    return finish_report(
        report,
        output_dir=output_dir,
        json_name="swarm_inference_beta_clean.json",
        markdown_name="swarm_inference_beta_clean.md",
        prompt_scope=prompt_scope_summary(args),
    )


def limitations() -> list[str]:
    return [
        "CPU-only tiny Hugging Face GPT stage0/stage1 split inference Beta; not production Swarm Inference",
        "Read-only fixed-scenario operator proof; not arbitrary public prompt serving",
        "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Swarm Inference Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- prompt scope: `{prompt_scope_text(prompt_scope)}`",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
    ]
    if report.get("mode") == "prepare":
        lines.extend(["## Stage Join Packs", ""])
        for item in report.get("stage_join_packs") or []:
            lines.append(f"- `{item.get('stage')}` miner `{item.get('miner_id')}` files `{item.get('directory')}`")
        lines.append("")
    if report.get("step"):
        step = report.get("step") or {}
        lines.extend(["## Step", "", f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`", ""])
    lines.extend(["## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coordinator-url", default=DEFAULT_COORDINATOR_URL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--miner-id-prefix", default="swarm-beta")
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--timeout-seconds", type=float, default=360.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or operate the user-facing Swarm Inference Beta package.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser("prepare", help="Create stage0/stage1 join packs and operator runbook.")
    add_common(prepare)
    prepare.add_argument("--observer-token", default="")
    prepare.add_argument("--admin-token", default="")
    prepare.add_argument("--replace", action="store_true")
    prepare.add_argument("--lease-seconds", type=float, default=15.0)
    prepare.add_argument("--compute-seconds", type=float, default=0.2)
    prepare.add_argument("--heartbeat-interval", type=float, default=0.1)
    prepare.add_argument("--max-request-attempts", type=int, default=120)

    coordinator = subparsers.add_parser("coordinator", help="Print or run the generated Coordinator command.")
    add_common(coordinator)
    coordinator.add_argument("--observer-token", default="")
    coordinator.add_argument("--admin-token", default="")
    coordinator.add_argument("--lease-seconds", type=float, default=15.0)
    coordinator.add_argument("--run", action="store_true", help="execute the long-running Coordinator command")

    miner = subparsers.add_parser("miner", help="Print or run a generated stage Miner command.")
    add_common(miner)
    miner.add_argument("--stage", choices=list(STAGES), required=True)
    miner.add_argument("--compute-seconds", type=float, default=0.2)
    miner.add_argument("--heartbeat-interval", type=float, default=0.1)
    miner.add_argument("--max-request-attempts", type=int, default=120)
    miner.add_argument("--run", action="store_true", help="execute the long-running Miner command")

    verify = subparsers.add_parser("verify", help="Verify a running two-stage real tiny-LLM split inference session.")
    add_common(verify)
    verify.add_argument("--observer-token", default="")
    verify.add_argument("--admin-token", default="")
    verify.add_argument("--prompt-texts", default=",".join(DEFAULT_PROMPTS))
    verify.add_argument("--prompt-texts-file", default="", help="newline-delimited bounded batch of up to 4 prompts")
    verify.add_argument("--real-internet-beta-report", default="")

    collect = subparsers.add_parser("collect", help="Collect redacted evidence and support bundle from a running Beta.")
    add_common(collect)
    collect.add_argument("--observer-token", default="")
    collect.add_argument("--admin-token", default="")
    collect.add_argument("--miner-id", default="")
    collect.add_argument("--artifact-timeout", type=float, default=60.0)

    live = subparsers.add_parser("live", help="Run the side-effectful public Kaggle auto proof for Swarm Inference Beta.")
    live.add_argument("--output-dir", default=DEFAULT_LIVE_OUTPUT_DIR)
    live.add_argument("--public-host", default="24.199.118.54")
    live.add_argument("--bind-host", default="0.0.0.0")
    live.add_argument("--port", type=int, default=DEFAULT_LIVE_PORT)
    live.add_argument("--base-port", type=int, default=DEFAULT_LIVE_PORT + 1)
    live.add_argument("--ready-url", default="")
    live.add_argument("--coordinator-url", default="")
    live.add_argument("--miner-id-prefix", default="swarm-beta-live")
    live.add_argument("--request-count", type=int, default=2)
    live.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    live.add_argument("--hf-cache-dir", default="")
    live.add_argument("--kaggle-owner", default="")
    live.add_argument("--dataset-slug", default="")
    live.add_argument("--dataset-title", default="CrowdTensor Swarm Inference Beta Live")
    live.add_argument("--kernel-slug-prefix", default="crowdtensor-swarm-inference-beta-live")
    live.add_argument("--kernel-title-prefix", default="CrowdTensor Swarm Inference Beta Live")
    live.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    live.add_argument("--skip-kaggle-cleanup", action="store_true")
    live.add_argument("--keep-live-private-artifacts", action="store_true")
    live.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="none")
    live.add_argument("--timeout-seconds", type=float, default=300.0)
    live.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    live.add_argument("--startup-timeout", type=float, default=60.0)
    live.add_argument("--process-exit-timeout", type=float, default=10.0)
    live.add_argument("--poll-interval", type=float, default=1.0)
    live.add_argument("--http-timeout", type=float, default=30.0)
    live.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    live.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    live.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    live.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    live.add_argument("--lease-seconds", type=float, default=15.0)
    live.add_argument("--compute-seconds", type=float, default=0.2)
    live.add_argument("--victim-compute-seconds", type=float, default=45.0)
    live.add_argument("--heartbeat-interval", type=float, default=0.1)
    live.add_argument("--idle-sleep", type=float, default=0.2)
    live.add_argument("--claim-observe-timeout", type=float, default=180.0)
    live.add_argument("--requeue-timeout", type=float, default=120.0)
    live.add_argument("--max-request-attempts", type=int, default=240)
    live.add_argument("--json", action="store_true")

    clean = subparsers.add_parser("clean", help="Dry-run or delete known Swarm Inference Beta generated files.")
    clean.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    clean.add_argument("--apply", action="store_true")
    clean.add_argument("--include-private", action="store_true")
    clean.add_argument("--remove-empty-dir", action="store_true")
    clean.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if hasattr(args, "request_count") and (args.request_count < 1 or args.request_count > 4):
        raise SystemExit("--request-count must be between 1 and 4")
    for name in ["port", "timeout_seconds", "remote_timeout_seconds", "http_timeout"]:
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ["base_port", "startup_timeout", "process_exit_timeout", "poll_interval", "kaggle_push_timeout_seconds", "kaggle_delete_timeout_seconds", "kaggle_status_timeout_seconds", "kaggle_status_poll_interval", "idle_sleep", "victim_compute_seconds", "claim_observe_timeout", "requeue_timeout"]:
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "lease_seconds") and args.lease_seconds <= 0:
        raise SystemExit("--lease-seconds must be positive")
    if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if hasattr(args, "heartbeat_interval") and args.heartbeat_interval <= 0:
        raise SystemExit("--heartbeat-interval must be positive")
    if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    if hasattr(args, "artifact_timeout") and args.artifact_timeout <= 0:
        raise SystemExit("--artifact-timeout must be positive")
    if getattr(args, "action", "") == "verify":
        raw_argv = list(argv if argv is not None else sys.argv[1:])
        prompt_texts_explicit = "--prompt-texts" in raw_argv or any(item.startswith("--prompt-texts=") for item in raw_argv)
        if getattr(args, "prompt_texts_file", "") and prompt_texts_explicit:
            raise SystemExit("swarm_inference_beta verify accepts either --prompt-texts or --prompt-texts-file, not both")
        try:
            if getattr(args, "prompt_texts_file", ""):
                args.prompt_texts_list = read_prompt_texts_file(args.prompt_texts_file)
                args.prompt_texts = ""
            else:
                args.prompt_texts_list = parse_prompt_texts_arg("", args.prompt_texts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if getattr(args, "action", "") == "live" and getattr(args, "failure_mode", "none") != "none":
        if args.victim_compute_seconds <= args.lease_seconds:
            args.victim_compute_seconds = args.lease_seconds + 30.0
        if args.requeue_timeout <= args.lease_seconds:
            args.requeue_timeout = args.lease_seconds + 45.0
    return args


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    if args.action == "prepare":
        return build_prepare(args)
    if args.action == "coordinator":
        return build_process_action(args, action="coordinator")
    if args.action == "miner":
        return build_process_action(args, action="miner")
    if args.action == "verify":
        return build_verify(args, runner=runner)
    if args.action == "collect":
        return build_collect(args, runner=runner)
    if args.action == "live":
        return build_live(args, runner=runner)
    if args.action == "clean":
        return build_clean(args)
    raise SystemExit(f"unknown action: {args.action}")


def print_human(report: dict[str, Any]) -> None:
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    print("CrowdTensor Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    if report.get("command_text"):
        print(f"  command: {report.get('command_text')}")
    if prompt_scope:
        print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
