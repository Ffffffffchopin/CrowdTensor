"""Public-safe Kaggle command helpers for bounded CUDA training gates."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scripts.glm52_kaggle_stage_worker_push_probe import isolated_kaggle_env, parse_raw_token_file


STATUS_RE = re.compile(r'has status "([^"]+)"')
CODE_URL_RE = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^?\s]+)")
SENSITIVE = (
    "KAGGLE_KEY",
    "KAGGLE_API_TOKEN",
    "Authorization:",
    "Bearer ",
    "Cookie:",
    "Set-Cookie",
    "CROWDTENSOR_MINER_TOKEN",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    return re.sub(r"-+", "-", slug)[:63].strip("-") or "ct-cuda-training"


def safe_output(value: str, *, limit: int = 3000) -> str:
    text = str(value or "")[-limit:]
    text = re.sub(r"(?i)(kaggle[_-]?(?:key|api_token)|token|cookie|oauth)[=:]\S+", r"\1=<redacted>", text)
    text = re.sub(r"(?i)(bearer\s+)[a-z0-9._=-]+", r"\1<redacted>", text)
    for fragment in SENSITIVE:
        text = text.replace(fragment, "<redacted>")
    text = re.sub(r"/(?:root|tmp|home|kaggle)/[^\s]+", "<local-path>", text)
    return text


def public_command(command: list[str]) -> list[str]:
    result: list[str] = []
    redact_next_path = False
    for item in command:
        value = str(item)
        if redact_next_path or value.startswith("/"):
            result.append("<local-path>")
            redact_next_path = False
            continue
        if value in {"-p", "--path", "--output-dir"}:
            result.append(value)
            redact_next_path = True
            continue
        if any(word in value.lower() for word in ("token", "cookie", "credential")):
            result.append("<private>")
        else:
            result.append(value)
    return result


def run_command(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "ok": process.returncode == 0,
            "returncode": process.returncode,
            "timed_out": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_public": public_command(command),
            "output_tail": safe_output(process.stdout or ""),
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return {
            "ok": False,
            "returncode": None,
            "timed_out": True,
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_public": public_command(command),
            "output_tail": safe_output(output),
        }


@contextmanager
def kaggle_env(raw_token_file: str | Path, *, username_hint: str) -> Iterator[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="crowdtensor-cuda-kaggle-config-") as config_dir:
        token_env = parse_raw_token_file(Path(raw_token_file).expanduser(), username_hint=username_hint)
        yield isolated_kaggle_env(token_env, config_dir)


def authenticated_owner(env: dict[str, str]) -> str:
    code = (
        "from kaggle.api.kaggle_api_extended import KaggleApi; "
        "api=KaggleApi(); api.authenticate(); "
        "print(api.config_values.get('username') or api.config_values.get('user') or '')"
    )
    step = run_command([os.sys.executable, "-c", code], env=env, timeout=30.0)
    if not step.get("ok"):
        return ""
    lines = [line.strip() for line in str(step.get("output_tail") or "").splitlines() if line.strip()]
    return safe_slug(lines[-1]) if lines else ""


def extract_kernel_ref(output: str, fallback: str = "") -> str:
    match = CODE_URL_RE.search(str(output or ""))
    return f"{match.group(1)}/{match.group(2)}" if match else str(fallback)


def push_accepted(step: dict[str, Any]) -> bool:
    output = str(step.get("output_tail") or "")
    return bool(step.get("ok")) and "successfully pushed" in output.lower()


def delete_succeeded_or_absent(step: dict[str, Any]) -> bool:
    if step.get("ok") is True:
        return True
    output = str(step.get("output_tail") or "").lower()
    return any(
        marker in output
        for marker in (
            "not found",
            "does not exist",
            "already deleted",
            "notebook not found",
        )
    )


def status_class(output: str) -> str:
    match = STATUS_RE.search(str(output or ""))
    status = match.group(1) if match else str(output or "").splitlines()[-1:] or [""]
    value = status if isinstance(status, str) else status[0]
    upper = str(value).upper()
    if "RUNNING" in upper:
        return "running"
    if "COMPLETE" in upper or "SUCCESS" in upper:
        return "complete"
    if any(word in upper for word in ("ERROR", "FAILED", "CANCEL")):
        return "failed"
    if any(word in upper for word in ("QUEUE", "PENDING", "PREPAR", "INITIAL")):
        return "queued"
    return "unknown"


def public_safety_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True)
    lowered = encoded.lower()
    patterns = [
        "kaggle_key",
        "kaggle_api_token",
        "bearer ",
        "authorization:",
        "cookie:",
        '"payload_b64":',
        '"activation_gradient":',
        '"raw_training_text":',
        "/root/",
        "/tmp/",
        "/home/",
    ]
    return [item for item in patterns if item in lowered]
