#!/usr/bin/env python3
"""Probe Kaggle GPU weekly quota availability for token-file accounts.

Kaggle does not expose a stable public "remaining weekly GPU hours" API.  This
probe performs the smallest practical live check: authenticate each token-file
section, submit one private GPU script kernel, classify the push response, and
delete accepted kernels immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_gpu_token_weekly_quota_probe_v1"
WORKER_REPORT_NAME = "kaggle_gpu_token_quota_worker_report.json"
CODE_URL_RE = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^?\s]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:63].strip("-") or "ct-gpu-quota-probe"


def redact_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"KGA[A-Za-z0-9_-]+", "KGA<redacted>", text)
    text = re.sub(r"(?i)(kaggle[_-]?key|api[_-]?key|token|cookie|oauth)[=:]\S+", r"\1=<redacted>", text)
    text = re.sub(r"(?i)(bearer\s+)[a-z0-9._=-]+", r"\1<redacted>", text)
    return text


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return [item for item in sections if item["label"] != "END"]


def parse_raw_token_file(path: Path, *, username_hint: str = "", label: str = "") -> dict[str, Any]:
    """Read a private raw Kaggle token file without exposing the key in reports."""

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError("kaggle_raw_token_file_empty")
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        loaded = None
    env: dict[str, str] = {}
    if isinstance(loaded, dict):
        username = str(loaded.get("username") or loaded.get("KAGGLE_USERNAME") or username_hint or "").strip()
        key = str(loaded.get("key") or loaded.get("KAGGLE_KEY") or loaded.get("KAGGLE_API_TOKEN") or "").strip()
        if username and key:
            env = {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key, "KAGGLE_API_TOKEN": key}
    if not env:
        compact = [
            item.strip().strip("'\"")
            for item in re.split(r"[\s,]+", raw)
            if item.strip() and not item.strip().startswith("#")
        ]
        if len(compact) >= 2:
            env = {"KAGGLE_USERNAME": compact[0], "KAGGLE_KEY": compact[1], "KAGGLE_API_TOKEN": compact[1]}
        elif len(compact) == 1 and username_hint:
            env = {"KAGGLE_USERNAME": str(username_hint), "KAGGLE_KEY": compact[0], "KAGGLE_API_TOKEN": compact[0]}
    if not env:
        raise RuntimeError("kaggle_raw_token_file_format_unrecognized")
    return {
        "label": str(label or username_hint or path.stem),
        "env": env,
        "raw_token_file": True,
    }


def clean_env(token_env: dict[str, str], *, config_dir: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("KAGGLE_") and key != "MY_KAGGLE_TOKEN"
    }
    env.update(token_env)
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    return env


def run_command(command: list[str], *, env: dict[str, str], timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout,
        )
        output = redact_text(proc.stdout or "")
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_public": command,
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return {
            "ok": False,
            "returncode": None,
            "timed_out": True,
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_public": command,
            "output_tail": redact_text(output)[-4000:],
        }


def _seconds(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "total_seconds"):
        return round(float(value.total_seconds()), 6)
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _quota_to_public_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    used = _seconds(getattr(value, "time_used", None))
    reserved = _seconds(getattr(value, "time_reserved", None))
    total = _seconds(getattr(value, "total_time_allowed", None))
    minimum = _seconds(getattr(value, "minimum_time_allowed", None))
    remaining = max(0.0, total - used)
    effective_remaining = max(0.0, total - used - reserved)
    return {
        "present": True,
        "time_used_seconds": used,
        "time_reserved_seconds": reserved,
        "total_time_allowed_seconds": total,
        "minimum_time_allowed_seconds": minimum,
        "remaining_seconds": round(remaining, 6),
        "effective_remaining_after_reserved_seconds": round(effective_remaining, 6),
        "has_ever_run": bool(getattr(value, "has_ever_run", False)),
        "quota_exhausted_by_used": bool(total > 0 and used >= total),
        "reserved_exceeds_remaining": bool(total > 0 and used + reserved >= total),
    }


def fetch_accelerator_quota(env: dict[str, str]) -> dict[str, Any]:
    old_env = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        from kaggle import KaggleApi

        api = KaggleApi()
        api.authenticate()
        response = api.quota_view()
        return {
            "ok": True,
            "quota_refresh_time": response.quota_refresh_time.isoformat() if response.quota_refresh_time else "",
            "gpu_quota": _quota_to_public_dict(response.gpu_quota),
            "tpu_quota": _quota_to_public_dict(response.tpu_quota),
        }
    except Exception as exc:  # noqa: BLE001 - quota API is best-effort diagnostic evidence.
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_public": redact_text(str(exc))[-500:],
        }
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def infer_owner_from_kernel_list(output: str, fallback_label: str) -> str:
    for line in str(output or "").splitlines():
        match = re.match(r"\s*([a-z0-9-]+)/[a-z0-9-]+\s+", line.strip())
        if match:
            return match.group(1)
    return safe_slug(fallback_label)


def render_kernel() -> str:
    return "\n".join(
        [
            "import json",
            "import os",
            "from datetime import datetime, timezone",
            "",
            f"REPORT_NAME = {WORKER_REPORT_NAME!r}",
            "",
            "def utc_now():",
            "    return datetime.now(timezone.utc).isoformat()",
            "",
            "report = {",
            "    'schema': 'kaggle_gpu_token_quota_worker_report_v1',",
            "    'ok': False,",
            "    'started_at': utc_now(),",
            "    'finished_at': '',",
            "    'cuda_available': False,",
            "    'cuda_device_count': 0,",
            "    'public_artifact_safe': True,",
            "    'raw_gpu_names_public': False,",
            "}",
            "try:",
            "    import torch",
            "    report['torch_version'] = getattr(torch, '__version__', '')",
            "    report['cuda_available'] = bool(torch.cuda.is_available())",
            "    report['cuda_device_count'] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0",
            "    report['ok'] = bool(report['cuda_available'] and report['cuda_device_count'] >= 1)",
            "except Exception as exc:",
            "    report['error_type'] = type(exc).__name__",
            "    report['error_public'] = str(exc)[-300:]",
            "report['finished_at'] = utc_now()",
            "with open(os.path.join('/kaggle/working', REPORT_NAME), 'w', encoding='utf-8') as handle:",
            "    json.dump(report, handle, indent=2, sort_keys=True)",
            "print(json.dumps({",
            "    'schema': report['schema'],",
            "    'ok': report['ok'],",
            "    'cuda_device_count': report['cuda_device_count'],",
            "    'public_artifact_safe': True,",
            "}, sort_keys=True), flush=True)",
            "",
        ]
    )


def write_kernel_package(
    kernel_dir: Path,
    *,
    owner: str,
    slug: str,
    accelerator: str,
) -> str:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(render_kernel(), encoding="utf-8")
    metadata = {
        "id": f"{owner}/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": accelerator,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return f"{owner}/{slug}"


def extract_kernel_ref(text: str, fallback: str) -> str:
    match = CODE_URL_RE.search(text or "")
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return fallback


def push_accepted(step: dict[str, Any]) -> bool:
    output = str(step.get("output_tail") or "")
    return bool(step.get("ok")) and "Kernel version" in output and "successfully pushed" in output


def classify_push(step: dict[str, Any]) -> str:
    text = str(step.get("output_tail") or "").lower()
    if push_accepted(step):
        return "gpu_submission_accepted"
    if "maximum weekly gpu quota" in text:
        return "weekly_gpu_quota_exhausted"
    if "quota" in text and "gpu" in text:
        return "gpu_quota_rejected"
    if "session" in text and any(fragment in text for fragment in ["maximum", "limit", "too many"]):
        return "gpu_session_limit_rejected"
    if "401" in text or "unauthorized" in text or "unauthorised" in text:
        return "auth_failed"
    if "403" in text or "forbidden" in text or "permission" in text:
        return "permission_denied"
    if step.get("timed_out"):
        return "push_timeout"
    return "push_failed_unclassified"


def public_redaction_errors(payload: Any) -> list[str]:
    strings: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            strings.append(value)

    visit(payload)
    combined = "\n".join(strings)
    checks = {
        "raw_kga_token": re.search(r"KGA[A-Za-z0-9_-]{8,}", combined) is not None,
        "api_key": "api_key" in combined.lower(),
        "oauth": "oauth" in combined.lower(),
        "cookie": "cookie" in combined.lower(),
        "bearer": re.search(r"(?i)bearer\s+[a-z0-9._=-]{8,}", combined) is not None,
        "proxy_url": "proxy_url" in combined.lower(),
        "runtimeproxy": "runtimeproxy" in combined.lower(),
    }
    return [name for name, failed in checks.items() if failed]


def probe_account(
    section: dict[str, Any],
    *,
    output_dir: Path,
    accelerator: str,
    push_timeout_seconds: float,
    delete_timeout_seconds: float,
    kernel_timeout_seconds: int,
    slug_prefix: str,
) -> dict[str, Any]:
    label = str(section.get("label") or "")
    result: dict[str, Any] = {
        "label": label,
        "auth_ok": False,
        "owner": "",
        "push_accepted": False,
        "quota_class": "",
        "weekly_gpu_quota_available_inferred": False,
        "weekly_gpu_quota_exhausted": False,
        "cleanup": {"attempted": False, "deleted": False, "failed": False},
    }

    with tempfile.TemporaryDirectory(prefix="kaggle-token-quota-config-") as config_tmp:
        env = clean_env(dict(section.get("env") or {}), config_dir=Path(config_tmp))
        quota = fetch_accelerator_quota(env)
        result["accelerator_quota"] = quota
        gpu_quota = quota.get("gpu_quota") if isinstance(quota.get("gpu_quota"), dict) else {}
        result["weekly_gpu_quota_exhausted_by_api"] = bool(gpu_quota.get("quota_exhausted_by_used"))
        result["gpu_reserved_exceeds_remaining_by_api"] = bool(gpu_quota.get("reserved_exceeds_remaining"))
        auth_step = run_command(
            ["kaggle", "kernels", "list", "--mine", "--page-size", "5"],
            env=env,
            timeout=60,
        )
        result["auth_step"] = auth_step
        result["auth_ok"] = bool(auth_step.get("ok"))
        owner = infer_owner_from_kernel_list(str(auth_step.get("output_tail") or ""), label)
        result["owner"] = owner
        if not auth_step.get("ok"):
            result["quota_class"] = "auth_failed"
            return result

        slug = safe_slug(f"{slug_prefix}-{safe_slug(label)}-{str(int(time.time()))[-8:]}")
        kernel_dir = output_dir / "private-kaggle-gpu-quota-kernels" / safe_slug(label)
        declared_ref = write_kernel_package(kernel_dir, owner=owner, slug=slug, accelerator=accelerator)
        result["declared_kernel_ref"] = declared_ref

        push_step = run_command(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(kernel_dir),
                "-t",
                str(int(kernel_timeout_seconds)),
                "--accelerator",
                accelerator,
            ],
            env=env,
            timeout=push_timeout_seconds,
        )
        result["push_step"] = push_step
        result["quota_class"] = classify_push(push_step)
        result["push_accepted"] = push_accepted(push_step)
        result["weekly_gpu_quota_available_inferred"] = bool(result["push_accepted"])
        result["weekly_gpu_quota_exhausted"] = bool(
            result["quota_class"] == "weekly_gpu_quota_exhausted"
            or result.get("weekly_gpu_quota_exhausted_by_api")
        )

        if result["push_accepted"]:
            kernel_ref = extract_kernel_ref(str(push_step.get("output_tail") or ""), declared_ref)
            result["kernel_ref"] = kernel_ref
            result["cleanup"]["attempted"] = True
            delete_step = run_command(
                ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                env=env,
                timeout=delete_timeout_seconds,
            )
            result["cleanup"]["step"] = delete_step
            result["cleanup"]["deleted"] = bool(delete_step.get("ok"))
            result["cleanup"]["failed"] = not delete_step.get("ok")
        return result


def build_summary(accounts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "account_count": len(accounts),
        "auth_ok_count": sum(1 for item in accounts if item.get("auth_ok")),
        "gpu_submission_accepted_count": sum(1 for item in accounts if item.get("push_accepted")),
        "weekly_gpu_quota_exhausted_count": sum(1 for item in accounts if item.get("weekly_gpu_quota_exhausted")),
        "weekly_gpu_quota_exhausted_by_api_count": sum(1 for item in accounts if item.get("weekly_gpu_quota_exhausted_by_api")),
        "gpu_reserved_exceeds_remaining_by_api_count": sum(1 for item in accounts if item.get("gpu_reserved_exceeds_remaining_by_api")),
        "gpu_session_limit_rejected_count": sum(1 for item in accounts if item.get("quota_class") == "gpu_session_limit_rejected"),
        "auth_failed_count": sum(1 for item in accounts if item.get("quota_class") == "auth_failed"),
        "quota_classes": {str(item.get("label")): str(item.get("quota_class") or "") for item in accounts},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", default="~/.config/crowdtensor/kaggle-tokens.md")
    parser.add_argument("--raw-token-file", default="")
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument("--raw-token-label", default="")
    parser.add_argument("--output-dir", default="dist/kaggle-gpu-token-weekly-quota-probe")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--kernel-timeout-seconds", type=int, default=120)
    parser.add_argument("--push-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--slug-prefix", default="ct-gpu-quota")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    private_dir = output_dir / "private-kaggle-gpu-quota-kernels"
    if private_dir.exists():
        shutil.rmtree(private_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "started_at": utc_now(),
        "finished_at": "",
        "token_file_public": str(Path(args.token_file).expanduser()),
        "raw_token_file_public": str(Path(args.raw_token_file).expanduser()) if str(args.raw_token_file or "") else "",
        "raw_token_username_public": str(args.raw_token_username or ""),
        "token_values_public": False,
        "requested_accelerator": str(args.accelerator),
        "accounts": [],
        "summary": {},
        "private_kernel_payloads_removed": False,
        "public_artifact_safe": True,
        "notes": [
            "Kaggle does not expose exact remaining weekly GPU hours via this probe.",
            "Accepted GPU submission implies weekly GPU quota/session capacity was available at probe time.",
            "Accepted kernels are deleted immediately to avoid holding a GPU session.",
        ],
    }
    try:
        sections = []
        if str(args.token_file or ""):
            sections.extend(parse_token_sections(Path(args.token_file).expanduser()))
        if str(args.raw_token_file or ""):
            sections.append(
                parse_raw_token_file(
                    Path(args.raw_token_file).expanduser(),
                    username_hint=str(args.raw_token_username or ""),
                    label=str(args.raw_token_label or ""),
                )
            )
        for section in sections:
            account_result = probe_account(
                section,
                output_dir=output_dir,
                accelerator=str(args.accelerator),
                push_timeout_seconds=float(args.push_timeout_seconds),
                delete_timeout_seconds=float(args.delete_timeout_seconds),
                kernel_timeout_seconds=int(args.kernel_timeout_seconds),
                slug_prefix=str(args.slug_prefix),
            )
            report["accounts"].append(account_result)
            report["summary"] = build_summary(report["accounts"])
            write_json(output_dir / "kaggle_gpu_token_weekly_quota_probe.json", report)
    finally:
        shutil.rmtree(private_dir, ignore_errors=True)
        report["private_kernel_payloads_removed"] = not private_dir.exists()
        report["finished_at"] = utc_now()
        report["summary"] = build_summary([item for item in report.get("accounts") or [] if isinstance(item, dict)])
        report["public_artifact_safe"] = not public_redaction_errors(report)
        write_json(output_dir / "kaggle_gpu_token_weekly_quota_probe.json", report)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "kaggle_gpu_token_weekly_quota_probe: "
            f"auth_ok={report['summary'].get('auth_ok_count', 0)}/{report['summary'].get('account_count', 0)} "
            f"accepted={report['summary'].get('gpu_submission_accepted_count', 0)} "
            f"weekly_exhausted={report['summary'].get('weekly_gpu_quota_exhausted_count', 0)} "
            f"classes={report['summary'].get('quota_classes', {})}"
        )
    return 0 if report["public_artifact_safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
