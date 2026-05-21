#!/usr/bin/env python3
"""Run the core CrowdTensorD browser acceptance smoke pack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tail_text(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def resolve_browser(explicit_browser: str = "") -> str:
    if explicit_browser:
        return explicit_browser
    return shutil.which("google-chrome") or shutil.which("chromium") or ""


def playwright_available() -> bool:
    return importlib.util.find_spec("playwright") is not None


def dependency_skip_reason(browser: str) -> str:
    if not playwright_available():
        return "Playwright Python is not installed"
    if not browser:
        return "Chromium-compatible browser was not found"
    return ""


def browser_checks(args: argparse.Namespace, state_root: Path, browser: str) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "webrtc",
            "port": args.base_port,
            "state_dir": str(state_root / "webrtc"),
            "command": [
                sys.executable,
                str(ROOT / "scripts" / "webrtc_smoke.py"),
                "--host",
                args.host,
                "--port",
                str(args.base_port),
                "--timeout",
                str(args.timeout),
                "--browser",
                browser,
            ],
        },
        {
            "name": "runtime_contract",
            "port": args.base_port + 1,
            "web_port": args.base_port + 2,
            "state_dir": str(state_root / "runtime_contract"),
            "command": [
                sys.executable,
                str(ROOT / "scripts" / "runtime_contract_check.py"),
                "--host",
                args.host,
                "--coordinator-port",
                str(args.base_port + 1),
                "--web-port",
                str(args.base_port + 2),
                "--state-dir",
                str(state_root / "runtime_contract"),
                "--browser",
                browser,
            ],
        },
        {
            "name": "browser_miner",
            "port": args.base_port + 3,
            "web_port": args.base_port + 4,
            "state_dir": str(state_root / "browser_miner"),
            "command": [
                sys.executable,
                str(ROOT / "scripts" / "browser_miner_smoke.py"),
                "--host",
                args.host,
                "--coordinator-port",
                str(args.base_port + 3),
                "--web-port",
                str(args.base_port + 4),
                "--state-dir",
                str(state_root / "browser_miner"),
                "--timeout",
                str(args.timeout),
                "--browser",
                browser,
            ],
        },
    ]
    if args.headful:
        for check in checks:
            check["command"].append("--headful")
    return checks


def run_check(check: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    Path(check["state_dir"]).mkdir(parents=True, exist_ok=True)
    started = utc_now()
    start_time = time.monotonic()
    result = {
        "name": check["name"],
        "port": check["port"],
        "state_dir": check["state_dir"],
        "command": check["command"],
        "started_at": started,
        "ok": False,
    }
    if "web_port" in check:
        result["web_port"] = check["web_port"]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(
            check["command"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        result.update({
            "finished_at": utc_now(),
            "duration_seconds": round(time.monotonic() - start_time, 3),
            "returncode": None,
            "error": f"timed out after {timeout_seconds:.1f}s",
            "stdout": tail_text(exc.stdout or ""),
            "stderr": tail_text(exc.stderr or ""),
        })
        return result

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    result.update({
        "finished_at": utc_now(),
        "duration_seconds": round(time.monotonic() - start_time, 3),
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": tail_text(stdout),
        "stderr": tail_text(stderr),
    })
    if completed.returncode == 0:
        result["summary"] = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    else:
        result["error"] = (
            stderr.strip().splitlines()[-1]
            if stderr.strip()
            else stdout.strip().splitlines()[-1]
            if stdout.strip()
            else "browser check failed"
        )
    return result


def build_report(
    *,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    checks: list[dict[str, Any]],
    browser: str,
    skipped: bool = False,
    skip_reason: str = "",
) -> dict[str, Any]:
    return {
        "ok": True if skipped else all(check.get("ok") for check in checks) if checks else False,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "browser": browser,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration_seconds, 3),
        "checks": checks,
    }


def write_report(report: dict[str, Any], path: str) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run core CrowdTensorD browser acceptance checks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=9310)
    parser.add_argument("--state-root", default="")
    parser.add_argument("--browser", default="")
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--check-timeout", type=float, default=120.0)
    parser.add_argument("--report", default="")
    parser.add_argument("--allow-skip", action="store_true", help="exit 0 when Playwright or Chromium is unavailable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    browser = resolve_browser(args.browser)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.state_root:
        state_root = Path(args.state_root)
        state_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_browser_acceptance_")
        state_root = Path(temp_dir.name)

    started_at = utc_now()
    start_time = time.monotonic()
    try:
        skip_reason = dependency_skip_reason(browser)
        if skip_reason:
            finished_at = utc_now()
            report = build_report(
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=time.monotonic() - start_time,
                checks=[],
                browser=browser,
                skipped=True,
                skip_reason=skip_reason,
            )
            if args.report:
                write_report(report, args.report)
            print(json.dumps(report, sort_keys=True))
            raise SystemExit(0 if args.allow_skip else 1)

        checks = [
            run_check(check, timeout_seconds=args.check_timeout)
            for check in browser_checks(args, state_root, browser)
        ]
        finished_at = utc_now()
        report = build_report(
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - start_time,
            checks=checks,
            browser=browser,
        )
        if args.report:
            write_report(report, args.report)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 1)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
