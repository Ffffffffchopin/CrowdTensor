#!/usr/bin/env python3
"""First-run diagnostics for a CrowdTensorD checkout."""

from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import security_preflight  # noqa: E402
import runtime_matrix  # noqa: E402


def severity_rank(severity: str) -> int:
    return {"info": 0, "warning": 1, "error": 2}.get(severity, 0)


def make_check(
    check_id: str,
    *,
    severity: str,
    ok: bool,
    message: str,
    remediation: str = "",
) -> dict[str, Any]:
    return {
        "id": check_id,
        "severity": severity,
        "ok": bool(ok),
        "message": message,
        "remediation": remediation,
    }


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def required_files(root: Path) -> list[str]:
    return [
        "README.md",
        "pyproject.toml",
        "coordinator.py",
        "miner_cli.py",
        "crowdtensor",
        "scripts/runtime_acceptance_pack.py",
    ]


def check_python_version() -> dict[str, Any]:
    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 11)
    return make_check(
        "python_version",
        severity="info" if ok else "error",
        ok=ok,
        message=f"Python {current}",
        remediation="install Python 3.11 or newer" if not ok else "",
    )


def check_required_files(root: Path) -> dict[str, Any]:
    missing = [relative for relative in required_files(root) if not (root / relative).exists()]
    return make_check(
        "required_files",
        severity="info" if not missing else "error",
        ok=not missing,
        message="required project files are present" if not missing else "missing required files: " + ", ".join(missing),
        remediation="run doctor from a complete CrowdTensorD checkout" if missing else "",
    )


def check_dependency(name: str, package_hint: str) -> dict[str, Any]:
    ok = module_available(name)
    return make_check(
        f"dependency_{name}",
        severity="info" if ok else "error",
        ok=ok,
        message=f"Python module {name} is available" if ok else f"Python module {name} is not available",
        remediation=f"install project dependencies with: pip install -r requirements.txt or pip install -e {package_hint}" if not ok else "",
    )


def check_imports() -> dict[str, Any]:
    try:
        import coordinator  # noqa: F401
        import miner_cli  # noqa: F401
        import crowdtensor.state_store  # noqa: F401
    except Exception as exc:
        return make_check(
            "project_imports",
            severity="error",
            ok=False,
            message=f"project imports failed: {exc}",
            remediation="install the package in editable mode with: pip install -e .[dev]",
        )
    return make_check(
        "project_imports",
        severity="info",
        ok=True,
        message="Coordinator, Miner, and core package imports succeeded",
    )


def check_entrypoints() -> dict[str, Any]:
    missing = [name for name in ["crowdtensord", "crowdtensor-miner"] if shutil.which(name) is None]
    if missing:
        return make_check(
            "console_entrypoints",
            severity="warning",
            ok=True,
            message="console entrypoints are not on PATH: " + ", ".join(missing),
            remediation="install the package with pip install -e . or run scripts with python3 directly",
        )
    return make_check(
        "console_entrypoints",
        severity="info",
        ok=True,
        message="crowdtensord and crowdtensor-miner are on PATH",
    )


def check_state_dir(path: Path) -> dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return make_check(
            "state_dir",
            severity="error",
            ok=False,
            message=f"state directory cannot be created: {path} ({exc})",
            remediation="choose a writable --state-dir",
        )
    if not os.access(path, os.W_OK):
        return make_check(
            "state_dir",
            severity="error",
            ok=False,
            message=f"state directory is not writable: {path}",
            remediation="fix permissions or choose a writable --state-dir",
        )
    return make_check(
        "state_dir",
        severity="info",
        ok=True,
        message=f"state directory is writable: {path}",
    )


def check_port_available(host: str, port: int) -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError as exc:
        if getattr(exc, "errno", None) in {errno.EPERM, errno.EACCES}:
            return make_check(
                "port_bind",
                severity="warning",
                ok=True,
                message=f"socket bind probe is not permitted for {host}:{port}: {exc}",
                remediation="run acceptance checks in a normal shell or CI host if this sandbox blocks localhost sockets",
            )
        return make_check(
            "port_bind",
            severity="error",
            ok=False,
            message=f"cannot bind {host}:{port}: {exc}",
            remediation="stop the process using the port or pass a different --port",
        )
    return make_check(
        "port_bind",
        severity="info",
        ok=True,
        message=f"{host}:{port} is available",
    )


def check_browser_dependencies(explicit_browser: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    playwright_ok = module_available("playwright")
    checks.append(make_check(
        "browser_playwright",
        severity="info" if playwright_ok else "warning",
        ok=True,
        message="Playwright Python module is available" if playwright_ok else "Playwright Python module is not installed",
        remediation="install browser extras with: pip install -e .[browser]" if not playwright_ok else "",
    ))
    browser_path = explicit_browser or shutil.which("google-chrome") or shutil.which("chromium") or ""
    checks.append(make_check(
        "browser_chromium",
        severity="info" if browser_path else "warning",
        ok=True,
        message=f"Chromium-compatible browser found: {browser_path}" if browser_path else "Chromium-compatible browser was not found",
        remediation="install Google Chrome/Chromium or pass --browser /path/to/browser" if not browser_path else "",
    ))
    return checks


def security_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        host=args.host,
        miner_token=args.miner_token,
        observer_token=args.observer_token,
        admin_token=args.admin_token,
        miner_token_registry=args.miner_token_registry,
        cors_origins=args.cors_origins,
        strict=args.strict,
    )


def adapt_security_preflight(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for check in report.get("checks", []):
        checks.append(make_check(
            f"security_{check.get('id', 'unknown')}",
            severity=str(check.get("severity", "info")),
            ok=bool(check.get("ok")),
            message=str(check.get("message", "")),
            remediation=str(check.get("remediation", "")),
        ))
    return checks


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    state_dir = Path(args.state_dir)
    if not state_dir.is_absolute():
        state_dir = root / state_dir
    matrix = runtime_matrix.build_matrix(root=root, browser_path=args.browser_path)

    checks: list[dict[str, Any]] = [
        check_python_version(),
        check_required_files(root),
        check_dependency("fastapi", ".[dev]"),
        check_dependency("uvicorn", ".[dev]"),
        check_imports(),
        check_entrypoints(),
        check_state_dir(state_dir),
        check_port_available(args.host, args.port),
    ]
    if args.browser:
        checks.extend(check_browser_dependencies(args.browser_path))
    if args.remote_demo:
        checks.extend(adapt_security_preflight(security_preflight.run_preflight(security_args(args))))

    errors = [check for check in checks if check["severity"] == "error" and not check["ok"]]
    warnings = [check for check in checks if check["severity"] == "warning"]
    ok = not errors and (not args.strict or not warnings)
    return {
        "ok": ok,
        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
            "info": len([check for check in checks if check["severity"] == "info"]),
            "strict": bool(args.strict),
            "remote_demo": bool(args.remote_demo),
            "browser": bool(args.browser),
            "runtime_matrix_available": matrix["summary"]["available"],
            "runtime_matrix_optional_missing": matrix["summary"]["optional_missing"],
            "runtime_matrix_blocked": matrix["summary"]["blocked"],
        },
        "runtime_matrix": {
            "ok": matrix["ok"],
            "summary": matrix["summary"],
            "configured_runtimes": matrix["configured_runtimes"],
            "recommended_next_commands": matrix["recommended_next_commands"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "root": str(root),
            "host": args.host,
            "port": args.port,
            "state_dir": str(state_dir),
        },
        "checks": sorted(checks, key=lambda item: (severity_rank(item["severity"]), item["id"])),
    }


def print_human(report: dict[str, Any]) -> None:
    status = "ok" if report["ok"] else "FAIL"
    summary = report["summary"]
    print(
        f"{status} doctor "
        f"errors={summary['errors']} warnings={summary['warnings']} "
        f"remote_demo={summary['remote_demo']} browser={summary['browser']}"
    )
    for check in report["checks"]:
        marker = "ok" if check["ok"] else "FAIL"
        print(f"{marker} {check['severity']} {check['id']}: {check['message']}")
        if check.get("remediation"):
            print(f"  - {check['remediation']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrowdTensorD first-run diagnostics.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default="state/doctor")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--remote-demo", action="store_true", help="include remote-demo security checks")
    parser.add_argument("--browser", action="store_true", help="include lightweight browser dependency checks")
    parser.add_argument("--browser-path", default="")
    parser.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    parser.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    parser.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    parser.add_argument(
        "--miner-token-registry",
        default=os.environ.get("CROWDTENSOR_MINER_TOKEN_REGISTRY", ""),
    )
    parser.add_argument("--cors-origin", action="append", dest="cors_origins", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
