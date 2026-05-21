#!/usr/bin/env python3
"""Offline security preflight for CrowdTensorD Coordinator configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.auth import HASH_PREFIX, validate_token_verifier  # noqa: E402


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEMO_TOKENS = {"local-miner", "local-observer", "local-admin"}


def is_loopback_host(host: str) -> bool:
    value = str(host or "").strip().lower()
    return value in LOOPBACK_HOSTS or value.startswith("127.")


def is_remote_bind(host: str) -> bool:
    value = str(host or "").strip().lower()
    return bool(value and not is_loopback_host(value))


def is_hashed_token(token: str) -> bool:
    return str(token or "").strip().startswith(HASH_PREFIX)


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


def token_state(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return "missing"
    if is_hashed_token(token):
        return "hashed"
    return "plaintext"


def validate_shared_token(
    checks: list[dict[str, Any]],
    *,
    token_name: str,
    token_value: str,
    remote_bind: bool,
    missing_severity: str | None = None,
) -> None:
    state = token_state(token_value)
    check_id = f"{token_name}_token"
    if state == "missing":
        severity = missing_severity or ("error" if remote_bind else "warning")
        checks.append(make_check(
            check_id,
            severity=severity,
            ok=severity != "error",
            message=f"{token_name} token is not configured",
            remediation=f"set --{token_name}-token or CROWDTENSOR_{token_name.upper()}_TOKEN before remote demos",
        ))
        return

    try:
        validate_token_verifier(token_value, field_name=f"{token_name} token")
    except ValueError as exc:
        checks.append(make_check(
            check_id,
            severity="error",
            ok=False,
            message=str(exc),
            remediation=f"generate a valid verifier with scripts/hash_token.py for the {token_name} token",
        ))
        return

    if remote_bind and token_value in DEMO_TOKENS:
        checks.append(make_check(
            check_id,
            severity="error",
            ok=False,
            message=f"{token_name} token uses a local demo value on a remote bind",
            remediation=f"replace {token_value!r} with a long random token or sha256 verifier",
        ))
        return

    if state == "plaintext":
        checks.append(make_check(
            check_id,
            severity="warning",
            ok=True,
            message=f"{token_name} token is configured as plaintext",
            remediation="store Coordinator config as sha256:<digest>; clients still send the original token",
        ))
        return

    checks.append(make_check(
        check_id,
        severity="info",
        ok=True,
        message=f"{token_name} token uses a sha256 verifier",
    ))


def load_registry(path: str) -> tuple[list[dict[str, Any]], str]:
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], f"invalid registry JSON: {exc}"
    except OSError as exc:
        return [], f"could not read miner token registry: {exc}"
    if not isinstance(payload, dict):
        return [], "miner token registry must be a JSON object"
    miners = payload.get("miners")
    if not isinstance(miners, list):
        return [], "miner token registry must contain a miners list"
    return miners, ""


def validate_registry(checks: list[dict[str, Any]], *, registry_path: str) -> bool:
    path = str(registry_path or "").strip()
    if not path:
        checks.append(make_check(
            "miner_token_registry",
            severity="warning",
            ok=True,
            message="per-Miner token registry is not configured",
            remediation="use scripts/create_miner_invite.py and --miner-token-registry for remote Miners",
        ))
        return False

    miners, error = load_registry(path)
    if error:
        checks.append(make_check(
            "miner_token_registry",
            severity="error",
            ok=False,
            message=error,
            remediation="fix the registry JSON or regenerate it with scripts/create_miner_invite.py",
        ))
        return False

    seen: set[str] = set()
    plaintext_entries: list[str] = []
    disabled_count = 0
    enabled_count = 0
    for index, entry in enumerate(miners):
        if not isinstance(entry, dict):
            checks.append(make_check(
                "miner_token_registry",
                severity="error",
                ok=False,
                message=f"miner token registry entry {index} must be an object",
                remediation="regenerate registry entries with scripts/create_miner_invite.py",
            ))
            return False
        miner_id = str(entry.get("miner_id", "")).strip()
        token = str(entry.get("token", "")).strip()
        enabled = entry.get("enabled", True)
        if not miner_id:
            checks.append(make_check(
                "miner_token_registry",
                severity="error",
                ok=False,
                message=f"miner token registry entry {index} missing miner_id",
                remediation="add a non-empty miner_id",
            ))
            return False
        if miner_id in seen:
            checks.append(make_check(
                "miner_token_registry",
                severity="error",
                ok=False,
                message=f"duplicate miner token registry miner_id: {miner_id}",
                remediation="keep one entry per miner_id",
            ))
            return False
        seen.add(miner_id)
        if not token:
            checks.append(make_check(
                "miner_token_registry",
                severity="error",
                ok=False,
                message=f"miner token registry entry {index} missing token",
                remediation="regenerate registry entries with scripts/create_miner_invite.py",
            ))
            return False
        try:
            validate_token_verifier(token, field_name=f"miner token registry entry {index} token")
        except ValueError as exc:
            checks.append(make_check(
                "miner_token_registry",
                severity="error",
                ok=False,
                message=str(exc),
                remediation="use sha256:<64 hex digest> registry token verifiers",
            ))
            return False
        if not is_hashed_token(token):
            plaintext_entries.append(miner_id)
        if not isinstance(enabled, bool):
            checks.append(make_check(
                "miner_token_registry",
                severity="error",
                ok=False,
                message=f"miner token registry entry {index} enabled must be boolean",
                remediation="set enabled to true or false",
            ))
            return False
        if not enabled:
            disabled_count += 1
        else:
            enabled_count += 1

    if plaintext_entries:
        checks.append(make_check(
            "miner_token_registry",
            severity="error",
            ok=False,
            message="miner token registry contains plaintext token verifiers",
            remediation="rotate these entries with scripts/create_miner_invite.py: " + ", ".join(plaintext_entries),
        ))
        return False

    if enabled_count == 0:
        checks.append(make_check(
            "miner_token_registry",
            severity="warning",
            ok=True,
            message="registry contains no enabled miner token entries",
            remediation="create an enabled entry with scripts/create_miner_invite.py before relying on registry auth",
        ))
        return False

    checks.append(make_check(
        "miner_token_registry",
        severity="info",
        ok=True,
        message=(
            f"registry contains {len(miners)} hashed miner token entries"
            + (f" ({disabled_count} disabled)" if disabled_count else "")
        ),
    ))
    return True


def validate_cors(checks: list[dict[str, Any]], *, cors_origins: list[str]) -> None:
    if not cors_origins:
        checks.append(make_check(
            "cors_origins",
            severity="info",
            ok=True,
            message="no explicit CORS origins configured",
        ))
        return
    risky = [
        origin for origin in cors_origins
        if origin.strip() == "*" or origin.lower().startswith("http://")
    ]
    if risky:
        checks.append(make_check(
            "cors_origins",
            severity="warning",
            ok=True,
            message="CORS origins include wildcard or cleartext HTTP origins",
            remediation="use exact HTTPS origins for remote browser demos: " + ", ".join(risky),
        ))
        return
    checks.append(make_check(
        "cors_origins",
        severity="info",
        ok=True,
        message="CORS origins are explicit and non-wildcard",
    ))


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    host = str(args.host or "").strip()
    remote_bind = is_remote_bind(host)
    checks.append(make_check(
        "bind_host",
        severity="warning" if remote_bind else "info",
        ok=True,
        message=f"Coordinator bind host is {host or '<empty>'}",
        remediation="bind to 127.0.0.1 for local demos; use VPN/TLS before remote exposure" if remote_bind else "",
    ))

    registry_auth_available = validate_registry(checks, registry_path=args.miner_token_registry)
    validate_shared_token(
        checks,
        token_name="miner",
        token_value=args.miner_token,
        remote_bind=remote_bind,
        missing_severity="warning" if registry_auth_available else None,
    )
    validate_shared_token(checks, token_name="observer", token_value=args.observer_token, remote_bind=remote_bind)
    validate_shared_token(checks, token_name="admin", token_value=args.admin_token, remote_bind=remote_bind)
    validate_cors(checks, cors_origins=args.cors_origins or [])

    errors = [check for check in checks if check["severity"] == "error" and not check["ok"]]
    warnings = [check for check in checks if check["severity"] == "warning"]
    ok = not errors and (not args.strict or not warnings)
    return {
        "ok": ok,
        "summary": {
            "host": host,
            "remote_bind": remote_bind,
            "strict": bool(args.strict),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "checks": sorted(checks, key=lambda item: (severity_rank(item["severity"]), item["id"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline CrowdTensorD security preflight checks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    parser.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    parser.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    parser.add_argument(
        "--miner-token-registry",
        default=os.environ.get("CROWDTENSOR_MINER_TOKEN_REGISTRY", ""),
    )
    parser.add_argument("--cors-origin", action="append", dest="cors_origins", default=[])
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args()


def print_human(report: dict[str, Any]) -> None:
    status = "ok" if report["ok"] else "FAIL"
    summary = report["summary"]
    print(
        f"{status} security_preflight "
        f"host={summary['host']} remote_bind={summary['remote_bind']} "
        f"errors={summary['errors']} warnings={summary['warnings']}"
    )
    for check in report["checks"]:
        marker = "ok" if check["ok"] else "FAIL"
        print(f"{marker} {check['severity']} {check['id']}: {check['message']}")
        if check.get("remediation"):
            print(f"  - {check['remediation']}")


def main() -> None:
    args = parse_args()
    report = run_preflight(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
