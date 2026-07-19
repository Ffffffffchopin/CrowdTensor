#!/usr/bin/env python3
"""Allocate or refresh a Colab TPU session through the Colab assignment API.

This uses an existing local OAuth token cache with the `colaboratory` scope.
It writes runtime proxy details only to the local colab-cli session state file;
the public report contains hashes and status only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib import error, parse, request
import uuid


SCHEMA = "colab_tpu_session_probe_v1"
XSSI_PREFIX = ")]}'\n"


def sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any], *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if mode is not None:
        path.chmod(mode)


class PublicHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = int(status)
        self.body = str(body or "")
        super().__init__(f"http_status_{self.status}")


def http_request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, body: bytes | None = None, timeout: float = 60.0) -> tuple[int, str]:
    req = request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - controlled official endpoints
            return int(getattr(resp, "status", 0) or 0), resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read(2000).decode("utf-8", "replace")
        raise PublicHTTPError(int(getattr(exc, "code", 0) or 0), body) from exc


def parse_colab_json(text: str) -> dict[str, Any]:
    if text.startswith(XSSI_PREFIX):
        text = text[len(XSSI_PREFIX) :]
    return json.loads(text) if text else {}


def refresh_access_token(token_cache: Path) -> str:
    token = load_json(token_cache)
    payload = parse.urlencode(
        {
            "client_id": token["client_id"],
            "client_secret": token["client_secret"],
            "refresh_token": token["refresh_token"],
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    status, text = http_request(
        "https://oauth2.googleapis.com/token",
        method="POST",
        headers={"content-type": "application/x-www-form-urlencoded"},
        body=payload,
        timeout=30,
    )
    data = json.loads(text)
    if status != 200 or not data.get("access_token"):
        raise RuntimeError("oauth_refresh_failed")
    return str(data["access_token"])


def uuid_to_web_safe_base64(value: uuid.UUID) -> str:
    text = str(value).replace("-", "_")
    return text + "." * (44 - len(str(value)))


def list_assignments(access_token: str, *, authuser: str = "0") -> list[dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
    }
    status, text = http_request(
        "https://colab.research.google.com/tun/m/assignments?" + parse.urlencode({"authuser": str(authuser)}),
        headers=headers,
    )
    if status != 200:
        raise RuntimeError("colab_assignment_list_failed")
    payload = parse_colab_json(text)
    assignments = payload.get("assignments")
    return assignments if isinstance(assignments, list) else []


def unassign(access_token: str, endpoint: str, *, authuser: str = "0") -> int:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
    }
    url = (
        "https://colab.research.google.com/tun/m/unassign/"
        + parse.quote(endpoint, safe="")
        + "?"
        + parse.urlencode({"authuser": str(authuser)})
    )
    status, text = http_request(url, headers=headers)
    payload = parse_colab_json(text)
    xsrf = payload.get("token")
    if not xsrf:
        return status
    status, _ = http_request(url, method="POST", headers={**headers, "X-Goog-Colab-Token": str(xsrf)}, timeout=30)
    return status


def allocate(access_token: str, accelerator: str, *, authuser: str = "0") -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
    }
    nbh = uuid_to_web_safe_base64(uuid.uuid4())
    url = (
        "https://colab.research.google.com/tun/m/assign?"
        + parse.urlencode({"nbh": nbh, "variant": "TPU", "accelerator": accelerator, "authuser": str(authuser)})
    )
    status, text = http_request(url, headers=headers, timeout=60)
    payload = parse_colab_json(text)
    xsrf = payload.get("token")
    if status != 200 or not xsrf:
        raise RuntimeError("colab_assignment_prepare_failed")
    status, text = http_request(url, method="POST", headers={**headers, "X-Goog-Colab-Token": str(xsrf)}, timeout=120)
    assigned = parse_colab_json(text)
    if status != 200 or not assigned.get("endpoint") or not isinstance(assigned.get("runtimeProxyInfo"), dict):
        raise RuntimeError("colab_assignment_post_failed")
    return assigned


def save_session(state_path: Path, session_name: str, assignment: dict[str, Any]) -> dict[str, Any]:
    state = load_json(state_path)
    proxy = assignment["runtimeProxyInfo"]
    session = {
        "name": session_name,
        "token": proxy["token"],
        "url": proxy["url"],
        "endpoint": assignment["endpoint"],
        "variant": "TPU",
        "accelerator": assignment.get("accelerator", ""),
        "kernel_id": None,
        "session_id": None,
        "last_execution": None,
        "running": None,
        "keep_alive_pid": None,
    }
    state[session_name] = session
    write_json(state_path, state, mode=0o600)
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-name", default="ct-colab-tpu-v5e1")
    parser.add_argument("--accelerator", choices=["V5E1", "V6E1"], default="V5E1")
    parser.add_argument("--token-cache", default=os.path.expanduser("~/.config/colab-exec/token.json"))
    parser.add_argument("--state-path", default=os.path.expanduser("~/.config/colab-cli/sessions.json"))
    parser.add_argument("--authuser", default="0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cleanup-other-tpu", action="store_true")
    parser.add_argument("--cleanup-before-tpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "session_name": args.session_name,
        "accelerator_requested": args.accelerator,
        "authuser": str(args.authuser),
        "public_artifact_safe": True,
        "oauth_token_public": False,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "cleanup_other_tpu": bool(args.cleanup_other_tpu),
        "cleanup_before_tpu": bool(args.cleanup_before_tpu),
        "cleaned_assignments": [],
        "precleaned_assignments": [],
    }
    try:
        access_token = refresh_access_token(Path(args.token_cache))
        before = list_assignments(access_token, authuser=str(args.authuser))
        precleaned: list[dict[str, Any]] = []
        if args.cleanup_before_tpu:
            for item in before:
                endpoint = str(item.get("endpoint") or "")
                accelerator = str(item.get("accelerator") or "")
                if endpoint and accelerator.startswith("V"):
                    precleaned.append({"endpoint_hash": sha256_short(endpoint), "status": unassign(access_token, endpoint, authuser=str(args.authuser))})
            if precleaned:
                before = list_assignments(access_token, authuser=str(args.authuser))
        assignment = allocate(access_token, args.accelerator, authuser=str(args.authuser))
        session = save_session(Path(args.state_path), args.session_name, assignment)
        retained_endpoint = str(session["endpoint"])
        cleaned: list[dict[str, Any]] = []
        if args.cleanup_other_tpu:
            for item in before:
                endpoint = str(item.get("endpoint") or "")
                accelerator = str(item.get("accelerator") or "")
                if endpoint and endpoint != retained_endpoint and accelerator.startswith("V"):
                    cleaned.append({"endpoint_hash": sha256_short(endpoint), "status": unassign(access_token, endpoint, authuser=str(args.authuser))})
        report.update(
            {
                "ok": True,
                "colab_tpu_session_allocated": True,
                "accelerator": str(assignment.get("accelerator") or ""),
                "variant": str(assignment.get("variant") or ""),
                "endpoint_hash": sha256_short(retained_endpoint),
                "runtime_proxy_host_hash": sha256_short(parse.urlparse(str(session["url"])).netloc),
                "cleaned_assignments": cleaned,
                "precleaned_assignments": precleaned,
            }
        )
    except Exception as exc:  # noqa: BLE001 - public-safe failure
        body = getattr(exc, "body", "")
        status = getattr(exc, "status", None)
        report.update(
            {
                "ok": False,
                "colab_tpu_session_allocated": False,
                "error_type": type(exc).__name__,
                "error_digest": "sha256:" + hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                "http_status": status,
                "http_body_digest": ("sha256:" + hashlib.sha256(str(body).encode("utf-8")).hexdigest()) if body else "",
                "diagnosis_codes": [
                    code
                    for code, needle in {
                        "colab_assignment_quota_or_entitlement_rejected": "quota",
                        "colab_assignment_too_many_assignments": "too many",
                        "colab_assignment_temporarily_unavailable": "temporarily",
                        "colab_assignment_resource_unavailable": "unavailable",
                        "colab_assignment_permission_denied": "permission",
                    }.items()
                    if needle in str(body).lower()
                ],
            }
        )
    finally:
        report["duration_seconds"] = round(time.time() - started, 3)
        path = output_dir / "colab_tpu_session_probe.json"
        write_json(path, report)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(path)
        if not report.get("ok"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
