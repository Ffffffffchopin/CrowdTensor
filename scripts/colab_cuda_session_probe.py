#!/usr/bin/env python3
"""Allocate or refresh a Colab CUDA GPU session through the Colab assignment API.

This is the GPU counterpart to the retained TPU session probe. It writes
runtime proxy details only to the local colab-cli session state file; the public
report contains hashes and status only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib import parse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import colab_tpu_session_probe as base


SCHEMA = "colab_cuda_session_probe_v1"
DEFAULT_SESSION_NAME = "ct-colab-cuda-gpu"


def sha256_short(value: str) -> str:
    return base.sha256_short(value)


def allocate_gpu(access_token: str, accelerator: str, *, authuser: str = "0") -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
    }
    nbh = base.uuid_to_web_safe_base64(__import__("uuid").uuid4())
    query = {"nbh": nbh, "variant": "GPU", "authuser": str(authuser)}
    if accelerator:
        query["accelerator"] = accelerator
    url = "https://colab.research.google.com/tun/m/assign?" + parse.urlencode(query)
    status, text = base.http_request(url, headers=headers, timeout=60)
    payload = base.parse_colab_json(text)
    xsrf = payload.get("token")
    if status != 200 or not xsrf:
        raise RuntimeError("colab_gpu_assignment_prepare_failed")
    status, text = base.http_request(url, method="POST", headers={**headers, "X-Goog-Colab-Token": str(xsrf)}, timeout=120)
    assigned = base.parse_colab_json(text)
    if status != 200 or not assigned.get("endpoint") or not isinstance(assigned.get("runtimeProxyInfo"), dict):
        raise RuntimeError("colab_gpu_assignment_post_failed")
    return assigned


def save_gpu_session(state_path: Path, session_name: str, assignment: dict[str, Any]) -> dict[str, Any]:
    state = base.load_json(state_path)
    proxy = assignment["runtimeProxyInfo"]
    session = {
        "name": session_name,
        "token": proxy["token"],
        "url": proxy["url"],
        "endpoint": assignment["endpoint"],
        "variant": "GPU",
        "accelerator": assignment.get("accelerator", ""),
        "kernel_id": None,
        "session_id": None,
        "last_execution": None,
        "running": None,
        "keep_alive_pid": None,
    }
    state[session_name] = session
    base.write_json(state_path, state, mode=0o600)
    return session


def assignment_is_gpu(item: dict[str, Any]) -> bool:
    variant = str(item.get("variant") or "").upper()
    accelerator = str(item.get("accelerator") or "").upper()
    return variant == "GPU" or accelerator in {"T4", "L4", "A100", "V100", "P100", "H100", "GPU"}


def build_failure_report(args: argparse.Namespace, exc: Exception, *, started: float) -> dict[str, Any]:
    body = getattr(exc, "body", "")
    status = getattr(exc, "status", None)
    return {
        "schema": SCHEMA,
        "ok": False,
        "colab_cuda_session_allocated": False,
        "session_name": args.session_name,
        "accelerator_requested": str(args.accelerator or ""),
        "authuser": str(args.authuser),
        "error_type": type(exc).__name__,
        "error_digest": "sha256:" + hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        "http_status": status,
        "http_body_digest": ("sha256:" + hashlib.sha256(str(body).encode("utf-8")).hexdigest()) if body else "",
        "diagnosis_codes": [
            code
            for code, needle in {
                "colab_gpu_assignment_quota_or_entitlement_rejected": "quota",
                "colab_gpu_assignment_too_many_assignments": "too many",
                "colab_gpu_assignment_temporarily_unavailable": "temporarily",
                "colab_gpu_assignment_resource_unavailable": "unavailable",
                "colab_gpu_assignment_permission_denied": "permission",
            }.items()
            if needle in str(body).lower()
        ],
        "blockers": [
            "colab_cuda_session_not_allocated",
            f"colab_gpu_assignment_http_{status}" if status else "colab_gpu_assignment_error",
        ],
        "duration_seconds": round(time.time() - started, 3),
        "public_artifact_safe": True,
        "oauth_token_public": False,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "credentials_public": False,
        "private_runtime_state_public": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-name", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--accelerator", default="T4")
    parser.add_argument("--token-cache", default=os.path.expanduser("~/.config/colab-exec/token.json"))
    parser.add_argument("--state-path", default=os.path.expanduser("~/.config/colab-cli/sessions.json"))
    parser.add_argument("--authuser", default="0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cleanup-other-gpu", action="store_true")
    parser.add_argument("--cleanup-before-gpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "session_name": args.session_name,
        "accelerator_requested": str(args.accelerator or ""),
        "authuser": str(args.authuser),
        "cleanup_other_gpu": bool(args.cleanup_other_gpu),
        "cleanup_before_gpu": bool(args.cleanup_before_gpu),
        "cleaned_assignments": [],
        "precleaned_assignments": [],
        "public_artifact_safe": True,
        "oauth_token_public": False,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "credentials_public": False,
        "private_runtime_state_public": False,
    }
    try:
        access_token = base.refresh_access_token(Path(args.token_cache))
        before = base.list_assignments(access_token, authuser=str(args.authuser))
        precleaned: list[dict[str, Any]] = []
        if args.cleanup_before_gpu:
            for item in before:
                endpoint = str(item.get("endpoint") or "")
                if endpoint and assignment_is_gpu(item):
                    precleaned.append({"endpoint_hash": sha256_short(endpoint), "status": base.unassign(access_token, endpoint, authuser=str(args.authuser))})
            if precleaned:
                before = base.list_assignments(access_token, authuser=str(args.authuser))
        assignment = allocate_gpu(access_token, str(args.accelerator or ""), authuser=str(args.authuser))
        session = save_gpu_session(Path(args.state_path), args.session_name, assignment)
        retained_endpoint = str(session["endpoint"])
        cleaned: list[dict[str, Any]] = []
        if args.cleanup_other_gpu:
            for item in before:
                endpoint = str(item.get("endpoint") or "")
                if endpoint and endpoint != retained_endpoint and assignment_is_gpu(item):
                    cleaned.append({"endpoint_hash": sha256_short(endpoint), "status": base.unassign(access_token, endpoint, authuser=str(args.authuser))})
        report.update(
            {
                "ok": True,
                "colab_cuda_session_allocated": True,
                "accelerator": str(assignment.get("accelerator") or ""),
                "variant": str(assignment.get("variant") or "GPU"),
                "endpoint_hash": sha256_short(retained_endpoint),
                "runtime_proxy_host_hash": sha256_short(parse.urlparse(str(session["url"])).netloc),
                "cleaned_assignments": cleaned,
                "precleaned_assignments": precleaned,
                "duration_seconds": round(time.time() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001 - public-safe failure
        report = build_failure_report(args, exc, started=started)
    finally:
        path = output_dir / "colab_cuda_session_probe.json"
        base.write_json(path, report)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(path)
        if not report.get("ok"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
