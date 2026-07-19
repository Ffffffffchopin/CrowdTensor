#!/usr/bin/env python3
"""Public-safe Colab CUDA GPU runtime probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import colab_cli_runtime  # noqa: E402
from scripts import colab_cuda_session_manager  # noqa: E402


SCHEMA = "colab_cuda_runtime_probe_v1"


def sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_session(config: Path, session_name: str) -> dict[str, Any]:
    data = json.loads(config.read_text())
    if session_name not in data:
        raise SystemExit(f"Session {session_name!r} not found in {config}")
    session = data[session_name]
    required = ["url", "token", "endpoint"]
    missing = [key for key in required if not session.get(key)]
    if missing:
        raise SystemExit(f"Session {session_name!r} missing required fields: {missing}")
    return session


def parse_probe_stdout(outputs: list[dict[str, Any]], marker: str) -> dict[str, Any]:
    text_parts: list[str] = []
    for output in outputs:
        text = output.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    text = "\n".join(text_parts)
    match = re.search(re.escape(marker) + r"\s+(\{.*\})", text)
    if not match:
        return {
            "ok": False,
            "error": "probe_marker_missing",
            "output_types": [output.get("output_type") for output in outputs],
        }
    try:
        loaded = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"probe_json_decode_failed: {exc}"}
    return loaded if isinstance(loaded, dict) else {"ok": False, "error": "probe_payload_not_object"}


def public_device_summary(device: dict[str, Any]) -> dict[str, Any]:
    name = str(device.get("name") or "")
    return {
        "index": int(device.get("index") or 0),
        "name_hash": sha256_short(name),
        "name_public": False,
        "total_memory_mb": int(device.get("total_memory_mb") or 0),
        "major": int(device.get("major") or 0),
        "minor": int(device.get("minor") or 0),
    }


def run_probe(
    session: dict[str, Any],
    timeout: float,
    matrix_size: int,
    *,
    session_name: str = "ct-colab-cuda-gpu",
    config: Path | str = os.path.expanduser("~/.config/colab-cli/sessions.json"),
    token_cache: Path | str = os.path.expanduser("~/.config/colab-exec/token.json"),
    max_attempts: int = 1,
    reacquire_before: bool = False,
    accelerator: str = "T4",
    authuser: str = "0",
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = "CT_COLAB_CUDA_RUNTIME_PROBE"
    started = time.time()
    kernel_error: str | None = None
    payload: dict[str, Any] = {}
    manager_result: dict[str, Any] = {}
    try:
        code = f"""
import json, os, platform, time
out = {{"pid": os.getpid(), "python": platform.python_version()}}
try:
    import torch
    out["torch_version"] = torch.__version__
    out["cuda_version"] = str(torch.version.cuda)
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    devices = []
    for idx in range(out["cuda_device_count"]):
        props = torch.cuda.get_device_properties(idx)
        devices.append({{
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "total_memory_mb": int(props.total_memory // 1024 // 1024),
            "major": int(props.major),
            "minor": int(props.minor),
        }})
    out["devices"] = devices
    if out["cuda_device_count"] > 0:
        start = time.time()
        x = torch.ones(({matrix_size}, {matrix_size}), device="cuda:0", dtype=torch.float16)
        y = (x @ x).detach()
        torch.cuda.synchronize()
        out["cuda_matmul_ready"] = True
        out["matmul_shape"] = list(y.shape)
        out["matmul_dtype"] = str(y.dtype)
        out["max_allocated_mb"] = round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3)
        out["elapsed_ms"] = round((time.time() - start) * 1000, 3)
    else:
        out["cuda_matmul_ready"] = False
except Exception as exc:
    out["cuda_available"] = False
    out["cuda_device_count"] = 0
    out["cuda_matmul_ready"] = False
    out["error_type"] = type(exc).__name__
    out["error"] = str(exc)[:300]
print("{marker} " + json.dumps(out, sort_keys=True))
"""
        outputs, session, manager_result = colab_cuda_session_manager.execute_with_retry(
            code,
            session_name=session_name,
            state_path=config,
            timeout=timeout,
            max_attempts=max_attempts,
            token_cache=token_cache,
            accelerator=accelerator,
            authuser=authuser,
            force_reacquire_before=reacquire_before,
            heartbeat_code='print("CT_COLAB_CUDA_HEARTBEAT")',
        )
        if not manager_result.get("ok"):
            raise RuntimeError(str(manager_result.get("blocker") or "colab_cuda_execute_failed"))
        payload = parse_probe_stdout(outputs, marker)
    except Exception as exc:  # noqa: BLE001 - public-safe failure
        kernel_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    updated_session = dict(session)
    devices = [
        public_device_summary(item)
        for item in payload.get("devices", [])
        if isinstance(item, dict)
    ]
    ready = bool(
        kernel_error is None
        and payload.get("cuda_available") is True
        and int(payload.get("cuda_device_count") or 0) >= 1
        and payload.get("cuda_matmul_ready") is True
    )
    report = {
        "schema": SCHEMA,
        "ok": ready,
        "colab_cuda_runtime_ready": ready,
        "runtime_proxy_connected": bool(payload) and kernel_error is None,
        "cuda_available": payload.get("cuda_available") is True,
        "cuda_device_count": int(payload.get("cuda_device_count") or 0),
        "cuda_matmul_ready": payload.get("cuda_matmul_ready") is True,
        "torch_version": str(payload.get("torch_version") or ""),
        "cuda_version": str(payload.get("cuda_version") or ""),
        "devices": devices,
        "observed_device_count_max": int(payload.get("cuda_device_count") or 0),
        "matrix_size": int(matrix_size),
        "matmul_dtype": str(payload.get("matmul_dtype") or ""),
        "max_allocated_mb": payload.get("max_allocated_mb", 0),
        "kernel_error": kernel_error,
        "kernel_error_digest": ("sha256:" + hashlib.sha256(str(kernel_error).encode("utf-8")).hexdigest()) if kernel_error else "",
        "runtime_error_type": str(payload.get("error_type") or ""),
        "runtime_error_digest": ("sha256:" + hashlib.sha256(str(payload.get("error") or "").encode("utf-8")).hexdigest()) if payload.get("error") else "",
        "duration_seconds": round(time.time() - started, 3),
        "kernel_id_hash": sha256_short(str(updated_session.get("kernel_id") or "")) if updated_session.get("kernel_id") else "",
        "session_id_hash": sha256_short(str(updated_session.get("session_id") or "")) if updated_session.get("session_id") else "",
        "session_manager": public_manager_result(manager_result),
    }
    if not ready:
        blockers = ["colab_cuda_runtime_not_ready"]
        if kernel_error:
            blockers.append("colab_cuda_runtime_proxy_error")
        elif payload.get("cuda_available") is not True:
            blockers.append("colab_cuda_device_missing")
        elif payload.get("cuda_matmul_ready") is not True:
            blockers.append("colab_cuda_matmul_not_ready")
        report["blockers"] = blockers
    else:
        report["blockers"] = []
    return report, updated_session


def public_manager_result(result: dict[str, Any]) -> dict[str, Any]:
    attempts = []
    for item in list(result.get("attempts") or []):
        if not isinstance(item, dict):
            continue
        attempts.append({
            key: value
            for key, value in item.items()
            if key
            not in {
                "runtime_proxy_token",
                "runtime_proxy_url",
                "endpoint",
                "token",
                "url",
            }
        })
    return {
        "ok": result.get("ok") is True,
        "blocker": str(result.get("blocker") or ""),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "public_artifact_safe": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="ct-colab-cuda-gpu")
    parser.add_argument("--config", default=os.path.expanduser("~/.config/colab-cli/sessions.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--reacquire-before", action="store_true")
    parser.add_argument("--token-cache", default=os.path.expanduser("~/.config/colab-exec/token.json"))
    parser.add_argument("--accelerator", default="T4")
    parser.add_argument("--authuser", default="0")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    session = load_session(Path(args.config), args.session)
    report, updated_session = run_probe(
        session=session,
        timeout=float(args.timeout),
        matrix_size=int(args.matrix_size),
        session_name=args.session,
        config=Path(args.config),
        token_cache=Path(args.token_cache),
        max_attempts=int(args.max_attempts),
        reacquire_before=bool(args.reacquire_before),
        accelerator=str(args.accelerator or "T4"),
        authuser=str(args.authuser),
    )
    state_path = Path(args.config)
    try:
        state = json.loads(state_path.read_text())
        if isinstance(state, dict) and args.session in state and report.get("runtime_proxy_connected"):
            state[args.session] = updated_session
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            state_path.chmod(0o600)
    except Exception:
        report["session_state_update_failed"] = True
    public_session = updated_session if isinstance(updated_session, dict) and updated_session.get("url") else session
    parsed = urlparse(public_session["url"])
    report.update(
        {
            "session_name": args.session,
            "accelerator": str(public_session.get("accelerator") or ""),
            "variant": str(public_session.get("variant") or ""),
            "endpoint_hash": sha256_short(str(public_session.get("endpoint") or "")),
            "runtime_proxy_host_hash": sha256_short(parsed.netloc),
            "runtime_proxy_token_public": False,
            "runtime_proxy_url_public": False,
            "endpoint_public": False,
            "credentials_public": False,
            "private_runtime_state_public": False,
            "host_python": platform.python_version(),
            "public_artifact_safe": True,
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "colab_cuda_runtime_probe.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(output_path)
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
