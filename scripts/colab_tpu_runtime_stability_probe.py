#!/usr/bin/env python3
"""Public-safe Colab TPU runtime stability probe.

This script reuses a locally tracked google-colab-cli session and executes a
bounded JAX workload through the runtime proxy. It never writes proxy tokens,
raw URLs, notebook hashes, or endpoint ids to the report.
"""

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


SCHEMA = "colab_tpu_runtime_stability_probe_v1"


def sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_session(config: Path, session_name: str) -> dict[str, Any]:
    data = json.loads(config.read_text())
    if session_name not in data:
        raise SystemExit(f"Session {session_name!r} not found in {config}")
    session = data[session_name]
    required = ["url", "token", "endpoint"]
    missing = [k for k in required if not session.get(k)]
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
            "output_types": [o.get("output_type") for o in outputs],
        }
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"probe_json_decode_failed: {exc}"}
    return payload


def run_probe(session: dict[str, Any], rounds: int, interval_seconds: float, matrix_size: int, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    ColabRuntime = colab_cli_runtime.load_colab_runtime_class()

    started_at = time.time()
    runtime = ColabRuntime(
        session["url"],
        session["token"],
        kernel_id=session.get("kernel_id"),
        session_id=session.get("session_id"),
    )
    observations: list[dict[str, Any]] = []
    marker = "CT_COLAB_TPU_STABILITY_PROBE"
    kernel_error: str | None = None
    try:
        for round_index in range(rounds):
            code = f"""
import json, os, platform, time
out = {{"round": {round_index}, "pid": os.getpid(), "python": platform.python_version()}}
try:
    import jax, jax.numpy as jnp
    devices = jax.devices()
    out["jax_version"] = jax.__version__
    out["device_count"] = len(devices)
    out["devices_public"] = [str(d) for d in devices]
    start = time.time()
    x = jnp.ones(({matrix_size}, {matrix_size}), dtype=jnp.bfloat16)
    y = (x @ x).block_until_ready()
    out["matmul_ready"] = True
    out["matmul_shape"] = list(y.shape)
    out["matmul_dtype"] = str(y.dtype)
    out["elapsed_ms"] = round((time.time() - start) * 1000, 3)
except Exception as exc:
    out["matmul_ready"] = False
    out["error_type"] = type(exc).__name__
    out["error"] = str(exc)[:300]
print("{marker} " + json.dumps(out, sort_keys=True))
"""
            executed_at = time.time()
            outputs = runtime.execute_code(code, timeout=timeout)
            payload = parse_probe_stdout(outputs, marker)
            payload["round"] = round_index
            payload["wall_elapsed_ms"] = round((time.time() - executed_at) * 1000, 3)
            observations.append(payload)
            if round_index != rounds - 1:
                time.sleep(interval_seconds)
    except Exception as exc:  # noqa: BLE001 - report public-safe failure
        kernel_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        try:
            runtime.stop()
        except Exception:
            pass

    ready_rounds = [
        obs
        for obs in observations
        if obs.get("device_count", 0) >= 1 and obs.get("matmul_ready") is True
    ]
    device_strings = sorted(
        {
            device
            for obs in observations
            for device in obs.get("devices_public", [])
            if isinstance(device, str)
        }
    )
    updated_session = dict(session)
    if runtime.kernel_id:
        updated_session["kernel_id"] = runtime.kernel_id
    if runtime.session_id:
        updated_session["session_id"] = runtime.session_id
    return {
        "schema": SCHEMA,
        "ok": kernel_error is None and len(ready_rounds) == rounds,
        "colab_tpu_runtime_stably_acquired": kernel_error is None and len(ready_rounds) == rounds,
        "runtime_proxy_connected": bool(observations),
        "rounds_requested": rounds,
        "rounds_completed": len(observations),
        "rounds_ready": len(ready_rounds),
        "interval_seconds": interval_seconds,
        "matrix_size": matrix_size,
        "kernel_error": kernel_error,
        "kernel_id_hash": sha256_short(runtime.kernel_id or "") if runtime.kernel_id else "",
        "session_id_hash": sha256_short(runtime.session_id or "") if runtime.session_id else "",
        "observed_device_count_max": max([obs.get("device_count", 0) for obs in observations] or [0]),
        "observed_devices_public": device_strings,
        "observations": observations,
        "duration_seconds": round(time.time() - started_at, 3),
    }, updated_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="ct-colab-tpu-v5e1")
    parser.add_argument("--config", default=os.path.expanduser("~/.config/colab-cli/sessions.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    session = load_session(Path(args.config), args.session)
    report, updated_session = run_probe(
        session=session,
        rounds=args.rounds,
        interval_seconds=args.interval_seconds,
        matrix_size=args.matrix_size,
        timeout=args.timeout,
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
    parsed_url = urlparse(session["url"])
    report.update(
        {
            "session_name": args.session,
            "accelerator": session.get("accelerator", ""),
            "variant": session.get("variant", ""),
            "endpoint_hash": sha256_short(session.get("endpoint", "")),
            "runtime_proxy_host_hash": sha256_short(parsed_url.netloc),
            "runtime_proxy_token_public": False,
            "runtime_proxy_url_public": False,
            "endpoint_public": False,
            "public_artifact_safe": True,
            "host_python": platform.python_version(),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "colab_tpu_runtime_stability_probe.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(output_path)


if __name__ == "__main__":
    main()
