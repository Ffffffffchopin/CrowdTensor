#!/usr/bin/env python3
"""Validate the optional libp2p/Kad discovery sidecar locally."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.p2p_lite import stable_peer_secret  # noqa: E402
from crowdtensor.session_protocol import build_session_request  # noqa: E402


SCHEMA = "libp2p_discovery_alpha_check_v1"


def request_json(method: str, base_url: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 5.0) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def popen_process(command: list[str]) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("NODE_NO_WARNINGS", "1")
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def stop_process(proc: subprocess.Popen[str] | None) -> dict[str, Any]:
    if proc is None:
        return {}
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate(timeout=5)
    else:
        stdout, stderr = proc.communicate(timeout=1)
    return {
        "returncode": proc.returncode,
        "stdout_tail": (stdout or "")[-1200:],
        "stderr_tail": (stderr or "")[-1200:],
    }


def wait_health(base_url: str, proc: subprocess.Popen[str], *, timeout: float) -> tuple[bool, dict[str, Any], str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        if proc.poll() is not None:
            return False, {}, f"process exited early: {stop_process(proc)}"
        try:
            payload = request_json("GET", base_url, "/real-p2p/health", timeout=2.0)
            if payload.get("ok") is True:
                return True, payload, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    return False, {}, last_error or "health timeout"


def wait_catalog(base_url: str, *, timeout: float, http_timeout: float) -> tuple[bool, dict[str, Any], str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    last_payload: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        try:
            last_payload = request_json("GET", base_url, "/real-p2p/providers", timeout=http_timeout)
            peers = last_payload.get("peers") if isinstance(last_payload.get("peers"), list) else []
            roles = {str(peer.get("peer_id") or ""): peer for peer in peers if isinstance(peer, dict)}
            if {"coord", "stage0", "stage1"}.issubset(set(roles)):
                return True, last_payload, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    return False, last_payload, last_error or "catalog sync timeout"


def libp2p_multiaddr(health: dict[str, Any]) -> str:
    libp2p = health.get("libp2p") if isinstance(health.get("libp2p"), dict) else {}
    addrs = libp2p.get("listen_multiaddrs") if isinstance(libp2p.get("listen_multiaddrs"), list) else []
    for item in addrs:
        text = str(item)
        if "/p2p/" in text:
            return text
    return str(addrs[0] if addrs else "")


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("node") is None:
        return write_report(output_dir, {
            "schema": SCHEMA,
            "ok": False,
            "diagnosis_codes": ["node_runtime_missing", "libp2p_discovery_alpha_blocked"],
        })
    if not (ROOT / "node_modules" / "libp2p").exists():
        return write_report(output_dir, {
            "schema": SCHEMA,
            "ok": False,
            "diagnosis_codes": ["node_modules_missing", "libp2p_discovery_alpha_blocked"],
            "operator_action": "Run npm install before libp2p alpha checks.",
        })
    secret = stable_peer_secret("libp2p-discovery-alpha")
    tmp_root = Path(tempfile.mkdtemp(prefix="crowdtensor_libp2p_alpha_"))
    bootstrap_proc: subprocess.Popen[str] | None = None
    worker_proc: subprocess.Popen[str] | None = None
    steps: list[dict[str, Any]] = []
    try:
        bootstrap_url = f"http://127.0.0.1:{args.base_port}"
        worker_url = f"http://127.0.0.1:{args.base_port + 1}"
        common = [
            sys.executable,
            str(ROOT / "scripts" / "real_p2p_daemon.py"),
            "--host",
            "127.0.0.1",
            "--swarm-id",
            args.swarm_id,
            "--record-secret",
            secret,
            "--require-signed",
            "--discovery-backend",
            "libp2p-kad",
            "--ttl-seconds",
            "30",
        ]
        bootstrap_proc = popen_process(common + [
            "--port",
            str(args.base_port),
            "--node-id",
            "coord",
            "--role",
            "coordinator",
            "--coordinator-url",
            "http://127.0.0.1:9191",
            "--backend",
            "cpu",
            "--peer-key-file",
            str(tmp_root / "bootstrap-peer-key.json"),
        ])
        ready, bootstrap_health, error = wait_health(bootstrap_url, bootstrap_proc, timeout=args.startup_timeout)
        steps.append({"name": "bootstrap_libp2p_daemon", "ok": ready, "error": error})
        if not ready:
            return finalize(args, output_dir, steps, {}, {}, {}, {}, bootstrap_proc, worker_proc)
        bootstrap_addr = libp2p_multiaddr(bootstrap_health)
        steps.append({"name": "bootstrap_multiaddr", "ok": bool(bootstrap_addr), "multiaddr": bootstrap_addr})
        worker_proc = popen_process(common + [
            "--port",
            str(args.base_port + 1),
            "--node-id",
            "stage0",
            "--role",
            "miner",
            "--backend",
            "cpu",
            "--stage-role",
            "stage0",
            "--stage-capability",
            "real_llm_sharded_stage0",
            "--bootstrap",
            bootstrap_addr,
            "--peer-key-file",
            str(tmp_root / "worker-peer-key.json"),
        ])
        ready, worker_health, error = wait_health(worker_url, worker_proc, timeout=args.startup_timeout)
        steps.append({"name": "worker_libp2p_daemon", "ok": ready, "error": error})
        if not ready:
            return finalize(args, output_dir, steps, bootstrap_health, worker_health, {}, {}, bootstrap_proc, worker_proc)

        stage1_peer = {
            "schema": "p2p_lite_peer_v1",
            "swarm_id": args.swarm_id,
            "peer_id": "stage1",
            "role": "miner",
            "backend": "cpu",
            "stage_role": "stage1",
            "capabilities": {
                "runtime": "python-cli",
                "backend": "cpu",
                "real_llm_sharded_stage_role": "stage1",
                "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
            },
            "ttl_seconds": 30,
        }
        # Let the Python signing path create an HMAC-compatible record through the CLI daemon API.
        import crowdtensor.real_p2p as real_p2p  # local import keeps script startup light

        record = real_p2p.build_provider_record(stage1_peer, secret)
        announce = request_json("POST", worker_url, "/real-p2p/announce", record, timeout=args.http_timeout)
        steps.append({"name": "worker_announce_stage1_record", "ok": bool(announce.get("ok")), "schema": announce.get("schema")})

        synced, catalog, error = wait_catalog(bootstrap_url, timeout=args.sync_timeout, http_timeout=args.http_timeout)
        steps.append({"name": "bootstrap_catalog_sync", "ok": synced, "error": error, "provider_count": catalog.get("provider_count")})
        session = build_session_request(
            prompt_text="CrowdTensor libp2p alpha",
            backend="cpu",
            stage_mode="split",
            max_new_tokens=2,
            route_source="real-p2p-discovery",
        )
        route = request_json("POST", bootstrap_url, "/real-p2p/route", {"session_request": session}, timeout=args.http_timeout)
        steps.append({"name": "bootstrap_route_lookup", "ok": bool(route.get("ok")), "schema": route.get("schema")})
        return finalize(args, output_dir, steps, bootstrap_health, worker_health, catalog, route, bootstrap_proc, worker_proc)
    finally:
        if worker_proc is not None and worker_proc.poll() is None:
            stop_process(worker_proc)
        if bootstrap_proc is not None and bootstrap_proc.poll() is None:
            stop_process(bootstrap_proc)
        shutil.rmtree(tmp_root, ignore_errors=True)


def finalize(
    args: argparse.Namespace,
    output_dir: Path,
    steps: list[dict[str, Any]],
    bootstrap_health: dict[str, Any],
    worker_health: dict[str, Any],
    catalog: dict[str, Any],
    route: dict[str, Any],
    bootstrap_proc: subprocess.Popen[str] | None,
    worker_proc: subprocess.Popen[str] | None,
) -> dict[str, Any]:
    peers = catalog.get("peers") if isinstance(catalog.get("peers"), list) else []
    peer_ids = {str(peer.get("peer_id") or "") for peer in peers if isinstance(peer, dict)}
    registry = catalog.get("registry") if isinstance(catalog.get("registry"), dict) else {}
    libp2p = catalog.get("libp2p") if isinstance(catalog.get("libp2p"), dict) else {}
    provider_sync = libp2p.get("provider_sync") if isinstance(libp2p.get("provider_sync"), dict) else {}
    ready = bool(
        all(step.get("ok") for step in steps)
        and {"coord", "stage0", "stage1"}.issubset(peer_ids)
        and int(registry.get("signed_provider_record_count") or 0) >= 3
        and route.get("ok") is True
        and libp2p.get("peer_id")
        and int(provider_sync.get("provider_stream_sync_success") or 0) >= 1
    )
    codes = set()
    if ready:
        codes.update({
            "libp2p_discovery_alpha_check_ready",
            "libp2p_discovery_backend_ready",
            "p2p_peer_identity_ready",
            "p2p_provider_dht_ready",
            "libp2p_provider_stream_sync_ready",
            "libp2p_route_lookup_ready",
        })
    else:
        codes.add("libp2p_discovery_alpha_blocked")
        if not {"coord", "stage0", "stage1"}.issubset(peer_ids):
            codes.add("libp2p_provider_record_sync_blocked")
        if not route.get("ok"):
            codes.add("libp2p_route_lookup_blocked")
    report = {
        "schema": SCHEMA,
        "ok": ready,
        "output_dir": str(output_dir),
        "swarm_id": args.swarm_id,
        "diagnosis_codes": sorted(codes),
        "bootstrap": {
            "schema": bootstrap_health.get("schema"),
            "libp2p": bootstrap_health.get("libp2p"),
        },
        "worker": {
            "schema": worker_health.get("schema"),
            "libp2p": worker_health.get("libp2p"),
        },
        "catalog": {
            "schema": catalog.get("schema"),
            "provider_count": catalog.get("provider_count"),
            "peer_ids": sorted(peer_ids),
            "registry": registry,
            "libp2p": libp2p,
        },
        "route": route.get("route") if isinstance(route.get("route"), dict) else {},
        "steps": steps,
        "processes": {
            "bootstrap": stop_process(bootstrap_proc),
            "worker": stop_process(worker_proc),
        },
        "safety": {
            "tokens_gossiped": False,
            "raw_prompts_gossiped": False,
            "activations_gossiped": False,
            "coordinator_result_fallback": True,
            "not_production": True,
            "not_hivemind_petals_parity": True,
            "not_large_model_serving": True,
        },
    }
    return write_report(output_dir, report)


def write_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "libp2p_discovery_alpha_check.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local libp2p discovery alpha check.")
    parser.add_argument("--output-dir", default="dist/libp2p-discovery-alpha-check")
    parser.add_argument("--swarm-id", default="libp2p-discovery-alpha")
    parser.add_argument("--base-port", type=int, default=9870)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--sync-timeout", type=float, default=20.0)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    for name in ["startup_timeout", "sync_timeout", "http_timeout"]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = run_check(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"libp2p discovery alpha ready: {report.get('ok')}")
        print(f"diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
