#!/usr/bin/env python3
"""Run the CrowdTensor real-P2P provider discovery daemon."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.real_p2p import (  # noqa: E402
    DEFAULT_DISCOVERY_BACKEND,
    DISCOVERY_BACKENDS,
    LIBP2P_KAD_BACKEND,
    LIBP2P_KAD_COMPAT_BACKEND,
    ProviderRecordStore,
    build_provider_record,
    create_app,
    stable_node_id,
)


def parse_capability(value: str) -> tuple[str, str]:
    if "=" not in value:
        return value.strip(), "true"
    key, item = value.split("=", 1)
    return key.strip(), item.strip()


def build_local_record(args: argparse.Namespace) -> dict:
    capabilities = {}
    for item in args.capability or []:
        key, value = parse_capability(item)
        if not key:
            continue
        capabilities[key] = [part.strip() for part in value.split(",") if part.strip()] if "," in value else value
    if args.backend:
        capabilities.setdefault("backend", args.backend)
    if args.stage_role:
        capabilities.setdefault("real_llm_sharded_stage_role", args.stage_role)
    if args.stage_capability:
        capabilities.setdefault("real_llm_sharded_stage_capabilities", list(args.stage_capability))
    urls = {}
    if args.peer_url:
        urls["peer"] = args.peer_url
    if args.coordinator_url:
        urls["coordinator"] = args.coordinator_url
    peer = {
        "schema": "p2p_lite_peer_v1",
        "swarm_id": args.swarm_id,
        "peer_id": args.node_id or stable_node_id(f"{args.swarm_id}:{args.role}:{args.peer_url}:{args.coordinator_url}"),
        "role": args.role,
        "urls": urls,
        "capabilities": capabilities,
        "stage_role": args.stage_role,
        "backend": args.backend,
        "ttl_seconds": args.ttl_seconds,
    }
    return build_provider_record(peer, args.record_secret, default_ttl=args.ttl_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CrowdTensor real-P2P provider discovery daemon.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--public-host", default="")
    parser.add_argument("--swarm-id", default="default")
    parser.add_argument("--node-id", default="")
    parser.add_argument("--role", choices=["coordinator", "miner", "observer"], default="observer")
    parser.add_argument("--peer-url", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--backend", choices=["", "cpu", "cuda"], default="")
    parser.add_argument("--stage-role", choices=["", "stage0", "stage1", "both"], default="")
    parser.add_argument("--stage-capability", action="append", default=[])
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--bootstrap", action="append", default=[])
    parser.add_argument("--ttl-seconds", type=float, default=60.0)
    parser.add_argument("--record-secret", "--peer-secret", dest="record_secret", default="")
    parser.add_argument("--require-signed", action="store_true")
    parser.add_argument("--signature-max-age-seconds", type=float, default=3600.0)
    parser.add_argument("--discovery-backend", choices=sorted(DISCOVERY_BACKENDS), default=DEFAULT_DISCOVERY_BACKEND)
    parser.add_argument("--libp2p-host", default="127.0.0.1")
    parser.add_argument("--libp2p-port", type=int, default=0)
    parser.add_argument("--libp2p-public-host", default="")
    parser.add_argument("--peer-key-file", default="")
    parser.add_argument("--kad-protocol", default="/crowdtensor/kad/1.0.0")
    parser.add_argument("--print-record", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.ttl_seconds <= 0:
        raise SystemExit("--ttl-seconds must be positive")
    if args.require_signed and not args.record_secret:
        raise SystemExit("--require-signed requires --record-secret")
    if args.signature_max_age_seconds <= 0:
        raise SystemExit("--signature-max-age-seconds must be positive")
    if args.discovery_backend in {LIBP2P_KAD_BACKEND, LIBP2P_KAD_COMPAT_BACKEND} and args.libp2p_port < 0:
        raise SystemExit("--libp2p-port must be non-negative")
    return args


def run_libp2p_sidecar(args: argparse.Namespace) -> None:
    script = ROOT / "scripts" / "libp2p_kad_daemon.mjs"
    polyfill = ROOT / "scripts" / "libp2p_node20_polyfill.mjs"
    command = [
        "node",
    ]
    if polyfill.is_file():
        command.extend(["--import", polyfill.as_uri()])
    command.extend([
        str(script),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--public-host",
        args.public_host,
        "--swarm-id",
        args.swarm_id,
        "--role",
        args.role,
        "--ttl-seconds",
        str(args.ttl_seconds),
        "--record-secret",
        args.record_secret,
        "--signature-max-age-seconds",
        str(args.signature_max_age_seconds),
        "--discovery-backend",
        LIBP2P_KAD_BACKEND,
        "--libp2p-host",
        args.libp2p_host,
        "--libp2p-port",
        str(args.libp2p_port),
        "--libp2p-public-host",
        args.libp2p_public_host,
        "--kad-protocol",
        args.kad_protocol,
    ])
    if args.node_id:
        command.extend(["--node-id", args.node_id])
    if args.peer_url:
        command.extend(["--peer-url", args.peer_url])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.backend:
        command.extend(["--backend", args.backend])
    if args.stage_role:
        command.extend(["--stage-role", args.stage_role])
    for capability in args.stage_capability or []:
        command.extend(["--stage-capability", capability])
    for capability in args.capability or []:
        command.extend(["--capability", capability])
    for bootstrap in args.bootstrap or []:
        command.extend(["--bootstrap", bootstrap])
    if args.require_signed:
        command.append("--require-signed")
    if args.peer_key_file:
        command.extend(["--peer-key-file", args.peer_key_file])
    if args.print_record:
        command.append("--print-record")
    env = dict(os.environ)
    env.setdefault("NODE_NO_WARNINGS", "1")
    completed = subprocess.run(command, cwd=str(ROOT), env=env, text=True)
    raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()
    if args.discovery_backend in {LIBP2P_KAD_BACKEND, LIBP2P_KAD_COMPAT_BACKEND}:
        run_libp2p_sidecar(args)
    local_record = build_local_record(args)
    if args.print_record:
        print(json.dumps(local_record, sort_keys=True), flush=True)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit("uvicorn is not installed. Run: pip install -r requirements.txt") from exc
    store = ProviderRecordStore(
        swarm_id=args.swarm_id,
        ttl_seconds=args.ttl_seconds,
        record_secret=args.record_secret,
        require_signed=args.require_signed,
        signature_max_age_seconds=args.signature_max_age_seconds,
        discovery_backend=args.discovery_backend,
    )
    app = create_app(
        store=store,
        local_record=local_record,
        bootstrap_urls=list(args.bootstrap or []),
        bind_host=args.host,
        public_host=args.public_host,
        listen_port=args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
