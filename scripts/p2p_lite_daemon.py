#!/usr/bin/env python3
"""Run the CrowdTensor P2P-lite discovery daemon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.p2p_lite import PeerCatalog, create_app, stable_peer_id  # noqa: E402


def parse_capability(value: str) -> tuple[str, str]:
    if "=" not in value:
        return value, "true"
    key, item = value.split("=", 1)
    return key.strip(), item.strip()


def build_local_peer(args: argparse.Namespace) -> dict:
    capabilities = {}
    for item in args.capability or []:
        key, value = parse_capability(item)
        if key:
            if "," in value:
                capabilities[key] = [part.strip() for part in value.split(",") if part.strip()]
            else:
                capabilities[key] = value
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
        "peer_id": args.peer_id or stable_peer_id(f"{args.swarm_id}:{args.role}:{args.peer_url}:{args.coordinator_url}"),
        "role": args.role,
        "urls": urls,
        "capabilities": capabilities,
        "stage_role": args.stage_role,
        "backend": args.backend,
        "ttl_seconds": args.ttl_seconds,
    }
    return peer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CrowdTensor P2P-lite discovery daemon.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--swarm-id", default="default")
    parser.add_argument("--peer-id", default="")
    parser.add_argument("--role", choices=["coordinator", "miner", "observer"], default="observer")
    parser.add_argument("--peer-url", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--backend", choices=["", "cpu", "cuda"], default="")
    parser.add_argument("--stage-role", choices=["", "stage0", "stage1", "both"], default="")
    parser.add_argument("--stage-capability", action="append", default=[])
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--bootstrap", action="append", default=[])
    parser.add_argument("--ttl-seconds", type=float, default=60.0)
    parser.add_argument("--print-peer", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.ttl_seconds <= 0:
        raise SystemExit("--ttl-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    local_peer = build_local_peer(args)
    if args.print_peer:
        print(json.dumps(local_peer, sort_keys=True), flush=True)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit("uvicorn is not installed. Run: pip install -r requirements.txt") from exc
    catalog = PeerCatalog(swarm_id=args.swarm_id, ttl_seconds=args.ttl_seconds)
    app = create_app(catalog=catalog, local_peer=local_peer, bootstrap_urls=list(args.bootstrap or []))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
