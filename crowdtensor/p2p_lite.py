"""P2P-lite peer catalog for CrowdTensor public RC routing.

This is intentionally not libp2p, not a DHT, and not NAT traversal.  It is a
small HTTP-gossip discovery layer that removes hard-coded Coordinator URLs from
operator runbooks while leaving task leasing and execution authority in the
Coordinator.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .session_protocol import build_route_decision, public_leak_paths


PEER_SCHEMA = "p2p_lite_peer_v1"
CATALOG_SCHEMA = "p2p_lite_catalog_v1"
RESOLVE_SCHEMA = "p2p_lite_resolve_v1"
DEFAULT_TTL_SECONDS = 60.0
SECRET_KEYS = {
    "admin_token",
    "miner_token",
    "observer_token",
    "token",
    "api_key",
    "authorization",
    "lease_token",
    "idempotency_key",
    "prompt",
    "prompt_text",
    "generated_text",
    "generated_token_ids",
    "activation_results",
}


def now_epoch() -> float:
    return time.time()


def stable_peer_id(seed: str | None = None) -> str:
    if seed:
        return "peer-" + hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:16]
    return "peer-" + uuid.uuid4().hex[:16]


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def sanitize_peer(value: dict[str, Any], *, now: float | None = None, default_ttl: float = DEFAULT_TTL_SECONDS) -> dict[str, Any]:
    current = now_epoch() if now is None else float(now)
    peer_id = str(value.get("peer_id") or stable_peer_id(json.dumps(value, sort_keys=True, default=str))).strip()
    role = str(value.get("role") or "observer").strip().lower()
    if role not in {"coordinator", "miner", "observer"}:
        role = "observer"
    ttl = float(value.get("ttl_seconds") or default_ttl)
    ttl = max(1.0, min(ttl, 3600.0))
    urls = _safe_dict(value.get("urls"))
    capabilities = _safe_dict(value.get("capabilities"))
    peer = {
        "schema": PEER_SCHEMA,
        "swarm_id": str(value.get("swarm_id") or "default"),
        "peer_id": peer_id,
        "role": role,
        "urls": {
            key: str(item)
            for key, item in sorted(urls.items())
            if key in {"coordinator", "peer", "metrics", "health"} and str(item)
        },
        "capabilities": capabilities,
        "stage_role": str(value.get("stage_role") or capabilities.get("real_llm_sharded_stage_role") or ""),
        "backend": str(value.get("backend") or capabilities.get("backend") or ""),
        "ttl_seconds": ttl,
        "last_seen": float(value.get("last_seen") or current),
    }
    peer["expires_at"] = peer["last_seen"] + ttl
    leaks = peer_leak_paths(peer)
    if leaks:
        raise ValueError("peer metadata contains private fields: " + ", ".join(leaks[:5]))
    return peer


def peer_leak_paths(value: Any, *, path: str = "$") -> list[str]:
    leaks = public_leak_paths(value, path=path)
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            child_path = f"{path}.{key}"
            safe_boolean = key_text in {
                "tokens_gossiped",
                "raw_prompts_gossiped",
                "activations_gossiped",
            } and isinstance(item, bool)
            if not safe_boolean and (key_text in SECRET_KEYS or "token" in key_text or "secret" in key_text):
                leaks.append(child_path)
            leaks.extend(peer_leak_paths(item, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaks.extend(peer_leak_paths(item, path=f"{path}[{index}]"))
    return sorted(set(leaks))


class PeerCatalog:
    def __init__(self, *, swarm_id: str = "default", ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.swarm_id = str(swarm_id or "default")
        self.ttl_seconds = float(ttl_seconds)
        self._peers: dict[str, dict[str, Any]] = {}

    def announce(self, peer: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
        sanitized = sanitize_peer(
            {**peer, "swarm_id": peer.get("swarm_id") or self.swarm_id},
            now=now,
            default_ttl=self.ttl_seconds,
        )
        existing = self._peers.get(sanitized["peer_id"])
        if existing and float(existing.get("last_seen") or 0.0) > float(sanitized.get("last_seen") or 0.0):
            return dict(existing)
        self._peers[sanitized["peer_id"]] = sanitized
        return dict(sanitized)

    def merge(self, peers: list[dict[str, Any]], *, now: float | None = None) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for peer in peers:
            if not isinstance(peer, dict):
                continue
            try:
                merged.append(self.announce(peer, now=now))
            except ValueError:
                continue
        self.prune(now=now)
        return merged

    def prune(self, *, now: float | None = None) -> int:
        current = now_epoch() if now is None else float(now)
        expired = [
            peer_id for peer_id, peer in self._peers.items()
            if float(peer.get("expires_at") or 0.0) <= current
        ]
        for peer_id in expired:
            self._peers.pop(peer_id, None)
        return len(expired)

    def peers(self, *, now: float | None = None) -> list[dict[str, Any]]:
        self.prune(now=now)
        return sorted((dict(peer) for peer in self._peers.values()), key=lambda item: str(item.get("peer_id")))

    def catalog_payload(self, *, now: float | None = None) -> dict[str, Any]:
        peers = self.peers(now=now)
        return {
            "schema": CATALOG_SCHEMA,
            "ok": True,
            "swarm_id": self.swarm_id,
            "peer_count": len(peers),
            "peers": peers,
            "safety": {
                "tokens_gossiped": False,
                "raw_prompts_gossiped": False,
                "activations_gossiped": False,
                "not_libp2p": True,
                "not_dht": True,
                "not_nat_traversal": True,
            },
        }

    def resolve(self, session_request: dict[str, Any], *, coordinator_url: str = "", now: float | None = None) -> dict[str, Any]:
        route = build_route_decision(
            session_request,
            coordinator_url=coordinator_url,
            peer_catalog=self.peers(now=now),
        )
        return {
            "schema": RESOLVE_SCHEMA,
            "ok": bool(route.get("usable_now")),
            "swarm_id": self.swarm_id,
            "route": route,
            "diagnosis_codes": list(route.get("diagnosis_codes") or []),
        }


def load_catalog_file(path: str | Path) -> PeerCatalog:
    payload = {}
    file_path = Path(path)
    if file_path.is_file():
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    catalog = PeerCatalog(swarm_id=str(payload.get("swarm_id") or "default"))
    peers = payload.get("peers") if isinstance(payload, dict) else []
    if isinstance(peers, list):
        catalog.merge([peer for peer in peers if isinstance(peer, dict)])
    return catalog


def write_catalog_file(path: str | Path, catalog: PeerCatalog) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(catalog.catalog_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_peer_catalog(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(f"{str(url).rstrip('/')}/peer/catalog", method="GET")
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def post_announce(url: str, peer: dict[str, Any], *, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(
        f"{str(url).rstrip('/')}/peer/announce",
        data=json.dumps(peer).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def gossip_once(catalog: PeerCatalog, bootstrap_urls: list[str], *, local_peer: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for url in bootstrap_urls:
        try:
            if local_peer:
                post_announce(url, local_peer, timeout=timeout)
            payload = fetch_peer_catalog(url, timeout=timeout)
            peers = payload.get("peers") if isinstance(payload, dict) else []
            merged = catalog.merge([peer for peer in peers if isinstance(peer, dict)])
            results.append({"url": url, "ok": True, "merged_peer_count": len(merged)})
        except (OSError, URLError, json.JSONDecodeError, ValueError) as exc:
            results.append({"url": url, "ok": False, "error": type(exc).__name__, "detail": str(exc)[:200]})
    return {
        "schema": "p2p_lite_gossip_once_v1",
        "ok": all(item.get("ok") for item in results) if results else True,
        "bootstrap_count": len(bootstrap_urls),
        "results": results,
    }


def create_app(*, catalog: PeerCatalog | None = None, local_peer: dict[str, Any] | None = None, bootstrap_urls: list[str] | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Run: pip install -r requirements.txt") from exc

    peer_catalog = catalog or PeerCatalog()
    if local_peer:
        peer_catalog.announce(local_peer)
    bootstraps = list(bootstrap_urls or [])

    class AnnounceRequest(BaseModel):
        schema: str = PEER_SCHEMA
        swarm_id: str = "default"
        peer_id: str = ""
        role: str = "observer"
        urls: dict[str, Any] = {}
        capabilities: dict[str, Any] = {}
        stage_role: str = ""
        backend: str = ""
        ttl_seconds: float = DEFAULT_TTL_SECONDS
        last_seen: float | None = None

    class ResolveRequest(BaseModel):
        session_request: dict[str, Any]
        coordinator_url: str = ""

    app = FastAPI(title="CrowdTensor P2P-lite discovery", version="0.1.0a0")
    app.state.catalog = peer_catalog

    @app.get("/peer/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "schema": "p2p_lite_health_v1", "swarm_id": peer_catalog.swarm_id}

    @app.post("/peer/announce")
    def announce(request: AnnounceRequest) -> dict[str, Any]:
        try:
            request_payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
            peer = peer_catalog.announce(request_payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True, "schema": "p2p_lite_announce_v1", "peer": peer}

    @app.get("/peer/catalog")
    def catalog_payload() -> dict[str, Any]:
        if bootstraps:
            gossip_once(peer_catalog, bootstraps, local_peer=local_peer)
        return peer_catalog.catalog_payload()

    @app.post("/peer/resolve")
    def resolve(request: ResolveRequest) -> dict[str, Any]:
        return peer_catalog.resolve(request.session_request, coordinator_url=request.coordinator_url)

    return app
