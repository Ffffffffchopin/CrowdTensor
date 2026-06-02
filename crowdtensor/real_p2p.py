"""Replaceable real-P2P provider discovery core for CrowdTensor.

This module is the first step beyond ``p2p_lite``: it models stage providers as
signed provider records, exposes a daemon-friendly API, and keeps discovery
physically separate from the Coordinator.  The default backend is still a small
HTTP provider-record store so local and CI validation remain deterministic.
libp2p/Kademlia can replace the store behind the same record and route APIs.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .p2p_lite import (
    DEFAULT_SIGNATURE_MAX_AGE_SECONDS,
    DEFAULT_TTL_SECONDS,
    PEER_SCHEMA,
    peer_health_score,
    peer_leak_paths,
    sanitize_peer,
    sign_peer_announcement,
    stable_peer_id,
    verify_peer_announcement,
)
from .session_protocol import build_route_decision


PROVIDER_RECORD_SCHEMA = "real_p2p_provider_record_v1"
PROVIDER_CATALOG_SCHEMA = "real_p2p_provider_catalog_v1"
ROUTE_LOOKUP_SCHEMA = "real_p2p_route_lookup_v1"
NODE_SCHEMA = "real_p2p_node_v1"
HEALTH_SCHEMA = "real_p2p_health_v1"
ANNOUNCE_SCHEMA = "real_p2p_announce_v1"
DIAGNOSTICS_SCHEMA = "real_p2p_nat_relay_diagnostics_v1"
PEER_SCORING_SCHEMA = "real_p2p_peer_scoring_v1"
DEFAULT_DISCOVERY_BACKEND = "http-provider-store"
LIBP2P_KAD_BACKEND = "libp2p-kad"
LIBP2P_KAD_COMPAT_BACKEND = "libp2p-kad-adapter"
DISCOVERY_BACKENDS = {
    "http-provider-store",
    "memory-provider-store",
    LIBP2P_KAD_BACKEND,
    LIBP2P_KAD_COMPAT_BACKEND,
}


def now_epoch() -> float:
    return time.time()


def stable_node_id(seed: str | None = None) -> str:
    return stable_peer_id(seed).replace("peer-", "node-", 1)


def _record_digest(provider: dict[str, Any]) -> str:
    payload = {
        "swarm_id": provider.get("swarm_id"),
        "peer_id": provider.get("peer_id"),
        "role": provider.get("role"),
        "urls": provider.get("urls") if isinstance(provider.get("urls"), dict) else {},
        "capabilities": provider.get("capabilities") if isinstance(provider.get("capabilities"), dict) else {},
        "stage_role": provider.get("stage_role"),
        "backend": provider.get("backend"),
        "identity_hash": ((provider.get("peer_identity") or {}) if isinstance(provider.get("peer_identity"), dict) else {}).get("identity_hash"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def provider_record_id(provider: dict[str, Any]) -> str:
    return "provider-" + _record_digest(provider)[:20]


def _provider_safety() -> dict[str, Any]:
    return {
        "tokens_gossiped": False,
        "raw_prompts_gossiped": False,
        "activations_gossiped": False,
        "peer_secret_gossiped": False,
        "coordinator_backed_task_execution": True,
        "not_large_model_serving": True,
    }


def _metric_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def provider_result_metrics(provider: dict[str, Any]) -> dict[str, int]:
    caps = provider.get("capabilities") if isinstance(provider.get("capabilities"), dict) else {}
    trust = caps.get("trust") if isinstance(caps.get("trust"), dict) else {}
    metrics = caps.get("peer_metrics") if isinstance(caps.get("peer_metrics"), dict) else {}
    return {
        "accepted_result_count": _metric_int(
            caps.get("accepted_result_count", trust.get("accepted_result_count", metrics.get("accepted_result_count")))
        ),
        "failed_result_count": _metric_int(
            caps.get("failed_result_count", trust.get("failed_result_count", metrics.get("failed_result_count")))
        ),
        "stale_result_count": _metric_int(
            caps.get("stale_result_count", trust.get("stale_result_count", metrics.get("stale_result_count")))
        ),
    }


def provider_peer_scoring(provider: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    caps = provider.get("capabilities") if isinstance(provider.get("capabilities"), dict) else {}
    trust = caps.get("trust") if isinstance(caps.get("trust"), dict) else {}
    metrics = provider_result_metrics(provider)
    heartbeat_score = peer_health_score(provider, now=now)
    success_bonus = min(20, metrics["accepted_result_count"] * 2)
    failure_penalty = min(60, metrics["failed_result_count"] * 10 + metrics["stale_result_count"] * 5)
    explicit_quarantine = bool(caps.get("quarantined") or trust.get("quarantined"))
    score = max(0, min(100, heartbeat_score + success_bonus - failure_penalty))
    quarantined = bool(explicit_quarantine or score <= 0)
    return {
        "schema": PEER_SCORING_SCHEMA,
        "score": 0 if explicit_quarantine else score,
        "status": "quarantined" if quarantined else "ready",
        "quarantined": quarantined,
        "heartbeat_score": heartbeat_score,
        "accepted_result_count": metrics["accepted_result_count"],
        "failed_result_count": metrics["failed_result_count"],
        "stale_result_count": metrics["stale_result_count"],
        "route_priority": 0 if quarantined else score,
        "diagnosis_codes": (
            ["peer_quarantined", "peer_scoring_ready"]
            if quarantined
            else ["peer_scoring_ready"]
        ),
    }


def peer_sort_key(provider: dict[str, Any], *, now: float | None = None) -> tuple[int, int, str]:
    scoring = provider.get("peer_scoring") if isinstance(provider.get("peer_scoring"), dict) else provider_peer_scoring(provider, now=now)
    priority = int(scoring.get("route_priority") or 0)
    accepted = int(scoring.get("accepted_result_count") or 0)
    return (-priority, -accepted, str(provider.get("peer_id") or ""))


def discovery_boundaries(*, discovery_backend: str = DEFAULT_DISCOVERY_BACKEND) -> dict[str, Any]:
    libp2p_runtime = discovery_backend in {LIBP2P_KAD_BACKEND, LIBP2P_KAD_COMPAT_BACKEND}
    return {
        "replaceable_discovery_backend": True,
        "discovery_backend": discovery_backend,
        "provider_record_store_ready": True,
        "libp2p_runtime_ready": bool(libp2p_runtime),
        "dht_runtime_ready": bool(libp2p_runtime),
        "provider_record_transport": "libp2p-stream" if libp2p_runtime else "http-provider-store",
        "dht_provider_record_value_store_ready": False,
        "nat_traversal_ready": False,
        "relay_ready": False,
        "production_p2p_security_ready": False,
        "hivemind_petals_parity": False,
    }


def build_provider_record(
    peer: dict[str, Any],
    record_secret: str = "",
    *,
    signed_at: float | None = None,
    now: float | None = None,
    default_ttl: float = DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    current = now_epoch() if now is None else float(now)
    provider = sanitize_peer(
        {**peer, "last_seen": float(peer.get("last_seen") or current)},
        now=current,
        default_ttl=default_ttl,
    )
    if record_secret:
        provider = sanitize_peer(
            sign_peer_announcement(provider, record_secret, signed_at=current if signed_at is None else float(signed_at)),
            now=current,
            default_ttl=default_ttl,
        )
    capabilities = provider.get("capabilities") if isinstance(provider.get("capabilities"), dict) else {}
    record = {
        "schema": PROVIDER_RECORD_SCHEMA,
        "record_id": provider_record_id(provider),
        "swarm_id": provider.get("swarm_id"),
        "provider": provider,
        "role": provider.get("role"),
        "stage_role": provider.get("stage_role") or capabilities.get("real_llm_sharded_stage_role") or "",
        "stage_capabilities": list(capabilities.get("real_llm_sharded_stage_capabilities") or []),
        "backend": provider.get("backend") or capabilities.get("backend") or "",
        "peer_scoring": provider_peer_scoring(provider, now=current),
        "ttl_seconds": provider.get("ttl_seconds"),
        "last_seen": provider.get("last_seen"),
        "expires_at": provider.get("expires_at"),
        "signed_provider_record": bool(provider.get("peer_signature")),
        "signature_algorithm": ((provider.get("peer_signature") or {}) if isinstance(provider.get("peer_signature"), dict) else {}).get("algorithm", ""),
        "safety": _provider_safety(),
    }
    leaks = peer_leak_paths(record)
    if leaks:
        raise ValueError("provider record contains private fields: " + ", ".join(leaks[:5]))
    return record


def provider_from_record(record_or_peer: dict[str, Any]) -> dict[str, Any]:
    if record_or_peer.get("schema") == PROVIDER_RECORD_SCHEMA and isinstance(record_or_peer.get("provider"), dict):
        return dict(record_or_peer["provider"])
    return dict(record_or_peer)


class ProviderRecordStore:
    def __init__(
        self,
        *,
        swarm_id: str = "default",
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        record_secret: str = "",
        require_signed: bool = False,
        signature_max_age_seconds: float = DEFAULT_SIGNATURE_MAX_AGE_SECONDS,
        discovery_backend: str = DEFAULT_DISCOVERY_BACKEND,
    ) -> None:
        self.swarm_id = str(swarm_id or "default")
        self.ttl_seconds = float(ttl_seconds)
        self.record_secret = str(record_secret or "")
        self.require_signed = bool(require_signed)
        self.signature_max_age_seconds = float(signature_max_age_seconds)
        self.discovery_backend = discovery_backend if discovery_backend in DISCOVERY_BACKENDS else DEFAULT_DISCOVERY_BACKEND
        self._records: dict[str, dict[str, Any]] = {}

    def announce(self, record_or_peer: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
        current = now_epoch() if now is None else float(now)
        provider = provider_from_record(record_or_peer)
        provider = sanitize_peer(
            {**provider, "swarm_id": provider.get("swarm_id") or self.swarm_id},
            now=current,
            default_ttl=self.ttl_seconds,
        )
        if str(provider.get("swarm_id") or "") != self.swarm_id:
            raise ValueError(f"provider record swarm mismatch: expected {self.swarm_id}")
        verification: dict[str, Any] = {"ok": False, "diagnosis_code": "real_p2p_unsigned_provider_record"}
        if self.record_secret:
            verification = verify_peer_announcement(
                provider,
                self.record_secret,
                now=current,
                max_age_seconds=self.signature_max_age_seconds,
            )
        if self.require_signed and not verification.get("ok"):
            raise ValueError(f"signed provider record required: {verification.get('diagnosis_code')}")
        provider["identity_verified"] = bool(verification.get("ok"))
        provider["signature_verification"] = {
            key: value
            for key, value in verification.items()
            if key in {"ok", "schema", "diagnosis_code", "identity_hash", "signed_at"}
        }
        provider["health_score"] = peer_health_score(provider, now=current)
        provider["health_status"] = "ready" if provider["health_score"] > 0 else "expired"
        provider["peer_scoring"] = provider_peer_scoring(provider, now=current)
        provider["trust_score"] = int(provider["peer_scoring"].get("score") or 0)
        provider["trust_status"] = str(provider["peer_scoring"].get("status") or "")
        record = build_provider_record(provider, now=current, default_ttl=self.ttl_seconds)
        store_ttl = min(float(record.get("ttl_seconds") or self.ttl_seconds), max(1.0, self.ttl_seconds))
        record["ttl_seconds"] = store_ttl
        record["expires_at"] = current + store_ttl
        record["provider"] = provider
        record["record_id"] = provider_record_id(provider)
        record["identity_verified"] = bool(provider.get("identity_verified"))
        record["signature_verification"] = provider.get("signature_verification")
        record["health_score"] = provider.get("health_score")
        record["health_status"] = provider.get("health_status")
        record["peer_scoring"] = provider.get("peer_scoring")
        record["trust_score"] = provider.get("trust_score")
        record["trust_status"] = provider.get("trust_status")
        existing = self._records.get(str(provider.get("peer_id") or ""))
        if existing:
            existing_provider = existing.get("provider") if isinstance(existing.get("provider"), dict) else {}
            if float(existing_provider.get("last_seen") or 0.0) > float(provider.get("last_seen") or 0.0):
                return dict(existing)
        self._records[str(provider.get("peer_id"))] = record
        return dict(record)

    def merge(self, records_or_peers: list[dict[str, Any]], *, now: float | None = None) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for item in records_or_peers:
            if not isinstance(item, dict):
                continue
            try:
                merged.append(self.announce(item, now=now))
            except ValueError:
                continue
        self.prune(now=now)
        return merged

    def prune(self, *, now: float | None = None) -> int:
        current = now_epoch() if now is None else float(now)
        expired = [
            peer_id
            for peer_id, record in self._records.items()
            if float(record.get("expires_at") or 0.0) <= current
        ]
        for peer_id in expired:
            self._records.pop(peer_id, None)
        return len(expired)

    def provider_records(self, *, now: float | None = None) -> list[dict[str, Any]]:
        self.prune(now=now)
        return sorted((dict(record) for record in self._records.values()), key=lambda item: str(item.get("record_id")))

    def peers(self, *, now: float | None = None) -> list[dict[str, Any]]:
        peers: list[dict[str, Any]] = []
        for record in self.provider_records(now=now):
            provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
            if provider:
                peers.append(dict(provider))
        return sorted(peers, key=lambda item: str(item.get("peer_id")))

    def ranked_peers(self, *, now: float | None = None) -> list[dict[str, Any]]:
        peers = []
        for peer in self.peers(now=now):
            scored = dict(peer)
            scored["peer_scoring"] = provider_peer_scoring(scored, now=now)
            scored["trust_score"] = int(scored["peer_scoring"].get("score") or 0)
            scored["trust_status"] = str(scored["peer_scoring"].get("status") or "")
            peers.append(scored)
        return sorted(peers, key=lambda item: peer_sort_key(item, now=now))

    def peer_scoring_payload(self, *, now: float | None = None) -> dict[str, Any]:
        ranked = self.ranked_peers(now=now)
        quarantined = [peer for peer in ranked if ((peer.get("peer_scoring") or {}) if isinstance(peer.get("peer_scoring"), dict) else {}).get("quarantined")]
        return {
            "schema": PEER_SCORING_SCHEMA,
            "ok": True,
            "enabled": True,
            "peer_count": len(ranked),
            "quarantined_peer_count": len(quarantined),
            "ranked_peer_ids": [str(peer.get("peer_id") or "") for peer in ranked],
            "scores": {
                str(peer.get("peer_id") or ""): {
                    "score": int(((peer.get("peer_scoring") or {}) if isinstance(peer.get("peer_scoring"), dict) else {}).get("score") or 0),
                    "status": str(((peer.get("peer_scoring") or {}) if isinstance(peer.get("peer_scoring"), dict) else {}).get("status") or ""),
                    "accepted_result_count": int(((peer.get("peer_scoring") or {}) if isinstance(peer.get("peer_scoring"), dict) else {}).get("accepted_result_count") or 0),
                    "failed_result_count": int(((peer.get("peer_scoring") or {}) if isinstance(peer.get("peer_scoring"), dict) else {}).get("failed_result_count") or 0),
                    "stale_result_count": int(((peer.get("peer_scoring") or {}) if isinstance(peer.get("peer_scoring"), dict) else {}).get("stale_result_count") or 0),
                }
                for peer in ranked
            },
            "route_prefers_higher_score": True,
            "diagnosis_codes": ["peer_scoring_ready"],
        }

    def catalog_payload(self, *, now: float | None = None) -> dict[str, Any]:
        providers = self.provider_records(now=now)
        peers = self.peers(now=now)
        scoring = self.peer_scoring_payload(now=now)
        signed_count = sum(1 for record in providers if record.get("identity_verified") is True)
        healthy_count = sum(1 for record in providers if int(record.get("health_score") or 0) > 0)
        return {
            "schema": PROVIDER_CATALOG_SCHEMA,
            "ok": True,
            "swarm_id": self.swarm_id,
            "provider_count": len(providers),
            "peer_count": len(peers),
            "providers": providers,
            "peers": peers,
            "registry": {
                "signed_provider_record_required": self.require_signed,
                "signed_provider_record_count": signed_count,
                "healthy_provider_count": healthy_count,
                "peer_scoring_ready": True,
                "swarm_isolation_ready": True,
                "ttl_seconds": self.ttl_seconds,
                "discovery_backend": self.discovery_backend,
            },
            "peer_scoring": scoring,
            "diagnosis_codes": [
                "real_p2p_provider_store_ready",
                "real_p2p_swarm_isolation_ready",
                "replaceable_discovery_backend_ready",
                "peer_scoring_ready",
            ],
            "safety": _provider_safety(),
            "boundaries": discovery_boundaries(discovery_backend=self.discovery_backend),
        }

    def route_lookup(
        self,
        session_request: dict[str, Any],
        *,
        coordinator_url: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        route_peers = [
            peer for peer in self.ranked_peers(now=now)
            if not (((peer.get("peer_scoring") or {}) if isinstance(peer.get("peer_scoring"), dict) else {}).get("quarantined"))
        ]
        route = build_route_decision(
            session_request,
            coordinator_url=coordinator_url,
            peer_catalog=route_peers,
        )
        codes = list(route.get("diagnosis_codes") or [])
        codes.append("real_p2p_swarm_isolation_ready")
        if route.get("usable_now"):
            codes.append("real_p2p_route_lookup_ready")
        else:
            codes.append("real_p2p_route_lookup_blocked")
        return {
            "schema": ROUTE_LOOKUP_SCHEMA,
            "ok": bool(route.get("usable_now")),
            "swarm_id": self.swarm_id,
            "route": route,
            "provider_count": len(self.provider_records(now=now)),
            "diagnosis_codes": sorted(set(codes)),
            "boundaries": discovery_boundaries(discovery_backend=self.discovery_backend),
        }


def nat_relay_diagnostics(
    *,
    bind_host: str = "127.0.0.1",
    public_host: str = "",
    listen_port: int = 0,
    bootstrap_urls: list[str] | None = None,
    discovery_backend: str = DEFAULT_DISCOVERY_BACKEND,
) -> dict[str, Any]:
    public_bind = bind_host in {"0.0.0.0", "::"}
    codes = ["real_p2p_nat_relay_diagnostics_ready"]
    if public_bind and public_host:
        codes.append("manual_public_bind_advertised")
    else:
        codes.append("manual_public_bind_required")
    if bootstrap_urls:
        codes.append("bootstrap_peer_configured")
    else:
        codes.append("bootstrap_peer_missing")
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "ok": True,
        "bind_host": bind_host,
        "public_host": public_host,
        "listen_port": int(listen_port or 0),
        "bootstrap_count": len(bootstrap_urls or []),
        "discovery_backend": discovery_backend,
        "nat_traversal_ready": False,
        "relay_ready": False,
        "operator_action": (
            "Expose the daemon through a trusted public endpoint, VPN, tunnel, or future relay backend; "
            "automatic NAT traversal is not enabled in this RC."
        ),
        "diagnosis_codes": codes,
        "boundaries": discovery_boundaries(discovery_backend=discovery_backend),
    }


def fetch_provider_catalog(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(f"{str(url).rstrip('/')}/real-p2p/providers", method="GET")
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def post_provider_record(url: str, record_or_peer: dict[str, Any], *, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(
        f"{str(url).rstrip('/')}/real-p2p/announce",
        data=json.dumps(record_or_peer).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def post_route_lookup(
    url: str,
    session_request: dict[str, Any],
    *,
    coordinator_url: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    request = Request(
        f"{str(url).rstrip('/')}/real-p2p/route",
        data=json.dumps({"session_request": session_request, "coordinator_url": coordinator_url}).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def bootstrap_once(
    store: ProviderRecordStore,
    bootstrap_urls: list[str],
    *,
    local_record: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for url in bootstrap_urls:
        try:
            if local_record:
                post_provider_record(url, local_record, timeout=timeout)
            payload = fetch_provider_catalog(url, timeout=timeout)
            incoming = payload.get("providers") if isinstance(payload.get("providers"), list) else payload.get("peers")
            incoming_items = [item for item in incoming or [] if isinstance(item, dict)]
            merged: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            for item in incoming_items:
                try:
                    merged.append(store.announce(item))
                except ValueError as exc:
                    rejected.append({
                        "peer_id": str((provider_from_record(item).get("peer_id") if isinstance(item, dict) else "") or ""),
                        "error": str(exc)[:200],
                    })
            store.prune()
            results.append({
                "url": url,
                "ok": True,
                "incoming_provider_count": len(incoming_items),
                "merged_provider_count": len(merged),
                "rejected_provider_count": len(rejected),
                "rejected": rejected[:5],
            })
        except (OSError, URLError, json.JSONDecodeError, ValueError) as exc:
            results.append({"url": url, "ok": False, "error": type(exc).__name__, "detail": str(exc)[:200]})
    success_count = sum(1 for item in results if item.get("ok"))
    merged_total = sum(int(item.get("merged_provider_count") or 0) for item in results)
    rejected_total = sum(int(item.get("rejected_provider_count") or 0) for item in results)
    codes = ["real_p2p_bootstrap_sync_not_configured"] if not bootstrap_urls else []
    if bootstrap_urls:
        codes.append("real_p2p_bootstrap_sync_ready" if success_count == len(bootstrap_urls) else "real_p2p_bootstrap_sync_partial")
    if merged_total:
        codes.append("real_p2p_bootstrap_records_merged")
    if rejected_total:
        codes.extend(["real_p2p_bootstrap_records_rejected", "real_p2p_swarm_isolation_ready"])
    return {
        "schema": "real_p2p_bootstrap_once_v1",
        "ok": all(item.get("ok") for item in results) if results else True,
        "bootstrap_count": len(bootstrap_urls),
        "success_count": success_count,
        "merged_provider_count": merged_total,
        "rejected_provider_count": rejected_total,
        "results": results,
        "diagnosis_codes": codes,
    }


def create_app(
    *,
    store: ProviderRecordStore | None = None,
    local_record: dict[str, Any] | None = None,
    bootstrap_urls: list[str] | None = None,
    bind_host: str = "127.0.0.1",
    public_host: str = "",
    listen_port: int = 0,
):
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Run: pip install -r requirements.txt") from exc

    provider_store = store or ProviderRecordStore()
    if local_record:
        provider_store.announce(local_record)
    bootstraps = list(bootstrap_urls or [])

    app = FastAPI(title="CrowdTensor real-P2P provider discovery", version="0.1.0a0")
    app.state.provider_store = provider_store

    @app.get("/real-p2p/health")
    def health() -> dict[str, Any]:
        payload = provider_store.catalog_payload()
        return {
            "ok": True,
            "schema": HEALTH_SCHEMA,
            "swarm_id": provider_store.swarm_id,
            "provider_count": payload.get("provider_count"),
            "registry": payload.get("registry"),
            "boundaries": payload.get("boundaries"),
        }

    @app.post("/real-p2p/announce")
    def announce(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            record = provider_store.announce(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
        return {"ok": True, "schema": ANNOUNCE_SCHEMA, "record": record, "peer": provider}

    @app.get("/real-p2p/providers")
    def providers() -> dict[str, Any]:
        if bootstraps:
            bootstrap_once(provider_store, bootstraps, local_record=local_record)
        return provider_store.catalog_payload()

    @app.post("/real-p2p/route")
    def route(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        bootstrap_sync: dict[str, Any] = {}
        if bootstraps:
            bootstrap_sync = bootstrap_once(provider_store, bootstraps, local_record=local_record)
        session_request = request.get("session_request") if isinstance(request.get("session_request"), dict) else {}
        payload = provider_store.route_lookup(session_request, coordinator_url=str(request.get("coordinator_url") or ""))
        if bootstrap_sync:
            payload["bootstrap_sync"] = bootstrap_sync
            route_codes = set(payload.get("diagnosis_codes") or [])
            route_codes.update(code for code in bootstrap_sync.get("diagnosis_codes") or [] if isinstance(code, str))
            route_codes.add(
                "real_p2p_route_bootstrap_sync_ready"
                if bootstrap_sync.get("ok")
                else "real_p2p_route_bootstrap_sync_failed"
            )
            payload["diagnosis_codes"] = sorted(route_codes)
        return payload

    @app.get("/real-p2p/diagnostics")
    def diagnostics() -> dict[str, Any]:
        return nat_relay_diagnostics(
            bind_host=bind_host,
            public_host=public_host,
            listen_port=listen_port,
            bootstrap_urls=bootstraps,
            discovery_backend=provider_store.discovery_backend,
        )

    @app.get("/peer/catalog")
    def legacy_catalog() -> dict[str, Any]:
        payload = provider_store.catalog_payload()
        return {
            "schema": "p2p_lite_catalog_v1",
            "ok": True,
            "swarm_id": payload.get("swarm_id"),
            "peer_count": payload.get("peer_count"),
            "peers": payload.get("peers") or [],
            "registry": payload.get("registry") or {},
            "safety": payload.get("safety") or {},
            "compatibility": {"served_by": PROVIDER_CATALOG_SCHEMA},
        }

    @app.post("/peer/announce")
    def legacy_announce(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            record = provider_store.announce(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        provider = record.get("provider") if isinstance(record.get("provider"), dict) else {}
        return {"ok": True, "schema": "p2p_lite_announce_v1", "peer": provider}

    return app
