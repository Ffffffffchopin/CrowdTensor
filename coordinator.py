#!/usr/bin/env python3
"""FastAPI Coordinator for CrowdTensorD Phase 1."""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from crowdtensor.auth import token_matches, validate_token_verifier
from crowdtensor.protocol import (
    DEFAULT_PROTOCOL_VERSION,
    DEFAULT_WORKLOAD_TYPE,
    LeaseConflict,
    NoTaskAvailable,
    ResultRejected,
)
from crowdtensor.outer_optimizer import (
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    OPTIMIZER_DILOCO_MOMENTUM,
    SUPPORTED_DELTA_FORMATS,
    SUPPORTED_OUTER_OPTIMIZERS,
)
from crowdtensor.state_store import StateStore


SERVICE_NAME = "crowdtensord-coordinator"
SERVICE_VERSION = "0.1.0a0"
API_STATUS = "alpha"


def load_miner_token_registry(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load a minimal per-miner token registry from JSON."""
    if not path:
        return {}
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid miner token registry JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"could not read miner token registry: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("miner token registry must be a JSON object")
    miners = payload.get("miners")
    if not isinstance(miners, list):
        raise ValueError("miner token registry must contain a miners list")

    registry: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(miners):
        if not isinstance(entry, dict):
            raise ValueError(f"miner token registry entry {index} must be an object")
        miner_id = str(entry.get("miner_id", "")).strip()
        token = str(entry.get("token", "")).strip()
        if not miner_id:
            raise ValueError(f"miner token registry entry {index} missing miner_id")
        if not token:
            raise ValueError(f"miner token registry entry {index} missing token")
        token = validate_token_verifier(token, field_name=f"miner token registry entry {index} token")
        if miner_id in registry:
            raise ValueError(f"duplicate miner token registry miner_id: {miner_id}")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"miner token registry entry {index} enabled must be boolean")
        label = entry.get("label", "")
        registry[miner_id] = {
            "token": token,
            "enabled": enabled,
            "label": str(label or ""),
        }
    return registry


def _metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return str(numeric)


def _metric_label(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _nested_workload_count(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    total = 0
    for workloads in value.values():
        if isinstance(workloads, dict):
            total += len(workloads)
        elif isinstance(workloads, list):
            total += len(workloads)
    return total


def _trust_override_mode_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts = {"allow": 0, "block": 0}
    overrides = summary.get("miner_trust_overrides") or {}
    if not isinstance(overrides, dict):
        return counts
    for workload_overrides in overrides.values():
        if not isinstance(workload_overrides, dict):
            continue
        for override in workload_overrides.values():
            if not isinstance(override, dict):
                continue
            mode = str(override.get("mode", ""))
            if mode in counts:
                counts[mode] += 1
    return counts


def build_metrics_text(summary: dict[str, Any]) -> str:
    """Render safe aggregate Coordinator metrics in Prometheus text format."""
    lines: list[str] = []

    def add(name: str, value: Any, labels: dict[str, Any] | None = None) -> None:
        if labels:
            label_text = ",".join(
                f'{key}="{_metric_label(label_value)}"'
                for key, label_value in sorted(labels.items())
            )
            lines.append(f"{name}{{{label_text}}} {_metric_value(value)}")
        else:
            lines.append(f"{name} {_metric_value(value)}")

    lines.extend([
        "# HELP crowdtensord_event_index Last applied append-only event index.",
        "# TYPE crowdtensord_event_index gauge",
    ])
    add("crowdtensord_event_index", summary.get("event_index", 0))

    model = summary.get("model") or {}
    lines.extend([
        "# HELP crowdtensord_model_step Current model step counters.",
        "# TYPE crowdtensord_model_step gauge",
    ])
    add("crowdtensord_model_step", model.get("global_step", 0), {"step": "global"})
    add("crowdtensord_model_step", model.get("optimizer_step", 0), {"step": "optimizer"})
    add("crowdtensord_model_step", model.get("adapter_step", 0), {"step": "adapter"})
    micro_transformer = model.get("micro_transformer") or {}
    if isinstance(micro_transformer, dict):
        add(
            "crowdtensord_model_step",
            micro_transformer.get("optimizer_step", 0),
            {"step": "micro_transformer"},
        )
    model_bundle = model.get("model_bundle") or {}
    if isinstance(model_bundle, dict):
        add(
            "crowdtensord_model_step",
            model_bundle.get("optimizer_step", 0),
            {"step": "model_bundle"},
        )

    lines.extend([
        "# HELP crowdtensord_task_count Current task count by status.",
        "# TYPE crowdtensord_task_count gauge",
    ])
    task_counts = summary.get("task_counts") or {}
    if isinstance(task_counts, dict):
        for status in ("queued", "leased", "completed", "rejected"):
            add("crowdtensord_task_count", task_counts.get(status, 0), {"status": status})

    lines.extend([
        "# HELP crowdtensord_results_total Accepted and rejected result counters.",
        "# TYPE crowdtensord_results_total counter",
    ])
    add("crowdtensord_results_total", summary.get("accepted_results", 0), {"result": "accepted"})
    add("crowdtensord_results_total", summary.get("rejected_results", 0), {"result": "rejected"})

    lines.extend([
        "# HELP crowdtensord_model_updates_total Applied model update counters.",
        "# TYPE crowdtensord_model_updates_total counter",
    ])
    add("crowdtensord_model_updates_total", summary.get("model_updates", 0), {"target": "dense"})
    add("crowdtensord_model_updates_total", summary.get("adapter_updates", 0), {"target": "adapter"})
    add(
        "crowdtensord_model_updates_total",
        summary.get("micro_transformer_updates", 0),
        {"target": "micro_transformer"},
    )
    add(
        "crowdtensord_model_updates_total",
        summary.get("model_bundle_updates", 0),
        {"target": "model_bundle"},
    )

    lines.extend([
        "# HELP crowdtensord_audit_results_total Replay audit counters.",
        "# TYPE crowdtensord_audit_results_total counter",
    ])
    add("crowdtensord_audit_results_total", summary.get("audit_results", 0), {"result": "total"})
    add("crowdtensord_audit_results_total", summary.get("audit_rejections", 0), {"result": "rejected"})

    lines.extend([
        "# HELP crowdtensord_claims_total Claim routing counters.",
        "# TYPE crowdtensord_claims_total counter",
    ])
    add("crowdtensord_claims_total", summary.get("blocked_claims", 0), {"result": "blocked"})
    add("crowdtensord_claims_total", summary.get("incompatible_claims", 0), {"result": "incompatible"})

    lines.extend([
        "# HELP crowdtensord_miner_workload_blocks Current blocked miner/workload pairs.",
        "# TYPE crowdtensord_miner_workload_blocks gauge",
    ])
    add("crowdtensord_miner_workload_blocks", _nested_workload_count(summary.get("quarantined_miners")), {"source": "auto"})
    add(
        "crowdtensord_miner_workload_blocks",
        _nested_workload_count(summary.get("effective_quarantined_miners")),
        {"source": "effective"},
    )
    add("crowdtensord_miner_workload_blocks", _nested_workload_count(summary.get("manual_blocked_miners")), {"source": "manual"})

    lines.extend([
        "# HELP crowdtensord_trust_overrides Current manual trust override count by mode.",
        "# TYPE crowdtensord_trust_overrides gauge",
    ])
    for mode, count in _trust_override_mode_counts(summary).items():
        add("crowdtensord_trust_overrides", count, {"mode": mode})

    lines.extend([
        "# HELP crowdtensord_staleness Current accepted-result staleness summary.",
        "# TYPE crowdtensord_staleness gauge",
    ])
    add("crowdtensord_staleness", summary.get("max_staleness", 0), {"summary": "max"})
    add("crowdtensord_staleness", summary.get("avg_staleness", 0.0), {"summary": "avg"})

    return "\n".join(lines) + "\n"


def version_payload() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "default_workload_type": DEFAULT_WORKLOAD_TYPE,
        "api_status": API_STATUS,
    }


def readiness_payload(
    summary: dict[str, Any],
    *,
    miner_required: bool,
    observer_required: bool,
    admin_configured: bool,
    miner_registry_configured: bool,
) -> dict[str, Any]:
    return {
        "ok": True,
        **version_payload(),
        "event_index": summary.get("event_index", 0),
        "task_counts": dict(summary.get("task_counts") or {}),
        "task_lanes": [dict(lane) for lane in summary.get("task_lanes") or []],
        "auth": {
            "miner_required": bool(miner_required),
            "observer_required": bool(observer_required),
            "admin_configured": bool(admin_configured),
            "miner_registry_configured": bool(miner_registry_configured),
        },
    }


def create_app(
    *,
    state_dir: str | Path = "state",
    lease_seconds: float = 15.0,
    inner_steps: int = 500,
    backlog: int = 1,
    task_lanes: list[dict[str, Any]] | None = None,
    reaper_interval: float = 1.0,
    cors_origins: list[str] | None = None,
    replay_audit: bool = False,
    outer_optimizer: str = OPTIMIZER_DILOCO_MOMENTUM,
    delta_format: str = "dense_float",
    admin_token: str | None = None,
    miner_token: str | None = None,
    miner_token_registry: str | Path | None = None,
    observer_token: str | None = None,
):
    try:
        from fastapi import FastAPI, Header, HTTPException, Query, Response
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI is not installed. Run: pip install -r requirements.txt") from exc

    store = StateStore(
        state_dir,
        lease_seconds=lease_seconds,
        inner_steps=inner_steps,
        backlog=backlog,
        task_lanes=task_lanes,
        replay_audit=replay_audit,
        outer_optimizer=outer_optimizer,
        delta_format=delta_format,
    )
    configured_admin_token = admin_token if admin_token is not None else os.environ.get("CROWDTENSOR_ADMIN_TOKEN", "")
    configured_miner_token = miner_token if miner_token is not None else os.environ.get("CROWDTENSOR_MINER_TOKEN", "")
    configured_miner_registry_path = (
        miner_token_registry
        if miner_token_registry is not None
        else os.environ.get("CROWDTENSOR_MINER_TOKEN_REGISTRY", "")
    )
    configured_miner_registry = load_miner_token_registry(configured_miner_registry_path)
    configured_observer_token = observer_token if observer_token is not None else os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", "")
    if configured_admin_token:
        configured_admin_token = validate_token_verifier(configured_admin_token, field_name="admin token")
    if configured_miner_token:
        configured_miner_token = validate_token_verifier(configured_miner_token, field_name="miner token")
    if configured_observer_token:
        configured_observer_token = validate_token_verifier(configured_observer_token, field_name="observer token")

    class ClaimRequest(BaseModel):
        miner_id: str = Field(default="anonymous", min_length=1)
        capabilities: dict[str, Any] = Field(default_factory=dict)

    class LeaseRequest(BaseModel):
        lease_token: str = Field(min_length=1)
        attempt: int = Field(ge=1)
        runtime_status: dict[str, Any] = Field(default_factory=dict)

    class ResultRequest(LeaseRequest):
        idempotency_key: str | None = None
        local_delta: list[float] | None = None
        pseudo_gradient: list[float] | None = None
        compressed_delta: dict[str, Any] | None = None
        probe_result: dict[str, Any] | None = None
        adapter_delta: dict[str, Any] | None = None
        bundle_delta: dict[str, Any] | None = None
        inference_result: dict[str, Any] | None = None
        inference_results: list[dict[str, Any]] | None = None
        external_llm_result: dict[str, Any] | None = None
        external_llm_results: list[dict[str, Any]] | None = None
        metrics: dict[str, Any] = Field(default_factory=dict)

    class TrustOverrideRequest(BaseModel):
        miner_id: str = Field(min_length=1)
        workload_type: str = Field(min_length=1)
        mode: str = Field(min_length=1)
        reason: str = ""

    def require_admin(token: str | None) -> None:
        if not configured_admin_token:
            raise HTTPException(status_code=403, detail="admin token is not configured")
        if not token_matches(token, configured_admin_token):
            raise HTTPException(status_code=403, detail="invalid admin token")

    def token_matches_registry(token: str | None) -> bool:
        if token is None:
            return False
        for entry in configured_miner_registry.values():
            if entry["enabled"] and token_matches(token, entry["token"]):
                return True
        return False

    def token_matches_shared(token: str | None) -> bool:
        if not configured_miner_token:
            return False
        return token_matches(token, configured_miner_token)

    def require_miner(token: str | None) -> None:
        if not configured_miner_token and not configured_miner_registry:
            return
        if token_matches_registry(token) or token_matches_shared(token):
            return
        raise HTTPException(status_code=401, detail="invalid miner token")

    def require_claim_miner(miner_id: str, token: str | None) -> None:
        if not configured_miner_token and not configured_miner_registry:
            return
        miner_name = str(miner_id or "")
        if miner_name in configured_miner_registry:
            entry = configured_miner_registry[miner_name]
            if not entry["enabled"]:
                raise HTTPException(status_code=401, detail="miner token is disabled")
            if token_matches(token, entry["token"]):
                return
            raise HTTPException(status_code=401, detail="invalid miner token")
        if token_matches_shared(token):
            return
        raise HTTPException(status_code=401, detail="invalid miner token")

    def require_observer(token: str | None) -> None:
        if not configured_observer_token:
            return
        if not token_matches(token, configured_observer_token):
            raise HTTPException(status_code=401, detail="invalid observer token")

    async def reaper_loop() -> None:
        while True:
            store.reap_expired()
            await asyncio.sleep(reaper_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(reaper_loop())
        app.state.reaper_task = task
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="CrowdTensorD Coordinator", version=SERVICE_VERSION, lifespan=lifespan)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=[
                "content-type",
                "x-crowdtensor-admin-token",
                "x-crowdtensor-miner-token",
                "x-crowdtensor-observer-token",
            ],
        )
    app.state.store = store

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": SERVICE_NAME, "version": SERVICE_VERSION}

    @app.get("/version")
    def version() -> dict:
        return version_payload()

    @app.get("/ready")
    def ready() -> dict:
        try:
            store.reap_expired()
            summary = store.summary()
            return readiness_payload(
                summary,
                miner_required=bool(configured_miner_token or configured_miner_registry),
                observer_required=bool(configured_observer_token),
                admin_configured=bool(configured_admin_token),
                miner_registry_configured=bool(configured_miner_registry),
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"ok": False, "reason": str(exc)}) from exc

    @app.get("/state")
    def state(
        x_crowdtensor_observer_token: str | None = Header(default=None),
    ) -> dict:
        require_observer(x_crowdtensor_observer_token)
        store.reap_expired()
        return store.summary()

    @app.get("/metrics")
    def metrics(
        x_crowdtensor_observer_token: str | None = Header(default=None),
    ) -> Response:
        require_observer(x_crowdtensor_observer_token)
        store.reap_expired()
        return Response(
            build_metrics_text(store.summary()),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/admin/events")
    def admin_events(
        limit: int = Query(default=50, ge=0, le=500),
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token)
        return {
            "events": store.event_tail(limit=limit),
            "limit": min(500, max(0, int(limit))),
        }

    @app.get("/admin/results")
    def admin_results(
        limit: int = Query(default=50, ge=0, le=500),
        status: str = Query(default="any"),
        miner_id: str = Query(default=""),
        workload_type: str = Query(default=""),
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token)
        try:
            return {
                "results": store.result_ledger(
                    limit=limit,
                    status=status,
                    miner_id=miner_id,
                    workload_type=workload_type,
                ),
                "limit": min(500, max(0, int(limit))),
                "status": status,
                "miner_id": miner_id,
                "workload_type": workload_type,
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/admin/trust-overrides")
    def admin_trust_overrides(
        request: TrustOverrideRequest,
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token)
        try:
            return store.set_trust_override(
                request.miner_id,
                request.workload_type,
                request.mode,
                reason=request.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/tasks/claim")
    def claim_task(
        request: ClaimRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict:
        require_claim_miner(request.miner_id, x_crowdtensor_miner_token)
        store.reap_expired()
        try:
            return store.claim_task(request.miner_id, capabilities=request.capabilities)
        except NoTaskAvailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/tasks/{task_id}/heartbeat")
    def heartbeat(
        task_id: str,
        request: LeaseRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict:
        require_miner(x_crowdtensor_miner_token)
        try:
            return store.heartbeat(
                task_id,
                lease_token=request.lease_token,
                attempt=request.attempt,
                runtime_status=request.runtime_status,
            )
        except LeaseConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/tasks/{task_id}/result")
    def result(
        task_id: str,
        request: ResultRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict:
        require_miner(x_crowdtensor_miner_token)
        try:
            return store.complete_task(
                task_id,
                lease_token=request.lease_token,
                attempt=request.attempt,
                idempotency_key=request.idempotency_key,
                local_delta=request.local_delta,
                pseudo_gradient=request.pseudo_gradient,
                compressed_delta=request.compressed_delta,
                probe_result=request.probe_result,
                adapter_delta=request.adapter_delta,
                bundle_delta=request.bundle_delta,
                inference_result=request.inference_result,
                inference_results=request.inference_results,
                external_llm_result=request.external_llm_result,
                external_llm_results=request.external_llm_results,
                metrics=request.metrics,
            )
        except ResultRejected as exc:
            raise HTTPException(status_code=422, detail=exc.validation) from exc
        except LeaseConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def parse_task_lane(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("task lane must use runtime:backend:count[:workload_type]")
    runtime, backend, count_text = parts[:3]
    workload_type = parts[3] if len(parts) == 4 else DEFAULT_WORKLOAD_TYPE
    try:
        count = int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("task lane count must be an integer") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("task lane count must be non-negative")
    return {
        "runtime": runtime or "any",
        "backend": backend or "any",
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "workload_type": workload_type or DEFAULT_WORKLOAD_TYPE,
        "count": count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CrowdTensorD Phase 1 Coordinator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--inner-steps", type=int, default=500)
    parser.add_argument("--backlog", type=int, default=1)
    parser.add_argument(
        "--task-lane",
        action="append",
        type=parse_task_lane,
        dest="task_lanes",
        default=None,
        help="runtime:backend:count[:workload_type] lane to keep queued; repeat for multiple lanes",
    )
    parser.add_argument("--reaper-interval", type=float, default=1.0)
    parser.add_argument(
        "--replay-audit",
        action="store_true",
        help="enable deterministic replay audit for supported training workloads",
    )
    parser.add_argument(
        "--outer-optimizer",
        choices=sorted(SUPPORTED_OUTER_OPTIMIZERS),
        default=OPTIMIZER_DILOCO_MOMENTUM,
        help="outer optimizer for new dense DiLoCo state",
    )
    parser.add_argument(
        "--delta-format",
        choices=sorted(SUPPORTED_DELTA_FORMATS),
        default="dense_float",
        help="claim-time delta transport format for diloco_train tasks",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help="admin token for control-plane endpoints; falls back to CROWDTENSOR_ADMIN_TOKEN",
    )
    parser.add_argument(
        "--miner-token",
        default=None,
        help="shared token for Miner task endpoints; falls back to CROWDTENSOR_MINER_TOKEN",
    )
    parser.add_argument(
        "--miner-token-registry",
        default=None,
        help="JSON per-miner token registry path; falls back to CROWDTENSOR_MINER_TOKEN_REGISTRY",
    )
    parser.add_argument(
        "--observer-token",
        default=None,
        help="shared token for /state and /metrics; falls back to CROWDTENSOR_OBSERVER_TOKEN",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        default=None,
        help="allowed browser origin for local browser Miner clients; repeat for multiple origins",
    )
    args = parser.parse_args()
    if args.replay_audit and args.delta_format == DELTA_FORMAT_SIGN_COMPRESSED_EF:
        parser.error("--delta-format sign_compressed_ef cannot be used with --replay-audit")
    return args


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit("uvicorn is not installed. Run: pip install -r requirements.txt") from exc

    app = create_app(
        state_dir=args.state_dir,
        lease_seconds=args.lease_seconds,
        inner_steps=args.inner_steps,
        backlog=args.backlog,
        task_lanes=args.task_lanes,
        reaper_interval=args.reaper_interval,
        cors_origins=args.cors_origins or ["http://127.0.0.1:8765", "http://localhost:8765"],
        replay_audit=args.replay_audit,
        outer_optimizer=args.outer_optimizer,
        delta_format=args.delta_format,
        admin_token=args.admin_token,
        miner_token=args.miner_token,
        miner_token_registry=args.miner_token_registry,
        observer_token=args.observer_token,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
