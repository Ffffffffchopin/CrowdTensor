"""Role-separated Community API routes layered over the training runtime."""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from .community_security import (
    CredentialAuthority,
    ReplayWindow,
    RestrictedExecutionPolicy,
    SecurityContractError,
    TLSProxyPolicy,
    TaskEnvelopeSigner,
    UpdateAnomalyDetector,
    authorize,
)
from .community_workflow import CommunityWorkflow
from .version import COMMUNITY_PROTOCOL_VERSION


API_SCHEMA = "crowdtensor_community_api_v1"


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


@dataclass
class CommunityPrivateCredentials:
    owner: str
    miner: str
    observer: str


class CommunitySecurityContext:
    def __init__(
        self,
        *,
        issuer: str,
        signing_key: bytes | None = None,
        tls_policy: TLSProxyPolicy | None = None,
        execution_policy: RestrictedExecutionPolicy | None = None,
    ) -> None:
        key = signing_key or secrets.token_bytes(32)
        self.authority = CredentialAuthority(issuer=issuer, key=key)
        self.task_signer = TaskEnvelopeSigner(key)
        self.replay = ReplayWindow()
        self.tls_policy = tls_policy or TLSProxyPolicy(require_https=True)
        self.execution_policy = execution_policy or RestrictedExecutionPolicy()
        self.anomaly_detector = UpdateAnomalyDetector()
        self.audit_events: list[dict[str, Any]] = []
        self.credentials = self._issue_all()

    def _issue_all(self) -> CommunityPrivateCredentials:
        return CommunityPrivateCredentials(
            owner=self.authority.issue(subject="owner", role="owner")[0],
            miner=self.authority.issue(subject="miner", role="miner")[0],
            observer=self.authority.issue(subject="observer", role="observer")[0],
        )

    def rotate(self) -> tuple[CommunityPrivateCredentials, dict[str, Any]]:
        report = self.authority.rotate(retain_previous=1)
        self.credentials = self._issue_all()
        return self.credentials, report

    def audit(self, *, action: str, role: str, allowed: bool) -> None:
        self.audit_events.append(
            {
                "sequence": len(self.audit_events) + 1,
                "action": str(action),
                "role": str(role),
                "allowed": bool(allowed),
                "recorded_at": int(time.time()),
                "credential_value_public": False,
            }
        )


def create_community_app(
    workflow: CommunityWorkflow,
    *,
    context: CommunitySecurityContext,
    base_app: Any | None = None,
) -> Any:
    from fastapi import FastAPI, Header, HTTPException

    app = base_app or FastAPI(title="CrowdTensor Community API", docs_url=None, redoc_url=None)

    def transport(request: Request) -> None:
        try:
            context.tls_policy.validate(
                scheme=request.url.scheme,
                forwarded_proto=str(request.headers.get("x-forwarded-proto") or ""),
                proxy_identity=str(request.headers.get("x-crowdtensor-proxy-id") or ""),
            )
        except SecurityContractError as exc:
            raise HTTPException(status_code=426, detail=str(exc)) from exc

    def authorize_request(value: str | None, permission: str) -> str:
        if not value or not value.startswith("Bearer "):
            context.audit(action=permission, role="unknown", allowed=False)
            raise HTTPException(status_code=401, detail="community_bearer_credential_required")
        try:
            claims = context.authority.verify(value[7:])
        except SecurityContractError as exc:
            context.audit(action=permission, role="unknown", allowed=False)
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        role = str(claims.get("role") or "")
        decision = authorize(role, permission)
        context.audit(action=permission, role=role, allowed=decision["allowed"])
        if not decision["allowed"]:
            raise HTTPException(status_code=403, detail="community_rbac_permission_denied")
        return role

    @app.get("/v1/community/health")
    def health(request: Request) -> dict[str, Any]:
        transport(request)
        return {
            "schema": API_SCHEMA,
            "ok": True,
            "protocol_version": COMMUNITY_PROTOCOL_VERSION,
            "tls_contract_enforced": context.tls_policy.require_https,
            "public_artifact_safe": True,
        }

    @app.get("/v1/community/status")
    def status(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        transport(request)
        authorize_request(authorization, "job:read")
        return workflow.status()

    @app.post("/v1/community/control/{action}")
    def control(
        action: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        transport(request)
        authorize_request(authorization, "job:control")
        if action not in {"pause", "resume", "rebalance", "stop", "cleanup"}:
            raise HTTPException(status_code=404, detail="community_control_action_unsupported")
        method = getattr(workflow, action)
        return method(dry_run=True)

    @app.post("/v1/community/tasks/verify")
    async def verify_task(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        transport(request)
        authorize_request(authorization, "task:claim")
        envelope = await request.json()
        try:
            payload = context.task_signer.verify(
                envelope,
                replay_window=context.replay,
                expected_protocol_version=COMMUNITY_PROTOCOL_VERSION,
            )
        except (SecurityContractError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "schema": "crowdtensor_community_task_verification_v1",
            "ok": True,
            "payload_hash": _hash(str(payload)),
            "payload_values_public": False,
            "public_artifact_safe": True,
        }

    @app.post("/v1/community/execution/validate")
    async def validate_execution(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        transport(request)
        authorize_request(authorization, "task:claim")
        value = await request.json()
        return context.execution_policy.validate(
            value.get("command") or [],
            file_paths=value.get("file_paths") or [],
            network_urls=value.get("network_urls") or [],
            memory_bytes=int(value.get("memory_bytes") or 0),
            cpu_seconds=int(value.get("cpu_seconds") or 0),
            output_bytes=int(value.get("output_bytes") or 0),
        )

    @app.post("/v1/community/updates/inspect")
    async def inspect_update(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        transport(request)
        authorize_request(authorization, "task:submit")
        value = await request.json()
        return context.anomaly_detector.inspect(
            miner_id=str(value.get("miner_id") or ""),
            values=value.get("values") or [],
            expected_count=int(value.get("expected_count") or 0),
        )

    @app.get("/v1/community/audit")
    def audit_log(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        transport(request)
        authorize_request(authorization, "events:read_redacted")
        return {
            "schema": "crowdtensor_community_audit_log_v1",
            "events": list(context.audit_events[-200:]),
            "credential_values_public": False,
            "public_artifact_safe": True,
        }

    app.state.community_security_context = context
    return app
