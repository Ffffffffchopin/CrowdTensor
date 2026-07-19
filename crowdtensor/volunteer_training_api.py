"""FastAPI transport for the Volunteer Training Protocol Alpha."""

from __future__ import annotations

import json
import secrets
import time
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response

from .community_security import SecurityContractError, TLSProxyPolicy
from .volunteer_training_coordinator import VolunteerTrainingCoordinator
from .volunteer_training_protocol import (
    MAX_SUBMISSION_METADATA_BYTES,
    SUBMISSION_SCHEMA,
    VolunteerProtocolError,
    decode_submission_envelope,
    hash_cell_id,
    public_error,
    with_public_safety,
)
from .volunteer_training_storage import (
    DEFAULT_CHUNK_BYTES,
    LocalVolunteerBlobStore,
    ResumableUploadManager,
    VolunteerBlobStore,
)


SERVICE_SCHEMA = "crowdtensor_volunteer_training_http_service_v1"
PROJECT_SITE_MEDIA_TYPES = {
    "favicon.png": "image/png",
    "site.css": "text/css; charset=utf-8",
    "site.js": "text/javascript; charset=utf-8",
    "hero-dashboard.png": "image/png",
}


def _dashboard_asset(name: str) -> bytes:
    allowed = {"index.html", "dashboard.css", "dashboard.js"}
    if name not in allowed:
        raise VolunteerProtocolError(
            "volunteer_dashboard_asset_not_found", status_code=404
        )
    return resources.files("crowdtensor.volunteer_dashboard").joinpath(name).read_bytes()


def _dashboard_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'"
        ),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }


def _project_site_asset(name: str) -> bytes:
    allowed = {"index.html", *PROJECT_SITE_MEDIA_TYPES}
    if name not in allowed:
        raise VolunteerProtocolError("project_site_asset_not_found", status_code=404)
    return resources.files("crowdtensor.project_site").joinpath(name).read_bytes()


def _project_site_headers(*, cache_assets: bool = False) -> dict[str, str]:
    return {
        "Cache-Control": "public, max-age=300" if cache_assets else "no-cache",
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        ),
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }


def _bearer_token(request: Request) -> str:
    value = str(request.headers.get("authorization") or "")
    if not value.startswith("Bearer ") or not value[7:].strip():
        raise VolunteerProtocolError(
            "volunteer_invite_authentication_required", status_code=401
        )
    return value[7:].strip()


def _request_nonce(request: Request) -> str:
    return str(request.headers.get("x-crowdtensor-nonce") or "")


async def _bounded_body(request: Request, *, max_bytes: int) -> bytes:
    value = bytearray()
    async for chunk in request.stream():
        value.extend(chunk)
        if len(value) > int(max_bytes):
            raise VolunteerProtocolError("volunteer_request_body_too_large", status_code=413)
    return bytes(value)


async def _json_object(
    request: Request, *, max_bytes: int = 64 * 1024
) -> dict[str, Any]:
    try:
        value = json.loads(
            (await _bounded_body(request, max_bytes=int(max_bytes))).decode("utf-8")
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VolunteerProtocolError("volunteer_request_json_invalid", status_code=400) from exc
    if not isinstance(value, dict):
        raise VolunteerProtocolError("volunteer_request_object_required", status_code=400)
    return value


def create_volunteer_training_app(
    coordinator: VolunteerTrainingCoordinator,
    *,
    tls_policy: TLSProxyPolicy | None = None,
    upload_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    upload_blob_store: VolunteerBlobStore | None = None,
) -> FastAPI:
    app = FastAPI(title="CrowdTensor Volunteer Training Alpha", version="1.0")
    policy = tls_policy or TLSProxyPolicy(require_https=False)
    campaign_manifest = coordinator.campaign_manifest()
    upload_store = upload_blob_store or LocalVolunteerBlobStore(
        coordinator.private / "upload-object-store"
    )
    upload_manager = ResumableUploadManager(
        coordinator.private / "resumable-uploads",
        blob_store=upload_store,
        max_upload_bytes=int(campaign_manifest["update_admission"]["max_delta_bytes"]),
        chunk_bytes=int(upload_chunk_bytes),
        clock=getattr(coordinator, "clock", time.time),
    )

    def _authenticated_bearer(
        request: Request,
        *,
        cell_id: str = "",
        scope: str = "",
        upload_bytes: int = 0,
    ) -> str:
        token = _bearer_token(request)
        if cell_id and scope and hasattr(coordinator, "authorize_cell_request"):
            coordinator.authorize_cell_request(
                token=token,
                cell_id=cell_id,
                scope=scope,
                request_nonce=_request_nonce(request),
                upload_bytes=int(upload_bytes),
            )
        else:
            coordinator.authenticate_invite(token)
        return token

    @app.middleware("http")
    async def enforce_tls_contract(request: Request, call_next: Any) -> Any:
        try:
            report = policy.validate(
                scheme=str(request.url.scheme),
                forwarded_proto=str(request.headers.get("x-forwarded-proto") or ""),
                proxy_identity=str(request.headers.get("x-crowdtensor-proxy-id") or ""),
            )
        except SecurityContractError as exc:
            return JSONResponse(public_error(str(exc)), status_code=400)
        request.state.volunteer_tls_contract = report
        return await call_next(request)

    @app.exception_handler(VolunteerProtocolError)
    async def protocol_error(_request: Request, exc: VolunteerProtocolError) -> JSONResponse:
        return JSONResponse(public_error(exc.code), status_code=exc.status_code)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def project_site() -> HTMLResponse:
        return HTMLResponse(
            _project_site_asset("index.html"), headers=_project_site_headers()
        )

    @app.head("/", include_in_schema=False)
    async def project_site_head() -> Response:
        return Response(
            media_type="text/html; charset=utf-8", headers=_project_site_headers()
        )

    @app.get("/assets/{asset_name}", include_in_schema=False)
    async def project_site_asset(asset_name: str) -> Response:
        media_type = PROJECT_SITE_MEDIA_TYPES.get(asset_name)
        if media_type is None:
            raise VolunteerProtocolError("project_site_asset_not_found", status_code=404)
        return Response(
            _project_site_asset(asset_name),
            media_type=media_type,
            headers=_project_site_headers(cache_assets=True),
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def project_site_favicon() -> Response:
        return Response(
            _project_site_asset("favicon.png"),
            media_type="image/png",
            headers=_project_site_headers(cache_assets=True),
        )

    @app.get("/v1/volunteer/health")
    async def health() -> dict[str, Any]:
        return with_public_safety(
            {
                "schema": SERVICE_SCHEMA,
                "ok": True,
                "service": "volunteer_training_coordinator",
                "campaign_id": coordinator.campaign_manifest()["campaign_id"],
                "binary_safetensors_submission": True,
                "authenticated_artifact_download": True,
                "resumable_chunk_upload": True,
                "tls_required": bool(policy.require_https),
                "trusted_forwarded_headers": bool(policy.trust_forwarded_headers),
            }
        )

    @app.get("/v1/volunteer/campaign")
    async def campaign() -> dict[str, Any]:
        return coordinator.campaign_manifest()

    @app.get("/v1/volunteer/status")
    async def status() -> dict[str, Any]:
        return coordinator.status()

    @app.get("/v1/volunteer/public-snapshot")
    async def public_snapshot() -> dict[str, Any]:
        if hasattr(coordinator, "public_campaign_snapshot"):
            return coordinator.public_campaign_snapshot()
        return with_public_safety(
            {
                "schema": "crowdtensor_volunteer_public_campaign_snapshot_v1",
                "ok": True,
                "campaign": coordinator.campaign_manifest(),
                "progress": coordinator.status(),
                "rounds": [],
                "activity": [],
            }
        )

    @app.get("/v1/volunteer/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(
            _dashboard_asset("index.html"), headers=_dashboard_headers()
        )

    @app.get("/v1/volunteer/dashboard/assets/{asset_name}")
    async def dashboard_asset(asset_name: str) -> Response:
        media_types = {
            "dashboard.css": "text/css; charset=utf-8",
            "dashboard.js": "text/javascript; charset=utf-8",
        }
        if asset_name not in media_types:
            raise VolunteerProtocolError(
                "volunteer_dashboard_asset_not_found", status_code=404
            )
        return Response(
            _dashboard_asset(asset_name),
            media_type=media_types[asset_name],
            headers=_dashboard_headers(),
        )

    @app.get("/v1/volunteer/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        if hasattr(coordinator, "prometheus_metrics"):
            return coordinator.prometheus_metrics()
        return "crowdtensor_volunteer_service_up 1\n"

    @app.post("/v1/volunteer/credentials/issue")
    async def credential_issue(request: Request) -> dict[str, Any]:
        token = _bearer_token(request)
        payload = await _json_object(request)
        if not hasattr(coordinator, "issue_cell_credential"):
            raise VolunteerProtocolError(
                "volunteer_cell_credentials_unavailable", status_code=404
            )
        scopes = payload.get("scopes")
        if scopes is not None and not isinstance(scopes, list):
            raise VolunteerProtocolError(
                "volunteer_credential_scope_invalid", status_code=400
            )
        ttl = payload.get("ttl_seconds")
        return coordinator.issue_cell_credential(
            invite_token=token,
            cell_id=str(payload.get("cell_id") or ""),
            scopes=scopes,
            ttl_seconds=int(ttl) if ttl is not None else None,
        )

    @app.post("/v1/volunteer/credentials/revoke")
    async def credential_revoke(request: Request) -> dict[str, Any]:
        token = _bearer_token(request)
        payload = await _json_object(request)
        if not hasattr(coordinator, "revoke_cell_credential"):
            raise VolunteerProtocolError(
                "volunteer_cell_credentials_unavailable", status_code=404
            )
        return coordinator.revoke_cell_credential(
            invite_token=token,
            credential_id=str(payload.get("credential_id") or ""),
        )

    @app.post("/v1/volunteer/work/claim")
    async def claim(request: Request) -> dict[str, Any]:
        token = _bearer_token(request)
        payload = await _json_object(request)
        kwargs = {
            "cell_id": str(payload.get("cell_id") or ""),
            "invite_token": token,
            "capability": payload.get("capability")
            if isinstance(payload.get("capability"), dict)
            else {},
        }
        if hasattr(coordinator, "authorize_cell_request"):
            kwargs["request_nonce"] = _request_nonce(request)
        return coordinator.claim(**kwargs)

    @app.post("/v1/volunteer/work/heartbeat")
    async def heartbeat(request: Request) -> dict[str, Any]:
        token = _bearer_token(request)
        payload = await _json_object(request)
        kwargs = dict(
            cell_id=str(payload.get("cell_id") or ""),
            invite_token=token,
            work_id=str(payload.get("work_id") or ""),
            lease_generation=int(payload.get("lease_generation") or 0),
            lease_token=str(payload.get("lease_token") or ""),
        )
        if hasattr(coordinator, "authorize_cell_request"):
            kwargs["request_nonce"] = _request_nonce(request)
        return coordinator.heartbeat(**kwargs)

    @app.get("/v1/volunteer/artifacts/{artifact_id}")
    async def artifact(artifact_id: str, request: Request) -> FileResponse:
        token = _bearer_token(request)
        kwargs = {"invite_token": token}
        if hasattr(coordinator, "authorize_cell_request"):
            kwargs.update(
                {
                    "cell_id": str(request.headers.get("x-crowdtensor-cell-id") or ""),
                    "request_nonce": _request_nonce(request),
                }
            )
        path = coordinator.artifact_path(artifact_id, **kwargs)
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename="crowdtensor-artifact.bin",
        )

    @app.post("/v1/volunteer/work/submit")
    async def submit(request: Request) -> dict[str, Any]:
        token = _bearer_token(request)
        campaign = coordinator.campaign_manifest()
        max_delta_bytes = int(campaign["update_admission"]["max_delta_bytes"])
        request_limit = max_delta_bytes + MAX_SUBMISSION_METADATA_BYTES + 8
        try:
            content_length = int(request.headers.get("content-length") or 0)
        except ValueError as exc:
            raise VolunteerProtocolError(
                "volunteer_content_length_invalid", status_code=400
            ) from exc
        if content_length < 0 or content_length > request_limit:
            raise VolunteerProtocolError(
                "volunteer_submission_delta_too_large", status_code=413
            )
        body = await _bounded_body(request, max_bytes=request_limit)
        metadata, delta = decode_submission_envelope(
            body, max_delta_bytes=max_delta_bytes
        )
        if metadata.get("schema") != SUBMISSION_SCHEMA:
            raise VolunteerProtocolError(
                "volunteer_submission_schema_mismatch", status_code=400
            )
        manifest = metadata.get("delta_manifest")
        if not isinstance(manifest, dict):
            raise VolunteerProtocolError(
                "volunteer_submission_manifest_missing", status_code=400
            )
        upload_dir = coordinator.private / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload = upload_dir / f"upload-{secrets.token_hex(16)}.safetensors"
        try:
            upload.write_bytes(delta)
            upload.chmod(0o600)
            private_manifest = dict(manifest)
            private_manifest["delta_path"] = str(upload)
            kwargs = dict(
                cell_id=str(metadata.get("cell_id") or ""),
                invite_token=token,
                work_id=str(metadata.get("work_id") or ""),
                lease_generation=int(metadata.get("lease_generation") or 0),
                lease_token=str(metadata.get("lease_token") or ""),
                delta_manifest=private_manifest,
            )
            if hasattr(coordinator, "authorize_cell_request"):
                kwargs["request_nonce"] = _request_nonce(request)
            return coordinator.submit(**kwargs)
        finally:
            upload.unlink(missing_ok=True)

    @app.post("/v1/volunteer/uploads/start")
    async def upload_start(request: Request) -> dict[str, Any]:
        payload = await _json_object(
            request, max_bytes=MAX_SUBMISSION_METADATA_BYTES + 64 * 1024
        )
        cell_id = str(payload.get("cell_id") or "")
        _authenticated_bearer(
            request, cell_id=cell_id, scope="upload:write"
        )
        metadata = payload.get("submission")
        if not isinstance(metadata, dict) or metadata.get("schema") != SUBMISSION_SCHEMA:
            raise VolunteerProtocolError(
                "volunteer_submission_schema_mismatch", status_code=400
            )
        if str(metadata.get("cell_id") or "") != cell_id:
            raise VolunteerProtocolError(
                "volunteer_upload_cell_identity_mismatch", status_code=403
            )
        return upload_manager.start(
            owner_cell_hash=hash_cell_id(cell_id),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            expected_blob_hash=str(payload.get("expected_blob_hash") or ""),
            total_bytes=int(payload.get("total_bytes") or 0),
            private_metadata=metadata,
        )

    def _upload_cell_hash(request: Request) -> str:
        cell_id = str(request.headers.get("x-crowdtensor-cell-id") or "")
        if not cell_id:
            raise VolunteerProtocolError("volunteer_cell_id_missing", status_code=400)
        return hash_cell_id(cell_id)

    @app.get("/v1/volunteer/uploads-report")
    async def uploads_report(request: Request) -> dict[str, Any]:
        _authenticated_bearer(request)
        return upload_manager.public_report()

    @app.get("/v1/volunteer/uploads/{upload_id}")
    async def upload_status(upload_id: str, request: Request) -> dict[str, Any]:
        cell_id = str(request.headers.get("x-crowdtensor-cell-id") or "")
        _authenticated_bearer(
            request, cell_id=cell_id, scope="upload:read"
        )
        return upload_manager.status(
            upload_id, owner_cell_hash=_upload_cell_hash(request)
        )

    @app.put("/v1/volunteer/uploads/{upload_id}/chunks/{chunk_index}")
    async def upload_chunk(
        upload_id: str, chunk_index: int, request: Request
    ) -> dict[str, Any]:
        cell_id = str(request.headers.get("x-crowdtensor-cell-id") or "")
        try:
            content_length = int(request.headers.get("content-length") or 0)
        except ValueError as exc:
            raise VolunteerProtocolError(
                "volunteer_content_length_invalid", status_code=400
            ) from exc
        _authenticated_bearer(
            request,
            cell_id=cell_id,
            scope="upload:write",
            upload_bytes=content_length,
        )
        chunk_hash = str(request.headers.get("x-crowdtensor-chunk-sha256") or "")
        value = await _bounded_body(request, max_bytes=int(upload_chunk_bytes))
        return upload_manager.put_chunk(
            upload_id,
            owner_cell_hash=_upload_cell_hash(request),
            chunk_index=int(chunk_index),
            chunk_hash=chunk_hash,
            value=value,
        )

    @app.post("/v1/volunteer/uploads/{upload_id}/complete")
    async def upload_complete(upload_id: str, request: Request) -> dict[str, Any]:
        cell_id = str(request.headers.get("x-crowdtensor-cell-id") or "")
        token = _authenticated_bearer(
            request, cell_id=cell_id, scope="upload:write"
        )
        owner_hash = _upload_cell_hash(request)
        completed = upload_manager.complete(upload_id, owner_cell_hash=owner_hash)
        metadata = completed.get("private_metadata")
        if not isinstance(metadata, dict):
            raise VolunteerProtocolError(
                "volunteer_submission_metadata_invalid", status_code=409
            )
        manifest = metadata.get("delta_manifest")
        if not isinstance(manifest, dict):
            raise VolunteerProtocolError(
                "volunteer_submission_manifest_missing", status_code=400
            )
        private_manifest = dict(manifest)
        private_manifest["delta_path"] = str(
            upload_manager.completed_blob_path(upload_id, owner_cell_hash=owner_hash)
        )
        kwargs = dict(
            cell_id=str(metadata.get("cell_id") or ""),
            invite_token=token,
            work_id=str(metadata.get("work_id") or ""),
            lease_generation=int(metadata.get("lease_generation") or 0),
            lease_token=str(metadata.get("lease_token") or ""),
            delta_manifest=private_manifest,
        )
        if hasattr(coordinator, "authorize_cell_request"):
            kwargs["request_nonce"] = _request_nonce(request) + ":submit"
        response = coordinator.submit(**kwargs)
        response["resumable_upload"] = {
            key: value
            for key, value in completed.items()
            if key != "private_metadata"
        }
        return response

    return app


def service_contract() -> dict[str, Any]:
    return with_public_safety(
        {
            "schema": SERVICE_SCHEMA,
            "routes": [
                "GET /v1/volunteer/health",
                "GET /v1/volunteer/campaign",
                "GET /v1/volunteer/status",
                "GET /v1/volunteer/public-snapshot",
                "GET /v1/volunteer/dashboard",
                "GET /v1/volunteer/metrics",
                "POST /v1/volunteer/credentials/issue",
                "POST /v1/volunteer/credentials/revoke",
                "POST /v1/volunteer/work/claim",
                "POST /v1/volunteer/work/heartbeat",
                "GET /v1/volunteer/artifacts/{artifact_id}",
                "POST /v1/volunteer/work/submit",
                "POST /v1/volunteer/uploads/start",
                "GET /v1/volunteer/uploads/{upload_id}",
                "PUT /v1/volunteer/uploads/{upload_id}/chunks/{chunk_index}",
                "POST /v1/volunteer/uploads/{upload_id}/complete",
                "GET /v1/volunteer/uploads-report",
            ],
            "invite_bearer_required_for_private_routes": True,
            "per_cell_short_lived_credentials": True,
            "credential_scope_and_revocation_enforced": True,
            "request_nonce_replay_protection": True,
            "request_upload_quota_and_rate_limits": True,
            "invite_bearer_validated_before_upload_allocation": True,
            "binary_safetensors_submission": True,
            "raw_tensor_json_transport": False,
            "external_tls_termination_required": True,
            "trusted_proxy_contract_available": True,
            "resumable_chunk_upload": True,
            "upload_state_survives_coordinator_restart": True,
            "content_addressed_upload_completion": True,
        }
    )
