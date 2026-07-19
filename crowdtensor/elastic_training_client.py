"""Private HTTP client used by elastic volunteer training Miners."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .elastic_training_runtime import (
    build_qwen_stage_checkpoint_archive,
    restore_qwen_stage_checkpoint_archive,
    sign_checkpoint_submission,
)
from .heterogeneous_tensor_transport import (
    decode_tensor_payload,
    encode_tensor_message,
)
from .heterogeneous_training_checkpoint import (
    build_stage_checkpoint_archive,
    restore_stage_checkpoint_archive,
)


PERSISTENT_HTTP_MAX_BODY_BYTES = 4 * 1024 * 1024


def _sha(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class ElasticTrainingHTTPClient:
    """Bounded, retrying Miner client that keeps all fencing tokens private."""

    def __init__(
        self,
        *,
        coordinator_url: str,
        coordinator_token: str,
        run_id: str,
        miner_id_hash: str,
        registration_nonce: str,
        supported_stage_ids: list[int],
        slot_count: int,
        accelerator: str = "cuda",
        capability: dict[str, Any] | None = None,
        timeout: float = 60.0,
        retry_attempts: int = 8,
        retry_base_seconds: float = 0.5,
        heartbeat_interval_seconds: float = 5.0,
        persistent_http_after_step: int = -1,
        persistent_http_max_body_bytes: int = PERSISTENT_HTTP_MAX_BODY_BYTES,
    ) -> None:
        self._url = str(coordinator_url).rstrip("/")
        self._coordinator_token = str(coordinator_token)
        self.run_id = str(run_id)
        self.miner_id_hash = str(miner_id_hash)
        self._registration_nonce = str(registration_nonce)
        self.supported_stage_ids = sorted({int(value) for value in supported_stage_ids})
        self.slot_count = int(slot_count)
        self.accelerator = str(accelerator)
        self.capability = dict(capability or {})
        self.timeout = float(timeout)
        self.retry_attempts = int(retry_attempts)
        self.retry_base_seconds = float(retry_base_seconds)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.persistent_http_after_step = int(persistent_http_after_step)
        self.persistent_http_max_body_bytes = int(
            persistent_http_max_body_bytes
        )
        self._current_step = 0
        self._httpx_client: Any | None = None
        self._httpx_lock = threading.RLock()
        self._session_id = ""
        self._session_token = ""
        self._registration: dict[str, Any] = {}
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: BaseException | None = None
        self._heartbeat_failure_count = 0
        self._heartbeat_recovery_count = 0
        self._request_count = 0
        self._retry_count = 0
        self._checkpoint_upload_count = 0
        self._checkpoint_download_count = 0
        self._barrier_commit_count = 0
        self._tensor_upload_count = 0
        self._tensor_download_count = 0
        self._inline_tensor_upload_count = 0
        self._inline_tensor_download_count = 0
        self._stage_profile_count = 0
        self._persistent_request_count = 0
        self._legacy_request_count = 0
        self._persistent_transport_fallback_count = 0
        self._large_payload_connection_isolation_count = 0
        if (
            not self._url
            or not self._coordinator_token
            or not self.run_id
            or not self.miner_id_hash.startswith("sha256:")
            or not self._registration_nonce
            or not self.supported_stage_ids
            or self.slot_count < 1
            or self.retry_attempts < 1
            or self.persistent_http_after_step < -1
            or self.persistent_http_max_body_bytes < 1
        ):
            raise ValueError("elastic_client_configuration_invalid")

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code == 429 or exc.code >= 500
        response = getattr(exc, "response", None)
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code:
            return status_code == 429 or status_code >= 500
        if type(exc).__module__.startswith("httpx"):
            return True
        return isinstance(
            exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError)
        )

    @staticmethod
    def _error(exc: BaseException) -> RuntimeError:
        if isinstance(exc, urllib.error.HTTPError):
            detail = ""
            try:
                parsed = json.loads(exc.read().decode("utf-8"))
                detail = str(parsed.get("detail") or "") if isinstance(parsed, dict) else ""
            except BaseException:
                detail = ""
            code = detail or f"elastic_coordinator_http_{exc.code}"
            return RuntimeError(code)
        response = getattr(exc, "response", None)
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code:
            detail = ""
            try:
                parsed = response.json()
                detail = str(parsed.get("detail") or "") if isinstance(parsed, dict) else ""
            except BaseException:
                detail = ""
            return RuntimeError(
                detail or f"elastic_coordinator_http_{status_code}"
            )
        return RuntimeError("elastic_coordinator_transport_unavailable")

    def set_current_step(self, step: int) -> None:
        if int(step) < 0:
            raise ValueError("elastic_client_current_step_invalid")
        self._current_step = int(step)

    def _persistent_transport_enabled(self) -> bool:
        return bool(
            self.persistent_http_after_step >= 0
            and self._current_step > self.persistent_http_after_step
        )

    def _persistent_request_eligible(self, payload: bytes | None) -> bool:
        return bool(
            self._persistent_transport_enabled()
            and (
                payload is None
                or len(payload) <= self.persistent_http_max_body_bytes
            )
        )

    def _persistent_request(
        self,
        *,
        path: str,
        method: str,
        payload: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[bytes, dict[str, str]]:
        try:
            import httpx
        except ImportError as exc:
            self._persistent_transport_fallback_count += 1
            raise RuntimeError("elastic_httpx_transport_unavailable") from exc
        with self._httpx_lock:
            if self._httpx_client is None:
                self._httpx_client = httpx.Client(
                    timeout=httpx.Timeout(timeout),
                    limits=httpx.Limits(
                        max_connections=16,
                        max_keepalive_connections=8,
                        keepalive_expiry=60.0,
                    ),
                    follow_redirects=False,
                )
            client = self._httpx_client
        response = client.request(
            method,
            f"{self._url}{path}",
            content=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return bytes(response.content), {
            str(key).lower(): str(value) for key, value in response.headers.items()
        }

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        json_value: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        retry_attempts: int | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        if json_value is not None and body is not None:
            raise ValueError("elastic_client_request_body_conflict")
        payload = body
        request_headers = {
            "User-Agent": "crowdtensor-elastic-training-miner/1",
            "x-crowdtensor-miner-token": self._coordinator_token,
            **dict(headers or {}),
        }
        if json_value is not None:
            payload = json.dumps(json_value, separators=(",", ":")).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        last_error: BaseException | None = None
        attempts = int(retry_attempts or self.retry_attempts)
        request_timeout = float(timeout or self.timeout)
        for attempt in range(attempts):
            try:
                self._request_count += 1
                if self._persistent_request_eligible(payload):
                    try:
                        result = self._persistent_request(
                            path=path,
                            method=method,
                            payload=payload,
                            headers=request_headers,
                            timeout=request_timeout,
                        )
                        self._persistent_request_count += 1
                        return result
                    except RuntimeError as exc:
                        if str(exc) != "elastic_httpx_transport_unavailable":
                            raise
                        self._legacy_request_count += 1
                else:
                    self._legacy_request_count += 1
                    if (
                        self._persistent_transport_enabled()
                        and payload is not None
                        and len(payload) > self.persistent_http_max_body_bytes
                    ):
                        self._large_payload_connection_isolation_count += 1
                request = urllib.request.Request(
                    f"{self._url}{path}",
                    data=payload,
                    headers=request_headers,
                    method=method,
                )
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    return response.read(), {
                        str(key).lower(): str(value)
                        for key, value in response.headers.items()
                    }
            except BaseException as exc:
                last_error = exc
            if (
                last_error is None
                or not self._retryable(last_error)
                or attempt + 1 >= attempts
            ):
                raise self._error(last_error or RuntimeError("unknown")) from last_error
            self._retry_count += 1
            jitter_byte = hashlib.sha256(
                f"{self.run_id}:{path}:{attempt}".encode("utf-8")
            ).digest()[0]
            jitter = 0.75 + (float(jitter_byte) / 255.0) * 0.5
            delay = min(
                5.0,
                self.retry_base_seconds * (2 ** min(attempt, 4)) * jitter,
            )
            time.sleep(delay)
        raise self._error(last_error or RuntimeError("unknown"))

    def _json_request(
        self,
        path: str,
        *,
        method: str = "GET",
        value: dict[str, Any] | None = None,
        session: bool = False,
        timeout: float | None = None,
        retry_attempts: int | None = None,
    ) -> dict[str, Any]:
        headers = {}
        if session:
            self._require_registered()
            headers["x-crowdtensor-elastic-session-token"] = self._session_token
        raw, _headers = self._request(
            path,
            method=method,
            json_value=value,
            headers=headers,
            timeout=timeout,
            retry_attempts=retry_attempts,
        )
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("elastic_coordinator_response_invalid") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("elastic_coordinator_response_invalid")
        return parsed

    def _require_registered(self) -> None:
        if not self._session_id or not self._session_token:
            raise RuntimeError("elastic_miner_not_registered")

    def _raise_heartbeat_error(self) -> None:
        if self._heartbeat_error is not None:
            raise RuntimeError("elastic_miner_heartbeat_failed") from self._heartbeat_error

    def register(self) -> dict[str, Any]:
        response = self._json_request(
            "/elastic-training/miners/register",
            method="POST",
            value={
                "run_id": self.run_id,
                "miner_id_hash": self.miner_id_hash,
                "registration_nonce": self._registration_nonce,
                "supported_stage_ids": self.supported_stage_ids,
                "slot_count": self.slot_count,
                "accelerator": self.accelerator,
                "capability": self.capability or None,
            },
        )
        session_id = str(response.get("session_id") or "")
        session_token = str(response.get("session_token") or "")
        if not session_id or not session_token:
            raise RuntimeError("elastic_miner_registration_response_invalid")
        self._session_id = session_id
        self._session_token = session_token
        self._registration = dict(response)
        return response

    def capabilities(self) -> dict[str, Any]:
        return self._json_request("/elastic-training/capabilities")

    def _assignment_request(
        self,
        path: str,
        *,
        assignment: dict[str, Any],
        method: str = "GET",
        json_value: dict[str, Any] | None = None,
        body: bytes | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        self._require_registered()
        return self._request(
            path,
            method=method,
            json_value=json_value,
            body=body,
            headers={
                "x-crowdtensor-elastic-session-id": self._session_id,
                "x-crowdtensor-elastic-session-token": self._session_token,
                "x-crowdtensor-elastic-assignment-token": str(
                    assignment["assignment_token"]
                ),
            },
        )

    @staticmethod
    def _parsed_json(value: bytes) -> dict[str, Any]:
        try:
            result = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("elastic_coordinator_response_invalid") from exc
        if not isinstance(result, dict):
            raise RuntimeError("elastic_coordinator_response_invalid")
        return result

    def report_stage_runtime(
        self,
        assignment: dict[str, Any],
        *,
        event_type: str = "profile",
        forward_latency_ms: float = 0.0,
        backward_latency_ms: float = 0.0,
        peak_memory_bytes: int = 0,
        sample_count: int = 1,
        compile_latency_ms: float = 0.0,
        steady_forward_latency_ms: float = 0.0,
        steady_backward_latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        raw, _headers = self._assignment_request(
            "/elastic-training/miners/"
            f"{urllib.parse.quote(self._session_id)}/stage-runtime",
            assignment=assignment,
            method="POST",
            json_value={
                "placement_generation": int(assignment["placement_generation"]),
                "stage_id": int(assignment["stage_id"]),
                "device_id": str(assignment["device_id"]),
                "event_type": str(event_type),
                "forward_latency_ms": float(forward_latency_ms),
                "backward_latency_ms": float(backward_latency_ms),
                "peak_memory_bytes": int(peak_memory_bytes),
                "sample_count": int(sample_count),
                "compile_latency_ms": float(compile_latency_ms),
                "steady_forward_latency_ms": float(steady_forward_latency_ms),
                "steady_backward_latency_ms": float(steady_backward_latency_ms),
            },
        )
        self._stage_profile_count += event_type == "profile"
        return self._parsed_json(raw)

    def send_tensors(
        self,
        assignment: dict[str, Any],
        tensors: dict[str, Any],
        *,
        target_stage_id: int,
        direction: str,
        microbatch_id: int,
        manifest_hash: str,
        chunk_bytes: int = 4 * 1024 * 1024,
        ttl_seconds: float = 300.0,
    ) -> dict[str, Any]:
        envelope, chunks = encode_tensor_message(
            tensors,
            job_id=self.run_id,
            manifest_hash=str(manifest_hash),
            global_step=int(assignment["target_step"]),
            microbatch_id=int(microbatch_id),
            source_stage_id=int(assignment["stage_id"]),
            target_stage_id=int(target_stage_id),
            direction=str(direction),
            placement_generation=int(assignment["placement_generation"]),
            assignment_token_hash=str(assignment["assignment_token_hash"]),
            chunk_bytes=int(chunk_bytes),
            ttl_seconds=float(ttl_seconds),
            max_delivery_attempts=min(3, self.retry_attempts),
        )
        if self._persistent_transport_enabled() and len(chunks) == 1:
            raw, _headers = self._assignment_request(
                "/elastic-training/tensors/inline",
                assignment=assignment,
                method="POST",
                json_value={
                    "envelope": envelope,
                    "chunk_b64": base64.b64encode(chunks[0]).decode("ascii"),
                },
            )
            inline = self._parsed_json(raw)
            if inline.get("complete") is not True:
                raise RuntimeError("elastic_inline_tensor_upload_incomplete")
            self._tensor_upload_count += 1
            self._inline_tensor_upload_count += 1
            return {
                "schema": "crowdtensor_elastic_tensor_send_v1",
                "message_id": envelope["message_id"],
                "payload_hash": envelope["payload_hash"],
                "payload_bytes": int(envelope["payload_bytes"]),
                "chunk_count": 1,
                "begin_complete": True,
                "delivery_complete": True,
                "inline_transport_used": True,
                "tensor_values_public": False,
                "public_artifact_safe": True,
            }
        raw, _headers = self._assignment_request(
            "/elastic-training/tensors/begin",
            assignment=assignment,
            method="POST",
            json_value=envelope,
        )
        begin = self._parsed_json(raw)
        for index, chunk in enumerate(chunks):
            raw, _headers = self._assignment_request(
                "/elastic-training/tensors/"
                f"{urllib.parse.quote(envelope['message_id'], safe='')}/{index}",
                assignment=assignment,
                method="PUT",
                body=chunk,
            )
            self._parsed_json(raw)
            self._tensor_upload_count += 1
        return {
            "schema": "crowdtensor_elastic_tensor_send_v1",
            "message_id": envelope["message_id"],
            "payload_hash": envelope["payload_hash"],
            "payload_bytes": int(envelope["payload_bytes"]),
            "chunk_count": len(chunks),
            "begin_complete": bool(begin.get("complete")) or not begin.get(
                "missing_chunk_indices"
            ),
            "delivery_complete": True,
            "inline_transport_used": False,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }

    def receive_tensors(
        self,
        assignment: dict[str, Any],
        *,
        source_stage_id: int,
        direction: str,
        microbatch_id: int,
        timeout: float,
        target_device: str,
        target_dtype: str | None = None,
        poll_interval: float = 0.25,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        deadline = time.monotonic() + float(timeout)
        query = urllib.parse.urlencode(
            {
                "global_step": int(assignment["target_step"]),
                "microbatch_id": int(microbatch_id),
                "source_stage_id": int(source_stage_id),
                "target_stage_id": int(assignment["stage_id"]),
                "direction": str(direction),
                "placement_generation": int(assignment["placement_generation"]),
            }
        )
        lookup: dict[str, Any] = {}
        while time.monotonic() < deadline:
            lookup_path = (
                f"/elastic-training/tensors/inline?{query}"
                if self._persistent_transport_enabled()
                else f"/elastic-training/tensors/lookup?{query}"
            )
            raw, _headers = self._assignment_request(
                lookup_path,
                assignment=assignment,
            )
            lookup = self._parsed_json(raw)
            if lookup.get("found") is True and (lookup.get("status") or {}).get(
                "complete"
            ) is True:
                break
            time.sleep(min(float(poll_interval), max(0.0, deadline - time.monotonic())))
        else:
            raise TimeoutError("elastic_tensor_receive_timeout")
        envelope = dict(lookup["envelope"])
        chunks = []
        if lookup.get("inline_payload") is True:
            try:
                chunk = base64.b64decode(
                    str(lookup.get("chunk_b64") or ""), validate=True
                )
            except (binascii.Error, ValueError) as exc:
                raise RuntimeError("elastic_inline_tensor_payload_invalid") from exc
            if (
                int(envelope["chunk_count"]) != 1
                or _sha(chunk) != envelope["chunk_hashes"][0]
            ):
                raise RuntimeError("elastic_tensor_chunk_response_hash_mismatch")
            chunks.append(chunk)
            self._tensor_download_count += 1
            self._inline_tensor_download_count += 1
        else:
            for index in range(int(envelope["chunk_count"])):
                raw, headers = self._assignment_request(
                    "/elastic-training/tensors/"
                    f"{urllib.parse.quote(envelope['message_id'], safe='')}/{index}",
                    assignment=assignment,
                )
                if headers.get("x-crowdtensor-tensor-chunk-hash") != envelope[
                    "chunk_hashes"
                ][index]:
                    raise RuntimeError("elastic_tensor_chunk_response_hash_mismatch")
                chunks.append(raw)
                self._tensor_download_count += 1
        tensors = decode_tensor_payload(
            b"".join(chunks),
            envelope,
            target_device=target_device,
            target_dtype=target_dtype,
        )
        return tensors, {
            "message_id": envelope["message_id"],
            "payload_hash": envelope["payload_hash"],
            "payload_bytes": int(envelope["payload_bytes"]),
            "chunk_count": len(chunks),
            "indexed_lookup_enabled": lookup.get("indexed_lookup_enabled") is True,
            "inline_transport_used": lookup.get("inline_payload") is True,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }

    def heartbeat(self) -> dict[str, Any]:
        self._require_registered()
        return self._json_request(
            f"/elastic-training/miners/{urllib.parse.quote(self._session_id)}/heartbeat",
            method="POST",
            session=True,
        )

    def assignments(self) -> dict[str, Any]:
        self._raise_heartbeat_error()
        self._require_registered()
        return self._json_request(
            f"/elastic-training/miners/{urllib.parse.quote(self._session_id)}/assignments",
            session=True,
        )

    def wait_for_assignments(
        self,
        *,
        expected_stage_ids: list[int] | None,
        timeout: float,
        expected_base_step: int | None = None,
        expected_assignment_count: int | None = None,
        allowed_stage_groups: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        expected = (
            sorted({int(value) for value in expected_stage_ids})
            if expected_stage_ids is not None
            else None
        )
        allowed = {
            tuple(sorted({int(value) for value in group}))
            for group in (allowed_stage_groups or [])
        }
        deadline = time.monotonic() + float(timeout)
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            self._raise_heartbeat_error()
            last = self.heartbeat()
            assignments = list(last.get("assignments") or [])
            stage_ids = sorted(int(item.get("stage_id", -1)) for item in assignments)
            base_steps = {int(item.get("base_step", -1)) for item in assignments}
            stage_match = (
                stage_ids == expected
                if expected is not None
                else (
                    (expected_assignment_count is None or len(stage_ids) == expected_assignment_count)
                    and (not allowed or tuple(stage_ids) in allowed)
                )
            )
            if stage_match and (
                expected_base_step is None or base_steps == {int(expected_base_step)}
            ):
                return last
            if last.get("runtime_state") == "completed":
                raise RuntimeError("elastic_training_already_completed")
            time.sleep(0.5)
        raise TimeoutError("elastic_stage_assignment_wait_timeout")

    def start_heartbeat(self) -> None:
        self._require_registered()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_error = None
        self._heartbeat_failure_count = 0

        def run() -> None:
            while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
                try:
                    self._require_registered()
                    self._json_request(
                        f"/elastic-training/miners/{urllib.parse.quote(self._session_id)}/heartbeat",
                        method="POST",
                        session=True,
                        timeout=min(10.0, self.timeout),
                        retry_attempts=1,
                    )
                except BaseException as exc:
                    self._heartbeat_failure_count += 1
                    if self._heartbeat_failure_count >= max(
                        12, self.retry_attempts * 2
                    ):
                        self._heartbeat_error = exc
                    continue
                if self._heartbeat_failure_count:
                    self._heartbeat_recovery_count += 1
                self._heartbeat_failure_count = 0
                self._heartbeat_error = None

        self._heartbeat_thread = threading.Thread(
            target=run,
            name="elastic-training-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(2.0, self.timeout + 1.0))
        self._heartbeat_thread = None

    @staticmethod
    def _assignment_headers(
        *,
        session_id: str,
        session_token: str,
        assignment_token: str,
    ) -> dict[str, str]:
        return {
            "x-crowdtensor-elastic-session-id": session_id,
            "x-crowdtensor-elastic-session-token": session_token,
            "x-crowdtensor-elastic-assignment-token": assignment_token,
        }

    def download_checkpoint(
        self,
        assignment: dict[str, Any],
        *,
        checkpoint_dir: str | Path,
        training_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._raise_heartbeat_error()
        self._require_registered()
        if assignment.get("restore_required") is not True:
            raise ValueError("elastic_checkpoint_restore_not_required")
        epoch_id = int(assignment["epoch_id"])
        stage_id = int(assignment["stage_id"])
        raw, headers = self._request(
            f"/elastic-training/checkpoints/{epoch_id}/{stage_id}",
            headers=self._assignment_headers(
                session_id=self._session_id,
                session_token=self._session_token,
                assignment_token=str(assignment["assignment_token"]),
            ),
        )
        expected_hash = str(
            assignment.get("committed_checkpoint_archive_hash") or ""
        )
        if _sha(raw) != expected_hash or headers.get(
            "x-crowdtensor-checkpoint-hash", ""
        ) != expected_hash:
            raise RuntimeError("elastic_downloaded_checkpoint_hash_mismatch")
        if training_manifest is not None:
            report = restore_stage_checkpoint_archive(
                raw,
                checkpoint_dir,
                training_manifest=training_manifest,
                expected_stage_id=stage_id,
                expected_step=int(assignment["base_step"]),
                expected_dataset_cursor=int(assignment["base_dataset_cursor"]),
            )
        else:
            report = restore_qwen_stage_checkpoint_archive(
                raw,
                checkpoint_dir,
                expected_stage_id=stage_id,
                expected_step=int(assignment["base_step"]),
                expected_dataset_cursor=int(assignment["base_dataset_cursor"]),
                validate_tensor_payloads=bool(
                    self._registration.get("checkpoint_tensor_validation_required")
                ),
            )
        self._checkpoint_download_count += 1
        return report

    def submit_checkpoint(
        self,
        assignment: dict[str, Any],
        *,
        checkpoint_dir: str | Path,
        training_manifest: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._raise_heartbeat_error()
        self._require_registered()
        stage_id = int(assignment["stage_id"])
        epoch_id = int(assignment["epoch_id"])
        if training_manifest is not None:
            archive, archive_report = build_stage_checkpoint_archive(
                checkpoint_dir,
                training_manifest=training_manifest,
                stage_id=stage_id,
            )
        else:
            archive, archive_report = build_qwen_stage_checkpoint_archive(
                checkpoint_dir, stage_id=stage_id
            )
        assignment_token = str(assignment["assignment_token"])
        signature = sign_checkpoint_submission(
            session_token=self._session_token,
            run_id=self.run_id,
            session_id=self._session_id,
            epoch_id=epoch_id,
            stage_id=stage_id,
            assignment_token=assignment_token,
            archive_hash=str(archive_report["archive_hash"]),
        )
        raw, _headers = self._request(
            f"/elastic-training/checkpoints/{epoch_id}/{stage_id}",
            method="POST",
            body=archive,
            headers={
                **self._assignment_headers(
                    session_id=self._session_id,
                    session_token=self._session_token,
                    assignment_token=assignment_token,
                ),
                "Content-Type": "application/vnd.crowdtensor.stage-checkpoint+zip",
                "x-crowdtensor-checkpoint-signature": signature,
            },
        )
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("elastic_checkpoint_submission_response_invalid") from exc
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError("elastic_checkpoint_submission_response_invalid")
        self._checkpoint_upload_count += 1
        if response.get("global_commit_created") is True:
            self._barrier_commit_count += 1
        return response, archive_report

    def barrier_status(self, *, epoch_id: int) -> dict[str, Any]:
        self._raise_heartbeat_error()
        self._require_registered()
        raw, _headers = self._request(
            f"/elastic-training/barriers/{int(epoch_id)}",
            headers={
                "x-crowdtensor-elastic-session-id": self._session_id,
                "x-crowdtensor-elastic-session-token": self._session_token,
            },
        )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("elastic_barrier_response_invalid") from exc
        if not isinstance(value, dict):
            raise RuntimeError("elastic_barrier_response_invalid")
        return value

    def wait_barrier(self, *, epoch_id: int, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            value = self.barrier_status(epoch_id=epoch_id)
            state = str(value.get("state") or "")
            if state == "committed":
                return value
            if state == "aborted":
                raise RuntimeError("elastic_barrier_epoch_aborted")
            time.sleep(0.25)
        raise TimeoutError("elastic_barrier_wait_timeout")

    def offline(self) -> dict[str, Any]:
        self.stop_heartbeat()
        self._require_registered()
        result = self._json_request(
            f"/elastic-training/miners/{urllib.parse.quote(self._session_id)}/offline",
            method="POST",
            session=True,
        )
        self.close()
        return result

    def close(self) -> None:
        with self._httpx_lock:
            if self._httpx_client is not None:
                self._httpx_client.close()
                self._httpx_client = None

    def report_device_telemetry(
        self,
        *,
        device_id: str,
        free_memory_bytes: int,
        utilization_fraction: float,
        throughput_units_per_second: float,
        network_bandwidth_bytes_per_second: float = 0.0,
        network_latency_ms: float = 0.0,
        checkpoint_step: int = 0,
        health_score: float = 1.0,
    ) -> dict[str, Any]:
        self._raise_heartbeat_error()
        self._require_registered()
        return self._json_request(
            f"/elastic-training/miners/{urllib.parse.quote(self._session_id)}/telemetry",
            method="POST",
            session=True,
            value={
                "device_id": str(device_id),
                "free_memory_bytes": int(free_memory_bytes),
                "utilization_fraction": float(utilization_fraction),
                "throughput_units_per_second": float(
                    throughput_units_per_second
                ),
                "network_bandwidth_bytes_per_second": float(
                    network_bandwidth_bytes_per_second
                ),
                "network_latency_ms": float(network_latency_ms),
                "checkpoint_step": int(checkpoint_step),
                "health_score": float(health_score),
            },
        )

    def report_device_failure(
        self,
        *,
        device_id: str,
        failure_class: str,
        quarantine_threshold: int = 3,
        quarantine_seconds: float = 300.0,
    ) -> dict[str, Any]:
        self._require_registered()
        return self._json_request(
            f"/elastic-training/miners/{urllib.parse.quote(self._session_id)}/device-failure",
            method="POST",
            session=True,
            value={
                "device_id": str(device_id),
                "failure_class": str(failure_class),
                "quarantine_threshold": int(quarantine_threshold),
                "quarantine_seconds": float(quarantine_seconds),
            },
        )

    def public_report(self) -> dict[str, Any]:
        return {
            "schema": "crowdtensor_elastic_training_client_report_v1",
            "registered": bool(self._session_id),
            "miner_id_hash": self.miner_id_hash,
            "miner_session_hash": _sha(self._session_id) if self._session_id else "",
            "supported_stage_ids": list(self.supported_stage_ids),
            "slot_count": self.slot_count,
            "accelerator": self.accelerator,
            "capability_hash": str(self.capability.get("content_hash") or ""),
            "request_count": self._request_count,
            "retry_count": self._retry_count,
            "bounded_retry_attempts": self.retry_attempts,
            "exponential_backoff_with_deterministic_jitter": True,
            "retry_delay_cap_seconds": 5.0,
            "persistent_http_after_step": self.persistent_http_after_step,
            "persistent_http_max_body_bytes": self.persistent_http_max_body_bytes,
            "persistent_http_active": self._persistent_transport_enabled(),
            "persistent_http_request_count": self._persistent_request_count,
            "legacy_http_request_count": self._legacy_request_count,
            "persistent_transport_fallback_count": self._persistent_transport_fallback_count,
            "large_payload_connection_isolation_count": self._large_payload_connection_isolation_count,
            "large_payload_connection_isolation_enabled": True,
            "connection_pool_reuse_enabled": self.persistent_http_after_step >= 0,
            "checkpoint_upload_count": self._checkpoint_upload_count,
            "checkpoint_download_count": self._checkpoint_download_count,
            "barrier_commit_response_count": self._barrier_commit_count,
            "tensor_chunk_upload_count": self._tensor_upload_count,
            "tensor_chunk_download_count": self._tensor_download_count,
            "inline_tensor_message_upload_count": self._inline_tensor_upload_count,
            "inline_tensor_message_download_count": self._inline_tensor_download_count,
            "inline_tensor_transport_enabled": self.persistent_http_after_step >= 0,
            "stage_profile_report_count": self._stage_profile_count,
            "heartbeat_failure_count": self._heartbeat_failure_count,
            "heartbeat_recovery_count": self._heartbeat_recovery_count,
            "checkpoint_signatures_sent": self._checkpoint_upload_count,
            "checkpoint_tensor_validation_required": bool(
                self._registration.get("checkpoint_tensor_validation_required")
            ),
            "heartbeat_failed": self._heartbeat_error is not None,
            "coordinator_url_public": False,
            "coordinator_token_public": False,
            "session_id_public": False,
            "session_token_public": False,
            "registration_nonce_public": False,
            "assignment_tokens_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
