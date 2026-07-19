"""Persistent two-stage Adapter runtime for bounded Kaggle reliability gates."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import Request, Response

from .adapter_stage_training import (
    configure_adapter_stage_model,
    owned_lora_state as _owned_lora_state,
    tensor_state_hash as _tensor_state_hash,
)
from .model_adapter import SmolLMModelAdapter, get_model_adapter, stable_hash


MODEL_ID = SmolLMModelAdapter.default_model_id
MODEL_REVISION = SmolLMModelAdapter.default_revision


STATE_SCHEMA = "crowdtensor_community_live_private_state_v1"
STATUS_SCHEMA = "crowdtensor_community_live_status_v1"
WORKER_REPORT_SCHEMA = "crowdtensor_community_live_worker_v1"
API_SCHEMA = "crowdtensor_community_live_api_v1"
CHECKPOINT_SCHEMA = "crowdtensor_community_live_stage_checkpoint_v1"


def _sha256(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(4) + ".tmp")
    try:
        temporary.write_text(json.dumps(value, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_private(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
        raise RuntimeError("community_live_private_state_invalid")
    return value


def _tensor_bytes(value: Any) -> bytes:
    from safetensors.torch import save

    return save({"value": value.detach().cpu().contiguous()})


def _tensor_from_bytes(value: bytes) -> Any:
    from safetensors.torch import load

    return load(value)["value"]


def _encode_tensor(value: Any) -> tuple[str, str]:
    payload = _tensor_bytes(value)
    return base64.b64encode(payload).decode("ascii"), _sha256(payload)


def _decode_tensor(value: str, expected_hash: str) -> Any:
    payload = base64.b64decode(value, validate=True)
    if _sha256(payload) != expected_hash:
        raise RuntimeError("community_live_tensor_payload_hash_invalid")
    return _tensor_from_bytes(payload)


class CommunityLiveCoordinator:
    """Atomic finite-state journal; private tensors never enter public status."""

    def __init__(
        self,
        state_path: str | Path,
        *,
        run_id: str,
        target_steps: int = 100,
        sequence_length: int = 8,
        lease_seconds: float = 120.0,
        checkpoint_steps: tuple[int, ...] = (30, 50, 100),
        model_adapter_id: str = SmolLMModelAdapter.adapter_id,
        model_id: str = "",
        model_revision: str = "",
        model_config: dict[str, Any] | None = None,
        split_index: int = 0,
    ) -> None:
        self.path = Path(state_path).expanduser().resolve()
        self.lock = threading.RLock()
        if self.path.is_file():
            self.state = _read_private(self.path)
            if self.state.get("run_id") != run_id:
                raise RuntimeError("community_live_run_id_mismatch")
        else:
            if target_steps < 1 or target_steps > 100:
                raise ValueError("community_live_target_steps_invalid")
            now = time.time()
            adapter = get_model_adapter(model_adapter_id)
            selected_config = dict(model_config or adapter.canonical_config())
            canonical = adapter.validate_config(selected_config)
            stages = adapter.partition(canonical, stage_count=2)
            selected_split = int(split_index or stages[0].layer_end)
            if selected_split != int(stages[0].layer_end):
                raise ValueError("community_live_split_index_not_adapter_partition")
            self.state = {
                "schema": STATE_SCHEMA,
                "run_id": str(run_id),
                "target_steps": int(target_steps),
                "sequence_length": int(sequence_length),
                "lease_seconds": float(lease_seconds),
                "checkpoint_steps": sorted({int(item) for item in checkpoint_steps}),
                "model_adapter_id": adapter.adapter_id,
                "model_id": str(model_id or adapter.default_model_id),
                "model_revision": str(model_revision or adapter.default_revision),
                "model_config": selected_config,
                "split_index": selected_split,
                "committed_step": 0,
                "phase": "stage0_forward",
                "coordinator_generation": 1,
                "workers": {},
                "leases": {},
                "current": {},
                "checkpoints": {},
                "ledger": [],
                "events": [
                    {"sequence": 1, "operation": "coordinator_started", "generation": 1, "recorded_at": now}
                ],
                "started_at": now,
                "completed_at": 0.0,
                "duplicate_or_stale_rejections": 0,
                "restart_requested_after_step": 0,
                "claims_paused_for_restart": False,
                "public_artifact": False,
            }
            self._persist()

    def _persist(self) -> None:
        _write_private(self.path, self.state)

    def _event(self, operation: str, **fields: Any) -> None:
        self.state["events"].append(
            {
                "sequence": len(self.state["events"]) + 1,
                "operation": operation,
                "recorded_at": time.time(),
                **fields,
            }
        )

    def record_restart(self) -> dict[str, Any]:
        with self.lock:
            before = int(self.state["coordinator_generation"])
            self.state["coordinator_generation"] = before + 1
            self.state["leases"] = {}
            self.state["restart_requested_after_step"] = 0
            self.state["claims_paused_for_restart"] = False
            self._event("coordinator_started", generation=before + 1, restart=True)
            self._persist()
            return {
                "generation_before": before,
                "generation_after": before + 1,
                "journal_recovered": True,
                "committed_step": int(self.state["committed_step"]),
            }

    def _activate_restart_barrier(self) -> bool:
        requested = int(self.state.get("restart_requested_after_step") or 0)
        ready = bool(
            requested
            and int(self.state["committed_step"]) >= requested
            and self.state["phase"] == "stage0_forward"
            and not self.state["leases"]
        )
        if ready:
            self.state["claims_paused_for_restart"] = True
        return bool(self.state.get("claims_paused_for_restart"))

    def request_restart_barrier(self, *, after_step: int) -> dict[str, Any]:
        if after_step < 1 or after_step > int(self.state["target_steps"]):
            raise ValueError("community_live_restart_barrier_step_invalid")
        with self.lock:
            existing = int(self.state.get("restart_requested_after_step") or 0)
            if not existing:
                self.state["restart_requested_after_step"] = int(after_step)
                self._event("coordinator_restart_requested", after_step=int(after_step))
            elif existing != int(after_step):
                raise RuntimeError("community_live_restart_barrier_conflict")
            ready = self._activate_restart_barrier()
            self._persist()
            return {
                "requested_after_step": int(after_step),
                "ready": ready,
                "committed_step": int(self.state["committed_step"]),
                "phase": str(self.state["phase"]),
            }

    def register(self, *, worker_id_hash: str, role: str, backend: str) -> dict[str, Any]:
        if role not in {"stage0", "stage1"} or backend not in {"cpu", "cuda"}:
            raise ValueError("community_live_worker_capability_invalid")
        if not str(worker_id_hash).startswith("sha256:"):
            raise ValueError("community_live_worker_identity_invalid")
        with self.lock:
            previous = dict(self.state["workers"].get(role) or {})
            generation = int(previous.get("generation") or 0) + 1
            replacement = bool(previous and previous.get("worker_id_hash") != worker_id_hash)
            self.state["workers"][role] = {
                "worker_id_hash": worker_id_hash,
                "role": role,
                "backend": backend,
                "generation": generation,
                "registered_at": time.time(),
                "last_seen": time.time(),
                "replacement": replacement,
            }
            self._event(
                "worker_registered",
                role=role,
                backend=backend,
                worker_id_hash=worker_id_hash,
                worker_generation=generation,
                replacement=replacement,
                committed_step=int(self.state["committed_step"]),
            )
            self._persist()
            checkpoint = dict(self.state["checkpoints"].get(role) or {})
            return {
                "schema": API_SCHEMA,
                "ok": True,
                "run_id": self.state["run_id"],
                "role": role,
                "worker_generation": generation,
                "coordinator_generation": int(self.state["coordinator_generation"]),
                "target_steps": int(self.state["target_steps"]),
                "sequence_length": int(self.state["sequence_length"]),
                "model_adapter_id": str(
                    self.state.get("model_adapter_id") or SmolLMModelAdapter.adapter_id
                ),
                "model_id": str(self.state.get("model_id") or MODEL_ID),
                "model_revision": str(self.state.get("model_revision") or MODEL_REVISION),
                "model_config": dict(
                    self.state.get("model_config") or SmolLMModelAdapter.default_config
                ),
                "split_index": int(self.state.get("split_index") or 15),
                "checkpoint": checkpoint,
                "checkpoint_payload_public": False,
                "public_artifact": False,
            }

    def _expire_leases(self) -> None:
        now = time.time()
        expired = [lease for lease, value in self.state["leases"].items() if float(value["expires_at"]) <= now]
        for lease in expired:
            self.state["leases"].pop(lease, None)
            self._event("lease_expired", lease_hash=_sha256(lease))

    def _input(self, step: int) -> list[int]:
        length = int(self.state["sequence_length"])
        config = dict(self.state.get("model_config") or SmolLMModelAdapter.default_config)
        vocab = int(config.get("vocab_size") or 0)
        if vocab < 4:
            raise RuntimeError("community_live_model_vocab_invalid")
        return [int((step * 97 + index * 31 + 7) % vocab) for index in range(length)]

    def claim(
        self,
        *,
        worker_id_hash: str,
        role: str,
        generation: int,
        max_committed_step: int = 0,
    ) -> dict[str, Any]:
        with self.lock:
            self._expire_leases()
            worker = dict(self.state["workers"].get(role) or {})
            if worker.get("worker_id_hash") != worker_id_hash or int(worker.get("generation") or 0) != int(generation):
                raise RuntimeError("community_live_worker_generation_stale")
            worker["last_seen"] = time.time()
            self.state["workers"][role] = worker
            committed = int(self.state["committed_step"])
            stop_after = int(max_committed_step or 0)
            if stop_after < 0:
                raise ValueError("community_live_max_committed_step_invalid")
            if stop_after and committed >= stop_after:
                self._persist()
                return {
                    "schema": API_SCHEMA,
                    "task_available": False,
                    "completed": False,
                    "worker_stop_requested": True,
                    "committed_step": committed,
                }
            if committed >= int(self.state["target_steps"]):
                self._persist()
                return {"schema": API_SCHEMA, "task_available": False, "completed": True, "committed_step": committed}
            if self._activate_restart_barrier():
                self._persist()
                return {
                    "schema": API_SCHEMA,
                    "task_available": False,
                    "completed": False,
                    "committed_step": committed,
                    "restart_barrier": True,
                }
            phase = str(self.state["phase"])
            eligible = bool(
                (phase == "stage0_forward" and role == "stage0")
                or (phase == "stage1_backward" and role == "stage1")
                or (phase == "stage0_backward" and role == "stage0")
                or (phase == "commit" and role not in (self.state["current"].get("commit_acks") or {}))
            )
            if not eligible or any(value["role"] == role for value in self.state["leases"].values()):
                self._persist()
                return {"schema": API_SCHEMA, "task_available": False, "completed": False, "committed_step": committed}
            step = committed + 1
            lease = secrets.token_urlsafe(24)
            task: dict[str, Any] = {
                "schema": API_SCHEMA,
                "task_available": True,
                "lease": lease,
                "lease_hash": _sha256(lease),
                "step": step,
                "phase": phase,
                "checkpoint_required": step in self.state["checkpoint_steps"],
                "coordinator_generation": int(self.state["coordinator_generation"]),
                "worker_generation": int(generation),
                "private_payload": True,
            }
            if phase in {"stage0_forward", "stage0_backward"}:
                token_ids = self._input(step)
                task.update({"input_ids": token_ids, "labels": token_ids, "attention_mask": [1] * len(token_ids)})
            if phase == "stage1_backward":
                task.update(
                    {
                        "activation_b64": self.state["current"]["activation_b64"],
                        "activation_hash": self.state["current"]["activation_hash"],
                        "labels": self._input(step),
                        "attention_mask": [1] * int(self.state["sequence_length"]),
                    }
                )
            if phase == "stage0_backward":
                task.update(
                    {
                        "gradient_b64": self.state["current"]["gradient_b64"],
                        "gradient_hash": self.state["current"]["gradient_hash"],
                    }
                )
            self.state["leases"][lease] = {
                "role": role,
                "worker_id_hash": worker_id_hash,
                "worker_generation": int(generation),
                "coordinator_generation": int(self.state["coordinator_generation"]),
                "step": step,
                "phase": phase,
                "expires_at": time.time() + float(self.state["lease_seconds"]),
            }
            self._persist()
            return task

    def submit(self, *, worker_id_hash: str, lease: str, value: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            record = self.state["leases"].pop(str(lease), None)
            if not record or record.get("worker_id_hash") != worker_id_hash:
                self.state["duplicate_or_stale_rejections"] += 1
                self._persist()
                raise RuntimeError("community_live_duplicate_or_stale_result")
            step = int(record["step"])
            phase = str(record["phase"])
            if step != int(self.state["committed_step"]) + 1 or phase != self.state["phase"]:
                self.state["duplicate_or_stale_rejections"] += 1
                self._persist()
                raise RuntimeError("community_live_result_phase_stale")
            current = self.state["current"]
            if phase == "stage0_forward":
                payload = str(value.get("activation_b64") or "")
                digest = str(value.get("activation_hash") or "")
                _decode_tensor(payload, digest)
                current.update(
                    {
                        "activation_b64": payload,
                        "activation_hash": digest,
                        "activation_bytes": len(
                            base64.b64decode(payload, validate=True)
                        ),
                    }
                )
                self.state["phase"] = "stage1_backward"
            elif phase == "stage1_backward":
                payload = str(value.get("gradient_b64") or "")
                digest = str(value.get("gradient_hash") or "")
                _decode_tensor(payload, digest)
                loss = float(value.get("loss"))
                if not math.isfinite(loss):
                    raise RuntimeError("community_live_non_finite_loss")
                current.update(
                    {
                        "gradient_b64": payload,
                        "gradient_hash": digest,
                        "gradient_bytes": len(
                            base64.b64decode(payload, validate=True)
                        ),
                        "loss": loss,
                    }
                )
                self.state["phase"] = "stage0_backward"
            elif phase == "stage0_backward":
                if value.get("gradient_ready") is not True:
                    raise RuntimeError("community_live_stage0_gradient_not_ready")
                current["commit_acks"] = {}
                self.state["phase"] = "commit"
            elif phase == "commit":
                role = str(record["role"])
                checkpoint = value.get("checkpoint") if isinstance(value.get("checkpoint"), dict) else {}
                if step in self.state["checkpoint_steps"]:
                    payload = str(checkpoint.get("payload_b64") or "")
                    digest = str(checkpoint.get("payload_hash") or "")
                    raw = base64.b64decode(payload, validate=True)
                    if _sha256(raw) != digest:
                        raise RuntimeError("community_live_checkpoint_hash_invalid")
                    self.state["checkpoints"][role] = {
                        "schema": CHECKPOINT_SCHEMA,
                        "step": step,
                        "payload_b64": payload,
                        "payload_hash": digest,
                        "adapter_hash": str(checkpoint.get("adapter_hash") or ""),
                        "payload_public": False,
                    }
                    self._event(
                        "checkpoint_committed",
                        role=role,
                        step=step,
                        payload_bytes=len(raw),
                    )
                current.setdefault("commit_acks", {})[role] = {
                    "worker_id_hash": worker_id_hash,
                    "worker_generation": int(record["worker_generation"]),
                    "adapter_hash": str(value.get("adapter_hash") or ""),
                }
                if set(current["commit_acks"]) == {"stage0", "stage1"}:
                    self.state["ledger"].append(
                        {
                            "step": step,
                            "loss": float(current["loss"]),
                            "activation_hash": current["activation_hash"],
                            "gradient_hash": current["gradient_hash"],
                            "activation_bytes": int(current["activation_bytes"]),
                            "gradient_bytes": int(current["gradient_bytes"]),
                            "commit_acks": current["commit_acks"],
                            "committed_at": time.time(),
                        }
                    )
                    self.state["committed_step"] = step
                    self.state["phase"] = "stage0_forward"
                    self.state["current"] = {}
                    if step >= int(self.state["target_steps"]):
                        self.state["completed_at"] = time.time()
                    self._event("step_committed", step=step)
                    self._activate_restart_barrier()
            else:
                raise RuntimeError("community_live_submit_phase_invalid")
            self._persist()
            return {
                "schema": API_SCHEMA,
                "ok": True,
                "accepted": True,
                "committed_step": int(self.state["committed_step"]),
                "phase": str(self.state["phase"]),
                "public_artifact_safe": True,
            }

    def private_checkpoint(self, role: str) -> dict[str, Any]:
        with self.lock:
            return dict(self.state["checkpoints"].get(role) or {})

    def public_status(self) -> dict[str, Any]:
        with self.lock:
            workers = [
                {
                    "role": role,
                    "backend": item["backend"],
                    "worker_id_hash": item["worker_id_hash"],
                    "generation": int(item["generation"]),
                    "replacement": bool(item.get("replacement")),
                }
                for role, item in sorted(self.state["workers"].items())
            ]
            ledger = list(self.state["ledger"])
            checkpoint_summary = {
                role: {
                    "step": int(item["step"]),
                    "payload_hash": item["payload_hash"],
                    "adapter_hash": item["adapter_hash"],
                    "payload_public": False,
                }
                for role, item in sorted(self.state["checkpoints"].items())
            }
            events = [
                {key: value for key, value in item.items() if key not in {"lease", "payload_b64"}}
                for item in self.state["events"]
            ]
            report = {
                "schema": STATUS_SCHEMA,
                "ok": True,
                "run_id_hash": _sha256(self.state["run_id"]),
                "model_adapter_id": str(
                    self.state.get("model_adapter_id") or SmolLMModelAdapter.adapter_id
                ),
                "model_id": str(self.state.get("model_id") or MODEL_ID),
                "model_revision": str(self.state.get("model_revision") or MODEL_REVISION),
                "target_steps": int(self.state["target_steps"]),
                "committed_step": int(self.state["committed_step"]),
                "committed_step_ids": [int(item["step"]) for item in ledger],
                "strictly_contiguous_steps": [int(item["step"]) for item in ledger]
                == list(range(1, len(ledger) + 1)),
                "phase": str(self.state["phase"]),
                "coordinator_generation": int(self.state["coordinator_generation"]),
                "workers": workers,
                "checkpoint_summary": checkpoint_summary,
                "finite_losses": all(math.isfinite(float(item["loss"])) for item in ledger),
                "ledger_entry_count": len(ledger),
                "duplicate_or_stale_rejections": int(self.state["duplicate_or_stale_rejections"]),
                "restart_barrier_ready": bool(self.state.get("claims_paused_for_restart")),
                "events": events,
                "duration_seconds": (
                    float(self.state["completed_at"] or time.time()) - float(self.state["started_at"])
                ),
                "completed": int(self.state["committed_step"]) >= int(self.state["target_steps"]),
                "node_scope": "Kaggle logical multi-node",
                "physical_multi_machine_verified": False,
                "credential_values_public": False,
                "private_urls_public": False,
                "raw_training_text_public": False,
                "token_ids_public": False,
                "activation_values_public": False,
                "gradient_values_public": False,
                "checkpoint_tensor_values_public": False,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
            report["content_hash"] = stable_hash(report)
            return report


def create_live_app(
    coordinator: CommunityLiveCoordinator,
    *,
    miner_token: str,
    wheel_path: str | Path,
    adapter_wheel_path: str | Path | None = None,
) -> Any:
    from fastapi import FastAPI, Header, HTTPException

    if not miner_token:
        raise ValueError("community_live_miner_token_required")
    wheel = Path(wheel_path).expanduser().resolve()
    adapter_wheel = (
        Path(adapter_wheel_path).expanduser().resolve()
        if adapter_wheel_path is not None
        else None
    )
    app = FastAPI(title="CrowdTensor Community Live", docs_url=None, redoc_url=None)

    def auth(value: str | None) -> None:
        if value is None or not hmac.compare_digest(value, miner_token):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"schema": API_SCHEMA, "ok": True, "public_artifact_safe": True}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"schema": API_SCHEMA, "ok": True, "ready": True, "public_artifact_safe": True}

    @app.get("/v1/community-live/status")
    def status(x_crowdtensor_miner_token: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_crowdtensor_miner_token)
        return coordinator.public_status()

    @app.get("/v1/community-live/wheel")
    def download_wheel(x_crowdtensor_miner_token: str | None = Header(default=None)) -> Response:
        auth(x_crowdtensor_miner_token)
        if not wheel.is_file():
            raise HTTPException(status_code=503, detail="wheel_unavailable")
        return Response(
            wheel.read_bytes(),
            media_type="application/octet-stream",
            headers={
                "x-crowdtensor-wheel-sha256": _sha256(wheel.read_bytes()),
                "x-crowdtensor-wheel-filename": wheel.name,
            },
        )

    @app.get("/v1/community-live/adapter-wheel")
    def download_adapter_wheel(
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> Response:
        auth(x_crowdtensor_miner_token)
        if adapter_wheel is None or not adapter_wheel.is_file():
            raise HTTPException(status_code=503, detail="adapter_wheel_unavailable")
        payload = adapter_wheel.read_bytes()
        return Response(
            payload,
            media_type="application/octet-stream",
            headers={
                "x-crowdtensor-wheel-sha256": _sha256(payload),
                "x-crowdtensor-wheel-filename": adapter_wheel.name,
                "x-crowdtensor-wheel-kind": "model-adapter-plugin",
            },
        )

    @app.post("/v1/community-live/register")
    async def register(request: Request, x_crowdtensor_miner_token: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_crowdtensor_miner_token)
        value = await request.json()
        try:
            return coordinator.register(
                worker_id_hash=str(value.get("worker_id_hash") or ""),
                role=str(value.get("role") or ""),
                backend=str(value.get("backend") or ""),
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/community-live/claim")
    async def claim(request: Request, x_crowdtensor_miner_token: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_crowdtensor_miner_token)
        value = await request.json()
        try:
            return coordinator.claim(
                worker_id_hash=str(value.get("worker_id_hash") or ""),
                role=str(value.get("role") or ""),
                generation=int(value.get("worker_generation") or 0),
                max_committed_step=int(value.get("max_committed_step") or 0),
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/community-live/submit")
    async def submit(request: Request, x_crowdtensor_miner_token: str | None = Header(default=None)) -> dict[str, Any]:
        auth(x_crowdtensor_miner_token)
        value = await request.json()
        try:
            return coordinator.submit(
                worker_id_hash=str(value.get("worker_id_hash") or ""),
                lease=str(value.get("lease") or ""),
                value=dict(value.get("result") or {}),
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    app.state.community_live_coordinator = coordinator
    return app


def _request_json(
    method: str,
    url: str,
    *,
    token: str,
    value: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = json.dumps(value).encode() if value is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "x-crowdtensor-miner-token": token,
            "User-Agent": "crowdtensor-community-live-worker/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        raise RuntimeError(f"community_live_http_{exc.code}:{detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("community_live_http_response_invalid")
    return payload


def _checkpoint_bytes(model: Any, optimizer: Any, *, start: int, end: int, step: int) -> tuple[bytes, str]:
    from safetensors.torch import save

    adapter = _owned_lora_state(model, start=start, end=end)
    tensors = {"adapter." + key: value for key, value in adapter.items()}
    trainable = sorted(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )
    for index, (_name, parameter) in enumerate(trainable):
        state = optimizer.state.get(parameter) or {}
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                tensors[f"optimizer.{index}.{key}"] = state[key].detach().cpu().contiguous()
        scalar = state.get("step")
        if scalar is not None:
            import torch

            tensors[f"optimizer.{index}.step"] = torch.as_tensor(scalar).detach().cpu().reshape(1)
    payload = save(
        tensors,
        metadata={
            "schema": CHECKPOINT_SCHEMA,
            "step": str(int(step)),
            "trainable_parameter_count": str(len(trainable)),
        },
    )
    return payload, _tensor_state_hash(adapter)


def _restore_checkpoint(model: Any, optimizer: Any, payload: bytes, *, start: int, end: int) -> dict[str, Any]:
    from peft import set_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import load

    tensors = load(payload)
    adapter = {key[len("adapter."):]: value for key, value in tensors.items() if key.startswith("adapter.")}
    set_peft_model_state_dict(model, adapter)
    trainable = sorted(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )
    for index, (_name, parameter) in enumerate(trainable):
        state: dict[str, Any] = {}
        for key in ("exp_avg", "exp_avg_sq", "step"):
            name = f"optimizer.{index}.{key}"
            if name in tensors:
                item = tensors[name].to(parameter.device)
                state[key] = item.reshape(()) if key == "step" else item
        if state:
            optimizer.state[parameter] = state
    return {
        "checkpoint_restored": True,
        "adapter_hash": _tensor_state_hash(_owned_lora_state(model, start=start, end=end)),
        "optimizer_parameter_state_count": sum(bool(optimizer.state.get(item)) for _name, item in trainable),
    }


def run_remote_worker(
    *,
    coordinator_url: str,
    token: str,
    role: str,
    backend: str,
    output_path: str | Path,
    max_committed_step: int = 0,
    poll_seconds: float = 0.5,
    timeout_seconds: float = 2700.0,
    cache_dir: str | Path = "",
) -> dict[str, Any]:
    import torch

    if role not in {"stage0", "stage1"}:
        raise ValueError("community_live_role_invalid")
    device = "cuda:0" if backend == "cuda" else "cpu"
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("community_live_cuda_unavailable")
    started = time.monotonic()
    worker_id_hash = _sha256(secrets.token_bytes(32))
    registration = _request_json(
        "POST",
        coordinator_url.rstrip("/") + "/v1/community-live/register",
        token=token,
        value={"worker_id_hash": worker_id_hash, "role": role, "backend": backend},
    )
    model, causal, optimizer, start, end, stage_details = configure_adapter_stage_model(
        adapter_id=str(registration["model_adapter_id"]),
        model_id=str(registration["model_id"]),
        model_revision=str(registration["model_revision"]),
        model_config=dict(registration["model_config"]),
        stage_id=0 if role == "stage0" else 1,
        split_index=int(registration["split_index"]),
        device=device,
        cache_dir=str(cache_dir),
        rank=8,
        alpha=16,
    )
    restored = {"checkpoint_restored": False, "adapter_hash": "", "optimizer_parameter_state_count": 0}
    checkpoint = dict(registration.get("checkpoint") or {})
    if checkpoint:
        raw = base64.b64decode(str(checkpoint["payload_b64"]), validate=True)
        if _sha256(raw) != checkpoint.get("payload_hash"):
            raise RuntimeError("community_live_registration_checkpoint_hash_invalid")
        restored = _restore_checkpoint(model, optimizer, raw, start=start, end=end)
        if restored["adapter_hash"] != checkpoint.get("adapter_hash"):
            raise RuntimeError("community_live_registration_adapter_hash_invalid")
    initial_hash = _tensor_state_hash(_owned_lora_state(model, start=start, end=end))
    worker_generation = int(registration["worker_generation"])
    step_events: list[dict[str, Any]] = []
    pending_gradient_step = 0
    last_committed = int(checkpoint.get("step") or 0)
    retry_count = 0
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            task = _request_json(
                "POST", coordinator_url.rstrip("/") + "/v1/community-live/claim",
                token=token,
                value={
                    "worker_id_hash": worker_id_hash,
                    "role": role,
                    "worker_generation": worker_generation,
                    "max_committed_step": int(max_committed_step or 0),
                },
            )
        except (OSError, RuntimeError):
            retry_count += 1
            time.sleep(min(5.0, poll_seconds * max(1, retry_count)))
            continue
        retry_count = 0
        last_committed = max(last_committed, int(task.get("committed_step") or 0))
        if task.get("completed") is True or (max_committed_step and last_committed >= max_committed_step):
            break
        if task.get("task_available") is not True:
            time.sleep(poll_seconds)
            continue
        phase = str(task["phase"])
        step = int(task["step"])
        result: dict[str, Any]
        if phase == "stage0_forward" and role == "stage0":
            ids = torch.tensor([task["input_ids"]], dtype=torch.long, device=device)
            mask = torch.tensor([task["attention_mask"]], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden = causal.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            payload, digest = _encode_tensor(hidden.float())
            result = {"activation_b64": payload, "activation_hash": digest}
        elif phase == "stage1_backward" and role == "stage1":
            optimizer.zero_grad(set_to_none=True)
            hidden = _decode_tensor(task["activation_b64"], task["activation_hash"]).to(
                device=device, dtype=next(causal.parameters()).dtype
            )
            hidden.requires_grad_(True)
            mask = torch.tensor([task["attention_mask"]], dtype=torch.long, device=device)
            labels = torch.tensor([task["labels"]], dtype=torch.long, device=device)
            final = causal.model(inputs_embeds=hidden, attention_mask=mask, use_cache=False).last_hidden_state
            logits = causal.lm_head(final).float()
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("community_live_worker_non_finite_loss")
            loss.backward()
            payload, digest = _encode_tensor(hidden.grad.float())
            pending_gradient_step = step
            result = {"gradient_b64": payload, "gradient_hash": digest, "loss": float(loss.detach().cpu())}
        elif phase == "stage0_backward" and role == "stage0":
            optimizer.zero_grad(set_to_none=True)
            ids = torch.tensor([task["input_ids"]], dtype=torch.long, device=device)
            mask = torch.tensor([task["attention_mask"]], dtype=torch.long, device=device)
            hidden = causal.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            gradient = _decode_tensor(task["gradient_b64"], task["gradient_hash"]).to(device=device, dtype=hidden.dtype)
            hidden.backward(gradient)
            pending_gradient_step = step
            result = {"gradient_ready": True}
        elif phase == "commit":
            if pending_gradient_step != step:
                raise RuntimeError("community_live_commit_without_pending_gradient")
            torch.nn.utils.clip_grad_norm_([item for item in model.parameters() if item.requires_grad], 1.0)
            optimizer.step()
            adapter_hash = _tensor_state_hash(_owned_lora_state(model, start=start, end=end))
            result = {"adapter_hash": adapter_hash}
            if task.get("checkpoint_required") is True:
                checkpoint_payload, adapter_hash = _checkpoint_bytes(
                    model, optimizer, start=start, end=end, step=step
                )
                result["checkpoint"] = {
                    "payload_b64": base64.b64encode(checkpoint_payload).decode("ascii"),
                    "payload_hash": _sha256(checkpoint_payload),
                    "adapter_hash": adapter_hash,
                }
            pending_gradient_step = 0
        else:
            raise RuntimeError("community_live_worker_task_role_mismatch")
        response = _request_json(
            "POST", coordinator_url.rstrip("/") + "/v1/community-live/submit",
            token=token,
            value={"worker_id_hash": worker_id_hash, "lease": task["lease"], "result": result},
            timeout=300,
        )
        last_committed = max(last_committed, int(response.get("committed_step") or 0))
        step_events.append(
            {
                "step": step,
                "phase": phase,
                "accepted": response.get("accepted") is True,
                "coordinator_generation": int(task["coordinator_generation"]),
                "worker_generation": worker_generation,
            }
        )
    else:
        raise RuntimeError("community_live_worker_timeout")
    final_hash = _tensor_state_hash(_owned_lora_state(model, start=start, end=end))
    report = {
        "schema": WORKER_REPORT_SCHEMA,
        "ok": True,
        "role": role,
        "backend": backend,
        "device_type": "cuda" if backend == "cuda" else "cpu",
        "worker_id_hash": worker_id_hash,
        "worker_generation": worker_generation,
        "model_adapter_id": str(registration["model_adapter_id"]),
        "model_id": str(registration["model_id"]),
        "model_revision": str(registration["model_revision"]),
        "stage_runtime": stage_details,
        "real_model_weights_loaded": True,
        "checkpoint_restored": restored["checkpoint_restored"],
        "restored_checkpoint_step": int(checkpoint.get("step") or 0),
        "optimizer_state_restored": bool(
            not checkpoint or int(restored["optimizer_parameter_state_count"]) > 0
        ),
        "adapter_hash_before": initial_hash,
        "adapter_hash_after": final_hash,
        "adapter_updated": initial_hash != final_hash,
        "outdated_optional_torchao_dispatch_disabled": bool(
            getattr(model, "_crowdtensor_outdated_optional_torchao_dispatch_disabled", False)
        ),
        "last_committed_step": last_committed,
        "step_events": step_events,
        "http_retry_count": retry_count,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "install_source": (
            "wheel" if os.environ.get("CROWDTENSOR_INSTALL_SOURCE") == "wheel" else "unknown"
        ),
        "workspace_import_used": False,
        "node_scope": "Kaggle logical multi-node",
        "raw_process_id_public": False,
        "coordinator_url_public": False,
        "credential_values_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
