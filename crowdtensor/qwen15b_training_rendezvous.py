"""Authenticated private rendezvous for the Qwen 1.5B four-GPU Alpha."""

import base64
import binascii
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable


RENDEZVOUS_SCHEMA = "crowdtensor_qwen15b_four_gpu_rendezvous_v1"
PAYLOAD_KINDS = {"activation", "gradient", "stage_adapter"}
RUN_KINDS = {"baseline", "resumed", "elastic"}
ROLES = {"kernel_a", "kernel_b"}
EVENT_OPERATIONS = {
    "stage_loaded",
    "forward",
    "forward_backward",
    "backward",
    "optimizer_step",
    "checkpoint",
    "stage_stopped",
    "stage_restarted",
    "evaluation",
    "export",
}


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


class Qwen15BTrainingRendezvous:
    """Keep training tensors private while exposing hash-only progress."""

    def __init__(
        self,
        *,
        run_id: str,
        max_payload_bytes: int = 64 * 1024 * 1024,
        state_path: str | Path | None = None,
    ) -> None:
        self.run_id = str(run_id)
        self.max_payload_bytes = int(max_payload_bytes)
        self._state_path = Path(state_path).resolve() if state_path else None
        self._lock = threading.RLock()
        self._registrations: dict[str, dict[str, Any]] = {}
        self._registration_history: list[dict[str, Any]] = []
        self._payloads: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._event_keys: set[str] = set()
        self._completions: dict[str, dict[str, Any]] = {}
        self._restart_records: list[dict[str, Any]] = []
        self._generation = 0
        self._recovered_from_persistent_state = False
        self._created_at = time.time()
        if self._state_path and self._state_path.is_file():
            self._restore()

    @staticmethod
    def _event_key(value: dict[str, Any]) -> str:
        fields = {
            key: value.get(key)
            for key in (
                "role",
                "run_kind",
                "operation",
                "stage_id",
                "step",
                "microbatch",
                "pid",
                "started_ns",
                "ended_ns",
                "checkpoint_hash",
                "adapter_hash",
            )
        }
        encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return _sha256_bytes(encoded)

    def _snapshot_locked(self) -> dict[str, Any]:
        payloads = []
        for record in self._payloads.values():
            public_record = {key: value for key, value in record.items() if key != "payload_b64"}
            public_record["payload_file"] = str(record["payload_hash"]).split(":", 1)[-1] + ".bin"
            payloads.append(public_record)
        return {
            "schema": "crowdtensor_qwen15b_private_rendezvous_state_v1",
            "run_id": self.run_id,
            "max_payload_bytes": self.max_payload_bytes,
            "created_at": self._created_at,
            "generation": self._generation,
            "registrations": self._registrations,
            "registration_history": self._registration_history,
            "payloads": payloads,
            "events": self._events,
            "completions": self._completions,
            "restart_records": self._restart_records,
            "public_artifact": False,
        }

    def _persist_locked(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload_dir = self._state_path.parent / f"{self._state_path.stem}-payloads"
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_dir.chmod(0o700)
        for record in self._payloads.values():
            payload_file = payload_dir / (
                str(record["payload_hash"]).split(":", 1)[-1] + ".bin"
            )
            if not payload_file.is_file():
                payload_file.write_bytes(
                    base64.b64decode(str(record["payload_b64"]).encode("ascii"), validate=True)
                )
                payload_file.chmod(0o600)
        temporary = self._state_path.with_name(
            f".{self._state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(self._snapshot_locked(), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, self._state_path)
        self._state_path.chmod(0o600)

    def _restore(self) -> None:
        assert self._state_path is not None
        value = json.loads(self._state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != "crowdtensor_qwen15b_private_rendezvous_state_v1"
            or value.get("run_id") != self.run_id
        ):
            raise ValueError("qwen15b_training_persistent_state_invalid")
        self._created_at = float(value.get("created_at") or self._created_at)
        self._generation = int(value.get("generation") or 0)
        self._registrations = {
            str(key): dict(item)
            for key, item in dict(value.get("registrations") or {}).items()
        }
        self._registration_history = [
            dict(item) for item in value.get("registration_history") or []
        ]
        self._payloads = {}
        payload_dir = self._state_path.parent / f"{self._state_path.stem}-payloads"
        for item in value.get("payloads") or []:
            record = dict(item)
            payload_file = payload_dir / str(record.pop("payload_file", ""))
            if not payload_file.is_file():
                raise ValueError("qwen15b_training_persistent_payload_missing")
            raw = payload_file.read_bytes()
            if _sha256_bytes(raw) != record.get("payload_hash"):
                raise ValueError("qwen15b_training_persistent_payload_hash_mismatch")
            record["payload_b64"] = base64.b64encode(raw).decode("ascii")
            key = (
                str(record["run_kind"]),
                str(record["kind"]),
                int(record["step"]),
                int(record["microbatch"]),
            )
            self._payloads[key] = record
        self._events = [dict(item) for item in value.get("events") or []]
        self._event_keys = {
            self._event_key(item)
            for item in self._events
            if item.get("role") in ROLES and item.get("run_kind") in RUN_KINDS
        }
        self._completions = {
            str(key): dict(item)
            for key, item in dict(value.get("completions") or {}).items()
        }
        self._restart_records = [dict(item) for item in value.get("restart_records") or []]
        self._recovered_from_persistent_state = True

    def begin_coordinator_restart(self, *, after_step: int) -> dict[str, Any]:
        with self._lock:
            if self._restart_records and not self._restart_records[-1].get("completed_at"):
                return dict(self._restart_records[-1])
            self._generation += 1
            record = {
                "generation": self._generation,
                "after_step": int(after_step),
                "started_at": time.time(),
                "completed_at": 0.0,
                "recovered_payload_count": len(self._payloads),
                "recovered_event_count": len(self._events),
            }
            self._restart_records.append(record)
            self._persist_locked()
            return dict(record)

    def complete_coordinator_restart(self) -> dict[str, Any]:
        with self._lock:
            if not self._restart_records:
                raise ValueError("qwen15b_training_restart_not_started")
            record = self._restart_records[-1]
            if not record.get("completed_at"):
                record["completed_at"] = time.time()
                record["duration_seconds"] = float(record["completed_at"]) - float(
                    record["started_at"]
                )
            self._persist_locked()
            return dict(record)

    def _require_run(self, run_id: str) -> None:
        if str(run_id) != self.run_id:
            raise ValueError("qwen15b_training_run_id_mismatch")

    @staticmethod
    def _require_role(role: str) -> None:
        if role not in ROLES:
            raise ValueError("qwen15b_training_role_invalid")

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_run(str(payload.get("run_id") or ""))
        role = str(payload.get("role") or "")
        self._require_role(role)
        stage_ids = sorted({int(value) for value in payload.get("stage_ids") or []})
        expected = [0, 1] if role == "kernel_a" else [2, 3]
        pids = [int(value) for value in payload.get("stage_pids") or []]
        devices = [str(value) for value in payload.get("cuda_devices") or []]
        if stage_ids != expected or len(pids) != 2 or any(value <= 0 for value in pids):
            raise ValueError("qwen15b_training_registration_incomplete")
        if devices != ["cuda:0", "cuda:1"]:
            raise ValueError("qwen15b_training_registration_device_mismatch")
        now = time.time()
        public = {
            "role": role,
            "worker_id_hash": str(payload.get("worker_id_hash") or ""),
            "stage_ids": stage_ids,
            "stage_pids": pids,
            "cuda_devices": devices,
            "cuda_device_name_hashes": [
                str(value) for value in payload.get("cuda_device_name_hashes") or []
            ],
            "cuda_live": payload.get("cuda_live") is True,
            "registered_at": now,
            "coordinator_generation": self._generation,
        }
        if not public["worker_id_hash"] or len(public["cuda_device_name_hashes"]) != 2:
            raise ValueError("qwen15b_training_registration_incomplete")
        with self._lock:
            self._registrations[role] = public
            history_key = (
                role,
                self._generation,
                tuple(public["stage_pids"]),
            )
            existing_keys = {
                (
                    str(item.get("role") or ""),
                    int(item.get("coordinator_generation") or 0),
                    tuple(item.get("stage_pids") or []),
                )
                for item in self._registration_history
            }
            if history_key not in existing_keys:
                self._registration_history.append(dict(public))
                self._events.append(
                    {
                        "operation": "registered",
                        "role": role,
                        "coordinator_generation": self._generation,
                        "at": now,
                    }
                )
            self._persist_locked()
        return {"ok": True, "schema": RENDEZVOUS_SCHEMA, "role": role}

    @staticmethod
    def _payload_key(payload: dict[str, Any]) -> tuple[str, str, int, int]:
        run_kind = str(payload.get("run_kind") or "")
        kind = str(payload.get("kind") or "")
        step = int(payload.get("step", -1))
        microbatch = int(payload.get("microbatch", -1))
        if run_kind not in RUN_KINDS or kind not in PAYLOAD_KINDS:
            raise ValueError("qwen15b_training_payload_identity_invalid")
        if step < 0 or (kind != "stage_adapter" and microbatch < 0):
            raise ValueError("qwen15b_training_payload_position_invalid")
        if kind == "stage_adapter" and microbatch != -1:
            raise ValueError("qwen15b_training_adapter_position_invalid")
        return run_kind, kind, step, microbatch

    def put_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_run(str(payload.get("run_id") or ""))
        role = str(payload.get("producer_role") or "")
        self._require_role(role)
        run_kind, kind, step, microbatch = self._payload_key(payload)
        expected_role = "kernel_a" if kind in {"activation", "stage_adapter"} else "kernel_b"
        if role != expected_role:
            raise ValueError("qwen15b_training_payload_producer_invalid")
        encoded = str(payload.get("payload_b64") or "")
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as exc:
            raise ValueError("qwen15b_training_payload_base64_invalid") from exc
        if not raw or len(raw) > self.max_payload_bytes:
            raise ValueError("qwen15b_training_payload_size_invalid")
        actual_hash = _sha256_bytes(raw)
        if actual_hash != str(payload.get("payload_hash") or ""):
            raise ValueError("qwen15b_training_payload_hash_mismatch")
        now = time.time()
        record = {
            "run_kind": run_kind,
            "kind": kind,
            "step": step,
            "microbatch": microbatch,
            "producer_role": role,
            "payload_b64": encoded,
            "payload_hash": actual_hash,
            "byte_count": len(raw),
            "tensor_count": int(payload.get("tensor_count") or 0),
            "created_at": now,
        }
        key = (run_kind, kind, step, microbatch)
        with self._lock:
            previous = self._payloads.get(key)
            if previous and previous["payload_hash"] != actual_hash:
                raise ValueError("qwen15b_training_payload_conflict")
            self._payloads[key] = record
            if previous is None:
                self._events.append(
                    {
                        "operation": f"{kind}_available",
                        "run_kind": run_kind,
                        "step": step,
                        "microbatch": microbatch,
                        "payload_hash": actual_hash,
                        "byte_count": len(raw),
                        "at": now,
                    }
                )
            self._persist_locked()
        return {
            "ok": True,
            "schema": RENDEZVOUS_SCHEMA,
            "run_kind": run_kind,
            "kind": kind,
            "step": step,
            "microbatch": microbatch,
            "payload_hash": actual_hash,
        }

    def get_payload(
        self,
        *,
        run_id: str,
        run_kind: str,
        kind: str,
        step: int,
        microbatch: int,
    ) -> dict[str, Any] | None:
        self._require_run(run_id)
        key = self._payload_key(
            {
                "run_kind": run_kind,
                "kind": kind,
                "step": step,
                "microbatch": microbatch,
            }
        )
        with self._lock:
            value = self._payloads.get(key)
            return dict(value) if value else None

    def add_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_run(str(payload.get("run_id") or ""))
        role = str(payload.get("role") or "")
        self._require_role(role)
        operation = str(payload.get("operation") or "")
        run_kind = str(payload.get("run_kind") or "")
        if operation not in EVENT_OPERATIONS or run_kind not in RUN_KINDS:
            raise ValueError("qwen15b_training_event_invalid")
        public = {
            "role": role,
            "run_kind": run_kind,
            "operation": operation,
            "stage_id": int(payload.get("stage_id", -1)),
            "step": int(payload.get("step", -1)),
            "microbatch": int(payload.get("microbatch", -1)),
            "pid": int(payload.get("pid") or 0),
            "device": str(payload.get("device") or ""),
            "started_ns": int(payload.get("started_ns") or 0),
            "ended_ns": int(payload.get("ended_ns") or 0),
            "loss": float(payload["loss"]) if payload.get("loss") is not None else None,
            "gradient_norm": (
                float(payload["gradient_norm"])
                if payload.get("gradient_norm") is not None
                else None
            ),
            "checkpoint_hash": str(payload.get("checkpoint_hash") or ""),
            "adapter_hash": str(payload.get("adapter_hash") or ""),
            "at": time.time(),
        }
        if public["stage_id"] not in {0, 1, 2, 3}:
            raise ValueError("qwen15b_training_event_stage_invalid")
        with self._lock:
            key = self._event_key(public)
            if key not in self._event_keys:
                self._event_keys.add(key)
                self._events.append(public)
                self._persist_locked()
        return {"ok": True, "schema": RENDEZVOUS_SCHEMA, "operation": operation}

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_run(str(payload.get("run_id") or ""))
        role = str(payload.get("role") or "")
        self._require_role(role)
        summary = dict(payload.get("summary") or {})
        allowed = {
            "ok",
            "baseline_steps_completed",
            "resumed_steps_completed",
            "stage_ids",
            "final_adapter_hashes",
            "checkpoint_hashes",
            "controlled_restart_verified",
            "evaluation_verified",
            "export_verified",
            "coordinator_restart_owned_stages_verified",
            "transport_retry_count",
            "transport_reconnect_registration_count",
        }
        public = {key: summary[key] for key in allowed if key in summary}
        public.update({"role": role, "completed_at": time.time()})
        with self._lock:
            previous = self._completions.get(role)
            if previous:
                comparable_previous = {
                    key: value for key, value in previous.items() if key != "completed_at"
                }
                comparable_public = {
                    key: value for key, value in public.items() if key != "completed_at"
                }
                if comparable_previous != comparable_public:
                    raise ValueError("qwen15b_training_completion_conflict")
                return {"ok": True, "schema": RENDEZVOUS_SCHEMA, "role": role}
            self._completions[role] = public
            self._events.append(
                {"operation": "worker_completed", "role": role, "at": public["completed_at"]}
            )
            self._persist_locked()
        return {"ok": True, "schema": RENDEZVOUS_SCHEMA, "role": role}

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            payloads = [
                {key: value for key, value in record.items() if key != "payload_b64"}
                for record in self._payloads.values()
            ]
            post_restart_roles = sorted(
                {
                    str(item.get("role") or "")
                    for item in self._registration_history
                    if int(item.get("coordinator_generation") or 0) >= 1
                }
            )
            restart_records = [
                {
                    "generation": int(item.get("generation") or 0),
                    "after_step": int(item.get("after_step") or 0),
                    "started_at": float(item.get("started_at") or 0),
                    "completed_at": float(item.get("completed_at") or 0),
                    "duration_seconds": float(item.get("duration_seconds") or 0),
                    "recovered_payload_count": int(item.get("recovered_payload_count") or 0),
                    "recovered_event_count": int(item.get("recovered_event_count") or 0),
                }
                for item in self._restart_records
            ]
            return {
                "schema": RENDEZVOUS_SCHEMA,
                "run_id_hash": _sha256_bytes(self.run_id.encode("utf-8")),
                "registered_roles": sorted(self._registrations),
                "registrations": [self._registrations[key] for key in sorted(self._registrations)],
                "registration_history": list(self._registration_history),
                "coordinator_generation": self._generation,
                "persistent_state_enabled": self._state_path is not None,
                "recovered_from_persistent_state": self._recovered_from_persistent_state,
                "coordinator_restarts": restart_records,
                "post_restart_registered_roles": post_restart_roles,
                "coordinator_restart_verified": bool(
                    restart_records
                    and all(item["completed_at"] > item["started_at"] > 0 for item in restart_records)
                    and post_restart_roles == ["kernel_a", "kernel_b"]
                ),
                "payloads": sorted(
                    payloads,
                    key=lambda item: (
                        item["run_kind"],
                        item["step"],
                        item["microbatch"],
                        item["kind"],
                    ),
                ),
                "events": list(self._events),
                "completions": [self._completions[key] for key in sorted(self._completions)],
                "activation_values_public": False,
                "gradient_values_public": False,
                "adapter_tensor_values_public": False,
                "token_ids_public": False,
                "credentials_public": False,
                "public_artifact_safe": True,
            }

    def cleanup(self) -> dict[str, Any]:
        with self._lock:
            payload_count = len(self._payloads)
            payload_bytes = sum(int(value["byte_count"]) for value in self._payloads.values())
            self._payloads.clear()
            self._persist_locked()
            if self._state_path is not None:
                shutil.rmtree(
                    self._state_path.parent / f"{self._state_path.stem}-payloads",
                    ignore_errors=True,
                )
            return {
                "schema": "crowdtensor_qwen15b_four_gpu_rendezvous_cleanup_v1",
                "private_payload_count_removed": payload_count,
                "private_payload_bytes_removed": payload_bytes,
                "private_payloads_removed": True,
                "public_artifact_safe": True,
            }


def install_qwen15b_training_routes(
    app: Any,
    *,
    rendezvous: Qwen15BTrainingRendezvous,
    authorize: Callable[[str | None], None],
) -> None:
    """Mount private Qwen training routes on an existing Coordinator app."""

    from fastapi import Header, HTTPException, Query
    from pydantic import BaseModel, Field

    class RegistrationRequest(BaseModel):
        run_id: str = Field(min_length=1)
        role: str = Field(min_length=1)
        worker_id_hash: str = Field(min_length=1)
        stage_ids: list[int]
        stage_pids: list[int]
        cuda_devices: list[str]
        cuda_device_name_hashes: list[str]
        cuda_live: bool

    class PayloadRequest(BaseModel):
        run_id: str = Field(min_length=1)
        run_kind: str = Field(min_length=1)
        kind: str = Field(min_length=1)
        step: int = Field(ge=0)
        microbatch: int = Field(ge=-1)
        producer_role: str = Field(min_length=1)
        payload_b64: str = Field(min_length=1)
        payload_hash: str = Field(min_length=1)
        tensor_count: int = Field(ge=1)

    class EventRequest(BaseModel):
        run_id: str = Field(min_length=1)
        role: str = Field(min_length=1)
        run_kind: str = Field(min_length=1)
        operation: str = Field(min_length=1)
        stage_id: int
        step: int = -1
        microbatch: int = -1
        pid: int = 0
        device: str = ""
        started_ns: int = 0
        ended_ns: int = 0
        loss: float | None = None
        gradient_norm: float | None = None
        checkpoint_hash: str = ""
        adapter_hash: str = ""

    class CompletionRequest(BaseModel):
        run_id: str = Field(min_length=1)
        role: str = Field(min_length=1)
        summary: dict[str, Any] = Field(default_factory=dict)

    def guarded(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return call()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/qwen15b-training/register")
    def register(
        request: RegistrationRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return guarded(lambda: rendezvous.register(request.model_dump()))

    @app.post("/qwen15b-training/payload")
    def put_payload(
        request: PayloadRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return guarded(lambda: rendezvous.put_payload(request.model_dump()))

    @app.get("/qwen15b-training/payload/{run_kind}/{kind}/{step}/{microbatch}")
    def get_payload(
        run_kind: str,
        kind: str,
        step: int,
        microbatch: int,
        run_id: str = Query(min_length=1),
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        value = guarded(
            lambda: rendezvous.get_payload(
                run_id=run_id,
                run_kind=run_kind,
                kind=kind,
                step=step,
                microbatch=microbatch,
            )
            or {}
        )
        if not value:
            raise HTTPException(status_code=404, detail="qwen15b_training_payload_not_ready")
        return value

    @app.post("/qwen15b-training/event")
    def event(
        request: EventRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return guarded(lambda: rendezvous.add_event(request.model_dump()))

    @app.post("/qwen15b-training/complete")
    def complete(
        request: CompletionRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return guarded(lambda: rendezvous.complete(request.model_dump()))

    @app.get("/qwen15b-training/status")
    def status(
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return rendezvous.public_status()
