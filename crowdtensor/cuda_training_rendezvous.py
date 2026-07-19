"""Authenticated private tensor rendezvous for two-node CUDA training."""

import base64
import binascii
import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .training_contract import sha256_file


RENDEZVOUS_SCHEMA = "crowdtensor_cuda_training_rendezvous_v1"
PAYLOAD_KINDS = {"activation", "gradient"}


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class CUDATrainingRendezvous:
    """Keeps cross-node tensors private while exposing hash-only progress."""

    def __init__(self, *, run_id: str, max_payload_bytes: int = 16 * 1024 * 1024) -> None:
        self.run_id = str(run_id)
        self.max_payload_bytes = int(max_payload_bytes)
        self._lock = threading.RLock()
        self._registrations: dict[str, dict[str, Any]] = {}
        self._payloads: dict[tuple[str, int], dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._completions: dict[str, dict[str, Any]] = {}
        self._evaluations: dict[str, dict[str, Any]] = {}
        self._created_at = time.time()

    def _require_run(self, run_id: str) -> None:
        if str(run_id) != self.run_id:
            raise ValueError("cuda_training_run_id_mismatch")

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_run(str(payload.get("run_id") or ""))
        role = str(payload.get("role") or "")
        if role not in {"stage0", "stage1"}:
            raise ValueError("cuda_training_role_invalid")
        public = {
            "role": role,
            "worker_id_hash": str(payload.get("worker_id_hash") or ""),
            "pid": int(payload.get("pid") or 0),
            "cuda_device_index": int(payload.get("cuda_device_index", -1)),
            "cuda_device_name_hash": str(payload.get("cuda_device_name_hash") or ""),
            "cuda_live": payload.get("cuda_live") is True,
            "registered_at": time.time(),
        }
        if not public["worker_id_hash"] or public["pid"] <= 0 or public["cuda_device_index"] < 0:
            raise ValueError("cuda_training_registration_incomplete")
        with self._lock:
            self._registrations[role] = public
            self._events.append({"event": "registered", "role": role, "at": public["registered_at"]})
        return {"ok": True, "schema": RENDEZVOUS_SCHEMA, "role": role}

    def put_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_run(str(payload.get("run_id") or ""))
        kind = str(payload.get("kind") or "")
        if kind not in PAYLOAD_KINDS:
            raise ValueError("cuda_training_payload_kind_invalid")
        step = int(payload.get("step", -1))
        if step < 0:
            raise ValueError("cuda_training_payload_step_invalid")
        producer_role = str(payload.get("producer_role") or "")
        expected_role = "stage0" if kind == "activation" else "stage1"
        if producer_role != expected_role:
            raise ValueError("cuda_training_payload_producer_invalid")
        encoded = str(payload.get("payload_b64") or "")
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as exc:
            raise ValueError("cuda_training_payload_base64_invalid") from exc
        if not raw or len(raw) > self.max_payload_bytes:
            raise ValueError("cuda_training_payload_size_invalid")
        actual_hash = _sha256_bytes(raw)
        if actual_hash != str(payload.get("payload_hash") or ""):
            raise ValueError("cuda_training_payload_hash_mismatch")
        record = {
            "kind": kind,
            "step": step,
            "producer_role": producer_role,
            "payload_b64": encoded,
            "payload_hash": actual_hash,
            "byte_count": len(raw),
            "shape": [int(value) for value in payload.get("shape") or []],
            "dtype": str(payload.get("dtype") or ""),
            "gradient_scale": float(payload.get("gradient_scale") or 0.0) if kind == "gradient" else 0.0,
            "created_at": time.time(),
        }
        key = (kind, step)
        with self._lock:
            existing = self._payloads.get(key)
            if existing and existing["payload_hash"] != actual_hash:
                raise ValueError("cuda_training_payload_conflict")
            self._payloads[key] = record
            if not existing:
                self._events.append(
                    {
                        "event": f"{kind}_available",
                        "step": step,
                        "payload_hash": actual_hash,
                        "byte_count": len(raw),
                        "at": record["created_at"],
                    }
                )
        return {
            "ok": True,
            "schema": RENDEZVOUS_SCHEMA,
            "kind": kind,
            "step": step,
            "payload_hash": actual_hash,
        }

    def get_payload(self, *, run_id: str, kind: str, step: int) -> dict[str, Any] | None:
        self._require_run(run_id)
        if kind not in PAYLOAD_KINDS:
            raise ValueError("cuda_training_payload_kind_invalid")
        with self._lock:
            record = self._payloads.get((kind, int(step)))
            return dict(record) if record else None

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_run(str(payload.get("run_id") or ""))
        role = str(payload.get("role") or "")
        if role not in {"stage0", "stage1"}:
            raise ValueError("cuda_training_role_invalid")
        summary = dict(payload.get("summary") or {})
        allowed = {
            "steps_completed",
            "real_cuda_forward",
            "real_cuda_backward",
            "base_weights_frozen",
            "positive_lora_gradient_norms",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "final_adapter_hash",
            "checkpoint_hash",
            "loss_start",
            "loss_end",
            "loss_reduced",
        }
        public = {key: value for key, value in summary.items() if key in allowed}
        public.update({"role": role, "completed_at": time.time()})
        with self._lock:
            self._completions[role] = public
            self._events.append({"event": "worker_completed", "role": role, "at": public["completed_at"]})
        return {"ok": True, "schema": RENDEZVOUS_SCHEMA, "role": role}

    def put_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_run(str(payload.get("run_id") or ""))
        role = str(payload.get("role") or "")
        if role not in {"stage0", "stage1"}:
            raise ValueError("cuda_training_role_invalid")
        encoded = str(payload.get("logits_b64") or "")
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as exc:
            raise ValueError("cuda_training_evaluation_base64_invalid") from exc
        if not raw or len(raw) > self.max_payload_bytes:
            raise ValueError("cuda_training_evaluation_size_invalid")
        logits_hash = _sha256_bytes(raw)
        if logits_hash != str(payload.get("logits_hash") or ""):
            raise ValueError("cuda_training_evaluation_hash_mismatch")
        record = {
            "role": role,
            "logits_b64": encoded,
            "logits_hash": logits_hash,
            "byte_count": len(raw),
            "shape": [int(value) for value in payload.get("shape") or []],
            "dtype": str(payload.get("dtype") or ""),
            "before_loss": float(payload.get("before_loss")),
            "after_loss": float(payload.get("after_loss")),
            "adapter_changes_logits": payload.get("adapter_changes_logits") is True,
            "standard_peft_cuda_load": payload.get("standard_peft_cuda_load") is True,
            "created_at": time.time(),
        }
        with self._lock:
            self._evaluations[role] = record
            self._events.append({"event": "cuda_evaluation", "role": role, "at": record["created_at"]})
        return {"ok": True, "schema": RENDEZVOUS_SCHEMA, "role": role, "logits_hash": logits_hash}

    def private_evaluations(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {role: dict(value) for role, value in self._evaluations.items()}

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            payloads = [
                {
                    key: value
                    for key, value in record.items()
                    if key != "payload_b64"
                }
                for record in self._payloads.values()
            ]
            evaluations = [
                {key: value for key, value in record.items() if key != "logits_b64"}
                for record in self._evaluations.values()
            ]
            return {
                "schema": RENDEZVOUS_SCHEMA,
                "run_id_hash": _sha256_bytes(self.run_id.encode("utf-8")),
                "registered_roles": sorted(self._registrations),
                "registrations": [self._registrations[key] for key in sorted(self._registrations)],
                "payloads": sorted(payloads, key=lambda item: (item["step"], item["kind"])),
                "completions": [self._completions[key] for key in sorted(self._completions)],
                "evaluations": sorted(evaluations, key=lambda item: item["role"]),
                "events": list(self._events),
                "activation_values_public": False,
                "gradient_values_public": False,
                "evaluation_logits_public": False,
                "credentials_public": False,
                "raw_training_text_public": False,
                "public_artifact_safe": True,
            }

    def cleanup(self) -> dict[str, Any]:
        with self._lock:
            payload_count = len(self._payloads)
            evaluation_count = len(self._evaluations)
            self._payloads.clear()
            self._evaluations.clear()
            return {
                "schema": "crowdtensor_cuda_training_rendezvous_cleanup_v1",
                "private_payload_count_removed": payload_count,
                "private_evaluation_count_removed": evaluation_count,
                "private_payloads_removed": True,
                "public_artifact_safe": True,
            }


def install_cuda_training_routes(
    app: Any,
    *,
    rendezvous: CUDATrainingRendezvous,
    authorize: Callable[[str | None], None],
    store: Any,
    adapter_config_path: str | Path,
) -> None:
    """Mount private CUDA routes on the existing authenticated Coordinator app."""

    from fastapi import Header, HTTPException, Query
    from pydantic import BaseModel, Field

    class RegistrationRequest(BaseModel):
        run_id: str = Field(min_length=1)
        role: str = Field(min_length=1)
        worker_id_hash: str = Field(min_length=1)
        pid: int = Field(gt=0)
        cuda_device_index: int = Field(ge=0)
        cuda_device_name_hash: str = Field(min_length=1)
        cuda_live: bool

    class TensorRequest(BaseModel):
        run_id: str = Field(min_length=1)
        kind: str = Field(min_length=1)
        step: int = Field(ge=0)
        producer_role: str = Field(min_length=1)
        payload_b64: str = Field(min_length=1)
        payload_hash: str = Field(min_length=1)
        shape: list[int] = Field(default_factory=list)
        dtype: str = Field(min_length=1)
        gradient_scale: float | None = None

    class CompletionRequest(BaseModel):
        run_id: str = Field(min_length=1)
        role: str = Field(min_length=1)
        summary: dict[str, Any] = Field(default_factory=dict)

    class EvaluationRequest(BaseModel):
        run_id: str = Field(min_length=1)
        role: str = Field(min_length=1)
        logits_b64: str = Field(min_length=1)
        logits_hash: str = Field(min_length=1)
        shape: list[int] = Field(default_factory=list)
        dtype: str = Field(min_length=1)
        before_loss: float
        after_loss: float
        adapter_changes_logits: bool
        standard_peft_cuda_load: bool

    def guarded(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return call()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/cuda-training/register")
    def register(
        request: RegistrationRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return guarded(lambda: rendezvous.register(request.model_dump()))

    @app.post("/cuda-training/payload")
    def put_payload(
        request: TensorRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return guarded(lambda: rendezvous.put_payload(request.model_dump()))

    @app.get("/cuda-training/payload/{kind}/{step}")
    def get_payload(
        kind: str,
        step: int,
        run_id: str = Query(min_length=1),
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        value = guarded(lambda: rendezvous.get_payload(run_id=run_id, kind=kind, step=step) or {})
        if not value:
            raise HTTPException(status_code=404, detail="cuda_training_payload_not_ready")
        return value

    @app.post("/cuda-training/complete")
    def complete(
        request: CompletionRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return guarded(lambda: rendezvous.complete(request.model_dump()))

    @app.get("/cuda-training/global-adapter")
    def global_adapter(
        run_id: str = Query(min_length=1),
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        guarded(lambda: (rendezvous._require_run(run_id), {})[1])
        state = dict(store.training_state or {})
        if state.get("round_status") != "aggregated":
            raise HTTPException(status_code=404, detail="cuda_training_global_adapter_not_ready")
        adapter_path = Path(str(state.get("global_adapter_path") or ""))
        config_path = Path(adapter_config_path)
        if not adapter_path.is_file() or not config_path.is_file():
            raise HTTPException(status_code=503, detail="cuda_training_global_adapter_missing")
        adapter_bytes = adapter_path.read_bytes()
        config_bytes = config_path.read_bytes()
        return {
            "schema": "crowdtensor_cuda_training_global_adapter_private_v1",
            "adapter_b64": base64.b64encode(adapter_bytes).decode("ascii"),
            "adapter_hash": sha256_file(adapter_path),
            "adapter_config_b64": base64.b64encode(config_bytes).decode("ascii"),
            "adapter_config_hash": sha256_file(config_path),
            "adapter_version": int(state.get("adapter_version", 0)),
            "outer_step": int(state.get("outer_step", 0)),
        }

    @app.post("/cuda-training/evaluation")
    def evaluation(
        request: EvaluationRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return guarded(lambda: rendezvous.put_evaluation(request.model_dump()))

    @app.get("/cuda-training/status")
    def status(
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return rendezvous.public_status()
