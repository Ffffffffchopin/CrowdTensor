"""Durable task, checkpoint, validation, and ledger state for CrowdTensorD."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Iterable

from .audit import (
    AUDIT_MODE_NONE,
    AUDIT_MODE_REPLAY,
    verify_diloco_replay,
    verify_lora_replay,
    verify_micro_transformer_replay,
    verify_model_bundle_replay,
)
from .protocol import (
    DEFAULT_PROTOCOL_VERSION,
    DEFAULT_WORKLOAD_TYPE,
    EVENT_CLAIM_BLOCKED,
    EVENT_INCOMPATIBLE_CLAIM,
    EVENT_TASK_CLAIMED,
    EVENT_TASK_COMPLETED,
    EVENT_TASK_CREATED,
    EVENT_TASK_HEARTBEAT,
    EVENT_TASK_REJECTED,
    EVENT_TASK_REQUEUED,
    EVENT_TRUST_OVERRIDE_SET,
    LeaseConflict,
    NoTaskAvailable,
    ResultRejected,
    REQUIREMENT_ANY,
    STATUS_COMPLETED,
    STATUS_LEASED,
    STATUS_QUEUED,
    STATUS_REJECTED,
    WORKLOAD_BROWSER_PROBE,
    WORKLOAD_CPU_LORA_MOCK,
    WORKLOAD_DILOCO_TRAIN,
    WORKLOAD_EXTERNAL_LLM_INFER,
    WORKLOAD_MICRO_TRANSFORMER_LM,
    WORKLOAD_MODEL_BUNDLE_INFER,
    WORKLOAD_MODEL_BUNDLE_LM,
    new_lease_token,
    new_task_id,
    now_epoch,
)
from .diloco import (
    apply_outer_update,
    apply_outer_update_with_summary,
    default_model,
    loss,
    normalize_model,
    training_spec_for,
)
from .external_llm import (
    external_llm_inference_spec_for,
    validate_external_llm_inference,
)
from .lora_mock import (
    apply_adapter_update,
    adapter_loss,
    lora_training_spec_for,
    validate_adapter_delta,
)
from .micro_transformer import (
    apply_micro_transformer_update,
    micro_transformer_loss,
    micro_transformer_training_spec_for,
    micro_transformer_version,
    validate_micro_transformer_delta,
)
from .model_bundle import (
    apply_model_bundle_update,
    model_bundle_inference_spec_for,
    model_bundle_loss,
    model_bundle_training_spec_for,
    model_bundle_version,
    normalize_inference_scenario_id,
    validate_model_bundle_delta,
    validate_model_bundle_inference,
)
from .outer_optimizer import (
    DELTA_FORMAT_DENSE_FLOAT,
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    normalize_delta_format,
    optimizer_claim_spec,
)
from .outer_optimizer import decode_delta_payload
from .outer_optimizer import OPTIMIZER_DILOCO_MOMENTUM
from .validation import validate_local_delta


BROWSER_PROBE_SPEC = {
    "type": WORKLOAD_BROWSER_PROBE,
    "bytes": 1048576,
    "cols": 1024,
    "iterations": 8,
    "buffer_pattern": "sin_mod_v1",
}

QUARANTINE_CONSECUTIVE_REJECTIONS = 2
QUARANTINE_SCORE_THRESHOLD = -3.0
TRUST_OVERRIDE_ALLOW = "allow"
TRUST_OVERRIDE_BLOCK = "block"
TRUST_OVERRIDE_NONE = "none"
TRUST_OVERRIDE_MODES = {TRUST_OVERRIDE_ALLOW, TRUST_OVERRIDE_BLOCK, TRUST_OVERRIDE_NONE}
MAX_EVENT_TAIL_LIMIT = 500


class StateStore:
    """Append-only task log plus atomic global-model checkpoint."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        lease_seconds: float = 15.0,
        inner_steps: int = 500,
        backlog: int = 1,
        task_lanes: Iterable[dict] | None = None,
        replay_audit: bool = False,
        outer_optimizer: str = OPTIMIZER_DILOCO_MOMENTUM,
        delta_format: str = DELTA_FORMAT_DENSE_FLOAT,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.task_log_path = self.state_dir / "tasks.jsonl"
        self.checkpoint_path = self.state_dir / "global_model.json"
        self.lease_seconds = float(lease_seconds)
        self.inner_steps = int(inner_steps)
        self.backlog = int(backlog)
        self.replay_audit = bool(replay_audit)
        self.outer_optimizer = str(outer_optimizer or OPTIMIZER_DILOCO_MOMENTUM)
        self.delta_format = normalize_delta_format(delta_format)
        if self.replay_audit and self.delta_format == DELTA_FORMAT_SIGN_COMPRESSED_EF:
            raise ValueError("sign_compressed_ef cannot be used with replay_audit")
        self.task_lanes = self._normalize_task_lanes(task_lanes)
        self._lock = threading.RLock()
        self._event_index = 0
        self._tasks: dict[str, dict] = {}
        self._incompatible_claims: list[dict] = []
        self._blocked_claims: list[dict] = []
        self._trust_overrides: dict[str, dict[str, dict]] = {}
        self._model = default_model(outer_optimizer_type=self.outer_optimizer)

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._load()
        self._recover_inflight()
        self.ensure_backlog()

    @property
    def model(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._model))

    def claim_task(self, miner_id: str | None = None, capabilities: dict | None = None) -> dict:
        with self._lock:
            self.ensure_backlog()
            miner_name = miner_id or "anonymous"
            miner_capabilities = dict(capabilities or {})
            queued = [
                task for task in self._tasks.values()
                if task["status"] == STATUS_QUEUED
            ]
            if not queued:
                raise NoTaskAvailable("no queued task available")

            compatible = [
                task for task in queued
                if self._capabilities_match(task, miner_capabilities)
            ]
            if not compatible:
                now = now_epoch()
                event = self._append_event({
                    "type": EVENT_INCOMPATIBLE_CLAIM,
                    "miner_id": miner_name,
                    "capabilities": miner_capabilities,
                    "queued_task_count": len(queued),
                    "queued_requirements": sorted(
                        {self._requirement_key(task) for task in queued}
                    ),
                    "reason": "no compatible queued task available",
                    "ts": now,
                })
                self._apply_task_event(event)
                raise NoTaskAvailable("no compatible queued task available")

            blocked_decisions = [
                self._trust_decision(miner_name, self._workload_type(task))
                for task in compatible
            ]
            eligible = [
                task for task, decision in zip(compatible, blocked_decisions)
                if not decision["blocked"]
            ]
            if not eligible:
                reason = self._blocked_claim_reason(blocked_decisions)
                now = now_epoch()
                event = self._append_event({
                    "type": EVENT_CLAIM_BLOCKED,
                    "miner_id": miner_name,
                    "capabilities": miner_capabilities,
                    "queued_task_count": len(queued),
                    "compatible_task_count": len(compatible),
                    "blocked_workloads": sorted({self._workload_type(task) for task in compatible}),
                    "reason": reason,
                    "ts": now,
                })
                self._apply_task_event(event)
                raise NoTaskAvailable(reason)

            task = sorted(eligible, key=lambda item: item["created_at"])[0]
            model_version = self._model_version_for_task(task)
            claimed_contract = self._claim_contract(task, miner_name, model_version)
            optimizer_spec = dict(claimed_contract.get("claim_optimizer_spec") or {})
            now = now_epoch()
            event = self._append_event({
                "type": EVENT_TASK_CLAIMED,
                "task_id": task["task_id"],
                "attempt": int(task.get("attempt", 0)) + 1,
                "lease_token": new_lease_token(),
                "lease_expires_at": now + self.lease_seconds,
                "miner_id": miner_name,
                "capabilities": miner_capabilities,
                "model_version": model_version,
                "inner_steps": int(task.get("inner_steps", self.inner_steps)),
                **claimed_contract,
                "claim_optimizer_spec": optimizer_spec,
                "ts": now,
            })
            self._apply_task_event(event)
            claimed = self._tasks[task["task_id"]]
            training_spec = claimed.get("claim_training_spec") or training_spec_for(
                claimed["task_id"],
                claimed.get("miner_id") or "anonymous",
                int(claimed["model_version"]),
            )
            workload_spec = claimed.get("claim_workload_spec") or self._workload_spec(claimed)
            weights = claimed.get("claim_weights") or self._claim_weights_for_task(claimed, workload_spec)
            optimizer_spec = claimed.get("claim_optimizer_spec") or self._optimizer_spec_for_task(claimed)
            return {
                "task_id": claimed["task_id"],
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "lease_expires_at": claimed["lease_expires_at"],
                "model_version": claimed["model_version"],
                "weights": list(weights),
                "inner_steps": claimed["inner_steps"],
                "workload_type": claimed.get("workload_type", DEFAULT_WORKLOAD_TYPE),
                "workload_spec": workload_spec,
                "audit_mode": claimed.get("audit_mode", AUDIT_MODE_NONE),
                "heartbeat_interval": max(1.0, self.lease_seconds / 3.0),
                "schema_version": self._model.get("schema_version"),
                "optimizer_step": self._model.get("optimizer_step", 0),
                "optimizer_spec": optimizer_spec,
                "task_requirements": self._task_requirements(claimed),
                "training_spec": training_spec,
            }

    def heartbeat(
        self,
        task_id: str,
        *,
        lease_token: str,
        attempt: int,
        runtime_status: dict | None = None,
    ) -> dict:
        with self._lock:
            task = self._require_live_lease(task_id, lease_token=lease_token, attempt=attempt)
            now = now_epoch()
            event = self._append_event({
                "type": EVENT_TASK_HEARTBEAT,
                "task_id": task_id,
                "attempt": attempt,
                "lease_token": lease_token,
                "lease_expires_at": now + self.lease_seconds,
                "runtime_status": dict(runtime_status or {}),
                "ts": now,
            })
            self._apply_task_event(event)
            return {
                "task_id": task_id,
                "attempt": attempt,
                "lease_expires_at": self._tasks[task_id]["lease_expires_at"],
            }

    def set_trust_override(
        self,
        miner_id: str,
        workload_type: str,
        mode: str,
        *,
        reason: str = "",
        actor: str = "admin",
    ) -> dict:
        miner_name = str(miner_id or "").strip()
        workload = str(workload_type or "").strip()
        override_mode = str(mode or "").strip().lower()
        if not miner_name:
            raise ValueError("miner_id is required")
        if not workload:
            raise ValueError("workload_type is required")
        if override_mode not in TRUST_OVERRIDE_MODES:
            raise ValueError("trust override mode must be allow, block, or none")

        with self._lock:
            now = now_epoch()
            event = self._append_event({
                "type": EVENT_TRUST_OVERRIDE_SET,
                "miner_id": miner_name,
                "workload_type": workload,
                "mode": override_mode,
                "reason": str(reason or ""),
                "actor": str(actor or "admin"),
                "ts": now,
            })
            self._apply_task_event(event)
            return {
                "accepted": True,
                "miner_id": miner_name,
                "workload_type": workload,
                "mode": override_mode,
                "reason": str(reason or ""),
                "event_index": int(event["event_index"]),
            }

    def event_tail(self, *, limit: int = 50) -> list[dict]:
        capped = min(MAX_EVENT_TAIL_LIMIT, max(0, int(limit)))
        if capped == 0 or not self.task_log_path.exists():
            return []
        with self._lock:
            events: list[dict] = []
            with self.task_log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    events.append(json.loads(line))
            return [
                self._redact_event(event)
                for event in events[-capped:]
            ]

    def result_ledger(
        self,
        *,
        limit: int = 50,
        status: str = "any",
        miner_id: str | None = None,
        workload_type: str | None = None,
        task_id: str | None = None,
    ) -> list[dict]:
        capped = min(MAX_EVENT_TAIL_LIMIT, max(0, int(limit)))
        if capped == 0:
            return []
        wanted_status = str(status or "any").strip().lower()
        if wanted_status not in {"any", "accepted", "rejected"}:
            raise ValueError("status must be any, accepted, or rejected")
        wanted_miner = str(miner_id or "").strip()
        wanted_workload = str(workload_type or "").strip()
        wanted_task_id = str(task_id or "").strip()

        with self._lock:
            scored = self._miner_workload_scores()
            rows = [
                self._result_ledger_entry(task, scored)
                for task in self._tasks.values()
                if task.get("status") in {STATUS_COMPLETED, STATUS_REJECTED}
            ]
            rows.sort(key=lambda item: int(item.get("event_index", 0)), reverse=True)
            filtered = []
            for row in rows:
                if wanted_status == "accepted" and row["status"] != STATUS_COMPLETED:
                    continue
                if wanted_status == "rejected" and row["status"] != STATUS_REJECTED:
                    continue
                if wanted_miner and row["miner_id"] != wanted_miner:
                    continue
                if wanted_workload and row["workload_type"] != wanted_workload:
                    continue
                if wanted_task_id and row["task_id"] != wanted_task_id:
                    continue
                filtered.append(row)
                if len(filtered) >= capped:
                    break
            return filtered

    def create_readonly_inference_task(
        self,
        *,
        request_count: int = 4,
        scenario_id: str = "",
        required_runtime: str = "python-cli",
        required_backend: str = "cpu",
        required_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    ) -> dict:
        count = max(1, min(int(request_count), 8))
        scenario = normalize_inference_scenario_id(scenario_id)
        runtime = str(required_runtime or "").strip()
        backend = str(required_backend or "").strip()
        if runtime != "python-cli":
            raise ValueError("read-only inference sessions require runtime python-cli")
        if backend != "cpu":
            raise ValueError("read-only inference sessions require backend cpu")
        with self._lock:
            task_id = self._create_task(
                required_runtime=runtime,
                required_backend=backend,
                required_protocol_version=required_protocol_version,
                workload_type=WORKLOAD_MODEL_BUNDLE_INFER,
                inner_steps=count,
                workload_metadata={"scenario_id": scenario} if scenario else {},
            )
            task = self._tasks[task_id]
            return {
                "schema": "inference_session_request_v1",
                "accepted": True,
                "task_id": task_id,
                "status": task["status"],
                "workload_type": task["workload_type"],
                "request_count": task["inner_steps"],
                "scenario_id": scenario,
                "task_requirements": self._task_requirements(task),
            }

    def complete_task(
        self,
        task_id: str,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        local_delta: Iterable[float] | None = None,
        pseudo_gradient: Iterable[float] | None = None,
        compressed_delta: dict | None = None,
        probe_result: dict | None = None,
        adapter_delta: dict | None = None,
        bundle_delta: dict | None = None,
        inference_result: dict | None = None,
        inference_results: list[dict] | None = None,
        external_llm_result: dict | None = None,
        external_llm_results: list[dict] | None = None,
        metrics: dict | None = None,
    ) -> dict:
        with self._lock:
            terminal_response = self._terminal_idempotent_result(
                task_id,
                lease_token=lease_token,
                attempt=attempt,
                idempotency_key=idempotency_key,
            )
            if terminal_response is not None:
                if terminal_response.get("accepted") is True:
                    return terminal_response
                raise ResultRejected(terminal_response)

            task = self._require_live_lease(task_id, lease_token=lease_token, attempt=attempt)
            workload_type = task.get("workload_type", DEFAULT_WORKLOAD_TYPE) or DEFAULT_WORKLOAD_TYPE
            if workload_type == WORKLOAD_BROWSER_PROBE:
                return self._complete_probe_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    probe_result=probe_result,
                    metrics=metrics,
                )
            if workload_type == WORKLOAD_CPU_LORA_MOCK:
                return self._complete_lora_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    adapter_delta=adapter_delta,
                    metrics=metrics,
                )
            if workload_type == WORKLOAD_MICRO_TRANSFORMER_LM:
                return self._complete_micro_transformer_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    local_delta=local_delta if local_delta is not None else pseudo_gradient,
                    metrics=metrics,
                )
            if workload_type == WORKLOAD_MODEL_BUNDLE_LM:
                return self._complete_model_bundle_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    bundle_delta=bundle_delta,
                    metrics=metrics,
                )
            if workload_type == WORKLOAD_MODEL_BUNDLE_INFER:
                return self._complete_model_bundle_inference_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    inference_result=inference_result,
                    inference_results=inference_results,
                    metrics=metrics,
                )
            if workload_type == WORKLOAD_EXTERNAL_LLM_INFER:
                return self._complete_external_llm_inference_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    external_llm_result=external_llm_result,
                    external_llm_results=external_llm_results,
                    metrics=metrics,
                )
            if workload_type != WORKLOAD_DILOCO_TRAIN:
                raise ValueError(f"unsupported workload_type {workload_type}")

            staleness = int(self._model["version"]) - int(task["model_version"])
            raw_delta, delta_metadata = decode_delta_payload(
                local_delta=local_delta,
                pseudo_gradient=pseudo_gradient,
                compressed_delta=compressed_delta,
            )
            validation = validate_local_delta(self._model, raw_delta)
            validation = {**validation, **delta_metadata}
            validation = self._validate_delta_format_contract(task, validation)
            if validation["accepted"]:
                validation = self._audit_diloco_result(task, validation)
            else:
                validation = self._annotate_audit_skip(task, validation)
            if not validation["accepted"]:
                response = dict(validation)
                now = now_epoch()
                event = self._append_event({
                    "type": EVENT_TASK_REJECTED,
                    "task_id": task_id,
                    "attempt": attempt,
                    "lease_token": lease_token,
                    "miner_id": task.get("miner_id"),
                    "base_model_version": int(task["model_version"]),
                    "local_delta": validation.get("local_delta"),
                    "metrics": metrics or {},
                    "staleness": staleness,
                    "validation": validation,
                    "result_response": response,
                    **self._idempotency_event_fields(
                        idempotency_key,
                        lease_token=lease_token,
                    ),
                    "ts": now,
                })
                self._apply_task_event(event)
                self.ensure_backlog()
                raise ResultRejected(validation)

            delta = validation["local_delta"]
            next_model, optimizer_summary = apply_outer_update_with_summary(
                self._model,
                delta,
                delta_metadata=delta_metadata,
            )
            response = {
                "accepted": True,
                "model_version": next_model["version"],
                "global_step": next_model["global_step"],
                "optimizer_step": next_model["optimizer_step"],
                "weights": list(next_model["weights"]),
                "outer_velocity": list(next_model["outer_velocity"]),
                "optimizer": optimizer_summary,
                "loss": loss(next_model),
                "staleness": staleness,
            }
            now = now_epoch()
            event = self._append_event({
                "type": EVENT_TASK_COMPLETED,
                "task_id": task_id,
                "attempt": attempt,
                "lease_token": lease_token,
                "miner_id": task.get("miner_id"),
                "base_model_version": int(task["model_version"]),
                "local_delta": delta,
                "metrics": metrics or {},
                "staleness": staleness,
                "validation": validation,
                "optimizer": optimizer_summary,
                "result_model_version": int(next_model["version"]),
                "model_updated": True,
                "result_response": response,
                **self._idempotency_event_fields(
                    idempotency_key,
                    lease_token=lease_token,
                ),
                "ts": now,
            })
            self._apply_task_event(event)
            self._model = next_model
            self._write_checkpoint()
            self.ensure_backlog()
            return response

    def _complete_probe_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        probe_result: dict | None = None,
        metrics: dict | None = None,
    ) -> dict:
        validation = self._validate_probe_result(probe_result)
        staleness = int(self._model["version"]) - int(task["model_version"])
        now = now_epoch()
        response = {
            "accepted": True,
            "model_updated": False,
            "workload_type": WORKLOAD_BROWSER_PROBE,
            "model_version": self._model["version"],
            "global_step": self._model["global_step"],
            "optimizer_step": self._model["optimizer_step"],
            "weights": list(self._model["weights"]),
            "outer_velocity": list(self._model["outer_velocity"]),
            "loss": loss(self._model),
            "staleness": staleness,
            "probe_result": probe_result or {},
        } if validation["accepted"] else dict(validation)
        event = {
            "type": EVENT_TASK_COMPLETED if validation["accepted"] else EVENT_TASK_REJECTED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "probe_result": probe_result or {},
            "metrics": metrics or {},
            "staleness": staleness,
            "validation": validation,
            "result_model_version": int(self._model["version"]),
            "model_updated": False,
            "result_response": response,
            **self._idempotency_event_fields(
                idempotency_key,
                lease_token=lease_token,
            ),
            "ts": now,
        }
        self._apply_task_event(self._append_event(event))
        self.ensure_backlog()
        if not validation["accepted"]:
            raise ResultRejected(validation)

        return response

    def _complete_lora_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        adapter_delta: dict | None = None,
        metrics: dict | None = None,
    ) -> dict:
        validation = validate_adapter_delta(self._model, adapter_delta)
        staleness = int(self._model["version"]) - int(task["model_version"])
        now = now_epoch()
        if validation["accepted"]:
            validation = self._audit_lora_result(task, validation)
        else:
            validation = self._annotate_audit_skip(task, validation)
        if not validation["accepted"]:
            response = dict(validation)
            event = self._append_event({
                "type": EVENT_TASK_REJECTED,
                "task_id": task["task_id"],
                "attempt": attempt,
                "lease_token": lease_token,
                "miner_id": task.get("miner_id"),
                "base_model_version": int(task["model_version"]),
                "adapter_delta": validation.get("adapter_delta"),
                "metrics": metrics or {},
                "staleness": staleness,
                "validation": validation,
                "result_model_version": int(self._model["version"]),
                "model_updated": False,
                "adapter_updated": False,
                "result_response": response,
                **self._idempotency_event_fields(
                    idempotency_key,
                    lease_token=lease_token,
                ),
                "ts": now,
            })
            self._apply_task_event(event)
            self.ensure_backlog()
            raise ResultRejected(validation)

        normalized_delta = validation["adapter_delta"]
        next_model = apply_adapter_update(self._model, normalized_delta)
        response = {
            "accepted": True,
            "model_updated": False,
            "adapter_updated": True,
            "workload_type": WORKLOAD_CPU_LORA_MOCK,
            "model_version": next_model["version"],
            "global_step": next_model["global_step"],
            "optimizer_step": next_model["optimizer_step"],
            "adapter_step": next_model["adapter_step"],
            "weights": list(next_model["weights"]),
            "lora_adapter": dict(next_model["lora_adapter"]),
            "outer_velocity": list(next_model["outer_velocity"]),
            "loss": loss(next_model),
            "adapter_loss": adapter_loss(next_model),
            "staleness": staleness,
        }
        event = self._append_event({
            "type": EVENT_TASK_COMPLETED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "adapter_delta": normalized_delta,
            "metrics": metrics or {},
            "staleness": staleness,
            "validation": validation,
            "result_model_version": int(self._model["version"]),
            "model_updated": False,
            "adapter_updated": True,
            "adapter_step_result": int(next_model["adapter_step"]),
            "result_response": response,
            **self._idempotency_event_fields(
                idempotency_key,
                lease_token=lease_token,
            ),
            "ts": now,
        })
        self._apply_task_event(event)
        self._model = next_model
        self._write_checkpoint()
        self.ensure_backlog()
        return response

    def _complete_micro_transformer_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        local_delta: Iterable[float] | None = None,
        metrics: dict | None = None,
    ) -> dict:
        validation = validate_micro_transformer_delta(self._model, local_delta)
        staleness = micro_transformer_version(self._model) - int(task["model_version"])
        now = now_epoch()
        if validation["accepted"]:
            validation = self._audit_micro_transformer_result(task, validation)
        else:
            validation = self._annotate_audit_skip(task, validation)
        if not validation["accepted"]:
            response = dict(validation)
            event = self._append_event({
                "type": EVENT_TASK_REJECTED,
                "task_id": task["task_id"],
                "attempt": attempt,
                "lease_token": lease_token,
                "miner_id": task.get("miner_id"),
                "base_model_version": int(task["model_version"]),
                "local_delta": validation.get("local_delta"),
                "metrics": metrics or {},
                "staleness": staleness,
                "validation": validation,
                "result_model_version": micro_transformer_version(self._model),
                "model_updated": False,
                "micro_transformer_updated": False,
                "result_response": response,
                **self._idempotency_event_fields(
                    idempotency_key,
                    lease_token=lease_token,
                ),
                "ts": now,
            })
            self._apply_task_event(event)
            self.ensure_backlog()
            raise ResultRejected(validation)

        delta = validation["local_delta"]
        next_model = apply_micro_transformer_update(self._model, delta)
        next_micro = next_model["micro_transformer"]
        response = {
            "accepted": True,
            "model_updated": False,
            "micro_transformer_updated": True,
            "workload_type": WORKLOAD_MICRO_TRANSFORMER_LM,
            "model_version": int(next_micro["version"]),
            "micro_transformer_optimizer_step": int(next_micro["optimizer_step"]),
            "micro_transformer_loss": micro_transformer_loss(next_model),
            "staleness": staleness,
        }
        event = self._append_event({
            "type": EVENT_TASK_COMPLETED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "local_delta": delta,
            "metrics": metrics or {},
            "staleness": staleness,
            "validation": validation,
            "result_model_version": int(next_micro["version"]),
            "model_updated": False,
            "micro_transformer_updated": True,
            "micro_transformer_step_result": int(next_micro["optimizer_step"]),
            "result_response": response,
            **self._idempotency_event_fields(
                idempotency_key,
                lease_token=lease_token,
            ),
            "ts": now,
        })
        self._apply_task_event(event)
        self._model = next_model
        self._write_checkpoint()
        self.ensure_backlog()
        return response

    def _complete_model_bundle_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        bundle_delta: dict | None = None,
        metrics: dict | None = None,
    ) -> dict:
        validation = validate_model_bundle_delta(self._model, bundle_delta)
        staleness = model_bundle_version(self._model) - int(task["model_version"])
        now = now_epoch()
        if validation["accepted"]:
            validation = self._audit_model_bundle_result(task, validation)
        else:
            validation = self._annotate_audit_skip(task, validation)
        if not validation["accepted"]:
            response = dict(validation)
            event = self._append_event({
                "type": EVENT_TASK_REJECTED,
                "task_id": task["task_id"],
                "attempt": attempt,
                "lease_token": lease_token,
                "miner_id": task.get("miner_id"),
                "base_model_version": int(task["model_version"]),
                "bundle_delta": validation.get("bundle_delta"),
                "metrics": metrics or {},
                "staleness": staleness,
                "validation": validation,
                "result_model_version": model_bundle_version(self._model),
                "model_updated": False,
                "model_bundle_updated": False,
                "result_response": response,
                **self._idempotency_event_fields(
                    idempotency_key,
                    lease_token=lease_token,
                ),
                "ts": now,
            })
            self._apply_task_event(event)
            self.ensure_backlog()
            raise ResultRejected(validation)

        normalized_delta = validation["bundle_delta"]
        next_model = apply_model_bundle_update(self._model, normalized_delta)
        next_bundle = next_model["model_bundle"]
        response = {
            "accepted": True,
            "model_updated": False,
            "model_bundle_updated": True,
            "workload_type": WORKLOAD_MODEL_BUNDLE_LM,
            "bundle_id": next_bundle["bundle_id"],
            "model_version": int(next_bundle["version"]),
            "bundle_version": int(next_bundle["version"]),
            "bundle_optimizer_step": int(next_bundle["optimizer_step"]),
            "bundle_loss": model_bundle_loss(next_model),
            "artifact_hash": next_bundle["artifact_hash"],
            "staleness": staleness,
        }
        event = self._append_event({
            "type": EVENT_TASK_COMPLETED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "bundle_delta": normalized_delta,
            "metrics": metrics or {},
            "staleness": staleness,
            "validation": validation,
            "result_model_version": int(next_bundle["version"]),
            "model_updated": False,
            "model_bundle_updated": True,
            "model_bundle_step_result": int(next_bundle["optimizer_step"]),
            "result_response": response,
            **self._idempotency_event_fields(
                idempotency_key,
                lease_token=lease_token,
            ),
            "ts": now,
        })
        self._apply_task_event(event)
        self._model = next_model
        self._write_checkpoint()
        self.ensure_backlog()
        return response

    def _complete_model_bundle_inference_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        inference_result: dict | None = None,
        inference_results: list[dict] | None = None,
        metrics: dict | None = None,
    ) -> dict:
        validation = validate_model_bundle_inference(
            self._model,
            inference_result,
            inference_results=inference_results,
            expected_requests=(task.get("claim_workload_spec") or {}).get("requests"),
            expected_scenario_id=(task.get("claim_workload_spec") or {}).get("scenario_id"),
        )
        staleness = model_bundle_version(self._model) - int(task["model_version"])
        now = now_epoch()
        response = {
            "accepted": True,
            "model_updated": False,
            "model_bundle_updated": False,
            "workload_type": WORKLOAD_MODEL_BUNDLE_INFER,
            "bundle_id": validation.get("bundle_id"),
            "model_version": model_bundle_version(self._model),
            "bundle_version": validation.get("base_bundle_version"),
            "artifact_hash": validation.get("artifact_hash"),
            "predicted_token_id": validation.get("predicted_token_id"),
            "predicted_token": validation.get("predicted_token"),
            "target_token_id": validation.get("target_token_id"),
            "target_token": validation.get("target_token"),
            "correct": bool(validation.get("correct", False)),
            "request_count": int(validation.get("request_count", 1)),
            "correct_count": int(validation.get("correct_count", 1 if validation.get("correct") else 0)),
            "accuracy": float(validation.get("accuracy", 1.0 if validation.get("correct") else 0.0)),
            "scenario_schema": validation.get("scenario_schema"),
            "scenario_id": validation.get("scenario_id"),
            "scenario_description": validation.get("scenario_description"),
            "scenario_request_count": validation.get("scenario_request_count"),
            "staleness": staleness,
        } if validation["accepted"] else dict(validation)
        event = {
            "type": EVENT_TASK_COMPLETED if validation["accepted"] else EVENT_TASK_REJECTED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "inference_result": validation.get("inference_result") if validation["accepted"] else inference_result,
            "inference_results": validation.get("inference_results") if validation["accepted"] else inference_results,
            "metrics": metrics or {},
            "staleness": staleness,
            "validation": validation,
            "result_model_version": model_bundle_version(self._model),
            "model_updated": False,
            "model_bundle_updated": False,
            "result_response": response,
            **self._idempotency_event_fields(
                idempotency_key,
                lease_token=lease_token,
            ),
            "ts": now,
        }
        self._apply_task_event(self._append_event(event))
        self.ensure_backlog()
        if not validation["accepted"]:
            raise ResultRejected(validation)
        return response

    def _complete_external_llm_inference_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        external_llm_result: dict | None = None,
        external_llm_results: list[dict] | None = None,
        metrics: dict | None = None,
    ) -> dict:
        validation = validate_external_llm_inference(
            external_llm_result,
            external_llm_results=external_llm_results,
            expected_requests=(task.get("claim_workload_spec") or {}).get("requests"),
        )
        staleness = int(self._model["version"]) - int(task["model_version"])
        now = now_epoch()
        response = {
            "accepted": True,
            "model_updated": False,
            "model_bundle_updated": False,
            "workload_type": WORKLOAD_EXTERNAL_LLM_INFER,
            "model_version": int(self._model["version"]),
            "request_count": int(validation.get("request_count", 0)),
            "completion_count": int(validation.get("completion_count", 0)),
            "output_chars": int(validation.get("output_chars", 0)),
            "adapter_kind": validation.get("adapter_kind"),
            "model_id": validation.get("model_id"),
            "output_preview": validation.get("output_preview"),
            "staleness": staleness,
        } if validation["accepted"] else dict(validation)
        event = {
            "type": EVENT_TASK_COMPLETED if validation["accepted"] else EVENT_TASK_REJECTED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "external_llm_result": (
                validation.get("external_llm_result") if validation["accepted"] else external_llm_result
            ),
            "external_llm_results": (
                validation.get("external_llm_results") if validation["accepted"] else external_llm_results
            ),
            "metrics": metrics or {},
            "staleness": staleness,
            "validation": validation,
            "result_model_version": int(self._model["version"]),
            "model_updated": False,
            "model_bundle_updated": False,
            "result_response": response,
            **self._idempotency_event_fields(
                idempotency_key,
                lease_token=lease_token,
            ),
            "ts": now,
        }
        self._apply_task_event(self._append_event(event))
        self.ensure_backlog()
        if not validation["accepted"]:
            raise ResultRejected(validation)
        return response

    def reap_expired(self, *, now: float | None = None) -> list[str]:
        with self._lock:
            current = now_epoch() if now is None else float(now)
            expired = [
                task["task_id"] for task in self._tasks.values()
                if task["status"] == STATUS_LEASED
                and float(task.get("lease_expires_at", 0.0)) <= current
            ]
            for task_id in expired:
                self._requeue(task_id, reason="lease_timeout", ts=current)
            self.ensure_backlog()
            return expired

    def ensure_backlog(self) -> None:
        with self._lock:
            for lane in self.task_lanes:
                active = sum(
                    1 for task in self._tasks.values()
                    if task["status"] in {STATUS_QUEUED, STATUS_LEASED}
                    and self._task_matches_lane(task, lane)
                )
                while active < int(lane["count"]):
                    self._create_task(
                        required_runtime=lane["runtime"],
                        required_backend=lane["backend"],
                        required_protocol_version=lane["protocol_version"],
                        workload_type=lane["workload_type"],
                    )
                    active += 1

    def summary(self) -> dict:
        with self._lock:
            counts = {
                STATUS_QUEUED: 0,
                STATUS_LEASED: 0,
                STATUS_COMPLETED: 0,
                STATUS_REJECTED: 0,
            }
            for task in self._tasks.values():
                counts[task["status"]] = counts.get(task["status"], 0) + 1
            completed = [
                task for task in self._tasks.values()
                if task["status"] == STATUS_COMPLETED
            ]
            model_updates = [
                task for task in completed
                if bool(task.get("model_updated", True))
            ]
            adapter_updates = [
                task for task in completed
                if bool(task.get("adapter_updated", False))
            ]
            micro_transformer_updates = [
                task for task in completed
                if bool(task.get("micro_transformer_updated", False))
            ]
            model_bundle_updates = [
                task for task in completed
                if bool(task.get("model_bundle_updated", False))
            ]
            staleness_values = [
                int(task.get("staleness", 0) or 0)
                for task in completed
            ]
            last_completed = None
            if completed:
                last_completed_task = max(
                    completed,
                    key=lambda item: float(item.get("completed_at", item.get("updated_at", 0.0))),
                )
                last_completed = self._public_task(last_completed_task)
            rejected = [
                task for task in self._tasks.values()
                if task["status"] == STATUS_REJECTED
            ]
            audit_results = [
                task for task in completed + rejected
                if (task.get("validation") or {}).get("audit_mode") == AUDIT_MODE_REPLAY
                and (task.get("validation") or {}).get("audit_accepted") is not None
            ]
            audit_rejections = [
                task for task in audit_results
                if (task.get("validation") or {}).get("audit_accepted") is False
            ]
            last_rejected = None
            if rejected:
                last_rejected_task = max(
                    rejected,
                    key=lambda item: float(item.get("rejected_at", item.get("updated_at", 0.0))),
                )
                last_rejected = self._public_task(last_rejected_task)
            miner_workload_scores = self._miner_workload_scores()
            auto_quarantined = self._quarantined_miners(miner_workload_scores)
            trust_overrides = self._public_trust_overrides()

            return {
                "model": self.model,
                "loss": loss(self._model),
                "event_index": self._event_index,
                "task_counts": counts,
                "accepted_results": len(completed),
                "model_updates": len(model_updates),
                "adapter_updates": len(adapter_updates),
                "micro_transformer_updates": len(micro_transformer_updates),
                "model_bundle_updates": len(model_bundle_updates),
                "rejected_results": len(rejected),
                "audit_results": len(audit_results),
                "audit_rejections": len(audit_rejections),
                "max_staleness": max(staleness_values) if staleness_values else 0,
                "avg_staleness": (
                    sum(staleness_values) / len(staleness_values)
                    if staleness_values else 0.0
                ),
                "last_completed": last_completed,
                "last_rejected": last_rejected,
                "miner_scores": self._miner_scores(),
                "miner_workload_scores": miner_workload_scores,
                "quarantined_miners": auto_quarantined,
                "miner_trust_overrides": trust_overrides,
                "effective_quarantined_miners": self._effective_quarantined_miners(auto_quarantined, trust_overrides),
                "manual_blocked_miners": self._manual_blocked_miners(trust_overrides),
                "miner_profiles": self._miner_profiles(),
                "task_lanes": [dict(lane) for lane in self.task_lanes],
                "task_counts_by_requirement": self._task_counts_by_requirement(),
                "task_counts_by_lane": self._task_counts_by_lane(),
                "incompatible_claims": len(self._incompatible_claims),
                "last_incompatible_claim": (
                    dict(self._incompatible_claims[-1])
                    if self._incompatible_claims else None
                ),
                "blocked_claims": len(self._blocked_claims),
                "last_blocked_claim": (
                    dict(self._blocked_claims[-1])
                    if self._blocked_claims else None
                ),
                "tasks": [
                    self._public_task(task)
                    for task in sorted(self._tasks.values(), key=lambda item: item["created_at"])
                ],
            }

    def _load(self) -> None:
        checkpoint_event_index = 0
        if self.checkpoint_path.exists():
            with self.checkpoint_path.open("r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            self._model = normalize_model(checkpoint.get("model", default_model()))
            checkpoint_event_index = int(checkpoint.get("event_index", 0))

        events: list[dict] = []
        if self.task_log_path.exists():
            with self.task_log_path.open("r", encoding="utf-8") as handle:
                for fallback_index, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    event["event_index"] = int(event.get("event_index", fallback_index))
                    events.append(event)

        for event in events:
            self._event_index = max(self._event_index, int(event["event_index"]))
            self._apply_task_event(event)

        for event in events:
            if event["type"] != EVENT_TASK_COMPLETED or int(event["event_index"]) <= checkpoint_event_index:
                continue
            if bool(event.get("model_updated", True)):
                delta = event.get("local_delta", event.get("pseudo_gradient"))
                if delta is not None:
                    optimizer = event.get("optimizer") or {}
                    if optimizer.get("optimizer_type"):
                        replay_model = {
                            **self._model,
                            "outer_optimizer_type": optimizer["optimizer_type"],
                            "outer_optimizer_contract": {
                                **(self._model.get("outer_optimizer_contract") or {}),
                                "optimizer_type": optimizer["optimizer_type"],
                            },
                        }
                    else:
                        replay_model = {
                            **self._model,
                            "outer_optimizer_type": OPTIMIZER_DILOCO_MOMENTUM,
                            "outer_optimizer_contract": {
                                **(self._model.get("outer_optimizer_contract") or {}),
                                "optimizer_type": OPTIMIZER_DILOCO_MOMENTUM,
                            },
                        }
                    self._model = apply_outer_update(replay_model, delta)
            elif bool(event.get("adapter_updated", False)):
                delta = event.get("adapter_delta")
                if delta is not None:
                    self._model = apply_adapter_update(self._model, delta)
            elif bool(event.get("micro_transformer_updated", False)):
                delta = event.get("local_delta")
                if delta is not None:
                    self._model = apply_micro_transformer_update(self._model, delta)
            elif bool(event.get("model_bundle_updated", False)):
                delta = event.get("bundle_delta")
                if delta is not None:
                    self._model = apply_model_bundle_update(self._model, delta)

    def _create_task(
        self,
        *,
        required_runtime: str = REQUIREMENT_ANY,
        required_backend: str = REQUIREMENT_ANY,
        required_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        workload_type: str = DEFAULT_WORKLOAD_TYPE,
        inner_steps: int | None = None,
        workload_metadata: dict | None = None,
    ) -> str:
        now = now_epoch()
        task_id = new_task_id()
        event = self._append_event({
            "type": EVENT_TASK_CREATED,
            "task_id": task_id,
            "inner_steps": self.inner_steps if inner_steps is None else max(1, int(inner_steps)),
            "required_runtime": required_runtime or REQUIREMENT_ANY,
            "required_backend": required_backend or REQUIREMENT_ANY,
            "required_protocol_version": required_protocol_version or DEFAULT_PROTOCOL_VERSION,
            "workload_type": workload_type or DEFAULT_WORKLOAD_TYPE,
            "workload_metadata": dict(workload_metadata or {}),
            "audit_mode": self._audit_mode_for_workload(workload_type or DEFAULT_WORKLOAD_TYPE),
            "ts": now,
        })
        self._apply_task_event(event)
        return task_id

    def _recover_inflight(self) -> None:
        now = now_epoch()
        for task_id, task in list(self._tasks.items()):
            if task["status"] == STATUS_LEASED:
                self._requeue(task_id, reason="startup_recovery", ts=now)

    def _requeue(self, task_id: str, *, reason: str, ts: float) -> None:
        task = self._tasks.get(task_id)
        if not task or task["status"] != STATUS_LEASED:
            return
        event = self._append_event({
            "type": EVENT_TASK_REQUEUED,
            "task_id": task_id,
            "attempt": task.get("attempt", 0),
            "reason": reason,
            "ts": ts,
        })
        self._apply_task_event(event)

    def _append_event(self, event: dict) -> dict:
        self._event_index += 1
        durable = dict(event)
        durable["event_index"] = self._event_index
        with self.task_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(durable, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return durable

    def _write_checkpoint(self) -> None:
        payload = {
            "event_index": self._event_index,
            "model": normalize_model(self._model),
            "loss": loss(self._model),
            "updated_at": now_epoch(),
        }
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.checkpoint_path)

    def _normalize_task_lanes(self, task_lanes: Iterable[dict] | None) -> list[dict]:
        raw_lanes = (
            [{"runtime": REQUIREMENT_ANY, "backend": REQUIREMENT_ANY, "count": self.backlog}]
            if task_lanes is None else list(task_lanes)
        )
        lanes: list[dict] = []
        for lane in raw_lanes:
            count = int(lane.get("count", 0))
            if count < 0:
                raise ValueError("task lane count must be non-negative")
            lanes.append({
                "runtime": str(lane.get("runtime", lane.get("required_runtime", REQUIREMENT_ANY)) or REQUIREMENT_ANY),
                "backend": str(lane.get("backend", lane.get("required_backend", REQUIREMENT_ANY)) or REQUIREMENT_ANY),
                "protocol_version": str(
                    lane.get(
                        "protocol_version",
                        lane.get("required_protocol_version", DEFAULT_PROTOCOL_VERSION),
                    ) or DEFAULT_PROTOCOL_VERSION
                ),
                "workload_type": str(lane.get("workload_type", DEFAULT_WORKLOAD_TYPE) or DEFAULT_WORKLOAD_TYPE),
                "count": count,
            })
        return lanes

    def _apply_task_event(self, event: dict) -> None:
        event_type = event["type"]
        if event_type == EVENT_TRUST_OVERRIDE_SET:
            miner_id = event.get("miner_id") or "anonymous"
            workload_type = event.get("workload_type", DEFAULT_WORKLOAD_TYPE) or DEFAULT_WORKLOAD_TYPE
            mode = event.get("mode", TRUST_OVERRIDE_NONE) or TRUST_OVERRIDE_NONE
            if mode == TRUST_OVERRIDE_NONE:
                miner_overrides = self._trust_overrides.get(miner_id)
                if miner_overrides:
                    miner_overrides.pop(workload_type, None)
                    if not miner_overrides:
                        self._trust_overrides.pop(miner_id, None)
                return
            self._trust_overrides.setdefault(miner_id, {})[workload_type] = {
                "mode": mode,
                "reason": event.get("reason", ""),
                "actor": event.get("actor", "admin"),
                "event_index": int(event.get("event_index", self._event_index)),
                "updated_at": float(event["ts"]),
            }
            return

        if event_type == EVENT_CLAIM_BLOCKED:
            self._blocked_claims.append({
                "event_index": int(event.get("event_index", self._event_index)),
                "miner_id": event.get("miner_id") or "anonymous",
                "capabilities": event.get("capabilities", {}),
                "queued_task_count": int(event.get("queued_task_count", 0)),
                "compatible_task_count": int(event.get("compatible_task_count", 0)),
                "blocked_workloads": list(event.get("blocked_workloads", [])),
                "reason": event.get("reason", "miner quarantined for workload"),
                "ts": float(event["ts"]),
            })
            return

        if event_type == EVENT_INCOMPATIBLE_CLAIM:
            self._incompatible_claims.append({
                "event_index": int(event.get("event_index", self._event_index)),
                "miner_id": event.get("miner_id") or "anonymous",
                "capabilities": event.get("capabilities", {}),
                "queued_task_count": int(event.get("queued_task_count", 0)),
                "queued_requirements": list(event.get("queued_requirements", [])),
                "reason": event.get("reason", "no compatible queued task available"),
                "ts": float(event["ts"]),
            })
            return

        if event_type == EVENT_TASK_CREATED:
            self._tasks[event["task_id"]] = {
                "task_id": event["task_id"],
                "status": STATUS_QUEUED,
                "attempt": 0,
                "lease_token": None,
                "lease_expires_at": None,
                "miner_id": None,
                "model_version": None,
                "base_model_version": None,
                "result_model_version": None,
                "staleness": None,
                "metrics": {},
                "validation": {},
                "probe_result": {},
                "adapter_delta": {},
                "bundle_delta": {},
                "inference_result": {},
                "inference_results": [],
                "external_llm_result": {},
                "external_llm_results": [],
                "result_response": {},
                "result_idempotency_key_hash": "",
                "result_lease_token_hash": "",
                "result_event_index": None,
                "model_updated": False,
                "adapter_updated": False,
                "micro_transformer_updated": False,
                "model_bundle_updated": False,
                "capabilities": {},
                "runtime_status": {},
                "completed_at": None,
                "rejected_at": None,
                "inner_steps": int(event.get("inner_steps", self.inner_steps)),
                "required_runtime": event.get("required_runtime", REQUIREMENT_ANY) or REQUIREMENT_ANY,
                "required_backend": event.get("required_backend", REQUIREMENT_ANY) or REQUIREMENT_ANY,
                "required_protocol_version": (
                    event.get("required_protocol_version", DEFAULT_PROTOCOL_VERSION)
                    or DEFAULT_PROTOCOL_VERSION
                ),
                "workload_type": event.get("workload_type", DEFAULT_WORKLOAD_TYPE) or DEFAULT_WORKLOAD_TYPE,
                "workload_metadata": event.get("workload_metadata", {}) or {},
                "audit_mode": event.get("audit_mode", AUDIT_MODE_NONE) or AUDIT_MODE_NONE,
                "claim_weights": None,
                "claim_training_spec": {},
                "claim_optimizer_spec": {},
                "claim_workload_spec": {},
                "created_at": float(event["ts"]),
                "updated_at": float(event["ts"]),
            }
            return

        task = self._tasks[event["task_id"]]
        task["updated_at"] = float(event["ts"])

        if event_type == EVENT_TASK_CLAIMED:
            task.update({
                "status": STATUS_LEASED,
                "attempt": int(event["attempt"]),
                "lease_token": event["lease_token"],
                "lease_expires_at": float(event["lease_expires_at"]),
                "miner_id": event["miner_id"],
                "capabilities": event.get("capabilities", {}),
                "model_version": int(event["model_version"]),
                "inner_steps": int(event.get("inner_steps", task["inner_steps"])),
                "audit_mode": event.get("audit_mode", task.get("audit_mode", AUDIT_MODE_NONE)) or AUDIT_MODE_NONE,
                "claim_weights": event.get("claim_weights"),
                "claim_training_spec": event.get("claim_training_spec", {}),
                "claim_optimizer_spec": event.get("claim_optimizer_spec", {}),
                "claim_workload_spec": event.get("claim_workload_spec", {}),
            })
        elif event_type == EVENT_TASK_HEARTBEAT:
            task["lease_expires_at"] = float(event["lease_expires_at"])
            task["runtime_status"] = event.get("runtime_status", {})
        elif event_type == EVENT_TASK_COMPLETED:
            task.update({
                "status": STATUS_COMPLETED,
                "lease_token": None,
                "lease_expires_at": None,
                "base_model_version": int(event.get("base_model_version", task.get("model_version") or 0)),
                "result_model_version": int(event.get("result_model_version", 0)),
                "staleness": int(event.get("staleness", 0)),
                "metrics": event.get("metrics", {}),
                "validation": event.get("validation", {}),
                "optimizer": event.get("optimizer", {}),
                "probe_result": event.get("probe_result", {}),
                "adapter_delta": event.get("adapter_delta", {}),
                "bundle_delta": event.get("bundle_delta", {}),
                "inference_result": event.get("inference_result", {}),
                "inference_results": event.get("inference_results", []),
                "external_llm_result": event.get("external_llm_result", {}),
                "external_llm_results": event.get("external_llm_results", []),
                "result_response": event.get("result_response", {}),
                "result_idempotency_key_hash": event.get("result_idempotency_key_hash", ""),
                "result_lease_token_hash": event.get("result_lease_token_hash", ""),
                "result_event_index": int(event.get("event_index", self._event_index)),
                "model_updated": bool(event.get("model_updated", True)),
                "adapter_updated": bool(event.get("adapter_updated", False)),
                "micro_transformer_updated": bool(event.get("micro_transformer_updated", False)),
                "model_bundle_updated": bool(event.get("model_bundle_updated", False)),
                "adapter_step_result": event.get("adapter_step_result"),
                "micro_transformer_step_result": event.get("micro_transformer_step_result"),
                "model_bundle_step_result": event.get("model_bundle_step_result"),
                "completed_at": float(event["ts"]),
            })
        elif event_type == EVENT_TASK_REJECTED:
            task.update({
                "status": STATUS_REJECTED,
                "lease_token": None,
                "lease_expires_at": None,
                "base_model_version": int(event.get("base_model_version", task.get("model_version") or 0)),
                "result_model_version": None,
                "staleness": int(event.get("staleness", 0)),
                "metrics": event.get("metrics", {}),
                "validation": event.get("validation", {}),
                "optimizer": event.get("optimizer", {}),
                "probe_result": event.get("probe_result", {}),
                "adapter_delta": event.get("adapter_delta", {}),
                "bundle_delta": event.get("bundle_delta", {}),
                "inference_result": event.get("inference_result", {}),
                "inference_results": event.get("inference_results", []),
                "external_llm_result": event.get("external_llm_result", {}),
                "external_llm_results": event.get("external_llm_results", []),
                "result_response": event.get("result_response", {}),
                "result_idempotency_key_hash": event.get("result_idempotency_key_hash", ""),
                "result_lease_token_hash": event.get("result_lease_token_hash", ""),
                "result_event_index": int(event.get("event_index", self._event_index)),
                "model_updated": False,
                "adapter_updated": False,
                "micro_transformer_updated": False,
                "model_bundle_updated": False,
                "rejected_at": float(event["ts"]),
            })
        elif event_type == EVENT_TASK_REQUEUED:
            task.update({
                "status": STATUS_QUEUED,
                "lease_token": None,
                "lease_expires_at": None,
                "miner_id": None,
                "model_version": None,
                "base_model_version": None,
                "result_model_version": None,
                "staleness": None,
                "metrics": {},
                "validation": {},
                "probe_result": {},
                "adapter_delta": {},
                "bundle_delta": {},
                "inference_result": {},
                "inference_results": [],
                "external_llm_result": {},
                "external_llm_results": [],
                "result_response": {},
                "result_idempotency_key_hash": "",
                "result_lease_token_hash": "",
                "result_event_index": None,
                "model_updated": False,
                "adapter_updated": False,
                "micro_transformer_updated": False,
                "model_bundle_updated": False,
                "capabilities": {},
                "runtime_status": {},
                "claim_weights": None,
                "claim_training_spec": {},
                "claim_optimizer_spec": {},
                "claim_workload_spec": {},
                "completed_at": None,
                "rejected_at": None,
            })

    def _require_live_lease(self, task_id: str, *, lease_token: str, attempt: int) -> dict:
        task = self._tasks.get(task_id)
        if task is None:
            raise LeaseConflict(f"unknown task {task_id}")
        if task["status"] != STATUS_LEASED:
            raise LeaseConflict(f"task {task_id} is not leased")
        if task.get("lease_token") != lease_token or int(task.get("attempt", -1)) != int(attempt):
            raise LeaseConflict(f"task {task_id} lease token or attempt is stale")
        if float(task.get("lease_expires_at", 0.0)) <= now_epoch():
            raise LeaseConflict(f"task {task_id} lease expired")
        return task

    def _terminal_idempotent_result(
        self,
        task_id: str,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None,
    ) -> dict | None:
        normalized_key = self._normalize_idempotency_key(idempotency_key)
        if not normalized_key:
            return None
        task = self._tasks.get(task_id)
        if task is None or task.get("status") not in {STATUS_COMPLETED, STATUS_REJECTED}:
            return None
        if int(task.get("attempt", -1)) != int(attempt):
            raise LeaseConflict(f"task {task_id} lease token or attempt is stale")
        key_hash = self._hash_secret(normalized_key)
        lease_hash = self._hash_secret(lease_token)
        if (
            task.get("result_idempotency_key_hash") == key_hash
            and task.get("result_lease_token_hash") == lease_hash
        ):
            response = dict(task.get("result_response") or {})
            if not response:
                raise LeaseConflict(f"task {task_id} idempotent result response is unavailable")
            return response
        raise LeaseConflict(f"task {task_id} idempotency key or lease token does not match completed result")

    def _idempotency_event_fields(self, idempotency_key: str | None, *, lease_token: str) -> dict:
        normalized_key = self._normalize_idempotency_key(idempotency_key)
        if not normalized_key:
            return {}
        return {
            "result_idempotency_key_hash": self._hash_secret(normalized_key),
            "result_lease_token_hash": self._hash_secret(lease_token),
        }

    def _normalize_idempotency_key(self, value: str | None) -> str:
        return str(value or "").strip()

    def _hash_secret(self, value: str) -> str:
        return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    def _public_metrics(self, metrics: dict | None) -> dict:
        if not isinstance(metrics, dict):
            return {}
        public = dict(metrics)
        for field in (
            "local_delta",
            "pseudo_gradient",
            "compressed_delta",
            "adapter_delta",
            "bundle_delta",
            "inference_result",
            "inference_results",
            "external_llm_result",
            "external_llm_results",
            "output_text",
        ):
            public.pop(field, None)
        return public

    def _public_task(self, task: dict) -> dict:
        public = dict(task)
        if public.get("lease_token"):
            public["lease_token"] = "<redacted>"
        public.pop("result_idempotency_key_hash", None)
        public.pop("result_lease_token_hash", None)
        public.pop("result_response", None)
        public.pop("bundle_delta", None)
        public.pop("inference_result", None)
        public.pop("inference_results", None)
        public.pop("external_llm_result", None)
        public.pop("external_llm_results", None)
        public["claim_workload_spec"] = self._public_workload_spec(
            public.get("claim_workload_spec"),
            workload_type=self._workload_type(task),
        )
        public["metrics"] = self._public_metrics(public.get("metrics"))
        public["validation"] = self._public_validation(
            public.get("validation"),
            workload_type=self._workload_type(task),
        )
        return public

    def _public_workload_spec(self, workload_spec: dict | None, *, workload_type: str) -> dict:
        if not isinstance(workload_spec, dict):
            return {}
        public = json.loads(json.dumps(workload_spec))
        if workload_type == WORKLOAD_EXTERNAL_LLM_INFER:
            requests = public.get("requests")
            if isinstance(requests, list):
                for request in requests:
                    if isinstance(request, dict) and "prompt" in request:
                        request["prompt"] = "<redacted>"
        return public

    def _public_validation(self, validation: dict | None, *, workload_type: str) -> dict:
        if not isinstance(validation, dict):
            return {}
        public = dict(validation)
        for field in (
            "bundle_delta",
            "inference_result",
            "inference_results",
            "external_llm_result",
            "external_llm_results",
            "output_text",
        ):
            public.pop(field, None)
        if workload_type == WORKLOAD_EXTERNAL_LLM_INFER and "output_preview" in public:
            public["output_preview"] = "<redacted>"
        return public

    def _result_ledger_entry(self, task: dict, workload_scores: dict) -> dict:
        validation = task.get("validation") or {}
        workload_type = self._workload_type(task)
        miner_id = task.get("miner_id") or "anonymous"
        score = workload_scores.get(miner_id, {}).get(workload_type, {})
        terminal_at = task.get("completed_at") or task.get("rejected_at") or task.get("updated_at")
        return {
            "event_index": int(task.get("result_event_index") or 0),
            "task_id": task["task_id"],
            "status": task.get("status"),
            "accepted": task.get("status") == STATUS_COMPLETED,
            "miner_id": miner_id,
            "workload_type": workload_type,
            "attempt": int(task.get("attempt", 0) or 0),
            "base_model_version": task.get("base_model_version"),
            "result_model_version": task.get("result_model_version"),
            "staleness": int(task.get("staleness", 0) or 0),
            "model_updated": bool(task.get("model_updated", False)),
            "adapter_updated": bool(task.get("adapter_updated", False)),
            "micro_transformer_updated": bool(task.get("micro_transformer_updated", False)),
            "model_bundle_updated": bool(task.get("model_bundle_updated", False)),
            "idempotent": bool(task.get("result_idempotency_key_hash")),
            "terminal_at": float(terminal_at or 0.0),
            "validation": self._validation_summary(validation, workload_type=workload_type),
            "session_metrics": self._session_metrics_summary(task.get("metrics") or {}),
            "audit": self._audit_summary(validation),
            "optimizer": self._optimizer_summary(task.get("optimizer") or {}),
            "miner_workload_score": {
                "score": score.get("score"),
                "accepted": score.get("accepted", 0),
                "rejected": score.get("rejected", 0),
                "consecutive_rejections": score.get("consecutive_rejections", 0),
                "quarantined": bool(score.get("quarantined", False)),
                "last_rejection_code": score.get("last_rejection_code"),
            },
        }

    def _validation_summary(self, validation: dict, *, workload_type: str = "") -> dict:
        fields = [
            "accepted",
            "code",
            "reason",
            "delta_norm",
            "loss_before",
            "loss_after",
            "loss_delta",
            "max_delta_norm",
            "max_loss_delta",
            "delta_format",
            "decoded_delta_norm",
            "compression_ratio_estimate",
            "error_feedback",
            "residual_norm",
            "corrected_delta_norm",
            "ops",
            "elapsed_ms",
            "hash",
            "bundle_id",
            "base_bundle_version",
            "artifact_hash",
            "predicted_token_id",
            "predicted_token",
            "target_token_id",
            "target_token",
            "correct",
            "request_count",
            "correct_count",
            "accuracy",
            "request_trace",
            "request_trace_count",
            "request_trace_truncated",
            "scenario_schema",
            "scenario_id",
            "scenario_description",
            "scenario_request_count",
            "completion_count",
            "output_chars",
            "adapter_kind",
            "model_id",
            "output_preview",
        ]
        summary = {
            field: validation.get(field)
            for field in fields
            if field in validation
        }
        if workload_type == WORKLOAD_EXTERNAL_LLM_INFER and "output_preview" in summary:
            summary["output_preview"] = "<redacted>"
        return summary

    def _session_metrics_summary(self, metrics: dict) -> dict:
        fields = [
            "elapsed_ms",
            "request_count",
            "correct_count",
            "accuracy",
            "scenario_schema",
            "scenario_id",
            "scenario_description",
            "scenario_request_count",
            "completion_count",
            "output_chars",
            "requests_per_second",
            "adapter_kind",
            "model_id",
        ]
        return {
            field: metrics.get(field)
            for field in fields
            if field in metrics
        }

    def _audit_summary(self, validation: dict) -> dict:
        fields = [
            "audit_mode",
            "audit_accepted",
            "audit_code",
            "audit_reason",
            "audit_max_abs_error",
            "audit_tolerance",
            "audit_delta_format",
            "audit_expected_delta_norm",
        ]
        return {
            field: validation.get(field)
            for field in fields
            if field in validation
        }

    def _optimizer_summary(self, optimizer: dict) -> dict:
        fields = [
            "contract_version",
            "optimizer_type",
            "delta_format",
            "optimizer_step_before",
            "optimizer_step_after",
            "outer_lr",
            "outer_momentum",
            "weight_count",
            "delta_norm",
            "velocity_norm",
            "outer_update_norm",
            "error_feedback",
            "residual_norm",
            "corrected_delta_norm",
        ]
        return {
            field: optimizer.get(field)
            for field in fields
            if field in optimizer
        }

    def _redact_event(self, event: dict) -> dict:
        def redact(value):
            if isinstance(value, dict):
                redacted = {
                    key: (
                        "<redacted>"
                        if key in {
                            "lease_token",
                            "result_idempotency_key_hash",
                            "result_lease_token_hash",
                            "external_llm_result",
                            "external_llm_results",
                            "output_text",
                        }
                        else redact(item)
                    )
                    for key, item in value.items()
                    if key != "result_response"
                }
                workload_type = str(redacted.get("workload_type") or "")
                claim_spec = redacted.get("claim_workload_spec")
                is_external_llm = (
                    (
                        workload_type == WORKLOAD_EXTERNAL_LLM_INFER
                        or (isinstance(claim_spec, dict) and claim_spec.get("type") == WORKLOAD_EXTERNAL_LLM_INFER)
                        or "external_llm_result" in redacted
                        or "external_llm_results" in redacted
                    )
                )
                if is_external_llm and isinstance(claim_spec, dict):
                    redacted["claim_workload_spec"] = self._public_workload_spec(
                        claim_spec,
                        workload_type=WORKLOAD_EXTERNAL_LLM_INFER,
                    )
                if is_external_llm and isinstance(redacted.get("validation"), dict):
                    validation = dict(redacted["validation"])
                    if "output_preview" in validation:
                        validation["output_preview"] = "<redacted>"
                    redacted["validation"] = validation
                return redacted
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        return redact(event)

    def _trust_override_for(self, miner_id: str, workload_type: str) -> dict | None:
        return self._trust_overrides.get(miner_id, {}).get(workload_type)

    def _trust_decision(self, miner_id: str, workload_type: str) -> dict:
        override = self._trust_override_for(miner_id, workload_type)
        if override and override.get("mode") == TRUST_OVERRIDE_BLOCK:
            return {
                "blocked": True,
                "reason": "miner manually blocked for workload",
                "override_mode": TRUST_OVERRIDE_BLOCK,
            }
        if override and override.get("mode") == TRUST_OVERRIDE_ALLOW:
            return {
                "blocked": False,
                "reason": "miner manually allowed for workload",
                "override_mode": TRUST_OVERRIDE_ALLOW,
            }
        if self._miner_auto_quarantined_for_workload(miner_id, workload_type):
            return {
                "blocked": True,
                "reason": "miner quarantined for workload",
                "override_mode": TRUST_OVERRIDE_NONE,
            }
        return {
            "blocked": False,
            "reason": "eligible",
            "override_mode": TRUST_OVERRIDE_NONE,
        }

    def _blocked_claim_reason(self, decisions: list[dict]) -> str:
        if any(decision.get("reason") == "miner manually blocked for workload" for decision in decisions):
            return "miner manually blocked for workload"
        return "miner quarantined for workload"

    def _public_trust_overrides(self) -> dict:
        return json.loads(json.dumps(self._trust_overrides))

    def _audit_mode_for_workload(self, workload_type: str) -> str:
        if self.replay_audit and workload_type in {
            WORKLOAD_DILOCO_TRAIN,
            WORKLOAD_CPU_LORA_MOCK,
            WORKLOAD_MICRO_TRANSFORMER_LM,
            WORKLOAD_MODEL_BUNDLE_LM,
        }:
            return AUDIT_MODE_REPLAY
        return AUDIT_MODE_NONE

    def _model_version_for_task(self, task: dict) -> int:
        if self._workload_type(task) == WORKLOAD_MICRO_TRANSFORMER_LM:
            return micro_transformer_version(self._model)
        if self._workload_type(task) in {WORKLOAD_MODEL_BUNDLE_LM, WORKLOAD_MODEL_BUNDLE_INFER}:
            return model_bundle_version(self._model)
        return int(self._model["version"])

    def _claim_weights_for_task(self, task: dict, workload_spec: dict) -> list[float]:
        if self._workload_type(task) == WORKLOAD_EXTERNAL_LLM_INFER:
            return []
        if self._workload_type(task) == WORKLOAD_MICRO_TRANSFORMER_LM:
            return list(workload_spec.get("weights", []))
        if self._workload_type(task) in {WORKLOAD_MODEL_BUNDLE_LM, WORKLOAD_MODEL_BUNDLE_INFER}:
            return list(workload_spec.get("weights", []))
        return list(self._model["weights"])

    def _optimizer_spec_for_task(self, task: dict) -> dict:
        if self._workload_type(task) != WORKLOAD_DILOCO_TRAIN:
            return {}
        return {
            **optimizer_claim_spec(self._model),
            "delta_format": self.delta_format,
        }

    def _claim_contract(self, task: dict, miner_id: str, model_version: int) -> dict:
        audit_mode = task.get("audit_mode", AUDIT_MODE_NONE) or AUDIT_MODE_NONE
        optimizer_spec = self._optimizer_spec_for_task(task)
        if self._workload_type(task) in {WORKLOAD_MODEL_BUNDLE_INFER, WORKLOAD_EXTERNAL_LLM_INFER}:
            claim_task = {**task, "miner_id": miner_id, "model_version": model_version}
            workload_spec = self._workload_spec(claim_task)
            return {
                "audit_mode": audit_mode,
                "claim_weights": self._claim_weights_for_task(claim_task, workload_spec),
                "claim_training_spec": {},
                "claim_optimizer_spec": optimizer_spec,
                "claim_workload_spec": workload_spec,
            }
        if audit_mode != AUDIT_MODE_REPLAY:
            return {
                "audit_mode": audit_mode,
                "claim_weights": None,
                "claim_training_spec": {},
                "claim_optimizer_spec": optimizer_spec,
                "claim_workload_spec": {},
            }

        claim_task = {**task, "miner_id": miner_id, "model_version": model_version}
        workload_spec = self._workload_spec(claim_task)
        return {
            "audit_mode": AUDIT_MODE_REPLAY,
            "claim_weights": self._claim_weights_for_task(claim_task, workload_spec),
            "claim_training_spec": training_spec_for(task["task_id"], miner_id, model_version),
            "claim_optimizer_spec": optimizer_spec,
            "claim_workload_spec": workload_spec,
        }

    def _annotate_audit_skip(self, task: dict, validation: dict) -> dict:
        if task.get("audit_mode", AUDIT_MODE_NONE) != AUDIT_MODE_REPLAY:
            return validation
        return {
            **validation,
            "audit_mode": AUDIT_MODE_REPLAY,
            "audit_accepted": None,
            "audit_code": "skipped_prior_validation",
            "audit_reason": "replay audit skipped because base validation rejected the result",
        }

    def _merge_audit_validation(self, validation: dict, audit: dict, mismatch_code: str) -> dict:
        if audit.get("audit_accepted") is False:
            return {
                **validation,
                **audit,
                "accepted": False,
                "code": mismatch_code,
                "reason": audit.get("audit_reason", "replay audit failed"),
            }
        return {**validation, **audit}

    def _validate_delta_format_contract(self, task: dict, validation: dict) -> dict:
        expected = normalize_delta_format(
            (task.get("claim_optimizer_spec") or {}).get("delta_format")
        )
        actual = normalize_delta_format(validation.get("delta_format"))
        if expected == actual:
            return validation
        if (
            task.get("audit_mode", AUDIT_MODE_NONE) == AUDIT_MODE_REPLAY
            and actual == DELTA_FORMAT_SIGN_COMPRESSED_EF
        ):
            return validation
        return {
            **validation,
            "accepted": False,
            "code": "delta_format_mismatch",
            "reason": f"result delta_format {actual} does not match claim delta_format {expected}",
            "expected_delta_format": expected,
            "actual_delta_format": actual,
        }

    def _audit_diloco_result(self, task: dict, validation: dict) -> dict:
        if task.get("audit_mode", AUDIT_MODE_NONE) != AUDIT_MODE_REPLAY:
            return validation
        audit = verify_diloco_replay({**task, "validation": validation}, validation["local_delta"])
        return self._merge_audit_validation(validation, audit, "local_delta_replay_mismatch")

    def _audit_lora_result(self, task: dict, validation: dict) -> dict:
        if task.get("audit_mode", AUDIT_MODE_NONE) != AUDIT_MODE_REPLAY:
            return validation
        audit = verify_lora_replay(task, validation["adapter_delta"])
        return self._merge_audit_validation(validation, audit, "adapter_delta_replay_mismatch")

    def _audit_micro_transformer_result(self, task: dict, validation: dict) -> dict:
        if task.get("audit_mode", AUDIT_MODE_NONE) != AUDIT_MODE_REPLAY:
            return validation
        audit = verify_micro_transformer_replay(task, validation["local_delta"])
        return self._merge_audit_validation(validation, audit, "micro_transformer_delta_replay_mismatch")

    def _audit_model_bundle_result(self, task: dict, validation: dict) -> dict:
        if task.get("audit_mode", AUDIT_MODE_NONE) != AUDIT_MODE_REPLAY:
            return validation
        audit = verify_model_bundle_replay(task, validation["bundle_delta"])
        return self._merge_audit_validation(validation, audit, "model_bundle_delta_replay_mismatch")

    def _task_requirements(self, task: dict) -> dict:
        return {
            "runtime": task.get("required_runtime", REQUIREMENT_ANY) or REQUIREMENT_ANY,
            "backend": task.get("required_backend", REQUIREMENT_ANY) or REQUIREMENT_ANY,
            "protocol_version": (
                task.get("required_protocol_version", DEFAULT_PROTOCOL_VERSION)
                or DEFAULT_PROTOCOL_VERSION
            ),
        }

    def _workload_type(self, task: dict) -> str:
        return task.get("workload_type", DEFAULT_WORKLOAD_TYPE) or DEFAULT_WORKLOAD_TYPE

    def _workload_spec(self, task: dict) -> dict:
        workload_type = self._workload_type(task)
        if workload_type == WORKLOAD_BROWSER_PROBE:
            return dict(BROWSER_PROBE_SPEC)
        if workload_type == WORKLOAD_CPU_LORA_MOCK:
            return lora_training_spec_for(
                task["task_id"],
                task.get("miner_id") or "anonymous",
                self._model,
            )
        if workload_type == WORKLOAD_MICRO_TRANSFORMER_LM:
            return micro_transformer_training_spec_for(
                task["task_id"],
                task.get("miner_id") or "anonymous",
                self._model.get("micro_transformer", {}),
            )
        if workload_type == WORKLOAD_MODEL_BUNDLE_LM:
            return model_bundle_training_spec_for(
                task["task_id"],
                task.get("miner_id") or "anonymous",
                self._model.get("model_bundle", {}),
            )
        if workload_type == WORKLOAD_MODEL_BUNDLE_INFER:
            return model_bundle_inference_spec_for(
                task["task_id"],
                task.get("miner_id") or "anonymous",
                self._model.get("model_bundle", {}),
                request_count=int(task.get("inner_steps", 1)),
                scenario_id=(task.get("workload_metadata") or {}).get("scenario_id"),
            )
        if workload_type == WORKLOAD_EXTERNAL_LLM_INFER:
            return external_llm_inference_spec_for(
                task["task_id"],
                task.get("miner_id") or "anonymous",
                request_count=int(task.get("inner_steps", 1)),
            )
        return {"type": WORKLOAD_DILOCO_TRAIN}

    def _requirement_key(self, task: dict) -> str:
        requirements = self._task_requirements(task)
        return (
            f"{requirements['runtime']}/"
            f"{requirements['backend']}/"
            f"{requirements['protocol_version']}"
        )

    def _lane_key(self, task: dict) -> str:
        return f"{self._requirement_key(task)}/{self._workload_type(task)}"

    def _task_matches_lane(self, task: dict, lane: dict) -> bool:
        requirements = self._task_requirements(task)
        return (
            requirements["runtime"] == lane["runtime"]
            and requirements["backend"] == lane["backend"]
            and requirements["protocol_version"] == lane["protocol_version"]
            and self._workload_type(task) == lane["workload_type"]
        )

    def _capabilities_match(self, task: dict, capabilities: dict) -> bool:
        requirements = self._task_requirements(task)

        required_runtime = requirements["runtime"]
        if required_runtime != REQUIREMENT_ANY and capabilities.get("runtime") != required_runtime:
            return False

        required_backend = requirements["backend"]
        if required_backend != REQUIREMENT_ANY and capabilities.get("backend") != required_backend:
            return False

        required_protocol = requirements["protocol_version"]
        if required_protocol != REQUIREMENT_ANY:
            miner_protocol = capabilities.get("protocol_version")
            if not miner_protocol:
                if required_protocol != DEFAULT_PROTOCOL_VERSION:
                    return False
            elif miner_protocol != required_protocol:
                return False

        workload_type = self._workload_type(task)
        supported_workloads = capabilities.get("supported_workloads")
        if supported_workloads is None:
            supported = {WORKLOAD_DILOCO_TRAIN}
        elif isinstance(supported_workloads, str):
            supported = {supported_workloads}
        else:
            supported = {str(value) for value in supported_workloads}
        if workload_type not in supported:
            return False

        if workload_type != WORKLOAD_DILOCO_TRAIN:
            return True
        expected_format = self._optimizer_spec_for_task(task).get("delta_format", DELTA_FORMAT_DENSE_FLOAT)
        supported_delta_formats = capabilities.get("supported_delta_formats")
        if supported_delta_formats is None:
            return expected_format == DELTA_FORMAT_DENSE_FLOAT
        if isinstance(supported_delta_formats, str):
            formats = {supported_delta_formats}
        else:
            formats = {str(value) for value in supported_delta_formats}
        return expected_format in formats

    def _validate_probe_result(self, probe_result: dict | None) -> dict:
        if not isinstance(probe_result, dict):
            return {
                "accepted": False,
                "code": "probe_result_missing",
                "reason": "browser_probe requires a probe_result object",
                "probe_result": probe_result,
            }

        try:
            ops = int(probe_result.get("ops", 0))
            elapsed_ms = float(probe_result.get("elapsed_ms", probe_result.get("elapsedMs", 0.0)))
        except (TypeError, ValueError):
            return {
                "accepted": False,
                "code": "probe_result_not_numeric",
                "reason": "browser_probe ops and elapsed_ms must be numeric",
                "probe_result": probe_result,
            }

        verified = probe_result.get("verified") is True
        result_hash = str(probe_result.get("hash", ""))
        accepted = verified and ops > 0 and elapsed_ms > 0 and bool(result_hash)
        return {
            "accepted": accepted,
            "code": "ok" if accepted else "probe_result_invalid",
            "reason": "accepted" if accepted else "browser_probe result failed verification",
            "probe_result": probe_result,
            "ops": ops,
            "elapsed_ms": elapsed_ms,
            "hash": result_hash,
        }

    def _task_counts_by_requirement(self) -> dict:
        grouped: dict[str, dict] = {}
        for task in self._tasks.values():
            key = self._requirement_key(task)
            entry = grouped.setdefault(key, {
                STATUS_QUEUED: 0,
                STATUS_LEASED: 0,
                STATUS_COMPLETED: 0,
                STATUS_REJECTED: 0,
            })
            entry[task["status"]] = entry.get(task["status"], 0) + 1
        return grouped

    def _task_counts_by_lane(self) -> dict:
        grouped: dict[str, dict] = {}
        for task in self._tasks.values():
            key = self._lane_key(task)
            entry = grouped.setdefault(key, {
                STATUS_QUEUED: 0,
                STATUS_LEASED: 0,
                STATUS_COMPLETED: 0,
                STATUS_REJECTED: 0,
            })
            entry[task["status"]] = entry.get(task["status"], 0) + 1
        return grouped

    def _miner_scores(self) -> dict:
        scores: dict[str, dict] = {}
        for task in self._tasks.values():
            if task["status"] not in {STATUS_COMPLETED, STATUS_REJECTED}:
                continue
            miner_id = task.get("miner_id") or "anonymous"
            entry = scores.setdefault(miner_id, {
                "accepted": 0,
                "rejected": 0,
                "stale": 0,
                "score": 0.0,
                "staleness_total": 0,
                "avg_staleness": 0.0,
                "last_seen_at": None,
            })
            staleness = int(task.get("staleness", 0) or 0)
            entry["last_seen_at"] = max(
                float(entry["last_seen_at"] or 0.0),
                float(task.get("completed_at") or task.get("rejected_at") or task.get("updated_at") or 0.0),
            )
            if task["status"] == STATUS_COMPLETED:
                entry["accepted"] += 1
                entry["staleness_total"] += staleness
                entry["score"] += 1.0 - 0.05 * staleness
            elif task["status"] == STATUS_REJECTED:
                entry["rejected"] += 1
                entry["score"] -= 2.0

        for entry in scores.values():
            if entry["accepted"]:
                entry["avg_staleness"] = entry["staleness_total"] / entry["accepted"]
            entry["score"] = round(entry["score"], 6)
        return scores

    def _miner_workload_scores(self) -> dict:
        scores: dict[str, dict] = {}
        scored_tasks = [
            task for task in self._tasks.values()
            if task["status"] in {STATUS_COMPLETED, STATUS_REJECTED}
        ]
        scored_tasks.sort(
            key=lambda item: float(
                item.get("completed_at")
                or item.get("rejected_at")
                or item.get("updated_at")
                or 0.0
            )
        )

        for task in scored_tasks:
            miner_id = task.get("miner_id") or "anonymous"
            workload_type = self._workload_type(task)
            miner_scores = scores.setdefault(miner_id, {})
            entry = miner_scores.setdefault(workload_type, {
                "accepted": 0,
                "rejected": 0,
                "consecutive_rejections": 0,
                "score": 0.0,
                "staleness_total": 0,
                "avg_staleness": 0.0,
                "last_seen_at": None,
                "last_rejection_code": None,
                "quarantined": False,
            })
            staleness = int(task.get("staleness", 0) or 0)
            entry["last_seen_at"] = max(
                float(entry["last_seen_at"] or 0.0),
                float(task.get("completed_at") or task.get("rejected_at") or task.get("updated_at") or 0.0),
            )
            if task["status"] == STATUS_COMPLETED:
                entry["accepted"] += 1
                entry["consecutive_rejections"] = 0
                entry["staleness_total"] += staleness
                entry["score"] += 1.0 - 0.05 * staleness
            elif task["status"] == STATUS_REJECTED:
                entry["rejected"] += 1
                entry["consecutive_rejections"] += 1
                entry["score"] -= 2.0
                validation = task.get("validation") or {}
                entry["last_rejection_code"] = validation.get("code")

        for miner_scores in scores.values():
            for entry in miner_scores.values():
                if entry["accepted"]:
                    entry["avg_staleness"] = entry["staleness_total"] / entry["accepted"]
                entry["score"] = round(entry["score"], 6)
                entry["quarantined"] = self._score_is_quarantined(entry)
                entry.pop("staleness_total", None)
        return scores

    def _score_is_quarantined(self, entry: dict) -> bool:
        return (
            int(entry.get("consecutive_rejections", 0)) >= QUARANTINE_CONSECUTIVE_REJECTIONS
            or float(entry.get("score", 0.0)) <= QUARANTINE_SCORE_THRESHOLD
        )

    def _quarantined_miners(self, scores: dict | None = None) -> dict:
        workload_scores = self._miner_workload_scores() if scores is None else scores
        quarantined: dict[str, dict] = {}
        for miner_id, miner_scores in workload_scores.items():
            for workload_type, entry in miner_scores.items():
                if not entry.get("quarantined"):
                    continue
                miner_entry = quarantined.setdefault(miner_id, {})
                miner_entry[workload_type] = {
                    "score": entry["score"],
                    "rejected": entry["rejected"],
                    "consecutive_rejections": entry["consecutive_rejections"],
                    "last_rejection_code": entry.get("last_rejection_code"),
                }
        return quarantined

    def _effective_quarantined_miners(self, quarantined: dict, overrides: dict) -> dict:
        effective = json.loads(json.dumps(quarantined))
        for miner_id, workload_overrides in overrides.items():
            for workload_type, override in workload_overrides.items():
                mode = override.get("mode")
                if mode == TRUST_OVERRIDE_ALLOW:
                    miner_entry = effective.get(miner_id)
                    if miner_entry:
                        miner_entry.pop(workload_type, None)
                        if not miner_entry:
                            effective.pop(miner_id, None)
                elif mode == TRUST_OVERRIDE_BLOCK:
                    effective.setdefault(miner_id, {})[workload_type] = {
                        "manual_block": True,
                        "reason": override.get("reason", ""),
                        "event_index": override.get("event_index"),
                    }
        return effective

    def _manual_blocked_miners(self, overrides: dict) -> dict:
        blocked: dict[str, dict] = {}
        for miner_id, workload_overrides in overrides.items():
            for workload_type, override in workload_overrides.items():
                if override.get("mode") != TRUST_OVERRIDE_BLOCK:
                    continue
                blocked.setdefault(miner_id, {})[workload_type] = dict(override)
        return blocked

    def _miner_auto_quarantined_for_workload(self, miner_id: str, workload_type: str) -> bool:
        entry = self._miner_workload_scores().get(miner_id, {}).get(workload_type)
        return bool(entry and entry.get("quarantined"))

    def _miner_profiles(self) -> dict:
        profiles: dict[str, dict] = {}
        for task in self._tasks.values():
            miner_id = task.get("miner_id")
            if not miner_id:
                continue
            capabilities = dict(task.get("capabilities") or {})
            runtime = capabilities.get("runtime", "unknown")
            backend = capabilities.get("backend", "unknown")
            profile = profiles.setdefault(miner_id, {
                "runtime": runtime,
                "backend": backend,
                "accepted": 0,
                "rejected": 0,
                "leased": 0,
                "avg_staleness": 0.0,
                "staleness_total": 0,
                "avg_worker_elapsed_ms": 0.0,
                "worker_elapsed_total_ms": 0.0,
                "worker_elapsed_count": 0,
                "last_seen_at": None,
                "last_capabilities": {},
                "last_runtime_status": {},
            })
            profile["runtime"] = runtime
            profile["backend"] = backend
            profile["last_capabilities"] = capabilities
            if task.get("runtime_status"):
                profile["last_runtime_status"] = dict(task.get("runtime_status") or {})
            profile["last_seen_at"] = max(
                float(profile["last_seen_at"] or 0.0),
                float(task.get("completed_at") or task.get("rejected_at") or task.get("updated_at") or 0.0),
            )

            status = task.get("status")
            if status == STATUS_COMPLETED:
                profile["accepted"] += 1
                staleness = int(task.get("staleness", 0) or 0)
                profile["staleness_total"] += staleness
                metrics = task.get("metrics") or {}
                elapsed_ms = metrics.get("elapsed_ms")
                if isinstance(elapsed_ms, (int, float)):
                    profile["worker_elapsed_total_ms"] += float(elapsed_ms)
                    profile["worker_elapsed_count"] += 1
            elif status == STATUS_REJECTED:
                profile["rejected"] += 1
            elif status == STATUS_LEASED:
                profile["leased"] += 1

        for profile in profiles.values():
            if profile["accepted"]:
                profile["avg_staleness"] = profile["staleness_total"] / profile["accepted"]
            if profile["worker_elapsed_count"]:
                profile["avg_worker_elapsed_ms"] = (
                    profile["worker_elapsed_total_ms"] / profile["worker_elapsed_count"]
                )
            profile["avg_worker_elapsed_ms"] = round(profile["avg_worker_elapsed_ms"], 6)
            profile.pop("staleness_total", None)
            profile.pop("worker_elapsed_total_ms", None)
            profile.pop("worker_elapsed_count", None)
        return profiles
