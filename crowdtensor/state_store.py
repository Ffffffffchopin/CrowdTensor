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
    EVENT_CONTROL_PLANE_BLOCKED,
    EVENT_INCOMPATIBLE_CLAIM,
    EVENT_TASK_CLAIMED,
    EVENT_TASK_COMPLETED,
    EVENT_TASK_CREATED,
    EVENT_TASK_HEARTBEAT,
    EVENT_TASK_REJECTED,
    EVENT_TASK_REQUEUED,
    EVENT_TRUST_OVERRIDE_SET,
    LeaseConflict,
    MICRO_LLM_SHARDED_BOTH_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
    NoTaskAvailable,
    REAL_LLM_SHARDED_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
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
    WORKLOAD_MICRO_LLM_SHARDED_INFER,
    WORKLOAD_MICRO_TRANSFORMER_LM,
    WORKLOAD_MODEL_BUNDLE_INFER,
    WORKLOAD_MODEL_BUNDLE_LM,
    WORKLOAD_REAL_LLM_SHARDED_INFER,
    WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
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
    micro_llm_sharded_inference_spec_for,
    micro_transformer_artifact_hash,
    micro_transformer_loss,
    micro_transformer_training_spec_for,
    micro_transformer_version,
    validate_micro_llm_sharded_inference,
    validate_micro_transformer_delta,
)
from .micro_llm_artifact import load_micro_llm_artifact, prompt_requests_for_model
from .model_bundle import (
    apply_model_bundle_update,
    model_bundle_inference_spec_for,
    model_bundle_loss,
    model_bundle_training_spec_for,
    model_bundle_version,
    normalize_inference_scenario_id,
    sharded_model_bundle_inference_spec_for,
    validate_model_bundle_delta,
    validate_model_bundle_inference,
    validate_sharded_model_bundle_inference,
)
from .outer_optimizer import (
    DELTA_FORMAT_DENSE_FLOAT,
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    normalize_delta_format,
    optimizer_claim_spec,
)
from .outer_optimizer import decode_delta_payload
from .outer_optimizer import OPTIMIZER_DILOCO_MOMENTUM
from .real_llm import (
    BACKEND_CPU as REAL_LLM_BACKEND_CPU,
    BACKEND_CUDA as REAL_LLM_BACKEND_CUDA,
    DEFAULT_MODEL_ID as DEFAULT_REAL_LLM_MODEL_ID,
    DEFAULT_PROMPTS as DEFAULT_REAL_LLM_PROMPTS,
    PARTITION_MODE_FULL as REAL_LLM_PARTITION_MODE_FULL,
    inspect_real_llm_artifact,
    normalize_backend as normalize_real_llm_backend,
    normalize_partition_mode as normalize_real_llm_partition_mode,
    real_llm_sharded_inference_spec_for,
    validate_real_llm_sharded_inference,
)
from .session_protocol import safe_stream_events
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
        micro_llm_artifact: str | Path | None = None,
        real_llm_model_id: str = DEFAULT_REAL_LLM_MODEL_ID,
        real_llm_backend: str = REAL_LLM_BACKEND_CPU,
        real_llm_partition_mode: str = REAL_LLM_PARTITION_MODE_FULL,
        hf_cache_dir: str | Path | None = None,
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
        self.micro_llm_artifact_path = str(micro_llm_artifact or "")
        self.real_llm_model_id = str(real_llm_model_id or DEFAULT_REAL_LLM_MODEL_ID)
        self.real_llm_backend = normalize_real_llm_backend(real_llm_backend)
        self.real_llm_partition_mode = normalize_real_llm_partition_mode(real_llm_partition_mode)
        self.hf_cache_dir = str(hf_cache_dir or "")
        self._real_llm_artifact_cache: dict[str, dict] = {}
        self._real_llm_stage_affinity: dict[str, dict[int, str]] = {}
        if self.replay_audit and self.delta_format == DELTA_FORMAT_SIGN_COMPRESSED_EF:
            raise ValueError("sign_compressed_ef cannot be used with replay_audit")
        self.task_lanes = self._normalize_task_lanes(task_lanes)
        self._lock = threading.RLock()
        self._event_index = 0
        self._tasks: dict[str, dict] = {}
        self._incompatible_claims: list[dict] = []
        self._blocked_claims: list[dict] = []
        self._claim_events: list[dict] = []
        self._trust_overrides: dict[str, dict[str, dict]] = {}
        self._model = default_model(outer_optimizer_type=self.outer_optimizer)
        if self.micro_llm_artifact_path:
            self._model["micro_transformer"] = load_micro_llm_artifact(self.micro_llm_artifact_path)["model"]

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._load()
        if self.micro_llm_artifact_path:
            self._model["micro_transformer"] = load_micro_llm_artifact(self.micro_llm_artifact_path)["model"]
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
            compatible = self._prefer_real_llm_stage_affinity(compatible, miner_name)
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
            claimed_contract_preview = self._real_llm_stage_affinity_claim(task, miner_name)
            model_version = self._model_version_for_task(task)
            claimed_contract = self._claim_contract(task, miner_name, model_version)
            if claimed_contract_preview:
                workload_spec = dict(claimed_contract.get("claim_workload_spec") or {})
                workload_spec.setdefault("stage_affinity", claimed_contract_preview)
                claimed_contract["claim_workload_spec"] = workload_spec
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

    def record_claim_blocked(
        self,
        miner_id: str,
        *,
        capabilities: dict | None = None,
        reason: str,
        blocked_workloads: list[str] | None = None,
        compatible_task_count: int = 0,
    ) -> dict:
        with self._lock:
            queued_count = sum(1 for task in self._tasks.values() if task["status"] == STATUS_QUEUED)
            event = self._append_event({
                "type": EVENT_CLAIM_BLOCKED,
                "miner_id": str(miner_id or "anonymous"),
                "capabilities": dict(capabilities or {}),
                "queued_task_count": queued_count,
                "compatible_task_count": int(compatible_task_count),
                "blocked_workloads": list(blocked_workloads or []),
                "reason": str(reason or "claim blocked"),
                "ts": now_epoch(),
            })
            self._apply_task_event(event)
            return dict(self._blocked_claims[-1]) if self._blocked_claims else {}

    def record_control_plane_blocked(
        self,
        *,
        reason: str,
        endpoint: str,
        subject: str,
        workload_type: str = "",
        window_seconds: float = 0.0,
        limit: int = 0,
        observed_count: int = 0,
    ) -> dict:
        with self._lock:
            event = self._append_event({
                "type": EVENT_CONTROL_PLANE_BLOCKED,
                "reason": str(reason or "control plane request blocked"),
                "endpoint": str(endpoint or ""),
                "subject": str(subject or "anonymous"),
                "workload_type": str(workload_type or ""),
                "window_seconds": float(window_seconds or 0.0),
                "limit": int(limit or 0),
                "observed_count": int(observed_count or 0),
                "ts": now_epoch(),
            })
            return {
                "event_index": int(event["event_index"]),
                "reason": event["reason"],
                "endpoint": event["endpoint"],
                "subject": event["subject"],
                "workload_type": event["workload_type"],
                "window_seconds": event["window_seconds"],
                "limit": event["limit"],
                "observed_count": event["observed_count"],
            }

    def miner_claim_usage(self, miner_id: str) -> dict:
        miner_name = str(miner_id or "")
        with self._lock:
            leased = 0
            accepted = 0
            rejected = 0
            for task in self._tasks.values():
                if task.get("miner_id") != miner_name:
                    continue
                if task.get("status") == STATUS_LEASED:
                    leased += 1
                elif task.get("status") == STATUS_COMPLETED:
                    accepted += 1
                elif task.get("status") == STATUS_REJECTED:
                    rejected += 1
            return {
                "schema": "miner_claim_usage_v1",
                "miner_id": miner_name,
                "leased": leased,
                "accepted": accepted,
                "rejected": rejected,
                "claim_count": leased + accepted + rejected,
            }

    def miner_claim_rate_usage(
        self,
        miner_id: str,
        *,
        window_seconds: float,
        now: float | None = None,
    ) -> dict:
        miner_name = str(miner_id or "")
        window = max(0.0, float(window_seconds or 0.0))
        current = now_epoch() if now is None else float(now)
        window_started_at = current - window
        with self._lock:
            claimed = 0
            leased = 0
            accepted = 0
            rejected = 0
            requeued_or_reassigned = 0
            for event in self._claim_events:
                if event.get("miner_id") != miner_name:
                    continue
                if float(event.get("claimed_at") or 0.0) < window_started_at:
                    continue
                claimed += 1
                task = self._tasks.get(str(event.get("task_id") or ""))
                if not task or task.get("miner_id") != miner_name:
                    requeued_or_reassigned += 1
                elif task.get("status") == STATUS_LEASED:
                    leased += 1
                elif task.get("status") == STATUS_COMPLETED:
                    accepted += 1
                elif task.get("status") == STATUS_REJECTED:
                    rejected += 1
                else:
                    requeued_or_reassigned += 1
            return {
                "schema": "miner_claim_rate_usage_v1",
                "miner_id": miner_name,
                "window_seconds": window,
                "window_started_at": window_started_at,
                "now": current,
                "claimed": claimed,
                "leased": leased,
                "accepted": accepted,
                "rejected": rejected,
                "requeued_or_reassigned": requeued_or_reassigned,
                "claim_count": claimed,
            }

    def miner_accounting_summary(
        self,
        *,
        limit: int = 50,
        status: str = "any",
        miner_id: str | None = None,
        workload_type: str | None = None,
        session_id: str | None = None,
        created_by_subject: str | None = None,
    ) -> dict:
        capped = min(MAX_EVENT_TAIL_LIMIT, max(0, int(limit)))
        wanted_status = str(status or "any").strip().lower()
        if wanted_status not in {"any", "leased", "accepted", "rejected"}:
            raise ValueError("status must be any, leased, accepted, or rejected")
        wanted_miner = str(miner_id or "").strip()
        wanted_workload = str(workload_type or "").strip()
        wanted_session = str(session_id or "").strip()
        wanted_subject = str(created_by_subject or "").strip()
        with self._lock:
            rows: list[dict] = []
            for task in self._tasks.values():
                row = self._miner_accounting_row(task)
                if not row:
                    continue
                if wanted_status != "any" and row["accounting_status"] != wanted_status:
                    continue
                if wanted_miner and row["miner_id"] != wanted_miner:
                    continue
                if wanted_workload and row["workload_type"] != wanted_workload:
                    continue
                if wanted_session and row.get("session_id") != wanted_session:
                    continue
                if wanted_subject and row.get("created_by_subject") != wanted_subject:
                    continue
                rows.append(row)
            rows.sort(key=lambda item: (float(item.get("recorded_at", 0.0)), int(item.get("event_index", 0))), reverse=True)
            return {
                "schema": "miner_accounting_summary_v1",
                "rows": rows[:capped],
                "row_count": len(rows),
                "limit": capped,
                "status": wanted_status,
                "miner_id": wanted_miner,
                "workload_type": wanted_workload,
                "session_id": wanted_session,
                "created_by_subject": wanted_subject,
                "miner_totals": self._miner_accounting_totals(rows),
                "created_by_subject_totals": self._created_by_subject_accounting_totals(rows),
                "raw_prompts_public": False,
                "raw_outputs_public": False,
                "lease_material_public": False,
                "public_artifact_safe": True,
            }

    def miner_settlement_draft(
        self,
        *,
        limit: int = 50,
        miner_id: str | None = None,
        workload_type: str | None = None,
        session_id: str | None = None,
        created_by_subject: str | None = None,
        unit_price_microcredits: int = 0,
    ) -> dict:
        capped = min(MAX_EVENT_TAIL_LIMIT, max(0, int(limit)))
        wanted_miner = str(miner_id or "").strip()
        wanted_workload = str(workload_type or "").strip()
        wanted_session = str(session_id or "").strip()
        wanted_subject = str(created_by_subject or "").strip()
        price = max(0, int(unit_price_microcredits or 0))
        with self._lock:
            accounting_rows: list[dict] = []
            for task in self._tasks.values():
                row = self._miner_accounting_row(task)
                if not row or row.get("accounting_status") != "accepted":
                    continue
                if wanted_miner and row["miner_id"] != wanted_miner:
                    continue
                if wanted_workload and row["workload_type"] != wanted_workload:
                    continue
                if wanted_session and row.get("session_id") != wanted_session:
                    continue
                if wanted_subject and row.get("created_by_subject") != wanted_subject:
                    continue
                accounting_rows.append(row)
            accounting_rows.sort(
                key=lambda item: (float(item.get("recorded_at", 0.0)), int(item.get("event_index", 0))),
                reverse=True,
            )
            settlement_rows = [
                self._settlement_row_from_accounting(row, unit_price_microcredits=price)
                for row in accounting_rows
            ]
            return {
                "schema": "miner_settlement_draft_v1",
                "rows": settlement_rows[:capped],
                "row_count": len(settlement_rows),
                "limit": capped,
                "miner_id": wanted_miner,
                "workload_type": wanted_workload,
                "session_id": wanted_session,
                "created_by_subject": wanted_subject,
                "unit_price_microcredits": price,
                "currency": "operator_microcredit_v1",
                "settlement_totals": self._settlement_totals(settlement_rows),
                "created_by_subject_totals": self._created_by_subject_settlement_totals(settlement_rows),
                "draft_only": True,
                "payment_executed": False,
                "reward_accounts_public": False,
                "raw_prompts_public": False,
                "raw_outputs_public": False,
                "lease_material_public": False,
                "public_artifact_safe": True,
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
        session_id: str | None = None,
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
        wanted_session_id = str(session_id or "").strip()

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
                if wanted_session_id and row.get("session_id") != wanted_session_id:
                    continue
                filtered.append(row)
                if len(filtered) >= capped:
                    break
            return filtered

    def session_stream_events(
        self,
        *,
        session_id: str,
        max_new_tokens: int | None = None,
        limit: int = 50,
        workload_type: str = WORKLOAD_REAL_LLM_SHARDED_INFER,
    ) -> list[dict]:
        capped = min(MAX_EVENT_TAIL_LIMIT, max(0, int(limit)))
        if capped == 0:
            return []
        wanted_session = str(session_id or "").strip()
        if not wanted_session:
            return []
        wanted_workload = str(workload_type or WORKLOAD_REAL_LLM_SHARDED_INFER).strip()
        with self._lock:
            rows = [
                self._result_ledger_entry(task, {})
                for task in self._tasks.values()
                if task.get("status") == STATUS_COMPLETED
                and self._workload_type(task) == wanted_workload
                and (task.get("workload_metadata") or {}).get("session_id") == wanted_session
            ]
        events_by_key: dict[tuple[str, int], dict] = {}
        for row in rows:
            validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
            if int(validation.get("stage_id") or -1) != 1:
                continue
            generated_count = int(validation.get("generated_token_count") or 0)
            if generated_count <= 0:
                continue
            observed_at = row.get("terminal_at") if row.get("terminal_at") else None
            for event in safe_stream_events(row, max_new_tokens=max_new_tokens, observed_at=observed_at):
                event_count = int(event.get("generated_token_count") or 0)
                if event_count <= 0:
                    continue
                request_key = str(event.get("request_id") or event.get("prompt_hash") or "")
                key = (request_key, event_count)
                previous = events_by_key.get(key)
                if previous is None or int(event.get("generation_step") or 0) >= int(previous.get("generation_step") or 0):
                    events_by_key[key] = event
        return [
            event
            for _key, event in sorted(
                events_by_key.items(),
                key=lambda item: (
                    str(item[0][0]),
                    int(item[0][1]),
                    int(item[1].get("generation_step") or 0),
                ),
            )
        ][:capped]

    def create_readonly_inference_task(
        self,
        *,
        request_count: int = 4,
        scenario_id: str = "",
        required_runtime: str = "python-cli",
        required_backend: str = "cpu",
        required_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        created_by_subject: str = "",
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
            metadata = self._session_created_by_metadata(created_by_subject)
            if scenario:
                metadata["scenario_id"] = scenario
            task_id = self._create_task(
                required_runtime=runtime,
                required_backend=backend,
                required_protocol_version=required_protocol_version,
                workload_type=WORKLOAD_MODEL_BUNDLE_INFER,
                inner_steps=count,
                workload_metadata=metadata,
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

    def create_readonly_external_llm_task(
        self,
        *,
        request_count: int = 4,
        required_runtime: str = "python-cli",
        required_backend: str = "cpu",
        required_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        created_by_subject: str = "",
    ) -> dict:
        count = max(1, min(int(request_count), 8))
        runtime = str(required_runtime or "").strip()
        backend = str(required_backend or "").strip()
        if runtime != "python-cli":
            raise ValueError("read-only external LLM sessions require runtime python-cli")
        if backend != "cpu":
            raise ValueError("read-only external LLM sessions require backend cpu")
        with self._lock:
            metadata = self._session_created_by_metadata(created_by_subject)
            metadata["adapter_contract"] = "external_llm_runtime_v1"
            task_id = self._create_task(
                required_runtime=runtime,
                required_backend=backend,
                required_protocol_version=required_protocol_version,
                workload_type=WORKLOAD_EXTERNAL_LLM_INFER,
                inner_steps=count,
                workload_metadata=metadata,
            )
            task = self._tasks[task_id]
            return {
                "schema": "inference_session_request_v1",
                "accepted": True,
                "task_id": task_id,
                "status": task["status"],
                "workload_type": task["workload_type"],
                "request_count": task["inner_steps"],
                "scenario_id": "",
                "task_requirements": self._task_requirements(task),
            }

    def create_sharded_inference_session(
        self,
        *,
        request_count: int = 4,
        scenario_id: str = "",
        required_runtime: str = "python-cli",
        required_backend: str = "cpu",
        required_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        created_by_subject: str = "",
    ) -> dict:
        count = max(1, min(int(request_count), 8))
        scenario = normalize_inference_scenario_id(scenario_id)
        runtime = str(required_runtime or "").strip()
        backend = str(required_backend or "").strip()
        if runtime != "python-cli":
            raise ValueError("sharded inference sessions require runtime python-cli")
        if backend != "cpu":
            raise ValueError("sharded inference sessions require backend cpu")
        with self._lock:
            session_id = new_task_id().replace("task-", "shard-session-", 1)
            metadata = self._session_created_by_metadata(created_by_subject)
            metadata.update({
                "session_id": session_id,
                "stage_id": 0,
                "stage_count": 2,
                "scenario_id": scenario,
            })
            stage0_id = self._create_task(
                required_runtime=runtime,
                required_backend=backend,
                required_protocol_version=required_protocol_version,
                workload_type=WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
                inner_steps=count,
                workload_metadata=metadata,
            )
            task = self._tasks[stage0_id]
            return {
                "schema": "sharded_inference_session_v1",
                "accepted": True,
                "session_id": session_id,
                "stage_count": 2,
                "stage_0_task_id": stage0_id,
                "stage_1_task_id": "",
                "status": "stage_0_queued",
                "workload_type": WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
                "request_count": task["inner_steps"],
                "scenario_id": scenario,
                "task_requirements": self._task_requirements(task),
            }

    def create_micro_llm_sharded_inference_session(
        self,
        *,
        request_count: int = 4,
        decode_steps: int = 1,
        prompt_texts: list[str] | None = None,
        required_runtime: str = "python-cli",
        required_backend: str = "cpu",
        required_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        created_by_subject: str = "",
    ) -> dict:
        count = max(1, min(int(request_count), 8))
        steps = max(1, min(int(decode_steps), 4))
        runtime = str(required_runtime or "").strip()
        backend = str(required_backend or "").strip()
        if runtime != "python-cli":
            raise ValueError("micro LLM sharded inference sessions require runtime python-cli")
        if backend != "cpu":
            raise ValueError("micro LLM sharded inference sessions require backend cpu")
        with self._lock:
            micro_model = self._model.get("micro_transformer", {})
            requests = prompt_requests_for_model(micro_model, prompt_texts=prompt_texts) if prompt_texts else []
            if requests:
                count = min(count, len(requests))
                requests = requests[:count]
            session_id = new_task_id().replace("task-", "micro-llm-shard-session-", 1)
            metadata = self._session_created_by_metadata(created_by_subject)
            metadata.update({
                "session_id": session_id,
                "stage_id": 0,
                "stage_count": 2,
                "decode_steps": steps,
                "requests": requests,
                "artifact_schema": micro_model.get("artifact_schema", ""),
                "artifact_id": micro_model.get("artifact_id", ""),
                "artifact_version": micro_model.get("artifact_version"),
                "artifact_hash": micro_transformer_artifact_hash(micro_model),
                "tokenizer_schema": micro_model.get("tokenizer_schema", ""),
                "prompt_request_count": len(requests),
            })
            stage0_id = self._create_task(
                required_runtime=runtime,
                required_backend=backend,
                required_protocol_version=required_protocol_version,
                workload_type=WORKLOAD_MICRO_LLM_SHARDED_INFER,
                inner_steps=count,
                workload_metadata=metadata,
            )
            task = self._tasks[stage0_id]
            return {
                "schema": "micro_llm_sharded_session_v1",
                "accepted": True,
                "session_id": session_id,
                "stage_count": 2,
                "stage_0_task_id": stage0_id,
                "stage_1_task_id": "",
                "status": "stage_0_queued",
                "workload_type": WORKLOAD_MICRO_LLM_SHARDED_INFER,
                "request_count": task["inner_steps"],
                "decode_steps": steps,
                "artifact_schema": micro_model.get("artifact_schema", ""),
                "artifact_id": micro_model.get("artifact_id", ""),
                "artifact_version": micro_model.get("artifact_version"),
                "artifact_hash": micro_transformer_artifact_hash(micro_model),
                "tokenizer_schema": micro_model.get("tokenizer_schema", ""),
                "prompt_request_count": len(requests),
                "task_requirements": self._task_requirements(task),
            }

    def create_real_llm_sharded_inference_session(
        self,
        *,
        request_count: int = 1,
        max_new_tokens: int = 1,
        prompt_texts: list[str] | None = None,
        required_runtime: str = "python-cli",
        required_backend: str = "cpu",
        model_id: str | None = None,
        llm_backend: str | None = None,
        partition_mode: str | None = None,
        required_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        created_by_subject: str = "",
    ) -> dict:
        count = max(1, min(int(request_count), 4))
        generation_limit = max(1, min(int(max_new_tokens), 32))
        runtime = str(required_runtime or "").strip()
        backend = str(required_backend or "").strip()
        resolved_llm_backend = normalize_real_llm_backend(llm_backend or self.real_llm_backend)
        resolved_partition_mode = normalize_real_llm_partition_mode(partition_mode or self.real_llm_partition_mode)
        expected_backend = "cuda" if resolved_llm_backend == REAL_LLM_BACKEND_CUDA else "cpu"
        if runtime != "python-cli":
            raise ValueError("real LLM sharded inference sessions require runtime python-cli")
        if backend != expected_backend:
            raise ValueError(f"real LLM sharded inference sessions require backend {expected_backend}")
        with self._lock:
            artifact = self._real_llm_artifact_summary(resolved_llm_backend, model_id=model_id)
            artifact["partition_mode"] = resolved_partition_mode
            prompts = [str(item) for item in (prompt_texts or DEFAULT_REAL_LLM_PROMPTS) if str(item)]
            prompts = prompts[:count] or list(DEFAULT_REAL_LLM_PROMPTS[:count])
            session_id = new_task_id().replace("task-", "real-llm-shard-session-", 1)
            metadata = self._session_created_by_metadata(created_by_subject)
            metadata.update({
                "session_id": session_id,
                "stage_id": 0,
                "stage_count": 2,
                "requests": [],
                "prompt_texts": prompts,
                "max_new_tokens": generation_limit,
                "generation_step": 0,
                "artifact_schema": artifact.get("schema", ""),
                "artifact_hash": artifact.get("artifact_hash", ""),
                "model_id": artifact.get("model_id", self.real_llm_model_id),
                "backend": artifact.get("backend", "hf_transformers_cpu"),
                "partition_mode": resolved_partition_mode,
                "split_index": artifact.get("split_index"),
                "num_hidden_layers": artifact.get("num_hidden_layers"),
                "hidden_size": artifact.get("hidden_size"),
                "real_llm_artifact_ready": True,
            })
            stage0_id = self._create_task(
                required_runtime=runtime,
                required_backend=backend,
                required_protocol_version=required_protocol_version,
                workload_type=WORKLOAD_REAL_LLM_SHARDED_INFER,
                inner_steps=count,
                workload_metadata=metadata,
            )
            task = self._tasks[stage0_id]
            return {
                "schema": "real_llm_sharded_session_v1",
                "accepted": True,
                "session_id": session_id,
                "stage_count": 2,
                "stage_0_task_id": stage0_id,
                "stage_1_task_id": "",
                "status": "stage_0_queued",
                "workload_type": WORKLOAD_REAL_LLM_SHARDED_INFER,
                "request_count": task["inner_steps"],
                "max_new_tokens": generation_limit,
                "generation_step": 0,
                "artifact_schema": artifact.get("schema"),
                "artifact_hash": artifact.get("artifact_hash"),
                "model_id": artifact.get("model_id"),
                "backend": artifact.get("backend"),
                "partition_mode": resolved_partition_mode,
                "split_index": artifact.get("split_index"),
                "num_hidden_layers": artifact.get("num_hidden_layers"),
                "hidden_size": artifact.get("hidden_size"),
                "prompt_request_count": len(prompts),
                "task_requirements": self._task_requirements(task),
            }

    def _session_created_by_metadata(self, created_by_subject: str = "") -> dict:
        subject = str(created_by_subject or "").strip()
        return {"created_by_subject": subject} if subject else {}

    def _real_llm_stage_affinity_key(self, task: dict) -> tuple[str, int] | None:
        if self._workload_type(task) != WORKLOAD_REAL_LLM_SHARDED_INFER:
            return None
        metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
        session_id = str(metadata.get("session_id") or "").strip()
        if not session_id:
            return None
        try:
            stage_id = int(metadata.get("stage_id", -1))
        except (TypeError, ValueError):
            return None
        if stage_id not in {0, 1}:
            return None
        return session_id, stage_id

    def _real_llm_stage_affinity_for(self, task: dict) -> str:
        key = self._real_llm_stage_affinity_key(task)
        if key is None:
            return ""
        session_id, stage_id = key
        metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
        bound = str(metadata.get("stage_affinity_miner_id") or "").strip()
        if bound:
            return bound
        return self._real_llm_stage_affinity_for_session(session_id, stage_id)

    def _real_llm_stage_affinity_for_session(self, session_id: str, stage_id: int) -> str:
        session_affinity = self._real_llm_stage_affinity.get(session_id, {})
        return str(session_affinity.get(stage_id) or "").strip()

    def _remember_real_llm_stage_affinity(self, task: dict, miner_id: str) -> None:
        key = self._real_llm_stage_affinity_key(task)
        if key is None:
            return
        session_id, stage_id = key
        miner_name = str(miner_id or "").strip()
        if not miner_name:
            return
        self._real_llm_stage_affinity.setdefault(session_id, {})[stage_id] = miner_name
        metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
        metadata["stage_affinity_miner_id"] = miner_name
        metadata["stage_affinity_policy"] = "session_stage_sticky_v1"
        task["workload_metadata"] = metadata

    def _prefer_real_llm_stage_affinity(self, tasks: list[dict], miner_id: str) -> list[dict]:
        miner_name = str(miner_id or "").strip()
        unrestricted: list[dict] = []
        matching: list[dict] = []
        for task in tasks:
            if self._workload_type(task) == WORKLOAD_REAL_LLM_SHARDED_INFER and int(task.get("attempt", 0) or 0) > 0:
                unrestricted.append(task)
                continue
            bound = self._real_llm_stage_affinity_for(task)
            if not bound:
                unrestricted.append(task)
            elif bound == miner_name:
                matching.append(task)
        return matching or unrestricted

    def _real_llm_stage_affinity_claim(self, task: dict, miner_id: str) -> dict:
        if self._workload_type(task) != WORKLOAD_REAL_LLM_SHARDED_INFER:
            return {}
        key = self._real_llm_stage_affinity_key(task)
        if key is None:
            return {}
        miner_name = str(miner_id or "").strip()
        bound = miner_name if int(task.get("attempt", 0) or 0) > 0 else (self._real_llm_stage_affinity_for(task) or miner_name)
        if not bound:
            return {}
        return {
            "schema": "real_llm_stage_affinity_v1",
            "policy": "session_stage_sticky_v1",
            "session_id": key[0],
            "stage_id": key[1],
            "miner_id": bound,
            "matched": bound == miner_name,
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
        sharded_inference_result: dict | None = None,
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
            if workload_type == WORKLOAD_SHARDED_MODEL_BUNDLE_INFER:
                return self._complete_sharded_model_bundle_inference_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    sharded_inference_result=sharded_inference_result,
                    metrics=metrics,
                )
            if workload_type == WORKLOAD_MICRO_LLM_SHARDED_INFER:
                return self._complete_micro_llm_sharded_inference_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    sharded_inference_result=sharded_inference_result,
                    metrics=metrics,
                )
            if workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
                return self._complete_real_llm_sharded_inference_task(
                    task,
                    lease_token=lease_token,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    sharded_inference_result=sharded_inference_result,
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

    def _complete_sharded_model_bundle_inference_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        sharded_inference_result: dict | None = None,
        metrics: dict | None = None,
    ) -> dict:
        claim_spec = task.get("claim_workload_spec") or {}
        validation = validate_sharded_model_bundle_inference(
            self._model,
            sharded_inference_result,
            expected_spec=claim_spec,
        )
        staleness = model_bundle_version(self._model) - int(task["model_version"])
        now = now_epoch()
        stage_id = int(claim_spec.get("stage_id", validation.get("stage_id", 0)))
        response = {
            "accepted": True,
            "model_updated": False,
            "model_bundle_updated": False,
            "workload_type": WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
            "model_version": model_bundle_version(self._model),
            "bundle_version": validation.get("base_bundle_version"),
            "bundle_id": validation.get("bundle_id"),
            "artifact_hash": validation.get("artifact_hash"),
            "session_id": validation.get("session_id"),
            "stage_id": stage_id,
            "stage_count": 2,
            "activation_count": int(validation.get("activation_count", 0)),
            "activation_bytes": int(validation.get("activation_bytes", 0)),
            "activation_hashes": list(validation.get("activation_hashes") or []),
            "activation_transport_ready": bool(validation.get("activation_transport_ready", False)),
            "baseline_match": bool(validation.get("baseline_match", stage_id == 0)),
            "request_count": int(validation.get("request_count", 0)),
            "correct_count": int(validation.get("correct_count", 0)),
            "accuracy": float(validation.get("accuracy", 0.0)),
            "scenario_schema": validation.get("scenario_schema"),
            "scenario_id": validation.get("scenario_id"),
            "scenario_description": validation.get("scenario_description"),
            "scenario_request_count": validation.get("scenario_request_count"),
            "staleness": staleness,
        } if validation["accepted"] else dict(validation)
        if validation["accepted"] and stage_id == 1:
            response.update({
                "predicted_token_id": validation.get("predicted_token_id"),
                "predicted_token": validation.get("predicted_token"),
                "target_token_id": validation.get("target_token_id"),
                "target_token": validation.get("target_token"),
                "correct": bool(validation.get("correct", False)),
            })
        event = {
            "type": EVENT_TASK_COMPLETED if validation["accepted"] else EVENT_TASK_REJECTED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "sharded_inference_result": (
                validation.get("sharded_inference_result") if validation["accepted"] else sharded_inference_result
            ),
            "activation_results": validation.get("activation_results") if validation["accepted"] and stage_id == 0 else [],
            "inference_result": validation.get("inference_result") if validation["accepted"] and stage_id == 1 else {},
            "inference_results": validation.get("inference_results") if validation["accepted"] and stage_id == 1 else [],
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
        if validation["accepted"] and stage_id == 0:
            self._create_stage1_task_from_stage0(task["task_id"], validation)
        self.ensure_backlog()
        if not validation["accepted"]:
            raise ResultRejected(validation)
        return response

    def _create_stage1_task_from_stage0(self, stage0_task_id: str, validation: dict) -> str:
        stage0_task = self._tasks[stage0_task_id]
        metadata = dict(stage0_task.get("workload_metadata") or {})
        session_id = str(metadata.get("session_id") or validation.get("session_id") or "")
        for task in self._tasks.values():
            task_metadata = task.get("workload_metadata") or {}
            if (
                self._workload_type(task) == WORKLOAD_SHARDED_MODEL_BUNDLE_INFER
                and str(task_metadata.get("session_id")) == session_id
                and int(task_metadata.get("stage_id", -1)) == 1
            ):
                return task["task_id"]
        return self._create_task(
            required_runtime=stage0_task.get("required_runtime", REQUIREMENT_ANY),
            required_backend=stage0_task.get("required_backend", REQUIREMENT_ANY),
            required_protocol_version=stage0_task.get("required_protocol_version", DEFAULT_PROTOCOL_VERSION),
            workload_type=WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
            inner_steps=int(stage0_task.get("inner_steps", 1)),
            workload_metadata={
                **metadata,
                "stage_id": 1,
                "stage_count": 2,
                "parent_task_id": stage0_task_id,
                "activation_results": validation.get("activation_results") or [],
                "activation_hashes": validation.get("activation_hashes") or [],
                "activation_bytes": int(validation.get("activation_bytes", 0)),
                "artifact_schema": metadata.get("artifact_schema", ""),
                "artifact_id": metadata.get("artifact_id", ""),
                "artifact_version": metadata.get("artifact_version"),
                "artifact_hash": metadata.get("artifact_hash", ""),
                "tokenizer_schema": metadata.get("tokenizer_schema", ""),
            },
        )

    def _complete_micro_llm_sharded_inference_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        sharded_inference_result: dict | None = None,
        metrics: dict | None = None,
    ) -> dict:
        claim_spec = task.get("claim_workload_spec") or {}
        validation = validate_micro_llm_sharded_inference(
            self._model.get("micro_transformer", {}),
            sharded_inference_result,
            expected_spec=claim_spec,
        )
        staleness = micro_transformer_version(self._model) - int(task["model_version"])
        now = now_epoch()
        stage_id = int(claim_spec.get("stage_id", validation.get("stage_id", 0)))
        response = {
            "accepted": True,
            "model_updated": False,
            "model_bundle_updated": False,
            "micro_transformer_updated": False,
            "workload_type": WORKLOAD_MICRO_LLM_SHARDED_INFER,
            "model_version": micro_transformer_version(self._model),
            "base_model_version": validation.get("base_model_version"),
            "artifact_hash": validation.get("artifact_hash"),
            "session_id": validation.get("session_id"),
            "stage_id": stage_id,
            "stage_count": 2,
            "activation_count": int(validation.get("activation_count", 0)),
            "activation_bytes": int(validation.get("activation_bytes", 0)),
            "activation_hashes": list(validation.get("activation_hashes") or []),
            "activation_transport_ready": bool(validation.get("activation_transport_ready", False)),
            "artifact_schema": validation.get("artifact_schema"),
            "artifact_id": validation.get("artifact_id"),
            "artifact_version": validation.get("artifact_version"),
            "tokenizer_schema": validation.get("tokenizer_schema"),
            "baseline_match": bool(validation.get("baseline_match", stage_id == 0)),
            "decoded_tokens_match": bool(validation.get("decoded_tokens_match", stage_id == 0)),
            "request_count": int(validation.get("request_count", 0)),
            "decode_steps": int(validation.get("decode_steps", 0)),
            "generated_token_count": int(validation.get("generated_token_count", 0)),
            "generated_token_ids": list(validation.get("generated_token_ids") or []),
            "generated_text": validation.get("generated_text", ""),
            "staleness": staleness,
        } if validation["accepted"] else dict(validation)
        event = {
            "type": EVENT_TASK_COMPLETED if validation["accepted"] else EVENT_TASK_REJECTED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "sharded_inference_result": (
                validation.get("sharded_inference_result") if validation["accepted"] else sharded_inference_result
            ),
            "activation_results": validation.get("activation_results") if validation["accepted"] and stage_id == 0 else [],
            "inference_result": validation.get("inference_result") if validation["accepted"] and stage_id == 1 else {},
            "inference_results": validation.get("inference_results") if validation["accepted"] and stage_id == 1 else [],
            "metrics": metrics or {},
            "staleness": staleness,
            "validation": validation,
            "result_model_version": micro_transformer_version(self._model),
            "model_updated": False,
            "model_bundle_updated": False,
            "micro_transformer_updated": False,
            "result_response": response,
            **self._idempotency_event_fields(
                idempotency_key,
                lease_token=lease_token,
            ),
            "ts": now,
        }
        self._apply_task_event(self._append_event(event))
        if validation["accepted"] and stage_id == 0:
            self._create_micro_llm_stage1_task_from_stage0(task["task_id"], validation)
        self.ensure_backlog()
        if not validation["accepted"]:
            raise ResultRejected(validation)
        return response

    def _create_micro_llm_stage1_task_from_stage0(self, stage0_task_id: str, validation: dict) -> str:
        stage0_task = self._tasks[stage0_task_id]
        metadata = dict(stage0_task.get("workload_metadata") or {})
        session_id = str(metadata.get("session_id") or validation.get("session_id") or "")
        for task in self._tasks.values():
            task_metadata = task.get("workload_metadata") or {}
            if (
                self._workload_type(task) == WORKLOAD_MICRO_LLM_SHARDED_INFER
                and str(task_metadata.get("session_id")) == session_id
                and int(task_metadata.get("stage_id", -1)) == 1
            ):
                return task["task_id"]
        return self._create_task(
            required_runtime=stage0_task.get("required_runtime", REQUIREMENT_ANY),
            required_backend=stage0_task.get("required_backend", REQUIREMENT_ANY),
            required_protocol_version=stage0_task.get("required_protocol_version", DEFAULT_PROTOCOL_VERSION),
            workload_type=WORKLOAD_MICRO_LLM_SHARDED_INFER,
            inner_steps=int(stage0_task.get("inner_steps", 1)),
            workload_metadata={
                **metadata,
                "stage_id": 1,
                "stage_count": 2,
                "parent_task_id": stage0_task_id,
                "requests": list((stage0_task.get("claim_workload_spec") or {}).get("requests") or []),
                "activation_results": validation.get("activation_results") or [],
                "activation_hashes": validation.get("activation_hashes") or [],
                "activation_bytes": int(validation.get("activation_bytes", 0)),
                "artifact_schema": metadata.get("artifact_schema", ""),
                "artifact_id": metadata.get("artifact_id", ""),
                "artifact_version": metadata.get("artifact_version"),
                "artifact_hash": metadata.get("artifact_hash", ""),
                "tokenizer_schema": metadata.get("tokenizer_schema", ""),
            },
        )

    def _complete_real_llm_sharded_inference_task(
        self,
        task: dict,
        *,
        lease_token: str,
        attempt: int,
        idempotency_key: str | None = None,
        sharded_inference_result: dict | None = None,
        metrics: dict | None = None,
    ) -> dict:
        claim_spec = task.get("claim_workload_spec") or {}
        validation = validate_real_llm_sharded_inference(
            sharded_inference_result,
            expected_spec=claim_spec,
            cache_dir=self.hf_cache_dir,
            replay_runtime=(
                str(claim_spec.get("backend") or REAL_LLM_BACKEND_CPU) != REAL_LLM_BACKEND_CUDA
            ),
        )
        staleness = 0
        now = now_epoch()
        stage_id = int(claim_spec.get("stage_id", validation.get("stage_id", 0)))
        response = {
            "accepted": True,
            "model_updated": False,
            "model_bundle_updated": False,
            "micro_transformer_updated": False,
            "workload_type": WORKLOAD_REAL_LLM_SHARDED_INFER,
            "model_version": int(task.get("model_version") or 0),
            "artifact_schema": validation.get("artifact_schema"),
            "artifact_hash": validation.get("artifact_hash"),
            "model_id": validation.get("model_id"),
            "backend": validation.get("backend"),
            "partition_mode": validation.get("partition_mode", REAL_LLM_PARTITION_MODE_FULL),
            "stage_layer_range": list(validation.get("stage_layer_range") or []),
            "stage_parameter_count": int(validation.get("stage_parameter_count", 0)),
            "full_model_parameter_count": int(validation.get("full_model_parameter_count", 0)),
            "stage_parameter_fraction": validation.get("stage_parameter_fraction"),
            "device_parameter_count": int(validation.get("device_parameter_count", 0)),
            "partition_parameter_split_valid": bool(validation.get("partition_parameter_split_valid", False)),
            "stage_local_partition_ready": bool(validation.get("stage_local_partition_ready", False)),
            "stage0_partition_loaded": bool(validation.get("stage0_partition_loaded", False)),
            "stage1_partition_loaded": bool(validation.get("stage1_partition_loaded", False)),
            "stage_gpu_memory_reduced": bool(validation.get("stage_gpu_memory_reduced", False)),
            "stage_cpu_partition_ready": bool(validation.get("stage_cpu_partition_ready", False)),
            "baseline_device": validation.get("baseline_device", ""),
            "split_index": validation.get("split_index"),
            "session_id": validation.get("session_id"),
            "stage_id": stage_id,
            "stage_count": 2,
            "max_new_tokens": int(validation.get("max_new_tokens", 1)),
            "generation_step": int(validation.get("generation_step", 0)),
            "activation_count": int(validation.get("activation_count", 0)),
            "activation_bytes": int(validation.get("activation_bytes", 0)),
            "activation_hashes": list(validation.get("activation_hashes") or []),
            "activation_transport_ready": bool(validation.get("activation_transport_ready", False)),
            "real_llm_artifact_ready": bool(validation.get("real_llm_artifact_ready", False)),
            "runtime_replay_performed": bool(validation.get("runtime_replay_performed", False)),
            "remote_runtime_validation": bool(validation.get("remote_runtime_validation", False)),
            "baseline_match": bool(validation.get("baseline_match", stage_id == 0)),
            "decoded_tokens_match": bool(validation.get("decoded_tokens_match", stage_id == 0)),
            "request_count": int(validation.get("request_count", 0)),
            "generated_token_count": int(validation.get("generated_token_count", 0)),
            "generated_token_ids": list(validation.get("generated_token_ids") or []),
            "generated_text": validation.get("generated_text", ""),
            "generated_text_hash": validation.get("generated_text_hash", ""),
            "staleness": staleness,
        } if validation["accepted"] else dict(validation)
        event = {
            "type": EVENT_TASK_COMPLETED if validation["accepted"] else EVENT_TASK_REJECTED,
            "task_id": task["task_id"],
            "attempt": attempt,
            "lease_token": lease_token,
            "miner_id": task.get("miner_id"),
            "base_model_version": int(task["model_version"]),
            "sharded_inference_result": (
                validation.get("sharded_inference_result") if validation["accepted"] else sharded_inference_result
            ),
            "activation_results": validation.get("activation_results") if validation["accepted"] and stage_id == 0 else [],
            "inference_result": validation.get("inference_result") if validation["accepted"] and stage_id == 1 else {},
            "inference_results": validation.get("inference_results") if validation["accepted"] and stage_id == 1 else [],
            "metrics": metrics or {},
            "staleness": staleness,
            "validation": validation,
            "result_model_version": int(task.get("model_version") or 0),
            "model_updated": False,
            "model_bundle_updated": False,
            "micro_transformer_updated": False,
            "result_response": response,
            **self._idempotency_event_fields(
                idempotency_key,
                lease_token=lease_token,
            ),
            "ts": now,
        }
        self._apply_task_event(self._append_event(event))
        if validation["accepted"] and stage_id == 0:
            self._create_real_llm_stage1_task_from_stage0(task["task_id"], validation)
        if validation["accepted"] and stage_id == 1:
            self._create_next_real_llm_stage0_task_from_stage1(task["task_id"], validation)
        self.ensure_backlog()
        if not validation["accepted"]:
            raise ResultRejected(validation)
        return response

    def _create_real_llm_stage1_task_from_stage0(self, stage0_task_id: str, validation: dict) -> str:
        stage0_task = self._tasks[stage0_task_id]
        metadata = dict(stage0_task.get("workload_metadata") or {})
        session_id = str(metadata.get("session_id") or validation.get("session_id") or "")
        generation_step = int(metadata.get("generation_step", validation.get("generation_step", 0)))
        for task in self._tasks.values():
            task_metadata = task.get("workload_metadata") or {}
            if (
                self._workload_type(task) == WORKLOAD_REAL_LLM_SHARDED_INFER
                and str(task_metadata.get("session_id")) == session_id
                and int(task_metadata.get("stage_id", -1)) == 1
                and int(task_metadata.get("generation_step", 0)) == generation_step
            ):
                return task["task_id"]
        return self._create_task(
            required_runtime=stage0_task.get("required_runtime", REQUIREMENT_ANY),
            required_backend=stage0_task.get("required_backend", REQUIREMENT_ANY),
            required_protocol_version=stage0_task.get("required_protocol_version", DEFAULT_PROTOCOL_VERSION),
            workload_type=WORKLOAD_REAL_LLM_SHARDED_INFER,
            inner_steps=int(stage0_task.get("inner_steps", 1)),
            workload_metadata={
                **metadata,
                "stage_id": 1,
                "stage_count": 2,
                "parent_task_id": stage0_task_id,
                "generation_step": generation_step,
                "max_new_tokens": int(metadata.get("max_new_tokens", 1)),
                "stage_affinity_miner_id": self._real_llm_stage_affinity_for_session(session_id, 1),
                "stage_affinity_policy": "session_stage_sticky_v1",
                "requests": list((stage0_task.get("claim_workload_spec") or {}).get("requests") or []),
                "activation_results": validation.get("activation_results") or [],
                "activation_hashes": validation.get("activation_hashes") or [],
                "activation_bytes": int(validation.get("activation_bytes", 0)),
                "artifact_schema": metadata.get("artifact_schema", ""),
                "artifact_hash": metadata.get("artifact_hash", ""),
                "model_id": metadata.get("model_id", self.real_llm_model_id),
                "backend": metadata.get("backend", "hf_transformers_cpu"),
                "partition_mode": metadata.get("partition_mode", REAL_LLM_PARTITION_MODE_FULL),
                "split_index": metadata.get("split_index"),
                "num_hidden_layers": metadata.get("num_hidden_layers"),
                "hidden_size": metadata.get("hidden_size"),
                "real_llm_artifact_ready": metadata.get("real_llm_artifact_ready", True),
            },
        )

    def _create_next_real_llm_stage0_task_from_stage1(self, stage1_task_id: str, validation: dict) -> str:
        stage1_task = self._tasks[stage1_task_id]
        metadata = dict(stage1_task.get("workload_metadata") or {})
        session_id = str(metadata.get("session_id") or validation.get("session_id") or "")
        current_step = int(metadata.get("generation_step", validation.get("generation_step", 0)))
        max_new_tokens = max(1, min(int(metadata.get("max_new_tokens", validation.get("max_new_tokens", 1))), 32))
        next_step = current_step + 1
        if next_step >= max_new_tokens:
            return ""
        results = list(validation.get("inference_results") or [])
        if not results and isinstance(validation.get("inference_result"), dict):
            results = [dict(validation["inference_result"])]
        previous_requests = list(metadata.get("requests") or [])
        next_requests: list[dict] = []
        for index, previous in enumerate(previous_requests):
            result = results[index] if index < len(results) and isinstance(results[index], dict) else {}
            token_text = str(result.get("next_token_text") or "")
            base_prompt = str(previous.get("prompt") or "")
            generated_text = str(result.get("generated_text") or (str(previous.get("generated_text") or "") + token_text))
            token_ids = list(result.get("generated_token_ids") or list(previous.get("generated_token_ids") or []))
            next_requests.append({
                "request_id": str(previous.get("request_id") or f"req-{index + 1}"),
                "prompt": base_prompt[:256],
                "prompt_hash": str(previous.get("prompt_hash") or ""),
                "max_new_tokens": max_new_tokens,
                "generated_token_ids": token_ids,
                "generated_text": generated_text,
                "generation_step": next_step,
            })
        for task in self._tasks.values():
            task_metadata = task.get("workload_metadata") or {}
            if (
                self._workload_type(task) == WORKLOAD_REAL_LLM_SHARDED_INFER
                and str(task_metadata.get("session_id")) == session_id
                and int(task_metadata.get("stage_id", -1)) == 0
                and int(task_metadata.get("generation_step", 0)) == next_step
            ):
                return task["task_id"]
        return self._create_task(
            required_runtime=stage1_task.get("required_runtime", REQUIREMENT_ANY),
            required_backend=stage1_task.get("required_backend", REQUIREMENT_ANY),
            required_protocol_version=stage1_task.get("required_protocol_version", DEFAULT_PROTOCOL_VERSION),
            workload_type=WORKLOAD_REAL_LLM_SHARDED_INFER,
            inner_steps=int(stage1_task.get("inner_steps", 1)),
            workload_metadata={
                **metadata,
                "stage_id": 0,
                "stage_count": 2,
                "parent_task_id": stage1_task_id,
                "generation_step": next_step,
                "max_new_tokens": max_new_tokens,
                "stage_affinity_miner_id": self._real_llm_stage_affinity_for_session(session_id, 0),
                "stage_affinity_policy": "session_stage_sticky_v1",
                "requests": next_requests,
                "prompt_texts": [],
                "activation_results": [],
                "activation_hashes": [],
                "activation_bytes": 0,
                "artifact_schema": metadata.get("artifact_schema", ""),
                "artifact_hash": metadata.get("artifact_hash", ""),
                "model_id": metadata.get("model_id", self.real_llm_model_id),
                "backend": metadata.get("backend", "hf_transformers_cpu"),
                "partition_mode": metadata.get("partition_mode", REAL_LLM_PARTITION_MODE_FULL),
                "split_index": metadata.get("split_index"),
                "num_hidden_layers": metadata.get("num_hidden_layers"),
                "hidden_size": metadata.get("hidden_size"),
                "real_llm_artifact_ready": metadata.get("real_llm_artifact_ready", True),
            },
        )

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
                "sharded_inference_result": {},
                "activation_results": [],
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
                "claimed_at": None,
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
            self._claim_events.append({
                "event_index": int(event.get("event_index", self._event_index)),
                "task_id": event.get("task_id", ""),
                "miner_id": event.get("miner_id") or "anonymous",
                "workload_type": self._workload_type(task),
                "claimed_at": float(event["ts"]),
            })
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
                "claimed_at": float(event["ts"]),
                "claim_weights": event.get("claim_weights"),
                "claim_training_spec": event.get("claim_training_spec", {}),
                "claim_optimizer_spec": event.get("claim_optimizer_spec", {}),
                "claim_workload_spec": event.get("claim_workload_spec", {}),
            })
            self._remember_real_llm_stage_affinity(task, event.get("miner_id") or "anonymous")
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
                "sharded_inference_result": event.get("sharded_inference_result", {}),
                "activation_results": event.get("activation_results", []),
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
                "sharded_inference_result": event.get("sharded_inference_result", {}),
                "activation_results": event.get("activation_results", []),
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
                "sharded_inference_result": {},
                "activation_results": [],
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
                "claimed_at": None,
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
            "sharded_inference_result",
            "activation_result",
            "activation_results",
            "hidden_state",
            "input_ids",
            "logits",
            "output_text",
        ):
            public.pop(field, None)
        return public

    def _public_task(self, task: dict) -> dict:
        public = dict(task)
        workload_type = self._workload_type(task)
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
        public.pop("sharded_inference_result", None)
        public.pop("activation_results", None)
        if workload_type in {
            WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
            WORKLOAD_MICRO_LLM_SHARDED_INFER,
            WORKLOAD_REAL_LLM_SHARDED_INFER,
        } and isinstance(public.get("workload_metadata"), dict):
            metadata = dict(public["workload_metadata"])
            metadata.pop("activation_results", None)
            metadata.pop("activation_result", None)
            metadata.pop("prompt_texts", None)
            if workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER and isinstance(metadata.get("requests"), list):
                requests = json.loads(json.dumps(metadata["requests"]))
                for request in requests:
                    if isinstance(request, dict) and "prompt" in request:
                        request["prompt"] = "<redacted>"
                metadata["requests"] = requests
            public["workload_metadata"] = metadata
        public["claim_workload_spec"] = self._public_workload_spec(
            public.get("claim_workload_spec"),
            workload_type=workload_type,
        )
        public["metrics"] = self._public_metrics(public.get("metrics"))
        public["validation"] = self._public_validation(
            public.get("validation"),
            workload_type=workload_type,
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
        if workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
            requests = public.get("requests")
            if isinstance(requests, list):
                for request in requests:
                    if isinstance(request, dict) and "prompt" in request:
                        request["prompt"] = "<redacted>"
            artifact = public.get("artifact")
            if isinstance(artifact, dict):
                public["artifact"] = {
                    key: artifact.get(key)
                    for key in (
                        "schema",
                        "model_id",
                        "backend",
                        "model_type",
                        "num_hidden_layers",
                        "hidden_size",
                        "vocab_size",
                        "split_index",
                        "artifact_hash",
                        "read_only",
                    )
                    if key in artifact
                }
        if workload_type in {
            WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
            WORKLOAD_MICRO_LLM_SHARDED_INFER,
            WORKLOAD_REAL_LLM_SHARDED_INFER,
        }:
            public.pop("weights", None)
            public.pop("activation_results", None)
            public.pop("activation_result", None)
            public.pop("hidden_state", None)
            public.pop("input_ids", None)
            if "activation_hashes" in public:
                public["activation_hashes"] = list(public.get("activation_hashes") or [])
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
            "sharded_inference_result",
            "activation_result",
            "activation_results",
            "hidden_state",
            "input_ids",
            "logits",
            "output_text",
        ):
            public.pop(field, None)
        if workload_type == WORKLOAD_EXTERNAL_LLM_INFER and "output_preview" in public:
            public["output_preview"] = "<redacted>"
        return public

    def _result_ledger_entry(self, task: dict, workload_scores: dict) -> dict:
        validation = task.get("validation") or {}
        metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
        workload_type = self._workload_type(task)
        miner_id = task.get("miner_id") or "anonymous"
        score = workload_scores.get(miner_id, {}).get(workload_type, {})
        terminal_at = task.get("completed_at") or task.get("rejected_at") or task.get("updated_at")
        return {
            "event_index": int(task.get("result_event_index") or 0),
            "task_id": task["task_id"],
            "session_id": metadata.get("session_id"),
            "created_by_subject": metadata.get("created_by_subject", ""),
            "parent_task_id": metadata.get("parent_task_id"),
            "stage_id": (task.get("validation") or {}).get("stage_id"),
            "stage_count": (task.get("validation") or {}).get("stage_count"),
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

    def _miner_accounting_row(self, task: dict) -> dict:
        miner_id = task.get("miner_id")
        if not miner_id or task.get("status") not in {STATUS_LEASED, STATUS_COMPLETED, STATUS_REJECTED}:
            return {}
        status = task.get("status")
        accounting_status = {
            STATUS_LEASED: "leased",
            STATUS_COMPLETED: "accepted",
            STATUS_REJECTED: "rejected",
        }.get(status, "unknown")
        workload_type = self._workload_type(task)
        validation = task.get("validation") if isinstance(task.get("validation"), dict) else {}
        metrics = task.get("metrics") if isinstance(task.get("metrics"), dict) else {}
        metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
        stage_id = validation.get("stage_id", metadata.get("stage_id"))
        elapsed_ms = metrics.get("elapsed_ms", validation.get("elapsed_ms"))
        work_units = self._accounting_work_units(task, validation, metrics, metadata)
        recorded_at = (
            task.get("completed_at")
            or task.get("rejected_at")
            or task.get("claimed_at")
            or task.get("lease_expires_at")
            or task.get("updated_at")
            or 0.0
        )
        return {
            "schema": "miner_accounting_row_v1",
            "event_index": int(task.get("result_event_index") or 0),
            "task_id": task.get("task_id"),
            "session_id": metadata.get("session_id"),
            "created_by_subject": metadata.get("created_by_subject", ""),
            "miner_id": str(miner_id),
            "workload_type": workload_type,
            "accounting_status": accounting_status,
            "accepted": accounting_status == "accepted",
            "attempt": int(task.get("attempt", 0) or 0),
            "stage_id": stage_id,
            "stage_count": validation.get("stage_count", metadata.get("stage_count")),
            "backend": validation.get("backend", metadata.get("backend", (task.get("capabilities") or {}).get("backend", ""))),
            "model_id": validation.get("model_id", metadata.get("model_id", "")),
            "work_units": work_units,
            "elapsed_ms": round(float(elapsed_ms), 6) if isinstance(elapsed_ms, (int, float)) else None,
            "recorded_at": float(recorded_at or 0.0),
            "claimed_at": float(task.get("claimed_at") or 0.0) if task.get("claimed_at") is not None else None,
            "model_updated": bool(task.get("model_updated", False)),
            "read_only": not bool(
                task.get("model_updated")
                or task.get("adapter_updated")
                or task.get("micro_transformer_updated")
                or task.get("model_bundle_updated")
            ),
            "raw_payload_public": False,
        }

    def _accounting_work_units(self, task: dict, validation: dict, metrics: dict, metadata: dict) -> dict:
        workload_type = self._workload_type(task)
        if workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
            generated = validation.get("generated_token_count", metrics.get("generated_token_count"))
            activation_count = validation.get("activation_count", metrics.get("activation_count"))
            return {
                "unit": "generated_token_or_stage_row",
                "generated_token_count": int(generated or 0),
                "activation_count": int(activation_count or 0),
                "generation_step": int(validation.get("generation_step", metadata.get("generation_step", 0)) or 0),
                "max_new_tokens": int(validation.get("max_new_tokens", metadata.get("max_new_tokens", 0)) or 0),
            }
        if workload_type in {WORKLOAD_MODEL_BUNDLE_INFER, WORKLOAD_EXTERNAL_LLM_INFER}:
            request_count = validation.get("request_count", metrics.get("request_count", task.get("inner_steps", 0)))
            return {
                "unit": "request",
                "request_count": int(request_count or 0),
            }
        if workload_type in {WORKLOAD_SHARDED_MODEL_BUNDLE_INFER, WORKLOAD_MICRO_LLM_SHARDED_INFER}:
            return {
                "unit": "stage_task",
                "stage_rows": int(validation.get("activation_count", metrics.get("activation_count", 1)) or 1),
            }
        return {
            "unit": "inner_step",
            "inner_steps": int(task.get("inner_steps", 0) or 0),
        }

    def _miner_accounting_totals(self, rows: list[dict]) -> dict:
        totals: dict[str, dict] = {}
        for row in rows:
            miner_id = row["miner_id"]
            workload_type = row["workload_type"]
            key = f"{miner_id}/{workload_type}"
            item = totals.setdefault(key, {
                "miner_id": miner_id,
                "workload_type": workload_type,
                "leased": 0,
                "accepted": 0,
                "rejected": 0,
                "elapsed_ms": 0.0,
                "work_units": {},
            })
            status = row.get("accounting_status")
            if status in {"leased", "accepted", "rejected"}:
                item[status] += 1
            if isinstance(row.get("elapsed_ms"), (int, float)):
                item["elapsed_ms"] = round(float(item["elapsed_ms"]) + float(row["elapsed_ms"]), 6)
            self._add_accounting_work_units(item["work_units"], row.get("work_units"))
        return totals

    def _created_by_subject_accounting_totals(self, rows: list[dict]) -> dict:
        totals: dict[str, dict] = {}
        for row in rows:
            subject = str(row.get("created_by_subject") or "").strip()
            if not subject:
                continue
            workload_type = row["workload_type"]
            key = f"{subject}/{workload_type}"
            item = totals.setdefault(key, {
                "created_by_subject": subject,
                "workload_type": workload_type,
                "leased": 0,
                "accepted": 0,
                "rejected": 0,
                "elapsed_ms": 0.0,
                "work_units": {},
            })
            status = row.get("accounting_status")
            if status in {"leased", "accepted", "rejected"}:
                item[status] += 1
            if isinstance(row.get("elapsed_ms"), (int, float)):
                item["elapsed_ms"] = round(float(item["elapsed_ms"]) + float(row["elapsed_ms"]), 6)
            self._add_accounting_work_units(item["work_units"], row.get("work_units"))
        return totals

    def _add_accounting_work_units(self, target: dict, work_units: object) -> None:
        if not isinstance(work_units, dict):
            return
        for unit_key, value in work_units.items():
            if unit_key == "unit" or not isinstance(value, int):
                continue
            target[unit_key] = int(target.get(unit_key, 0)) + int(value)

    def _settlement_row_from_accounting(self, row: dict, *, unit_price_microcredits: int) -> dict:
        reward_unit, reward_units = self._settlement_reward_units(row.get("work_units") or {})
        amount = int(reward_units) * int(unit_price_microcredits)
        return {
            "schema": "miner_settlement_row_v1",
            "task_id": row.get("task_id"),
            "session_id": row.get("session_id"),
            "created_by_subject": row.get("created_by_subject", ""),
            "miner_id": row.get("miner_id"),
            "workload_type": row.get("workload_type"),
            "stage_id": row.get("stage_id"),
            "backend": row.get("backend"),
            "model_id": row.get("model_id"),
            "accepted": True,
            "accounting_event_index": int(row.get("event_index") or 0),
            "recorded_at": float(row.get("recorded_at") or 0.0),
            "reward_unit": reward_unit,
            "reward_units": int(reward_units),
            "unit_price_microcredits": int(unit_price_microcredits),
            "reward_amount_microcredits": amount,
            "work_units": json.loads(json.dumps(row.get("work_units") or {})),
            "reward_account_present": False,
            "settlement_status": "policy_not_joined",
            "draft_only": True,
            "payment_executed": False,
            "raw_payload_public": False,
        }

    def _settlement_reward_units(self, work_units: dict) -> tuple[str, int]:
        unit = str(work_units.get("unit") or "work_unit")
        if unit == "generated_token_or_stage_row":
            generated = int(work_units.get("generated_token_count") or 0)
            if generated > 0:
                return "generated_token", generated
            activations = int(work_units.get("activation_count") or 0)
            if activations > 0:
                return "activation_row", activations
            return "stage_row", 1
        if unit == "request":
            return "request", max(0, int(work_units.get("request_count") or 0))
        if unit == "stage_task":
            return "stage_row", max(0, int(work_units.get("stage_rows") or 0))
        if unit == "inner_step":
            return "inner_step", max(0, int(work_units.get("inner_steps") or 0))
        ignored = {"unit", "generation_step", "max_new_tokens"}
        fallback = sum(int(value) for key, value in work_units.items() if key not in ignored and isinstance(value, int))
        return unit, max(0, fallback)

    def _settlement_totals(self, rows: list[dict]) -> dict:
        totals: dict[str, dict] = {}
        for row in rows:
            miner_id = str(row.get("miner_id") or "")
            workload_type = str(row.get("workload_type") or "")
            key = f"{miner_id}/{workload_type}"
            item = totals.setdefault(key, {
                "miner_id": miner_id,
                "workload_type": workload_type,
                "accepted": 0,
                "reward_units": 0,
                "reward_amount_microcredits": 0,
                "unit_price_microcredits": int(row.get("unit_price_microcredits") or 0),
                "currency": "operator_microcredit_v1",
                "reward_account_present": False,
                "settlement_status": "policy_not_joined",
                "payment_executed": False,
            })
            item["accepted"] += 1
            item["reward_units"] += int(row.get("reward_units") or 0)
            item["reward_amount_microcredits"] += int(row.get("reward_amount_microcredits") or 0)
        return totals

    def _created_by_subject_settlement_totals(self, rows: list[dict]) -> dict:
        totals: dict[str, dict] = {}
        for row in rows:
            subject = str(row.get("created_by_subject") or "").strip()
            if not subject:
                continue
            workload_type = str(row.get("workload_type") or "")
            key = f"{subject}/{workload_type}"
            item = totals.setdefault(key, {
                "created_by_subject": subject,
                "workload_type": workload_type,
                "accepted": 0,
                "reward_units": 0,
                "reward_amount_microcredits": 0,
                "unit_price_microcredits": int(row.get("unit_price_microcredits") or 0),
                "currency": "operator_microcredit_v1",
                "draft_only": True,
                "payment_executed": False,
            })
            item["accepted"] += 1
            item["reward_units"] += int(row.get("reward_units") or 0)
            item["reward_amount_microcredits"] += int(row.get("reward_amount_microcredits") or 0)
        return totals

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
            "artifact_schema",
            "artifact_id",
            "artifact_version",
            "tokenizer_schema",
            "backend",
            "split_index",
            "real_llm_artifact_ready",
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
            "session_id",
            "stage_id",
            "stage_count",
            "max_new_tokens",
            "generation_step",
            "activation_count",
            "activation_bytes",
            "activation_hashes",
            "activation_transport_ready",
            "baseline_match",
            "decoded_tokens_match",
            "decode_steps",
            "generated_token_count",
            "generated_token_ids",
            "generated_text",
            "generated_text_hash",
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
            "backend",
            "split_index",
            "real_llm_artifact_ready",
            "stage_id",
            "stage_count",
            "activation_count",
            "activation_bytes",
            "baseline_match",
            "decoded_tokens_match",
            "decode_steps",
            "generated_token_count",
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
                            "sharded_inference_result",
                            "activation_result",
                            "activation_results",
                            "hidden_state",
                            "input_ids",
                            "logits",
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
                if isinstance(claim_spec, dict) and claim_spec.get("type") == WORKLOAD_REAL_LLM_SHARDED_INFER:
                    redacted["claim_workload_spec"] = self._public_workload_spec(
                        claim_spec,
                        workload_type=WORKLOAD_REAL_LLM_SHARDED_INFER,
                    )
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
        if self._workload_type(task) in {
            WORKLOAD_MICRO_TRANSFORMER_LM,
            WORKLOAD_MICRO_LLM_SHARDED_INFER,
        }:
            return micro_transformer_version(self._model)
        if self._workload_type(task) == WORKLOAD_REAL_LLM_SHARDED_INFER:
            return 0
        if self._workload_type(task) in {
            WORKLOAD_MODEL_BUNDLE_LM,
            WORKLOAD_MODEL_BUNDLE_INFER,
            WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
        }:
            return model_bundle_version(self._model)
        return int(self._model["version"])

    def _claim_weights_for_task(self, task: dict, workload_spec: dict) -> list[float]:
        if self._workload_type(task) in {
            WORKLOAD_EXTERNAL_LLM_INFER,
            WORKLOAD_REAL_LLM_SHARDED_INFER,
        }:
            return []
        if self._workload_type(task) in {
            WORKLOAD_MICRO_TRANSFORMER_LM,
            WORKLOAD_MICRO_LLM_SHARDED_INFER,
        }:
            return list(workload_spec.get("weights", []))
        if self._workload_type(task) in {
            WORKLOAD_MODEL_BUNDLE_LM,
            WORKLOAD_MODEL_BUNDLE_INFER,
            WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
        }:
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
        if self._workload_type(task) in {
            WORKLOAD_MODEL_BUNDLE_INFER,
            WORKLOAD_EXTERNAL_LLM_INFER,
            WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
            WORKLOAD_MICRO_LLM_SHARDED_INFER,
            WORKLOAD_REAL_LLM_SHARDED_INFER,
        }:
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
        requirements = {
            "runtime": task.get("required_runtime", REQUIREMENT_ANY) or REQUIREMENT_ANY,
            "backend": task.get("required_backend", REQUIREMENT_ANY) or REQUIREMENT_ANY,
            "protocol_version": (
                task.get("required_protocol_version", DEFAULT_PROTOCOL_VERSION)
                or DEFAULT_PROTOCOL_VERSION
            ),
        }
        stage_capability = self._required_stage_capability(task)
        if stage_capability:
            requirements["stage_capability"] = stage_capability
        return requirements

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
        if workload_type == WORKLOAD_SHARDED_MODEL_BUNDLE_INFER:
            metadata = task.get("workload_metadata") or {}
            return sharded_model_bundle_inference_spec_for(
                task["task_id"],
                task.get("miner_id") or "anonymous",
                self._model.get("model_bundle", {}),
                request_count=int(task.get("inner_steps", 1)),
                scenario_id=metadata.get("scenario_id"),
                session_id=str(metadata.get("session_id") or ""),
                stage_id=int(metadata.get("stage_id", 0)),
                parent_task_id=str(metadata.get("parent_task_id") or ""),
                activation_results=list(metadata.get("activation_results") or []),
            )
        if workload_type == WORKLOAD_MICRO_LLM_SHARDED_INFER:
            metadata = task.get("workload_metadata") or {}
            return micro_llm_sharded_inference_spec_for(
                task["task_id"],
                task.get("miner_id") or "anonymous",
                self._model.get("micro_transformer", {}),
                request_count=int(task.get("inner_steps", 1)),
                decode_steps=int(metadata.get("decode_steps", 1)),
                session_id=str(metadata.get("session_id") or ""),
                stage_id=int(metadata.get("stage_id", 0)),
                parent_task_id=str(metadata.get("parent_task_id") or ""),
                requests=list(metadata.get("requests") or []),
                activation_results=list(metadata.get("activation_results") or []),
            )
        if workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
            metadata = task.get("workload_metadata") or {}
            artifact = self._real_llm_artifact_from_metadata(metadata) or self._real_llm_artifact_summary()
            if isinstance(metadata, dict) and metadata.get("partition_mode"):
                artifact["partition_mode"] = normalize_real_llm_partition_mode(metadata.get("partition_mode"))
            return real_llm_sharded_inference_spec_for(
                task["task_id"],
                task.get("miner_id") or "anonymous",
                artifact,
                request_count=int(task.get("inner_steps", 1)),
                prompt_texts=list(metadata.get("prompt_texts") or []),
                session_id=str(metadata.get("session_id") or ""),
                stage_id=int(metadata.get("stage_id", 0)),
                parent_task_id=str(metadata.get("parent_task_id") or ""),
                max_new_tokens=int(metadata.get("max_new_tokens", 1)),
                generation_step=int(metadata.get("generation_step", 0)),
                requests=list(metadata.get("requests") or []),
                activation_results=list(metadata.get("activation_results") or []),
            )
        return {"type": WORKLOAD_DILOCO_TRAIN}

    def _real_llm_artifact_summary(self, backend: str | None = None, *, model_id: str | None = None) -> dict:
        resolved_backend = normalize_real_llm_backend(backend or self.real_llm_backend)
        resolved_model_id = str(model_id or self.real_llm_model_id or DEFAULT_REAL_LLM_MODEL_ID).strip() or DEFAULT_REAL_LLM_MODEL_ID
        use_default_cache = resolved_model_id == self.real_llm_model_id
        if use_default_cache:
            cached = self._real_llm_artifact_cache.get(resolved_backend)
            if cached is not None:
                return dict(cached)
        artifact = inspect_real_llm_artifact(
            model_id=resolved_model_id,
            cache_dir=self.hf_cache_dir,
            backend=resolved_backend,
            require_runtime=resolved_backend != REAL_LLM_BACKEND_CUDA,
        )
        if use_default_cache:
            self._real_llm_artifact_cache[resolved_backend] = dict(artifact)
        return dict(artifact)

    def _real_llm_artifact_from_metadata(self, metadata: dict | None) -> dict:
        if not isinstance(metadata, dict):
            return {}
        artifact_hash = str(metadata.get("artifact_hash") or "").strip()
        model_id = str(metadata.get("model_id") or self.real_llm_model_id or DEFAULT_REAL_LLM_MODEL_ID).strip()
        if not artifact_hash:
            return {}
        return {
            "schema": str(metadata.get("artifact_schema") or "real_llm_artifact_v1"),
            "artifact_hash": artifact_hash,
            "model_id": model_id,
            "backend": normalize_real_llm_backend(str(metadata.get("backend") or REAL_LLM_BACKEND_CPU)),
            "partition_mode": normalize_real_llm_partition_mode(metadata.get("partition_mode") or REAL_LLM_PARTITION_MODE_FULL),
            "model_type": str(metadata.get("model_type") or ""),
            "split_index": int(metadata.get("split_index") or 1),
            "num_hidden_layers": int(metadata.get("num_hidden_layers") or 2),
            "hidden_size": int(metadata.get("hidden_size") or 1),
            "vocab_size": int(metadata.get("vocab_size") or 0),
            "read_only": True,
        }

    def _requirement_key(self, task: dict) -> str:
        requirements = self._task_requirements(task)
        key = (
            f"{requirements['runtime']}/"
            f"{requirements['backend']}/"
            f"{requirements['protocol_version']}"
        )
        if requirements.get("stage_capability"):
            key = f"{key}/{requirements['stage_capability']}"
        return key

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

        required_stage_capability = self._required_stage_capability(task)
        if required_stage_capability and not self._miner_supports_stage_capability(capabilities, required_stage_capability):
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

    def _required_stage_capability(self, task: dict) -> str:
        workload_type = self._workload_type(task)
        if workload_type not in {WORKLOAD_MICRO_LLM_SHARDED_INFER, WORKLOAD_REAL_LLM_SHARDED_INFER}:
            return ""
        metadata = task.get("workload_metadata") or {}
        try:
            stage_id = int(metadata.get("stage_id", 0))
        except (TypeError, ValueError):
            stage_id = 0
        if workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
            real_backend = str(metadata.get("backend") or REAL_LLM_BACKEND_CPU)
            if stage_id == 0:
                return (
                    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY
                    if real_backend == REAL_LLM_BACKEND_CUDA
                    else REAL_LLM_SHARDED_STAGE0_CAPABILITY
                )
            if stage_id == 1:
                return (
                    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY
                    if real_backend == REAL_LLM_BACKEND_CUDA
                    else REAL_LLM_SHARDED_STAGE1_CAPABILITY
                )
            return ""
        if stage_id == 0:
            return MICRO_LLM_SHARDED_STAGE0_CAPABILITY
        if stage_id == 1:
            return MICRO_LLM_SHARDED_STAGE1_CAPABILITY
        return ""

    def _miner_supports_stage_capability(self, capabilities: dict, required_stage_capability: str) -> bool:
        if required_stage_capability in {
            REAL_LLM_SHARDED_STAGE0_CAPABILITY,
            REAL_LLM_SHARDED_STAGE1_CAPABILITY,
            REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
            REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
        }:
            return self._miner_supports_real_llm_stage(capabilities, required_stage_capability)
        return self._miner_supports_micro_llm_stage(capabilities, required_stage_capability)

    def _miner_supports_micro_llm_stage(self, capabilities: dict, required_stage_capability: str) -> bool:
        advertised = capabilities.get("micro_llm_sharded_stage_capabilities")
        if advertised is None:
            role = str(capabilities.get("micro_llm_sharded_stage_role") or "both").strip().lower()
            advertised_set = {
                "stage0": {MICRO_LLM_SHARDED_STAGE0_CAPABILITY},
                "stage1": {MICRO_LLM_SHARDED_STAGE1_CAPABILITY},
                "both": {
                    MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
                    MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
                    MICRO_LLM_SHARDED_BOTH_CAPABILITY,
                },
            }.get(role)
            if advertised_set is None:
                advertised_set = {
                    MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
                    MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
                    MICRO_LLM_SHARDED_BOTH_CAPABILITY,
                }
        elif isinstance(advertised, str):
            advertised_set = {advertised}
        else:
            advertised_set = {str(value) for value in advertised}
        return (
            MICRO_LLM_SHARDED_BOTH_CAPABILITY in advertised_set
            or required_stage_capability in advertised_set
        )

    def _miner_supports_real_llm_stage(self, capabilities: dict, required_stage_capability: str) -> bool:
        advertised = capabilities.get("real_llm_sharded_stage_capabilities")
        if advertised is None:
            role = str(capabilities.get("real_llm_sharded_stage_role") or "both").strip().lower()
            advertised_set = {
                "stage0": {REAL_LLM_SHARDED_STAGE0_CAPABILITY},
                "stage1": {REAL_LLM_SHARDED_STAGE1_CAPABILITY},
                "both": {
                    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
                    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
                    REAL_LLM_SHARDED_BOTH_CAPABILITY,
                },
            }.get(role)
            if advertised_set is None:
                advertised_set = {
                    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
                    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
                    REAL_LLM_SHARDED_BOTH_CAPABILITY,
                }
        elif isinstance(advertised, str):
            advertised_set = {advertised}
        else:
            advertised_set = {str(value) for value in advertised}
        if required_stage_capability in {
            REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
            REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
        }:
            runtime = capabilities.get("real_llm_runtime") if isinstance(capabilities.get("real_llm_runtime"), dict) else {}
            return (
                str(runtime.get("adapter_kind") or "") == REAL_LLM_BACKEND_CUDA
                and (
                    REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY in advertised_set
                    or required_stage_capability in advertised_set
                )
            )
        return (
            REAL_LLM_SHARDED_BOTH_CAPABILITY in advertised_set
            or required_stage_capability in advertised_set
        )

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
