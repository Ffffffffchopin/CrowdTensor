"""Protocol constants and small exceptions for CrowdTensorD Phase 1."""

from __future__ import annotations

import time
import uuid


STATUS_QUEUED = "queued"
STATUS_LEASED = "leased"
STATUS_COMPLETED = "completed"
STATUS_REJECTED = "rejected"

EVENT_TASK_CREATED = "task_created"
EVENT_TASK_CLAIMED = "task_claimed"
EVENT_TASK_HEARTBEAT = "task_heartbeat"
EVENT_TASK_COMPLETED = "task_completed"
EVENT_TASK_REQUEUED = "task_requeued"
EVENT_TASK_REJECTED = "task_rejected"
EVENT_INCOMPATIBLE_CLAIM = "incompatible_claim"
EVENT_CLAIM_BLOCKED = "claim_blocked"
EVENT_TRUST_OVERRIDE_SET = "trust_override_set"
EVENT_CONTROL_PLANE_BLOCKED = "control_plane_blocked"

REQUIREMENT_ANY = "any"
DEFAULT_PROTOCOL_VERSION = "runtime_contract_v1"
WORKLOAD_DILOCO_TRAIN = "diloco_train"
WORKLOAD_BROWSER_PROBE = "browser_probe"
WORKLOAD_CPU_LORA_MOCK = "cpu_lora_mock"
WORKLOAD_MICRO_TRANSFORMER_LM = "micro_transformer_lm"
WORKLOAD_MICRO_LLM_SHARDED_INFER = "micro_llm_sharded_infer"
MICRO_LLM_SHARDED_STAGE0_CAPABILITY = "micro_llm_sharded_stage0"
MICRO_LLM_SHARDED_STAGE1_CAPABILITY = "micro_llm_sharded_stage1"
MICRO_LLM_SHARDED_BOTH_CAPABILITY = "micro_llm_sharded_both"
MICRO_LLM_SHARDED_STAGE_CAPABILITIES = {
    MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
    MICRO_LLM_SHARDED_BOTH_CAPABILITY,
}
WORKLOAD_REAL_LLM_SHARDED_INFER = "real_llm_sharded_infer"
REAL_LLM_SHARDED_STAGE0_CAPABILITY = "real_llm_sharded_stage0"
REAL_LLM_SHARDED_STAGE1_CAPABILITY = "real_llm_sharded_stage1"
REAL_LLM_SHARDED_BOTH_CAPABILITY = "real_llm_sharded_both"
REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY = "real_llm_sharded_cuda_stage0"
REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY = "real_llm_sharded_cuda_stage1"
REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY = "real_llm_sharded_cuda_both"
REAL_LLM_SHARDED_STAGE_CAPABILITIES = {
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
}
WORKLOAD_MODEL_BUNDLE_LM = "model_bundle_lm"
WORKLOAD_MODEL_BUNDLE_INFER = "model_bundle_infer"
WORKLOAD_SHARDED_MODEL_BUNDLE_INFER = "sharded_model_bundle_infer"
WORKLOAD_EXTERNAL_LLM_INFER = "external_llm_infer"
DEFAULT_WORKLOAD_TYPE = WORKLOAD_DILOCO_TRAIN


class CrowdTensorError(RuntimeError):
    """Base exception for expected protocol failures."""


class LeaseConflict(CrowdTensorError):
    """Raised when a miner uses a stale, expired, or invalid lease."""


class NoTaskAvailable(CrowdTensorError):
    """Raised when no queued work exists."""


class ResultRejected(CrowdTensorError):
    """Raised when a valid lease submits a result that fails quality validation."""

    def __init__(self, validation: dict) -> None:
        super().__init__(validation.get("reason", "result rejected"))
        self.validation = validation


def now_epoch() -> float:
    return time.time()


def new_task_id() -> str:
    return f"task-{uuid.uuid4().hex}"


def new_lease_token() -> str:
    return uuid.uuid4().hex
