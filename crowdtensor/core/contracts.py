"""Versioned, framework-neutral contracts for collaborative training.

These objects deliberately contain no PyTorch, JAX, provider, or transport
code. They are the stable boundary between CrowdTensor orchestration and the
runtime adapters that perform model work.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


TRAINING_PROJECT_SCHEMA = "crowdtensor_training_project_v2"
WORK_UNIT_SCHEMA = "crowdtensor_work_unit_v2"
CHECKPOINT_REF_SCHEMA = "crowdtensor_checkpoint_ref_v2"
CHECKPOINT_LINEAGE_SCHEMA = "crowdtensor_checkpoint_lineage_v2"
CONTRIBUTION_RECEIPT_SCHEMA = "crowdtensor_contribution_receipt_v2"

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
_METRIC_NAME = re.compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")


class ContractError(ValueError):
    """Raised when a versioned core contract fails closed validation."""


class TrainingMode(str, Enum):
    """The two supported distributed-training coordination semantics."""

    ELASTIC_DELTA = "elastic_delta"
    STABLE_SHARDED = "stable_sharded"


class ReceiptOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


def canonical_json(value: Any) -> bytes:
    """Encode a public contract deterministically."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("contract_value_not_canonical_json") from exc


def stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field}_must_be_string")
    result = value.strip()
    if not result:
        raise ContractError(f"{field}_required")
    return result


def _require_identifier(value: Any, field: str) -> str:
    result = _require_string(value, field)
    if not _IDENTIFIER.fullmatch(result):
        raise ContractError(f"{field}_invalid")
    return result


def _require_hash(value: Any, field: str) -> str:
    result = _require_string(value, field)
    if not _HASH.fullmatch(result):
        raise ContractError(f"{field}_invalid")
    return result


def _require_timestamp(value: Any, field: str) -> str:
    result = _require_string(value, field)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{field}_invalid") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"{field}_timezone_required")
    return result


def _hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_hash"] = stable_hash(result)
    return result


def _verify_payload(
    value: Mapping[str, Any],
    *,
    schema: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("contract_object_required")
    payload = dict(value)
    if payload.get("schema") != schema:
        raise ContractError("contract_schema_mismatch")
    allowed = required | set(optional or ()) | {"schema", "content_hash"}
    unknown = set(payload) - allowed
    missing = required - set(payload)
    if unknown:
        raise ContractError("contract_unknown_fields:" + ",".join(sorted(unknown)))
    if missing:
        raise ContractError("contract_missing_fields:" + ",".join(sorted(missing)))
    claimed_hash = _require_hash(payload.pop("content_hash", ""), "content_hash")
    if stable_hash(payload) != claimed_hash:
        raise ContractError("contract_content_hash_mismatch")
    return payload


@dataclass(frozen=True)
class ArtifactRef:
    """A revision-pinned model, dataset, checkpoint, or delta artifact."""

    uri: str
    revision: str
    digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _require_string(self.uri, "artifact_uri"))
        object.__setattr__(
            self, "revision", _require_string(self.revision, "artifact_revision")
        )
        if self.digest is not None:
            object.__setattr__(
                self, "digest", _require_hash(self.digest, "artifact_digest")
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"uri": self.uri, "revision": self.revision}
        if self.digest is not None:
            result["digest"] = self.digest
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        if not isinstance(value, Mapping):
            raise ContractError("artifact_ref_object_required")
        unknown = set(value) - {"uri", "revision", "digest"}
        if unknown:
            raise ContractError("artifact_ref_unknown_fields:" + ",".join(sorted(unknown)))
        return cls(
            uri=value.get("uri", ""),
            revision=value.get("revision", ""),
            digest=value.get("digest"),
        )


@dataclass(frozen=True)
class TrainingProject:
    """Immutable training intent; runtime state is stored separately."""

    project_id: str
    mode: TrainingMode
    model: ArtifactRef
    dataset: ArtifactRef
    model_adapter: str
    training_backend: str
    target_steps: int
    optimization_plugins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id", _require_identifier(self.project_id, "project_id")
        )
        try:
            object.__setattr__(self, "mode", TrainingMode(self.mode))
        except ValueError as exc:
            raise ContractError("training_mode_invalid") from exc
        if not isinstance(self.model, ArtifactRef) or not isinstance(
            self.dataset, ArtifactRef
        ):
            raise ContractError("project_artifact_ref_invalid")
        object.__setattr__(
            self,
            "model_adapter",
            _require_identifier(self.model_adapter, "model_adapter"),
        )
        object.__setattr__(
            self,
            "training_backend",
            _require_identifier(self.training_backend, "training_backend"),
        )
        if isinstance(self.target_steps, bool) or int(self.target_steps) < 1:
            raise ContractError("target_steps_must_be_positive")
        object.__setattr__(self, "target_steps", int(self.target_steps))
        plugins = tuple(
            _require_identifier(item, "optimization_plugin")
            for item in self.optimization_plugins
        )
        if len(set(plugins)) != len(plugins):
            raise ContractError("optimization_plugins_duplicate")
        object.__setattr__(self, "optimization_plugins", plugins)

    def to_dict(self) -> dict[str, Any]:
        return _hashed(
            {
                "schema": TRAINING_PROJECT_SCHEMA,
                "project_id": self.project_id,
                "mode": self.mode.value,
                "model": self.model.to_dict(),
                "dataset": self.dataset.to_dict(),
                "model_adapter": self.model_adapter,
                "training_backend": self.training_backend,
                "target_steps": self.target_steps,
                "optimization_plugins": list(self.optimization_plugins),
                "public_artifact_safe": True,
            }
        )

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingProject:
        payload = _verify_payload(
            value,
            schema=TRAINING_PROJECT_SCHEMA,
            required={
                "project_id",
                "mode",
                "model",
                "dataset",
                "model_adapter",
                "training_backend",
                "target_steps",
                "optimization_plugins",
                "public_artifact_safe",
            },
        )
        if payload.get("public_artifact_safe") is not True:
            raise ContractError("project_public_safety_required")
        plugins = payload["optimization_plugins"]
        if not isinstance(plugins, list):
            raise ContractError("optimization_plugins_list_required")
        return cls(
            project_id=payload["project_id"],
            mode=payload["mode"],
            model=ArtifactRef.from_dict(payload["model"]),
            dataset=ArtifactRef.from_dict(payload["dataset"]),
            model_adapter=payload["model_adapter"],
            training_backend=payload["training_backend"],
            target_steps=payload["target_steps"],
            optimization_plugins=tuple(plugins),
        )


@dataclass(frozen=True)
class WorkUnit:
    """A bounded, generation-fenced unit issued to one runtime worker."""

    work_id: str
    project_hash: str
    mode: TrainingMode
    generation: int
    backend: str
    base_checkpoint_hash: str
    data_shard_hash: str
    step_start: int
    step_count: int
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_id", _require_identifier(self.work_id, "work_id"))
        object.__setattr__(
            self, "project_hash", _require_hash(self.project_hash, "project_hash")
        )
        try:
            object.__setattr__(self, "mode", TrainingMode(self.mode))
        except ValueError as exc:
            raise ContractError("training_mode_invalid") from exc
        if isinstance(self.generation, bool) or int(self.generation) < 1:
            raise ContractError("work_generation_must_be_positive")
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "backend", _require_identifier(self.backend, "backend"))
        object.__setattr__(
            self,
            "base_checkpoint_hash",
            _require_hash(self.base_checkpoint_hash, "base_checkpoint_hash"),
        )
        object.__setattr__(
            self,
            "data_shard_hash",
            _require_hash(self.data_shard_hash, "data_shard_hash"),
        )
        if isinstance(self.step_start, bool) or int(self.step_start) < 0:
            raise ContractError("step_start_must_be_non_negative")
        if isinstance(self.step_count, bool) or int(self.step_count) < 1:
            raise ContractError("step_count_must_be_positive")
        object.__setattr__(self, "step_start", int(self.step_start))
        object.__setattr__(self, "step_count", int(self.step_count))
        capabilities = tuple(
            _require_identifier(item, "required_capability")
            for item in self.required_capabilities
        )
        if len(set(capabilities)) != len(capabilities):
            raise ContractError("required_capabilities_duplicate")
        object.__setattr__(self, "required_capabilities", capabilities)

    def to_dict(self) -> dict[str, Any]:
        return _hashed(
            {
                "schema": WORK_UNIT_SCHEMA,
                "work_id": self.work_id,
                "project_hash": self.project_hash,
                "mode": self.mode.value,
                "generation": self.generation,
                "backend": self.backend,
                "base_checkpoint_hash": self.base_checkpoint_hash,
                "data_shard_hash": self.data_shard_hash,
                "step_start": self.step_start,
                "step_count": self.step_count,
                "required_capabilities": list(self.required_capabilities),
                "public_artifact_safe": True,
            }
        )

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkUnit:
        payload = _verify_payload(
            value,
            schema=WORK_UNIT_SCHEMA,
            required={
                "work_id",
                "project_hash",
                "mode",
                "generation",
                "backend",
                "base_checkpoint_hash",
                "data_shard_hash",
                "step_start",
                "step_count",
                "required_capabilities",
                "public_artifact_safe",
            },
        )
        if payload.get("public_artifact_safe") is not True:
            raise ContractError("work_unit_public_safety_required")
        capabilities = payload["required_capabilities"]
        if not isinstance(capabilities, list):
            raise ContractError("required_capabilities_list_required")
        return cls(
            work_id=payload["work_id"],
            project_hash=payload["project_hash"],
            mode=payload["mode"],
            generation=payload["generation"],
            backend=payload["backend"],
            base_checkpoint_hash=payload["base_checkpoint_hash"],
            data_shard_hash=payload["data_shard_hash"],
            step_start=payload["step_start"],
            step_count=payload["step_count"],
            required_capabilities=tuple(capabilities),
        )


@dataclass(frozen=True)
class CheckpointRef:
    """A content-addressed checkpoint reference, not the tensor payload itself."""

    checkpoint_id: str
    project_hash: str
    step: int
    generation: int
    artifact: ArtifactRef
    parent_hash: str | None = None
    created_by_work_id: str | None = None
    adapter_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_id",
            _require_identifier(self.checkpoint_id, "checkpoint_id"),
        )
        object.__setattr__(
            self, "project_hash", _require_hash(self.project_hash, "project_hash")
        )
        if isinstance(self.step, bool) or int(self.step) < 0:
            raise ContractError("checkpoint_step_must_be_non_negative")
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ContractError("checkpoint_generation_must_be_non_negative")
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "generation", int(self.generation))
        if not isinstance(self.artifact, ArtifactRef) or self.artifact.digest is None:
            raise ContractError("checkpoint_artifact_digest_required")
        if self.parent_hash is not None:
            object.__setattr__(
                self, "parent_hash", _require_hash(self.parent_hash, "parent_hash")
            )
        if self.created_by_work_id is not None:
            object.__setattr__(
                self,
                "created_by_work_id",
                _require_identifier(self.created_by_work_id, "created_by_work_id"),
            )
        if not isinstance(self.adapter_only, bool):
            raise ContractError("checkpoint_adapter_only_boolean_required")

    def to_dict(self) -> dict[str, Any]:
        return _hashed(
            {
                "schema": CHECKPOINT_REF_SCHEMA,
                "checkpoint_id": self.checkpoint_id,
                "project_hash": self.project_hash,
                "step": self.step,
                "generation": self.generation,
                "artifact": self.artifact.to_dict(),
                "parent_hash": self.parent_hash,
                "created_by_work_id": self.created_by_work_id,
                "adapter_only": self.adapter_only,
                "public_artifact_safe": True,
            }
        )

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointRef:
        payload = _verify_payload(
            value,
            schema=CHECKPOINT_REF_SCHEMA,
            required={
                "checkpoint_id",
                "project_hash",
                "step",
                "generation",
                "artifact",
                "parent_hash",
                "created_by_work_id",
                "adapter_only",
                "public_artifact_safe",
            },
        )
        if payload.get("public_artifact_safe") is not True:
            raise ContractError("checkpoint_public_safety_required")
        return cls(
            checkpoint_id=payload["checkpoint_id"],
            project_hash=payload["project_hash"],
            step=payload["step"],
            generation=payload["generation"],
            artifact=ArtifactRef.from_dict(payload["artifact"]),
            parent_hash=payload["parent_hash"],
            created_by_work_id=payload["created_by_work_id"],
            adapter_only=payload["adapter_only"],
        )


@dataclass(frozen=True)
class CheckpointLineage:
    """An append-only checkpoint chain with explicit parent hashes."""

    project_hash: str
    checkpoints: tuple[CheckpointRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_hash", _require_hash(self.project_hash, "project_hash")
        )
        checkpoints = tuple(self.checkpoints)
        if not checkpoints:
            raise ContractError("checkpoint_lineage_empty")
        ids: set[str] = set()
        previous: CheckpointRef | None = None
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, CheckpointRef):
                raise ContractError("checkpoint_lineage_item_invalid")
            if checkpoint.project_hash != self.project_hash:
                raise ContractError("checkpoint_lineage_project_mismatch")
            if checkpoint.checkpoint_id in ids:
                raise ContractError("checkpoint_lineage_duplicate_id")
            ids.add(checkpoint.checkpoint_id)
            if previous is None:
                if checkpoint.parent_hash is not None:
                    raise ContractError("checkpoint_lineage_genesis_parent_forbidden")
            else:
                if checkpoint.parent_hash != previous.content_hash:
                    raise ContractError("checkpoint_lineage_parent_mismatch")
                if checkpoint.step <= previous.step:
                    raise ContractError("checkpoint_lineage_step_not_increasing")
                if checkpoint.generation < previous.generation:
                    raise ContractError("checkpoint_lineage_generation_regressed")
            previous = checkpoint
        object.__setattr__(self, "checkpoints", checkpoints)

    def append(self, checkpoint: CheckpointRef) -> CheckpointLineage:
        return CheckpointLineage(self.project_hash, (*self.checkpoints, checkpoint))

    def to_dict(self) -> dict[str, Any]:
        return _hashed(
            {
                "schema": CHECKPOINT_LINEAGE_SCHEMA,
                "project_hash": self.project_hash,
                "checkpoints": [item.to_dict() for item in self.checkpoints],
                "public_artifact_safe": True,
            }
        )

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointLineage:
        payload = _verify_payload(
            value,
            schema=CHECKPOINT_LINEAGE_SCHEMA,
            required={"project_hash", "checkpoints", "public_artifact_safe"},
        )
        if payload.get("public_artifact_safe") is not True:
            raise ContractError("checkpoint_lineage_public_safety_required")
        checkpoints = payload["checkpoints"]
        if not isinstance(checkpoints, list):
            raise ContractError("checkpoint_lineage_list_required")
        return cls(
            project_hash=payload["project_hash"],
            checkpoints=tuple(CheckpointRef.from_dict(item) for item in checkpoints),
        )


@dataclass(frozen=True)
class ContributionReceipt:
    """Public-safe accounting record for one terminal work generation.

    An accepted elastic contribution does not necessarily advance the canonical
    checkpoint immediately. Quorum-based backends may accept several deltas
    before committing one aggregated checkpoint, so acceptance and commit are
    represented independently.
    """

    receipt_id: str
    project_hash: str
    work_id: str
    work_generation: int
    contributor_id_hash: str
    base_checkpoint_hash: str
    submitted_artifact_hash: str
    outcome: ReceiptOutcome
    completed_at: str
    steps: int
    samples: int
    tokens: int
    checkpoint_committed: bool = False
    output_checkpoint_hash: str | None = None
    rejection_code: str | None = None
    metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _require_identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self, "project_hash", _require_hash(self.project_hash, "project_hash")
        )
        object.__setattr__(self, "work_id", _require_identifier(self.work_id, "work_id"))
        if isinstance(self.work_generation, bool) or int(self.work_generation) < 1:
            raise ContractError("receipt_work_generation_must_be_positive")
        object.__setattr__(self, "work_generation", int(self.work_generation))
        object.__setattr__(
            self,
            "contributor_id_hash",
            _require_hash(self.contributor_id_hash, "contributor_id_hash"),
        )
        object.__setattr__(
            self,
            "base_checkpoint_hash",
            _require_hash(self.base_checkpoint_hash, "base_checkpoint_hash"),
        )
        object.__setattr__(
            self,
            "submitted_artifact_hash",
            _require_hash(self.submitted_artifact_hash, "submitted_artifact_hash"),
        )
        try:
            object.__setattr__(self, "outcome", ReceiptOutcome(self.outcome))
        except ValueError as exc:
            raise ContractError("receipt_outcome_invalid") from exc
        object.__setattr__(
            self, "completed_at", _require_timestamp(self.completed_at, "completed_at")
        )
        for field in ("steps", "samples", "tokens"):
            value = getattr(self, field)
            if isinstance(value, bool) or int(value) < 0:
                raise ContractError(f"receipt_{field}_must_be_non_negative")
            object.__setattr__(self, field, int(value))
        if not isinstance(self.checkpoint_committed, bool):
            raise ContractError("receipt_checkpoint_committed_boolean_required")
        if self.outcome is ReceiptOutcome.ACCEPTED:
            if self.rejection_code is not None:
                raise ContractError("accepted_receipt_rejection_code_forbidden")
            if self.steps < 1:
                raise ContractError("accepted_receipt_steps_must_be_positive")
            if self.checkpoint_committed and self.output_checkpoint_hash is None:
                raise ContractError("committed_receipt_output_checkpoint_required")
            if not self.checkpoint_committed and self.output_checkpoint_hash is not None:
                raise ContractError("uncommitted_receipt_output_checkpoint_forbidden")
        else:
            if self.checkpoint_committed:
                raise ContractError("rejected_receipt_checkpoint_commit_forbidden")
            if self.output_checkpoint_hash is not None:
                raise ContractError("rejected_receipt_output_checkpoint_forbidden")
            if self.rejection_code is None:
                raise ContractError("rejected_receipt_code_required")
        if self.output_checkpoint_hash is not None:
            object.__setattr__(
                self,
                "output_checkpoint_hash",
                _require_hash(self.output_checkpoint_hash, "output_checkpoint_hash"),
            )
        if self.rejection_code is not None:
            object.__setattr__(
                self,
                "rejection_code",
                _require_identifier(self.rejection_code, "rejection_code"),
            )
        metric_items: list[tuple[str, float]] = []
        seen: set[str] = set()
        for name, raw_value in self.metrics:
            metric_name = str(name)
            if not _METRIC_NAME.fullmatch(metric_name) or metric_name in seen:
                raise ContractError("receipt_metric_name_invalid")
            metric_value = float(raw_value)
            if not math.isfinite(metric_value):
                raise ContractError("receipt_metric_non_finite")
            seen.add(metric_name)
            metric_items.append((metric_name, metric_value))
        object.__setattr__(self, "metrics", tuple(sorted(metric_items)))

    def to_dict(self) -> dict[str, Any]:
        return _hashed(
            {
                "schema": CONTRIBUTION_RECEIPT_SCHEMA,
                "receipt_id": self.receipt_id,
                "project_hash": self.project_hash,
                "work_id": self.work_id,
                "work_generation": self.work_generation,
                "contributor_id_hash": self.contributor_id_hash,
                "base_checkpoint_hash": self.base_checkpoint_hash,
                "submitted_artifact_hash": self.submitted_artifact_hash,
                "outcome": self.outcome.value,
                "completed_at": self.completed_at,
                "steps": self.steps,
                "samples": self.samples,
                "tokens": self.tokens,
                "checkpoint_committed": self.checkpoint_committed,
                "output_checkpoint_hash": self.output_checkpoint_hash,
                "rejection_code": self.rejection_code,
                "metrics": dict(self.metrics),
                "contributor_identity_public": False,
                "public_artifact_safe": True,
            }
        )

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContributionReceipt:
        payload = _verify_payload(
            value,
            schema=CONTRIBUTION_RECEIPT_SCHEMA,
            required={
                "receipt_id",
                "project_hash",
                "work_id",
                "work_generation",
                "contributor_id_hash",
                "base_checkpoint_hash",
                "submitted_artifact_hash",
                "outcome",
                "completed_at",
                "steps",
                "samples",
                "tokens",
                "checkpoint_committed",
                "output_checkpoint_hash",
                "rejection_code",
                "metrics",
                "contributor_identity_public",
                "public_artifact_safe",
            },
        )
        if payload.get("public_artifact_safe") is not True:
            raise ContractError("receipt_public_safety_required")
        if payload.get("contributor_identity_public") is not False:
            raise ContractError("receipt_contributor_privacy_required")
        metrics = payload["metrics"]
        if not isinstance(metrics, Mapping):
            raise ContractError("receipt_metrics_object_required")
        return cls(
            receipt_id=payload["receipt_id"],
            project_hash=payload["project_hash"],
            work_id=payload["work_id"],
            work_generation=payload["work_generation"],
            contributor_id_hash=payload["contributor_id_hash"],
            base_checkpoint_hash=payload["base_checkpoint_hash"],
            submitted_artifact_hash=payload["submitted_artifact_hash"],
            outcome=payload["outcome"],
            completed_at=payload["completed_at"],
            steps=payload["steps"],
            samples=payload["samples"],
            tokens=payload["tokens"],
            checkpoint_committed=payload["checkpoint_committed"],
            output_checkpoint_hash=payload["output_checkpoint_hash"],
            rejection_code=payload["rejection_code"],
            metrics=tuple((str(key), float(item)) for key, item in metrics.items()),
        )


def validate_receipt_binding(
    receipt: ContributionReceipt,
    *,
    work: WorkUnit,
    base_checkpoint: CheckpointRef,
    output_checkpoint: CheckpointRef | None = None,
) -> None:
    """Verify that a receipt cannot be replayed across work generations."""

    if not all(
        item.project_hash == work.project_hash
        for item in (receipt, base_checkpoint)
    ):
        raise ContractError("receipt_binding_project_mismatch")
    if receipt.work_id != work.work_id:
        raise ContractError("receipt_binding_work_mismatch")
    if receipt.work_generation != work.generation:
        raise ContractError("receipt_binding_generation_mismatch")
    if receipt.base_checkpoint_hash != base_checkpoint.content_hash:
        raise ContractError("receipt_binding_base_checkpoint_mismatch")
    if work.base_checkpoint_hash != base_checkpoint.content_hash:
        raise ContractError("work_binding_base_checkpoint_mismatch")
    if receipt.outcome is ReceiptOutcome.ACCEPTED and receipt.checkpoint_committed:
        if output_checkpoint is None:
            raise ContractError("receipt_binding_output_checkpoint_required")
        if output_checkpoint.project_hash != work.project_hash:
            raise ContractError("receipt_binding_output_project_mismatch")
        if output_checkpoint.parent_hash != base_checkpoint.content_hash:
            raise ContractError("receipt_binding_output_parent_mismatch")
        # A quorum aggregation has no singular creating Work Unit. Its backend
        # validates membership before issuing receipts, while a single-work
        # checkpoint remains directly bound here.
        if output_checkpoint.created_by_work_id not in {None, work.work_id}:
            raise ContractError("receipt_binding_output_work_mismatch")
        if receipt.output_checkpoint_hash != output_checkpoint.content_hash:
            raise ContractError("receipt_binding_output_checkpoint_mismatch")
    elif output_checkpoint is not None:
        raise ContractError("receipt_binding_uncommitted_output_forbidden")
