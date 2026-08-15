"""Durable Work-Unit controller for Training Architecture v2.

This module owns orchestration state only. Backends remain responsible for
model execution, delta validation, aggregation, and checkpoint payloads.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .contracts import (
    CheckpointLineage,
    CheckpointRef,
    ContributionReceipt,
    ReceiptOutcome,
    TrainingMode,
    WorkUnit,
    stable_hash,
    validate_receipt_binding,
)
from .workspace import CONTROL_DIR, inspect_workspace, load_project


SESSION_CONTROLLER_SCHEMA = "crowdtensor_training_session_controller_v2"
SESSION_CONTROLLER_REPORT_SCHEMA = "crowdtensor_training_session_action_v2"
SESSION_CONTROLLER_FILE = "state/session-controller.json"
SESSION_CONTROLLER_LOCK = "state/session-controller.lock"
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_ACTIVE_WORK_UNITS = 1024


class SessionControllerError(ValueError):
    """Raised when a v2 session transition is stale, unsafe, or conflicting."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_hash", None)
    result["content_hash"] = stable_hash(result)
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_seal(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SessionControllerError("session_controller_state_invalid")
    supplied = str(payload.get("content_hash") or "")
    unsigned = dict(payload)
    unsigned.pop("content_hash", None)
    if supplied != stable_hash(unsigned):
        raise SessionControllerError("session_controller_state_invalid")
    return payload


def _work_key(work: WorkUnit) -> str:
    return f"{work.work_id}:{work.generation}"


def _same_work_attempt(left: WorkUnit, right: WorkUnit) -> bool:
    left_value = left.to_dict()
    right_value = right.to_dict()
    for value in (left_value, right_value):
        value.pop("content_hash", None)
        value.pop("generation", None)
    return left_value == right_value


def _lineage_is_prefix(prefix: CheckpointLineage, lineage: CheckpointLineage) -> bool:
    if prefix.project_hash != lineage.project_hash:
        return False
    if len(prefix.checkpoints) > len(lineage.checkpoints):
        return False
    return all(
        left.content_hash == right.content_hash
        for left, right in zip(prefix.checkpoints, lineage.checkpoints, strict=False)
    )


class SessionController:
    """Persist Work Units, terminal receipts, and canonical checkpoint lineage.

    Elastic Work Units may share the same committed base while contributors run
    in parallel. Stable-sharded execution admits exactly one rank-group Work
    Unit and delegates all collective and numerical behavior to its backend.
    Only a backend-verified result may append a checkpoint.
    """

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.project = load_project(self.workspace)
        self.state_path = self.workspace / CONTROL_DIR / SESSION_CONTROLLER_FILE
        self.lock_path = self.workspace / CONTROL_DIR / SESSION_CONTROLLER_LOCK

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any] | None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield self._load_state()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SessionControllerError("session_controller_state_invalid") from exc
        return self._validate_state(_verify_seal(payload))

    @staticmethod
    def _active_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the concurrent representation, upgrading the legacy shape."""

        raw = payload.get("active_work_units")
        if raw is None:
            work = payload.get("active_work")
            if work is None:
                return []
            return [
                {
                    "work": work,
                    "contributor_id_hash": payload.get(
                        "active_contributor_id_hash"
                    ),
                    "lease_expires_at": None,
                }
            ]
        if not isinstance(raw, list):
            raise SessionControllerError("session_controller_active_work_invalid")
        return [dict(item) if isinstance(item, dict) else item for item in raw]

    @classmethod
    def _sync_active_state(
        cls, payload: dict[str, Any], records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        result = dict(payload)
        ordered = sorted(
            records,
            key=lambda item: (
                str((item.get("work") or {}).get("work_id") or ""),
                int((item.get("work") or {}).get("generation") or 0),
            ),
        )
        result["controller_revision"] = 2
        result["active_work_units"] = ordered
        if len(ordered) == 1:
            result["active_work"] = ordered[0]["work"]
            result["active_contributor_id_hash"] = ordered[0].get(
                "contributor_id_hash"
            )
        else:
            # These fields are retained as a one-release compatibility view.
            result["active_work"] = None
            result["active_contributor_id_hash"] = None
        return result

    def _validate_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema") != SESSION_CONTROLLER_SCHEMA:
            raise SessionControllerError("session_controller_state_invalid")
        if payload.get("project_hash") != self.project.content_hash:
            raise SessionControllerError("session_controller_project_mismatch")
        if payload.get("mode") != self.project.mode.value:
            raise SessionControllerError("session_controller_mode_mismatch")
        if payload.get("training_backend") != self.project.training_backend:
            raise SessionControllerError("session_controller_backend_mismatch")
        if payload.get("public_artifact_safe") is not True:
            raise SessionControllerError("session_controller_public_safety_invalid")
        if payload.get("private_paths_public") is not False:
            raise SessionControllerError("session_controller_public_safety_invalid")
        if payload.get("credential_values_public") is not False:
            raise SessionControllerError("session_controller_public_safety_invalid")
        try:
            lineage = CheckpointLineage.from_dict(payload.get("lineage") or {})
            active_records = self._active_records(payload)
        except (TypeError, ValueError) as exc:
            raise SessionControllerError("session_controller_state_invalid") from exc
        if lineage.project_hash != self.project.content_hash:
            raise SessionControllerError("session_controller_lineage_project_mismatch")
        if len(active_records) > _MAX_ACTIVE_WORK_UNITS:
            raise SessionControllerError("session_controller_active_work_capacity")
        if self.project.mode is TrainingMode.STABLE_SHARDED and len(active_records) > 1:
            raise SessionControllerError("session_controller_stable_rank_group_conflict")
        active_keys: set[str] = set()
        active_ids: set[str] = set()
        checkpoint_hashes = {item.content_hash for item in lineage.checkpoints}
        for record in active_records:
            if not isinstance(record, dict) or set(record) != {
                "work",
                "contributor_id_hash",
                "lease_expires_at",
            }:
                raise SessionControllerError("session_controller_active_work_invalid")
            try:
                work = WorkUnit.from_dict(record.get("work") or {})
            except (TypeError, ValueError) as exc:
                raise SessionControllerError(
                    "session_controller_active_work_invalid"
                ) from exc
            self._validate_work(work, checkpoint_hashes=checkpoint_hashes)
            owner = record.get("contributor_id_hash")
            if owner is not None and not _HASH_PATTERN.fullmatch(str(owner)):
                raise SessionControllerError(
                    "session_controller_active_owner_invalid"
                )
            expiry = record.get("lease_expires_at")
            if expiry is not None and (
                isinstance(expiry, bool)
                or not isinstance(expiry, (int, float))
                or not math.isfinite(float(expiry))
                or float(expiry) <= 0
            ):
                raise SessionControllerError(
                    "session_controller_lease_expiry_invalid"
                )
            key = _work_key(work)
            if key in active_keys or work.work_id in active_ids:
                raise SessionControllerError("session_controller_active_work_conflict")
            active_keys.add(key)
            active_ids.add(work.work_id)
        terminals = payload.get("terminals")
        if not isinstance(terminals, list):
            raise SessionControllerError("session_controller_terminals_invalid")
        fenced = payload.get("fenced_work_units", [])
        if not isinstance(fenced, list):
            raise SessionControllerError("session_controller_fenced_work_invalid")
        keys: set[str] = set()
        receipt_ids: set[str] = set()
        for raw in terminals:
            if not isinstance(raw, dict):
                raise SessionControllerError("session_controller_terminal_invalid")
            try:
                work = WorkUnit.from_dict(raw.get("work") or {})
                receipt = ContributionReceipt.from_dict(raw.get("receipt") or {})
                base = CheckpointRef.from_dict(raw.get("base_checkpoint") or {})
                output = (
                    CheckpointRef.from_dict(raw["output_checkpoint"])
                    if raw.get("output_checkpoint") is not None
                    else None
                )
                validate_receipt_binding(
                    receipt,
                    work=work,
                    base_checkpoint=base,
                    output_checkpoint=output,
                )
            except (TypeError, ValueError) as exc:
                raise SessionControllerError("session_controller_terminal_invalid") from exc
            key = _work_key(work)
            if work.project_hash != self.project.content_hash:
                raise SessionControllerError("session_controller_work_project_mismatch")
            if work.mode is not self.project.mode:
                raise SessionControllerError("session_controller_work_mode_mismatch")
            if work.backend != self.project.training_backend:
                raise SessionControllerError("session_controller_work_backend_mismatch")
            if key in keys or receipt.receipt_id in receipt_ids:
                raise SessionControllerError("session_controller_terminal_replay")
            if base.content_hash not in checkpoint_hashes:
                raise SessionControllerError("session_controller_base_checkpoint_missing")
            if output is not None and output.content_hash not in checkpoint_hashes:
                raise SessionControllerError("session_controller_checkpoint_missing")
            keys.add(key)
            receipt_ids.add(receipt.receipt_id)
        if active_keys & keys:
            raise SessionControllerError("session_controller_active_terminal_conflict")
        fenced_keys: set[str] = set()
        for raw in fenced:
            if not isinstance(raw, dict) or set(raw) != {
                "work",
                "contributor_id_hash",
                "lease_expires_at",
                "fenced_at",
                "reason",
            }:
                raise SessionControllerError(
                    "session_controller_fenced_work_invalid"
                )
            try:
                work = WorkUnit.from_dict(raw.get("work") or {})
            except (TypeError, ValueError) as exc:
                raise SessionControllerError(
                    "session_controller_fenced_work_invalid"
                ) from exc
            self._validate_work(work, checkpoint_hashes=checkpoint_hashes)
            owner = raw.get("contributor_id_hash")
            if owner is not None and not _HASH_PATTERN.fullmatch(str(owner)):
                raise SessionControllerError(
                    "session_controller_fenced_work_invalid"
                )
            expiry = raw.get("lease_expires_at")
            if expiry is not None and (
                isinstance(expiry, bool)
                or not isinstance(expiry, (int, float))
                or not math.isfinite(float(expiry))
                or float(expiry) <= 0
            ):
                raise SessionControllerError(
                    "session_controller_fenced_work_invalid"
                )
            if not str(raw.get("fenced_at") or "") or not str(
                raw.get("reason") or ""
            ):
                raise SessionControllerError(
                    "session_controller_fenced_work_invalid"
                )
            key = _work_key(work)
            if key in fenced_keys or key in active_keys or key in keys:
                raise SessionControllerError(
                    "session_controller_fenced_work_conflict"
                )
            fenced_keys.add(key)
        upgraded = dict(payload)
        upgraded.setdefault("fenced_work_units", [])
        return self._sync_active_state(upgraded, active_records)

    def _validate_work(
        self,
        work: WorkUnit,
        *,
        checkpoint_hashes: set[str],
        required_base_hash: str | None = None,
    ) -> None:
        if work.project_hash != self.project.content_hash:
            raise SessionControllerError("session_controller_work_project_mismatch")
        if work.mode is not self.project.mode:
            raise SessionControllerError("session_controller_work_mode_mismatch")
        if work.backend != self.project.training_backend:
            raise SessionControllerError("session_controller_work_backend_mismatch")
        if work.base_checkpoint_hash not in checkpoint_hashes:
            raise SessionControllerError("session_controller_work_checkpoint_missing")
        if (
            required_base_hash is not None
            and work.base_checkpoint_hash != required_base_hash
        ):
            raise SessionControllerError("session_controller_work_stale_checkpoint")

    def _save(self, state: dict[str, Any]) -> dict[str, Any]:
        state = self._sync_active_state(state, self._active_records(state))
        state["updated_at"] = _utc_now()
        sealed = _seal(state)
        _atomic_json(self.state_path, sealed)
        self._write_projections(sealed)
        return sealed

    def _write_projections(self, state: dict[str, Any]) -> None:
        lineage = CheckpointLineage.from_dict(state["lineage"])
        for checkpoint in lineage.checkpoints:
            name = checkpoint.content_hash.split(":", 1)[1] + ".json"
            _atomic_json(
                self.workspace / CONTROL_DIR / "checkpoints" / name,
                checkpoint.to_dict(),
            )
        for terminal in state["terminals"]:
            receipt = ContributionReceipt.from_dict(terminal["receipt"])
            name = receipt.content_hash.split(":", 1)[1] + ".json"
            _atomic_json(
                self.workspace / CONTROL_DIR / "receipts" / name,
                receipt.to_dict(),
            )

    def initialize(self, lineage: CheckpointLineage) -> dict[str, Any]:
        """Bind the controller to an existing canonical checkpoint lineage."""

        if lineage.project_hash != self.project.content_hash:
            raise SessionControllerError("session_controller_lineage_project_mismatch")
        with self._locked() as state:
            if state is not None:
                existing = CheckpointLineage.from_dict(state["lineage"])
                if not _lineage_is_prefix(lineage, existing):
                    raise SessionControllerError("session_controller_lineage_conflict")
                self._write_projections(state)
                return self._report("initialize", state, idempotent_replay=True)
            state = self._save(
                {
                    "schema": SESSION_CONTROLLER_SCHEMA,
                    "project_hash": self.project.content_hash,
                    "mode": self.project.mode.value,
                    "training_backend": self.project.training_backend,
                    "lineage": lineage.to_dict(),
                    "controller_revision": 2,
                    "active_work_units": [],
                    "active_work": None,
                    "active_contributor_id_hash": None,
                    "terminals": [],
                    "fenced_work_units": [],
                    "updated_at": _utc_now(),
                    "credential_values_public": False,
                    "private_paths_public": False,
                    "public_artifact_safe": True,
                }
            )
            return self._report("initialize", state, idempotent_replay=False)

    def issue(
        self,
        work: WorkUnit,
        *,
        contributor_id_hash: str | None = None,
        replace_active: bool = False,
        lease_expires_at: float | None = None,
    ) -> dict[str, Any]:
        """Issue one Work Unit or generation-fence its prior active lease."""

        if contributor_id_hash is not None and not _HASH_PATTERN.fullmatch(
            str(contributor_id_hash)
        ):
            raise SessionControllerError("session_controller_active_owner_invalid")
        if lease_expires_at is not None and (
            isinstance(lease_expires_at, bool)
            or not isinstance(lease_expires_at, (int, float))
            or not math.isfinite(float(lease_expires_at))
            or float(lease_expires_at) <= 0
        ):
            raise SessionControllerError("session_controller_lease_expiry_invalid")
        if inspect_workspace(self.workspace)["lifecycle_state"] == "paused":
            raise SessionControllerError("session_controller_workspace_paused")
        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            lineage = CheckpointLineage.from_dict(state["lineage"])
            self._validate_work(
                work,
                checkpoint_hashes={item.content_hash for item in lineage.checkpoints},
                required_base_hash=lineage.checkpoints[-1].content_hash,
            )
            active_records = self._active_records(state)
            matching = [
                item
                for item in active_records
                if WorkUnit.from_dict(item["work"]).work_id == work.work_id
            ]
            active = WorkUnit.from_dict(matching[0]["work"]) if matching else None
            if active is not None:
                if active.content_hash == work.content_hash:
                    current_owner = matching[0].get("contributor_id_hash")
                    if current_owner is not None and current_owner != contributor_id_hash:
                        raise SessionControllerError(
                            "session_controller_active_owner_mismatch"
                        )
                    if (
                        lease_expires_at is not None
                        and (
                            matching[0].get("lease_expires_at") is None
                            or float(lease_expires_at)
                            > float(matching[0]["lease_expires_at"])
                        )
                    ):
                        matching[0]["lease_expires_at"] = float(lease_expires_at)
                        updated = self._sync_active_state(state, active_records)
                        updated = self._save(updated)
                        return self._report(
                            "issue", updated, idempotent_replay=True
                        )
                    return self._report("issue", state, idempotent_replay=True)
                if not replace_active:
                    raise SessionControllerError("session_controller_active_work_exists")
                if not _same_work_attempt(active, work):
                    raise SessionControllerError("session_controller_reassignment_mismatch")
                if work.generation != active.generation + 1:
                    raise SessionControllerError("session_controller_generation_not_next")
            elif replace_active:
                raise SessionControllerError("session_controller_active_work_missing")
            prior = [
                item
                for item in state["terminals"]
                if (item.get("work") or {}).get("work_id") == work.work_id
            ]
            fenced_prior = [
                item
                for item in state.get("fenced_work_units") or []
                if (item.get("work") or {}).get("work_id") == work.work_id
            ]
            latest_terminal = (
                max(
                    prior,
                    key=lambda item: int((item.get("work") or {}).get("generation") or 0),
                )
                if prior
                else None
            )
            latest_fenced = (
                max(
                    fenced_prior,
                    key=lambda item: int((item.get("work") or {}).get("generation") or 0),
                )
                if fenced_prior
                else None
            )
            terminal_generation = int(
                ((latest_terminal or {}).get("work") or {}).get("generation") or 0
            )
            fenced_generation = int(
                ((latest_fenced or {}).get("work") or {}).get("generation") or 0
            )
            if latest_terminal is not None and terminal_generation >= fenced_generation:
                latest_work = WorkUnit.from_dict(latest_terminal["work"])
                latest_receipt = ContributionReceipt.from_dict(
                    latest_terminal["receipt"]
                )
                if latest_receipt.outcome is ReceiptOutcome.ACCEPTED:
                    raise SessionControllerError("session_controller_accepted_work_terminal")
                if work.generation != latest_work.generation + 1:
                    raise SessionControllerError("session_controller_generation_not_next")
                if not _same_work_attempt(latest_work, work):
                    raise SessionControllerError("session_controller_reassignment_mismatch")
            elif latest_fenced is not None:
                latest_work = WorkUnit.from_dict(latest_fenced["work"])
                if latest_fenced.get("reason") != "lease_expired":
                    raise SessionControllerError("session_controller_generation_stale")
                if work.generation != latest_work.generation + 1:
                    raise SessionControllerError("session_controller_generation_not_next")
                if not _same_work_attempt(latest_work, work):
                    raise SessionControllerError("session_controller_reassignment_mismatch")
            if active is None and len(active_records) >= _MAX_ACTIVE_WORK_UNITS:
                raise SessionControllerError("session_controller_active_work_capacity")
            if (
                self.project.mode is TrainingMode.STABLE_SHARDED
                and active is None
                and active_records
            ):
                raise SessionControllerError(
                    "session_controller_stable_rank_group_active"
                )
            updated = dict(state)
            if matching:
                active_records.remove(matching[0])
                updated["fenced_work_units"] = [
                    *list(state.get("fenced_work_units") or []),
                    {
                        **matching[0],
                        "fenced_at": _utc_now(),
                        "reason": "generation_reassigned",
                    },
                ]
            active_records.append(
                {
                    "work": work.to_dict(),
                    "contributor_id_hash": contributor_id_hash,
                    "lease_expires_at": (
                        float(lease_expires_at)
                        if lease_expires_at is not None
                        else None
                    ),
                }
            )
            updated = self._sync_active_state(updated, active_records)
            updated = self._save(updated)
            return self._report(
                "reassign" if replace_active else "issue",
                updated,
                idempotent_replay=False,
            )

    def commit(
        self,
        work: WorkUnit,
        receipt: ContributionReceipt,
        *,
        base_checkpoint: CheckpointRef,
        output_checkpoint: CheckpointRef | None = None,
    ) -> dict[str, Any]:
        """Commit one terminal receipt and optional checkpoint exactly once."""

        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            for terminal in state["terminals"]:
                terminal_work = WorkUnit.from_dict(terminal["work"])
                if _work_key(terminal_work) != _work_key(work):
                    continue
                existing = ContributionReceipt.from_dict(terminal["receipt"])
                existing_output = terminal.get("output_checkpoint")
                if (
                    terminal_work.content_hash == work.content_hash
                    and existing.content_hash == receipt.content_hash
                    and existing_output
                    == (output_checkpoint.to_dict() if output_checkpoint else None)
                ):
                    self._write_projections(state)
                    return self._report("commit", state, idempotent_replay=True)
                raise SessionControllerError("session_controller_terminal_conflict")
            active_records = self._active_records(state)
            matching = [
                item
                for item in active_records
                if _work_key(WorkUnit.from_dict(item["work"])) == _work_key(work)
            ]
            if not matching:
                newer = [
                    WorkUnit.from_dict(item["work"])
                    for item in active_records
                    if WorkUnit.from_dict(item["work"]).work_id == work.work_id
                    and WorkUnit.from_dict(item["work"]).generation > work.generation
                ]
                newer.extend(
                    WorkUnit.from_dict(item["work"])
                    for item in state["terminals"]
                    if WorkUnit.from_dict(item["work"]).work_id == work.work_id
                    and WorkUnit.from_dict(item["work"]).generation > work.generation
                )
                fenced = [
                    WorkUnit.from_dict(item["work"])
                    for item in state.get("fenced_work_units") or []
                    if WorkUnit.from_dict(item["work"]).work_id == work.work_id
                    and WorkUnit.from_dict(item["work"]).generation >= work.generation
                ]
                if newer or fenced:
                    raise SessionControllerError("session_controller_generation_stale")
                raise SessionControllerError("session_controller_active_work_missing")
            active_record = matching[0]
            active = WorkUnit.from_dict(active_record["work"])
            if active.content_hash != work.content_hash:
                raise SessionControllerError("session_controller_active_work_mismatch")
            active_owner = active_record.get("contributor_id_hash")
            if (
                active_owner is not None
                and receipt.contributor_id_hash != active_owner
            ):
                raise SessionControllerError(
                    "session_controller_receipt_owner_mismatch"
                )
            lineage = CheckpointLineage.from_dict(state["lineage"])
            known_checkpoints = {
                item.content_hash: item for item in lineage.checkpoints
            }
            if base_checkpoint.content_hash != active.base_checkpoint_hash:
                raise SessionControllerError("session_controller_base_checkpoint_mismatch")
            known_base = known_checkpoints.get(base_checkpoint.content_hash)
            if known_base is None or known_base.to_dict() != base_checkpoint.to_dict():
                raise SessionControllerError("session_controller_base_checkpoint_missing")
            try:
                validate_receipt_binding(
                    receipt,
                    work=work,
                    base_checkpoint=base_checkpoint,
                    output_checkpoint=output_checkpoint,
                )
            except ValueError as exc:
                raise SessionControllerError("session_controller_receipt_invalid") from exc
            if output_checkpoint is not None:
                if base_checkpoint.content_hash != lineage.checkpoints[-1].content_hash:
                    raise SessionControllerError(
                        "session_controller_checkpoint_parent_stale"
                    )
                lineage = lineage.append(output_checkpoint)
            updated = dict(state)
            updated["lineage"] = lineage.to_dict()
            active_records.remove(active_record)
            fenced = list(state.get("fenced_work_units") or [])
            if output_checkpoint is not None:
                stale_records = [
                    item
                    for item in active_records
                    if WorkUnit.from_dict(item["work"]).base_checkpoint_hash
                    != output_checkpoint.content_hash
                ]
                active_records = [
                    item for item in active_records if item not in stale_records
                ]
                fenced.extend(
                    {
                        **item,
                        "fenced_at": _utc_now(),
                        "reason": "checkpoint_advanced",
                    }
                    for item in stale_records
                )
            updated["fenced_work_units"] = fenced
            updated = self._sync_active_state(updated, active_records)
            updated["terminals"] = [
                *state["terminals"],
                {
                    "work": work.to_dict(),
                    "receipt": receipt.to_dict(),
                    "base_checkpoint": base_checkpoint.to_dict(),
                    "output_checkpoint": (
                        output_checkpoint.to_dict() if output_checkpoint else None
                    ),
                },
            ]
            updated = self._save(updated)
            return self._report("commit", updated, idempotent_replay=False)

    def status(self) -> dict[str, Any]:
        with self._locked() as state:
            if state is None:
                return {
                    "schema": SESSION_CONTROLLER_REPORT_SCHEMA,
                    "initialized": False,
                    "project_hash": self.project.content_hash,
                    "state": "uninitialized",
                    "active_work": None,
                    "active_work_count": 0,
                    "active_work_units": [],
                    "concurrent_elastic_work_supported": (
                        self.project.mode is TrainingMode.ELASTIC_DELTA
                    ),
                    "stable_rank_group_restart_supported": (
                        self.project.mode is TrainingMode.STABLE_SHARDED
                    ),
                    "durable_lease_ownership": True,
                    "terminal_count": 0,
                    "fenced_work_count": 0,
                    "checkpoint_count": 0,
                    "credential_values_public": False,
                    "public_artifact_safe": True,
                    "private_paths_public": False,
                }
            self._write_projections(state)
            return self._report("status", state, idempotent_replay=False)

    def lineage(self) -> CheckpointLineage:
        """Return the validated canonical lineage without exposing state paths."""

        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            return CheckpointLineage.from_dict(state["lineage"])

    def active_work(self) -> WorkUnit | None:
        """Return the only active Work Unit, preserving the phase-2 API."""

        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            records = self._active_records(state)
            if len(records) > 1:
                raise SessionControllerError("session_controller_active_work_ambiguous")
            return WorkUnit.from_dict(records[0]["work"]) if records else None

    def active_works(self) -> tuple[WorkUnit, ...]:
        """Return all active Work Units in deterministic order."""

        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            return tuple(
                WorkUnit.from_dict(item["work"])
                for item in self._active_records(state)
            )

    def active_lease(
        self, work_id: str, generation: int | None = None
    ) -> tuple[WorkUnit, str | None, float | None] | None:
        """Return one active Work Unit with its durable pseudonymous owner."""

        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            matches = []
            for item in self._active_records(state):
                work = WorkUnit.from_dict(item["work"])
                if work.work_id == str(work_id) and (
                    generation is None or work.generation == int(generation)
                ):
                    matches.append((work, item))
            if not matches:
                return None
            if len(matches) != 1:
                raise SessionControllerError("session_controller_active_work_ambiguous")
            work, item = matches[0]
            owner = item.get("contributor_id_hash")
            expiry = item.get("lease_expires_at")
            return (
                work,
                str(owner) if owner is not None else None,
                float(expiry) if expiry is not None else None,
            )

    def active_contributor_id_hash(self) -> str | None:
        """Return the owner when exactly one Work Unit is active."""

        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            records = self._active_records(state)
            if len(records) > 1:
                raise SessionControllerError("session_controller_active_work_ambiguous")
            value = records[0].get("contributor_id_hash") if records else None
            return str(value) if value is not None else None

    def renew(
        self,
        work_id: str,
        generation: int,
        *,
        contributor_id_hash: str,
        lease_expires_at: float,
    ) -> dict[str, Any]:
        """Persist a monotonic lease extension after backend heartbeat success."""

        if not _HASH_PATTERN.fullmatch(str(contributor_id_hash)):
            raise SessionControllerError("session_controller_active_owner_invalid")
        if (
            isinstance(lease_expires_at, bool)
            or not isinstance(lease_expires_at, (int, float))
            or not math.isfinite(float(lease_expires_at))
            or float(lease_expires_at) <= 0
        ):
            raise SessionControllerError("session_controller_lease_expiry_invalid")
        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            records = self._active_records(state)
            matching = []
            for item in records:
                work = WorkUnit.from_dict(item["work"])
                if work.work_id == str(work_id):
                    matching.append((work, item))
            if not matching:
                raise SessionControllerError("session_controller_active_work_missing")
            work, record = matching[0]
            if work.generation > int(generation):
                raise SessionControllerError("session_controller_generation_stale")
            if work.generation != int(generation):
                raise SessionControllerError("session_controller_active_work_mismatch")
            if record.get("contributor_id_hash") != contributor_id_hash:
                raise SessionControllerError("session_controller_active_owner_mismatch")
            current = record.get("lease_expires_at")
            if current is not None and float(lease_expires_at) < float(current):
                raise SessionControllerError("session_controller_lease_expiry_regressed")
            if current is not None and float(lease_expires_at) == float(current):
                return self._report("renew", state, idempotent_replay=True)
            record["lease_expires_at"] = float(lease_expires_at)
            updated = self._save(self._sync_active_state(state, records))
            return self._report("renew", updated, idempotent_replay=False)

    def fence_expired(self, *, now: float) -> dict[str, Any]:
        """Move expired durable leases out of the active scheduling set."""

        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(float(now))
            or float(now) < 0
        ):
            raise SessionControllerError("session_controller_clock_invalid")
        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            records = self._active_records(state)
            expired = [
                item
                for item in records
                if item.get("lease_expires_at") is not None
                and float(item["lease_expires_at"]) <= float(now)
            ]
            if not expired:
                return self._report(
                    "fence_expired", state, idempotent_replay=True
                )
            active = [item for item in records if item not in expired]
            updated = dict(state)
            updated["fenced_work_units"] = [
                *list(state.get("fenced_work_units") or []),
                *(
                    {
                        **item,
                        "fenced_at": _utc_now(),
                        "reason": "lease_expired",
                    }
                    for item in expired
                ),
            ]
            updated = self._save(self._sync_active_state(updated, active))
            return self._report(
                "fence_expired", updated, idempotent_replay=False
            )

    def terminal(
        self, work_id: str, generation: int
    ) -> tuple[WorkUnit, ContributionReceipt, CheckpointRef, CheckpointRef | None] | None:
        """Return one validated terminal generation for idempotent recovery."""

        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            for raw in state["terminals"]:
                work = WorkUnit.from_dict(raw["work"])
                if work.work_id != str(work_id) or work.generation != int(generation):
                    continue
                return (
                    work,
                    ContributionReceipt.from_dict(raw["receipt"]),
                    CheckpointRef.from_dict(raw["base_checkpoint"]),
                    (
                        CheckpointRef.from_dict(raw["output_checkpoint"])
                        if raw.get("output_checkpoint") is not None
                        else None
                    ),
                )
            return None

    def next_generation(self, work_id: str) -> int:
        """Return the next fenced generation for one logical Work Unit."""

        candidate = str(work_id)
        with self._locked() as state:
            if state is None:
                raise SessionControllerError("session_controller_not_initialized")
            generations: list[int] = []
            for item in self._active_records(state):
                work = WorkUnit.from_dict(item["work"])
                if work.work_id == candidate:
                    generations.append(work.generation)
            for collection in (
                state["terminals"],
                state.get("fenced_work_units") or [],
            ):
                for item in collection:
                    work = WorkUnit.from_dict(item["work"])
                    if work.work_id == candidate:
                        generations.append(work.generation)
            return max(generations, default=0) + 1

    def _report(
        self, action: str, state: dict[str, Any], *, idempotent_replay: bool
    ) -> dict[str, Any]:
        lineage = CheckpointLineage.from_dict(state["lineage"])
        records = self._active_records(state)
        active = records[0]["work"] if len(records) == 1 else None
        payload = {
            "schema": SESSION_CONTROLLER_REPORT_SCHEMA,
            "initialized": True,
            "action": action,
            "project_hash": self.project.content_hash,
            "state": "work_active" if records else "idle",
            "active_work": active,
            "active_work_count": len(records),
            "active_work_units": [
                {
                    "work": item["work"],
                    "lease_expires_at": item.get("lease_expires_at"),
                }
                for item in records
            ],
            "concurrent_elastic_work_supported": (
                self.project.mode is TrainingMode.ELASTIC_DELTA
            ),
            "stable_rank_group_restart_supported": (
                self.project.mode is TrainingMode.STABLE_SHARDED
            ),
            "durable_lease_ownership": True,
            "terminal_count": len(state["terminals"]),
            "fenced_work_count": len(state.get("fenced_work_units") or []),
            "checkpoint_count": len(lineage.checkpoints),
            "lineage_hash": lineage.content_hash,
            "head_checkpoint_hash": lineage.checkpoints[-1].content_hash,
            "idempotent_replay": bool(idempotent_replay),
            "command_executed": False,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        return _seal(payload)


def inspect_session_controller(workspace: str | Path) -> dict[str, Any] | None:
    controller = SessionController(workspace)
    if not controller.state_path.is_file():
        return None
    return controller.status()
