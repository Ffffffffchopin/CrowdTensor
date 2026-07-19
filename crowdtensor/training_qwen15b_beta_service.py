"""Durable local control service for Qwen 1.5B four-GPU training jobs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .qwen15b_training import MODEL_ID, MODEL_REVISION


STORE_SCHEMA = "crowdtensor_training_qwen15b_beta_job_store_v1"
STATUS_SCHEMA = "crowdtensor_training_qwen15b_beta_job_status_v1"
SERVICE_SCHEMA = "crowdtensor_training_qwen15b_beta_service_v1"
EVENT_SCHEMA = "crowdtensor_training_qwen15b_beta_event_v1"
PHASES = (
    "model_resolution",
    "dataset",
    "account_preflight",
    "allocation",
    "kernel_launch",
    "stage_loading",
    "forward",
    "backward",
    "checkpoint",
    "recovery",
    "evaluation",
    "export",
    "cleanup",
)
TERMINAL_STATES = {"completed", "cancelled", "failed", "cleaned"}
RUNNABLE_STATES = {"queued", "recovery_required", "blocked"}
FORBIDDEN_PUBLIC_KEYS = {
    "credential",
    "credentials",
    "coordinator_token",
    "coordinator_url",
    "cookie",
    "job_dir",
    "output_dir",
    "token_file",
    "token_files",
    "token_path",
    "private_request",
}


def _now() -> float:
    return time.time()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _public_value(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_PUBLIC_KEYS or lowered.endswith("_path"):
                continue
            result[str(key)] = _public_value(item)
        return result
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return value


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


class TrainingBetaJobStore:
    """SQLite job/event ledger with idempotency and monotonic progress gates."""

    def __init__(self, path: str | Path, *, max_queue_size: int = 8) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_queue_size = int(max_queue_size)
        if self.max_queue_size < 1:
            raise ValueError("training_beta_queue_size_must_be_positive")
        with _connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    global_step INTEGER NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    lease_owner_hash TEXT NOT NULL DEFAULT '',
                    lease_expires_at REAL NOT NULL DEFAULT 0,
                    status_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(job_id, event_id),
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE INDEX IF NOT EXISTS jobs_state_created
                    ON jobs(state, created_at);
                """
            )

    @staticmethod
    def _validate_request(request: dict[str, Any]) -> None:
        if (
            request.get("model") != MODEL_ID
            or request.get("model_revision") not in {None, "", MODEL_REVISION}
            or request.get("topology") != "kaggle-2x-t4x2"
            or int(request.get("steps") or 0) != 8
        ):
            raise ValueError("training_beta_request_contract_invalid")
        if not str(request.get("job_dir") or ""):
            raise ValueError("training_beta_private_job_dir_required")

    def submit(
        self,
        request: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        self._validate_request(request)
        key = str(idempotency_key).strip()
        if not key:
            raise ValueError("training_beta_idempotency_key_required")
        request_hash = _stable_hash(
            {
                "model": request["model"],
                "model_revision": MODEL_REVISION,
                "topology": request["topology"],
                "steps": int(request["steps"]),
                "job_dir_hash": _stable_hash(str(Path(request["job_dir"]).resolve())),
            }
        )
        now = _now()
        with _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if previous is not None:
                if str(previous["request_hash"]) != request_hash:
                    connection.execute("ROLLBACK")
                    raise ValueError("training_beta_idempotency_conflict")
                connection.execute("COMMIT")
                return self._public_row(previous, connection=connection), False
            queued = int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state IN ('queued','recovery_required','blocked')"
                ).fetchone()[0]
            )
            if queued >= self.max_queue_size:
                connection.execute("ROLLBACK")
                raise RuntimeError("training_beta_queue_full")
            job_id = "qwen15b-beta-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            status = {
                "schema": STATUS_SCHEMA,
                "job_id": job_id,
                "backend": "cuda",
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "topology": "kaggle-2x-t4x2",
                "steps": 8,
                "overall_state": "queued",
                "current_phase": "model_resolution",
                "global_step": 0,
                "retry_count": 0,
                "phases": {phase: {"state": "pending"} for phase in PHASES},
                "blockers": [],
                "request_hash": request_hash,
                "credential_values_public": False,
                "credential_paths_public": False,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id,idempotency_key,request_hash,request_json,state,phase,
                    global_step,retry_count,revision,cancel_requested,status_json,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    key,
                    request_hash,
                    json.dumps(request, sort_keys=True),
                    "queued",
                    "model_resolution",
                    0,
                    0,
                    1,
                    0,
                    json.dumps(status, sort_keys=True),
                    now,
                    now,
                ),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_id=f"submit:{request_hash}",
                phase="model_resolution",
                value={"operation": "submitted", "request_hash": request_hash},
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._public_row(row), True

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        event_id: str,
        phase: str,
        value: dict[str, Any],
    ) -> bool:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO events(job_id,event_id,phase,event_json,created_at)
            VALUES(?,?,?,?,?)
            """,
            (
                job_id,
                str(event_id),
                str(phase),
                json.dumps(_public_value(value), sort_keys=True),
                _now(),
            ),
        )
        return cursor.rowcount == 1

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: float = 3600.0,
        preferred_job_id: str = "",
    ) -> dict[str, Any] | None:
        now = _now()
        owner_hash = _stable_hash(str(worker_id))
        with _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT job_id,lease_expires_at FROM jobs WHERE state = 'running' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if active is not None and float(active["lease_expires_at"] or 0) > now:
                connection.execute("COMMIT")
                return None
            if active is not None:
                connection.execute(
                    """
                    UPDATE jobs SET state='recovery_required', phase='recovery',
                        lease_owner_hash='', lease_expires_at=0, revision=revision+1,
                        updated_at=? WHERE job_id=?
                    """,
                    (now, active["job_id"]),
                )
            if preferred_job_id:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE job_id=? AND state IN ('queued','recovery_required','blocked')",
                    (preferred_job_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM jobs WHERE state IN ('queued','recovery_required','blocked')
                    ORDER BY CASE state WHEN 'recovery_required' THEN 0 ELSE 1 END, created_at
                    LIMIT 1
                    """
                ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            retry_count = int(row["retry_count"] or 0) + int(row["state"] != "queued")
            connection.execute(
                """
                UPDATE jobs SET state='running', phase=?, retry_count=?,
                    lease_owner_hash=?, lease_expires_at=?, cancel_requested=0,
                    revision=revision+1, updated_at=? WHERE job_id=?
                """,
                (
                    "recovery" if row["state"] == "recovery_required" else "model_resolution",
                    retry_count,
                    owner_hash,
                    now + float(lease_seconds),
                    now,
                    row["job_id"],
                ),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()
            self._insert_event(
                connection,
                job_id=str(row["job_id"]),
                event_id=f"claim:{int(claimed['revision'])}",
                phase=str(claimed["phase"]),
                value={"operation": "claimed", "worker_id_hash": owner_hash},
            )
            connection.execute("COMMIT")
            return {
                "public": self._public_row(claimed),
                "private_request": json.loads(str(claimed["request_json"])),
            }

    def update_status(
        self,
        job_id: str,
        status: dict[str, Any],
        *,
        event_id: str,
    ) -> dict[str, Any]:
        public = _public_value(status)
        phase = str(public.get("current_phase") or "")
        if phase not in PHASES:
            raise ValueError("training_beta_phase_invalid")
        new_step = int(public.get("global_step") or 0)
        new_retry = int(public.get("retry_count") or 0)
        state = str(public.get("overall_state") or "running")
        if state not in {"queued", "running", "blocked", *TERMINAL_STATES}:
            raise ValueError("training_beta_state_invalid")
        with _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise KeyError("training_beta_job_not_found")
            if new_step < int(row["global_step"] or 0):
                connection.execute("ROLLBACK")
                raise ValueError("training_beta_global_step_regression")
            inserted = self._insert_event(
                connection,
                job_id=job_id,
                event_id=event_id,
                phase=phase,
                value={
                    "operation": "status_updated",
                    "state": state,
                    "global_step": new_step,
                    "retry_count": new_retry,
                },
            )
            if inserted:
                connection.execute(
                    """
                    UPDATE jobs SET state=?,phase=?,global_step=?,retry_count=?,
                        status_json=?,revision=revision+1,updated_at=? WHERE job_id=?
                    """,
                    (
                        state,
                        phase,
                        new_step,
                        max(new_retry, int(row["retry_count"] or 0)),
                        json.dumps(public, sort_keys=True),
                        _now(),
                        job_id,
                    ),
                )
            updated = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._public_row(updated)

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        with _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise KeyError("training_beta_job_not_found")
            if bool(row["cancel_requested"]):
                connection.execute("COMMIT")
                return self._public_row(row)
            if row["state"] in {"queued", "blocked", "recovery_required"}:
                state = "cancelled"
            else:
                state = str(row["state"])
            connection.execute(
                """
                UPDATE jobs SET state=?,cancel_requested=1,revision=revision+1,
                    updated_at=? WHERE job_id=?
                """,
                (state, _now(), job_id),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_id="cancel-requested",
                phase=str(row["phase"]),
                value={"operation": "cancel_requested"},
            )
            updated = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._public_row(updated)

    def requeue(self, job_id: str) -> dict[str, Any]:
        with _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise KeyError("training_beta_job_not_found")
            if row["state"] == "completed":
                connection.execute("COMMIT")
                return self._public_row(row)
            if row["state"] == "cancelled":
                connection.execute("ROLLBACK")
                raise ValueError("training_beta_cancelled_job_not_resumable")
            connection.execute(
                """
                UPDATE jobs SET state='recovery_required',phase='recovery',
                    cancel_requested=0,lease_owner_hash='',lease_expires_at=0,
                    revision=revision+1,updated_at=? WHERE job_id=?
                """,
                (_now(), job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._public_row(updated)

    def private_request(self, job_id: str) -> dict[str, Any]:
        with _connect(self.path) as connection:
            row = connection.execute("SELECT request_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("training_beta_job_not_found")
        return json.loads(str(row["request_json"]))

    def update_private_inputs(self, job_id: str, values: dict[str, Any]) -> None:
        allowed = {
            "kaggle_token_files",
            "kaggle_raw_token_file",
            "kaggle_raw_token_username",
            "allocation_timeout_seconds",
        }
        patch = {key: values[key] for key in allowed if key in values}
        with _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT request_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise KeyError("training_beta_job_not_found")
            request = json.loads(str(row["request_json"]))
            request.update(patch)
            connection.execute(
                "UPDATE jobs SET request_json=?,revision=revision+1,updated_at=? WHERE job_id=?",
                (json.dumps(request, sort_keys=True), _now(), job_id),
            )
            connection.execute("COMMIT")

    def only_job_id(self) -> str:
        with _connect(self.path) as connection:
            rows = connection.execute("SELECT job_id FROM jobs ORDER BY created_at").fetchall()
        if len(rows) != 1:
            raise KeyError("training_beta_single_job_pointer_missing")
        return str(rows[0]["job_id"])

    def status(self, job_id: str) -> dict[str, Any]:
        with _connect(self.path) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("training_beta_job_not_found")
            return self._public_row(row, connection=connection)

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with _connect(self.path) as connection:
            rows = connection.execute(
                "SELECT sequence,event_id,phase,event_json,created_at FROM events WHERE job_id=? ORDER BY sequence",
                (job_id,),
            ).fetchall()
        return [
            {
                "schema": EVENT_SCHEMA,
                "sequence": int(row["sequence"]),
                "event_id": str(row["event_id"]),
                "phase": str(row["phase"]),
                "event": json.loads(str(row["event_json"])),
                "created_at": float(row["created_at"]),
                "public_artifact_safe": True,
            }
            for row in rows
        ]

    def _public_row(
        self,
        row: sqlite3.Row,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        owns = connection is None
        current = connection or _connect(self.path)
        try:
            queued_ahead = 0
            if row["state"] in {"queued", "recovery_required", "blocked"}:
                queued_ahead = int(
                    current.execute(
                        """
                        SELECT COUNT(*) FROM jobs
                        WHERE state IN ('queued','recovery_required','blocked') AND created_at < ?
                        """,
                        (float(row["created_at"]),),
                    ).fetchone()[0]
                )
            stored = _public_value(json.loads(str(row["status_json"])))
            stored.update(
                {
                    "schema": STATUS_SCHEMA,
                    "job_id": str(row["job_id"]),
                    "overall_state": str(row["state"]),
                    "current_phase": str(row["phase"]),
                    "global_step": int(row["global_step"]),
                    "retry_count": int(row["retry_count"]),
                    "revision": int(row["revision"]),
                    "cancel_requested": bool(row["cancel_requested"]),
                    "queue_position": queued_ahead + 1 if queued_ahead or row["state"] == "queued" else 0,
                    "lease_active": bool(
                        row["state"] == "running" and float(row["lease_expires_at"] or 0) > _now()
                    ),
                    "request_hash": str(row["request_hash"]),
                    "credential_values_public": False,
                    "credential_paths_public": False,
                    "private_paths_public": False,
                    "public_artifact_safe": True,
                }
            )
            return stored
        finally:
            if owns:
                current.close()

    def summary(self) -> dict[str, Any]:
        with _connect(self.path) as connection:
            counts = {
                str(row["state"]): int(row["count"])
                for row in connection.execute(
                    "SELECT state,COUNT(*) AS count FROM jobs GROUP BY state"
                ).fetchall()
            }
        return {
            "schema": STORE_SCHEMA,
            "job_counts": counts,
            "max_queue_size": self.max_queue_size,
            "one_live_gpu_job": True,
            "persistent_sqlite": True,
            "credential_values_public": False,
            "credential_paths_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }


Runner = Callable[[dict[str, Any]], dict[str, Any]]


class TrainingBetaSubmitRequest(BaseModel):
    model: str
    topology: str
    steps: int
    job_dir: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    kaggle_token_files: list[str] = Field(default_factory=list)
    kaggle_raw_token_file: str = ""
    kaggle_raw_token_username: str = ""
    allocation_timeout_seconds: float = 1800.0
    execute: bool = False


class TrainingBetaExportRequest(BaseModel):
    output_dir: str = ""


class TrainingBetaResumeRequest(BaseModel):
    kaggle_token_files: list[str] = Field(default_factory=list)
    kaggle_raw_token_file: str = ""
    kaggle_raw_token_username: str = ""
    allocation_timeout_seconds: float = 1800.0
    execute: bool = False


class TrainingBetaController:
    """Shared controller used by the CLI and authenticated HTTP routes."""

    def __init__(self, store: TrainingBetaJobStore, *, runner: Runner | None = None) -> None:
        self.store = store
        self.runner = runner or self._default_runner
        self._run_lock = threading.Lock()
        self._background_threads: dict[str, threading.Thread] = {}

    @staticmethod
    def _default_runner(request: dict[str, Any]) -> dict[str, Any]:
        from .training_qwen15b_job import run_qwen15b_training_job

        return run_qwen15b_training_job(
            request["job_dir"],
            model=MODEL_ID,
            topology="kaggle-2x-t4x2",
            steps=8,
            kaggle_token_files=list(request.get("kaggle_token_files") or []),
            kaggle_raw_token_file=str(request.get("kaggle_raw_token_file") or ""),
            kaggle_raw_token_username=str(request.get("kaggle_raw_token_username") or ""),
            allocation_timeout_seconds=float(request.get("allocation_timeout_seconds") or 1800),
            beta_mode=True,
        )

    def submit(
        self,
        request: dict[str, Any],
        *,
        idempotency_key: str,
        execute: bool = False,
    ) -> dict[str, Any]:
        status, created = self.store.submit(request, idempotency_key=idempotency_key)
        if execute and created:
            return self.execute(status["job_id"])
        return status

    def execute(self, job_id: str) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            return self.store.status(job_id)
        try:
            claimed = self.store.claim_next(
                worker_id=f"controller:{os.getpid()}",
                preferred_job_id=job_id,
            )
            if claimed is None:
                return self.store.status(job_id)
            request = dict(claimed["private_request"])
            try:
                result = self.runner(request)
            except BaseException as exc:
                result = {
                    "schema": STATUS_SCHEMA,
                    "job_id": job_id,
                    "overall_state": "failed",
                    "current_phase": "cleanup",
                    "global_step": int(claimed["public"].get("global_step") or 0),
                    "retry_count": int(claimed["public"].get("retry_count") or 0),
                    "blockers": [f"training_beta_runner_failed:{type(exc).__name__}"],
                    "failure_detail_public": False,
                    "private_paths_public": False,
                    "public_artifact_safe": True,
                }
            result = dict(result)
            result.setdefault("job_id", job_id)
            result.setdefault("global_step", 8 if result.get("overall_state") == "completed" else 0)
            result.setdefault("retry_count", int(claimed["public"].get("retry_count") or 0))
            result.setdefault("current_phase", "cleanup")
            result.setdefault("overall_state", "completed" if result.get("ok") else "blocked")
            return self.store.update_status(
                job_id,
                result,
                event_id=f"runner-finished:{_stable_hash(_public_value(result))}",
            )
        finally:
            self._run_lock.release()

    def start_background(self, job_id: str) -> dict[str, Any]:
        previous = self._background_threads.get(job_id)
        if previous is not None and previous.is_alive():
            return self.store.status(job_id)

        def target() -> None:
            try:
                self.execute(job_id)
            finally:
                self._background_threads.pop(job_id, None)

        thread = threading.Thread(
            target=target,
            name=f"training-beta-{job_id}",
            daemon=True,
        )
        self._background_threads[job_id] = thread
        thread.start()
        return self.store.status(job_id)

    def status(self, job_id: str) -> dict[str, Any]:
        return self.store.status(job_id)

    def resume(
        self,
        job_id: str,
        *,
        execute: bool = False,
        private_inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if private_inputs:
            self.store.update_private_inputs(job_id, private_inputs)
        status = self.store.requeue(job_id)
        return self.execute(job_id) if execute and status["overall_state"] != "completed" else status

    def cancel(self, job_id: str) -> dict[str, Any]:
        status = self.store.request_cancel(job_id)
        if status.get("overall_state") == "running":
            request = self.store.private_request(job_id)
            marker = (
                Path(str(request["job_dir"]))
                / ".private-service"
                / "cancel.requested"
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            if not marker.exists():
                marker.write_text("cancel requested\n", encoding="utf-8")
                marker.chmod(0o600)
        return {**status, "command_ok": True}

    def export(self, job_id: str, output_dir: str | Path | None = None) -> dict[str, Any]:
        from .training_qwen15b_job import export_qwen15b_training_job

        request = self.store.private_request(job_id)
        return export_qwen15b_training_job(request["job_dir"], output_dir)

    def cleanup(self, job_id: str) -> dict[str, Any]:
        from .training_qwen15b_job import cleanup_qwen15b_training_job

        current = self.store.status(job_id)
        existing_cleanup = current.get("cleanup")
        if (
            current.get("overall_state") == "cleaned"
            and isinstance(existing_cleanup, dict)
            and existing_cleanup.get("ok") is True
        ):
            return {**current, "command_ok": True}
        request = self.store.private_request(job_id)
        result = cleanup_qwen15b_training_job(
            request["job_dir"],
            kaggle_token_files=list(request.get("kaggle_token_files") or []),
            kaggle_raw_token_file=str(request.get("kaggle_raw_token_file") or ""),
            kaggle_raw_token_username=str(request.get("kaggle_raw_token_username") or ""),
        )
        current = self.store.status(job_id)
        status = {
            **current,
            "overall_state": "cleaned" if result.get("ok") else "blocked",
            "current_phase": "cleanup",
            "cleanup": result,
        }
        return {
            **self.store.update_status(
                job_id,
                status,
                event_id=f"cleanup-completed:{_stable_hash(result)}",
            ),
            "command_ok": bool(result.get("ok")),
        }


def create_training_beta_app(
    controller: TrainingBetaController,
    *,
    token: str,
) -> Any:
    """Create the authenticated local Beta API without persisting its token."""

    from fastapi import FastAPI, Header, HTTPException

    if not str(token):
        raise ValueError("training_beta_service_token_required")
    app = FastAPI(title="CrowdTensor Qwen Training Beta", docs_url=None, redoc_url=None)

    def authorize(value: str | None) -> None:
        import hmac

        if value is None or not hmac.compare_digest(value, token):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "schema": SERVICE_SCHEMA}

    @app.post("/v1/training/jobs")
    def submit(
        request: TrainingBetaSubmitRequest,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_training_token)
        value = request.model_dump()
        key = str(value.pop("idempotency_key"))
        execute = bool(value.pop("execute"))
        value["model_revision"] = MODEL_REVISION
        status_value = controller.submit(value, idempotency_key=key, execute=False)
        return (
            controller.start_background(status_value["job_id"])
            if execute and status_value["overall_state"] != "completed"
            else status_value
        )

    @app.get("/v1/training/jobs/{job_id}")
    def status(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_training_token)
        try:
            return controller.status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="training_beta_job_not_found") from exc

    @app.get("/v1/training/jobs/{job_id}/events")
    def events(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> list[dict[str, Any]]:
        authorize(x_crowdtensor_training_token)
        return controller.store.events(job_id)

    @app.post("/v1/training/jobs/{job_id}/resume")
    def resume(
        job_id: str,
        request: TrainingBetaResumeRequest | None = None,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_training_token)
        value = request.model_dump() if request is not None else {}
        execute = bool(value.pop("execute", False))
        status_value = controller.resume(
            job_id,
            execute=False,
            private_inputs=value or None,
        )
        return (
            controller.start_background(job_id)
            if execute and status_value["overall_state"] != "completed"
            else status_value
        )

    @app.post("/v1/training/jobs/{job_id}/cancel")
    def cancel(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_training_token)
        return controller.cancel(job_id)

    @app.post("/v1/training/jobs/{job_id}/export")
    def export(
        job_id: str,
        request: TrainingBetaExportRequest,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_training_token)
        return controller.export(job_id, request.output_dir or None)

    @app.post("/v1/training/jobs/{job_id}/cleanup")
    def cleanup(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_training_token)
        return controller.cleanup(job_id)

    @app.get("/v1/training/jobs/{job_id}/artifacts")
    def artifacts(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_training_token)
        status_value = controller.status(job_id)
        return {
            "schema": "crowdtensor_training_qwen15b_beta_artifacts_v1",
            "job_id": job_id,
            "artifacts": status_value.get("artifacts") or {},
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    return app
