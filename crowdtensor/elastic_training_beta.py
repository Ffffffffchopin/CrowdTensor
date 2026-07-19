"""Ordinary-user service path for elastic Qwen volunteer training."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .elastic_checkpoint_storage import checkpoint_blob_store_from_configuration
from .elastic_training_runtime import (
    ElasticTrainingRuntime,
    install_elastic_training_routes,
    restore_qwen_stage_checkpoint_archive,
)
from .qwen15b_training import (
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    export_qwen_standard_peft_adapter,
    fetch_bytes,
    sha256_bytes,
    sha256_file,
    stable_hash,
    _hf_url,
)
from .qwen15b_training_rendezvous import (
    Qwen15BTrainingRendezvous,
    install_qwen15b_training_routes,
)


JOB_SCHEMA = "crowdtensor_elastic_training_beta_job_v1"
STATUS_SCHEMA = "crowdtensor_elastic_training_beta_status_v1"
SERVICE_SCHEMA = "crowdtensor_elastic_training_beta_service_v1"
PRIVATE_SCHEMA = "crowdtensor_elastic_training_beta_private_job_v1"
CREDENTIAL_SCHEMA = "crowdtensor_elastic_training_beta_private_credentials_v1"


def _atomic_json(path: Path, value: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(3)}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("elastic_training_beta_job_file_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("elastic_training_beta_job_file_invalid")
    return value


def _copy_private_file(source: str | Path, destination: Path) -> None:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError("elastic_training_beta_private_input_missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp"
    )
    try:
        shutil.copyfile(source_path, temporary)
        temporary.chmod(0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_inputs(config_path: Path, tokenized_path: Path, *, target_steps: int) -> dict[str, Any]:
    config = _load_json(config_path)
    tokenized = _load_json(tokenized_path)
    train = tokenized.get("train")
    validation = tokenized.get("validation")
    if (
        config.get("model_type") != "qwen2"
        or int(config.get("num_hidden_layers") or 0) != 28
        or tokenized.get("schema") != "crowdtensor_qwen15b_tokenized_private_v1"
        or tokenized.get("model_id") != MODEL_ID
        or tokenized.get("model_revision") != MODEL_REVISION
        or not isinstance(train, list)
        or len(train) < int(target_steps) * 4
        or not isinstance(validation, list)
        or len(validation) < 4
    ):
        raise ValueError("elastic_training_beta_input_contract_invalid")
    return {
        "config_hash": sha256_file(config_path),
        "tokenized_payload_hash": sha256_file(tokenized_path),
        "train_sequence_count": len(train),
        "validation_sequence_count": len(validation),
        "sequence_length": int(tokenized.get("sequence_length") or 0),
        "raw_training_text_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


class ElasticTrainingBetaController:
    """One durable elastic job with owner and Miner product operations."""

    def __init__(self, job_dir: str | Path) -> None:
        self.job_dir = Path(job_dir).expanduser().resolve()
        self.private_dir = self.job_dir / ".private-elastic"
        self.private_job_path = self.private_dir / "job.json"
        self.credentials_path = self.private_dir / "credentials.json"
        self.public_status_path = self.job_dir / "elastic_training_status.json"
        self.public_cleanup_path = self.job_dir / "elastic_training_cleanup.json"
        private = _load_json(self.private_job_path)
        if private.get("schema") != PRIVATE_SCHEMA:
            raise ValueError("elastic_training_beta_private_job_schema_invalid")
        self.private = private
        self.job_id = str(private["job_id"])
        self.run_id = str(private["run_id"])
        self.runtime = ElasticTrainingRuntime.open_existing(
            self.private_dir / "elastic-training.sqlite3",
            run_id=self.run_id,
            lease_seconds=float(private.get("lease_seconds") or 30.0),
        )
        self.rendezvous = Qwen15BTrainingRendezvous(
            run_id=self.run_id,
            state_path=self.private_dir / "qwen-rendezvous-state.json",
        )

    @classmethod
    def create(
        cls,
        job_dir: str | Path,
        *,
        target_steps: int = 8,
        config_path: str | Path | None = None,
        tokenized_payload_path: str | Path | None = None,
        checkpoint_storage: dict[str, Any] | None = None,
        checkpoint_retention_steps: int = 2,
        lease_seconds: float = 30.0,
        max_online_miners: int = 16,
        max_rejected_submissions_per_session: int = 3,
        max_checkpoint_bytes_per_session: int = 0,
    ) -> "ElasticTrainingBetaController":
        output = Path(job_dir).expanduser().resolve()
        private_dir = output / ".private-elastic"
        private_job_path = private_dir / "job.json"
        if private_job_path.is_file():
            controller = cls(output)
            if int(controller.private.get("target_steps") or 0) != int(target_steps):
                raise ValueError("elastic_training_beta_job_contract_conflict")
            return controller
        if int(target_steps) != 8:
            raise ValueError("elastic_training_beta_target_steps_must_be_eight")
        output.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)
        private_dir.chmod(0o700)
        private_inputs = private_dir / "inputs"
        private_inputs.mkdir(parents=True, exist_ok=True)
        private_inputs.chmod(0o700)
        config_destination = private_inputs / "config.json"
        tokenized_destination = private_inputs / "qwen15b_tokenized_private.json"
        if config_path is not None or tokenized_payload_path is not None:
            if config_path is None or tokenized_payload_path is None:
                raise ValueError("elastic_training_beta_both_private_inputs_required")
            _copy_private_file(config_path, config_destination)
            _copy_private_file(tokenized_payload_path, tokenized_destination)
        else:
            from .training_qwen15b_job import _ensure_job_inputs

            prepared = _ensure_job_inputs(output)
            config_bytes = fetch_bytes(
                _hf_url(MODEL_ID, MODEL_REVISION, "config.json")
            )
            expected_config_hash = str(
                ((prepared.get("source") or {}).get("source") or {}).get(
                    "config_hash"
                )
                or ""
            )
            if expected_config_hash and sha256_bytes(config_bytes) != expected_config_hash:
                raise RuntimeError("elastic_training_beta_config_hash_mismatch")
            config_destination.write_bytes(config_bytes)
            config_destination.chmod(0o600)
            _copy_private_file(prepared["private_dataset"], tokenized_destination)
        input_report = _validate_inputs(
            config_destination, tokenized_destination, target_steps=target_steps
        )
        job_id = "elastic-beta-" + secrets.token_hex(8)
        run_id = "qwen15b-elastic-beta-" + secrets.token_hex(12)
        storage_configuration = dict(checkpoint_storage or {"backend": "local"})
        if storage_configuration.get("backend", "local") == "s3":
            storage_configuration.setdefault(
                "prefix", f"crowdtensor/elastic-training/{job_id}"
            )
        blob_store = checkpoint_blob_store_from_configuration(
            storage_configuration,
            default_root=private_dir / "checkpoint-blobs",
        )
        ElasticTrainingRuntime(
            private_dir / "elastic-training.sqlite3",
            run_id=run_id,
            target_steps=target_steps,
            microbatches_per_step=4,
            lease_seconds=float(lease_seconds),
            blob_store=blob_store,
            checkpoint_retention_steps=int(checkpoint_retention_steps),
            require_checkpoint_signatures=True,
            validate_checkpoint_tensors=True,
            max_online_miners=int(max_online_miners),
            max_rejected_submissions_per_session=int(
                max_rejected_submissions_per_session
            ),
            max_checkpoint_bytes_per_session=int(
                max_checkpoint_bytes_per_session
            ),
        )
        Qwen15BTrainingRendezvous(
            run_id=run_id,
            state_path=private_dir / "qwen-rendezvous-state.json",
        )
        now = time.time()
        private_job = {
            "schema": PRIVATE_SCHEMA,
            "job_id": job_id,
            "run_id": run_id,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "target_steps": int(target_steps),
            "microbatches_per_step": 4,
            "lease_seconds": float(lease_seconds),
            "config_path": str(config_destination),
            "tokenized_payload_path": str(tokenized_destination),
            "input_report": input_report,
            "created_at": now,
            "public_artifact": False,
        }
        credentials = {
            "schema": CREDENTIAL_SCHEMA,
            "owner_token": secrets.token_urlsafe(32),
            "miner_token": secrets.token_urlsafe(32),
            "created_at": now,
            "public_artifact": False,
        }
        _atomic_json(private_job_path, private_job)
        _atomic_json(private_dir / "credentials.json", credentials)
        controller = cls(output)
        controller.status()
        return controller

    @classmethod
    def is_job(cls, job_dir: str | Path) -> bool:
        return (
            Path(job_dir).expanduser().resolve()
            / ".private-elastic"
            / "job.json"
        ).is_file()

    def credentials(self) -> dict[str, Any]:
        value = _load_json(self.credentials_path)
        if value.get("schema") != CREDENTIAL_SCHEMA:
            raise ValueError("elastic_training_beta_credentials_invalid")
        return value

    def write_miner_invite(
        self,
        output_file: str | Path,
        *,
        coordinator_url: str,
    ) -> dict[str, Any]:
        target = Path(output_file).expanduser().resolve()
        if not str(coordinator_url).strip():
            raise ValueError("elastic_training_beta_invite_coordinator_url_required")
        credentials = self.credentials()
        invite = {
            "schema": "crowdtensor_elastic_training_beta_private_miner_invite_v1",
            "job_id": self.job_id,
            "coordinator_url": str(coordinator_url).rstrip("/"),
            "miner_token": str(credentials["miner_token"]),
            "created_at": time.time(),
            "public_artifact": False,
        }
        _atomic_json(target, invite)
        return {
            "schema": "crowdtensor_elastic_training_beta_miner_invite_write_v1",
            "ok": True,
            "command_ok": True,
            "job_id": self.job_id,
            "invite_file_written": target.is_file(),
            "invite_file_hash": sha256_file(target),
            "invite_contains_coordinator_url": True,
            "invite_contains_private_miner_token": True,
            "next_command": "crowdtensor-miner join --training --invite <invite.json>",
            "credential_values_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    def bootstrap(self) -> dict[str, Any]:
        config_path = Path(str(self.private["config_path"]))
        tokenized_path = Path(str(self.private["tokenized_payload_path"]))
        input_report = _validate_inputs(
            config_path,
            tokenized_path,
            target_steps=int(self.private["target_steps"]),
        )
        config = _load_json(config_path)
        tokenized = _load_json(tokenized_path)
        return {
            "schema": "crowdtensor_elastic_training_beta_miner_bootstrap_v1",
            "job_id": self.job_id,
            "run_id": self.run_id,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "target_steps": int(self.private["target_steps"]),
            "microbatches_per_step": int(self.private["microbatches_per_step"]),
            "config": config,
            "tokenized_payload": tokenized,
            "config_hash": stable_hash(config),
            "tokenized_payload_hash": stable_hash(tokenized),
            "stage_groups": [[0, 1], [2, 3]],
            "checkpoint_signatures_required": True,
            "checkpoint_tensor_validation_required": True,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "public_artifact": False,
        }

    def status(self) -> dict[str, Any]:
        runtime = self.runtime.public_status()
        rendezvous = self.rendezvous.public_status()
        state = str(runtime["runtime_state"])
        overall = {
            "completed": "completed",
            "cancelled": "cancelled",
            "cleaned": "cleaned",
            "running": "running",
            "paused_waiting_for_miners": "waiting_for_miners",
        }.get(state, "blocked")
        report = {
            "schema": STATUS_SCHEMA,
            "job_schema": JOB_SCHEMA,
            "job_id": self.job_id,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "execution_mode": "elastic_volunteer",
            "topology": "qwen15b-four-stage-dynamic-miners",
            "overall_state": overall,
            "current_phase": (
                "completed"
                if overall == "completed"
                else "waiting_for_miners"
                if overall == "waiting_for_miners"
                else "training"
                if overall == "running"
                else overall
            ),
            "global_step": int(runtime["committed_step"]),
            "committed_step": int(runtime["committed_step"]),
            "target_steps": int(runtime["target_steps"]),
            "progress_fraction": int(runtime["committed_step"])
            / int(runtime["target_steps"]),
            "online_miner_count": int(runtime["live_miner_count"]),
            "missing_stage_ids": list(runtime["missing_stage_ids"]),
            "pause_reason": str(runtime["pause_reason"]),
            "runtime": runtime,
            "rendezvous": rendezvous,
            "ordinary_user_create_status_cancel_export_ready": True,
            "miner_join_training_ready": True,
            "coordinator_restart_recovery_ready": True,
            "automatic_pause_wake_ready": True,
            "checkpoint_signature_verification_ready": True,
            "checkpoint_tensor_validation_ready": True,
            "malicious_miner_rejection_ready": True,
            "checkpoint_storage_backend": str(
                runtime.get("checkpoint_storage", {}).get("backend") or ""
            ),
            "blockers": [],
            "next_commands": {
                "serve": "crowdtensor train serve --elastic-job <job>",
                "status": "crowdtensor train status <job> --watch",
                "cancel": "crowdtensor train cancel <job>",
                "export": "crowdtensor train export <job>",
                "miner_join": (
                    "crowdtensor-miner join --training --coordinator <url> "
                    "--token-env CROWDTENSOR_MINER_TOKEN"
                ),
            },
            "credential_values_public": False,
            "credential_paths_public": False,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        if self.public_cleanup_path.is_file():
            cleanup = _load_json(self.public_cleanup_path)
            if cleanup.get("schema") == "crowdtensor_elastic_training_beta_cleanup_v1":
                report["cleanup"] = cleanup
        report["content_hash"] = stable_hash(report)
        _atomic_json(self.public_status_path, report, mode=0o644)
        return report

    def cancel(self) -> dict[str, Any]:
        self.runtime.cancel(reason="owner_cancelled")
        report = self.status()
        report["command_ok"] = True
        return report

    def cleanup(self) -> dict[str, Any]:
        if self.public_cleanup_path.is_file():
            existing = _load_json(self.public_cleanup_path)
            if (
                existing.get("schema")
                == "crowdtensor_elastic_training_beta_cleanup_v1"
                and existing.get("ok") is True
                and self.runtime.public_status()["runtime_state"] == "cleaned"
            ):
                return existing
        runtime_cleanup = self.runtime.cleanup()
        rendezvous_cleanup = self.rendezvous.cleanup()
        report = {
            "schema": "crowdtensor_elastic_training_beta_cleanup_v1",
            "ok": bool(
                runtime_cleanup.get("command_ok") is True
                and rendezvous_cleanup.get("private_payloads_removed") is True
            ),
            "command_ok": True,
            "job_id": self.job_id,
            "overall_state": "cleaned",
            "global_step": int(runtime_cleanup["committed_step"]),
            "active_miner_leases_revoked": runtime_cleanup["live_miner_count"] == 0,
            "uncommitted_checkpoint_cleanup": runtime_cleanup[
                "uncommitted_blob_cleanup"
            ],
            "checkpoint_retention": runtime_cleanup["checkpoint_retention"],
            "rendezvous_cleanup": rendezvous_cleanup,
            "exported_adapter_preserved": (
                not (self.job_dir / "exported_adapter").exists()
                or all(
                    (self.job_dir / "exported_adapter" / name).is_file()
                    for name in ("adapter_model.safetensors", "adapter_config.json")
                )
            ),
            "committed_checkpoints_retained_for_export": int(
                runtime_cleanup["committed_step"]
            )
            > 0,
            "public_evidence_preserved": True,
            "external_accelerator_resources_created_by_controller": False,
            "live_resources_left_running": False,
            "credential_values_public": False,
            "checkpoint_tensor_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["ok"] = bool(
            report["ok"]
            and report["active_miner_leases_revoked"]
            and report["exported_adapter_preserved"]
        )
        report["command_ok"] = report["ok"]
        report["content_hash"] = stable_hash(report)
        _atomic_json(self.public_cleanup_path, report, mode=0o644)
        self.status()
        return report

    def export(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        status = self.runtime.public_status()
        if status["runtime_state"] != "completed":
            raise RuntimeError("elastic_training_beta_export_requires_completed_job")
        destination = Path(
            output_dir or (self.job_dir / "exported_adapter")
        ).expanduser().resolve()
        with tempfile.TemporaryDirectory(
            prefix="crowdtensor-elastic-export-", dir=self.private_dir
        ) as temporary:
            root = Path(temporary)
            stage_states = []
            archive_hashes = []
            for stage_id in range(4):
                archive, archive_report = self.runtime.read_committed_checkpoint(
                    stage_id=stage_id,
                    target_step=int(status["committed_step"]),
                )
                restore_qwen_stage_checkpoint_archive(
                    archive,
                    root,
                    expected_stage_id=stage_id,
                    expected_step=int(status["committed_step"]),
                    expected_dataset_cursor=int(status["dataset_cursor"]),
                    validate_tensor_payloads=True,
                )
                from safetensors.torch import load_file

                stage_states.append(
                    load_file(str(root / f"stage{stage_id}_adapter.safetensors"))
                )
                archive_hashes.append(str(archive_report["archive_hash"]))
            exported = export_qwen_standard_peft_adapter(stage_states, destination)
        report = {
            "schema": "crowdtensor_elastic_training_beta_export_v1",
            "ok": exported.get("standard_peft_format") is True,
            "job_id": self.job_id,
            "global_step": int(status["committed_step"]),
            "checkpoint_set_hash": stable_hash(archive_hashes),
            **{key: value for key, value in exported.items() if key != "adapter_dir"},
            "private_paths_public": False,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        _atomic_json(self.job_dir / "elastic_training_export.json", report, mode=0o644)
        return report


class ElasticExportRequest(BaseModel):
    output_dir: str = ""


def create_elastic_training_beta_app(
    controller: ElasticTrainingBetaController,
    *,
    owner_token: str,
    miner_token: str,
) -> Any:
    """Create one restartable owner/Miner service for an elastic job."""

    from fastapi import FastAPI, Header, HTTPException

    if not owner_token or not miner_token:
        raise ValueError("elastic_training_beta_service_tokens_required")
    app = FastAPI(
        title="CrowdTensor Elastic Training Beta", docs_url=None, redoc_url=None
    )
    recovered = controller.rendezvous.public_status().get(
        "recovered_from_persistent_state"
    ) is True
    if recovered:
        committed_step = int(controller.runtime.public_status()["committed_step"])
        controller.rendezvous.begin_coordinator_restart(after_step=committed_step)
        controller.rendezvous.complete_coordinator_restart()

    def owner_authorize(value: str | None) -> None:
        if value is None or not hmac.compare_digest(value, owner_token):
            raise HTTPException(status_code=401, detail="unauthorized")

    def miner_authorize(value: str | None) -> None:
        if value is None or not hmac.compare_digest(value, miner_token):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "schema": SERVICE_SCHEMA,
            "ok": True,
            "job_id_hash": "sha256:"
            + hashlib.sha256(controller.job_id.encode("utf-8")).hexdigest(),
            "authenticated_owner_routes": True,
            "authenticated_miner_routes": True,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {
            "schema": SERVICE_SCHEMA,
            "ok": True,
            "service": "crowdtensor-elastic-training-beta",
            "version": "v1",
            "protocol_version": "elastic_training_beta_v1",
            "auth": {"miner_required": True, "owner_required": True},
        }

    @app.get("/v1/training/jobs/{job_id}")
    def status(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="elastic_training_beta_job_not_found")
        return controller.status()

    @app.post("/v1/training/jobs/{job_id}/cancel")
    def cancel(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="elastic_training_beta_job_not_found")
        return controller.cancel()

    @app.post("/v1/training/jobs/{job_id}/export")
    def export(
        job_id: str,
        request: ElasticExportRequest,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="elastic_training_beta_job_not_found")
        try:
            return controller.export(request.output_dir or None)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/training/jobs/{job_id}/cleanup")
    def cleanup(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="elastic_training_beta_job_not_found")
        return controller.cleanup()

    @app.get("/elastic-training/bootstrap")
    def bootstrap(
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        miner_authorize(x_crowdtensor_miner_token)
        return controller.bootstrap()

    install_elastic_training_routes(
        app, runtime=controller.runtime, authorize=miner_authorize
    )
    install_qwen15b_training_routes(
        app, rendezvous=controller.rendezvous, authorize=miner_authorize
    )
    app.state.elastic_training_beta_controller = controller
    return app
