"""Ordinary-user controller for heterogeneous CPU/GPU/JAX-TPU Qwen training."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .elastic_checkpoint_storage import checkpoint_blob_store_from_configuration
from .elastic_training_runtime import ElasticTrainingRuntime, install_elastic_training_routes
from .heterogeneous_qwen_source import (
    prepare_manifest_wikitext,
    resolve_qwen_source,
)
from .heterogeneous_training_checkpoint import restore_stage_checkpoint_archive
from .heterogeneous_training_manifest import (
    TPU_MANIFEST_SCHEMA,
    load_training_manifest,
    qwen25_7b_lora_manifest,
    qwen25_7b_lora_tpu_manifest,
    stable_hash,
    validate_training_manifest,
)
from .qwen15b_training import export_qwen_standard_peft_adapter, sha256_file


JOB_SCHEMA = "crowdtensor_heterogeneous_training_beta_private_job_v1"
STATUS_SCHEMA = "crowdtensor_heterogeneous_training_beta_status_v1"
SERVICE_SCHEMA = "crowdtensor_heterogeneous_training_beta_service_v1"
CREDENTIAL_SCHEMA = "crowdtensor_heterogeneous_training_beta_credentials_v1"


def _write_json(path: Path, value: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("heterogeneous_training_beta_private_state_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("heterogeneous_training_beta_private_state_invalid")
    return value


def _copy_private(source: str | Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source).expanduser().resolve(), target)
    target.chmod(0o600)


def _validate_inputs(
    manifest: dict[str, Any],
    config_path: Path,
    tokenized_path: Path,
) -> dict[str, Any]:
    config = _read_json(config_path)
    tokenized = _read_json(tokenized_path)
    model = manifest["model"]
    training = manifest["training"]
    train = tokenized.get("train")
    validation = tokenized.get("validation")
    required_rows = (
        int(training["target_steps"])
        * int(training["microbatches_per_step"])
        * int(training["microbatch_size"])
    )
    if (
        config.get("model_type") != model["model_type"]
        or int(config.get("num_hidden_layers") or 0) != int(model["num_hidden_layers"])
        or int(config.get("hidden_size") or 0) != int(model["hidden_size"])
        or tokenized.get("schema")
        != "crowdtensor_heterogeneous_tokenized_private_v1"
        or tokenized.get("training_manifest_hash") != manifest["content_hash"]
        or tokenized.get("model_id") != model["model_id"]
        or tokenized.get("model_revision") != model["model_revision"]
        or int(tokenized.get("sequence_length") or 0)
        != int(training["sequence_length"])
        or not isinstance(train, list)
        or len(train) < required_rows
        or not isinstance(validation, list)
        or len(validation) < 1
    ):
        raise ValueError("heterogeneous_training_beta_input_contract_invalid")
    if any(
        not isinstance(row, list)
        or len(row) != int(training["sequence_length"])
        or any(not isinstance(token, int) or token < 0 for token in row)
        for row in [*train, *validation]
    ):
        raise ValueError("heterogeneous_training_beta_token_rows_invalid")
    return {
        "config_hash": stable_hash(config),
        "tokenized_payload_hash": stable_hash(tokenized),
        "train_sequence_count": len(train),
        "validation_sequence_count": len(validation),
        "sequence_length": int(training["sequence_length"]),
        "raw_training_text_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


class HeterogeneousTrainingBetaController:
    def __init__(self, job_dir: str | Path) -> None:
        self.job_dir = Path(job_dir).expanduser().resolve()
        self.private_dir = self.job_dir / ".private-heterogeneous"
        self.private_job_path = self.private_dir / "job.json"
        self.credentials_path = self.private_dir / "credentials.json"
        self.public_status_path = self.job_dir / "heterogeneous_training_status.json"
        self.public_cleanup_path = self.job_dir / "heterogeneous_training_cleanup.json"
        self.private = _read_json(self.private_job_path)
        if self.private.get("schema") != JOB_SCHEMA:
            raise ValueError("heterogeneous_training_beta_job_schema_invalid")
        self.job_id = str(self.private["job_id"])
        self.run_id = str(self.private["run_id"])
        self.manifest = load_training_manifest(self.private["manifest_path"])
        self.runtime = ElasticTrainingRuntime.open_existing(
            self.private_dir / "heterogeneous-training.sqlite3",
            run_id=self.run_id,
            lease_seconds=float(self.private.get("lease_seconds") or 30.0),
        )

    @classmethod
    def is_job(cls, job_dir: str | Path) -> bool:
        return (
            Path(job_dir).expanduser().resolve()
            / ".private-heterogeneous"
            / "job.json"
        ).is_file()

    @classmethod
    def create(
        cls,
        job_dir: str | Path,
        *,
        manifest_path: str | Path | None = None,
        config_path: str | Path | None = None,
        tokenized_payload_path: str | Path | None = None,
        hf_token: str = "",
        checkpoint_storage: dict[str, Any] | None = None,
        checkpoint_retention_steps: int = 2,
        lease_seconds: float = 30.0,
        max_online_miners: int = 32,
        enable_jax_tpu: bool = False,
        tensor_lookup_optimization_after_step: int = 0,
    ) -> "HeterogeneousTrainingBetaController":
        output = Path(job_dir).expanduser().resolve()
        if cls.is_job(output):
            return cls(output)
        manifest = (
            load_training_manifest(manifest_path)
            if manifest_path is not None
            else (
                qwen25_7b_lora_tpu_manifest()
                if enable_jax_tpu
                else qwen25_7b_lora_manifest()
            )
        )
        if enable_jax_tpu and manifest["schema"] != TPU_MANIFEST_SCHEMA:
            raise ValueError("heterogeneous_training_tpu_manifest_v2_required")
        output.mkdir(parents=True, exist_ok=True)
        private = output / ".private-heterogeneous"
        private.mkdir(parents=True, exist_ok=True)
        private.chmod(0o700)
        inputs = private / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        inputs.chmod(0o700)
        manifest_destination = inputs / "training_manifest.json"
        config_destination = inputs / "config.json"
        tokenized_destination = inputs / "heterogeneous_tokenized_private.json"
        _write_json(manifest_destination, manifest)
        if (config_path is None) != (tokenized_payload_path is None):
            raise ValueError("heterogeneous_training_beta_both_inputs_required")
        source_report: dict[str, Any] = {}
        dataset_report: dict[str, Any] = {}
        if config_path is not None and tokenized_payload_path is not None:
            _copy_private(config_path, config_destination)
            _copy_private(tokenized_payload_path, tokenized_destination)
        else:
            config, _index, source_report = resolve_qwen_source(
                manifest, token=hf_token
            )
            _write_json(config_destination, config)
            dataset_report = prepare_manifest_wikitext(
                manifest, inputs, token=hf_token
            )
            prepared_path = Path(dataset_report["private_tokenized_path"]).resolve()
            if prepared_path != tokenized_destination.resolve():
                _copy_private(prepared_path, tokenized_destination)
        input_report = _validate_inputs(
            manifest, config_destination, tokenized_destination
        )
        job_id = "heterogeneous-beta-" + secrets.token_hex(8)
        run_id = "qwen-heterogeneous-" + secrets.token_hex(12)
        storage_configuration = dict(checkpoint_storage or {"backend": "local"})
        if storage_configuration.get("backend", "local") == "s3":
            storage_configuration.setdefault(
                "prefix", f"crowdtensor/heterogeneous-training/{job_id}"
            )
        blob_store = checkpoint_blob_store_from_configuration(
            storage_configuration,
            default_root=private / "checkpoint-blobs",
        )
        ElasticTrainingRuntime(
            private / "heterogeneous-training.sqlite3",
            run_id=run_id,
            target_steps=int(manifest["training"]["target_steps"]),
            microbatches_per_step=int(
                manifest["training"]["microbatches_per_step"]
            ),
            lease_seconds=float(lease_seconds),
            blob_store=blob_store,
            checkpoint_retention_steps=int(checkpoint_retention_steps),
            require_checkpoint_signatures=True,
            validate_checkpoint_tensors=True,
            max_online_miners=int(max_online_miners),
            max_rejected_submissions_per_session=3,
            tensor_lookup_optimization_after_step=int(
                tensor_lookup_optimization_after_step
            ),
            training_manifest=manifest,
        )
        now = time.time()
        _write_json(
            private / "job.json",
            {
                "schema": JOB_SCHEMA,
                "job_id": job_id,
                "run_id": run_id,
                "manifest_path": str(manifest_destination),
                "manifest_hash": manifest["content_hash"],
                "model_id": manifest["model"]["model_id"],
                "model_revision": manifest["model"]["model_revision"],
                "target_steps": int(manifest["training"]["target_steps"]),
                "microbatches_per_step": int(
                    manifest["training"]["microbatches_per_step"]
                ),
                "lease_seconds": float(lease_seconds),
                "config_path": str(config_destination),
                "tokenized_payload_path": str(tokenized_destination),
                "input_report": input_report,
                "source_report": source_report,
                "dataset_report": {
                    key: value
                    for key, value in dataset_report.items()
                    if key != "private_tokenized_path"
                },
                "created_at": now,
                "credential_values_public": False,
                "public_artifact": False,
            },
        )
        _write_json(
            private / "credentials.json",
            {
                "schema": CREDENTIAL_SCHEMA,
                "owner_token": secrets.token_urlsafe(32),
                "miner_token": secrets.token_urlsafe(32),
                "created_at": now,
                "public_artifact": False,
            },
        )
        controller = cls(output)
        controller.status()
        return controller

    def credentials(self) -> dict[str, Any]:
        value = _read_json(self.credentials_path)
        if value.get("schema") != CREDENTIAL_SCHEMA:
            raise ValueError("heterogeneous_training_beta_credentials_invalid")
        return value

    def write_miner_invite(
        self, output_file: str | Path, *, coordinator_url: str
    ) -> dict[str, Any]:
        if not str(coordinator_url).strip():
            raise ValueError("heterogeneous_training_beta_coordinator_url_required")
        credentials = self.credentials()
        target = Path(output_file).expanduser().resolve()
        _write_json(
            target,
            {
                "schema": "crowdtensor_elastic_training_beta_private_miner_invite_v1",
                "job_id": self.job_id,
                "coordinator_url": str(coordinator_url).rstrip("/"),
                "miner_token": credentials["miner_token"],
                "training_mode": "heterogeneous",
                "created_at": time.time(),
                "public_artifact": False,
            },
        )
        return {
            "schema": "crowdtensor_heterogeneous_training_invite_write_v1",
            "ok": True,
            "command_ok": True,
            "job_id": self.job_id,
            "invite_file_written": target.is_file(),
            "invite_file_hash": sha256_file(target),
            "next_command": "crowdtensor-miner join --training --invite <invite.json>",
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    def bootstrap(self) -> dict[str, Any]:
        config = _read_json(Path(self.private["config_path"]))
        tokenized = _read_json(Path(self.private["tokenized_payload_path"]))
        _validate_inputs(self.manifest, Path(self.private["config_path"]), Path(self.private["tokenized_payload_path"]))
        return {
            "schema": "crowdtensor_heterogeneous_training_beta_miner_bootstrap_v1",
            "job_id": self.job_id,
            "run_id": self.run_id,
            "model_id": self.manifest["model"]["model_id"],
            "model_revision": self.manifest["model"]["model_revision"],
            "target_steps": int(self.manifest["training"]["target_steps"]),
            "microbatches_per_step": int(
                self.manifest["training"]["microbatches_per_step"]
            ),
            "training_manifest": self.manifest,
            "config": config,
            "tokenized_payload": tokenized,
            "config_hash": stable_hash(config),
            "tokenized_payload_hash": stable_hash(tokenized),
            "checkpoint_signatures_required": True,
            "checkpoint_tensor_validation_required": True,
            "single_gpu_miner_supported": True,
            "cpu_trainable_stages_supported": True,
            "jax_tpu_trainable_stages_supported": self.manifest["schema"]
            == TPU_MANIFEST_SCHEMA,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "public_artifact": False,
        }

    def status(self) -> dict[str, Any]:
        runtime = self.runtime.public_status()
        state = str(runtime["runtime_state"])
        overall = {
            "completed": "completed",
            "cancelled": "cancelled",
            "cleaned": "cleaned",
            "running": "running",
            "paused_waiting_for_miners": "waiting_for_miners",
            "paused_by_owner": "paused",
        }.get(state, "blocked")
        report = {
            "schema": STATUS_SCHEMA,
            "job_id": self.job_id,
            "model_id": self.manifest["model"]["model_id"],
            "model_revision": self.manifest["model"]["model_revision"],
            "parameter_count": int(self.manifest["model"]["parameter_count"]),
            "training_manifest_hash": self.manifest["content_hash"],
            "execution_mode": "elastic_heterogeneous_pipeline",
            "topology": (
                "manifest-driven-cpu-cuda-jax-tpu-stages"
                if self.manifest["schema"] == TPU_MANIFEST_SCHEMA
                else "manifest-driven-cpu-cuda-stages"
            ),
            "overall_state": overall,
            "current_phase": (
                "completed"
                if overall == "completed"
                else "waiting_for_miners"
                if overall == "waiting_for_miners"
                else "paused"
                if overall == "paused"
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
            "placement_generation": int(runtime["placement_generation"]),
            "placement_plan": runtime["placement_plan"],
            "runtime": runtime,
            "ordinary_user_lifecycle_ready": True,
            "single_gpu_miner_ready": True,
            "cpu_trainable_stage_ready": True,
            "jax_tpu_trainable_stage_ready": self.manifest["schema"]
            == TPU_MANIFEST_SCHEMA,
            "required_device_types": list(
                self.manifest["scheduler"]["required_device_types"]
            ),
            "dynamic_placement_ready": True,
            "tensor_transport_ready": True,
            "blockers": [],
            "credential_values_public": False,
            "credential_paths_public": False,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "activation_values_public": False,
            "gradient_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        _write_json(self.public_status_path, report, mode=0o644)
        return report

    def cancel(self) -> dict[str, Any]:
        self.runtime.cancel(reason="owner_cancelled")
        return {**self.status(), "command_ok": True}

    def pause(self) -> dict[str, Any]:
        result = self.runtime.pause()
        return {**self.status(), "pause_transition_applied": result["pause_transition_applied"], "command_ok": True}

    def resume(self) -> dict[str, Any]:
        result = self.runtime.resume()
        return {**self.status(), "resume_transition_applied": result["resume_transition_applied"], "command_ok": True}

    def rebalance(self, *, reason: str = "owner_requested") -> dict[str, Any]:
        result = self.runtime.request_rebalance(reason=reason)
        return {**self.status(), "rebalance_transition_applied": result["rebalance_transition_applied"], "command_ok": True}

    def metrics(self) -> dict[str, Any]:
        return self.runtime.metrics_snapshot()

    def events(self, *, after_sequence: int = 0, limit: int = 200) -> dict[str, Any]:
        return self.runtime.event_tail(after_sequence=after_sequence, limit=limit)

    def export(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        status = self.runtime.public_status()
        if status["runtime_state"] != "completed":
            raise RuntimeError("heterogeneous_training_export_requires_completed_job")
        destination = Path(
            output_dir or self.job_dir / "exported_adapter"
        ).expanduser().resolve()
        with tempfile.TemporaryDirectory(
            prefix="crowdtensor-heterogeneous-export-", dir=self.private_dir
        ) as temporary:
            root = Path(temporary)
            states = []
            archive_hashes = []
            for stage_id in range(len(self.manifest["stages"])):
                archive, archive_report = self.runtime.read_committed_checkpoint(
                    stage_id=stage_id,
                    target_step=int(status["committed_step"]),
                )
                restore_stage_checkpoint_archive(
                    archive,
                    root,
                    training_manifest=self.manifest,
                    expected_stage_id=stage_id,
                    expected_step=int(status["committed_step"]),
                    expected_dataset_cursor=int(status["dataset_cursor"]),
                )
                from safetensors.torch import load_file

                states.append(
                    load_file(str(root / f"stage{stage_id}_adapter.safetensors"))
                )
                archive_hashes.append(archive_report["archive_hash"])
            exported = export_qwen_standard_peft_adapter(
                states,
                destination,
                lora_rank=int(self.manifest["lora"]["rank"]),
                lora_alpha=int(self.manifest["lora"]["alpha"]),
                lora_target_modules=self.manifest["lora"]["target_modules"],
                model_id=self.manifest["model"]["model_id"],
                model_revision=self.manifest["model"]["model_revision"],
            )
        report = {
            "schema": "crowdtensor_heterogeneous_training_beta_export_v1",
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
        _write_json(self.job_dir / "heterogeneous_training_export.json", report, mode=0o644)
        return report

    def cleanup(self) -> dict[str, Any]:
        runtime = self.runtime.cleanup()
        tensor_cleanup = (
            self.runtime.tensor_store.cleanup_all()
            if self.runtime.tensor_store is not None
            else {"all_messages_removed": True, "message_count_removed": 0}
        )
        report = {
            "schema": "crowdtensor_heterogeneous_training_beta_cleanup_v1",
            "ok": bool(
                runtime.get("command_ok") is True
                and tensor_cleanup.get("all_messages_removed") is True
            ),
            "command_ok": True,
            "job_id": self.job_id,
            "overall_state": "cleaned",
            "global_step": int(runtime["committed_step"]),
            "active_miner_leases_revoked": runtime["live_miner_count"] == 0,
            "checkpoint_retention": runtime["checkpoint_retention"],
            "tensor_transport_cleanup": tensor_cleanup,
            "live_resources_left_running": False,
            "credential_values_public": False,
            "checkpoint_tensor_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["command_ok"] = report["ok"]
        report["content_hash"] = stable_hash(report)
        _write_json(self.public_cleanup_path, report, mode=0o644)
        self.status()
        return report


class ExportRequest(BaseModel):
    output_dir: str = ""


class RebalanceRequest(BaseModel):
    reason: str = "owner_requested"


def create_heterogeneous_training_beta_app(
    controller: HeterogeneousTrainingBetaController,
    *,
    owner_token: str,
    miner_token: str,
) -> Any:
    from fastapi import FastAPI, Header, HTTPException

    if not owner_token or not miner_token:
        raise ValueError("heterogeneous_training_service_tokens_required")
    app = FastAPI(title="CrowdTensor Heterogeneous Training Beta", docs_url=None, redoc_url=None)
    coordinator_start = controller.runtime.record_coordinator_start(
        instance_id_hash="sha256:"
        + hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    )
    app.state.coordinator_start = coordinator_start

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
            + hashlib.sha256(controller.job_id.encode()).hexdigest(),
            "heterogeneous_scheduler": True,
            "coordinator_generation": int(
                coordinator_start["coordinator_generation"]
            ),
            "persistent_journal": True,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        status = controller.runtime.public_status()
        ready_state = str(status["runtime_state"]) not in {
            "failed",
            "cleaned",
        }
        return {
            "schema": SERVICE_SCHEMA,
            "ok": ready_state,
            "ready": ready_state,
            "runtime_state": str(status["runtime_state"]),
            "protocol_version": "v1",
            "public_artifact_safe": True,
        }

    @app.get("/metrics")
    def metrics(
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> Any:
        from fastapi import Response

        owner_authorize(x_crowdtensor_training_token)
        return Response(
            content=controller.runtime.prometheus_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/training/jobs/{job_id}")
    def status(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="job_not_found")
        return controller.status()

    @app.post("/v1/training/jobs/{job_id}/cancel")
    def cancel(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="job_not_found")
        return controller.cancel()

    @app.post("/v1/training/jobs/{job_id}/pause")
    def pause(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="job_not_found")
        return controller.pause()

    @app.post("/v1/training/jobs/{job_id}/resume")
    def resume(
        job_id: str,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="job_not_found")
        return controller.resume()

    @app.post("/v1/training/jobs/{job_id}/rebalance")
    def rebalance(
        job_id: str,
        request: RebalanceRequest,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="job_not_found")
        try:
            return controller.rebalance(reason=request.reason)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/training/jobs/{job_id}/events")
    def events(
        job_id: str,
        after_sequence: int = 0,
        limit: int = 200,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="job_not_found")
        try:
            return controller.events(
                after_sequence=after_sequence,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/training/jobs/{job_id}/export")
    def export(
        job_id: str,
        request: ExportRequest,
        x_crowdtensor_training_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner_authorize(x_crowdtensor_training_token)
        if job_id != controller.job_id:
            raise HTTPException(status_code=404, detail="job_not_found")
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
            raise HTTPException(status_code=404, detail="job_not_found")
        return controller.cleanup()

    @app.get("/elastic-training/bootstrap")
    def bootstrap(
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        miner_authorize(x_crowdtensor_miner_token)
        return controller.bootstrap()

    @app.get("/elastic-training/export-bundle")
    def export_bundle(
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> Any:
        import io
        import zipfile

        from fastapi import Response

        miner_authorize(x_crowdtensor_miner_token)
        try:
            exported = controller.export()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        adapter_dir = controller.job_dir / "exported_adapter"
        buffer = io.BytesIO()
        with zipfile.ZipFile(
            buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                archive.write(adapter_dir / name, name)
        payload = buffer.getvalue()
        return Response(
            content=payload,
            media_type="application/zip",
            headers={
                "x-crowdtensor-export-schema": str(exported["schema"]),
                "x-crowdtensor-export-hash": "sha256:"
                + hashlib.sha256(payload).hexdigest(),
                "x-crowdtensor-model-id-hash": hashlib.sha256(
                    controller.manifest["model"]["model_id"].encode("utf-8")
                ).hexdigest(),
            },
        )

    install_elastic_training_routes(
        app, runtime=controller.runtime, authorize=miner_authorize
    )
    app.state.heterogeneous_training_beta_controller = controller
    return app
