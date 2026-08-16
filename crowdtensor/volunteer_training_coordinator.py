"""Persistent round coordinator for WAN-friendly volunteer LoRA training."""

from __future__ import annotations

import copy
import fcntl
import hmac
import json
import math
import os
import secrets
import shutil
import tarfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .named_tensor_optimizer import (
    apply_diloco_outer_step,
    load_tensors,
    save_tensors,
    tensor_l2_norm,
)
from .training_contract import (
    public_delta_summary,
    sha256_file,
    sha256_json,
    tensor_specs,
    validate_adapter_delta,
)
from .volunteer_training_protocol import (
    CAMPAIGN_SCHEMA,
    CLAIM_SCHEMA,
    INVITE_SCHEMA,
    LEDGER_EVENT_SCHEMA,
    PROTOCOL_VERSION,
    STATUS_SCHEMA,
    WORK_UNIT_SCHEMA,
    VolunteerProtocolError,
    campaign_content_hash,
    hash_cell_id,
    public_error,
    public_safe,
    token_hash,
    validate_campaign_manifest,
    work_unit_content_hash,
    with_public_safety,
)
from .volunteer_training_storage import LocalVolunteerBlobStore
from .volunteer_training_operator import (
    BROWSER_CELL_SCOPES,
    NATIVE_CELL_SCOPES,
    STATE_SCHEMA_V2,
    authorize_cell_credential,
    default_operator_policy,
    issue_cell_credential,
    migrate_operator_state,
    public_policy_status,
    revoke_cell_credential,
)
from .volunteer_browser_probe import (
    BROWSER_PROBE_LEASE_SECONDS,
    BROWSER_PROBE_MAX_ACTIVE,
    BROWSER_PROBE_RESULT_RETENTION_SECONDS,
    BROWSER_PROBE_RESULT_SCHEMA,
    BROWSER_PROBE_ROUNDS,
    BROWSER_PROBE_RUNTIMES,
    BROWSER_PROBE_SCHEMA,
    BROWSER_PROBE_VECTOR_LENGTH,
    browser_probe_digest,
)


PRIVATE_STATE_SCHEMA = "crowdtensor_volunteer_training_coordinator_state_v1"
AGGREGATION_RECORD_SCHEMA = "crowdtensor_volunteer_training_aggregation_v1"
ARTIFACT_REF_SCHEMA = "crowdtensor_volunteer_training_artifact_ref_v1"
DEFAULT_MAX_DELTA_BYTES = 256 * 1024 * 1024
PAIRING_CODE_SCHEMA = "crowdtensor_volunteer_pairing_code_v1"
PAIRING_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
PAIRING_CODE_MIN_TTL_SECONDS = 60
PAIRING_CODE_MAX_TTL_SECONDS = 3600
PAIRING_CODE_RETENTION_SECONDS = 24 * 60 * 60
PAIRING_CODE_MAX_ACTIVE = 10_000
TRUSTED_EXTERNAL_EVALUATION_SCHEMA = (
    "crowdtensor_trusted_external_evaluation_result_v1"
)


def _normalize_pairing_code(value: str) -> str:
    normalized = "".join(character for character in str(value).upper() if character.isalnum())
    if not normalized.startswith("CT") or len(normalized) != 14:
        raise VolunteerProtocolError("volunteer_pairing_code_invalid", status_code=401)
    if any(character not in PAIRING_CODE_ALPHABET for character in normalized[2:]):
        raise VolunteerProtocolError("volunteer_pairing_code_invalid", status_code=401)
    return normalized


def _new_pairing_code() -> str:
    suffix = "".join(secrets.choice(PAIRING_CODE_ALPHABET) for _ in range(12))
    return "CT-" + "-".join(suffix[index : index + 4] for index in range(0, 12, 4))


def _atomic_write_bytes(path: Path, value: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any], *, mode: int = 0o600) -> Path:
    _atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        mode=mode,
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path.name}")
    return value


def _tensor_contract(tensors: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(item["name"]),
            "shape": list(item["shape"]),
            "dtype": str(item["dtype"]),
            "numel": int(item["numel"]),
            "byte_count": int(item["byte_count"]),
        }
        for item in tensor_specs(tensors)
    ]


def _deterministic_zip(source: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with path.open("rb") as source_handle, archive.open(
                info, "w", force_zip64=True
            ) as archive_handle:
                shutil.copyfileobj(source_handle, archive_handle, length=1024 * 1024)
    output.chmod(0o600)
    return output


def _safe_extract_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise VolunteerProtocolError(
                    "volunteer_evaluation_model_bundle_unsafe", status_code=400
                )
        archive.extractall(destination)


class VolunteerTrainingCoordinator:
    """Own the canonical adapter and fence every update to one leased round."""

    def __init__(
        self,
        campaign_dir: str | Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(campaign_dir).resolve()
        self.private = self.root / ".private"
        self.state_path = self.private / "coordinator_state.json"
        self.lock_path = self.private / "coordinator.lock"
        self.invite_path = self.private / "volunteer_invite.json"
        self.campaign_path = self.root / "campaign.json"
        self.ledger_path = self.root / "audit_ledger.jsonl"
        self.status_path = self.root / "status.json"
        self.migration_report_path = self.root / "migration.json"
        self.clock = clock
        if not self.state_path.is_file():
            raise FileNotFoundError("volunteer campaign coordinator state is missing")
        self._migrate_private_state()

    @classmethod
    def create_from_fixture(
        cls,
        campaign_dir: str | Path,
        fixture: dict[str, Any],
        *,
        campaign_id: str = "",
        target_rounds: int = 2,
        minimum_quorum: int = 2,
        lease_seconds: float = 30.0,
        outer_lr: float = 1.0,
        momentum: float = 0.0,
        clip_delta_norm: float = 10.0,
        hard_max_delta_norm: float = 100.0,
        max_loss_increase: float = 0.25,
        max_delta_bytes: int = DEFAULT_MAX_DELTA_BYTES,
        invite_token: str = "",
        clock: Callable[[], float] = time.time,
    ) -> "VolunteerTrainingCoordinator":
        root = Path(campaign_dir).resolve()
        state_path = root / ".private" / "coordinator_state.json"
        if state_path.is_file():
            return cls(root, clock=clock)
        if target_rounds < 1:
            raise ValueError("target_rounds must be positive")
        shards = list((fixture.get("dataset") or {}).get("shards") or [])
        if minimum_quorum < 2 or minimum_quorum > len(shards):
            raise ValueError("minimum_quorum must be between 2 and the shard count")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if max_delta_bytes < 1:
            raise ValueError("max_delta_bytes must be positive")

        private = root / ".private"
        artifacts_root = private / "artifacts"
        versions_root = private / "versions"
        private.mkdir(parents=True, exist_ok=True)
        artifacts_root.mkdir(parents=True, exist_ok=True)
        versions_root.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)

        base_model_source = Path(str(fixture["model"]["base_model_path"]))
        adapter_source = Path(str(fixture["lora"]["adapter_tensor_path"]))
        config_source = Path(str(fixture["lora"]["adapter_config_path"]))
        dataset_source = Path(str(fixture["dataset"]["private_dataset_path"]))
        for required in (base_model_source, adapter_source, config_source, dataset_source):
            if not required.exists():
                raise FileNotFoundError("volunteer campaign fixture artifact is missing")

        heldout_source_value = str(
            fixture["dataset"].get("private_validation_dataset_path") or ""
        )
        heldout_source = Path(heldout_source_value) if heldout_source_value else None
        if heldout_source is not None and not heldout_source.is_file():
            raise FileNotFoundError("volunteer campaign held-out dataset is missing")

        model_bundle = _deterministic_zip(base_model_source, artifacts_root / "base_model.zip")
        version0 = versions_root / "v000000"
        version0.mkdir(parents=True, exist_ok=True)
        adapter0 = version0 / "adapter_model.safetensors"
        config0 = version0 / "adapter_config.json"
        shutil.copyfile(adapter_source, adapter0)
        shutil.copyfile(config_source, config0)
        adapter0.chmod(0o600)
        config0.chmod(0o600)

        rows = [
            json.loads(line)
            for line in dataset_source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dataset_artifacts: dict[str, Path] = {}
        public_shards: list[dict[str, Any]] = []
        for shard in shards:
            shard_index = int(shard["shard_index"])
            indexes = [int(item) for item in shard["sample_indexes"]]
            path = artifacts_root / f"dataset_shard_{shard_index:04d}.jsonl"
            path.write_text(
                "".join(json.dumps(rows[index], sort_keys=True) + "\n" for index in indexes),
                encoding="utf-8",
            )
            path.chmod(0o600)
            dataset_artifacts[str(shard_index)] = path
            public_shards.append(
                {
                    "shard_index": shard_index,
                    "shard_hash": str(shard["shard_hash"]),
                    "sample_count": int(shard["sample_count"]),
                    "token_count": int(shard["token_count"]),
                    "artifact_hash": sha256_file(path),
                    "artifact_byte_count": path.stat().st_size,
                }
            )

        initial_tensors = load_tensors(adapter0)
        contract = _tensor_contract(initial_tensors)
        chosen_id = str(campaign_id or "").strip() or (
            "volunteer-" + sha256_json(
                {
                    "job_id": fixture["job_id"],
                    "model": fixture["model"]["manifest_hash"],
                    "dataset": fixture["dataset"]["manifest_hash"],
                }
            ).split(":", 1)[1][:16]
        )
        local = dict(fixture["local_training"])
        dtype_bytes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "float64": 8,
        }.get(str(fixture["model"].get("dtype") or "").lower(), 4)
        estimated_weight_bytes = int(fixture["model"]["parameter_count"]) * dtype_bytes
        recurring_download_bytes = (
            int(adapter0.stat().st_size)
            + int(config0.stat().st_size)
            + max(int(path.stat().st_size) for path in dataset_artifacts.values())
        )
        first_download_bytes = int(model_bundle.stat().st_size) + recurring_download_bytes
        evaluation_dataset_path = ""
        evaluation_contract: dict[str, Any] | None = None
        if heldout_source is not None:
            evaluation_root = private / "evaluation"
            evaluation_root.mkdir(parents=True, exist_ok=True)
            heldout_path = evaluation_root / "heldout.jsonl"
            shutil.copyfile(heldout_source, heldout_path)
            heldout_path.chmod(0o600)
            heldout_rows = [
                line
                for line in heldout_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not heldout_rows:
                raise ValueError("volunteer campaign held-out dataset is empty")
            evaluation_dataset_path = str(heldout_path)
            evaluation_contract = {
                "schema": "crowdtensor_volunteer_evaluation_contract_v1",
                "metric": "mean_token_cross_entropy",
                "heldout_dataset_hash": sha256_file(heldout_path),
                "heldout_sample_count": len(heldout_rows),
                "baseline_adapter_version": 0,
                "minimum_loss_reduction": 0.0,
                "statistical_significance_claimed": False,
                "raw_examples_public": False,
            }
        manifest: dict[str, Any] = {
            "schema": CAMPAIGN_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "campaign_id": chosen_id,
            "campaign_revision": 1,
            "model_id": fixture["model"]["model_id"],
            "model_revision": int(fixture["model"]["model_version"]),
            "model_manifest_hash": fixture["model"]["manifest_hash"],
            "base_model_hash": fixture["model"]["base_model_hash"],
            "model_parameter_count": int(fixture["model"]["parameter_count"]),
            "model_artifact_hash": sha256_file(model_bundle),
            "dataset_id": fixture["dataset"]["dataset_id"],
            "dataset_revision": int(fixture["dataset"]["dataset_version"]),
            "dataset_snapshot_hash": fixture["dataset"]["manifest_hash"],
            "dataset_shards": public_shards,
            "initial_adapter_hash": sha256_file(adapter0),
            "adapter_tensor_contract_hash": sha256_json(contract),
            "adapter_tensor_count": len(contract),
            "local_training": {
                "local_steps": int(local["local_steps"]),
                "max_local_steps": max(int(local["local_steps"]), 64),
                "learning_rate": float(local["learning_rate"]),
                "batch_size": int(local["batch_size"]),
                "sequence_length": int(local["sequence_length"]),
                "gradient_accumulation": int(local["gradient_accumulation"]),
                "optimizer_contract": str(local["optimizer_contract"]),
            },
            "resource_requirements": {
                "schema": "crowdtensor_volunteer_resource_requirements_v1",
                "supported_devices": ["cpu", "cuda"],
                "first_work_unit_download_bytes": first_download_bytes,
                "recurring_work_unit_download_bytes": recurring_download_bytes,
                "minimum_memory_bytes": max(
                    512 * 1024**2,
                    estimated_weight_bytes + estimated_weight_bytes // 4,
                ),
                "minimum_free_disk_bytes": max(
                    256 * 1024**2, first_download_bytes * 2
                ),
                "local_steps": int(local["local_steps"]),
                "estimate_only": True,
                "memory_estimate_basis": "model_dtype_plus_25_percent_overhead",
            },
            "round_policy": {
                "minimum_quorum": int(minimum_quorum),
                "target_rounds": int(target_rounds),
                "lease_seconds": float(lease_seconds),
                "aggregation_interval": "quorum_round",
                "distinct_cells_required": True,
                "stale_updates_rejected": True,
                "late_updates_rejected": True,
            },
            "outer_optimizer": {
                "optimizer_type": "diloco_momentum" if momentum else "local_sgd_mean",
                "outer_lr": float(outer_lr),
                "momentum": float(momentum),
                "named_tensor_aggregation": True,
            },
            "update_admission": {
                "tensor_name_shape_dtype_validation": True,
                "content_hash_validation": True,
                "finite_values_required": True,
                "clip_delta_norm": float(clip_delta_norm),
                "hard_max_delta_norm": float(hard_max_delta_norm),
                "max_loss_increase": float(max_loss_increase),
                "max_delta_bytes": int(max_delta_bytes),
            },
            "transport": {
                "work_unit_exchange": "authenticated_https_json",
                "delta_exchange": "length_prefixed_metadata_plus_safetensors",
                "artifact_exchange": "content_addressed_authenticated_download",
                "low_frequency_delta_only": True,
                "per_layer_activation_wan_transport": False,
                "external_tls_termination_required": True,
                "content_addressed_object_store": True,
                "resumable_chunk_upload": True,
                "s3_minio_presigned_download_contract": True,
            },
            "security_boundary": {
                "permission_mode": "invite_authenticated_trusted_cells_alpha",
                "permissionless_byzantine_safety": False,
                "sybil_resistance": False,
                "secure_aggregation": False,
                "data_admission_is_campaign_managed": True,
            },
            "physical_internet_multi_machine_verified": False,
            "hosted_logical_multi_node_only": False,
        }
        if evaluation_contract is not None:
            manifest["evaluation_contract"] = evaluation_contract
        campaign_import = fixture.get("campaign_import")
        model_source = fixture["model"].get("source_provenance")
        dataset_source = fixture["dataset"].get("source_provenance")
        if isinstance(campaign_import, dict):
            manifest.update(
                {
                    "model_adapter_id": str(
                        fixture["model"].get("model_adapter_id")
                        or campaign_import.get("model_adapter_id")
                        or ""
                    ),
                    "model_source": copy.deepcopy(model_source or {}),
                    "dataset_source": copy.deepcopy(dataset_source or {}),
                    "campaign_import": copy.deepcopy(campaign_import),
                }
            )
        manifest = with_public_safety(manifest)
        manifest["manifest_hash"] = campaign_content_hash(manifest)
        validate_campaign_manifest(manifest)
        _atomic_write_json(root / "campaign.json", manifest, mode=0o644)

        token = str(invite_token or secrets.token_urlsafe(32))
        invite = {
            "schema": INVITE_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "campaign_id": chosen_id,
            "campaign_manifest_hash": manifest["manifest_hash"],
            "coordinator_url": "",
            "invite_token": token,
            "credential_values_public": False,
            "private_invite": True,
        }
        _atomic_write_json(private / "volunteer_invite.json", invite, mode=0o600)

        state: dict[str, Any] = {
            "schema": PRIVATE_STATE_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "campaign": manifest,
            "invite_token_hash": token_hash(token),
            "adapter_version": 0,
            "outer_step": 0,
            "current_adapter_hash": sha256_file(adapter0),
            "current_adapter_path": str(adapter0),
            "current_config_path": str(config0),
            "current_velocity_path": "",
            "adapter_tensor_contract": contract,
            "artifact_registry": {},
            "blob_store_root": str(private / "object-store"),
            "rounds": [],
            "submissions": {},
            "campaign_complete": False,
            "ledger_sequence": 0,
            "ledger_head_hash": "sha256:" + "0" * 64,
            "accepted_update_count": 0,
            "rejected_update_count": 0,
            "expired_lease_count": 0,
            "reassigned_work_count": 0,
            "coordinator_recovery_count": 0,
            "uploaded_delta_bytes": 0,
            "created_at": float(clock()),
            "evaluation_dataset_path": evaluation_dataset_path,
        }
        migrate_operator_state(state, now=float(clock()))
        cls._register_artifact_in_state(state, "base_model", model_bundle)
        cls._register_artifact_in_state(state, "adapter_v0", adapter0)
        cls._register_artifact_in_state(state, "adapter_config_v0", config0)
        for index, path in dataset_artifacts.items():
            cls._register_artifact_in_state(state, f"dataset_shard_{index}", path)
        cls._start_round_in_state(state, now=float(clock()))
        _atomic_write_json(state_path, state, mode=0o600)
        coordinator = cls(root, clock=clock)
        with coordinator._locked_state() as loaded:
            coordinator._append_event(
                loaded,
                "campaign_created",
                {
                    "adapter_version": 0,
                    "target_rounds": int(target_rounds),
                    "minimum_quorum": int(minimum_quorum),
                },
            )
            coordinator._save_state(loaded)
            coordinator._write_status(loaded)
        return coordinator

    @staticmethod
    def _register_artifact_in_state(
        state: dict[str, Any], kind: str, path: Path
    ) -> dict[str, Any]:
        store = LocalVolunteerBlobStore(state["blob_store_root"])
        blob_ref = store.put_file(path)
        digest = str(blob_ref["blob_hash"])
        artifact_id = sha256_json(
            {"campaign_id": state["campaign"]["campaign_id"], "kind": kind, "hash": digest}
        ).split(":", 1)[1]
        record = {
            "schema": ARTIFACT_REF_SCHEMA,
            "artifact_id": artifact_id,
            "kind": str(kind),
            "sha256": digest,
            "byte_count": int(path.stat().st_size),
            "storage_backend": store.backend,
            "content_addressed": True,
            "blob_hash": digest,
            "local_path": str(store.local_path(digest)),
        }
        state["artifact_registry"][artifact_id] = record
        return record

    @staticmethod
    def _artifact_by_kind(state: dict[str, Any], kind: str) -> dict[str, Any]:
        matches = [
            item
            for item in state["artifact_registry"].values()
            if item.get("kind") == kind
        ]
        if len(matches) != 1:
            raise RuntimeError(f"volunteer artifact registry missing {kind}")
        return dict(matches[0])

    @staticmethod
    def _public_artifact_ref(record: dict[str, Any]) -> dict[str, Any]:
        return with_public_safety(record)

    @staticmethod
    def _start_round_in_state(state: dict[str, Any], *, now: float) -> dict[str, Any]:
        round_index = len(state["rounds"])
        campaign = state["campaign"]
        round_id = f"{campaign['campaign_id']}-round-{round_index:06d}"
        work_units: dict[str, dict[str, Any]] = {}
        for shard in campaign["dataset_shards"]:
            shard_index = int(shard["shard_index"])
            work_id = sha256_json(
                {
                    "campaign": campaign["campaign_id"],
                    "round": round_index,
                    "adapter_version": state["adapter_version"],
                    "shard": shard_index,
                }
            )
            work_units[work_id] = {
                "work_id": work_id,
                "dataset_shard_index": shard_index,
                "dataset_shard_hash": shard["shard_hash"],
                "state": "queued",
                "lease_generation": 0,
                "lease_token": "",
                "cell_id": "",
                "cell_id_hash": "",
                "lease_expires_at": 0.0,
                "accepted_result_id": "",
            }
        round_state = {
            "round_id": round_id,
            "round_index": round_index,
            "state": "active",
            "base_adapter_version": int(state["adapter_version"]),
            "base_adapter_hash": str(state["current_adapter_hash"]),
            "started_at": float(now),
            "completed_at": 0.0,
            "work_units": work_units,
            "accepted_result_ids": [],
            "accepted_cell_hashes": [],
            "aggregation": {},
        }
        state["rounds"].append(round_state)
        return round_state

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield _read_json(self.state_path)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _migrate_private_state(self) -> dict[str, Any]:
        """Upgrade durable private state before serving any request."""

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                state = _read_json(self.state_path)
                state, report = migrate_operator_state(
                    state, now=float(self.clock())
                )
                if report["migrated"]:
                    _atomic_write_json(self.state_path, state, mode=0o600)
                _atomic_write_json(self.migration_report_path, report, mode=0o644)
                return report
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _save_state(self, state: dict[str, Any]) -> None:
        _atomic_write_json(self.state_path, state, mode=0o600)

    def _authenticate(self, state: dict[str, Any], invite_token: str) -> None:
        actual = token_hash(str(invite_token or ""))
        if not hmac.compare_digest(actual, str(state["invite_token_hash"])):
            raise VolunteerProtocolError("volunteer_invite_authentication_failed", status_code=401)

    def _authorize_cell(
        self,
        state: dict[str, Any],
        *,
        token: str,
        cell_id: str,
        scope: str,
        request_nonce: str = "",
        upload_bytes: int = 0,
        submission: bool = False,
    ) -> dict[str, Any]:
        actual = token_hash(str(token or ""))
        if hmac.compare_digest(actual, str(state["invite_token_hash"])):
            state["policy_counters"]["legacy_invite_authentications"] += 1
            return {
                "authentication_mode": "legacy_operator_invite_compatibility",
                "cell_id_hash": hash_cell_id(cell_id),
                "required_scope": scope,
            }
        if "." not in str(token):
            self._authenticate(state, token)
        try:
            return authorize_cell_credential(
                state,
                token=token,
                cell_id=cell_id,
                required_scope=scope,
                nonce=request_nonce,
                now=float(self.clock()),
                upload_bytes=int(upload_bytes),
                submission=submission,
            )
        except VolunteerProtocolError:
            # Authentication policy counters are durable monitoring state even
            # when the protected operation itself is rejected.
            self._save_state(state)
            self._write_status(state)
            raise

    def authenticate_invite(self, invite_token: str) -> dict[str, Any]:
        """Validate a bearer before allocating upload or report resources."""

        with self._locked_state() as state:
            self._authenticate(state, invite_token)
            return with_public_safety(
                {
                    "ok": True,
                    "campaign_id": state["campaign"]["campaign_id"],
                    "invite_authenticated": True,
                }
            )

    def _prune_pairing_codes_in_state(
        self, state: dict[str, Any], *, now: float
    ) -> None:
        records = state.setdefault("pairing_codes", {})
        stale = [
            digest
            for digest, record in records.items()
            if isinstance(record, dict)
            and (
                float(record.get("expires_at") or 0.0)
                <= now - PAIRING_CODE_RETENTION_SECONDS
                or (
                    bool(record.get("redeemed"))
                    and float(record.get("redeemed_at") or 0.0)
                    <= now - PAIRING_CODE_RETENTION_SECONDS
                )
            )
        ]
        for digest in stale:
            records.pop(digest, None)

    def create_pairing_code(
        self,
        *,
        invite_token: str,
        mode: str = "agent",
        ttl_seconds: int = 600,
    ) -> dict[str, Any]:
        """Create one short, one-time enrollment code and persist only its hash."""

        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"agent", "browser"}:
            raise VolunteerProtocolError("volunteer_pairing_mode_invalid", status_code=400)
        ttl = int(ttl_seconds)
        if ttl < PAIRING_CODE_MIN_TTL_SECONDS or ttl > PAIRING_CODE_MAX_TTL_SECONDS:
            raise VolunteerProtocolError("volunteer_pairing_ttl_invalid", status_code=400)
        now = float(self.clock())
        with self._locked_state() as state:
            self._authenticate(state, invite_token)
            self._prune_pairing_codes_in_state(state, now=now)
            active_count = sum(
                isinstance(record, dict)
                and not record.get("redeemed")
                and float(record.get("expires_at") or 0.0) > now
                for record in state["pairing_codes"].values()
            )
            if active_count >= PAIRING_CODE_MAX_ACTIVE:
                raise VolunteerProtocolError(
                    "volunteer_pairing_capacity_exceeded", status_code=429
                )
            pairing_code = _new_pairing_code()
            normalized_code = _normalize_pairing_code(pairing_code)
            digest = token_hash(normalized_code)
            expires_at = now + ttl
            state["pairing_codes"][digest] = {
                "schema": PAIRING_CODE_SCHEMA,
                "code_hash": digest,
                "mode": normalized_mode,
                "created_at": now,
                "expires_at": expires_at,
                "redeemed": False,
                "redeemed_at": 0.0,
                "cell_id_hash": "",
            }
            state["pairing_counters"]["created"] += 1
            self._append_event(
                state,
                "pairing_code_created",
                {
                    "pairing_mode": normalized_mode,
                    "expires_at": expires_at,
                    "one_time": True,
                },
            )
            self._save_state(state)
            self._write_status(state)
            return {
                "schema": PAIRING_CODE_SCHEMA,
                "ok": True,
                "pairing_code": pairing_code,
                "pairing_mode": normalized_mode,
                "created_at": now,
                "expires_at": expires_at,
                "ttl_seconds": ttl,
                "one_time": True,
                "stored_as_hash_only": True,
                "credential_values_public": False,
                "public_artifact_safe": False,
            }

    def redeem_pairing_code(
        self, *, pairing_code: str, cell_id: str, expected_mode: str = ""
    ) -> dict[str, Any]:
        """Consume one code atomically and return one short-lived Cell credential."""

        normalized_code = _normalize_pairing_code(pairing_code)
        digest = token_hash(normalized_code)
        cell_hash = hash_cell_id(cell_id)
        now = float(self.clock())
        with self._locked_state() as state:
            self._prune_pairing_codes_in_state(state, now=now)
            record = state["pairing_codes"].get(digest)
            if not isinstance(record, dict):
                state["pairing_counters"]["rejected"] += 1
                self._save_state(state)
                self._write_status(state)
                raise VolunteerProtocolError("volunteer_pairing_code_invalid", status_code=401)
            if record.get("redeemed"):
                state["pairing_counters"]["rejected"] += 1
                self._save_state(state)
                self._write_status(state)
                raise VolunteerProtocolError("volunteer_pairing_code_consumed", status_code=409)
            if float(record.get("expires_at") or 0.0) <= now:
                state["pairing_counters"]["expired"] += 1
                state["pairing_counters"]["rejected"] += 1
                self._save_state(state)
                self._write_status(state)
                raise VolunteerProtocolError("volunteer_pairing_code_expired", status_code=410)
            mode = str(record.get("mode") or "")
            if expected_mode and str(expected_mode).strip().lower() != mode:
                state["pairing_counters"]["rejected"] += 1
                self._save_state(state)
                self._write_status(state)
                raise VolunteerProtocolError(
                    "volunteer_pairing_mode_mismatch", status_code=400
                )
            scopes = BROWSER_CELL_SCOPES if mode == "browser" else NATIVE_CELL_SCOPES
            credential_limit = 900 if mode == "browser" else 3600
            remaining_ttl = max(
                PAIRING_CODE_MIN_TTL_SECONDS,
                min(credential_limit, int(float(record["expires_at"]) - now)),
            )
            credential_token, public = issue_cell_credential(
                state,
                cell_id=cell_id,
                scopes=sorted(scopes),
                ttl_seconds=remaining_ttl,
                now=now,
            )
            record.update(
                {
                    "redeemed": True,
                    "redeemed_at": now,
                    "cell_id_hash": cell_hash,
                }
            )
            state["pairing_counters"]["redeemed"] += 1
            self._append_event(
                state,
                "cell_credential_issued",
                {
                    "credential_id": public["credential_id"],
                    "cell_id_hash": public["cell_id_hash"],
                    "scopes": public["scopes"],
                    "expires_at": public["expires_at"],
                    "enrollment_mode": "one_time_pairing_code",
                },
            )
            self._append_event(
                state,
                "pairing_code_redeemed",
                {
                    "pairing_mode": mode,
                    "cell_id_hash": cell_hash,
                    "credential_id": public["credential_id"],
                    "one_time": True,
                },
            )
            self._save_state(state)
            self._write_status(state)
            return {
                **public,
                "ok": True,
                "campaign_id": state["campaign"]["campaign_id"],
                "campaign_manifest_hash": state["campaign"]["manifest_hash"],
                "pairing_mode": mode,
                "pairing_code_consumed": True,
                "credential_token": credential_token,
            }

    def issue_cell_credential(
        self,
        *,
        invite_token: str,
        cell_id: str,
        scopes: list[str] | tuple[str, ...] | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        with self._locked_state() as state:
            self._authenticate(state, invite_token)
            try:
                token, public = issue_cell_credential(
                    state,
                    cell_id=cell_id,
                    scopes=scopes,
                    ttl_seconds=ttl_seconds,
                    now=float(self.clock()),
                )
            except VolunteerProtocolError:
                self._save_state(state)
                self._write_status(state)
                raise
            self._append_event(
                state,
                "cell_credential_issued",
                {
                    "credential_id": public["credential_id"],
                    "cell_id_hash": public["cell_id_hash"],
                    "scopes": public["scopes"],
                    "expires_at": public["expires_at"],
                },
            )
            self._save_state(state)
            self._write_status(state)
            # This is a private enrollment response and must never be packed.
            return {**public, "credential_token": token}

    def revoke_cell_credential(
        self, *, invite_token: str, credential_id: str
    ) -> dict[str, Any]:
        with self._locked_state() as state:
            self._authenticate(state, invite_token)
            report = revoke_cell_credential(
                state, credential_id=credential_id, now=float(self.clock())
            )
            self._append_event(
                state,
                "cell_credential_revoked",
                {
                    "credential_id": report["credential_id"],
                    "cell_id_hash": report["cell_id_hash"],
                    "idempotent": report["idempotent"],
                },
            )
            self._save_state(state)
            self._write_status(state)
            return report

    def configure_operator_policy(
        self, *, invite_token: str, updates: dict[str, int]
    ) -> dict[str, Any]:
        allowed = set(default_operator_policy()) - {"schema"}
        normalized: dict[str, int] = {}
        for key, value in updates.items():
            if key not in allowed or int(value) < 1:
                raise VolunteerProtocolError(
                    "volunteer_operator_policy_update_invalid", status_code=400
                )
            normalized[key] = int(value)
        with self._locked_state() as state:
            self._authenticate(state, invite_token)
            state["operator_policy"].update(normalized)
            self._append_event(
                state,
                "operator_policy_updated",
                {"updated_fields": sorted(normalized)},
            )
            self._save_state(state)
            self._write_status(state)
            return public_policy_status(state, now=float(self.clock()))

    def authorize_cell_request(
        self,
        *,
        token: str,
        cell_id: str,
        scope: str,
        request_nonce: str = "",
        upload_bytes: int = 0,
        submission: bool = False,
    ) -> dict[str, Any]:
        """Authorize non-Coordinator routes such as resumable upload chunks."""

        with self._locked_state() as state:
            try:
                result = self._authorize_cell(
                    state,
                    token=token,
                    cell_id=cell_id,
                    scope=scope,
                    request_nonce=request_nonce,
                    upload_bytes=upload_bytes,
                    submission=submission,
                )
            except VolunteerProtocolError:
                self._save_state(state)
                self._write_status(state)
                raise
            self._save_state(state)
            self._write_status(state)
            return with_public_safety({"ok": True, **result})

    def _append_event(
        self,
        state: dict[str, Any],
        event_type: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        sequence = int(state.get("ledger_sequence") or 0) + 1
        event = with_public_safety(
            {
                "schema": LEDGER_EVENT_SCHEMA,
                "sequence": sequence,
                "event_type": str(event_type),
                "campaign_id": state["campaign"]["campaign_id"],
                "previous_event_hash": state.get("ledger_head_hash"),
                "recorded_at": float(self.clock()),
                "details": public_safe(details),
            }
        )
        event["event_hash"] = sha256_json(
            {key: value for key, value in event.items() if key != "event_hash"}
        )
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("ab") as handle:
            handle.write(json.dumps(event, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        state["ledger_sequence"] = sequence
        state["ledger_head_hash"] = event["event_hash"]
        return event

    def _expire_browser_probes_in_state(
        self, state: dict[str, Any], *, now: float
    ) -> int:
        tasks = state.setdefault("browser_probe_tasks", {})
        expired = 0
        for task in tasks.values():
            if (
                isinstance(task, dict)
                and task.get("state") == "leased"
                and float(task.get("lease_expires_at") or 0.0) <= now
            ):
                task.update(
                    {
                        "state": "expired",
                        "lease_token": "",
                        "completed_at": now,
                    }
                )
                expired += 1
                state["browser_probe_counters"]["expired"] += 1
                self._append_event(
                    state,
                    "browser_probe_expired",
                    {
                        "task_id_hash": sha256_json(
                            {"browser_task_id": task.get("task_id")}
                        ),
                        "lease_generation": int(task.get("lease_generation") or 0),
                    },
                )
        stale = [
            task_id
            for task_id, task in tasks.items()
            if isinstance(task, dict)
            and task.get("state") in {"accepted", "rejected", "expired"}
            and float(task.get("completed_at") or 0.0)
            <= now - BROWSER_PROBE_RESULT_RETENTION_SECONDS
        ]
        for task_id in stale:
            tasks.pop(task_id, None)
        return expired

    @staticmethod
    def _browser_probe_payload(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": BROWSER_PROBE_SCHEMA,
            "task_id": task["task_id"],
            "lease_generation": int(task["lease_generation"]),
            "lease_token": task["lease_token"],
            "lease_expires_at": float(task["lease_expires_at"]),
            "seed": int(task["seed"]),
            "vector_length": int(task["vector_length"]),
            "rounds": int(task["rounds"]),
            "algorithm": "xorshift32_indexed_v1",
            "output_encoding": "little_endian_u32_sha256",
            "runtime_priority": ["webgpu", "wasm-cpu", "cpu-js"],
            "model_update": False,
            "browser_training": False,
            "scheduler_calibration": True,
        }

    def claim_browser_probe(
        self,
        *,
        cell_id: str,
        credential_token: str,
        capability: dict[str, Any] | None = None,
        request_nonce: str = "",
    ) -> dict[str, Any]:
        now = float(self.clock())
        cell_hash = hash_cell_id(cell_id)
        with self._locked_state() as state:
            auth = self._authorize_cell(
                state,
                token=credential_token,
                cell_id=cell_id,
                scope="browser:claim",
                request_nonce=request_nonce,
            )
            self._expire_browser_probes_in_state(state, now=now)
            lifecycle = str(state.get("campaign_lifecycle") or "running")
            if lifecycle != "running":
                self._save_state(state)
                self._write_status(state)
                return with_public_safety(
                    {
                        "schema": BROWSER_PROBE_SCHEMA,
                        "ok": True,
                        "state": "campaign_" + lifecycle,
                        "task": None,
                    }
                )
            existing = next(
                (
                    task
                    for task in state["browser_probe_tasks"].values()
                    if isinstance(task, dict)
                    and task.get("state") == "leased"
                    and task.get("cell_id_hash") == cell_hash
                ),
                None,
            )
            if existing is not None:
                self._save_state(state)
                self._write_status(state)
                return {
                    "schema": BROWSER_PROBE_SCHEMA,
                    "ok": True,
                    "state": "leased",
                    "idempotent_replay": True,
                    "task": self._browser_probe_payload(existing),
                }
            active_count = sum(
                isinstance(task, dict) and task.get("state") == "leased"
                for task in state["browser_probe_tasks"].values()
            )
            if active_count >= BROWSER_PROBE_MAX_ACTIVE:
                raise VolunteerProtocolError(
                    "volunteer_browser_probe_capacity_exceeded", status_code=429
                )
            seed = secrets.randbits(32)
            task_id = sha256_json(
                {
                    "campaign_id": state["campaign"]["campaign_id"],
                    "seed": seed,
                    "issued_at": now,
                    "nonce": secrets.token_hex(16),
                }
            )
            task = {
                "schema": BROWSER_PROBE_SCHEMA,
                "task_id": task_id,
                "state": "leased",
                "cell_id_hash": cell_hash,
                "lease_generation": 1,
                "lease_token": secrets.token_urlsafe(32),
                "lease_expires_at": now + BROWSER_PROBE_LEASE_SECONDS,
                "seed": seed,
                "vector_length": BROWSER_PROBE_VECTOR_LENGTH,
                "rounds": BROWSER_PROBE_ROUNDS,
                "created_at": now,
                "completed_at": 0.0,
                "runtime": "",
                "duration_ms": 0,
                "heartbeat_count": 0,
            }
            state["browser_probe_tasks"][task_id] = task
            self._append_event(
                state,
                "browser_probe_leased",
                {
                    "task_id_hash": sha256_json({"browser_task_id": task_id}),
                    "vector_length": BROWSER_PROBE_VECTOR_LENGTH,
                    "rounds": BROWSER_PROBE_ROUNDS,
                    "capability": public_safe(capability or {}),
                    "authentication_mode": auth["authentication_mode"],
                    "model_update": False,
                },
            )
            self._save_state(state)
            self._write_status(state)
            return {
                "schema": BROWSER_PROBE_SCHEMA,
                "ok": True,
                "state": "leased",
                "idempotent_replay": False,
                "task": self._browser_probe_payload(task),
            }

    def heartbeat_browser_probe(
        self,
        *,
        cell_id: str,
        credential_token: str,
        task_id: str,
        lease_generation: int,
        lease_token: str,
        request_nonce: str = "",
    ) -> dict[str, Any]:
        now = float(self.clock())
        cell_hash = hash_cell_id(cell_id)
        with self._locked_state() as state:
            self._authorize_cell(
                state,
                token=credential_token,
                cell_id=cell_id,
                scope="browser:heartbeat",
                request_nonce=request_nonce,
            )
            self._expire_browser_probes_in_state(state, now=now)
            task = state["browser_probe_tasks"].get(str(task_id))
            if not isinstance(task, dict) or task.get("state") != "leased":
                raise VolunteerProtocolError("volunteer_browser_probe_lease_not_active")
            if int(task.get("lease_generation") or 0) != int(lease_generation):
                raise VolunteerProtocolError("volunteer_browser_probe_generation_mismatch")
            if task.get("cell_id_hash") != cell_hash:
                raise VolunteerProtocolError(
                    "volunteer_browser_probe_cell_mismatch", status_code=403
                )
            if not hmac.compare_digest(str(task.get("lease_token")), str(lease_token)):
                raise VolunteerProtocolError(
                    "volunteer_browser_probe_token_mismatch", status_code=403
                )
            task["lease_expires_at"] = now + BROWSER_PROBE_LEASE_SECONDS
            task["heartbeat_count"] = int(task.get("heartbeat_count") or 0) + 1
            state["browser_probe_counters"]["heartbeats"] += 1
            self._save_state(state)
            self._write_status(state)
            return with_public_safety(
                {
                    "schema": BROWSER_PROBE_SCHEMA,
                    "ok": True,
                    "state": "leased",
                    "lease_expires_at": task["lease_expires_at"],
                    "heartbeat_count": task["heartbeat_count"],
                }
            )

    def submit_browser_probe(
        self,
        *,
        cell_id: str,
        credential_token: str,
        task_id: str,
        lease_generation: int,
        lease_token: str,
        output_sha256: str,
        runtime: str,
        duration_ms: int,
        request_nonce: str = "",
    ) -> dict[str, Any]:
        now = float(self.clock())
        cell_hash = hash_cell_id(cell_id)
        normalized_runtime = str(runtime).strip().lower()
        duration = int(duration_ms)
        if normalized_runtime not in BROWSER_PROBE_RUNTIMES:
            raise VolunteerProtocolError(
                "volunteer_browser_probe_runtime_invalid", status_code=400
            )
        if duration < 1 or duration > 10 * 60 * 1000:
            raise VolunteerProtocolError(
                "volunteer_browser_probe_duration_invalid", status_code=400
            )
        with self._locked_state() as state:
            self._authorize_cell(
                state,
                token=credential_token,
                cell_id=cell_id,
                scope="browser:submit",
                request_nonce=request_nonce,
                submission=True,
            )
            self._expire_browser_probes_in_state(state, now=now)
            task = state["browser_probe_tasks"].get(str(task_id))
            if not isinstance(task, dict) or task.get("state") != "leased":
                raise VolunteerProtocolError("volunteer_browser_probe_lease_not_active")
            if int(task.get("lease_generation") or 0) != int(lease_generation):
                raise VolunteerProtocolError("volunteer_browser_probe_generation_mismatch")
            if task.get("cell_id_hash") != cell_hash:
                raise VolunteerProtocolError(
                    "volunteer_browser_probe_cell_mismatch", status_code=403
                )
            if not hmac.compare_digest(str(task.get("lease_token")), str(lease_token)):
                raise VolunteerProtocolError(
                    "volunteer_browser_probe_token_mismatch", status_code=403
                )
            expected = browser_probe_digest(
                seed=int(task["seed"]),
                vector_length=int(task["vector_length"]),
                rounds=int(task["rounds"]),
            )
            supplied = str(output_sha256).strip().lower()
            if not hmac.compare_digest(expected, supplied):
                task.update(
                    {
                        "state": "rejected",
                        "lease_token": "",
                        "completed_at": now,
                        "runtime": normalized_runtime,
                        "duration_ms": duration,
                    }
                )
                state["browser_probe_counters"]["rejected"] += 1
                self._append_event(
                    state,
                    "browser_probe_rejected",
                    {
                        "task_id_hash": sha256_json({"browser_task_id": task_id}),
                        "runtime": normalized_runtime,
                        "reason": "output_hash_mismatch",
                    },
                )
                self._save_state(state)
                self._write_status(state)
                raise VolunteerProtocolError(
                    "volunteer_browser_probe_output_invalid", status_code=400
                )
            task.update(
                {
                    "state": "accepted",
                    "lease_token": "",
                    "completed_at": now,
                    "runtime": normalized_runtime,
                    "duration_ms": duration,
                    "output_sha256": expected,
                }
            )
            counters = state["browser_probe_counters"]
            counters["accepted"] += 1
            counters[normalized_runtime.replace("-", "_")] += 1
            counters["total_vector_elements"] += int(task["vector_length"])
            counters["total_duration_ms"] += duration
            self._append_event(
                state,
                "browser_probe_accepted",
                {
                    "task_id_hash": sha256_json({"browser_task_id": task_id}),
                    "runtime": normalized_runtime,
                    "duration_ms": duration,
                    "vector_length": int(task["vector_length"]),
                    "heartbeat_count": int(task.get("heartbeat_count") or 0),
                    "server_recomputed": True,
                    "model_update": False,
                },
            )
            self._save_state(state)
            self._write_status(state)
            return with_public_safety(
                {
                    "schema": BROWSER_PROBE_RESULT_SCHEMA,
                    "ok": True,
                    "accepted": True,
                    "state": "accepted",
                    "runtime": normalized_runtime,
                    "duration_ms": duration,
                    "vector_length": int(task["vector_length"]),
                    "heartbeat_count": int(task.get("heartbeat_count") or 0),
                    "server_recomputed": True,
                    "scheduler_calibration": True,
                    "model_update": False,
                    "browser_training": False,
                }
            )

    def _current_round(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if state.get("campaign_complete"):
            return None
        rounds = list(state.get("rounds") or [])
        return rounds[-1] if rounds and rounds[-1].get("state") == "active" else None

    def _expire_leases_in_state(self, state: dict[str, Any], *, now: float) -> int:
        round_state = self._current_round(state)
        if round_state is None:
            return 0
        expired = 0
        for work in round_state["work_units"].values():
            if work.get("state") != "leased":
                continue
            if float(work.get("lease_expires_at") or 0.0) > now:
                continue
            expired += 1
            previous_cell_hash = work.get("cell_id_hash")
            work.update(
                {
                    "state": "queued",
                    "lease_token": "",
                    "cell_id": "",
                    "cell_id_hash": "",
                    "lease_expires_at": 0.0,
                }
            )
            state["expired_lease_count"] = int(state.get("expired_lease_count") or 0) + 1
            self._append_event(
                state,
                "lease_expired",
                {
                    "round_id": round_state["round_id"],
                    "work_id": work["work_id"],
                    "previous_cell_id_hash": previous_cell_hash,
                    "lease_generation": int(work["lease_generation"]),
                },
            )
        return expired

    def expire_leases(self, *, invite_token: str) -> dict[str, Any]:
        with self._locked_state() as state:
            self._authenticate(state, invite_token)
            count = self._expire_leases_in_state(state, now=float(self.clock()))
            self._save_state(state)
            status = self._write_status(state)
            return with_public_safety(
                {"ok": True, "expired_lease_count": count, "status": status}
            )

    def _work_payload(
        self,
        state: dict[str, Any],
        round_state: dict[str, Any],
        work: dict[str, Any],
    ) -> dict[str, Any]:
        campaign = state["campaign"]
        version = int(round_state["base_adapter_version"])
        model_ref = self._public_artifact_ref(self._artifact_by_kind(state, "base_model"))
        adapter_ref = self._public_artifact_ref(
            self._artifact_by_kind(state, f"adapter_v{version}")
        )
        config_ref = self._public_artifact_ref(
            self._artifact_by_kind(state, f"adapter_config_v{version}")
        )
        dataset_ref = self._public_artifact_ref(
            self._artifact_by_kind(state, f"dataset_shard_{work['dataset_shard_index']}")
        )
        local = campaign["local_training"]
        payload: dict[str, Any] = {
            "schema": WORK_UNIT_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "campaign_id": campaign["campaign_id"],
            "campaign_manifest_hash": campaign["manifest_hash"],
            "round_id": round_state["round_id"],
            "round_index": int(round_state["round_index"]),
            "work_id": work["work_id"],
            "adapter_version": version,
            "base_adapter_hash": round_state["base_adapter_hash"],
            "model_manifest_hash": campaign["model_manifest_hash"],
            "base_model_hash": campaign["base_model_hash"],
            "model_revision": int(campaign["model_revision"]),
            "dataset_snapshot_hash": campaign["dataset_snapshot_hash"],
            "dataset_shard_index": int(work["dataset_shard_index"]),
            "dataset_shard_hash": work["dataset_shard_hash"],
            "dataset_sample_count": next(
                int(item["sample_count"])
                for item in campaign["dataset_shards"]
                if int(item["shard_index"]) == int(work["dataset_shard_index"])
            ),
            "local_steps": int(local["local_steps"]),
            "step_start": 0,
            "step_end": int(local["local_steps"]),
            "learning_rate": float(local["learning_rate"]),
            "batch_size": int(local["batch_size"]),
            "sequence_length": int(local["sequence_length"]),
            "gradient_accumulation": int(local["gradient_accumulation"]),
            "optimizer_contract": local["optimizer_contract"],
            "lease_id": sha256_json(
                {"work_id": work["work_id"], "generation": work["lease_generation"]}
            ),
            "lease_generation": int(work["lease_generation"]),
            "lease_expires_at": float(work["lease_expires_at"]),
            "idempotency_key": sha256_json(
                {
                    "campaign_id": campaign["campaign_id"],
                    "round_id": round_state["round_id"],
                    "work_id": work["work_id"],
                    "lease_generation": int(work["lease_generation"]),
                    "cell_id_hash": work["cell_id_hash"],
                }
            ),
            "artifact_refs": {
                "base_model": model_ref,
                "base_adapter": adapter_ref,
                "adapter_config": config_ref,
                "dataset_shard": dataset_ref,
            },
            "low_frequency_delta_only": True,
            "lease_token": work["lease_token"],
        }
        payload.update(
            {
                "credential_values_public": False,
                "private_paths_public": False,
                "raw_data_public": False,
                "tensor_values_public": False,
                "public_artifact_safe": True,
            }
        )
        payload["work_unit_hash"] = work_unit_content_hash(payload)
        return payload

    def claim(
        self,
        *,
        cell_id: str,
        invite_token: str,
        capability: dict[str, Any] | None = None,
        request_nonce: str = "",
    ) -> dict[str, Any]:
        cell_hash = hash_cell_id(cell_id)
        now = float(self.clock())
        with self._locked_state() as state:
            auth = self._authorize_cell(
                state,
                token=invite_token,
                cell_id=cell_id,
                scope="work:claim",
                request_nonce=request_nonce,
            )
            self._expire_leases_in_state(state, now=now)
            lifecycle = str(state.get("campaign_lifecycle") or "running")
            if lifecycle in {"paused", "finalized"}:
                self._save_state(state)
                status = self._write_status(state)
                return with_public_safety(
                    {
                        "schema": CLAIM_SCHEMA,
                        "ok": True,
                        "state": "campaign_" + lifecycle,
                        "work_unit": None,
                        "status": status,
                    }
                )
            round_state = self._current_round(state)
            if round_state is None:
                self._save_state(state)
                status = self._write_status(state)
                return with_public_safety(
                    {
                        "schema": CLAIM_SCHEMA,
                        "ok": True,
                        "state": "campaign_complete",
                        "work_unit": None,
                        "status": status,
                    }
                )

            for work in round_state["work_units"].values():
                if work.get("state") == "leased" and work.get("cell_id_hash") == cell_hash:
                    payload = self._work_payload(state, round_state, work)
                    self._save_state(state)
                    self._write_status(state)
                    return {
                        "schema": CLAIM_SCHEMA,
                        "ok": True,
                        "state": "leased",
                        "idempotent_replay": True,
                        "work_unit": payload,
                    }

            if cell_hash in set(round_state.get("accepted_cell_hashes") or []):
                self._save_state(state)
                self._write_status(state)
                return with_public_safety(
                    {
                        "schema": CLAIM_SCHEMA,
                        "ok": True,
                        "state": "waiting_next_round",
                        "work_unit": None,
                    }
                )

            queued = [
                item
                for item in round_state["work_units"].values()
                if item.get("state") == "queued"
            ]
            if not queued:
                self._save_state(state)
                self._write_status(state)
                return with_public_safety(
                    {
                        "schema": CLAIM_SCHEMA,
                        "ok": True,
                        "state": "waiting_for_work",
                        "work_unit": None,
                    }
                )
            active_for_cell = sum(
                item.get("state") == "leased" and item.get("cell_id_hash") == cell_hash
                for item in round_state["work_units"].values()
            )
            if active_for_cell >= int(
                state["operator_policy"]["maximum_active_leases_per_cell"]
            ):
                state["policy_counters"]["quota_rejections"] += 1
                self._save_state(state)
                self._write_status(state)
                raise VolunteerProtocolError(
                    "volunteer_cell_lease_concurrency_exceeded", status_code=429
                )
            work = sorted(queued, key=lambda item: (item["dataset_shard_index"], item["work_id"]))[0]
            previous_generation = int(work.get("lease_generation") or 0)
            generation = previous_generation + 1
            lease_token = secrets.token_urlsafe(32)
            work.update(
                {
                    "state": "leased",
                    "lease_generation": generation,
                    "lease_token": lease_token,
                    "cell_id": str(cell_id),
                    "cell_id_hash": cell_hash,
                    "lease_expires_at": now + float(state["campaign"]["round_policy"]["lease_seconds"]),
                }
            )
            if previous_generation:
                state["reassigned_work_count"] = int(state.get("reassigned_work_count") or 0) + 1
            self._append_event(
                state,
                "work_leased",
                {
                    "round_id": round_state["round_id"],
                    "work_id": work["work_id"],
                    "dataset_shard_index": int(work["dataset_shard_index"]),
                    "cell_id_hash": cell_hash,
                    "lease_generation": generation,
                    "reassigned": previous_generation > 0,
                    "capability": public_safe(capability or {}),
                    "authentication_mode": auth["authentication_mode"],
                },
            )
            payload = self._work_payload(state, round_state, work)
            self._save_state(state)
            self._write_status(state)
            return {
                "schema": CLAIM_SCHEMA,
                "ok": True,
                "state": "leased",
                "idempotent_replay": False,
                "work_unit": payload,
            }

    def heartbeat(
        self,
        *,
        cell_id: str,
        invite_token: str,
        work_id: str,
        lease_generation: int,
        lease_token: str,
        request_nonce: str = "",
    ) -> dict[str, Any]:
        cell_hash = hash_cell_id(cell_id)
        now = float(self.clock())
        with self._locked_state() as state:
            self._authorize_cell(
                state,
                token=invite_token,
                cell_id=cell_id,
                scope="work:heartbeat",
                request_nonce=request_nonce,
            )
            self._expire_leases_in_state(state, now=now)
            round_state = self._current_round(state)
            work = (round_state or {}).get("work_units", {}).get(str(work_id))
            if not isinstance(work, dict) or work.get("state") != "leased":
                raise VolunteerProtocolError("volunteer_lease_not_active")
            if int(work["lease_generation"]) != int(lease_generation):
                raise VolunteerProtocolError("volunteer_lease_generation_mismatch")
            if work.get("cell_id_hash") != cell_hash:
                raise VolunteerProtocolError("volunteer_lease_cell_mismatch", status_code=403)
            if not hmac.compare_digest(str(work["lease_token"]), str(lease_token)):
                raise VolunteerProtocolError("volunteer_lease_token_mismatch", status_code=403)
            work["lease_expires_at"] = now + float(state["campaign"]["round_policy"]["lease_seconds"])
            self._append_event(
                state,
                "lease_renewed",
                {
                    "round_id": round_state["round_id"],
                    "work_id": work["work_id"],
                    "cell_id_hash": cell_hash,
                    "lease_generation": int(work["lease_generation"]),
                },
            )
            payload = self._work_payload(state, round_state, work)
            self._save_state(state)
            self._write_status(state)
            return with_public_safety(
                {
                    "ok": True,
                    "work_unit_hash": payload["work_unit_hash"],
                    "lease_expires_at": payload["lease_expires_at"],
                }
            )

    def _reject_submission(
        self,
        state: dict[str, Any],
        *,
        code: str,
        round_id: str = "",
        work_id: str = "",
        cell_id_hash: str = "",
    ) -> None:
        state["rejected_update_count"] = int(state.get("rejected_update_count") or 0) + 1
        self._append_event(
            state,
            "update_rejected",
            {
                "code": code,
                "round_id": round_id,
                "work_id": work_id,
                "cell_id_hash": cell_id_hash,
            },
        )
        self._save_state(state)
        self._write_status(state)
        raise VolunteerProtocolError(code)

    def submit(
        self,
        *,
        cell_id: str,
        invite_token: str,
        work_id: str,
        lease_generation: int,
        lease_token: str,
        delta_manifest: dict[str, Any],
        request_nonce: str = "",
    ) -> dict[str, Any]:
        cell_hash = hash_cell_id(cell_id)
        now = float(self.clock())
        result_id = str((delta_manifest or {}).get("result_id") or "")
        with self._locked_state() as state:
            self._authorize_cell(
                state,
                token=invite_token,
                cell_id=cell_id,
                scope="work:submit",
                request_nonce=request_nonce,
                submission=True,
            )
            existing = state.get("submissions", {}).get(result_id)
            if isinstance(existing, dict):
                if (
                    existing.get("delta_file_hash") == delta_manifest.get("delta_file_hash")
                    and existing.get("cell_id_hash") == cell_hash
                    and existing.get("work_id") == str(work_id)
                ):
                    existing_round = next(
                        (
                            item
                            for item in state.get("rounds") or []
                            if item.get("round_id") == existing.get("round_id")
                        ),
                        {},
                    )
                    aggregation = existing_round.get("aggregation") or {}
                    input_result_ids = list(
                        aggregation.get("input_result_ids") or []
                    )
                    derived_aggregated = bool(
                        input_result_ids and input_result_ids[-1] == result_id
                    )
                    round_aggregated = bool(
                        existing.get("round_aggregated", derived_aggregated)
                    )
                    derived_version = (
                        int(aggregation.get("adapter_version_after") or 0)
                        if round_aggregated
                        else int(existing_round.get("base_adapter_version") or 0)
                    )
                    return with_public_safety(
                        {
                            "schema": "crowdtensor_volunteer_training_submission_response_v1",
                            "ok": True,
                            "accepted": True,
                            "idempotent_replay": True,
                            "result_id": result_id,
                            "round_id": existing["round_id"],
                            "round_aggregated": round_aggregated,
                            "adapter_version_after": int(
                                existing.get(
                                    "adapter_version_after", derived_version
                                )
                            ),
                            "accepted_at": float(existing["accepted_at"]),
                            "delta_clipped": bool(existing["delta_clipped"]),
                            "delta_norm_before_clip": float(
                                existing["delta_norm_before_clip"]
                            ),
                            "delta_norm_after_clip": float(
                                existing["delta_norm_after_clip"]
                            ),
                            "campaign_complete": bool(state["campaign_complete"]),
                        }
                    )
                self._reject_submission(
                    state,
                    code="volunteer_result_id_collision",
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )

            self._expire_leases_in_state(state, now=now)
            round_state = self._current_round(state)
            manifest_round = str((delta_manifest or {}).get("round_id") or "")
            try:
                submitted_version = int(delta_manifest.get("adapter_version", -1))
            except (TypeError, ValueError):
                submitted_version = -1
            if submitted_version < int(state["adapter_version"]):
                self._reject_submission(
                    state,
                    code="volunteer_stale_adapter_version_rejected",
                    round_id=manifest_round,
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            if round_state is None:
                self._reject_submission(
                    state,
                    code="volunteer_round_closed_stale_update",
                    round_id=manifest_round,
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            if manifest_round != round_state["round_id"]:
                self._reject_submission(
                    state,
                    code="volunteer_round_fork_rejected",
                    round_id=manifest_round,
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            work = round_state["work_units"].get(str(work_id))
            if not isinstance(work, dict) or work.get("state") != "leased":
                self._reject_submission(
                    state,
                    code="volunteer_lease_not_active",
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            if int(work["lease_generation"]) != int(lease_generation):
                self._reject_submission(
                    state,
                    code="volunteer_lease_generation_mismatch",
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            if work.get("cell_id_hash") != cell_hash:
                self._reject_submission(
                    state,
                    code="volunteer_lease_cell_mismatch",
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            if not hmac.compare_digest(str(work["lease_token"]), str(lease_token)):
                self._reject_submission(
                    state,
                    code="volunteer_lease_token_mismatch",
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            if float(work["lease_expires_at"]) <= now:
                self._reject_submission(
                    state,
                    code="volunteer_lease_expired",
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            if str(delta_manifest.get("miner_id") or "") != str(cell_id):
                self._reject_submission(
                    state,
                    code="volunteer_delta_cell_identity_mismatch",
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            if cell_hash in set(round_state.get("accepted_cell_hashes") or []):
                self._reject_submission(
                    state,
                    code="volunteer_distinct_cell_quorum_violation",
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )

            campaign = state["campaign"]
            expected = {
                "job_id": campaign["campaign_id"],
                "round_id": round_state["round_id"],
                "model_manifest_hash": campaign["model_manifest_hash"],
                "base_model_hash": campaign["base_model_hash"],
                "base_adapter_hash": round_state["base_adapter_hash"],
                "base_model_version": int(campaign["model_revision"]),
                "adapter_version": int(round_state["base_adapter_version"]),
                "dataset_shard_index": int(work["dataset_shard_index"]),
                "dataset_shard_hash": work["dataset_shard_hash"],
                "tensor_specs": state["adapter_tensor_contract"],
            }
            admission = campaign["update_admission"]
            validation = validate_adapter_delta(
                delta_manifest,
                expected=expected,
                seen_result_ids=state["submissions"].keys(),
                max_delta_norm=float(admission["hard_max_delta_norm"]),
                max_loss_increase=float(admission["max_loss_increase"]),
            )
            if validation.get("accepted") is not True:
                self._reject_submission(
                    state,
                    code=str(validation.get("code") or "volunteer_delta_validation_failed"),
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )

            source_path = Path(str(delta_manifest.get("delta_path") or ""))
            if source_path.stat().st_size > int(admission["max_delta_bytes"]):
                self._reject_submission(
                    state,
                    code="volunteer_submission_delta_too_large",
                    round_id=round_state["round_id"],
                    work_id=str(work_id),
                    cell_id_hash=cell_hash,
                )
            accepted_dir = self.private / "accepted" / round_state["round_id"]
            accepted_dir.mkdir(parents=True, exist_ok=True)
            accepted_path = accepted_dir / (result_id.replace("sha256:", "") + ".safetensors")
            tensors = load_tensors(source_path)
            norm_before = tensor_l2_norm(tensors)
            clip_norm = float(admission["clip_delta_norm"])
            clipped = norm_before > clip_norm
            if clipped:
                factor = clip_norm / max(norm_before, 1e-12)
                tensors = {name: value.float().mul(factor).to(value.dtype) for name, value in tensors.items()}
                save_tensors(tensors, accepted_path)
            else:
                shutil.copyfile(source_path, accepted_path)
            accepted_path.chmod(0o600)
            norm_after = tensor_l2_norm(load_tensors(accepted_path))
            accepted_hash = sha256_file(accepted_path)
            record = {
                "result_id": result_id,
                "round_id": round_state["round_id"],
                "work_id": str(work_id),
                "cell_id_hash": cell_hash,
                "delta_file_hash": str(delta_manifest["delta_file_hash"]),
                "accepted_delta_hash": accepted_hash,
                "accepted_delta_path": str(accepted_path),
                "delta_byte_count": int(source_path.stat().st_size),
                "delta_norm_before_clip": norm_before,
                "delta_norm_after_clip": norm_after,
                "delta_clipped": clipped,
                "loss_start": float(validation["loss_start"]),
                "loss_end": float(validation["loss_end"]),
                "samples_seen": int(delta_manifest.get("samples_seen") or 0),
                "tokens_seen": int(delta_manifest.get("tokens_seen") or 0),
                "public_delta_summary": public_delta_summary(delta_manifest),
                "accepted_at": now,
            }
            state["submissions"][result_id] = record
            state["accepted_update_count"] = int(state.get("accepted_update_count") or 0) + 1
            state["uploaded_delta_bytes"] = int(state.get("uploaded_delta_bytes") or 0) + int(
                source_path.stat().st_size
            )
            work["state"] = "accepted"
            work["accepted_result_id"] = result_id
            work["lease_token"] = ""
            round_state["accepted_result_ids"].append(result_id)
            round_state["accepted_cell_hashes"].append(cell_hash)
            self._append_event(
                state,
                "update_accepted",
                {
                    "round_id": round_state["round_id"],
                    "work_id": str(work_id),
                    "cell_id_hash": cell_hash,
                    "result_id": result_id,
                    "delta_file_hash": delta_manifest["delta_file_hash"],
                    "delta_byte_count": int(source_path.stat().st_size),
                    "delta_norm_before_clip": norm_before,
                    "delta_norm_after_clip": norm_after,
                    "delta_clipped": clipped,
                    "loss_start": float(validation["loss_start"]),
                    "loss_end": float(validation["loss_end"]),
                },
            )
            aggregated = False
            if len(round_state["accepted_result_ids"]) >= int(
                campaign["round_policy"]["minimum_quorum"]
            ):
                self._aggregate_round_in_state(state, round_state, now=now)
                aggregated = True
            record["round_aggregated"] = aggregated
            record["adapter_version_after"] = int(state["adapter_version"])
            self._save_state(state)
            self._write_status(state)
            return with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_training_submission_response_v1",
                    "ok": True,
                    "accepted": True,
                    "idempotent_replay": False,
                    "result_id": result_id,
                    "round_id": record["round_id"],
                    "validation": validation,
                    "delta_clipped": clipped,
                    "delta_norm_before_clip": norm_before,
                    "delta_norm_after_clip": norm_after,
                    "round_aggregated": aggregated,
                    "adapter_version_after": int(state["adapter_version"]),
                    "accepted_at": float(record["accepted_at"]),
                    "campaign_complete": bool(state["campaign_complete"]),
                }
            )

    def _aggregate_round_in_state(
        self,
        state: dict[str, Any],
        round_state: dict[str, Any],
        *,
        now: float,
    ) -> None:
        if round_state.get("state") != "active":
            raise RuntimeError("volunteer round is not active")
        campaign = state["campaign"]
        quorum = int(campaign["round_policy"]["minimum_quorum"])
        result_ids = list(round_state["accepted_result_ids"][:quorum])
        records = [state["submissions"][result_id] for result_id in result_ids]
        if len({item["cell_id_hash"] for item in records}) != quorum:
            raise RuntimeError("volunteer aggregation quorum is not cell-distinct")
        version_before = int(state["adapter_version"])
        version_after = version_before + 1
        pending = self.private / "pending" / f"v{version_after:06d}-{secrets.token_hex(4)}"
        pending.mkdir(parents=True, exist_ok=False)
        output_adapter = pending / "adapter_model.safetensors"
        output_config = pending / "adapter_config.json"
        output_velocity = pending / "outer_velocity.safetensors"
        current_velocity = str(state.get("current_velocity_path") or "")
        if current_velocity and Path(current_velocity).is_file():
            shutil.copyfile(current_velocity, output_velocity)
        outer = campaign["outer_optimizer"]
        aggregation = apply_diloco_outer_step(
            base_adapter_path=state["current_adapter_path"],
            delta_paths=[item["accepted_delta_path"] for item in records],
            output_adapter_path=output_adapter,
            velocity_path=output_velocity,
            outer_step=int(state["outer_step"]),
            adapter_version=version_before,
            outer_lr=float(outer["outer_lr"]),
            momentum=float(outer["momentum"]),
        )
        shutil.copyfile(state["current_config_path"], output_config)
        final = self.private / "versions" / f"v{version_after:06d}"
        if final.exists():
            shutil.rmtree(pending, ignore_errors=True)
            raise RuntimeError("volunteer canonical adapter version already exists")
        os.replace(pending, final)
        adapter_path = final / "adapter_model.safetensors"
        config_path = final / "adapter_config.json"
        velocity_path = final / "outer_velocity.safetensors"
        adapter_ref = self._register_artifact_in_state(state, f"adapter_v{version_after}", adapter_path)
        config_ref = self._register_artifact_in_state(
            state, f"adapter_config_v{version_after}", config_path
        )
        state.update(
            {
                "adapter_version": version_after,
                "outer_step": int(aggregation["outer_step_after"]),
                "current_adapter_hash": sha256_file(adapter_path),
                "current_adapter_path": str(adapter_path),
                "current_config_path": str(config_path),
                "current_velocity_path": str(velocity_path),
            }
        )
        aggregation_summary = {
            key: value
            for key, value in aggregation.items()
            if key
            not in {
                "global_adapter_tensor_specs",
                "global_adapter_path",
                "velocity_path",
            }
        }
        public_aggregation = with_public_safety(
            {
                **aggregation_summary,
                "schema": AGGREGATION_RECORD_SCHEMA,
                "round_id": round_state["round_id"],
                "quorum": quorum,
                "distinct_cell_count": len({item["cell_id_hash"] for item in records}),
                "input_result_ids": result_ids,
                "input_delta_hashes": [item["accepted_delta_hash"] for item in records],
                "canonical_adapter_hash": state["current_adapter_hash"],
                "canonical_adapter_tensor_contract_hash": state["campaign"][
                    "adapter_tensor_contract_hash"
                ],
                "canonical_adapter_artifact": self._public_artifact_ref(adapter_ref),
                "canonical_config_artifact": self._public_artifact_ref(config_ref),
                "atomic_staging_and_rename": True,
                "append_only_ledger_recorded": True,
                "completed_at": now,
            }
        )
        round_state.update(
            {
                "state": "completed",
                "completed_at": now,
                "aggregation": public_aggregation,
            }
        )
        round_dir = self.root / "rounds" / round_state["round_id"]
        _atomic_write_json(round_dir / "aggregation.json", public_aggregation, mode=0o644)
        self._append_event(
            state,
            "round_aggregated",
            {
                "round_id": round_state["round_id"],
                "round_index": int(round_state["round_index"]),
                "adapter_version_before": version_before,
                "adapter_version_after": version_after,
                "outer_step_after": int(state["outer_step"]),
                "quorum": quorum,
                "distinct_cell_count": quorum,
                "canonical_adapter_hash": state["current_adapter_hash"],
                "input_delta_hashes": [item["accepted_delta_hash"] for item in records],
            },
        )
        target_rounds = int(campaign["round_policy"]["target_rounds"])
        if version_after >= target_rounds:
            state["campaign_complete"] = True
            self._append_event(
                state,
                "campaign_target_reached",
                {
                    "adapter_version": version_after,
                    "outer_step": int(state["outer_step"]),
                    "completed_rounds": version_after,
                },
            )
        else:
            next_round = self._start_round_in_state(state, now=now)
            self._append_event(
                state,
                "round_started",
                {
                    "round_id": next_round["round_id"],
                    "round_index": int(next_round["round_index"]),
                    "base_adapter_version": int(next_round["base_adapter_version"]),
                    "base_adapter_hash": next_round["base_adapter_hash"],
                },
            )

    def artifact_path(
        self,
        artifact_id: str,
        *,
        invite_token: str,
        cell_id: str = "",
        request_nonce: str = "",
    ) -> Path:
        with self._locked_state() as state:
            if cell_id:
                self._authorize_cell(
                    state,
                    token=invite_token,
                    cell_id=cell_id,
                    scope="artifact:read",
                    request_nonce=request_nonce,
                )
            else:
                self._authenticate(state, invite_token)
            record = state["artifact_registry"].get(str(artifact_id))
            if not isinstance(record, dict):
                raise VolunteerProtocolError("volunteer_artifact_not_found", status_code=404)
            path = Path(str(record["local_path"]))
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise VolunteerProtocolError("volunteer_artifact_integrity_failed", status_code=409)
            self._save_state(state)
            return path

    def campaign_manifest(self) -> dict[str, Any]:
        return validate_campaign_manifest(_read_json(self.campaign_path))

    def validate_campaign(self) -> dict[str, Any]:
        manifest = self.campaign_manifest()
        ledger = self.verify_ledger()
        with self._locked_state() as state:
            artifact_errors: list[str] = []
            store = LocalVolunteerBlobStore(state["blob_store_root"])
            for record in state.get("artifact_registry", {}).values():
                blob_hash = str((record or {}).get("blob_hash") or "")
                try:
                    path = store.local_path(blob_hash)
                except VolunteerProtocolError:
                    artifact_errors.append("volunteer_campaign_artifact_reference_invalid")
                    continue
                if not path.is_file() or sha256_file(path) != blob_hash:
                    artifact_errors.append("volunteer_campaign_artifact_integrity_failed")
            evaluation_contract = manifest.get("evaluation_contract")
            heldout_path = Path(str(state.get("evaluation_dataset_path") or ""))
            heldout_validated = evaluation_contract is None
            if isinstance(evaluation_contract, dict):
                heldout_validated = bool(
                    heldout_path.is_file()
                    and sha256_file(heldout_path)
                    == evaluation_contract.get("heldout_dataset_hash")
                )
                if not heldout_validated:
                    artifact_errors.append(
                        "volunteer_campaign_heldout_dataset_integrity_failed"
                    )
            errors = artifact_errors + ([] if ledger.get("ok") else list(ledger["errors"]))
            state["validated_at"] = float(self.clock())
            report = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_campaign_validation_v1",
                    "ok": not errors,
                    "campaign_id": manifest["campaign_id"],
                    "campaign_manifest_hash": manifest["manifest_hash"],
                    "state_schema": state["schema"],
                    "state_revision": int(state.get("state_revision") or 0),
                    "manifest_validated": True,
                    "audit_ledger_validated": ledger.get("ok") is True,
                    "artifact_count": len(state.get("artifact_registry", {})),
                    "content_addressed_artifacts_validated": not artifact_errors,
                    "heldout_dataset_validated": heldout_validated,
                    "errors": errors,
                }
            )
            _atomic_write_json(self.root / "validation.json", report, mode=0o644)
            self._save_state(state)
            return report

    def _set_lifecycle(
        self, *, invite_token: str, target: str
    ) -> dict[str, Any]:
        if target not in {"running", "paused"}:
            raise ValueError("volunteer campaign lifecycle target invalid")
        with self._locked_state() as state:
            self._authenticate(state, invite_token)
            previous = str(state.get("campaign_lifecycle") or "running")
            if previous == "finalized":
                raise VolunteerProtocolError(
                    "volunteer_campaign_already_finalized", status_code=409
                )
            changed = previous != target
            state["campaign_lifecycle"] = target
            if changed:
                self._append_event(
                    state,
                    "campaign_" + ("started" if target == "running" else "paused"),
                    {"previous_lifecycle": previous, "lifecycle": target},
                )
            self._save_state(state)
            status = self._write_status(state)
            return with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_campaign_lifecycle_v1",
                    "ok": True,
                    "campaign_id": state["campaign"]["campaign_id"],
                    "previous_lifecycle": previous,
                    "lifecycle": target,
                    "changed": changed,
                    "status": status,
                }
            )

    def start_campaign(self, *, invite_token: str) -> dict[str, Any]:
        validation = self.validate_campaign()
        if validation.get("ok") is not True:
            raise VolunteerProtocolError("volunteer_campaign_validation_failed")
        return self._set_lifecycle(invite_token=invite_token, target="running")

    def pause_campaign(self, *, invite_token: str) -> dict[str, Any]:
        return self._set_lifecycle(invite_token=invite_token, target="paused")

    def resume_campaign(self, *, invite_token: str) -> dict[str, Any]:
        return self._set_lifecycle(invite_token=invite_token, target="running")

    def _write_evaluation_report(
        self,
        *,
        campaign: dict[str, Any],
        adapter_version: int,
        current_adapter_hash: str,
        accepted: list[dict[str, Any]],
        lineage: dict[str, Any],
        quality: dict[str, Any] | None,
        evaluation_scope: str,
        trusted_external_evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        loss_starts = [float(item.get("loss_start") or 0.0) for item in accepted]
        loss_ends = [float(item.get("loss_end") or 0.0) for item in accepted]
        payload: dict[str, Any] = {
            "schema": "crowdtensor_volunteer_campaign_evaluation_v1",
            "ok": lineage.get("ok") is True,
            "campaign_id": campaign["campaign_id"],
            "campaign_manifest_hash": campaign["manifest_hash"],
            "adapter_version": int(adapter_version),
            "canonical_adapter_hash": current_adapter_hash,
            "accepted_update_count": len(accepted),
            "accepted_tokens_seen": sum(
                int(item.get("tokens_seen") or 0) for item in accepted
            ),
            "mean_reported_loss_start": (
                sum(loss_starts) / len(loss_starts) if loss_starts else None
            ),
            "mean_reported_loss_end": (
                sum(loss_ends) / len(loss_ends) if loss_ends else None
            ),
            "checkpoint_lineage_verified": lineage.get("ok") is True,
            "completed_round_count": int(lineage["completed_round_count"]),
            "evaluation_scope": str(evaluation_scope),
            "held_out_quality_benchmark_performed": quality is not None,
            "quality": quality,
            "quality_improvement_verified": bool(
                quality and quality["quality_improvement_verified"]
            ),
            "statistical_significance_claimed": False,
        }
        if trusted_external_evaluation is not None:
            payload["trusted_external_evaluation"] = trusted_external_evaluation
        report = with_public_safety(payload)
        report["content_hash"] = sha256_json(report)
        _atomic_write_json(self.root / "evaluation.json", report, mode=0o644)
        return report

    def evaluate_campaign(
        self, *, heldout_quality: bool = False, device: str = "cpu"
    ) -> dict[str, Any]:
        with self._locked_state() as state:
            accepted = list(state.get("submissions", {}).values())
            lineage = self._checkpoint_lineage_from_state(state)
            campaign = copy.deepcopy(state["campaign"])
            adapter_version = int(state["adapter_version"])
            current_adapter_hash = str(state["current_adapter_hash"])
            current_adapter_path = Path(str(state["current_adapter_path"]))
            baseline_adapter_path = self.private / "versions" / "v000000"
            evaluation_dataset_path = Path(
                str(state.get("evaluation_dataset_path") or "")
            )
            base_model_record = self._artifact_by_kind(state, "base_model")
        quality: dict[str, Any] | None = None
        if heldout_quality:
            contract = campaign.get("evaluation_contract")
            if not isinstance(contract, dict) or not evaluation_dataset_path.is_file():
                raise VolunteerProtocolError(
                    "volunteer_campaign_heldout_evaluation_unavailable", status_code=409
                )
            if sha256_file(evaluation_dataset_path) != contract["heldout_dataset_hash"]:
                raise VolunteerProtocolError(
                    "volunteer_campaign_heldout_evaluation_changed", status_code=409
                )
            model_archive = Path(str(base_model_record["local_path"]))
            model_root = self.private / "evaluation" / "base-model"
            lock_path = self.private / "evaluation" / "base-model.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    if not (model_root / "config.json").is_file():
                        temporary = model_root.with_name(
                            f".{model_root.name}.{secrets.token_hex(4)}.tmp"
                        )
                        shutil.rmtree(temporary, ignore_errors=True)
                        temporary.mkdir(parents=True, exist_ok=True)
                        _safe_extract_zip(model_archive, temporary)
                        shutil.rmtree(model_root, ignore_errors=True)
                        os.replace(temporary, model_root)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            from .hf_lora_training import evaluate_adapter

            indexes = list(range(int(contract["heldout_sample_count"])))
            baseline = evaluate_adapter(
                base_model_path=model_root,
                adapter_path=baseline_adapter_path,
                dataset_path=evaluation_dataset_path,
                sample_indexes=indexes,
                device=device,
            )
            candidate = evaluate_adapter(
                base_model_path=model_root,
                adapter_path=current_adapter_path.parent,
                dataset_path=evaluation_dataset_path,
                sample_indexes=indexes,
                device=device,
            )
            baseline_loss = float(baseline["mean_loss"])
            candidate_loss = float(candidate["mean_loss"])
            if not math.isfinite(baseline_loss) or not math.isfinite(candidate_loss):
                raise VolunteerProtocolError(
                    "volunteer_campaign_heldout_evaluation_non_finite", status_code=409
                )
            quality = {
                "metric": contract["metric"],
                "heldout_dataset_hash": contract["heldout_dataset_hash"],
                "heldout_sample_count": int(contract["heldout_sample_count"]),
                "baseline_adapter_version": int(contract["baseline_adapter_version"]),
                "candidate_adapter_version": adapter_version,
                "baseline_mean_loss": baseline_loss,
                "candidate_mean_loss": candidate_loss,
                "loss_reduction": baseline_loss - candidate_loss,
                "baseline_perplexity": math.exp(min(80.0, baseline_loss)),
                "candidate_perplexity": math.exp(min(80.0, candidate_loss)),
                "baseline_logits_hash": baseline["logits_hash"],
                "candidate_logits_hash": candidate["logits_hash"],
                "quality_improvement_verified": candidate_loss < baseline_loss,
                "evaluation_device": str(candidate["device"]),
                "statistical_significance_claimed": False,
            }
        return self._write_evaluation_report(
            campaign=campaign,
            adapter_version=adapter_version,
            current_adapter_hash=current_adapter_hash,
            accepted=accepted,
            lineage=lineage,
            quality=quality,
            evaluation_scope=(
                "heldout_loss_perplexity_and_checkpoint_integrity"
                if quality is not None
                else "aggregated_training_metrics_and_checkpoint_integrity"
            ),
        )

    def import_trusted_external_evaluation(
        self, report: dict[str, Any], *, invite_token: str
    ) -> dict[str, Any]:
        """Record an operator-authenticated evaluation performed outside the Session."""

        def reject(cause: BaseException | None = None) -> None:
            error = VolunteerProtocolError(
                "volunteer_trusted_external_evaluation_invalid", status_code=400
            )
            if cause is None:
                raise error
            raise error from cause

        if not isinstance(report, dict):
            reject()
        expected_fields = {
            "schema",
            "campaign_id",
            "campaign_manifest_hash",
            "adapter_version",
            "baseline_adapter_hash",
            "candidate_adapter_hash",
            "model_source_snapshot_hash",
            "heldout_dataset_hash",
            "heldout_sample_count",
            "baseline",
            "candidate",
            "runtime",
            "credential_values_public",
            "private_paths_public",
            "raw_data_public",
            "tensor_values_public",
            "public_artifact_safe",
            "content_hash",
        }
        if set(report) != expected_fields or report.get(
            "schema"
        ) != TRUSTED_EXTERNAL_EVALUATION_SCHEMA:
            reject()
        supplied_hash = str(report.get("content_hash") or "")
        if supplied_hash != sha256_json(
            {key: value for key, value in report.items() if key != "content_hash"}
        ):
            reject()
        if any(
            report.get(field) is not expected
            for field, expected in (
                ("credential_values_public", False),
                ("private_paths_public", False),
                ("raw_data_public", False),
                ("tensor_values_public", False),
                ("public_artifact_safe", True),
            )
        ):
            reject()
        for field in ("adapter_version", "heldout_sample_count"):
            if isinstance(report.get(field), bool) or not isinstance(
                report.get(field), int
            ):
                reject()

        runtime = report.get("runtime")
        runtime_fields = {
            "backend",
            "device",
            "source_revision",
            "model_source_verified",
            "baseline_artifact_verified",
            "candidate_artifact_verified",
            "heldout_artifact_verified",
            "credential_values_public",
            "public_artifact_safe",
        }
        if not isinstance(runtime, dict) or set(runtime) != runtime_fields:
            reject()
        source_revision = str(runtime.get("source_revision") or "").lower()
        backend = str(runtime.get("backend") or "")
        device = str(runtime.get("device") or "")
        backend_characters = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        )
        device_characters = backend_characters + ":"
        if (
            len(source_revision) != 40
            or any(value not in "0123456789abcdef" for value in source_revision)
            or not backend
            or len(backend) > 128
            or any(value not in backend_characters for value in backend)
            or not device
            or len(device) > 64
            or any(value not in device_characters for value in device)
            or any(
                runtime.get(field) is not True
                for field in (
                    "model_source_verified",
                    "baseline_artifact_verified",
                    "candidate_artifact_verified",
                    "heldout_artifact_verified",
                    "public_artifact_safe",
                )
            )
            or runtime.get("credential_values_public") is not False
        ):
            reject()

        evaluations: list[dict[str, Any]] = []
        evaluation_fields = {
            "schema",
            "adapter_loaded",
            "sample_count",
            "mean_loss",
            "logits_hash",
            "logits_norm",
            "device",
        }
        for name in ("baseline", "candidate"):
            value = report.get(name)
            if (
                not isinstance(value, dict)
                or set(value) != evaluation_fields
                or value.get("schema") != "crowdtensor_lora_evaluation_v1"
                or value.get("adapter_loaded") is not True
                or value.get("device") != device
            ):
                reject()
            try:
                sample_count = value.get("sample_count")
                mean_loss = float(value.get("mean_loss"))
                logits_norm = float(value.get("logits_norm"))
            except (TypeError, ValueError) as exc:
                reject(exc)
            logits_hash = str(value.get("logits_hash") or "")
            if (
                isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count < 1
                or isinstance(value.get("mean_loss"), bool)
                or not math.isfinite(mean_loss)
                or mean_loss < 0.0
                or isinstance(value.get("logits_norm"), bool)
                or not math.isfinite(logits_norm)
                or logits_norm < 0.0
                or len(logits_hash) != 71
                or not logits_hash.startswith("sha256:")
                or any(
                    character not in "0123456789abcdef"
                    for character in logits_hash[7:]
                )
            ):
                reject()
            evaluations.append(value)

        with self._locked_state() as state:
            self._authenticate(state, invite_token)
            campaign = copy.deepcopy(state["campaign"])
            contract = campaign.get("evaluation_contract")
            model_source = campaign.get("model_source")
            if not isinstance(contract, dict) or not isinstance(model_source, dict):
                raise VolunteerProtocolError(
                    "volunteer_campaign_heldout_evaluation_unavailable", status_code=409
                )
            try:
                bindings_valid = bool(
                    report["campaign_id"] == campaign["campaign_id"]
                    and report["campaign_manifest_hash"] == campaign["manifest_hash"]
                    and int(report["adapter_version"]) == int(state["adapter_version"])
                    and report["baseline_adapter_hash"]
                    == campaign["initial_adapter_hash"]
                    and report["candidate_adapter_hash"]
                    == state["current_adapter_hash"]
                    and report["model_source_snapshot_hash"]
                    == model_source["imported_snapshot_hash"]
                    and report["heldout_dataset_hash"]
                    == contract["heldout_dataset_hash"]
                    and int(report["heldout_sample_count"])
                    == int(contract["heldout_sample_count"])
                    and all(
                        int(value["sample_count"])
                        == int(contract["heldout_sample_count"])
                        for value in evaluations
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                reject(exc)
            if not bindings_valid:
                reject()

            baseline, candidate = evaluations
            baseline_loss = float(baseline["mean_loss"])
            candidate_loss = float(candidate["mean_loss"])
            quality = {
                "metric": contract["metric"],
                "heldout_dataset_hash": contract["heldout_dataset_hash"],
                "heldout_sample_count": int(contract["heldout_sample_count"]),
                "baseline_adapter_version": int(contract["baseline_adapter_version"]),
                "candidate_adapter_version": int(state["adapter_version"]),
                "baseline_mean_loss": baseline_loss,
                "candidate_mean_loss": candidate_loss,
                "loss_reduction": baseline_loss - candidate_loss,
                "baseline_perplexity": math.exp(min(80.0, baseline_loss)),
                "candidate_perplexity": math.exp(min(80.0, candidate_loss)),
                "baseline_logits_hash": baseline["logits_hash"],
                "candidate_logits_hash": candidate["logits_hash"],
                "quality_improvement_verified": candidate_loss < baseline_loss,
                "evaluation_device": device,
                "statistical_significance_claimed": False,
            }
            accepted = list(state.get("submissions", {}).values())
            lineage = self._checkpoint_lineage_from_state(state)
            external_summary = with_public_safety(
                {
                    "schema": TRUSTED_EXTERNAL_EVALUATION_SCHEMA,
                    "result_content_hash": supplied_hash,
                    "runtime_backend": backend,
                    "evaluation_device": device,
                    "source_revision": source_revision,
                    "operator_authenticated": True,
                    "model_source_verified": True,
                    "baseline_artifact_verified": True,
                    "candidate_artifact_verified": True,
                    "heldout_artifact_verified": True,
                }
            )
            canonical = self._write_evaluation_report(
                campaign=campaign,
                adapter_version=int(state["adapter_version"]),
                current_adapter_hash=str(state["current_adapter_hash"]),
                accepted=accepted,
                lineage=lineage,
                quality=quality,
                evaluation_scope=(
                    "trusted_external_heldout_loss_perplexity_and_checkpoint_integrity"
                ),
                trusted_external_evaluation=external_summary,
            )
            self._append_event(
                state,
                "trusted_external_evaluation_imported",
                {
                    "adapter_version": int(state["adapter_version"]),
                    "evaluation_content_hash": canonical["content_hash"],
                    "external_result_content_hash": supplied_hash,
                },
            )
            self._save_state(state)
            self._write_status(state)
            return canonical

    def finalize_campaign(
        self, *, invite_token: str, allow_incomplete: bool = False
    ) -> dict[str, Any]:
        for _attempt in range(3):
            with self._locked_state() as state:
                self._authenticate(state, invite_token)
                if not state.get("campaign_complete") and not allow_incomplete:
                    raise VolunteerProtocolError(
                        "volunteer_campaign_target_rounds_incomplete", status_code=409
                    )
                evaluation = self._current_evaluation_from_state(state)
                if evaluation is not None:
                    previous = str(state.get("campaign_lifecycle") or "running")
                    state["campaign_lifecycle"] = "finalized"
                    state["finalized_at"] = float(self.clock())
                    self._append_event(
                        state,
                        "campaign_finalized",
                        {
                            "previous_lifecycle": previous,
                            "adapter_version": int(state["adapter_version"]),
                            "campaign_complete": bool(state["campaign_complete"]),
                            "evaluation_content_hash": evaluation["content_hash"],
                        },
                    )
                    self._save_state(state)
                    status = self._write_status(state)
                    return with_public_safety(
                        {
                            "schema": "crowdtensor_volunteer_campaign_finalize_v1",
                            "ok": True,
                            "campaign_id": state["campaign"]["campaign_id"],
                            "lifecycle": "finalized",
                            "campaign_complete": bool(state["campaign_complete"]),
                            "adapter_version": int(state["adapter_version"]),
                            "canonical_adapter_hash": state["current_adapter_hash"],
                            "evaluation_content_hash": evaluation["content_hash"],
                            "status": status,
                        }
                    )
            self.evaluate_campaign()
        raise VolunteerProtocolError(
            "volunteer_campaign_evaluation_changed_during_finalize", status_code=409
        )

    def export_campaign(self, destination: str | Path) -> dict[str, Any]:
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(3):
            with self._locked_state() as state:
                evaluation = self._current_evaluation_from_state(state)
                if evaluation is not None:
                    lineage = self._checkpoint_lineage_from_state(state)
                    lineage_path = _atomic_write_json(
                        self.root / "checkpoint-lineage.json", lineage, mode=0o644
                    )
                    sources = {
                        "campaign.json": self.campaign_path,
                        "status.json": self.status_path,
                        "evaluation.json": self.root / "evaluation.json",
                        "checkpoint-lineage.json": lineage_path,
                        "audit_ledger.jsonl": self.ledger_path,
                        "adapter_model.safetensors": Path(
                            state["current_adapter_path"]
                        ),
                        "adapter_config.json": Path(state["current_config_path"]),
                    }
                    temporary = output.with_name(
                        f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
                    )
                    try:
                        with zipfile.ZipFile(
                            temporary, "w", compression=zipfile.ZIP_DEFLATED
                        ) as archive:
                            for name, source in sorted(sources.items()):
                                info = zipfile.ZipInfo(
                                    name, date_time=(2026, 1, 1, 0, 0, 0)
                                )
                                info.compress_type = zipfile.ZIP_DEFLATED
                                info.external_attr = 0o644 << 16
                                archive.writestr(info, source.read_bytes())
                        temporary.chmod(0o644)
                        os.replace(temporary, output)
                    finally:
                        temporary.unlink(missing_ok=True)
                    report = with_public_safety(
                        {
                            "schema": "crowdtensor_volunteer_campaign_export_v1",
                            "ok": True,
                            "campaign_id": state["campaign"]["campaign_id"],
                            "campaign_manifest_hash": state["campaign"]["manifest_hash"],
                            "adapter_version": int(state["adapter_version"]),
                            "canonical_adapter_hash": state["current_adapter_hash"],
                            "export_hash": sha256_file(output),
                            "export_byte_count": output.stat().st_size,
                            "file_count": len(sources),
                            "evaluation_content_hash": evaluation["content_hash"],
                            "checkpoint_lineage_content_hash": lineage["content_hash"],
                            "credential_values_included": False,
                            "private_runtime_state_included": False,
                        }
                    )
                    _atomic_write_json(
                        output.with_suffix(output.suffix + ".json"),
                        report,
                        mode=0o644,
                    )
                    return report
            self.evaluate_campaign()
        raise VolunteerProtocolError(
            "volunteer_campaign_evaluation_changed_during_export", status_code=409
        )

    def backup_campaign(self, destination: str | Path) -> dict[str, Any]:
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(
            f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        with self._locked_state() as state:
            try:
                with tarfile.open(temporary, "w:gz") as archive:
                    for path in sorted(
                        item for item in self.root.rglob("*") if item.is_file()
                    ):
                        if path == self.lock_path or path == output or path == temporary:
                            continue
                        archive.add(path, arcname=path.relative_to(self.root).as_posix())
                temporary.chmod(0o600)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            return with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_campaign_backup_v1",
                    "ok": True,
                    "campaign_id": state["campaign"]["campaign_id"],
                    "campaign_manifest_hash": state["campaign"]["manifest_hash"],
                    "adapter_version": int(state["adapter_version"]),
                    "backup_hash": sha256_file(output),
                    "backup_byte_count": output.stat().st_size,
                    "private_backup": True,
                    "backup_permissions_restricted": (output.stat().st_mode & 0o077) == 0,
                }
            )

    @classmethod
    def restore_campaign(
        cls, backup: str | Path, destination: str | Path
    ) -> tuple["VolunteerTrainingCoordinator", dict[str, Any]]:
        source = Path(backup).expanduser().resolve()
        target = Path(destination).expanduser().resolve()
        if target.exists() and any(target.iterdir()):
            raise VolunteerProtocolError(
                "volunteer_restore_destination_not_empty", status_code=409
            )
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = Path(member.name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise VolunteerProtocolError(
                        "volunteer_backup_member_unsafe", status_code=400
                    )
            archive.extractall(target, members=members, filter="data")
        state_path = target / ".private" / "coordinator_state.json"
        state = _read_json(state_path)
        adapter_path = Path(str(state.get("current_adapter_path") or "/"))
        old_private = next(
            (parent for parent in adapter_path.parents if parent.name == ".private"),
            None,
        )
        if old_private is None:
            raise VolunteerProtocolError(
                "volunteer_backup_state_paths_invalid", status_code=409
            )
        old_root = old_private.parent

        def rebase(value: str) -> str:
            path = Path(str(value or ""))
            try:
                relative = path.relative_to(old_root)
            except ValueError:
                return str(value)
            return str(target / relative)

        for key in (
            "current_adapter_path",
            "current_config_path",
            "current_velocity_path",
            "blob_store_root",
        ):
            if state.get(key):
                state[key] = rebase(str(state[key]))
        for record in state.get("artifact_registry", {}).values():
            if isinstance(record, dict) and record.get("local_path"):
                record["local_path"] = rebase(str(record["local_path"]))
        for record in state.get("submissions", {}).values():
            if isinstance(record, dict) and record.get("delta_path"):
                record["delta_path"] = rebase(str(record["delta_path"]))
        _atomic_write_json(state_path, state, mode=0o600)
        coordinator = cls(target)
        recovery = coordinator.recover_after_restart()
        report = with_public_safety(
            {
                "schema": "crowdtensor_volunteer_campaign_restore_v1",
                "ok": recovery.get("ok") is True,
                "campaign_id": state["campaign"]["campaign_id"],
                "campaign_manifest_hash": state["campaign"]["manifest_hash"],
                "backup_hash": sha256_file(source),
                "state_rebased": True,
                "coordinator_recovery_verified": recovery.get("ok") is True,
                "audit_ledger_verified": coordinator.verify_ledger().get("ok") is True,
            }
        )
        _atomic_write_json(target / "restore.json", report, mode=0o644)
        return coordinator, report

    def private_invite(self) -> dict[str, Any]:
        invite = _read_json(self.invite_path)
        if invite.get("schema") != INVITE_SCHEMA:
            raise RuntimeError("volunteer invite schema mismatch")
        return invite

    def write_invite(self, coordinator_url: str) -> dict[str, Any]:
        value = self.private_invite()
        value["coordinator_url"] = str(coordinator_url).rstrip("/")
        _atomic_write_json(self.invite_path, value, mode=0o600)
        return value

    def _public_round(self, round_state: dict[str, Any]) -> dict[str, Any]:
        work = list(round_state.get("work_units", {}).values())
        return with_public_safety(
            {
                "round_id": round_state["round_id"],
                "round_index": int(round_state["round_index"]),
                "state": round_state["state"],
                "base_adapter_version": int(round_state["base_adapter_version"]),
                "base_adapter_hash": round_state["base_adapter_hash"],
                "work_unit_count": len(work),
                "queued_work_count": sum(item.get("state") == "queued" for item in work),
                "leased_work_count": sum(item.get("state") == "leased" for item in work),
                "accepted_work_count": sum(item.get("state") == "accepted" for item in work),
                "accepted_result_count": len(round_state.get("accepted_result_ids") or []),
                "distinct_accepted_cell_count": len(
                    set(round_state.get("accepted_cell_hashes") or [])
                ),
                "active_leases": [
                    {
                        "work_id": item["work_id"],
                        "cell_id_hash": item["cell_id_hash"],
                        "lease_generation": int(item["lease_generation"]),
                        "lease_expires_at": float(item["lease_expires_at"]),
                    }
                    for item in work
                    if item.get("state") == "leased"
                ],
                "aggregation": round_state.get("aggregation") or {},
            }
        )

    def _status_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        rounds = [self._public_round(item) for item in state.get("rounds") or []]
        accepted = list(state.get("submissions", {}).values())
        completed_rounds = sum(item.get("state") == "completed" for item in state.get("rounds") or [])
        total_tokens = sum(int(item.get("tokens_seen") or 0) for item in accepted)
        lifecycle = str(state.get("campaign_lifecycle") or "running")
        now = float(self.clock())
        browser_tasks = list(state.get("browser_probe_tasks", {}).values())
        browser_counters = dict(state.get("browser_probe_counters") or {})
        pairing_records = list(state.get("pairing_codes", {}).values())
        status = with_public_safety(
            {
                "schema": STATUS_SCHEMA,
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "campaign_id": state["campaign"]["campaign_id"],
                "campaign_manifest_hash": state["campaign"]["manifest_hash"],
                "campaign_complete": bool(state["campaign_complete"]),
                "overall_state": (
                    "finalized"
                    if lifecycle == "finalized"
                    else "completed"
                    if state["campaign_complete"]
                    else lifecycle
                ),
                "campaign_lifecycle": lifecycle,
                "adapter_version": int(state["adapter_version"]),
                "outer_step": int(state["outer_step"]),
                "canonical_adapter_hash": state["current_adapter_hash"],
                "target_rounds": int(state["campaign"]["round_policy"]["target_rounds"]),
                "completed_rounds": completed_rounds,
                "accepted_update_count": int(state.get("accepted_update_count") or 0),
                "rejected_update_count": int(state.get("rejected_update_count") or 0),
                "expired_lease_count": int(state.get("expired_lease_count") or 0),
                "reassigned_work_count": int(state.get("reassigned_work_count") or 0),
                "coordinator_recovery_count": int(
                    state.get("coordinator_recovery_count") or 0
                ),
                "uploaded_delta_bytes": int(state.get("uploaded_delta_bytes") or 0),
                "accepted_tokens_seen": total_tokens,
                "communication_bytes_per_token": (
                    float(state.get("uploaded_delta_bytes") or 0) / max(1, total_tokens)
                ),
                "rounds": rounds,
                "ledger_sequence": int(state.get("ledger_sequence") or 0),
                "ledger_head_hash": state.get("ledger_head_hash"),
                "append_only_audit_ledger": True,
                "atomic_canonical_version_advance": True,
                "low_frequency_delta_only": True,
                "physical_internet_multi_machine_verified": False,
                "permissionless_byzantine_safety": False,
                "sybil_resistance": False,
                "operator_policy": public_policy_status(
                    state, now=now
                ),
                "contributor_access": {
                    "controlled_enrollment": True,
                    "one_time_pairing_codes": True,
                    "active_pairing_code_count": sum(
                        isinstance(item, dict)
                        and not item.get("redeemed")
                        and float(item.get("expires_at") or 0.0) > now
                        for item in pairing_records
                    ),
                    "pairing_counters": dict(state.get("pairing_counters") or {}),
                    "pairing_code_values_public": False,
                },
                "browser_calibration": {
                    "accepted_task_count": int(browser_counters.get("accepted") or 0),
                    "rejected_task_count": int(browser_counters.get("rejected") or 0),
                    "expired_task_count": int(browser_counters.get("expired") or 0),
                    "active_task_count": sum(
                        isinstance(item, dict)
                        and item.get("state") == "leased"
                        and float(item.get("lease_expires_at") or 0.0) > now
                        for item in browser_tasks
                    ),
                    "webgpu_task_count": int(browser_counters.get("webgpu") or 0),
                    "wasm_cpu_task_count": int(browser_counters.get("wasm_cpu") or 0),
                    "cpu_js_task_count": int(browser_counters.get("cpu_js") or 0),
                    "heartbeat_count": int(browser_counters.get("heartbeats") or 0),
                    "total_vector_elements": int(
                        browser_counters.get("total_vector_elements") or 0
                    ),
                    "total_duration_ms": int(
                        browser_counters.get("total_duration_ms") or 0
                    ),
                    "server_recomputed": True,
                    "model_update_count": 0,
                    "browser_training": False,
                },
                "state_schema": state.get("schema"),
                "state_revision": int(state.get("state_revision") or 0),
                "migration_count": len(state.get("migration_history") or []),
                "artifact_store": LocalVolunteerBlobStore(
                    state["blob_store_root"]
                ).public_report(),
            }
        )
        return status

    def _write_status(self, state: dict[str, Any]) -> dict[str, Any]:
        status = self._status_from_state(state)
        _atomic_write_json(self.status_path, status, mode=0o644)
        return status

    def status(self, *, invite_token: str = "") -> dict[str, Any]:
        with self._locked_state() as state:
            if invite_token:
                self._authenticate(state, invite_token)
            self._expire_browser_probes_in_state(state, now=float(self.clock()))
            self._save_state(state)
            return self._write_status(state)

    def _current_evaluation_from_state(
        self, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        path = self.root / "evaluation.json"
        if not path.is_file():
            return None
        try:
            report = _read_json(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None
        supplied_hash = str(report.get("content_hash") or "")
        expected_hash = sha256_json(
            {key: value for key, value in report.items() if key != "content_hash"}
        )
        campaign = state["campaign"]
        try:
            valid = bool(
                report.get("schema")
                == "crowdtensor_volunteer_campaign_evaluation_v1"
                and report.get("campaign_id") == campaign["campaign_id"]
                and report.get("campaign_manifest_hash") == campaign["manifest_hash"]
                and int(report.get("adapter_version", -1))
                == int(state["adapter_version"])
                and report.get("canonical_adapter_hash")
                == state["current_adapter_hash"]
                and supplied_hash == expected_hash
                and report.get("public_artifact_safe") is True
            )
        except (TypeError, ValueError):
            return None
        if not valid:
            return None
        return report

    def public_campaign_snapshot(self) -> dict[str, Any]:
        """Return dashboard data without Cell, work, lease, or credential IDs."""

        with self._locked_state() as state:
            self._expire_browser_probes_in_state(state, now=float(self.clock()))
            self._save_state(state)
            campaign = state["campaign"]
            rounds = list(state.get("rounds") or [])
            submissions = list(state.get("submissions", {}).values())
            lifecycle = str(state.get("campaign_lifecycle") or "running")
            completed_rounds = sum(item.get("state") == "completed" for item in rounds)
            target_rounds = int(campaign["round_policy"]["target_rounds"])
            current = self._current_round(state)
            current_work = list((current or {}).get("work_units", {}).values())
            browser_tasks = list(state.get("browser_probe_tasks", {}).values())
            browser_counters = dict(state.get("browser_probe_counters") or {})
            loss_starts = [float(item.get("loss_start") or 0.0) for item in submissions]
            loss_ends = [float(item.get("loss_end") or 0.0) for item in submissions]
            model_source = campaign.get("model_source")
            dataset_source = campaign.get("dataset_source")
            lineage = self._checkpoint_lineage_from_state(state)
            current_evaluation = self._current_evaluation_from_state(state)
            data_packs: list[dict[str, Any]] = []
            if isinstance(dataset_source, dict):
                for entry in dataset_source.get("data_packs") or []:
                    if not isinstance(entry, dict):
                        continue
                    manifest = entry.get("manifest")
                    if not isinstance(manifest, dict):
                        continue
                    data_packs.append(
                        {
                            "role": str(entry.get("role") or "unknown"),
                            "pack_id": str(manifest.get("pack_id") or ""),
                            "data_pack_hash": str(manifest.get("content_hash") or ""),
                            "record_count": int(manifest.get("record_count") or 0),
                            "license_spdx": str(manifest.get("license_spdx") or ""),
                            "languages": sorted(
                                str(item) for item in manifest.get("languages") or []
                            ),
                            "domains": sorted(
                                str(item) for item in manifest.get("domains") or []
                            ),
                            "admission_ready": manifest.get("admission_ready") is True,
                            "public_records": manifest.get("public_records") is True,
                        }
                    )
            evaluation_summary: dict[str, Any] = {
                "accepted_update_count": len(submissions),
                "mean_reported_loss_start": (
                    sum(loss_starts) / len(loss_starts) if loss_starts else None
                ),
                "mean_reported_loss_end": (
                    sum(loss_ends) / len(loss_ends) if loss_ends else None
                ),
                "checkpoint_lineage_verified": lineage.get("ok") is True,
                "held_out_quality_benchmark_performed": False,
                "quality_improvement_verified": False,
                "statistical_significance_claimed": False,
                "current_evaluation_available": current_evaluation is not None,
            }
            if current_evaluation is not None:
                evaluation_summary.update(
                    {
                        "evaluation_content_hash": current_evaluation["content_hash"],
                        "evaluation_scope": current_evaluation.get("evaluation_scope"),
                        "held_out_quality_benchmark_performed": current_evaluation.get(
                            "held_out_quality_benchmark_performed"
                        )
                        is True,
                        "quality": public_safe(current_evaluation.get("quality")),
                        "quality_improvement_verified": current_evaluation.get(
                            "quality_improvement_verified"
                        )
                        is True,
                        "statistical_significance_claimed": current_evaluation.get(
                            "statistical_significance_claimed"
                        )
                        is True,
                    }
                )
            round_summaries = [
                {
                    "round_index": int(item["round_index"]),
                    "state": str(item["state"]),
                    "accepted_update_count": len(item.get("accepted_result_ids") or []),
                    "distinct_contributor_count": len(
                        set(item.get("accepted_cell_hashes") or [])
                    ),
                    "work_unit_count": len(item.get("work_units") or {}),
                    "adapter_version_before": int(item["base_adapter_version"]),
                    "adapter_version_after": int(
                        (item.get("aggregation") or {}).get(
                            "adapter_version_after", item["base_adapter_version"]
                        )
                    ),
                    "started_at": float(item.get("started_at") or 0.0),
                    "completed_at": float(item.get("completed_at") or 0.0),
                }
                for item in rounds
            ]
            events: list[dict[str, Any]] = []
            if self.ledger_path.is_file():
                for line in self.ledger_path.read_text(encoding="utf-8").splitlines()[-12:]:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events.append(
                        {
                            "sequence": int(event.get("sequence") or 0),
                            "event_type": str(event.get("event_type") or "event"),
                            "recorded_at": float(event.get("recorded_at") or 0.0),
                        }
                    )
            snapshot = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_public_campaign_snapshot_v1",
                    "ok": True,
                    "campaign": {
                        "campaign_id": campaign["campaign_id"],
                        "campaign_manifest_hash": campaign["manifest_hash"],
                        "campaign_profile": (
                            campaign.get("campaign_import") or {}
                        ).get("profile", "custom"),
                        "model_id": campaign["model_id"],
                        "model_revision": int(campaign["model_revision"]),
                        "model_parameter_count": int(
                            campaign.get("model_parameter_count") or 0
                        ),
                        "dataset_id": campaign["dataset_id"],
                        "dataset_revision": int(campaign["dataset_revision"]),
                        "dataset_shard_count": len(campaign["dataset_shards"]),
                        "protocol_version": campaign["protocol_version"],
                    },
                    "progress": {
                        "lifecycle": lifecycle,
                        "campaign_complete": bool(state["campaign_complete"]),
                        "adapter_version": int(state["adapter_version"]),
                        "outer_step": int(state["outer_step"]),
                        "completed_rounds": completed_rounds,
                        "target_rounds": target_rounds,
                        "progress_fraction": min(
                            1.0, float(completed_rounds) / max(1, target_rounds)
                        ),
                        "accepted_update_count": int(
                            state.get("accepted_update_count") or 0
                        ),
                        "accepted_token_count": sum(
                            int(item.get("tokens_seen") or 0) for item in submissions
                        ),
                        "active_contributor_count": sum(
                            item.get("state") == "leased" for item in current_work
                        ),
                        "queued_work_count": sum(
                            item.get("state") == "queued" for item in current_work
                        ),
                        "uploaded_delta_bytes": int(
                            state.get("uploaded_delta_bytes") or 0
                        ),
                        "accepted_browser_task_count": int(
                            browser_counters.get("accepted") or 0
                        ),
                        "active_browser_contributor_count": sum(
                            isinstance(item, dict) and item.get("state") == "leased"
                            for item in browser_tasks
                        ),
                    },
                    "browser_calibration": {
                        "accepted_task_count": int(
                            browser_counters.get("accepted") or 0
                        ),
                        "webgpu_task_count": int(browser_counters.get("webgpu") or 0),
                        "fallback_task_count": int(
                            browser_counters.get("wasm_cpu") or 0
                        )
                        + int(browser_counters.get("cpu_js") or 0),
                        "server_recomputed": True,
                        "model_update_count": 0,
                        "browser_training": False,
                    },
                    "rounds": round_summaries,
                    "data": {
                        "data_pack_count": len(data_packs),
                        "training_data_pack_count": sum(
                            item["role"] == "train" for item in data_packs
                        ),
                        "evaluation_data_pack_count": sum(
                            item["role"] == "evaluation" for item in data_packs
                        ),
                        "public_record_count": sum(
                            item["record_count"] for item in data_packs
                        ),
                        "licenses": sorted(
                            {item["license_spdx"] for item in data_packs if item["license_spdx"]}
                        ),
                        "all_data_packs_admission_ready": bool(data_packs)
                        and all(item["admission_ready"] for item in data_packs),
                        "benchmark_overlap_detected": bool(
                            isinstance(dataset_source, dict)
                            and dataset_source.get("benchmark_overlap_detected") is True
                        ),
                        "raw_records_embedded": False,
                        "data_packs": data_packs,
                    },
                    "checkpoint_lineage": lineage,
                    "evaluation": evaluation_summary,
                    "reliability": {
                        "expired_lease_count": int(
                            state.get("expired_lease_count") or 0
                        ),
                        "reassigned_work_count": int(
                            state.get("reassigned_work_count") or 0
                        ),
                        "coordinator_recovery_count": int(
                            state.get("coordinator_recovery_count") or 0
                        ),
                        "rejected_update_count": int(
                            state.get("rejected_update_count") or 0
                        ),
                    },
                    "provenance": {
                        "model_source": public_safe(model_source or {}),
                        "dataset_source": public_safe(dataset_source or {}),
                        "dataset_snapshot_hash": campaign["dataset_snapshot_hash"],
                        "initial_adapter_hash": campaign["initial_adapter_hash"],
                        "canonical_adapter_hash": state["current_adapter_hash"],
                        "append_only_ledger_head_hash": state["ledger_head_hash"],
                    },
                    "activity": events,
                    "trust_boundary": {
                        "permissionless": False,
                        "sybil_resistance": False,
                        "semantic_poisoning_safety": False,
                        "secure_aggregation": False,
                        "physical_multi_host_verified": False,
                        "quality_improvement_verified": evaluation_summary[
                            "quality_improvement_verified"
                        ],
                        "browser_model_training": False,
                        "browser_large_model_sharding": False,
                    },
                    "privacy": {
                        "cell_identifiers_public": False,
                        "work_identifiers_public": False,
                        "lease_material_public": False,
                        "credential_identifiers_public": False,
                        "pairing_code_values_public": False,
                        "raw_training_data_public": False,
                    },
                }
            )
            snapshot["content_hash"] = sha256_json(snapshot)
            return snapshot

    def prometheus_metrics(self) -> str:
        """Return a credential-free Prometheus text exposition."""

        with self._locked_state() as state:
            status = self._status_from_state(state)
            policy = status["operator_policy"]
            values = {
                "crowdtensor_volunteer_adapter_version": status["adapter_version"],
                "crowdtensor_volunteer_completed_rounds": status["completed_rounds"],
                "crowdtensor_volunteer_accepted_updates_total": status[
                    "accepted_update_count"
                ],
                "crowdtensor_volunteer_rejected_updates_total": status[
                    "rejected_update_count"
                ],
                "crowdtensor_volunteer_expired_leases_total": status[
                    "expired_lease_count"
                ],
                "crowdtensor_volunteer_recoveries_total": status[
                    "coordinator_recovery_count"
                ],
                "crowdtensor_volunteer_uploaded_delta_bytes_total": status[
                    "uploaded_delta_bytes"
                ],
                "crowdtensor_volunteer_active_credentials": policy[
                    "active_credential_count"
                ],
                "crowdtensor_volunteer_revoked_credentials_total": policy[
                    "revoked_credential_count"
                ],
                "crowdtensor_volunteer_rate_limit_rejections_total": policy[
                    "counters"
                ]["rate_limit_rejections"],
                "crowdtensor_volunteer_replay_rejections_total": policy[
                    "counters"
                ]["replay_rejections"],
                "crowdtensor_volunteer_browser_tasks_accepted_total": status[
                    "browser_calibration"
                ]["accepted_task_count"],
                "crowdtensor_volunteer_browser_tasks_active": status[
                    "browser_calibration"
                ]["active_task_count"],
                "crowdtensor_volunteer_pairing_codes_active": status[
                    "contributor_access"
                ]["active_pairing_code_count"],
            }
            return "".join(f"{name} {int(value)}\n" for name, value in values.items())

    def cleanup(self) -> dict[str, Any]:
        uploads = self.private / "uploads"
        if uploads.exists():
            shutil.rmtree(uploads)
        resumable_uploads = self.private / "resumable-uploads"
        if resumable_uploads.exists():
            shutil.rmtree(resumable_uploads)
        upload_object_store = self.private / "upload-object-store"
        if upload_object_store.exists():
            shutil.rmtree(upload_object_store)
        report = with_public_safety(
            {
                "schema": "crowdtensor_volunteer_training_cleanup_v1",
                "ok": True,
                "temporary_uploads_removed": not uploads.exists(),
                "resumable_uploads_removed": not resumable_uploads.exists(),
                "completed_upload_blobs_removed": not upload_object_store.exists(),
                "canonical_adapters_preserved": True,
                "content_addressed_artifacts_preserved": True,
                "audit_ledger_preserved": self.ledger_path.is_file(),
                "live_resources_left_running": False,
                "cleanup_verified": True,
            }
        )
        _atomic_write_json(self.root / "cleanup.json", report, mode=0o644)
        return report

    def recover_after_restart(self) -> dict[str, Any]:
        """Verify durable state without fencing valid in-flight leases."""

        ledger_before = self.verify_ledger()
        if ledger_before.get("ok") is not True:
            raise RuntimeError("volunteer_coordinator_recovery_ledger_invalid")
        with self._locked_state() as state:
            adapter_path = Path(str(state.get("current_adapter_path") or ""))
            config_path = Path(str(state.get("current_config_path") or ""))
            if (
                not adapter_path.is_file()
                or sha256_file(adapter_path) != state.get("current_adapter_hash")
                or not config_path.is_file()
            ):
                raise RuntimeError("volunteer_coordinator_recovery_canonical_adapter_invalid")
            store = LocalVolunteerBlobStore(state["blob_store_root"])
            for record in state.get("artifact_registry", {}).values():
                if not isinstance(record, dict):
                    raise RuntimeError("volunteer_coordinator_recovery_artifact_registry_invalid")
                blob_hash = str(record.get("blob_hash") or "")
                path = store.local_path(blob_hash)
                if not path.is_file() or sha256_file(path) != blob_hash:
                    raise RuntimeError("volunteer_coordinator_recovery_blob_invalid")
            active_round = self._current_round(state)
            active_leases = sum(
                item.get("state") == "leased"
                for item in (active_round or {}).get("work_units", {}).values()
            )
            state["coordinator_recovery_count"] = int(
                state.get("coordinator_recovery_count") or 0
            ) + 1
            self._append_event(
                state,
                "coordinator_recovered",
                {
                    "adapter_version": int(state["adapter_version"]),
                    "outer_step": int(state["outer_step"]),
                    "active_lease_count_preserved": active_leases,
                    "artifact_count_verified": len(state.get("artifact_registry", {})),
                    "recovery_count": int(state["coordinator_recovery_count"]),
                },
            )
            self._save_state(state)
            self._write_status(state)
            return with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_coordinator_recovery_v1",
                    "ok": True,
                    "coordinator_state_reloaded": True,
                    "canonical_adapter_verified": True,
                    "audit_ledger_verified_before_recovery": True,
                    "content_addressed_artifacts_verified": True,
                    "active_lease_count_preserved": active_leases,
                    "adapter_version": int(state["adapter_version"]),
                    "outer_step": int(state["outer_step"]),
                    "recovery_count": int(state["coordinator_recovery_count"]),
                }
            )

    @staticmethod
    def _checkpoint_lineage_from_state(state: dict[str, Any]) -> dict[str, Any]:
        expected_base = str(state["campaign"]["initial_adapter_hash"])
        entries: list[dict[str, Any]] = []
        errors: list[str] = []
        for round_state in state.get("rounds") or []:
            if round_state.get("state") != "completed":
                continue
            aggregation = round_state.get("aggregation") or {}
            base_hash = str(round_state.get("base_adapter_hash") or "")
            output_hash = str(aggregation.get("canonical_adapter_hash") or "")
            version_before = int(round_state.get("base_adapter_version") or 0)
            version_after = int(aggregation.get("adapter_version_after") or 0)
            entry_ok = bool(
                base_hash == expected_base
                and version_after == version_before + 1
                and output_hash.startswith("sha256:")
                and int(aggregation.get("input_delta_count") or 0)
                >= int(state["campaign"]["round_policy"]["minimum_quorum"])
            )
            if not entry_ok:
                errors.append(
                    "volunteer_checkpoint_lineage_round_invalid:"
                    + str(round_state.get("round_id") or "")
                )
            entries.append(
                {
                    "round_id": round_state["round_id"],
                    "round_index": int(round_state["round_index"]),
                    "adapter_version_before": version_before,
                    "adapter_version_after": version_after,
                    "base_adapter_hash": base_hash,
                    "canonical_adapter_hash": output_hash,
                    "input_delta_hashes": list(
                        aggregation.get("input_delta_hashes") or []
                    ),
                    "distinct_cell_count": int(
                        aggregation.get("distinct_cell_count") or 0
                    ),
                    "lineage_link_verified": entry_ok,
                }
            )
            expected_base = output_hash
        if expected_base != state.get("current_adapter_hash"):
            errors.append("volunteer_checkpoint_lineage_head_mismatch")
        report = with_public_safety(
            {
                "schema": "crowdtensor_volunteer_checkpoint_lineage_v1",
                "ok": not errors,
                "campaign_id": state["campaign"]["campaign_id"],
                "campaign_manifest_hash": state["campaign"]["manifest_hash"],
                "initial_adapter_hash": state["campaign"]["initial_adapter_hash"],
                "canonical_adapter_hash": state["current_adapter_hash"],
                "adapter_version": int(state["adapter_version"]),
                "outer_step": int(state["outer_step"]),
                "completed_round_count": len(entries),
                "entries": entries,
                "errors": errors,
                "append_only_ledger_head_hash": state["ledger_head_hash"],
            }
        )
        report["content_hash"] = sha256_json(report)
        return report

    def checkpoint_lineage(self) -> dict[str, Any]:
        with self._locked_state() as state:
            return self._checkpoint_lineage_from_state(state)

    def verify_ledger(self) -> dict[str, Any]:
        previous = "sha256:" + "0" * 64
        count = 0
        errors: list[str] = []
        if not self.ledger_path.is_file():
            errors.append("volunteer_audit_ledger_missing")
        else:
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                count += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    errors.append("volunteer_audit_ledger_json_invalid")
                    break
                if event.get("previous_event_hash") != previous:
                    errors.append("volunteer_audit_ledger_chain_broken")
                    break
                computed = sha256_json(
                    {key: value for key, value in event.items() if key != "event_hash"}
                )
                if event.get("event_hash") != computed:
                    errors.append("volunteer_audit_ledger_event_hash_mismatch")
                    break
                previous = computed
        with self._locked_state() as state:
            if previous != state.get("ledger_head_hash"):
                errors.append("volunteer_audit_ledger_head_mismatch")
            if count != int(state.get("ledger_sequence") or 0):
                errors.append("volunteer_audit_ledger_sequence_mismatch")
        return with_public_safety(
            {
                "schema": "crowdtensor_volunteer_training_ledger_check_v1",
                "ok": not errors,
                "event_count": count,
                "head_hash": previous,
                "errors": errors,
            }
        )


def safe_coordinator_error(exc: VolunteerProtocolError) -> dict[str, Any]:
    return public_error(exc.code)
