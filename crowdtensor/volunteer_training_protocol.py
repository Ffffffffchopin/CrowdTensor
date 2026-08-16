"""Versioned contracts for low-frequency volunteer LoRA training rounds."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from typing import Any

from .core.data_packs import DataPack, DataPackError
from .training_contract import sha256_bytes, sha256_json


PROTOCOL_VERSION = "volunteer_training_v1.0"
CAMPAIGN_SCHEMA = "crowdtensor_volunteer_training_campaign_v1"
WORK_UNIT_SCHEMA = "crowdtensor_volunteer_training_work_unit_v1"
SUBMISSION_SCHEMA = "crowdtensor_volunteer_training_submission_v1"
CLAIM_SCHEMA = "crowdtensor_volunteer_training_claim_v1"
STATUS_SCHEMA = "crowdtensor_volunteer_training_status_v1"
LEDGER_EVENT_SCHEMA = "crowdtensor_volunteer_training_ledger_event_v1"
INVITE_SCHEMA = "crowdtensor_volunteer_training_invite_v1"
CELL_STATE_SCHEMA = "crowdtensor_volunteer_training_cell_state_v1"
MAX_SUBMISSION_METADATA_BYTES = 4 * 1024 * 1024

_PRIVATE_KEYS = {
    "adapter_path",
    "artifact_path",
    "base_model_path",
    "cache_path",
    "checkpoint_path",
    "coordinator_url",
    "dataset_path",
    "delta_path",
    "invite_token",
    "pairing_code",
    "lease_token",
    "local_path",
    "private_path",
    "raw_data",
    "session_token",
    "tensor_values",
}


class VolunteerProtocolError(RuntimeError):
    """A stable, public-safe rejection raised at a protocol boundary."""

    def __init__(self, code: str, *, status_code: int = 409) -> None:
        super().__init__(str(code))
        self.code = str(code)
        self.status_code = int(status_code)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def hash_cell_id(cell_id: str) -> str:
    value = str(cell_id).strip()
    if not value:
        raise VolunteerProtocolError("volunteer_cell_id_missing", status_code=400)
    return sha256_bytes(value.encode("utf-8"))


def token_hash(token: str) -> str:
    return sha256_bytes(str(token).encode("utf-8"))


def public_safe(value: Any) -> Any:
    """Recursively remove local paths and bearer material from a public object."""

    if isinstance(value, dict):
        public: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key)
            if (
                normalized in _PRIVATE_KEYS
                or normalized.endswith("_path")
                or normalized.endswith("_token")
                or normalized.endswith("_url")
            ):
                continue
            public[normalized] = public_safe(item)
        return public
    if isinstance(value, list):
        return [public_safe(item) for item in value]
    return value


def with_public_safety(value: dict[str, Any]) -> dict[str, Any]:
    public = dict(public_safe(value))
    public.update(
        {
            "credential_values_public": False,
            "private_paths_public": False,
            "raw_data_public": False,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }
    )
    return public


def _require_hash(value: Any, code: str) -> str:
    text = str(value or "")
    if not text.startswith("sha256:") or len(text) != 71:
        raise VolunteerProtocolError(code, status_code=400)
    return text


def _finite_positive(value: Any, code: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VolunteerProtocolError(code, status_code=400) from exc
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        raise VolunteerProtocolError(code, status_code=400)
    return number


def _validate_campaign_import(manifest: dict[str, Any]) -> None:
    imported = manifest.get("campaign_import")
    if imported is None:
        return
    if not isinstance(imported, dict) or imported.get("schema") != (
        "crowdtensor_volunteer_campaign_import_v1"
    ):
        raise VolunteerProtocolError("volunteer_campaign_import_schema_invalid", status_code=400)
    adapter_id = str(manifest.get("model_adapter_id") or "")
    if not adapter_id or imported.get("model_adapter_id") != adapter_id:
        raise VolunteerProtocolError("volunteer_campaign_model_adapter_invalid", status_code=400)
    if imported.get("source_verified") is not True:
        raise VolunteerProtocolError("volunteer_campaign_import_unverified", status_code=400)
    supplied_import_hash = str(imported.get("content_hash") or "")
    expected_import_hash = sha256_json(
        {key: value for key, value in imported.items() if key != "content_hash"}
    )
    if supplied_import_hash != expected_import_hash:
        raise VolunteerProtocolError("volunteer_campaign_import_hash_mismatch", status_code=400)

    model_source = manifest.get("model_source")
    dataset_source = manifest.get("dataset_source")
    if not isinstance(model_source, dict) or not isinstance(dataset_source, dict):
        raise VolunteerProtocolError("volunteer_campaign_source_missing", status_code=400)
    if model_source != imported.get("model_source") or dataset_source != imported.get(
        "dataset_source"
    ):
        raise VolunteerProtocolError("volunteer_campaign_source_binding_mismatch", status_code=400)
    if not all(
        source.get("source_verified") is True
        and source.get("source_public") is True
        and source.get("immutable_revision") is True
        and len(str(source.get("revision") or "")) == 40
        for source in (model_source, dataset_source)
    ):
        raise VolunteerProtocolError("volunteer_campaign_source_unverified", status_code=400)

    model_files = model_source.get("imported_files")
    if not isinstance(model_files, list) or not model_files:
        raise VolunteerProtocolError("volunteer_campaign_model_files_missing", status_code=400)
    for record in model_files:
        if not isinstance(record, dict) or not str(record.get("relative_name") or ""):
            raise VolunteerProtocolError("volunteer_campaign_model_file_invalid", status_code=400)
        _require_hash(record.get("sha256"), "volunteer_campaign_model_file_hash_invalid")
        if int(record.get("byte_count") or 0) < 1:
            raise VolunteerProtocolError("volunteer_campaign_model_file_size_invalid", status_code=400)
    if model_source.get("imported_snapshot_hash") != sha256_json(model_files):
        raise VolunteerProtocolError("volunteer_campaign_model_snapshot_hash_invalid", status_code=400)
    runtime_fetch = model_source.get("runtime_fetch")
    if runtime_fetch is not None:
        expected_names = sorted(str(record["relative_name"]) for record in model_files)
        if not (
            isinstance(runtime_fetch, dict)
            and runtime_fetch.get("schema")
            == "crowdtensor_huggingface_snapshot_fetch_v1"
            and runtime_fetch.get("provider") == "huggingface_hub"
            and runtime_fetch.get("repo_id") == model_source.get("model_id")
            and runtime_fetch.get("revision") == model_source.get("revision")
            and runtime_fetch.get("file_manifest_hash")
            == model_source.get("imported_snapshot_hash")
            and runtime_fetch.get("allow_patterns") == expected_names
            and runtime_fetch.get("trust_remote_code") is False
            and model_source.get("gated") is False
            and model_source.get("private") is False
        ):
            raise VolunteerProtocolError(
                "volunteer_campaign_model_runtime_fetch_invalid", status_code=400
            )

    dataset_files = dataset_source.get("source_files")
    if not isinstance(dataset_files, list) or len(dataset_files) < 2:
        raise VolunteerProtocolError("volunteer_campaign_dataset_files_missing", status_code=400)
    for record in dataset_files:
        if not isinstance(record, dict) or not str(record.get("relative_name") or ""):
            raise VolunteerProtocolError("volunteer_campaign_dataset_file_invalid", status_code=400)
        _require_hash(record.get("sha256"), "volunteer_campaign_dataset_file_hash_invalid")
        if int(record.get("byte_count") or 0) < 1:
            raise VolunteerProtocolError("volunteer_campaign_dataset_file_size_invalid", status_code=400)
    if dataset_source.get("source_snapshot_hash") != sha256_json(dataset_files):
        raise VolunteerProtocolError("volunteer_campaign_dataset_snapshot_hash_invalid", status_code=400)

    if imported.get("profile") != "commons_instruction_sft_lora_v1":
        return
    if not (
        imported.get("deterministic_tokenization") is True
        and imported.get("response_only_supervision") is True
        and imported.get("raw_text_public") is False
        and imported.get("token_ids_public") is False
    ):
        raise VolunteerProtocolError(
            "volunteer_commons_training_contract_invalid", status_code=400
        )
    entries = dataset_source.get("data_packs")
    if not isinstance(entries, list) or len(entries) < 2:
        raise VolunteerProtocolError(
            "volunteer_commons_data_packs_missing", status_code=400
        )
    packs: list[tuple[str, DataPack]] = []
    seen_pack_ids: set[str] = set()
    try:
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"role", "manifest"}:
                raise DataPackError("data_pack_campaign_entry_invalid")
            role = str(entry.get("role") or "")
            if role not in {"train", "evaluation"}:
                raise DataPackError("data_pack_campaign_role_invalid")
            pack = DataPack.from_dict(entry.get("manifest"))
            if (
                pack.pack_id in seen_pack_ids
                or not pack.admission_ready
                or not pack.public_records
            ):
                raise DataPackError("data_pack_campaign_admission_invalid")
            seen_pack_ids.add(pack.pack_id)
            packs.append((role, pack))
    except (DataPackError, TypeError, ValueError) as exc:
        raise VolunteerProtocolError(
            "volunteer_commons_data_pack_invalid", status_code=400
        ) from exc
    training = [pack for role, pack in packs if role == "train"]
    evaluation = [pack for role, pack in packs if role == "evaluation"]
    if not training or len(evaluation) != 1:
        raise VolunteerProtocolError(
            "volunteer_commons_data_pack_roles_invalid", status_code=400
        )

    indexed_files = {
        (
            str(record.get("split") or ""),
            str(record.get("pack_id") or ""),
            str(record.get("relative_name") or ""),
        ): record
        for record in dataset_files
    }
    if len(indexed_files) != len(dataset_files) or len(dataset_files) != 2 * len(packs):
        raise VolunteerProtocolError(
            "volunteer_commons_data_pack_files_invalid", status_code=400
        )
    for role, pack in packs:
        manifest_name = f"data-packs/{pack.pack_id}/data-pack.json"
        records_name = f"data-packs/{pack.pack_id}/records.jsonl"
        manifest_record = indexed_files.get((role, pack.pack_id, manifest_name)) or {}
        records_record = indexed_files.get((role, pack.pack_id, records_name)) or {}
        encoded_manifest = (
            json.dumps(
                pack.to_dict(), indent=2, sort_keys=True, ensure_ascii=True
            )
            + "\n"
        ).encode("utf-8")
        if not (
            manifest_record.get("sha256") == sha256_bytes(encoded_manifest)
            and int(manifest_record.get("byte_count") or 0) == len(encoded_manifest)
            and records_record.get("sha256") == pack.records_hash
            and int(records_record.get("byte_count") or 0) == pack.byte_count
        ):
            raise VolunteerProtocolError(
                "volunteer_commons_data_pack_file_binding_invalid", status_code=400
            )
    if not (
        int(dataset_source.get("training_data_pack_count") or 0) == len(training)
        and dataset_source.get("evaluation_data_pack_id") == evaluation[0].pack_id
        and int(dataset_source.get("public_training_record_count") or 0)
        == sum(pack.record_count for pack in training)
        and int(dataset_source.get("public_evaluation_record_count") or 0)
        == evaluation[0].record_count
        and dataset_source.get("licenses")
        == sorted({pack.license_spdx for _role, pack in packs})
        and dataset_source.get("all_data_packs_admission_ready") is True
        and dataset_source.get("all_records_redistributable") is True
        and dataset_source.get("benchmark_overlap_detected") is False
    ):
        raise VolunteerProtocolError(
            "volunteer_commons_data_pack_summary_invalid", status_code=400
        )


def campaign_content_hash(manifest: dict[str, Any]) -> str:
    content = {
        key: value
        for key, value in public_safe(manifest).items()
        if key not in {"manifest_hash", "public_artifact_safe"}
    }
    return sha256_json(content)


def validate_campaign_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema") != CAMPAIGN_SCHEMA:
        raise VolunteerProtocolError("volunteer_campaign_schema_mismatch", status_code=400)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise VolunteerProtocolError("volunteer_protocol_version_incompatible", status_code=400)
    if not str(manifest.get("campaign_id") or "").strip():
        raise VolunteerProtocolError("volunteer_campaign_id_missing", status_code=400)
    _require_hash(manifest.get("model_manifest_hash"), "volunteer_model_manifest_hash_invalid")
    _require_hash(manifest.get("dataset_snapshot_hash"), "volunteer_dataset_snapshot_hash_invalid")
    _require_hash(manifest.get("initial_adapter_hash"), "volunteer_initial_adapter_hash_invalid")
    _require_hash(manifest.get("adapter_tensor_contract_hash"), "volunteer_adapter_contract_hash_invalid")

    shards = manifest.get("dataset_shards")
    if not isinstance(shards, list) or len(shards) < 2:
        raise VolunteerProtocolError("volunteer_dataset_shards_insufficient", status_code=400)
    indexes: set[int] = set()
    for shard in shards:
        if not isinstance(shard, dict):
            raise VolunteerProtocolError("volunteer_dataset_shard_invalid", status_code=400)
        try:
            index = int(shard["shard_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VolunteerProtocolError("volunteer_dataset_shard_index_invalid", status_code=400) from exc
        if index in indexes or index < 0:
            raise VolunteerProtocolError("volunteer_dataset_shard_index_invalid", status_code=400)
        indexes.add(index)
        _require_hash(shard.get("shard_hash"), "volunteer_dataset_shard_hash_invalid")
        if int(shard.get("sample_count") or 0) < 1:
            raise VolunteerProtocolError("volunteer_dataset_shard_empty", status_code=400)

    local = manifest.get("local_training")
    if not isinstance(local, dict):
        raise VolunteerProtocolError("volunteer_local_training_policy_missing", status_code=400)
    local_steps = int(local.get("local_steps") or 0)
    if local_steps < 1 or local_steps > int(local.get("max_local_steps") or 0):
        raise VolunteerProtocolError("volunteer_local_steps_out_of_bounds", status_code=400)
    _finite_positive(local.get("learning_rate"), "volunteer_learning_rate_invalid")

    rounds = manifest.get("round_policy")
    if not isinstance(rounds, dict):
        raise VolunteerProtocolError("volunteer_round_policy_missing", status_code=400)
    quorum = int(rounds.get("minimum_quorum") or 0)
    if quorum < 2 or quorum > len(shards):
        raise VolunteerProtocolError("volunteer_round_quorum_invalid", status_code=400)
    if int(rounds.get("target_rounds") or 0) < 1:
        raise VolunteerProtocolError("volunteer_target_rounds_invalid", status_code=400)
    _finite_positive(rounds.get("lease_seconds"), "volunteer_lease_seconds_invalid")

    outer = manifest.get("outer_optimizer")
    if not isinstance(outer, dict) or outer.get("optimizer_type") not in {
        "diloco_momentum",
        "local_sgd_mean",
    }:
        raise VolunteerProtocolError("volunteer_outer_optimizer_invalid", status_code=400)
    _finite_positive(outer.get("outer_lr"), "volunteer_outer_lr_invalid")
    momentum = _finite_positive(
        outer.get("momentum"), "volunteer_outer_momentum_invalid", allow_zero=True
    )
    if momentum >= 1.0:
        raise VolunteerProtocolError("volunteer_outer_momentum_invalid", status_code=400)

    admission = manifest.get("update_admission")
    if not isinstance(admission, dict):
        raise VolunteerProtocolError("volunteer_update_admission_missing", status_code=400)
    clip_norm = _finite_positive(admission.get("clip_delta_norm"), "volunteer_clip_norm_invalid")
    hard_norm = _finite_positive(admission.get("hard_max_delta_norm"), "volunteer_hard_norm_invalid")
    if hard_norm < clip_norm:
        raise VolunteerProtocolError("volunteer_hard_norm_below_clip_norm", status_code=400)

    requirements = manifest.get("resource_requirements")
    if requirements is not None:
        if (
            not isinstance(requirements, dict)
            or requirements.get("schema")
            != "crowdtensor_volunteer_resource_requirements_v1"
        ):
            raise VolunteerProtocolError(
                "volunteer_resource_requirements_invalid", status_code=400
            )
        devices = requirements.get("supported_devices")
        if (
            not isinstance(devices, list)
            or not devices
            or any(value not in {"cpu", "cuda"} for value in devices)
            or len(devices) != len(set(devices))
        ):
            raise VolunteerProtocolError(
                "volunteer_resource_devices_invalid", status_code=400
            )
        numeric_fields = (
            "first_work_unit_download_bytes",
            "recurring_work_unit_download_bytes",
            "minimum_memory_bytes",
            "minimum_free_disk_bytes",
        )
        if any(int(requirements.get(name) or 0) < 1 for name in numeric_fields):
            raise VolunteerProtocolError(
                "volunteer_resource_limit_invalid", status_code=400
            )
        if int(requirements["first_work_unit_download_bytes"]) < int(
            requirements["recurring_work_unit_download_bytes"]
        ):
            raise VolunteerProtocolError(
                "volunteer_resource_download_contract_invalid", status_code=400
            )
        if int(requirements.get("local_steps") or 0) != local_steps:
            raise VolunteerProtocolError(
                "volunteer_resource_step_contract_invalid", status_code=400
            )

    evaluation = manifest.get("evaluation_contract")
    if evaluation is not None:
        if (
            not isinstance(evaluation, dict)
            or evaluation.get("schema")
            != "crowdtensor_volunteer_evaluation_contract_v1"
            or evaluation.get("metric") != "mean_token_cross_entropy"
            or int(evaluation.get("heldout_sample_count") or 0) < 1
            or evaluation.get("statistical_significance_claimed") is not False
        ):
            raise VolunteerProtocolError(
                "volunteer_evaluation_contract_invalid", status_code=400
            )
        _require_hash(
            evaluation.get("heldout_dataset_hash"),
            "volunteer_evaluation_dataset_hash_invalid",
        )

    _validate_campaign_import(manifest)

    expected_hash = campaign_content_hash(manifest)
    if manifest.get("manifest_hash") != expected_hash:
        raise VolunteerProtocolError("volunteer_campaign_manifest_hash_mismatch", status_code=400)
    return with_public_safety(manifest)


def work_unit_content_hash(work: dict[str, Any]) -> str:
    content = {
        key: value
        for key, value in public_safe(work).items()
        if key not in {"work_unit_hash", "public_artifact_safe"}
    }
    return sha256_json(content)


def validate_work_unit(
    work: dict[str, Any],
    *,
    campaign: dict[str, Any],
    now: float | None = None,
) -> dict[str, Any]:
    if not isinstance(work, dict) or work.get("schema") != WORK_UNIT_SCHEMA:
        raise VolunteerProtocolError("volunteer_work_unit_schema_mismatch", status_code=400)
    if work.get("protocol_version") != PROTOCOL_VERSION:
        raise VolunteerProtocolError("volunteer_protocol_version_incompatible", status_code=400)
    if work.get("campaign_id") != campaign.get("campaign_id"):
        raise VolunteerProtocolError("volunteer_work_campaign_mismatch", status_code=409)
    if work.get("campaign_manifest_hash") != campaign.get("manifest_hash"):
        raise VolunteerProtocolError("volunteer_work_campaign_version_mismatch", status_code=409)
    if int(work.get("adapter_version", -1)) < 0:
        raise VolunteerProtocolError("volunteer_work_adapter_version_invalid", status_code=400)
    _require_hash(work.get("base_adapter_hash"), "volunteer_work_base_adapter_hash_invalid")
    _require_hash(work.get("dataset_shard_hash"), "volunteer_work_dataset_shard_hash_invalid")
    if int(work.get("local_steps") or 0) < 1:
        raise VolunteerProtocolError("volunteer_work_local_steps_invalid", status_code=400)
    if int(work.get("lease_generation") or 0) < 1:
        raise VolunteerProtocolError("volunteer_work_lease_generation_invalid", status_code=400)
    if now is not None and float(work.get("lease_expires_at") or 0.0) <= float(now):
        raise VolunteerProtocolError("volunteer_work_lease_expired", status_code=409)
    expected_hash = work_unit_content_hash(work)
    if work.get("work_unit_hash") != expected_hash:
        raise VolunteerProtocolError("volunteer_work_unit_hash_mismatch", status_code=400)
    return with_public_safety(work)


def encode_submission_envelope(metadata: dict[str, Any], delta: bytes) -> bytes:
    header = canonical_json(metadata)
    if len(header) > MAX_SUBMISSION_METADATA_BYTES:
        raise VolunteerProtocolError("volunteer_submission_metadata_too_large", status_code=413)
    if not delta:
        raise VolunteerProtocolError("volunteer_submission_delta_empty", status_code=400)
    return struct.pack(">Q", len(header)) + header + bytes(delta)


def decode_submission_envelope(
    value: bytes,
    *,
    max_delta_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    if len(value) < 9:
        raise VolunteerProtocolError("volunteer_submission_envelope_truncated", status_code=400)
    header_size = int(struct.unpack(">Q", value[:8])[0])
    if header_size < 2 or header_size > MAX_SUBMISSION_METADATA_BYTES:
        raise VolunteerProtocolError("volunteer_submission_metadata_size_invalid", status_code=413)
    delta_start = 8 + header_size
    if delta_start >= len(value):
        raise VolunteerProtocolError("volunteer_submission_delta_empty", status_code=400)
    delta = value[delta_start:]
    if len(delta) > int(max_delta_bytes):
        raise VolunteerProtocolError("volunteer_submission_delta_too_large", status_code=413)
    try:
        metadata = json.loads(value[8:delta_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VolunteerProtocolError("volunteer_submission_metadata_invalid", status_code=400) from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != SUBMISSION_SCHEMA:
        raise VolunteerProtocolError("volunteer_submission_schema_mismatch", status_code=400)
    return metadata, delta


def encode_json_header(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(canonical_json(value)).decode("ascii")


def decode_json_header(value: str) -> dict[str, Any]:
    try:
        decoded = base64.urlsafe_b64decode(str(value).encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise VolunteerProtocolError("volunteer_header_invalid", status_code=400) from exc
    if not isinstance(payload, dict):
        raise VolunteerProtocolError("volunteer_header_invalid", status_code=400)
    return payload


def public_error(code: str) -> dict[str, Any]:
    return with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_error_v1",
            "ok": False,
            "error": str(code),
        }
    )
