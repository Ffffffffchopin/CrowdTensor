"""Framework-neutral contracts for reviewable community data contributions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import ContractError, stable_hash


DATA_PACK_SCHEMA = "crowdtensor_data_pack_v1"
DATA_PACK_FORMATS = frozenset({"instruction_sft_jsonl_v1"})
DATA_SOURCE_KINDS = frozenset(
    {"contributor_authored", "permissive_source", "generated_with_provenance"}
)
DATA_MODERATION_STATES = frozenset({"pending", "approved", "rejected"})

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
_SPDX = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{1,63}\Z")
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*\Z")


class DataPackError(ContractError):
    """Raised when a Data Pack or its review boundary is malformed."""


def _text(value: Any, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise DataPackError(f"data_pack_{field}_must_be_string")
    result = value.strip()
    if not result or len(result) > maximum:
        raise DataPackError(f"data_pack_{field}_invalid")
    return result


def _hash(value: Any, field: str) -> str:
    result = _text(value, field, maximum=71)
    if not _HASH.fullmatch(result):
        raise DataPackError(f"data_pack_{field}_invalid")
    return result


def _strings(
    values: Any,
    field: str,
    *,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise DataPackError(f"data_pack_{field}_list_required")
    result = tuple(sorted({_text(item, field, maximum=512) for item in values}))
    if not result and not allow_empty:
        raise DataPackError(f"data_pack_{field}_required")
    if pattern is not None and any(not pattern.fullmatch(item) for item in result):
        raise DataPackError(f"data_pack_{field}_invalid")
    return result


@dataclass(frozen=True)
class DataPack:
    """A content-addressed data contribution without embedded raw records."""

    pack_id: str
    data_format: str
    license_spdx: str
    source_kind: str
    source_revision: str
    source_uris: tuple[str, ...]
    records_hash: str
    record_count: int
    byte_count: int
    languages: tuple[str, ...]
    domains: tuple[str, ...]
    contributor_id_hash: str
    redistribution_allowed: bool
    training_allowed: bool
    personal_data_reviewed: bool
    copyright_reviewed: bool
    benchmark_contamination_reviewed: bool
    moderation_status: str
    public_records: bool = False

    def __post_init__(self) -> None:
        pack_id = _text(self.pack_id, "pack_id", maximum=128)
        if not _IDENTIFIER.fullmatch(pack_id):
            raise DataPackError("data_pack_pack_id_invalid")
        object.__setattr__(self, "pack_id", pack_id)

        data_format = _text(self.data_format, "format", maximum=64)
        if data_format not in DATA_PACK_FORMATS:
            raise DataPackError("data_pack_format_unsupported")
        object.__setattr__(self, "data_format", data_format)

        license_spdx = _text(self.license_spdx, "license_spdx", maximum=64)
        if not _SPDX.fullmatch(license_spdx):
            raise DataPackError("data_pack_license_spdx_invalid")
        object.__setattr__(self, "license_spdx", license_spdx)

        source_kind = _text(self.source_kind, "source_kind", maximum=64)
        if source_kind not in DATA_SOURCE_KINDS:
            raise DataPackError("data_pack_source_kind_unsupported")
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(
            self, "source_revision", _text(self.source_revision, "source_revision")
        )

        uris = _strings(self.source_uris, "source_uris", allow_empty=True)
        if any(not item.startswith(("https://", "urn:")) for item in uris):
            raise DataPackError("data_pack_source_uris_invalid")
        object.__setattr__(self, "source_uris", uris)
        object.__setattr__(self, "records_hash", _hash(self.records_hash, "records_hash"))
        object.__setattr__(
            self,
            "contributor_id_hash",
            _hash(self.contributor_id_hash, "contributor_id_hash"),
        )

        for field in ("record_count", "byte_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DataPackError(f"data_pack_{field}_invalid")

        object.__setattr__(
            self, "languages", _strings(self.languages, "languages", pattern=_LANGUAGE)
        )
        object.__setattr__(
            self, "domains", _strings(self.domains, "domains", pattern=_IDENTIFIER)
        )
        for field in (
            "redistribution_allowed",
            "training_allowed",
            "personal_data_reviewed",
            "copyright_reviewed",
            "benchmark_contamination_reviewed",
            "public_records",
        ):
            if not isinstance(getattr(self, field), bool):
                raise DataPackError(f"data_pack_{field}_boolean_required")

        moderation = _text(self.moderation_status, "moderation_status", maximum=32)
        if moderation not in DATA_MODERATION_STATES:
            raise DataPackError("data_pack_moderation_status_invalid")
        object.__setattr__(self, "moderation_status", moderation)

    @property
    def admission_ready(self) -> bool:
        return bool(
            self.redistribution_allowed
            and self.training_allowed
            and self.personal_data_reviewed
            and self.copyright_reviewed
            and self.benchmark_contamination_reviewed
            and self.moderation_status == "approved"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DATA_PACK_SCHEMA,
            "pack_id": self.pack_id,
            "format": self.data_format,
            "license_spdx": self.license_spdx,
            "source_kind": self.source_kind,
            "source_revision": self.source_revision,
            "source_uris": list(self.source_uris),
            "records_hash": self.records_hash,
            "record_count": self.record_count,
            "byte_count": self.byte_count,
            "languages": list(self.languages),
            "domains": list(self.domains),
            "contributor_id_hash": self.contributor_id_hash,
            "redistribution_allowed": self.redistribution_allowed,
            "training_allowed": self.training_allowed,
            "personal_data_reviewed": self.personal_data_reviewed,
            "copyright_reviewed": self.copyright_reviewed,
            "benchmark_contamination_reviewed": self.benchmark_contamination_reviewed,
            "moderation_status": self.moderation_status,
            "public_records": self.public_records,
            "admission_ready": self.admission_ready,
            "raw_records_in_manifest": False,
            "public_artifact_safe": True,
        }
        payload["content_hash"] = stable_hash(payload)
        return payload

    @property
    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataPack":
        if not isinstance(value, Mapping) or value.get("schema") != DATA_PACK_SCHEMA:
            raise DataPackError("data_pack_schema_mismatch")
        expected = {
            "schema",
            "pack_id",
            "format",
            "license_spdx",
            "source_kind",
            "source_revision",
            "source_uris",
            "records_hash",
            "record_count",
            "byte_count",
            "languages",
            "domains",
            "contributor_id_hash",
            "redistribution_allowed",
            "training_allowed",
            "personal_data_reviewed",
            "copyright_reviewed",
            "benchmark_contamination_reviewed",
            "moderation_status",
            "public_records",
            "admission_ready",
            "raw_records_in_manifest",
            "public_artifact_safe",
            "content_hash",
        }
        if set(value) != expected:
            raise DataPackError("data_pack_fields_invalid")
        supplied_hash = _hash(value.get("content_hash"), "content_hash")
        unsigned = {key: item for key, item in value.items() if key != "content_hash"}
        if stable_hash(unsigned) != supplied_hash:
            raise DataPackError("data_pack_content_hash_mismatch")
        if value.get("raw_records_in_manifest") is not False:
            raise DataPackError("data_pack_raw_records_in_manifest")
        if value.get("public_artifact_safe") is not True:
            raise DataPackError("data_pack_public_safety_invalid")
        pack = cls(
            pack_id=value["pack_id"],
            data_format=value["format"],
            license_spdx=value["license_spdx"],
            source_kind=value["source_kind"],
            source_revision=value["source_revision"],
            source_uris=tuple(value["source_uris"]),
            records_hash=value["records_hash"],
            record_count=value["record_count"],
            byte_count=value["byte_count"],
            languages=tuple(value["languages"]),
            domains=tuple(value["domains"]),
            contributor_id_hash=value["contributor_id_hash"],
            redistribution_allowed=value["redistribution_allowed"],
            training_allowed=value["training_allowed"],
            personal_data_reviewed=value["personal_data_reviewed"],
            copyright_reviewed=value["copyright_reviewed"],
            benchmark_contamination_reviewed=value[
                "benchmark_contamination_reviewed"
            ],
            moderation_status=value["moderation_status"],
            public_records=value["public_records"],
        )
        if value.get("admission_ready") is not pack.admission_ready:
            raise DataPackError("data_pack_admission_state_mismatch")
        return pack
