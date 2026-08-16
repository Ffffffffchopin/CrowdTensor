"""Model-neutral text and community Data Pack preparation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from crowdtensor.core.contracts import stable_hash
from crowdtensor.core.data_packs import DataPack, DataPackError


DATA_PACK_MANIFEST_FILE = "data-pack.json"
DATA_PACK_RECORDS_FILE = "records.jsonl"
DATA_PACK_CREATE_RESULT_SCHEMA = "crowdtensor_data_pack_create_result_v1"
DATA_PACK_VALIDATION_SCHEMA = "crowdtensor_data_pack_validation_v1"

_RECORD_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?\Z")
_SENSITIVE_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pack_root(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    return path.parent if path.name == DATA_PACK_MANIFEST_FILE else path


def _field(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise DataPackError(f"data_pack_record_{name}_must_be_string")
    result = unicodedata.normalize("NFC", value).strip()
    if not result or len(result) > maximum or "\x00" in result:
        raise DataPackError(f"data_pack_record_{name}_invalid")
    return result


def _canonical_instruction_record(
    value: Any, *, default_language: str
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise DataPackError("data_pack_record_object_required")
    unknown = set(value) - {
        "record_id",
        "prompt",
        "response",
        "language",
        "source_ref",
    }
    if unknown:
        raise DataPackError("data_pack_record_unknown_fields")
    prompt = _field(value.get("prompt"), "prompt", maximum=32_768)
    response = _field(value.get("response"), "response", maximum=131_072)
    combined = prompt + "\n" + response
    if any(pattern.search(combined) for pattern in _SENSITIVE_PATTERNS):
        raise DataPackError("data_pack_record_obvious_secret_detected")
    record_id = str(value.get("record_id") or "").strip()
    if not record_id:
        record_id = "record-" + hashlib.sha256(
            (prompt + "\x00" + response).encode("utf-8")
        ).hexdigest()[:24]
    if not _RECORD_ID.fullmatch(record_id):
        raise DataPackError("data_pack_record_id_invalid")
    language = str(value.get("language") or default_language).strip().lower()
    result = {
        "record_id": record_id,
        "prompt": prompt,
        "response": response,
        "language": language,
    }
    source_ref = str(value.get("source_ref") or "").strip()
    if source_ref:
        if len(source_ref) > 512 or not source_ref.startswith(("https://", "urn:")):
            raise DataPackError("data_pack_record_source_ref_invalid")
        result["source_ref"] = source_ref
    return result


def _canonical_records(
    source: str | Path, *, languages: tuple[str, ...]
) -> tuple[list[dict[str, str]], bytes]:
    source_path = Path(source).expanduser()
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(line.encode("utf-8")) > 512 * 1024:
                    raise DataPackError("data_pack_record_line_too_large")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataPackError(
                        f"data_pack_record_json_invalid:{line_number}"
                    ) from exc
                record = _canonical_instruction_record(
                    raw, default_language=languages[0]
                )
                if record["language"] not in languages:
                    raise DataPackError("data_pack_record_language_not_declared")
                if record["record_id"] in seen:
                    raise DataPackError("data_pack_record_id_duplicate")
                seen.add(record["record_id"])
                records.append(record)
    except (OSError, UnicodeError) as exc:
        raise DataPackError("data_pack_records_unreadable") from exc
    if not records:
        raise DataPackError("data_pack_records_empty")
    encoded = b"".join(
        (
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    return records, encoded


def _result(pack: DataPack, *, created: bool) -> dict[str, Any]:
    report = {
        "schema": DATA_PACK_CREATE_RESULT_SCHEMA,
        "ok": True,
        "created": bool(created),
        "pack_id": pack.pack_id,
        "data_pack_hash": pack.content_hash,
        "records_hash": pack.records_hash,
        "record_count": pack.record_count,
        "admission_ready": pack.admission_ready,
        "raw_records_public": pack.public_records,
        "private_paths_public": False,
        "contributor_identity_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def create_instruction_data_pack(
    source: str | Path,
    output_dir: str | Path,
    *,
    pack_id: str,
    license_spdx: str,
    source_kind: str,
    languages: tuple[str, ...],
    domains: tuple[str, ...],
    contributor_id: str,
    source_revision: str = "",
    source_uris: tuple[str, ...] = (),
    redistribution_allowed: bool = False,
    training_allowed: bool = False,
    personal_data_reviewed: bool = False,
    copyright_reviewed: bool = False,
    benchmark_contamination_reviewed: bool = False,
    moderation_status: str = "pending",
    public_records: bool = False,
) -> dict[str, Any]:
    """Create an immutable instruction Data Pack from canonical JSONL records."""

    normalized_languages = tuple(sorted({str(item).strip().lower() for item in languages}))
    if not normalized_languages:
        raise DataPackError("data_pack_languages_required")
    normalized_contributor_id = str(contributor_id).strip()
    if not normalized_contributor_id:
        raise DataPackError("data_pack_contributor_id_required")
    records, encoded = _canonical_records(source, languages=normalized_languages)
    records_hash = _sha256_bytes(encoded)
    pack = DataPack(
        pack_id=pack_id,
        data_format="instruction_sft_jsonl_v1",
        license_spdx=license_spdx,
        source_kind=source_kind,
        source_revision=source_revision or records_hash,
        source_uris=tuple(source_uris),
        records_hash=records_hash,
        record_count=len(records),
        byte_count=len(encoded),
        languages=normalized_languages,
        domains=tuple(domains),
        contributor_id_hash=stable_hash(normalized_contributor_id),
        redistribution_allowed=redistribution_allowed,
        training_allowed=training_allowed,
        personal_data_reviewed=personal_data_reviewed,
        copyright_reviewed=copyright_reviewed,
        benchmark_contamination_reviewed=benchmark_contamination_reviewed,
        moderation_status=moderation_status,
        public_records=public_records,
    )
    root = _pack_root(output_dir)
    manifest_bytes = (
        json.dumps(pack.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    manifest_path = root / DATA_PACK_MANIFEST_FILE
    records_path = root / DATA_PACK_RECORDS_FILE
    if root.exists():
        if (
            manifest_path.is_file()
            and records_path.is_file()
            and manifest_path.read_bytes() == manifest_bytes
            and records_path.read_bytes() == encoded
        ):
            validate_instruction_data_pack(root)
            return _result(pack, created=False)
        raise DataPackError("data_pack_output_conflict")
    root.mkdir(parents=True)
    _atomic_write(records_path, encoded, mode=0o600)
    _atomic_write(manifest_path, manifest_bytes, mode=0o644)
    validate_instruction_data_pack(root)
    return _result(pack, created=True)


def load_data_pack(value: str | Path) -> DataPack:
    root = _pack_root(value)
    try:
        manifest = json.loads(
            (root / DATA_PACK_MANIFEST_FILE).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataPackError("data_pack_manifest_unreadable") from exc
    return DataPack.from_dict(manifest)


def validate_instruction_data_pack(value: str | Path) -> dict[str, Any]:
    root = _pack_root(value)
    pack = load_data_pack(root)
    records_path = root / DATA_PACK_RECORDS_FILE
    records, encoded = _canonical_records(records_path, languages=pack.languages)
    errors: list[str] = []
    if _sha256_bytes(encoded) != pack.records_hash:
        errors.append("data_pack_records_hash_mismatch")
    if len(encoded) != pack.byte_count:
        errors.append("data_pack_records_byte_count_mismatch")
    if len(records) != pack.record_count:
        errors.append("data_pack_records_count_mismatch")
    if records_path.read_bytes() != encoded:
        errors.append("data_pack_records_not_canonical")
    report = {
        "schema": DATA_PACK_VALIDATION_SCHEMA,
        "ok": not errors,
        "pack_id": pack.pack_id,
        "data_pack_hash": pack.content_hash,
        "records_hash": pack.records_hash,
        "record_count": pack.record_count,
        "records_verified": not errors,
        "admission_ready": pack.admission_ready and not errors,
        "moderation_status": pack.moderation_status,
        "errors": sorted(errors),
        "raw_records_in_report": False,
        "private_paths_public": False,
        "contributor_identity_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def load_instruction_records(value: str | Path) -> list[dict[str, str]]:
    root = _pack_root(value)
    report = validate_instruction_data_pack(root)
    if report["ok"] is not True:
        raise DataPackError("data_pack_records_integrity_failed")
    pack = load_data_pack(root)
    records, _encoded = _canonical_records(
        root / DATA_PACK_RECORDS_FILE, languages=pack.languages
    )
    return records


def tokenize_instruction_records(
    records: list[dict[str, str]],
    tokenizer: Any,
    *,
    sequence_length: int,
    sequence_count: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create deterministic response-supervised rows with prompt labels masked."""

    length = int(sequence_length)
    count = int(sequence_count) if int(sequence_count) > 0 else len(records)
    if length < 32 or length > 4096:
        raise ValueError("instruction_sequence_length_out_of_bounds")
    if count < 1 or count > len(records):
        raise ValueError("instruction_sequence_count_out_of_bounds")
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    bos = getattr(tokenizer, "bos_token_id", None)
    if eos is None:
        raise ValueError("instruction_tokenizer_eos_required")
    if pad is None:
        pad = eos

    rows: list[dict[str, Any]] = []
    record_ids: list[str] = []
    for record in records[:count]:
        prefix = (
            "### Instruction:\n"
            + record["prompt"]
            + "\n\n### Response:\n"
        )
        prefix_ids = [int(item) for item in tokenizer.encode(
            prefix, add_special_tokens=False
        )]
        response_ids = [int(item) for item in tokenizer.encode(
            record["response"], add_special_tokens=False
        )]
        if not response_ids:
            raise ValueError("instruction_response_tokenization_empty")
        if response_ids[-1] != int(eos):
            response_ids.append(int(eos))
        leading = [int(bos)] if bos is not None else []
        prompt_budget = max(1, min(len(prefix_ids), length // 2))
        response_budget = length - len(leading) - prompt_budget
        if response_budget < 1:
            raise ValueError("instruction_sequence_has_no_response_budget")
        selected_prefix = prefix_ids[:prompt_budget]
        selected_response = response_ids[:response_budget]
        input_ids = leading + selected_prefix + selected_response
        labels = [-100] * (len(leading) + len(selected_prefix)) + selected_response
        attention_mask = [1] * len(input_ids)
        padding = length - len(input_ids)
        input_ids.extend([int(pad)] * padding)
        labels.extend([-100] * padding)
        attention_mask.extend([0] * padding)
        if not any(item != -100 for item in labels):
            raise ValueError("instruction_sequence_has_no_supervised_tokens")
        rows.append(
            {
                "sample_id": record["record_id"],
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )
        record_ids.append(record["record_id"])
    return rows, record_ids


def tokenize_fixed_sequences(
    texts: list[Any],
    tokenizer: Any,
    *,
    sequence_length: int,
    sequence_count: int,
) -> tuple[list[list[int]], list[int]]:
    """Tokenize a bounded prefix into deterministic fixed-length sequences."""

    length = int(sequence_length)
    count = int(sequence_count)
    if length <= 0 or count <= 0:
        raise ValueError("fixed_sequence_shape_invalid")
    required = length * count
    tokens: list[int] = []
    row_indexes: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("tokenizer_eos_token_required")
    for index, value in enumerate(texts):
        text = str(value or "").strip()
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        row_indexes.append(index)
        tokens.extend(int(token) for token in encoded)
        tokens.append(int(eos))
        if len(tokens) >= required:
            break
    if len(tokens) < required:
        raise RuntimeError("text_split_did_not_provide_enough_fixed_tokens")
    rows = [
        tokens[offset : offset + length]
        for offset in range(0, required, length)
    ]
    return rows, row_indexes
