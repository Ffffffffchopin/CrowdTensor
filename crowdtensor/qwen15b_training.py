"""Qwen 1.5B source, stage ownership, and stage-selective loading helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MODEL_FILENAME = "model.safetensors"
MODEL_PARAMETER_COUNT = 1_543_714_304
MODEL_TENSOR_BYTES = 3_087_428_608
MODEL_FILE_SIZE = 3_087_467_144
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-2-raw-v1"
SOURCE_SCHEMA = "crowdtensor_qwen15b_training_source_v1"
STAGE_INDEX_SCHEMA = "crowdtensor_qwen15b_stage_index_v1"
STAGE_SHARD_SCHEMA = "crowdtensor_qwen15b_stage_shard_v1"
LAYER_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _hf_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def fetch_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "crowdtensor-qwen15b-training/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, *, timeout: float = 120.0) -> dict[str, Any]:
    value = json.loads(fetch_bytes(url, timeout=timeout))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {url}")
    return value


def fetch_range(url: str, start: int, end: int, *, timeout: float = 180.0) -> bytes:
    if start < 0 or end < start:
        raise ValueError("invalid HTTP byte range")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={int(start)}-{int(end)}",
            "User-Agent": "crowdtensor-qwen15b-stage-loader/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(int(end - start + 2))
        status = int(getattr(response, "status", 0) or 0)
        content_range = str(response.headers.get("Content-Range") or "")
    if len(payload) != end - start + 1:
        raise RuntimeError("stage-selective range response length mismatch")
    if status != 206 and not content_range:
        raise RuntimeError("stage-selective HTTP source ignored byte range")
    return payload


def fetch_safetensors_header(
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    filename: str = MODEL_FILENAME,
    range_reader: Callable[[str, int, int], bytes] | None = None,
) -> tuple[int, dict[str, Any]]:
    url = _hf_url(model_id, revision, filename)
    reader = range_reader or (lambda target, start, end: fetch_range(target, start, end))
    prefix = reader(url, 0, 7)
    if len(prefix) != 8:
        raise RuntimeError("safetensors header prefix missing")
    header_length = int(struct.unpack("<Q", prefix)[0])
    if header_length <= 0 or header_length > 64 * 1024 * 1024:
        raise RuntimeError("safetensors header length outside safety bound")
    header = json.loads(reader(url, 8, 7 + header_length).decode("utf-8"))
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header is not an object")
    return header_length, header


def read_safetensors_header(path: str | Path) -> tuple[int, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError("safetensors header prefix missing")
        header_length = int(struct.unpack("<Q", prefix)[0])
        header = json.loads(handle.read(header_length).decode("utf-8"))
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header is not an object")
    return header_length, header


def tensor_entries(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(name): dict(meta)
        for name, meta in header.items()
        if name != "__metadata__" and isinstance(meta, dict)
    }


def _dtype_nbytes(dtype: str) -> int:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "U16": 2,
        "I16": 2,
        "F16": 2,
        "BF16": 2,
        "U32": 4,
        "I32": 4,
        "F32": 4,
        "U64": 8,
        "I64": 8,
        "F64": 8,
    }
    try:
        return sizes[str(dtype).upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype: {dtype}") from exc


def tensor_numel(meta: dict[str, Any]) -> int:
    shape = [int(value) for value in meta.get("shape") or []]
    return int(math.prod(shape)) if shape else 1


def tensor_byte_count(meta: dict[str, Any]) -> int:
    offsets = [int(value) for value in meta.get("data_offsets") or []]
    if len(offsets) == 2:
        return offsets[1] - offsets[0]
    return tensor_numel(meta) * _dtype_nbytes(str(meta.get("dtype") or ""))


def build_weight_index(
    header: dict[str, Any],
    *,
    filename: str = MODEL_FILENAME,
) -> dict[str, Any]:
    entries = tensor_entries(header)
    index = {
        "schema": STAGE_INDEX_SCHEMA,
        "metadata": {
            "total_size": sum(tensor_byte_count(meta) for meta in entries.values()),
            "parameter_count": sum(tensor_numel(meta) for meta in entries.values()),
            "tensor_count": len(entries),
            "generated_from_single_safetensors_header": True,
            "official_index_present": False,
        },
        "weight_map": {name: filename for name in sorted(entries)},
    }
    index["content_hash"] = stable_hash(index)
    return index


@dataclass(frozen=True)
class QwenStageSpec:
    stage_id: int
    kernel_slot: str
    device_index: int
    layer_start: int
    layer_end: int
    owns_embedding: bool = False
    owns_norm: bool = False
    owns_lm_head: bool = False

    @property
    def device(self) -> str:
        return f"cuda:{self.device_index}"

    def public_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "kernel_slot": self.kernel_slot,
            "device": self.device,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "layer_count": self.layer_end - self.layer_start,
            "owns_embedding": self.owns_embedding,
            "owns_norm": self.owns_norm,
            "owns_lm_head": self.owns_lm_head,
        }


def canonical_stage_specs(layer_count: int = 28) -> list[QwenStageSpec]:
    if int(layer_count) != 28:
        raise ValueError("Qwen 1.5B Alpha requires exactly 28 decoder layers")
    return [
        QwenStageSpec(0, "A", 0, 0, 7, owns_embedding=True),
        QwenStageSpec(1, "A", 1, 7, 14),
        QwenStageSpec(2, "B", 0, 14, 21),
        QwenStageSpec(3, "B", 1, 21, 28, owns_norm=True, owns_lm_head=True),
    ]


def select_stage_source_keys(
    header: dict[str, Any],
    spec: QwenStageSpec,
) -> tuple[list[str], dict[str, str]]:
    entries = tensor_entries(header)
    selected: set[str] = set()
    for name in entries:
        match = LAYER_KEY_PATTERN.match(name)
        if match and spec.layer_start <= int(match.group(1)) < spec.layer_end:
            selected.add(name)
    aliases: dict[str, str] = {}
    if spec.owns_embedding:
        selected.add("model.embed_tokens.weight")
    if spec.owns_norm:
        selected.add("model.norm.weight")
    if spec.owns_lm_head:
        if "lm_head.weight" in entries:
            selected.add("lm_head.weight")
        else:
            selected.add("model.embed_tokens.weight")
            aliases["lm_head.weight"] = "model.embed_tokens.weight"
    missing = sorted(name for name in selected if name not in entries)
    if missing:
        raise RuntimeError(f"stage source tensors missing: {missing}")
    return sorted(selected), aliases


def build_stage_ownership(
    config: dict[str, Any],
    header: dict[str, Any],
) -> dict[str, Any]:
    layer_count = int(config.get("num_hidden_layers") or 0)
    specs = canonical_stage_specs(layer_count)
    entries = tensor_entries(header)
    stage_records: list[dict[str, Any]] = []
    key_owners: dict[str, list[int]] = {}
    for spec in specs:
        keys, aliases = select_stage_source_keys(header, spec)
        for key in keys:
            key_owners.setdefault(key, []).append(spec.stage_id)
        stage_records.append(
            {
                **spec.public_dict(),
                "source_tensor_count": len(keys),
                "source_parameter_count": sum(tensor_numel(entries[key]) for key in keys),
                "source_byte_count": sum(tensor_byte_count(entries[key]) for key in keys),
                "source_keys_hash": stable_hash(keys),
                "aliases": aliases,
            }
        )
    uncovered = sorted(set(entries) - set(key_owners))
    duplicates = {key: owners for key, owners in key_owners.items() if len(owners) > 1}
    allowed_tied = {"model.embed_tokens.weight": [0, 3]}
    report = {
        "schema": "crowdtensor_qwen15b_four_stage_ownership_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_count": layer_count,
        "stages": stage_records,
        "all_source_tensors_covered": not uncovered,
        "uncovered_source_keys": uncovered,
        "duplicate_source_key_owners": duplicates,
        "only_tied_embedding_lm_head_duplicated": duplicates == allowed_tied,
        "embedding_owner_stage": 0,
        "norm_owner_stage": 3,
        "lm_head_owner_stage": 3,
        "four_distinct_kernel_device_placements": len(
            {(item.kernel_slot, item.device_index) for item in specs}
        )
        == 4,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _tokenizer_hashes(model_id: str, revision: str) -> dict[str, Any]:
    files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    digest = hashlib.sha256()
    for filename in files:
        payload = fetch_bytes(_hf_url(model_id, revision, filename))
        hashes[filename] = sha256_bytes(payload)
        sizes[filename] = len(payload)
        digest.update(filename.encode("utf-8") + b"\0" + payload)
    return {
        "files": hashes,
        "file_sizes": sizes,
        "tokenizer_hash": "sha256:" + digest.hexdigest(),
    }


def _tree_entries(repo_kind: str, repo: str, revision: str, path: str = "") -> list[dict[str, Any]]:
    prefix = "datasets/" if repo_kind == "dataset" else "models/"
    suffix = f"/{path}" if path else ""
    url = (
        f"https://huggingface.co/api/{prefix}{repo}/tree/{revision}{suffix}"
        "?recursive=true&expand=true"
    )
    value = json.loads(fetch_bytes(url))
    if not isinstance(value, list):
        raise RuntimeError("Hugging Face tree response was not a list")
    return [dict(item) for item in value if isinstance(item, dict)]


def _lfs_file_entry(entries: list[dict[str, Any]], path: str) -> dict[str, Any]:
    for item in entries:
        if item.get("type") == "file" and item.get("path") == path:
            lfs = dict(item.get("lfs") or {})
            return {
                "path": path,
                "size": int(item.get("size") or lfs.get("size") or 0),
                "lfs_sha256": "sha256:" + str(lfs.get("oid") or ""),
                "xet_hash": str(item.get("xetHash") or ""),
            }
    raise RuntimeError(f"required Hugging Face file missing: {path}")


def resolve_dataset_source() -> dict[str, Any]:
    api = fetch_json(f"https://huggingface.co/api/datasets/{DATASET_ID}")
    if str(api.get("sha") or "") != DATASET_REVISION:
        raise RuntimeError("WikiText source revision changed from pinned Alpha revision")
    entries = _tree_entries("dataset", DATASET_ID, DATASET_REVISION, DATASET_CONFIG)
    train = _lfs_file_entry(
        entries,
        f"{DATASET_CONFIG}/train-00000-of-00001.parquet",
    )
    validation = _lfs_file_entry(
        entries,
        f"{DATASET_CONFIG}/validation-00000-of-00001.parquet",
    )
    report = {
        "schema": "crowdtensor_qwen15b_wikitext_source_v1",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": DATASET_CONFIG,
        "license": list((api.get("cardData") or {}).get("license") or []),
        "gated": api.get("gated") is True,
        "private": api.get("private") is True,
        "train": train,
        "validation": validation,
        "raw_text_public": False,
        "token_ids_public": False,
        "public_artifact_safe": True,
    }
    report["source_verified"] = bool(
        report["dataset_revision"] == DATASET_REVISION
        and train["size"] == 6_357_543
        and train["lfs_sha256"]
        == "sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
        and validation["size"] == 657_209
        and validation["lfs_sha256"]
        == "sha256:204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"
        and report["gated"] is False
        and report["private"] is False
    )
    report["content_hash"] = stable_hash(report)
    return report


def resolve_source_manifest() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    api = fetch_json(f"https://huggingface.co/api/models/{MODEL_ID}")
    revision = str(api.get("sha") or "")
    if revision != MODEL_REVISION:
        raise RuntimeError("Qwen 1.5B source revision changed from pinned Alpha revision")
    config_bytes = fetch_bytes(_hf_url(MODEL_ID, MODEL_REVISION, "config.json"))
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise RuntimeError("Qwen config was not an object")
    header_length, header = fetch_safetensors_header()
    index = build_weight_index(header)
    metadata = dict(index["metadata"])
    tokenizer = _tokenizer_hashes(MODEL_ID, MODEL_REVISION)
    model_tree = _tree_entries("model", MODEL_ID, MODEL_REVISION)
    model_file = _lfs_file_entry(model_tree, MODEL_FILENAME)
    dataset = resolve_dataset_source()
    source = {
        "schema": SOURCE_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_type": config.get("model_type"),
        "architectures": list(config.get("architectures") or []),
        "license": (api.get("cardData") or {}).get("license"),
        "gated": api.get("gated") is True,
        "private": api.get("private") is True,
        "parameter_count": int(metadata.get("parameter_count") or 0),
        "tensor_count": int(metadata.get("tensor_count") or 0),
        "weight_bytes": int(metadata.get("total_size") or 0),
        "model_file_size": model_file["size"],
        "model_file_lfs_sha256": model_file["lfs_sha256"],
        "model_file_xet_hash": model_file["xet_hash"],
        "config_hash": sha256_bytes(config_bytes),
        "safetensors_header_hash": stable_hash(header),
        "safetensors_header_length": header_length,
        "weight_index_hash": index["content_hash"],
        "weight_index_generated_from_header": True,
        "official_weight_index_present": False,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "public_source": True,
        "raw_text_public": False,
        "token_ids_public": False,
        "public_artifact_safe": True,
    }
    source["source_verified"] = bool(
        source["model_id"] == MODEL_ID
        and source["model_revision"] == MODEL_REVISION
        and source["model_type"] == "qwen2"
        and source["architectures"] == ["Qwen2ForCausalLM"]
        and source["license"] == "apache-2.0"
        and source["parameter_count"] == MODEL_PARAMETER_COUNT
        and source["weight_bytes"] == MODEL_TENSOR_BYTES
        and source["model_file_size"] == MODEL_FILE_SIZE
        and source["model_file_lfs_sha256"]
        == "sha256:a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
        and dataset["source_verified"] is True
        and source["gated"] is False
        and source["private"] is False
    )
    source["content_hash"] = stable_hash(source)
    return source, config, header


def _dataset_file_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{path}"


def _tokenize_split(
    texts: list[Any],
    tokenizer: Any,
    *,
    sequence_length: int,
    sequence_count: int,
) -> tuple[list[list[int]], list[int]]:
    required = int(sequence_length) * int(sequence_count)
    tokens: list[int] = []
    row_indexes: list[int] = []
    eos = int(tokenizer.eos_token_id)
    for index, value in enumerate(texts):
        text = str(value or "").strip()
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        row_indexes.append(index)
        tokens.extend(int(token) for token in encoded)
        tokens.append(eos)
        if len(tokens) >= required:
            break
    if len(tokens) < required:
        raise RuntimeError("WikiText split did not provide enough fixed tokens")
    rows = [
        tokens[offset : offset + int(sequence_length)]
        for offset in range(0, required, int(sequence_length))
    ]
    return rows, row_indexes


def prepare_tokenized_wikitext(
    output_dir: str | Path,
    *,
    sequence_length: int = 64,
    train_sequence_count: int = 32,
    validation_sequence_count: int = 8,
) -> dict[str, Any]:
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    if int(sequence_length) < 32 or int(sequence_length) > 256:
        raise ValueError("Qwen 1.5B Alpha sequence length must be in [32, 256]")
    if int(train_sequence_count) < 32:
        raise ValueError("Qwen 1.5B Alpha needs at least 32 training sequences")
    if int(validation_sequence_count) < 4:
        raise ValueError("Qwen 1.5B Alpha needs at least four validation sequences")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / ".private-raw-wikitext"
    cache_dir = output / ".private-tokenizer-cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "train": (
            f"{DATASET_CONFIG}/train-00000-of-00001.parquet",
            "sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
        ),
        "validation": (
            f"{DATASET_CONFIG}/validation-00000-of-00001.parquet",
            "sha256:204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
        ),
    }
    local_files: dict[str, Path] = {}
    try:
        for split, (remote_path, expected_hash) in files.items():
            payload = fetch_bytes(_dataset_file_url(remote_path), timeout=180.0)
            if sha256_bytes(payload) != expected_hash:
                raise RuntimeError(f"pinned WikiText {split} parquet hash mismatch")
            path = raw_dir / f"{split}.parquet"
            path.write_bytes(payload)
            local_files[split] = path
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=False,
            trust_remote_code=False,
        )
        train_texts = parquet.read_table(local_files["train"], columns=["text"])["text"].to_pylist()
        validation_texts = parquet.read_table(
            local_files["validation"], columns=["text"]
        )["text"].to_pylist()
        train_rows, train_indexes = _tokenize_split(
            train_texts,
            tokenizer,
            sequence_length=sequence_length,
            sequence_count=train_sequence_count,
        )
        validation_rows, validation_indexes = _tokenize_split(
            validation_texts,
            tokenizer,
            sequence_length=sequence_length,
            sequence_count=validation_sequence_count,
        )
        private_payload = {
            "schema": "crowdtensor_qwen15b_tokenized_private_v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "sequence_length": int(sequence_length),
            "train": train_rows,
            "validation": validation_rows,
        }
        private_path = output / "qwen15b_tokenized_private.json"
        private_path.write_text(
            json.dumps(private_payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        manifest = {
            "schema": "crowdtensor_qwen15b_tokenized_dataset_manifest_v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_config": DATASET_CONFIG,
            "sequence_length": int(sequence_length),
            "train_sequence_count": len(train_rows),
            "validation_sequence_count": len(validation_rows),
            "train_row_indexes": train_indexes,
            "validation_row_indexes": validation_indexes,
            "train_row_indexes_hash": stable_hash(train_indexes),
            "validation_row_indexes_hash": stable_hash(validation_indexes),
            "train_token_hash": stable_hash(train_rows),
            "validation_token_hash": stable_hash(validation_rows),
            "private_tokenized_payload_hash": sha256_file(private_path),
            "raw_text_public": False,
            "token_ids_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        manifest["content_hash"] = stable_hash(manifest)
        return {**manifest, "private_tokenized_path": str(private_path.resolve())}
    finally:
        import shutil

        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I32": torch.int32,
        "F32": torch.float32,
        "I64": torch.int64,
        "F64": torch.float64,
    }
    try:
        return mapping[str(name).upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported torch safetensors dtype: {name}") from exc


def _group_tensor_ranges(
    entries: dict[str, dict[str, Any]],
    keys: Iterable[str],
    *,
    data_start: int,
    max_group_bytes: int,
) -> list[tuple[int, int, list[str]]]:
    tensors = []
    for key in keys:
        offsets = [int(value) for value in entries[key].get("data_offsets") or []]
        if len(offsets) != 2 or offsets[1] <= offsets[0]:
            raise RuntimeError(f"invalid safetensors offsets for {key}")
        tensors.append((data_start + offsets[0], data_start + offsets[1] - 1, key))
    tensors.sort()
    groups: list[tuple[int, int, list[str]]] = []
    for start, end, key in tensors:
        if groups and start == groups[-1][1] + 1 and end - groups[-1][0] + 1 <= max_group_bytes:
            old_start, _, old_keys = groups[-1]
            groups[-1] = (old_start, end, [*old_keys, key])
        else:
            groups.append((start, end, [key]))
    return groups


def materialize_stage_shard(
    *,
    spec: QwenStageSpec,
    header_length: int,
    header: dict[str, Any],
    output_path: str | Path,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    filename: str = MODEL_FILENAME,
    range_reader: Callable[[str, int, int], bytes] | None = None,
    max_group_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    entries = tensor_entries(header)
    keys, aliases = select_stage_source_keys(header, spec)
    data_start = 8 + int(header_length)
    groups = _group_tensor_ranges(
        entries,
        keys,
        data_start=data_start,
        max_group_bytes=int(max_group_bytes),
    )
    url = _hf_url(model_id, revision, filename)
    reader = range_reader or (lambda target, start, end: fetch_range(target, start, end))
    tensors: dict[str, Any] = {}
    downloaded_bytes = 0
    for group_start, group_end, group_keys in groups:
        payload = reader(url, group_start, group_end)
        downloaded_bytes += len(payload)
        for key in group_keys:
            offsets = [int(value) for value in entries[key]["data_offsets"]]
            absolute_start = data_start + offsets[0]
            relative_start = absolute_start - group_start
            length = offsets[1] - offsets[0]
            raw = bytearray(payload[relative_start : relative_start + length])
            tensor = torch.frombuffer(raw, dtype=_torch_dtype(str(entries[key]["dtype"]))).clone()
            tensors[key] = tensor.reshape([int(value) for value in entries[key]["shape"]])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output)
    summary = {
        "schema": STAGE_SHARD_SCHEMA,
        **spec.public_dict(),
        "source_model_id": model_id,
        "source_revision": revision,
        "source_filename": filename,
        "source_keys_hash": stable_hash(keys),
        "source_tensor_count": len(keys),
        "source_parameter_count": sum(tensor_numel(entries[key]) for key in keys),
        "source_tensor_bytes": sum(tensor_byte_count(entries[key]) for key in keys),
        "downloaded_range_bytes": downloaded_bytes,
        "range_group_count": len(groups),
        "full_model_file_downloaded": False,
        "stage_selective_loading": True,
        "aliases": aliases,
        "shard_file_hash": sha256_file(output),
        "shard_byte_count": output.stat().st_size,
        "private_paths_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }
    summary["content_hash"] = stable_hash(summary)
    return {**summary, "shard_path": str(output.resolve())}


def materialize_stage_shard_from_layout(
    *,
    spec: QwenStageSpec,
    source_layout: dict[str, Any],
    output_path: str | Path,
    range_reader: Callable[[str, int, int], bytes] | None = None,
    max_group_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Materialize one stage from a pinned multi-file safetensors layout."""

    import torch
    from safetensors.torch import save_file

    model_id = str(source_layout.get("model_id") or "")
    revision = str(source_layout.get("model_revision") or "")
    weight_map = {
        str(name): str(filename)
        for name, filename in dict(source_layout.get("weight_map") or {}).items()
    }
    shards = {
        str(filename): dict(value)
        for filename, value in dict(source_layout.get("shards") or {}).items()
        if isinstance(value, dict)
    }
    if not model_id or not revision or not weight_map or not shards:
        raise ValueError("Qwen multi-file source layout is incomplete")

    combined_header: dict[str, Any] = {}
    entries_by_file: dict[str, dict[str, dict[str, Any]]] = {}
    for filename, shard in shards.items():
        header = dict(shard.get("header") or {})
        entries = tensor_entries(header)
        entries_by_file[filename] = entries
        for name, metadata in entries.items():
            if name in combined_header:
                raise RuntimeError(f"duplicate Qwen source tensor across shards: {name}")
            combined_header[name] = metadata
    if set(weight_map) != set(combined_header):
        raise RuntimeError("Qwen multi-file weight map/header tensor mismatch")
    if any(name not in entries_by_file.get(filename, {}) for name, filename in weight_map.items()):
        raise RuntimeError("Qwen multi-file weight map points to an invalid shard")

    keys, aliases = select_stage_source_keys(combined_header, spec)
    keys_by_file: dict[str, list[str]] = {}
    for key in keys:
        keys_by_file.setdefault(weight_map[key], []).append(key)

    reader = range_reader or (lambda target, start, end: fetch_range(target, start, end))
    tensors: dict[str, Any] = {}
    downloaded_bytes = 0
    range_group_count = 0
    source_files: list[str] = []
    for filename in sorted(keys_by_file):
        shard = shards.get(filename)
        if not isinstance(shard, dict):
            raise RuntimeError(f"Qwen source shard metadata missing: {filename}")
        header_length = int(shard.get("header_length") or 0)
        if header_length <= 0:
            raise RuntimeError(f"Qwen source shard header length invalid: {filename}")
        entries = entries_by_file[filename]
        data_start = 8 + header_length
        groups = _group_tensor_ranges(
            entries,
            keys_by_file[filename],
            data_start=data_start,
            max_group_bytes=int(max_group_bytes),
        )
        source_files.append(filename)
        range_group_count += len(groups)
        url = _hf_url(model_id, revision, filename)
        for group_start, group_end, group_keys in groups:
            payload = reader(url, group_start, group_end)
            downloaded_bytes += len(payload)
            for key in group_keys:
                offsets = [int(value) for value in entries[key]["data_offsets"]]
                absolute_start = data_start + offsets[0]
                relative_start = absolute_start - group_start
                length = offsets[1] - offsets[0]
                raw = bytearray(payload[relative_start : relative_start + length])
                tensor = torch.frombuffer(
                    raw, dtype=_torch_dtype(str(entries[key]["dtype"]))
                ).clone()
                tensors[key] = tensor.reshape(
                    [int(value) for value in entries[key]["shape"]]
                )

    if set(tensors) != set(keys):
        raise RuntimeError("Qwen multi-file stage materialization tensor mismatch")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output)
    summary = {
        "schema": "crowdtensor_qwen_multifile_stage_shard_v1",
        **spec.public_dict(),
        "source_model_id": model_id,
        "source_revision": revision,
        "source_layout_hash": str(source_layout.get("content_hash") or ""),
        "source_filenames": source_files,
        "source_filenames_hash": stable_hash(source_files),
        "source_keys_hash": stable_hash(keys),
        "source_tensor_count": len(keys),
        "source_parameter_count": sum(
            tensor_numel(combined_header[key]) for key in keys
        ),
        "source_tensor_bytes": sum(
            tensor_byte_count(combined_header[key]) for key in keys
        ),
        "downloaded_range_bytes": downloaded_bytes,
        "range_group_count": range_group_count,
        "full_model_file_downloaded": False,
        "stage_selective_loading": True,
        "multi_file_source": True,
        "aliases": aliases,
        "shard_file_hash": sha256_file(output),
        "shard_byte_count": output.stat().st_size,
        "private_paths_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }
    summary["content_hash"] = stable_hash(summary)
    return {**summary, "shard_path": str(output.resolve())}


QWEN_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
QWEN_STAGE_CHECKPOINT_SCHEMA = "crowdtensor_qwen15b_stage_checkpoint_v1"


def qwen_config_from_dict(value: dict[str, Any]) -> Any:
    """Build the pinned Qwen config with a deterministic training attention path."""

    from transformers import Qwen2Config

    config = Qwen2Config.from_dict(dict(value))
    config.use_cache = False
    config.attention_dropout = 0.0
    config._attn_implementation = "eager"
    return config


def _resolve_compute_dtype(device: Any, compute_dtype: Any | None) -> Any:
    import torch

    if compute_dtype is not None:
        if isinstance(compute_dtype, str):
            names = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            try:
                return names[compute_dtype.lower()]
            except KeyError as exc:
                raise ValueError(f"unsupported Qwen stage compute dtype: {compute_dtype}") from exc
        return compute_dtype
    return torch.float16 if torch.device(device).type == "cuda" else torch.float32


def _build_meta_qwen_stage_module(
    config: Any,
    spec: QwenStageSpec,
    *,
    device: Any,
    gradient_checkpointing: bool,
) -> Any:
    """Instantiate only one stage's owned modules, initially on the meta device."""

    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer,
        Qwen2RMSNorm,
        Qwen2RotaryEmbedding,
    )

    class QwenPipelineStage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.stage_spec = spec
            self.gradient_checkpointing_enabled = bool(gradient_checkpointing)
            with torch.device("meta"):
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleDict(
                    {
                        str(layer_index): Qwen2DecoderLayer(config, layer_index)
                        for layer_index in range(spec.layer_start, spec.layer_end)
                    }
                )
                if spec.owns_embedding:
                    self.model.embed_tokens = torch.nn.Embedding(
                        config.vocab_size,
                        config.hidden_size,
                        config.pad_token_id,
                    )
                if spec.owns_norm:
                    self.model.norm = Qwen2RMSNorm(
                        config.hidden_size,
                        eps=config.rms_norm_eps,
                    )
                if spec.owns_lm_head:
                    self.lm_head = torch.nn.Linear(
                        config.hidden_size,
                        config.vocab_size,
                        bias=False,
                    )
            # RoPE has no source weights. Construct its non-persistent buffers on
            # the execution device after the source-owned parameters exist on meta.
            self.rotary_emb = Qwen2RotaryEmbedding(config=config, device=device)

        def forward(
            self,
            value: Any,
            *,
            attention_mask: Any | None = None,
            position_ids: Any | None = None,
        ) -> Any:
            if self.stage_spec.owns_embedding:
                if value.dtype not in (torch.int32, torch.int64):
                    raise ValueError("Qwen stage0 requires integer input_ids")
                hidden_states = self.model.embed_tokens(value)
            else:
                hidden_states = value
            if hidden_states.ndim != 3:
                raise ValueError("Qwen stage hidden states must be rank three")
            batch_size, sequence_length = hidden_states.shape[:2]
            if position_ids is None:
                position_ids = torch.arange(
                    sequence_length,
                    device=hidden_states.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1)
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer_index in range(self.stage_spec.layer_start, self.stage_spec.layer_end):
                layer = self.model.layers[str(layer_index)]

                def layer_forward(states: Any, current_layer: Any = layer) -> Any:
                    result = current_layer(
                        states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                    )
                    return result[0] if isinstance(result, tuple) else result

                if self.gradient_checkpointing_enabled and self.training and hidden_states.requires_grad:
                    hidden_states = checkpoint(
                        layer_forward,
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = layer_forward(hidden_states)
            if self.stage_spec.owns_norm:
                hidden_states = self.model.norm(hidden_states)
            if self.stage_spec.owns_lm_head:
                return self.lm_head(hidden_states)
            return hidden_states

    return QwenPipelineStage()


def _load_stage_source_state(
    module: Any,
    spec: QwenStageSpec,
    shard_path: str | Path,
    *,
    device: Any,
    compute_dtype: Any,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    source = load_file(str(shard_path), device="cpu")
    source_keys = sorted(source)
    expected = set(module.state_dict())
    assigned: dict[str, Any] = {}
    for name, tensor in source.items():
        target_name = name
        if (
            name == "model.embed_tokens.weight"
            and "model.embed_tokens.weight" not in expected
            and "lm_head.weight" in expected
            and "lm_head.weight" not in source
        ):
            target_name = "lm_head.weight"
        if target_name not in expected:
            raise RuntimeError(f"stage shard contains a tensor not owned by stage {spec.stage_id}: {name}")
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=compute_dtype)
        assigned[target_name] = tensor.to(device=device)
    missing = sorted(expected - set(assigned))
    unexpected = sorted(set(assigned) - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"stage {spec.stage_id} source assignment mismatch: missing={missing}, unexpected={unexpected}"
        )
    result = module.load_state_dict(assigned, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Qwen stage strict source load did not consume exactly its shard")
    meta_parameters = [name for name, parameter in module.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"Qwen stage retained meta parameters: {meta_parameters}")
    return {
        "source_tensor_count": len(source_keys),
        "source_keys_hash": stable_hash(source_keys),
        "source_shard_hash": sha256_file(shard_path),
        "source_shard_byte_count": Path(shard_path).stat().st_size,
        "all_parameters_materialized": True,
    }


def load_qwen_pipeline_stage(
    config: dict[str, Any] | Any,
    spec: QwenStageSpec,
    shard_path: str | Path,
    *,
    device: str | Any = "cpu",
    compute_dtype: Any | None = None,
    inject_lora: bool = True,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    lora_target_modules: Iterable[str] = QWEN_LORA_TARGET_MODULES,
    lora_dropout: float = 0.0,
    lora_seed: int = 20260712,
    gradient_checkpointing: bool = True,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> tuple[Any, dict[str, Any]]:
    """Load one stage shard without ever constructing or loading the full model."""

    import torch
    from peft import LoraConfig, inject_adapter_in_model

    resolved_config = qwen_config_from_dict(config) if isinstance(config, dict) else config
    resolved_config.use_cache = False
    resolved_config._attn_implementation = "eager"
    target_device = torch.device(device)
    resolved_targets = tuple(str(item) for item in lora_target_modules)
    if not resolved_targets:
        raise ValueError("Qwen LoRA target modules cannot be empty")
    dtype = _resolve_compute_dtype(target_device, compute_dtype)
    module = _build_meta_qwen_stage_module(
        resolved_config,
        spec,
        device=target_device,
        gradient_checkpointing=gradient_checkpointing,
    )
    meta_parameter_count = sum(int(parameter.numel()) for parameter in module.parameters())
    meta_constructed = all(
        parameter.is_meta
        for name, parameter in module.named_parameters()
        if name != "rotary_emb.inv_freq"
    )
    source_report = _load_stage_source_state(
        module,
        spec,
        shard_path,
        device=target_device,
        compute_dtype=dtype,
    )
    for parameter in module.parameters():
        parameter.requires_grad = False
    if inject_lora:
        if int(lora_rank) <= 0 or int(lora_alpha) <= 0:
            raise ValueError("Qwen LoRA rank and alpha must be positive")
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(int(lora_seed) + int(spec.stage_id))
        try:
            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                target_modules=list(resolved_targets),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
            )
            module = inject_adapter_in_model(lora_config, module)
        finally:
            torch.random.set_rng_state(rng_state)
        # GradScaler cannot unscale FP16 parameter gradients. Keep LoRA
        # parameters in FP32. Qwen's BF16 source can overflow T4 FP16 compute,
        # so only stage-boundary tensors use FP16 in the runtime contract.
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
    trainable_names = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    trainable_dtypes = sorted(
        {
            str(parameter.dtype).replace("torch.", "")
            for parameter in module.parameters()
            if parameter.requires_grad
        }
    )
    frozen_dtypes = sorted(
        {
            str(parameter.dtype).replace("torch.", "")
            for parameter in module.parameters()
            if not parameter.requires_grad
        }
    )
    if inject_lora and (
        not trainable_names or any("lora_" not in name for name in trainable_names)
    ):
        raise RuntimeError("Qwen stage has missing or non-LoRA trainable parameters")
    loaded_layers = [
        int(name)
        for name in module.model.layers.keys()
    ]
    report = {
        "schema": (
            "crowdtensor_qwen15b_stage_load_v1"
            if str(model_id) == MODEL_ID and str(model_revision) == MODEL_REVISION
            else "crowdtensor_qwen_stage_load_v2"
        ),
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        **spec.public_dict(),
        **source_report,
        "meta_device_construction": meta_constructed,
        "meta_parameter_count": meta_parameter_count,
        "loaded_layer_indexes": loaded_layers,
        "loaded_layer_indexes_hash": stable_hash(loaded_layers),
        "loaded_full_model": False,
        "foreign_layer_count": 0,
        "stage_owned_module_construction": True,
        "compute_dtype": str(dtype).replace("torch.", ""),
        "source_dtype_cast_for_t4": bool(target_device.type == "cuda" and dtype == torch.float16),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "lora_injected": bool(inject_lora),
        "lora_rank": int(lora_rank) if inject_lora else 0,
        "lora_alpha": int(lora_alpha) if inject_lora else 0,
        "lora_target_modules": list(resolved_targets) if inject_lora else [],
        "parameter_count": sum(int(parameter.numel()) for parameter in module.parameters()),
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad
        ),
        "trainable_tensor_count": len(trainable_names),
        "trainable_tensor_names_hash": stable_hash(trainable_names),
        "only_lora_trainable": bool(
            not inject_lora or (trainable_names and all("lora_" in name for name in trainable_names))
        ),
        "trainable_parameter_dtypes": trainable_dtypes,
        "frozen_parameter_dtypes": frozen_dtypes,
        "fp32_lora_parameters_for_grad_scaler": bool(
            inject_lora and trainable_dtypes == ["float32"]
        ),
        "cuda_fp16_autocast": False,
        "cuda_fp32_stable_compute": target_device.type == "cuda" and dtype == torch.float32,
        "stage_boundary_dtype": (
            "float16" if target_device.type == "cuda" else str(dtype).replace("torch.", "")
        ),
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return module, report


def qwen_stage_adapter_state(module: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    return {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in get_peft_model_state_dict(module).items()
    }


def qwen_stage_adapter_hash(module: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(qwen_stage_adapter_state(module).items()):
        digest.update(name.encode("utf-8") + b"\0")
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little") + raw)
    return "sha256:" + digest.hexdigest()


def assemble_qwen_standard_peft_state(
    stage_states: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Map stage-owned adapter names to standard PeftModel adapter names."""

    assembled: dict[str, Any] = {}
    for state in stage_states:
        for name, tensor in state.items():
            if not str(name).startswith("model.layers.") or ".lora_" not in str(name):
                raise ValueError(f"non-Qwen-stage LoRA tensor cannot be exported: {name}")
            standard_name = f"base_model.model.{name}"
            if standard_name in assembled:
                raise ValueError(f"duplicate Qwen PEFT adapter tensor: {standard_name}")
            assembled[standard_name] = tensor.detach().cpu().contiguous()
    if not assembled:
        raise ValueError("Qwen PEFT adapter assembly received no tensors")
    return assembled


def export_qwen_standard_peft_adapter(
    stage_states: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    lora_target_modules: Iterable[str] = QWEN_LORA_TARGET_MODULES,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    from peft import LoraConfig
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = assemble_qwen_standard_peft_state(stage_states)
    resolved_targets = tuple(str(item) for item in lora_target_modules)
    adapter_config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        target_modules=list(resolved_targets),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=True,
        base_model_name_or_path=str(model_id),
        revision=str(model_revision),
    )
    adapter_config.save_pretrained(output)
    adapter_path = output / "adapter_model.safetensors"
    save_file(state, str(adapter_path))
    config_path = output / "adapter_config.json"
    layer_indexes = sorted(
        {
            int(match.group(1))
            for name in state
            if (match := re.search(r"\.layers\.(\d+)\.", name)) is not None
        }
    )
    report = {
        "schema": (
            "crowdtensor_qwen15b_standard_peft_export_v1"
            if str(model_id) == MODEL_ID and str(model_revision) == MODEL_REVISION
            else "crowdtensor_qwen_standard_peft_export_v2"
        ),
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "adapter_file": adapter_path.name,
        "adapter_file_hash": sha256_file(adapter_path),
        "adapter_file_byte_count": adapter_path.stat().st_size,
        "adapter_config_file": config_path.name,
        "adapter_config_hash": sha256_file(config_path),
        "adapter_tensor_count": len(state),
        "adapter_tensor_names_hash": stable_hash(sorted(state)),
        "layer_indexes": layer_indexes,
        "layer_indexes_hash": stable_hash(layer_indexes),
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "lora_target_modules": list(resolved_targets),
        "standard_peft_format": True,
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return {**report, "adapter_dir": str(output.resolve())}


def qwen_stage_base_hash(module: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if "lora_" in name:
            continue
        value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(value).to_bytes(8, "little") + value)
    return "sha256:" + digest.hexdigest()


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_qwen_stage_checkpoint(
    module: Any,
    optimizer: Any,
    scaler: Any,
    checkpoint_dir: str | Path,
    *,
    spec: QwenStageSpec,
    global_step: int,
    dataset_cursor: int,
    device: str,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter_path = output / f"stage{spec.stage_id}_adapter.safetensors"
    optimizer_path = output / f"stage{spec.stage_id}_optimizer.pt"
    scaler_path = output / f"stage{spec.stage_id}_grad_scaler.pt"
    rng_path = output / f"stage{spec.stage_id}_rng.pt"
    manifest_path = output / f"stage{spec.stage_id}_checkpoint.json"
    adapter = qwen_stage_adapter_state(module)
    save_file(adapter, str(adapter_path))
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scaler.state_dict(), scaler_path)
    rng_state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    target = torch.device(device)
    if target.type == "cuda":
        rng_state["cuda"] = torch.cuda.get_rng_state(target)
    torch.save(rng_state, rng_path)
    manifest = {
        "schema": QWEN_STAGE_CHECKPOINT_SCHEMA,
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "stage_id": int(spec.stage_id),
        "layer_start": int(spec.layer_start),
        "layer_end": int(spec.layer_end),
        "global_step": int(global_step),
        "optimizer_step": int(global_step),
        "dataset_cursor": int(dataset_cursor),
        "device": str(device),
        "adapter_file": adapter_path.name,
        "adapter_file_hash": sha256_file(adapter_path),
        "adapter_tensor_hash": qwen_stage_adapter_hash(module),
        "adapter_tensor_count": len(adapter),
        "optimizer_file": optimizer_path.name,
        "optimizer_file_hash": sha256_file(optimizer_path),
        "grad_scaler_file": scaler_path.name,
        "grad_scaler_file_hash": sha256_file(scaler_path),
        "grad_scaler_state_present": True,
        "rng_file": rng_path.name,
        "rng_file_hash": sha256_file(rng_path),
        "rng_state_present": True,
        "tensor_values_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    manifest["content_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def load_qwen_stage_checkpoint(
    module: Any,
    optimizer: Any,
    scaler: Any,
    checkpoint_dir: str | Path,
    *,
    spec: QwenStageSpec,
    device: str,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    root = Path(checkpoint_dir)
    manifest_path = root / f"stage{spec.stage_id}_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != QWEN_STAGE_CHECKPOINT_SCHEMA:
        raise RuntimeError("Qwen stage checkpoint schema mismatch")
    if (
        manifest.get("model_id") != str(model_id)
        or manifest.get("model_revision") != str(model_revision)
        or int(manifest.get("stage_id", -1)) != int(spec.stage_id)
        or int(manifest.get("layer_start", -1)) != int(spec.layer_start)
        or int(manifest.get("layer_end", -1)) != int(spec.layer_end)
    ):
        raise RuntimeError("Qwen stage checkpoint ownership mismatch")
    files = {
        "adapter": root / str(manifest["adapter_file"]),
        "optimizer": root / str(manifest["optimizer_file"]),
        "scaler": root / str(manifest["grad_scaler_file"]),
        "rng": root / str(manifest["rng_file"]),
    }
    expected_hashes = {
        "adapter": manifest["adapter_file_hash"],
        "optimizer": manifest["optimizer_file_hash"],
        "scaler": manifest["grad_scaler_file_hash"],
        "rng": manifest["rng_file_hash"],
    }
    if any(sha256_file(path) != expected_hashes[name] for name, path in files.items()):
        raise RuntimeError("Qwen stage checkpoint file hash mismatch")
    adapter = load_file(str(files["adapter"]), device="cpu")
    incompatible = set_peft_model_state_dict(module, adapter, adapter_name="default")
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"Qwen stage checkpoint has unexpected adapter keys: {unexpected}")
    optimizer.load_state_dict(torch.load(files["optimizer"], map_location="cpu", weights_only=True))
    _optimizer_to_device(optimizer, torch.device(device))
    scaler.load_state_dict(torch.load(files["scaler"], map_location="cpu", weights_only=True))
    rng = torch.load(files["rng"], map_location="cpu", weights_only=True)
    torch.random.set_rng_state(rng["cpu"])
    target = torch.device(device)
    if target.type == "cuda" and "cuda" in rng:
        torch.cuda.set_rng_state(rng["cuda"], target)
    if qwen_stage_adapter_hash(module) != manifest["adapter_tensor_hash"]:
        raise RuntimeError("Qwen stage adapter tensor hash mismatch after restore")
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def _new_qwen_grad_scaler(device: Any, *, init_scale: float) -> Any:
    import torch

    enabled = torch.device(device).type == "cuda"
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
            init_scale=float(init_scale),
            growth_interval=1000,
        )
    except (AttributeError, TypeError):  # pragma: no cover - older Kaggle torch
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
            init_scale=float(init_scale),
            growth_interval=1000,
        )


def _qwen_trainable_grad_norm(module: Any) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)


class QwenStageTrainer:
    """One stage's optimizer and private activation/gradient training state."""

    def __init__(
        self,
        module: Any,
        spec: QwenStageSpec,
        *,
        device: str,
        checkpoint_dir: str | Path,
        learning_rate: float = 5e-4,
        gradient_clip_norm: float = 1.0,
        grad_scaler_init_scale: float = 128.0,
        resume: bool = False,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
    ) -> None:
        import torch

        self.module = module
        self.spec = spec
        self.device = str(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.trainable_parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        if not self.trainable_parameters:
            raise RuntimeError("Qwen stage has no trainable LoRA parameters")
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=float(learning_rate),
            weight_decay=0.0,
        )
        self.scaler = _new_qwen_grad_scaler(device, init_scale=grad_scaler_init_scale)
        self.cached_outputs: dict[int, Any] = {}
        self.cached_inputs: dict[int, Any] = {}
        self.compute_intervals: list[dict[str, Any]] = []
        self.loaded_checkpoint: dict[str, Any] | None = None
        if resume:
            self.loaded_checkpoint = load_qwen_stage_checkpoint(
                self.module,
                self.optimizer,
                self.scaler,
                self.checkpoint_dir,
                spec=self.spec,
                device=self.device,
                model_id=self.model_id,
                model_revision=self.model_revision,
            )

    def _synchronize(self) -> None:
        import torch

        target = torch.device(self.device)
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    def _autocast(self) -> Any:
        import contextlib

        return contextlib.nullcontext()

    def _activation_dtype(self) -> Any:
        import torch

        if torch.device(self.device).type == "cuda":
            return torch.float32
        return next(self.module.parameters()).dtype

    def _record_interval(self, operation: str, microbatch_id: int, started_ns: int) -> None:
        import time

        self.compute_intervals.append(
            {
                "operation": operation,
                "microbatch_id": int(microbatch_id),
                "started_ns": int(started_ns),
                "ended_ns": int(time.time_ns()),
                "stage_id": int(self.spec.stage_id),
                "device": self.device,
            }
        )

    def begin_step(self) -> None:
        import torch

        if self.cached_outputs or self.cached_inputs:
            raise RuntimeError("Qwen stage cannot begin a step with retained graphs")
        self.optimizer.zero_grad(set_to_none=True)
        # Intermediate stages consume an already-scaled external gradient. This
        # initializes the matching GradScaler state before unscale_/step.
        self.scaler.scale(torch.zeros((), device=self.device))

    def forward(self, microbatch_id: int, value: Any) -> dict[str, Any]:
        import time
        import torch

        target = torch.device(self.device)
        if self.spec.owns_embedding:
            stage_input = torch.as_tensor(value, dtype=torch.long, device=target)
        else:
            stage_input = torch.as_tensor(value).to(
                device=target,
                dtype=self._activation_dtype(),
            )
            stage_input.requires_grad_(True)
            self.cached_inputs[int(microbatch_id)] = stage_input
        self._synchronize()
        started_ns = time.time_ns()
        with self._autocast():
            output = self.module(stage_input)
        if not bool(torch.isfinite(output.detach()).all()):
            raise RuntimeError("qwen15b_non_finite_stage_activation")
        self._synchronize()
        self._record_interval("forward", microbatch_id, started_ns)
        if self.spec.owns_lm_head:
            raise RuntimeError("final Qwen stage must use loss_backward")
        self.cached_outputs[int(microbatch_id)] = output
        private = output.detach()
        if target.type == "cuda":
            private = private.to(dtype=torch.float16)
        if not bool(torch.isfinite(private).all()):
            raise RuntimeError("qwen15b_non_finite_stage_boundary_activation")
        private = private.to("cpu").contiguous()
        return {
            "activation": private,
            "shape": list(private.shape),
            "dtype": str(private.dtype).replace("torch.", ""),
            "activation_hash": "sha256:"
            + hashlib.sha256(private.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "compute_interval": dict(self.compute_intervals[-1]),
        }

    def loss_backward(
        self,
        microbatch_id: int,
        hidden_states: Any,
        labels: Any,
        *,
        microbatch_count: int,
    ) -> dict[str, Any]:
        import time
        import torch

        if not self.spec.owns_lm_head:
            raise RuntimeError("only final Qwen stage computes causal LM loss")
        hidden = torch.as_tensor(hidden_states).to(
            device=self.device,
            dtype=self._activation_dtype(),
        )
        hidden.requires_grad_(True)
        target = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        self._synchronize()
        started_ns = time.time_ns()
        with self._autocast():
            logits = self.module(hidden)
            if not bool(torch.isfinite(logits.detach()).all()):
                raise RuntimeError("qwen15b_non_finite_logits")
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = target[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError("qwen15b_non_finite_loss")
        self.scaler.scale(loss / max(1, int(microbatch_count))).backward()
        self._synchronize()
        self._record_interval("forward_backward", microbatch_id, started_ns)
        gradient = hidden.grad.detach()
        if torch.device(self.device).type == "cuda":
            gradient = gradient.to(dtype=torch.float16)
        gradient = gradient.to("cpu").contiguous()
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("qwen15b_non_finite_activation_gradient")
        logits_probe = logits[:, -1:, : min(256, logits.shape[-1])].detach().to("cpu").contiguous()
        return {
            "activation_gradient": gradient,
            "loss": float(loss.detach().float().item()),
            "gradient_scale": float(self.scaler.get_scale()),
            "gradient_hash": "sha256:"
            + hashlib.sha256(gradient.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "logits_probe_hash": "sha256:"
            + hashlib.sha256(logits_probe.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "compute_interval": dict(self.compute_intervals[-1]),
        }

    def backward(self, microbatch_id: int, activation_gradient: Any) -> dict[str, Any]:
        import time
        import torch

        key = int(microbatch_id)
        output = self.cached_outputs.pop(key)
        incoming = torch.as_tensor(activation_gradient).to(
            device=self.device,
            dtype=output.dtype,
        )
        if not bool(torch.isfinite(incoming).all()):
            raise RuntimeError("qwen15b_non_finite_incoming_gradient")
        self._synchronize()
        started_ns = time.time_ns()
        output.backward(incoming)
        self._synchronize()
        self._record_interval("backward", microbatch_id, started_ns)
        previous = None
        if not self.spec.owns_embedding:
            stage_input = self.cached_inputs.pop(key)
            previous = stage_input.grad.detach()
            if torch.device(self.device).type == "cuda":
                previous = previous.to(dtype=torch.float16)
            previous = previous.to("cpu").contiguous()
            if not bool(torch.isfinite(previous).all()):
                raise RuntimeError("qwen15b_non_finite_activation_gradient")
        return {
            "activation_gradient": previous,
            "gradient_scale": float(self.scaler.get_scale()),
            "incoming_gradient_hash": "sha256:"
            + hashlib.sha256(incoming.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            "compute_interval": dict(self.compute_intervals[-1]),
        }

    def finish_step(self, *, global_step: int, dataset_cursor: int) -> dict[str, Any]:
        import torch

        if self.cached_outputs or self.cached_inputs:
            raise RuntimeError("Qwen stage cannot step with unfinished microbatches")
        scale_before = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        gradient_norm = _qwen_trainable_grad_norm(self.module)
        if not math.isfinite(gradient_norm):
            raise RuntimeError("qwen15b_non_finite_lora_gradient")
        torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        checkpoint = save_qwen_stage_checkpoint(
            self.module,
            self.optimizer,
            self.scaler,
            self.checkpoint_dir,
            spec=self.spec,
            global_step=global_step,
            dataset_cursor=dataset_cursor,
            device=self.device,
            model_id=self.model_id,
            model_revision=self.model_revision,
        )
        target = torch.device(self.device)
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0
        )
        return {
            "global_step": int(global_step),
            "dataset_cursor": int(dataset_cursor),
            "gradient_scale_before": scale_before,
            "gradient_scale_after": float(self.scaler.get_scale()),
            "lora_gradient_norm": gradient_norm,
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "gradient_clipping_applied": True,
            "optimizer_step_applied": True,
            "checkpoint_hash": checkpoint["content_hash"],
            "adapter_tensor_hash": checkpoint["adapter_tensor_hash"],
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }
