"""Pinned, stage-selective Hugging Face source loader for Qwen training."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .heterogeneous_training_manifest import stable_hash, validate_training_manifest
from .qwen15b_training import QwenStageSpec, sha256_file


SOURCE_SCHEMA = "crowdtensor_heterogeneous_qwen_source_v1"
SHARD_SCHEMA = "crowdtensor_heterogeneous_qwen_stage_shard_v1"
LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


def _request_bytes(
    url: str,
    *,
    token: str = "",
    byte_range: tuple[int, int] | None = None,
    timeout: float = 180.0,
    attempts: int = 5,
) -> bytes:
    headers = {"User-Agent": "crowdtensor-heterogeneous-qwen-training/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last_error: BaseException | None = None
    for attempt in range(int(attempts)):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                expected = (
                    int(byte_range[1] - byte_range[0] + 1)
                    if byte_range is not None
                    else None
                )
                value = response.read((expected + 1) if expected is not None else -1)
                if expected is not None:
                    content_range = str(response.headers.get("Content-Range") or "")
                    if len(value) != expected or (
                        int(getattr(response, "status", 0) or 0) != 206
                        and not content_range
                    ):
                        raise RuntimeError("heterogeneous_qwen_source_range_ignored")
                return value
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            last_error = exc
        if attempt + 1 < int(attempts):
            time.sleep(min(8.0, 0.5 * 2**attempt))
    raise RuntimeError("heterogeneous_qwen_source_fetch_failed") from last_error


def _hf_url(model_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"


def _json_bytes(value: bytes, *, code: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(code) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(code)
    return parsed


def resolve_qwen_source(
    training_manifest: dict[str, Any],
    *,
    token: str = "",
    source_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve pinned config/index metadata without downloading model tensors."""

    manifest = validate_training_manifest(training_manifest)
    model = manifest["model"]
    model_id = str(model["model_id"])
    revision = str(model["model_revision"])
    root = Path(source_root).expanduser().resolve() if source_root else None
    if root is not None:
        config_bytes = (root / "config.json").read_bytes()
        index_bytes = (root / "model.safetensors.index.json").read_bytes()
        source_kind = "attached_local_model"
    else:
        config_bytes = _request_bytes(
            _hf_url(model_id, revision, "config.json"), token=token
        )
        index_bytes = _request_bytes(
            _hf_url(model_id, revision, "model.safetensors.index.json"), token=token
        )
        source_kind = "huggingface_range"
    config = _json_bytes(config_bytes, code="heterogeneous_qwen_config_invalid")
    index = _json_bytes(index_bytes, code="heterogeneous_qwen_index_invalid")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("heterogeneous_qwen_weight_map_invalid")
    if (
        config.get("model_type") != model["model_type"]
        or int(config.get("num_hidden_layers") or 0) != int(model["num_hidden_layers"])
        or int(config.get("hidden_size") or 0) != int(model["hidden_size"])
        or int(config.get("intermediate_size") or 0)
        != int(model["intermediate_size"])
        or int(config.get("vocab_size") or 0) != int(model["vocab_size"])
        or int((index.get("metadata") or {}).get("total_size") or 0)
        != int(model["weight_bytes"])
    ):
        raise RuntimeError("heterogeneous_qwen_source_manifest_mismatch")
    files = sorted({str(item) for item in weight_map.values()})
    report = {
        "schema": SOURCE_SCHEMA,
        "model_id": model_id,
        "model_revision": revision,
        "source_kind": source_kind,
        "config_hash": "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
        "weight_index_hash": "sha256:" + hashlib.sha256(index_bytes).hexdigest(),
        "weight_bytes": int((index.get("metadata") or {}).get("total_size") or 0),
        "weight_file_count": len(files),
        "weight_file_names_hash": stable_hash(files),
        "weight_tensor_count": len(weight_map),
        "stage_selective_loading": True,
        "full_model_download_required": False,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["source_verified"] = True
    report["content_hash"] = stable_hash(report)
    return config, index, report


def _header_from_reader(reader: Callable[[int, int], bytes]) -> tuple[int, dict[str, Any]]:
    prefix = reader(0, 7)
    if len(prefix) != 8:
        raise RuntimeError("heterogeneous_qwen_safetensors_header_missing")
    header_length = int(struct.unpack("<Q", prefix)[0])
    if header_length < 2 or header_length > 128 * 1024 * 1024:
        raise RuntimeError("heterogeneous_qwen_safetensors_header_invalid")
    header = _json_bytes(
        reader(8, 7 + header_length),
        code="heterogeneous_qwen_safetensors_header_invalid",
    )
    return header_length, header


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "I64": torch.int64,
        }[str(name).upper()]
    except KeyError as exc:
        raise RuntimeError("heterogeneous_qwen_source_dtype_unsupported") from exc


def _selected_keys(
    weight_map: dict[str, Any],
    stage: dict[str, Any],
) -> list[str]:
    selected = []
    for name in weight_map:
        match = LAYER_PATTERN.match(str(name))
        if match and int(stage["layer_start"]) <= int(match.group(1)) < int(
            stage["layer_end"]
        ):
            selected.append(str(name))
    if stage["owns_embedding"]:
        selected.append("model.embed_tokens.weight")
    if stage["owns_norm"]:
        selected.append("model.norm.weight")
    if stage["owns_lm_head"]:
        selected.append("lm_head.weight")
    missing = sorted(set(selected) - set(weight_map))
    if missing:
        raise RuntimeError("heterogeneous_qwen_stage_source_keys_missing")
    return sorted(set(selected))


def _groups(
    entries: dict[str, dict[str, Any]],
    keys: list[str],
    *,
    data_start: int,
    max_group_bytes: int,
) -> list[tuple[int, int, list[str]]]:
    values = []
    for key in keys:
        offsets = [int(item) for item in entries[key].get("data_offsets") or []]
        if len(offsets) != 2 or offsets[1] <= offsets[0]:
            raise RuntimeError("heterogeneous_qwen_tensor_offsets_invalid")
        values.append((data_start + offsets[0], data_start + offsets[1] - 1, key))
    values.sort()
    groups: list[tuple[int, int, list[str]]] = []
    for start, end, key in values:
        if groups and start == groups[-1][1] + 1 and end - groups[-1][0] + 1 <= int(
            max_group_bytes
        ):
            old_start, _old_end, old_keys = groups[-1]
            groups[-1] = (old_start, end, [*old_keys, key])
        else:
            groups.append((start, end, [key]))
    return groups


def materialize_qwen_stage_shard(
    training_manifest: dict[str, Any],
    *,
    stage_id: int,
    output_path: str | Path,
    token: str = "",
    source_root: str | Path | None = None,
    max_group_bytes: int = 128 * 1024 * 1024,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Download/read only tensors owned by one manifest stage."""

    import torch
    from safetensors.torch import save_file

    manifest = validate_training_manifest(training_manifest)
    config, index, source = resolve_qwen_source(
        manifest, token=token, source_root=source_root
    )
    stage = dict(manifest["stages"][int(stage_id)])
    if int(stage["stage_id"]) != int(stage_id):
        raise ValueError("heterogeneous_qwen_stage_id_invalid")
    weight_map = {str(key): str(value) for key, value in index["weight_map"].items()}
    keys = _selected_keys(weight_map, stage)
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "source_resolved",
                "stage_id": int(stage_id),
                "source_file_count": len(by_file),
                "source_tensor_count": len(keys),
                "downloaded_bytes": 0,
            }
        )
    root = Path(source_root).expanduser().resolve() if source_root else None
    tensors: dict[str, Any] = {}
    downloaded_bytes = 0
    range_group_count = 0
    source_files = []
    for source_file_index, filename in enumerate(sorted(by_file), start=1):
        source_files.append(filename)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "source_file_started",
                    "stage_id": int(stage_id),
                    "source_file_index": source_file_index,
                    "source_file_count": len(by_file),
                    "downloaded_bytes": downloaded_bytes,
                }
            )
        if root is not None:
            source_path = root / filename

            def read_range(start: int, end: int, path: Path = source_path) -> bytes:
                with path.open("rb") as handle:
                    handle.seek(start)
                    return handle.read(end - start + 1)

        else:
            url = _hf_url(
                manifest["model"]["model_id"],
                manifest["model"]["model_revision"],
                filename,
            )

            def read_range(start: int, end: int, target: str = url) -> bytes:
                return _request_bytes(
                    target,
                    token=token,
                    byte_range=(start, end),
                    timeout=300.0,
                )

        header_length, header = _header_from_reader(read_range)
        entries = {
            str(name): dict(item)
            for name, item in header.items()
            if name != "__metadata__" and isinstance(item, dict)
        }
        file_keys = sorted(by_file[filename])
        if any(key not in entries for key in file_keys):
            raise RuntimeError("heterogeneous_qwen_weight_index_header_mismatch")
        data_start = 8 + header_length
        groups = _groups(
            entries,
            file_keys,
            data_start=data_start,
            max_group_bytes=max_group_bytes,
        )
        range_group_count += len(groups)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "source_file_planned",
                    "stage_id": int(stage_id),
                    "source_file_index": source_file_index,
                    "source_file_count": len(by_file),
                    "source_file_range_group_count": len(groups),
                    "downloaded_bytes": downloaded_bytes,
                }
            )
        for source_file_group_index, (group_start, group_end, group_keys) in enumerate(
            groups, start=1
        ):
            payload = read_range(group_start, group_end)
            if len(payload) != group_end - group_start + 1:
                raise RuntimeError("heterogeneous_qwen_source_range_length_invalid")
            downloaded_bytes += len(payload)
            for key in group_keys:
                metadata = entries[key]
                offsets = [int(item) for item in metadata["data_offsets"]]
                absolute_start = data_start + offsets[0]
                relative_start = absolute_start - group_start
                length = offsets[1] - offsets[0]
                raw = bytearray(payload[relative_start : relative_start + length])
                tensor = torch.frombuffer(
                    raw, dtype=_torch_dtype(str(metadata["dtype"]))
                ).clone()
                tensors[key] = tensor.reshape(
                    [int(item) for item in metadata["shape"]]
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "range_group_downloaded",
                        "stage_id": int(stage_id),
                        "source_file_index": source_file_index,
                        "source_file_count": len(by_file),
                        "source_file_range_group_index": source_file_group_index,
                        "source_file_range_group_count": len(groups),
                        "downloaded_bytes": downloaded_bytes,
                        "materialized_tensor_count": len(tensors),
                    }
                )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output))
    del tensors
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "stage_shard_saved",
                "stage_id": int(stage_id),
                "source_file_count": len(source_files),
                "range_group_count": range_group_count,
                "downloaded_bytes": downloaded_bytes,
                "source_tensor_count": len(keys),
            }
        )
    report = {
        "schema": SHARD_SCHEMA,
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "stage_id": int(stage_id),
        "layer_start": int(stage["layer_start"]),
        "layer_end": int(stage["layer_end"]),
        "source_tensor_count": len(keys),
        "source_keys_hash": stable_hash(keys),
        "source_file_count": len(source_files),
        "source_file_names_hash": stable_hash(source_files),
        "source_tensor_bytes": int(stage["estimated_weight_bytes"]),
        "read_range_bytes": downloaded_bytes,
        "range_group_count": range_group_count,
        "stage_selective_loading": True,
        "full_model_downloaded": False,
        "source_kind": source["source_kind"],
        "shard_file_hash": sha256_file(output),
        "shard_byte_count": output.stat().st_size,
        "credential_values_public": False,
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return {**report, "shard_path": str(output)}


def qwen_stage_spec(
    training_manifest: dict[str, Any],
    *,
    stage_id: int,
    device_index: int = 0,
) -> QwenStageSpec:
    manifest = validate_training_manifest(training_manifest)
    stage = manifest["stages"][int(stage_id)]
    return QwenStageSpec(
        stage_id=int(stage_id),
        kernel_slot="dynamic",
        device_index=int(device_index),
        layer_start=int(stage["layer_start"]),
        layer_end=int(stage["layer_end"]),
        owns_embedding=bool(stage["owns_embedding"]),
        owns_norm=bool(stage["owns_norm"]),
        owns_lm_head=bool(stage["owns_lm_head"]),
    )


def prepare_manifest_wikitext(
    training_manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    token: str = "",
) -> dict[str, Any]:
    """Prepare the pinned manifest dataset while keeping text and tokens private."""

    import shutil

    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    from .qwen15b_training import (
        _dataset_file_url,
        _tokenize_split,
        sha256_bytes,
    )

    manifest = validate_training_manifest(training_manifest)
    training = manifest["training"]
    dataset = manifest["dataset"]
    required_train = (
        int(training["target_steps"])
        * int(training["microbatches_per_step"])
        * int(training["microbatch_size"])
    )
    train_count = max(required_train, 8)
    validation_count = 4
    sequence_length = int(training["sequence_length"])
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw_root = root / ".private-raw-wikitext"
    tokenizer_cache = root / ".private-tokenizer-cache"
    raw_root.mkdir(parents=True, exist_ok=True)
    files = {
        "train": (
            "wikitext-2-raw-v1/train-00000-of-00001.parquet",
            "sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
        ),
        "validation": (
            "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
            "sha256:204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
        ),
    }
    local = {}
    try:
        for split, (remote, expected_hash) in files.items():
            payload = _request_bytes(_dataset_file_url(remote), timeout=180.0)
            if sha256_bytes(payload) != expected_hash:
                raise RuntimeError("heterogeneous_wikitext_source_hash_mismatch")
            path = raw_root / f"{split}.parquet"
            path.write_bytes(payload)
            local[split] = path
        tokenizer = AutoTokenizer.from_pretrained(
            manifest["model"]["model_id"],
            revision=manifest["model"]["model_revision"],
            cache_dir=tokenizer_cache,
            token=token or None,
            trust_remote_code=False,
        )
        train_text = parquet.read_table(local["train"], columns=["text"])[
            "text"
        ].to_pylist()
        validation_text = parquet.read_table(
            local["validation"], columns=["text"]
        )["text"].to_pylist()
        train_rows, train_indexes = _tokenize_split(
            train_text,
            tokenizer,
            sequence_length=sequence_length,
            sequence_count=train_count,
        )
        validation_rows, validation_indexes = _tokenize_split(
            validation_text,
            tokenizer,
            sequence_length=sequence_length,
            sequence_count=validation_count,
        )
        private = {
            "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
            "training_manifest_hash": manifest["content_hash"],
            "model_id": manifest["model"]["model_id"],
            "model_revision": manifest["model"]["model_revision"],
            "dataset_id": dataset["dataset_id"],
            "dataset_revision": dataset["dataset_revision"],
            "sequence_length": sequence_length,
            "train": train_rows,
            "validation": validation_rows,
        }
        private_path = root / "heterogeneous_tokenized_private.json"
        private_path.write_text(
            json.dumps(private, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        private_path.chmod(0o600)
        report = {
            "schema": "crowdtensor_heterogeneous_tokenized_dataset_v1",
            "training_manifest_hash": manifest["content_hash"],
            "model_id": manifest["model"]["model_id"],
            "model_revision": manifest["model"]["model_revision"],
            "dataset_id": dataset["dataset_id"],
            "dataset_revision": dataset["dataset_revision"],
            "sequence_length": sequence_length,
            "train_sequence_count": len(train_rows),
            "validation_sequence_count": len(validation_rows),
            "train_row_indexes_hash": stable_hash(train_indexes),
            "validation_row_indexes_hash": stable_hash(validation_indexes),
            "train_token_hash": stable_hash(train_rows),
            "validation_token_hash": stable_hash(validation_rows),
            "private_tokenized_payload_hash": sha256_file(private_path),
            "raw_training_text_public": False,
            "token_ids_public": False,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        return {**report, "private_tokenized_path": str(private_path)}
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)
        shutil.rmtree(tokenizer_cache, ignore_errors=True)
