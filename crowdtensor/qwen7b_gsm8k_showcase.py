"""Pinned Qwen2.5-7B-Instruct and GSM8K showcase inputs."""

from __future__ import annotations

import json
import random
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .qwen15b_training import (
    _hf_url,
    fetch_bytes,
    fetch_json,
    fetch_safetensors_header,
    sha256_bytes,
    sha256_file,
    stable_hash,
    tensor_byte_count,
    tensor_entries,
    tensor_numel,
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_PARAMETER_COUNT = 7_615_616_512
MODEL_WEIGHT_BYTES = 15_231_233_024
MODEL_INDEX_FILENAME = "model.safetensors.index.json"
MODEL_SHARD_FILENAMES = tuple(
    f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)
)
DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET_CONFIG = "main"
DATASET_FILES = {
    "train": {
        "path": "main/train-00000-of-00001.parquet",
        "size": 2_306_545,
        "sha256": "sha256:ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    },
    "test": {
        "path": "main/test-00000-of-00001.parquet",
        "size": 419_088,
        "sha256": "sha256:ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    },
}
SOURCE_LAYOUT_SCHEMA = "crowdtensor_qwen7b_instruct_source_layout_v1"
PRIVATE_TRAIN_SCHEMA = "crowdtensor_qwen7b_gsm8k_private_train_v1"
PRIVATE_BENCHMARK_SCHEMA = "crowdtensor_qwen7b_gsm8k_private_benchmark_v1"
DATASET_MANIFEST_SCHEMA = "crowdtensor_qwen7b_gsm8k_dataset_manifest_v1"
DATASET_MANIFEST_WORKFLOW_FIELDS = frozenset(
    {"private_train_payload_created", "private_benchmark_payload_created"}
)
SYSTEM_PROMPT = (
    "You are a careful math solver. Show the reasoning, then end with exactly "
    "one line in the form #### <number>."
)
ANSWER_PATTERN = re.compile(r"####\s*([-+]?[$]?[0-9][0-9,]*(?:\.[0-9]+)?)")


def dataset_manifest_content_hash(value: Mapping[str, Any]) -> str:
    """Hash immutable dataset facts while excluding local workflow status."""
    return stable_hash(
        {
            key: item
            for key, item in value.items()
            if key != "content_hash" and key not in DATASET_MANIFEST_WORKFLOW_FIELDS
        }
    )


def _dataset_url(path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/{path}"
    )


def normalize_numeric_answer(value: str) -> str:
    text = str(value or "").strip().replace("$", "").replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return ""
    if number.is_integer():
        return str(int(number))
    return (f"{number:.12f}").rstrip("0").rstrip(".")


def extract_gsm8k_answer(value: str, *, require_marker: bool = False) -> str:
    text = str(value or "")
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return normalize_numeric_answer(matches[-1])
    if require_marker:
        return ""
    fallback = re.findall(r"[-+]?[$]?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    return normalize_numeric_answer(fallback[-1]) if fallback else ""


def build_source_layout() -> dict[str, Any]:
    api = fetch_json(f"https://huggingface.co/api/models/{MODEL_ID}")
    if str(api.get("sha") or "") != MODEL_REVISION:
        raise RuntimeError("Qwen2.5-7B-Instruct source revision changed")
    config_bytes = fetch_bytes(_hf_url(MODEL_ID, MODEL_REVISION, "config.json"))
    config = json.loads(config_bytes)
    index_bytes = fetch_bytes(
        _hf_url(MODEL_ID, MODEL_REVISION, MODEL_INDEX_FILENAME)
    )
    index = json.loads(index_bytes)
    weight_map = {
        str(name): str(filename)
        for name, filename in dict(index.get("weight_map") or {}).items()
    }
    if sorted(set(weight_map.values())) != sorted(MODEL_SHARD_FILENAMES):
        raise RuntimeError("Qwen2.5-7B-Instruct shard index changed")

    shards: dict[str, Any] = {}
    parameter_count = 0
    weight_bytes = 0
    tensor_count = 0
    for filename in MODEL_SHARD_FILENAMES:
        header_length, header = fetch_safetensors_header(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            filename=filename,
        )
        entries = tensor_entries(header)
        shard_parameter_count = sum(tensor_numel(item) for item in entries.values())
        shard_weight_bytes = sum(tensor_byte_count(item) for item in entries.values())
        parameter_count += shard_parameter_count
        weight_bytes += shard_weight_bytes
        tensor_count += len(entries)
        shards[filename] = {
            "header_length": header_length,
            "header": header,
            "header_hash": stable_hash(header),
            "tensor_count": len(entries),
            "parameter_count": shard_parameter_count,
            "weight_bytes": shard_weight_bytes,
            "expected_file_size": shard_weight_bytes + header_length + 8,
        }
    if set(weight_map) != {
        name for shard in shards.values() for name in tensor_entries(shard["header"])
    }:
        raise RuntimeError("Qwen2.5-7B-Instruct index/header mismatch")
    if parameter_count != MODEL_PARAMETER_COUNT or weight_bytes != MODEL_WEIGHT_BYTES:
        raise RuntimeError("Qwen2.5-7B-Instruct pinned tensor budget mismatch")
    if int((index.get("metadata") or {}).get("total_size") or 0) != weight_bytes:
        raise RuntimeError("Qwen2.5-7B-Instruct index byte budget mismatch")

    report = {
        "schema": SOURCE_LAYOUT_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameter_count": parameter_count,
        "weight_bytes": weight_bytes,
        "tensor_count": tensor_count,
        "config": config,
        "config_hash": sha256_bytes(config_bytes),
        "weight_index_file": MODEL_INDEX_FILENAME,
        "weight_index_hash": sha256_bytes(index_bytes),
        "weight_map": weight_map,
        "weight_map_hash": stable_hash(weight_map),
        "shards": shards,
        "architecture": str((config.get("architectures") or [""])[0]),
        "model_type": str(config.get("model_type") or ""),
        "num_hidden_layers": int(config.get("num_hidden_layers") or 0),
        "hidden_size": int(config.get("hidden_size") or 0),
        "intermediate_size": int(config.get("intermediate_size") or 0),
        "num_attention_heads": int(config.get("num_attention_heads") or 0),
        "num_key_value_heads": int(config.get("num_key_value_heads") or 0),
        "vocab_size": int(config.get("vocab_size") or 0),
        "license": str((api.get("cardData") or {}).get("license") or ""),
        "gated": api.get("gated") is True,
        "private": api.get("private") is True,
        "stage_selective_range_loading": True,
        "full_model_download_required_per_worker": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }
    report["source_verified"] = bool(
        report["architecture"] == "Qwen2ForCausalLM"
        and report["model_type"] == "qwen2"
        and report["num_hidden_layers"] == 28
        and report["license"] == "apache-2.0"
        and report["gated"] is False
        and report["private"] is False
    )
    report["content_hash"] = stable_hash(report)
    return report


def _prompt_ids(tokenizer: Any, question: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(question).strip()},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if (
        isinstance(encoded, (list, tuple))
        and len(encoded) == 1
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = encoded[0]
    result = [int(value) for value in list(encoded or [])]
    if not result:
        raise RuntimeError("Qwen chat template produced no prompt tokens")
    return result


def _supervised_example(tokenizer: Any, question: str, answer: str) -> tuple[list[int], list[int]]:
    prompt = _prompt_ids(tokenizer, question)
    completion = tokenizer.encode(
        str(answer).strip(), add_special_tokens=False
    ) + [int(tokenizer.eos_token_id)]
    return prompt + completion, [-100] * len(prompt) + completion


def _packed_rows(
    examples: list[tuple[list[int], list[int]]],
    *,
    sequence_length: int,
    row_count: int,
) -> list[dict[str, Any]]:
    required = int(sequence_length) * int(row_count)
    input_stream: list[int] = []
    label_stream: list[int] = []
    index = 0
    while len(input_stream) < required:
        inputs, labels = examples[index % len(examples)]
        input_stream.extend(inputs)
        label_stream.extend(labels)
        index += 1
    rows = []
    for offset in range(0, required, int(sequence_length)):
        inputs = input_stream[offset : offset + int(sequence_length)]
        labels = label_stream[offset : offset + int(sequence_length)]
        supervised = sum(value != -100 for value in labels)
        if supervised < 1:
            raise RuntimeError("GSM8K packed row has no supervised answer tokens")
        rows.append(
            {
                "input_ids": inputs,
                "labels": labels,
                "non_padding_token_count": len(inputs),
                "supervised_token_count": supervised,
            }
        )
    return rows


def select_benchmark_indexes(
    *,
    total_count: int,
    example_count: int,
    seed: int,
    excluded_indexes: set[int] | None = None,
) -> list[int]:
    excluded = {int(value) for value in (excluded_indexes or set())}
    candidates = [index for index in range(int(total_count)) if index not in excluded]
    if len(candidates) < int(example_count):
        raise ValueError("GSM8K confirmatory benchmark candidate budget is incomplete")
    random.Random(int(seed)).shuffle(candidates)
    return sorted(candidates[: int(example_count)])


def prepare_gsm8k_payloads(
    output_dir: str | Path,
    *,
    sequence_length: int = 256,
    train_sequence_count: int = 1024,
    validation_sequence_count: int = 16,
    benchmark_example_count: int = 128,
    seed: int = 20260719,
    benchmark_seed: int | None = None,
    excluded_benchmark_indexes: set[int] | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    if not 128 <= int(sequence_length) <= 512:
        raise ValueError("GSM8K showcase sequence length must be in [128, 512]")
    if int(train_sequence_count) < 1024:
        raise ValueError("GSM8K showcase needs at least 1024 training sequences")
    if int(validation_sequence_count) < 8 or int(benchmark_example_count) < 128:
        raise ValueError("GSM8K showcase evaluation budget is incomplete")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / ".private-raw-gsm8k"
    cache_dir = output / ".private-tokenizer-cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        local: dict[str, Path] = {}
        for split, metadata in DATASET_FILES.items():
            payload = fetch_bytes(_dataset_url(str(metadata["path"])), timeout=180.0)
            if len(payload) != int(metadata["size"]) or sha256_bytes(payload) != metadata["sha256"]:
                raise RuntimeError(f"pinned GSM8K {split} parquet mismatch")
            path = raw_dir / f"{split}.parquet"
            path.write_bytes(payload)
            local[split] = path

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=False,
            trust_remote_code=False,
        )
        train_table = parquet.read_table(local["train"], columns=["question", "answer"])
        test_table = parquet.read_table(local["test"], columns=["question", "answer"])
        train_questions = train_table["question"].to_pylist()
        train_answers = train_table["answer"].to_pylist()
        test_questions = test_table["question"].to_pylist()
        test_answers = test_table["answer"].to_pylist()

        rng = random.Random(int(seed))
        train_indexes = list(range(len(train_questions)))
        rng.shuffle(train_indexes)
        validation_source_indexes = train_indexes[:128]
        training_source_indexes = train_indexes[128:]
        training_examples = [
            _supervised_example(
                tokenizer,
                train_questions[index],
                train_answers[index],
            )
            for index in training_source_indexes
        ]
        validation_examples = [
            _supervised_example(
                tokenizer,
                train_questions[index],
                train_answers[index],
            )
            for index in validation_source_indexes
        ]
        train_rows = _packed_rows(
            training_examples,
            sequence_length=sequence_length,
            row_count=train_sequence_count,
        )
        validation_rows = _packed_rows(
            validation_examples,
            sequence_length=sequence_length,
            row_count=validation_sequence_count,
        )

        excluded_indexes = {
            int(value) for value in (excluded_benchmark_indexes or set())
        }
        selected_benchmark_seed = (
            int(seed) + 1 if benchmark_seed is None else int(benchmark_seed)
        )
        benchmark_indexes = select_benchmark_indexes(
            total_count=len(test_questions),
            example_count=int(benchmark_example_count),
            seed=selected_benchmark_seed,
            excluded_indexes=excluded_indexes,
        )
        benchmark_examples = []
        for index in benchmark_indexes:
            gold = extract_gsm8k_answer(str(test_answers[index]), require_marker=True)
            if not gold:
                raise RuntimeError("GSM8K benchmark gold answer is invalid")
            benchmark_examples.append(
                {
                    "example_index": index,
                    "prompt_input_ids": _prompt_ids(tokenizer, str(test_questions[index])),
                    "gold_answer": gold,
                }
            )

        train_payload = {
            "schema": PRIVATE_TRAIN_SCHEMA,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_config": DATASET_CONFIG,
            "sequence_length": int(sequence_length),
            "train": train_rows,
            "validation": validation_rows,
        }
        benchmark_payload = {
            "schema": PRIVATE_BENCHMARK_SCHEMA,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_config": DATASET_CONFIG,
            "split": "test",
            "system_prompt": SYSTEM_PROMPT,
            "examples": benchmark_examples,
        }
        train_path = output / "qwen7b_gsm8k_train_private.json"
        benchmark_path = output / "qwen7b_gsm8k_benchmark_private.json"
        train_path.write_text(
            json.dumps(train_payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        benchmark_path.write_text(
            json.dumps(benchmark_payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        non_padding_tokens = sum(
            int(row["non_padding_token_count"]) for row in train_rows
        )
        supervised_tokens = sum(
            int(row["supervised_token_count"]) for row in train_rows
        )
        manifest = {
            "schema": DATASET_MANIFEST_SCHEMA,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_config": DATASET_CONFIG,
            "license": ["mit"],
            "sequence_length": int(sequence_length),
            "train_sequence_count": len(train_rows),
            "validation_sequence_count": len(validation_rows),
            "benchmark_example_count": len(benchmark_examples),
            "benchmark_seed": selected_benchmark_seed,
            "benchmark_excluded_example_count": len(excluded_indexes),
            "benchmark_excluded_indexes_hash": stable_hash(
                sorted(excluded_indexes)
            ),
            "benchmark_overlap_with_excluded_count": len(
                set(benchmark_indexes) & excluded_indexes
            ),
            "confirmatory_fresh_holdout": bool(excluded_indexes),
            "training_non_padding_token_count": non_padding_tokens,
            "training_supervised_token_count": supervised_tokens,
            "training_split": "train",
            "validation_source_split": "train_reserved",
            "benchmark_split": "test",
            "train_test_split_isolation_verified": True,
            "training_source_indexes_hash": stable_hash(training_source_indexes),
            "validation_source_indexes_hash": stable_hash(validation_source_indexes),
            "benchmark_indexes_hash": stable_hash(benchmark_indexes),
            "train_token_hash": stable_hash(train_rows),
            "validation_token_hash": stable_hash(validation_rows),
            "benchmark_prompt_hash": stable_hash(
                [item["prompt_input_ids"] for item in benchmark_examples]
            ),
            "benchmark_gold_hash": stable_hash(
                [item["gold_answer"] for item in benchmark_examples]
            ),
            "private_train_payload_hash": sha256_file(train_path),
            "private_benchmark_payload_hash": sha256_file(benchmark_path),
            "raw_text_public": False,
            "token_ids_public": False,
            "gold_answers_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        manifest["content_hash"] = dataset_manifest_content_hash(manifest)
        return {
            **manifest,
            "private_train_payload_path": str(train_path),
            "private_benchmark_payload_path": str(benchmark_path),
        }
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
