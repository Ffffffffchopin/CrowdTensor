#!/usr/bin/env python3
"""Prepare isolated private GSM8K SFT and benchmark payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crowdtensor.qwen7b_gsm8k_showcase import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PRIVATE_BENCHMARK_SCHEMA,
    dataset_manifest_content_hash,
    prepare_gsm8k_payloads,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-sequence-count", type=int, default=1024)
    parser.add_argument("--validation-sequence-count", type=int, default=16)
    parser.add_argument("--benchmark-example-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--benchmark-seed", type=int)
    parser.add_argument("--exclude-benchmark-payload", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    excluded_indexes: set[int] = set()
    for value in args.exclude_benchmark_payload:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        if (
            payload.get("schema") != PRIVATE_BENCHMARK_SCHEMA
            or payload.get("model_id") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("dataset_id") != DATASET_ID
            or payload.get("dataset_revision") != DATASET_REVISION
        ):
            raise ValueError("excluded GSM8K benchmark payload schema invalid")
        excluded_indexes.update(
            int(item["example_index"]) for item in payload.get("examples") or []
        )
    result = prepare_gsm8k_payloads(
        output,
        sequence_length=args.sequence_length,
        train_sequence_count=args.train_sequence_count,
        validation_sequence_count=args.validation_sequence_count,
        benchmark_example_count=args.benchmark_example_count,
        seed=args.seed,
        benchmark_seed=args.benchmark_seed,
        excluded_benchmark_indexes=excluded_indexes,
    )
    public = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "private_train_payload_path",
            "private_benchmark_payload_path",
        }
    }
    public["private_train_payload_created"] = True
    public["private_benchmark_payload_created"] = True
    if public.get("content_hash") != dataset_manifest_content_hash(public):
        raise RuntimeError("GSM8K public manifest content hash mismatch")
    path = output / "training_qwen7b_gsm8k_dataset_prepare.json"
    path.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "schema": public["schema"],
                    "training_non_padding_token_count": public[
                        "training_non_padding_token_count"
                    ],
                    "training_supervised_token_count": public[
                        "training_supervised_token_count"
                    ],
                    "benchmark_example_count": public["benchmark_example_count"],
                    "content_hash": public["content_hash"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
