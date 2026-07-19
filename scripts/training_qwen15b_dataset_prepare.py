#!/usr/bin/env python3
"""Prepare the pinned private WikiText token payload for Qwen 1.5B Alpha."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.qwen15b_training import (  # noqa: E402
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    prepare_tokenized_wikitext,
)


SCHEMA = "crowdtensor_qwen15b_dataset_prepare_v1"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(
    output_dir: str | Path,
    *,
    sequence_length: int = 64,
    train_sequence_count: int = 32,
    validation_sequence_count: int = 8,
    preparer: Callable[..., dict[str, Any]] = prepare_tokenized_wikitext,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    private_result = preparer(
        output,
        sequence_length=sequence_length,
        train_sequence_count=train_sequence_count,
        validation_sequence_count=validation_sequence_count,
    )
    private_path = Path(str(private_result.pop("private_tokenized_path")))
    report = {
        "schema": SCHEMA,
        "ok": bool(
            private_result.get("model_id") == MODEL_ID
            and private_result.get("model_revision") == MODEL_REVISION
            and private_result.get("dataset_id") == DATASET_ID
            and private_result.get("dataset_revision") == DATASET_REVISION
            and int(private_result.get("sequence_length") or 0) == int(sequence_length)
            and int(private_result.get("train_sequence_count") or 0) >= 32
            and int(private_result.get("validation_sequence_count") or 0) >= 4
            and private_result.get("raw_text_public") is False
            and private_result.get("token_ids_public") is False
            and private_path.is_file()
        ),
        "manifest": private_result,
        "private_payload_present": private_path.is_file(),
        "private_payload_hash": private_result.get("private_tokenized_payload_hash"),
        "private_payload_name_public": False,
        "raw_text_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["blockers"] = [] if report["ok"] else ["qwen15b_tokenized_dataset_incomplete"]
    _write(output / "training_qwen15b_dataset_prepare.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--train-sequence-count", type=int, default=32)
    parser.add_argument("--validation-sequence-count", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build(
        args.output_dir,
        sequence_length=args.sequence_length,
        train_sequence_count=args.train_sequence_count,
        validation_sequence_count=args.validation_sequence_count,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"training_qwen15b_dataset_prepare ok={report['ok']}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
