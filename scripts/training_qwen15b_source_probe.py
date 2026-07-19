#!/usr/bin/env python3
"""Resolve and record the pinned Qwen 1.5B and WikiText Alpha sources."""

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
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    build_stage_ownership,
    build_weight_index,
    resolve_source_manifest,
    sha256_file,
)


SCHEMA = "crowdtensor_qwen15b_training_source_probe_v1"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(
    output_dir: str | Path,
    *,
    resolver: Callable[[], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
    = resolve_source_manifest,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, config, header = resolver()
    index = build_weight_index(header)
    ownership = build_stage_ownership(config, header)
    source_path = output / "qwen15b_source_manifest.json"
    index_path = output / "qwen15b_generated_weight_index.json"
    ownership_path = output / "qwen15b_four_stage_ownership.json"
    _write(source_path, source)
    _write(index_path, index)
    _write(ownership_path, ownership)
    report = {
        "schema": SCHEMA,
        "ok": bool(
            source.get("source_verified") is True
            and source.get("model_id") == MODEL_ID
            and source.get("model_revision") == MODEL_REVISION
            and int(source.get("parameter_count") or 0) == MODEL_PARAMETER_COUNT
            and source.get("parameter_count", 0) >= 1_000_000_000
            and (source.get("dataset") or {}).get("source_verified") is True
            and ownership.get("all_source_tensors_covered") is True
            and ownership.get("only_tied_embedding_lm_head_duplicated") is True
            and ownership.get("four_distinct_kernel_device_placements") is True
        ),
        "source": source,
        "weight_index": {
            "schema": index["schema"],
            "content_hash": index["content_hash"],
            "metadata": index["metadata"],
            "full_weight_map_public_artifact": True,
        },
        "ownership": ownership,
        "artifacts": {
            "source_manifest": source_path.name,
            "weight_index": index_path.name,
            "four_stage_ownership": ownership_path.name,
        },
        "artifact_hashes": {
            "source_manifest": sha256_file(source_path),
            "weight_index": sha256_file(index_path),
            "four_stage_ownership": sha256_file(ownership_path),
        },
        "raw_text_public": False,
        "token_ids_public": False,
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if not report["ok"]:
        report["blockers"] = ["qwen15b_training_source_contract_incomplete"]
    else:
        report["blockers"] = []
    _write(output / "training_qwen15b_source_probe.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build(args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"training_qwen15b_source_probe ok={report['ok']}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
