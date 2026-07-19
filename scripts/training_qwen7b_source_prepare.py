#!/usr/bin/env python3
"""Build the pinned multi-shard Qwen2.5-7B-Instruct source layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crowdtensor.qwen7b_gsm8k_showcase import build_source_layout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build_source_layout()
    path = output / "qwen7b_source_layout.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.get("source_verified") is True,
                    "schema": report["schema"],
                    "model_id": report["model_id"],
                    "model_revision": report["model_revision"],
                    "parameter_count": report["parameter_count"],
                    "weight_bytes": report["weight_bytes"],
                    "content_hash": report["content_hash"],
                },
                sort_keys=True,
            )
        )
    return 0 if report.get("source_verified") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
