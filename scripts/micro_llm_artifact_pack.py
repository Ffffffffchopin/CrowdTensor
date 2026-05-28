#!/usr/bin/env python3
"""Build or inspect a dependency-free file-backed micro-LLM artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.micro_llm_artifact import (  # noqa: E402
    DEFAULT_ARTIFACT_ID,
    ARTIFACT_SCHEMA_VERSION,
    build_default_micro_llm_artifact,
    inspect_micro_llm_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or inspect a micro_llm_artifact_v1 package.")
    parser.add_argument("--output-dir", default="dist/micro-llm-artifact")
    parser.add_argument("--artifact-id", default=DEFAULT_ARTIFACT_ID)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--inspect", action="store_true", help="inspect an existing artifact instead of building it")
    parser.add_argument("--json-out", default="")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.output_dir)
    try:
        report = inspect_micro_llm_artifact(artifact_dir) if args.inspect else build_default_micro_llm_artifact(
            artifact_dir,
            artifact_id=args.artifact_id,
            version=args.version,
        )
        report = {
            **report,
            "schema": ARTIFACT_SCHEMA_VERSION,
            "output_dir": str(artifact_dir.resolve()),
        }
        if args.json_out:
            output = Path(args.json_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"schema": ARTIFACT_SCHEMA_VERSION, "ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
