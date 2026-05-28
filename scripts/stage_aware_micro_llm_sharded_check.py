#!/usr/bin/env python3
"""Stage-aware acceptance wrapper for CPU-only micro-LLM sharded inference."""

from __future__ import annotations

import json
import sys

from micro_llm_sharded_inference_check import main as base_main


def main() -> None:
    if "--stage-mode" not in sys.argv and not any(arg.startswith("--stage-mode=") for arg in sys.argv):
        sys.argv.extend(["--stage-mode", "split"])
    if "--require-distinct-stage-miners" not in sys.argv:
        sys.argv.append("--require-distinct-stage-miners")
    base_main()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"schema": "stage_aware_micro_llm_sharded_check_v1", "ok": False, "error": str(exc)}, sort_keys=True))
        raise SystemExit(1) from exc
