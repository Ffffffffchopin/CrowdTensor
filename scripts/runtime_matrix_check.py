#!/usr/bin/env python3
"""Deterministic acceptance check for the runtime capability matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import runtime_matrix  # noqa: E402


CPU_REQUIRED = {
    "diloco_train",
    "cpu_lora_mock",
    "micro_transformer_lm",
    "model_bundle_lm",
    "model_bundle_infer",
    "external_llm_infer_mock",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CrowdTensorD runtime capability matrix output.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--root", default=str(ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = runtime_matrix.build_matrix(root=Path(args.root))
    workloads = {row["name"]: row for row in matrix["workloads"]}
    missing = [name for name in sorted(CPU_REQUIRED) if workloads.get(name, {}).get("status") != "available"]
    if missing:
        raise SystemExit(f"CPU baseline workloads must be available: {missing}")
    for name in ["external_llm_infer_command", "external_llm_infer_http", "browser_probe"]:
        status = workloads.get(name, {}).get("status")
        if status not in {"available", "configured", "optional_missing", "blocked"}:
            raise SystemExit(f"unexpected status for {name}: {status}")
        if status == "blocked" and name != "external_llm_infer_command":
            raise SystemExit(f"{name} should be optional_missing when not configured")
    payload = json.dumps(matrix, sort_keys=True)
    for secret_fragment in ["local-runtime-key", "CROWDTENSOR_LLM_RUNTIME_API_KEY=", "Bearer "]:
        if secret_fragment in payload:
            raise SystemExit("runtime matrix leaked secret-like material")
    print(json.dumps({
        "ok": matrix["ok"],
        "available": matrix["summary"]["available"],
        "optional_missing": matrix["summary"]["optional_missing"],
        "blocked": matrix["summary"]["blocked"],
        "cpu_required": sorted(CPU_REQUIRED),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
