#!/usr/bin/env python3
"""Run the bounded real SmolLM2 two-process LoRA live proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crowdtensor.smollm_training import run_two_stage_lora


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument(
        "--node-scope",
        choices=["local logical multi-process", "Kaggle logical multi-node"],
        default="local logical multi-process",
    )
    parser.add_argument("--clean-install", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if len(devices) != 2:
        raise SystemExit("--devices must contain exactly two device values")
    report = run_two_stage_lora(
        args.output_dir,
        steps=args.steps,
        sequence_length=args.sequence_length,
        devices=(devices[0], devices[1]),
        timeout_seconds=args.timeout_seconds,
        node_scope=args.node_scope,
        clean_install=args.clean_install,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"smollm_live_ok={report['ok']} logical_miners={report['logical_miner_count']} "
            f"steps={len(report['committed_step_ids'])} reload={report['reload']['adapter_reload_verified']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
