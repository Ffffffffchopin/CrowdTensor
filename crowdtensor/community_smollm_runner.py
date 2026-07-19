"""Installed-module runner for the Kaggle dual-GPU SmolLM proof."""

from __future__ import annotations

import argparse

from .smollm_training import run_two_stage_lora


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    args = parser.parse_args(argv)
    report = run_two_stage_lora(
        args.output_dir,
        steps=args.steps,
        sequence_length=8,
        devices=("cuda:0", "cuda:1"),
        timeout_seconds=args.timeout_seconds,
        node_scope="Kaggle logical multi-node",
        clean_install=True,
    )
    print(f"smollm_dual_gpu_ok={{report['ok']}}")


if __name__ == "__main__":
    main()
