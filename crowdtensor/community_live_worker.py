"""Restricted executable entrypoint for a Community live stage worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .community_live_training import run_remote_worker


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["stage0", "stage1"], required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-committed-step", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=2700)
    parser.add_argument("--cache-dir", default="")
    args = parser.parse_args(argv)
    coordinator = str(os.environ.get("CROWDTENSOR_PRIVATE_COORDINATOR_URL") or "")
    token = str(os.environ.get("CROWDTENSOR_PRIVATE_MINER_TOKEN") or "")
    if not coordinator or not token:
        raise SystemExit("private coordinator environment is required")
    report = run_remote_worker(
        coordinator_url=coordinator,
        token=token,
        role=args.role,
        backend=args.backend,
        output_path=args.output,
        max_committed_step=args.max_committed_step,
        timeout_seconds=args.timeout_seconds,
        cache_dir=args.cache_dir,
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "role": report["role"],
                "last_committed_step": report["last_committed_step"],
                "content_hash": report["content_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
