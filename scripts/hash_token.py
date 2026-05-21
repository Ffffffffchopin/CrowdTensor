#!/usr/bin/env python3
"""Generate a CrowdTensorD sha256 token verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.auth import hash_token  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hash a CrowdTensorD token for Coordinator configuration.")
    parser.add_argument("token", help="plaintext token to hash")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("token must be non-empty")
    return args


def main() -> None:
    args = parse_args()
    token_hash = hash_token(args.token)
    if args.json:
        print(json.dumps({"algorithm": "sha256", "token_hash": token_hash}, sort_keys=True))
    else:
        print(token_hash)


if __name__ == "__main__":
    main()
