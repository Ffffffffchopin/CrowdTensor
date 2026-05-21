#!/usr/bin/env python3
"""Create a remote Miner invite and hashed registry entry."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.auth import hash_token, validate_token_verifier  # noqa: E402


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {"miners": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid registry JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("registry must be a JSON object")
    miners = payload.get("miners")
    if miners is None:
        payload["miners"] = []
    elif not isinstance(miners, list):
        raise ValueError("registry miners must be a list")
    return payload


def write_registry(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def create_invite(
    *,
    registry_path: Path,
    miner_id: str,
    coordinator_url: str,
    label: str = "",
    token: str = "",
    replace: bool = False,
) -> dict:
    miner_name = str(miner_id or "").strip()
    if not miner_name:
        raise ValueError("miner_id is required")
    coordinator = str(coordinator_url or "").strip().rstrip("/")
    if not coordinator:
        raise ValueError("coordinator_url is required")

    plaintext_token = token or secrets.token_urlsafe(32)
    token_hash = hash_token(plaintext_token)
    registry = load_registry(registry_path)
    miners = registry.setdefault("miners", [])
    now = int(time.time())
    entry = {
        "enabled": True,
        "label": str(label or ""),
        "miner_id": miner_name,
        "token": validate_token_verifier(token_hash, field_name="miner token"),
        "updated_at": now,
    }

    existing_index = next(
        (index for index, item in enumerate(miners) if isinstance(item, dict) and item.get("miner_id") == miner_name),
        None,
    )
    if existing_index is not None and not replace:
        raise ValueError(f"miner_id {miner_name!r} already exists; pass --replace to update it")
    if existing_index is None:
        entry["created_at"] = now
        miners.append(entry)
    else:
        previous = miners[existing_index]
        entry["created_at"] = int(previous.get("created_at", now)) if isinstance(previous, dict) else now
        miners[existing_index] = entry

    write_registry(registry_path, registry)
    command = (
        f"CROWDTENSOR_MINER_TOKEN={plaintext_token} "
        f"crowdtensor-miner --coordinator {coordinator} --miner-id {miner_name} "
        "--max-tasks 1"
    )
    return {
        "coordinator_url": coordinator,
        "env": {
            "CROWDTENSOR_MINER_TOKEN": plaintext_token,
        },
        "miner_id": miner_name,
        "registry": str(registry_path),
        "run_command": command,
        "token_hash": token_hash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CrowdTensorD remote Miner invite.")
    parser.add_argument("--registry", required=True, help="path to miner token registry JSON")
    parser.add_argument("--miner-id", required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--token", default="", help="plaintext token to use; defaults to a generated random token")
    parser.add_argument("--replace", action="store_true", help="replace an existing registry entry for this miner_id")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        invite = create_invite(
            registry_path=Path(args.registry),
            miner_id=args.miner_id,
            coordinator_url=args.coordinator_url,
            label=args.label,
            token=args.token,
            replace=args.replace,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(invite, sort_keys=True))
        return

    print("CrowdTensorD remote Miner invite")
    print(f"miner_id: {invite['miner_id']}")
    print(f"registry: {invite['registry']}")
    print(f"token_hash: {invite['token_hash']}")
    print()
    print("Remote .env:")
    print(f"CROWDTENSOR_MINER_TOKEN={invite['env']['CROWDTENSOR_MINER_TOKEN']}")
    print()
    print("Remote command:")
    print(invite["run_command"])


if __name__ == "__main__":
    main()
