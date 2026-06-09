#!/usr/bin/env python3
"""Create a remote Miner invite and hashed registry entry."""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shlex
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
    stage: str = "both",
    backend: str = "cpu",
    hf_model_id: str = "sshleifer/tiny-gpt2",
    max_tasks: int = 0,
    max_runtime_seconds: float = 0.0,
    trust_tier: str = "new",
    quota_task_limit: int = 0,
    claim_rate_limit: int = 0,
    claim_rate_window_seconds: float = 0.0,
    reward_account: str = "",
    invite_file: Path | None = None,
) -> dict:
    miner_name = str(miner_id or "").strip()
    if not miner_name:
        raise ValueError("miner_id is required")
    coordinator = str(coordinator_url or "").strip().rstrip("/")
    if not coordinator:
        raise ValueError("coordinator_url is required")
    stage_value = str(stage or "both").strip()
    if stage_value not in {"stage0", "stage1", "both"}:
        raise ValueError("stage must be one of: stage0, stage1, both")
    backend_value = str(backend or "cpu").strip()
    if backend_value not in {"cpu", "cuda"}:
        raise ValueError("backend must be one of: cpu, cuda")
    model_id = str(hf_model_id or "sshleifer/tiny-gpt2").strip()
    if not model_id:
        raise ValueError("hf_model_id is required")
    task_limit = int(max_tasks or 0)
    if task_limit < 0:
        raise ValueError("max_tasks must be non-negative")
    runtime_limit = float(max_runtime_seconds or 0.0)
    if runtime_limit < 0:
        raise ValueError("max_runtime_seconds must be non-negative")
    quota_limit = int(quota_task_limit or 0)
    if quota_limit < 0:
        raise ValueError("quota_task_limit must be non-negative")
    rate_limit = int(claim_rate_limit or 0)
    if rate_limit < 0:
        raise ValueError("claim_rate_limit must be non-negative")
    rate_window = float(claim_rate_window_seconds or 0.0)
    if rate_window < 0:
        raise ValueError("claim_rate_window_seconds must be non-negative")
    if (rate_limit > 0) != (rate_window > 0):
        raise ValueError("claim_rate_limit and claim_rate_window_seconds must be set together")

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
        "join_policy": {
            "schema": "crowdtensor_miner_join_policy_v1",
            "coordinator_url": coordinator,
            "stage": stage_value,
            "backend": backend_value,
            "hf_model_id": model_id,
            "max_tasks": task_limit,
            "max_runtime_seconds": runtime_limit,
            "trust_tier": str(trust_tier or "new"),
            "quota_task_limit": quota_limit,
            "claim_rate_limit": rate_limit,
            "claim_rate_window_seconds": rate_window,
            "reward_account": str(reward_account or ""),
            "read_only_workload": "real_llm_sharded_infer",
            "not_production": True,
        },
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
    legacy_command = (
        f"CROWDTENSOR_MINER_TOKEN={plaintext_token} "
        f"crowdtensor-miner --coordinator {coordinator} --miner-id {miner_name} "
        "--max-tasks 1"
    )
    product_command_parts = [
        "crowdtensor",
        "join",
        "--coordinator-url",
        coordinator,
        "--miner-id",
        miner_name,
        "--stage",
        stage_value,
        "--backend",
        backend_value,
        "--hf-model-id",
        model_id,
    ]
    if task_limit > 0:
        product_command_parts.extend(["--max-tasks", str(task_limit)])
    if runtime_limit > 0:
        product_command_parts.extend(["--max-runtime-seconds", str(runtime_limit)])
    product_command_parts.append("--run")
    join_invite = {
        "schema": "crowdtensor_miner_join_invite_v1",
        "coordinator_url": coordinator,
        "miner_id": miner_name,
        "stage": stage_value,
        "backend": backend_value,
        "hf_model_id": model_id,
        "miner_token": plaintext_token,
        "token_hash": token_hash,
        "policy": entry["join_policy"],
        "public_artifact_safe": False,
    }
    invite_code = base64.urlsafe_b64encode(
        json.dumps(join_invite, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    invite_file_path = ""
    if invite_file is not None:
        invite_file.parent.mkdir(parents=True, exist_ok=True)
        invite_file.write_text(json.dumps(join_invite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        invite_file_path = str(invite_file)
    return {
        "coordinator_url": coordinator,
        "env": {
            "CROWDTENSOR_MINER_TOKEN": plaintext_token,
        },
        "invite_file": invite_file_path,
        "join_invite": join_invite,
        "join_invite_code": invite_code,
        "miner_id": miner_name,
        "product_join_command": (
            f"CROWDTENSOR_MINER_TOKEN={shlex.quote(plaintext_token)} "
            f"{shlex.join(product_command_parts)}"
        ),
        "registry": str(registry_path),
        "run_command": legacy_command,
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
    parser.add_argument("--stage", choices=["stage0", "stage1", "both"], default="both")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    parser.add_argument("--trust-tier", default="new")
    parser.add_argument("--quota-task-limit", type=int, default=0)
    parser.add_argument("--claim-rate-limit", type=int, default=0)
    parser.add_argument("--claim-rate-window-seconds", type=float, default=0.0)
    parser.add_argument("--reward-account", default="")
    parser.add_argument("--invite-file", default="", help="write the plaintext Miner join invite JSON to this private path")
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
            stage=args.stage,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            max_tasks=args.max_tasks,
            max_runtime_seconds=args.max_runtime_seconds,
            trust_tier=args.trust_tier,
            quota_task_limit=args.quota_task_limit,
            claim_rate_limit=args.claim_rate_limit,
            claim_rate_window_seconds=args.claim_rate_window_seconds,
            reward_account=args.reward_account,
            invite_file=Path(args.invite_file) if args.invite_file else None,
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
    print()
    print("Product join command:")
    print(invite["product_join_command"])
    if invite.get("invite_file"):
        print()
        print("Invite file:")
        print(invite["invite_file"])
    print()
    print("Invite code:")
    print(invite["join_invite_code"])


if __name__ == "__main__":
    main()
