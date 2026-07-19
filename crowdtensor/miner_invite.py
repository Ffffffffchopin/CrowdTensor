"""Packaged Miner invite creation used by the installed CrowdTensor CLI."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shlex
import time
from pathlib import Path

from .auth import hash_token, validate_token_verifier


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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
    peer_bootstrap: str = "",
    p2p_backend: str = "lite",
    swarm_id: str = "default",
    route_preference: str = "",
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
    runtime_limit = float(max_runtime_seconds or 0.0)
    quota_limit = int(quota_task_limit or 0)
    rate_limit = int(claim_rate_limit or 0)
    rate_window = float(claim_rate_window_seconds or 0.0)
    if any(value < 0 for value in (task_limit, runtime_limit, quota_limit, rate_limit, rate_window)):
        raise ValueError("invite limits must be non-negative")
    if (rate_limit > 0) != (rate_window > 0):
        raise ValueError(
            "claim_rate_limit and claim_rate_window_seconds must be set together"
        )
    peer_bootstrap_url = str(peer_bootstrap or "").strip().rstrip("/")
    p2p_backend_value = str(p2p_backend or "lite").strip()
    if p2p_backend_value not in {"lite", "real"}:
        raise ValueError("p2p_backend must be one of: lite, real")
    swarm_id_value = str(swarm_id or "default").strip() or "default"
    route_preference_value = str(route_preference or "").strip()
    if route_preference_value not in {"", "coordinator-url", "peer-bootstrap"}:
        raise ValueError(
            "route_preference must be one of: coordinator-url, peer-bootstrap"
        )
    if route_preference_value == "peer-bootstrap" and not peer_bootstrap_url:
        raise ValueError("route_preference=peer-bootstrap requires peer_bootstrap")
    if not route_preference_value:
        route_preference_value = (
            "peer-bootstrap" if peer_bootstrap_url else "coordinator-url"
        )

    plaintext_token = token or secrets.token_urlsafe(32)
    verifier = hash_token(plaintext_token)
    registry = load_registry(registry_path)
    miners = registry.setdefault("miners", [])
    now = int(time.time())
    policy = {
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
    }
    entry = {
        "enabled": True,
        "label": str(label or ""),
        "miner_id": miner_name,
        "token": validate_token_verifier(verifier, field_name="miner token"),
        "updated_at": now,
        "join_policy": policy,
    }
    existing_index = next(
        (
            index
            for index, item in enumerate(miners)
            if isinstance(item, dict) and item.get("miner_id") == miner_name
        ),
        None,
    )
    if existing_index is not None and not replace:
        raise ValueError(
            f"miner_id {miner_name!r} already exists; pass --replace to update it"
        )
    if existing_index is None:
        entry["created_at"] = now
        miners.append(entry)
    else:
        previous = miners[existing_index]
        entry["created_at"] = (
            int(previous.get("created_at", now))
            if isinstance(previous, dict)
            else now
        )
        miners[existing_index] = entry

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
        product_command_parts.extend(
            ["--max-runtime-seconds", str(runtime_limit)]
        )
    invite = {
        "schema": "crowdtensor_miner_join_invite_v1",
        "coordinator_url": coordinator,
        "miner_id": miner_name,
        "stage": stage_value,
        "backend": backend_value,
        "hf_model_id": model_id,
        "miner_token": plaintext_token,
        "token_hash": verifier,
        "policy": policy,
        "public_artifact_safe": False,
    }
    if peer_bootstrap_url:
        discovery = {
            "schema": "crowdtensor_miner_join_discovery_v1",
            "enabled": True,
            "peer_bootstrap": peer_bootstrap_url,
            "p2p_backend": p2p_backend_value,
            "swarm_id": swarm_id_value,
            "route_preference": route_preference_value,
            "not_nat_traversal": True,
            "public_artifact_safe": True,
        }
        policy["discovery"] = discovery
        invite["discovery"] = discovery
        product_command_parts.extend(
            [
                "--p2p",
                "--p2p-backend",
                p2p_backend_value,
                "--peer-bootstrap",
                peer_bootstrap_url,
                "--swarm-id",
                swarm_id_value,
            ]
        )
    product_command_parts.append("--run")
    write_registry(registry_path, registry)
    invite_code = base64.urlsafe_b64encode(
        json.dumps(invite, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    invite_file_path = ""
    if invite_file is not None:
        invite_file.parent.mkdir(parents=True, exist_ok=True)
        invite_file.write_text(
            json.dumps(invite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        invite_file_path = str(invite_file)
    legacy_command = (
        f"CROWDTENSOR_MINER_TOKEN={plaintext_token} "
        f"crowdtensor-miner --coordinator {coordinator} --miner-id {miner_name} "
        "--max-tasks 1"
    )
    return {
        "coordinator_url": coordinator,
        "env": {"CROWDTENSOR_MINER_TOKEN": plaintext_token},
        "invite_file": invite_file_path,
        "join_invite": invite,
        "join_invite_code": invite_code,
        "miner_id": miner_name,
        "product_join_command": (
            f"CROWDTENSOR_MINER_TOKEN={shlex.quote(plaintext_token)} "
            f"{shlex.join(product_command_parts)}"
        ),
        "registry": str(registry_path),
        "run_command": legacy_command,
        "token_hash": verifier,
    }
