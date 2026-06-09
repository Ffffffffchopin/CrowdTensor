#!/usr/bin/env python3
"""FastAPI Coordinator for CrowdTensorD Phase 1."""

import argparse
import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from crowdtensor.auth import token_matches, validate_token_verifier
from crowdtensor.protocol import (
    DEFAULT_PROTOCOL_VERSION,
    DEFAULT_WORKLOAD_TYPE,
    LeaseConflict,
    NoTaskAvailable,
    REAL_LLM_SHARDED_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
    ResultRejected,
    WORKLOAD_EXTERNAL_LLM_INFER,
    WORKLOAD_MICRO_LLM_SHARDED_INFER,
    WORKLOAD_MODEL_BUNDLE_INFER,
    WORKLOAD_REAL_LLM_SHARDED_INFER,
    WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
)
from crowdtensor.outer_optimizer import (
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    OPTIMIZER_DILOCO_MOMENTUM,
    SUPPORTED_DELTA_FORMATS,
    SUPPORTED_OUTER_OPTIMIZERS,
)
from crowdtensor.state_store import StateStore


SERVICE_NAME = "crowdtensord-coordinator"
SERVICE_VERSION = "0.1.0a0"
API_STATUS = "alpha"
OPERATOR_ROLE_OWNER = "owner"
OPERATOR_ROLE_ADMIN = "admin"
OPERATOR_ROLE_ACCOUNTING = "accounting"
OPERATOR_ROLE_AUDITOR = "auditor"
OPERATOR_ALLOWED_ROLES = {
    OPERATOR_ROLE_OWNER,
    OPERATOR_ROLE_ADMIN,
    OPERATOR_ROLE_ACCOUNTING,
    OPERATOR_ROLE_AUDITOR,
}


def load_miner_token_registry(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load a minimal per-miner token registry from JSON."""
    if not path:
        return {}
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid miner token registry JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"could not read miner token registry: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("miner token registry must be a JSON object")
    miners = payload.get("miners")
    if not isinstance(miners, list):
        raise ValueError("miner token registry must contain a miners list")

    registry: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(miners):
        if not isinstance(entry, dict):
            raise ValueError(f"miner token registry entry {index} must be an object")
        miner_id = str(entry.get("miner_id", "")).strip()
        token = str(entry.get("token", "")).strip()
        if not miner_id:
            raise ValueError(f"miner token registry entry {index} missing miner_id")
        if not token:
            raise ValueError(f"miner token registry entry {index} missing token")
        token = validate_token_verifier(token, field_name=f"miner token registry entry {index} token")
        if miner_id in registry:
            raise ValueError(f"duplicate miner token registry miner_id: {miner_id}")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"miner token registry entry {index} enabled must be boolean")
        label = entry.get("label", "")
        registry[miner_id] = {
            "token": token,
            "enabled": enabled,
            "label": str(label or ""),
            "join_policy": _safe_miner_join_policy(entry.get("join_policy")),
        }
    return registry


def load_operator_token_registry(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load per-operator control-plane tokens from JSON."""
    if not path:
        return {}
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid operator token registry JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"could not read operator token registry: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("operator token registry must be a JSON object")
    operators = payload.get("operators")
    if not isinstance(operators, list):
        raise ValueError("operator token registry must contain an operators list")

    registry: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(operators):
        if not isinstance(entry, dict):
            raise ValueError(f"operator token registry entry {index} must be an object")
        operator_id = str(entry.get("operator_id", "")).strip()
        token = str(entry.get("token", "")).strip()
        if not operator_id:
            raise ValueError(f"operator token registry entry {index} missing operator_id")
        if not token:
            raise ValueError(f"operator token registry entry {index} missing token")
        token = validate_token_verifier(token, field_name=f"operator token registry entry {index} token")
        if operator_id in registry:
            raise ValueError(f"duplicate operator token registry operator_id: {operator_id}")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"operator token registry entry {index} enabled must be boolean")
        roles_raw = entry.get("roles", [OPERATOR_ROLE_AUDITOR])
        if isinstance(roles_raw, str):
            roles = {roles_raw.strip()}
        elif isinstance(roles_raw, list):
            roles = {str(role).strip() for role in roles_raw}
        else:
            raise ValueError(f"operator token registry entry {index} roles must be a string or list")
        roles = {role for role in roles if role}
        unknown = sorted(role for role in roles if role not in OPERATOR_ALLOWED_ROLES)
        if unknown:
            raise ValueError(f"operator token registry entry {index} has unknown roles: {', '.join(unknown)}")
        if not roles:
            raise ValueError(f"operator token registry entry {index} must include at least one role")
        label = entry.get("label", "")
        registry[operator_id] = {
            "token": token,
            "enabled": enabled,
            "label": str(label or ""),
            "roles": sorted(roles),
        }
    return registry


def operator_registry_summary(registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    operators: list[dict[str, Any]] = []
    for operator_id, entry in sorted(registry.items()):
        operators.append({
            "operator_id": operator_id,
            "enabled": bool(entry.get("enabled")),
            "label": str(entry.get("label") or ""),
            "roles": list(entry.get("roles") or []),
        })
    return {
        "schema": "crowdtensor_operator_registry_summary_v1",
        "operator_count": len(registry),
        "operators": operators,
        "plaintext_tokens_public": False,
        "public_artifact_safe": True,
    }


def _safe_miner_join_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    return {
        "schema": str(policy.get("schema") or "crowdtensor_miner_join_policy_v1"),
        "coordinator_url_present": bool(str(policy.get("coordinator_url") or "").strip()),
        "stage": str(policy.get("stage") or ""),
        "backend": str(policy.get("backend") or ""),
        "hf_model_id": str(policy.get("hf_model_id") or ""),
        "max_tasks": int(policy.get("max_tasks") or 0),
        "max_runtime_seconds": float(policy.get("max_runtime_seconds") or 0.0),
        "trust_tier": str(policy.get("trust_tier") or "new"),
        "quota_task_limit": int(policy.get("quota_task_limit") or 0),
        "claim_rate_limit": int(policy.get("claim_rate_limit") or 0),
        "claim_rate_window_seconds": float(policy.get("claim_rate_window_seconds") or 0.0),
        "reward_account_present": bool(str(policy.get("reward_account") or "").strip()),
        "read_only_workload": str(policy.get("read_only_workload") or ""),
        "not_production": policy.get("not_production", True) is not False,
    }


def miner_registry_policy_summary(registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    miners: list[dict[str, Any]] = []
    for miner_id, entry in sorted(registry.items()):
        policy = entry.get("join_policy") if isinstance(entry.get("join_policy"), dict) else {}
        if not policy:
            continue
        miners.append({
            "miner_id": miner_id,
            "enabled": bool(entry.get("enabled")),
            "label": str(entry.get("label") or ""),
            "policy": dict(policy),
        })
    return {
        "schema": "crowdtensor_miner_registry_policy_summary_v1",
        "miner_count": len(registry),
        "policy_count": len(miners),
        "miners": miners,
        "plaintext_tokens_public": False,
        "reward_accounts_public": False,
        "public_artifact_safe": True,
    }


def _policy_allowed_stage_capabilities(policy: dict[str, Any]) -> list[str]:
    stage = str(policy.get("stage") or "both").strip()
    backend = str(policy.get("backend") or "cpu").strip()
    if backend == "cuda":
        return {
            "stage0": [REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY],
            "stage1": [REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY],
            "both": [
                REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
                REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
                REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
            ],
        }.get(stage, [])
    return {
        "stage0": [REAL_LLM_SHARDED_STAGE0_CAPABILITY],
        "stage1": [REAL_LLM_SHARDED_STAGE1_CAPABILITY],
        "both": [
            REAL_LLM_SHARDED_STAGE0_CAPABILITY,
            REAL_LLM_SHARDED_STAGE1_CAPABILITY,
            REAL_LLM_SHARDED_BOTH_CAPABILITY,
        ],
    }.get(stage, [])


def enforce_miner_join_policy(
    *,
    miner_id: str,
    capabilities: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    entry = registry.get(str(miner_id or ""))
    if not entry:
        return dict(capabilities or {}), ""
    policy = entry.get("join_policy") if isinstance(entry.get("join_policy"), dict) else {}
    if not policy:
        return dict(capabilities or {}), ""

    restricted = dict(capabilities or {})
    allowed_workload = str(policy.get("read_only_workload") or WORKLOAD_REAL_LLM_SHARDED_INFER)
    supported_workloads = restricted.get("supported_workloads")
    if supported_workloads is None:
        advertised_workloads = {DEFAULT_WORKLOAD_TYPE}
    elif isinstance(supported_workloads, str):
        advertised_workloads = {supported_workloads}
    else:
        advertised_workloads = {str(value) for value in supported_workloads}
    if allowed_workload not in advertised_workloads:
        return restricted, "join_policy_workload_not_advertised"
    restricted["supported_workloads"] = [allowed_workload]

    backend = str(policy.get("backend") or "").strip()
    if backend and backend != "any":
        advertised_backend = str(restricted.get("backend") or "")
        if advertised_backend and advertised_backend != backend:
            return restricted, "join_policy_backend_mismatch"
        restricted["backend"] = backend

    model_id = str(policy.get("hf_model_id") or "").strip()
    runtime = restricted.get("real_llm_runtime") if isinstance(restricted.get("real_llm_runtime"), dict) else {}
    if model_id or backend in {"cpu", "cuda"}:
        advertised_model = str(runtime.get("model_id") or "").strip()
        if model_id and advertised_model and advertised_model != model_id:
            return restricted, "join_policy_model_mismatch"
        runtime = dict(runtime)
        if model_id:
            runtime["model_id"] = model_id
        if backend == "cuda":
            runtime["adapter_kind"] = "hf_transformers_cuda"
        elif backend == "cpu":
            runtime["adapter_kind"] = "hf_transformers_cpu"
        restricted["real_llm_runtime"] = runtime

    allowed_stage_capabilities = _policy_allowed_stage_capabilities(policy)
    if allowed_stage_capabilities:
        advertised_caps = restricted.get("real_llm_sharded_stage_capabilities")
        if isinstance(advertised_caps, str):
            advertised_stage_capabilities = {advertised_caps}
        elif advertised_caps is None:
            advertised_stage_capabilities = set(allowed_stage_capabilities)
        else:
            advertised_stage_capabilities = {str(value) for value in advertised_caps}
        allowed_set = set(allowed_stage_capabilities)
        if advertised_stage_capabilities and not (advertised_stage_capabilities & allowed_set):
            return restricted, "join_policy_stage_mismatch"
        restricted["real_llm_sharded_stage_capabilities"] = allowed_stage_capabilities
        restricted["real_llm_sharded_stage_role"] = str(policy.get("stage") or "both")

    return restricted, ""


def miner_join_policy_for(miner_id: str, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entry = registry.get(str(miner_id or ""))
    if not isinstance(entry, dict):
        return {}
    policy = entry.get("join_policy")
    return dict(policy) if isinstance(policy, dict) else {}


def apply_accounting_policy_summary(accounting: dict[str, Any], registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = json.loads(json.dumps(accounting))
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        policy = miner_join_policy_for(str(row.get("miner_id") or ""), registry)
        if not policy:
            continue
        row["join_policy"] = {
            "trust_tier": str(policy.get("trust_tier") or "new"),
            "quota_task_limit": int(policy.get("quota_task_limit") or 0),
            "claim_rate_limit": int(policy.get("claim_rate_limit") or 0),
            "claim_rate_window_seconds": float(policy.get("claim_rate_window_seconds") or 0.0),
            "reward_account_present": bool(
                policy.get("reward_account_present")
                or str(policy.get("reward_account") or "").strip()
            ),
            "read_only_workload": str(policy.get("read_only_workload") or ""),
        }
    for item in payload.get("miner_totals", {}).values():
        if not isinstance(item, dict):
            continue
        policy = miner_join_policy_for(str(item.get("miner_id") or ""), registry)
        if not policy:
            continue
        item["join_policy"] = {
            "trust_tier": str(policy.get("trust_tier") or "new"),
            "quota_task_limit": int(policy.get("quota_task_limit") or 0),
            "claim_rate_limit": int(policy.get("claim_rate_limit") or 0),
            "claim_rate_window_seconds": float(policy.get("claim_rate_window_seconds") or 0.0),
            "reward_account_present": bool(
                policy.get("reward_account_present")
                or str(policy.get("reward_account") or "").strip()
            ),
            "read_only_workload": str(policy.get("read_only_workload") or ""),
        }
    payload["registry_policy_joined"] = bool(registry)
    return payload


def _safe_settlement_policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    reward_account_present = bool(
        policy.get("reward_account_present")
        or str(policy.get("reward_account") or "").strip()
    )
    return {
        "trust_tier": str(policy.get("trust_tier") or "new"),
        "quota_task_limit": int(policy.get("quota_task_limit") or 0),
        "claim_rate_limit": int(policy.get("claim_rate_limit") or 0),
        "claim_rate_window_seconds": float(policy.get("claim_rate_window_seconds") or 0.0),
        "reward_account_present": reward_account_present,
        "read_only_workload": str(policy.get("read_only_workload") or ""),
    }


def apply_settlement_policy_summary(settlement: dict[str, Any], registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = json.loads(json.dumps(settlement))
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        policy = miner_join_policy_for(str(row.get("miner_id") or ""), registry)
        if not policy:
            continue
        summary = _safe_settlement_policy_summary(policy)
        row["join_policy"] = summary
        row["reward_account_present"] = bool(summary["reward_account_present"])
        row["settlement_status"] = "payable_draft" if row["reward_account_present"] else "missing_reward_account"
    for item in payload.get("settlement_totals", {}).values():
        if not isinstance(item, dict):
            continue
        policy = miner_join_policy_for(str(item.get("miner_id") or ""), registry)
        if not policy:
            continue
        summary = _safe_settlement_policy_summary(policy)
        item["join_policy"] = summary
        item["reward_account_present"] = bool(summary["reward_account_present"])
        item["settlement_status"] = "payable_draft" if item["reward_account_present"] else "missing_reward_account"
    payload["registry_policy_joined"] = bool(registry)
    return payload


def _metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return str(numeric)


def _metric_label(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _nested_workload_count(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    total = 0
    for workloads in value.values():
        if isinstance(workloads, dict):
            total += len(workloads)
        elif isinstance(workloads, list):
            total += len(workloads)
    return total


def _trust_override_mode_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts = {"allow": 0, "block": 0}
    overrides = summary.get("miner_trust_overrides") or {}
    if not isinstance(overrides, dict):
        return counts
    for workload_overrides in overrides.values():
        if not isinstance(workload_overrides, dict):
            continue
        for override in workload_overrides.values():
            if not isinstance(override, dict):
                continue
            mode = str(override.get("mode", ""))
            if mode in counts:
                counts[mode] += 1
    return counts


def build_metrics_text(summary: dict[str, Any]) -> str:
    """Render safe aggregate Coordinator metrics in Prometheus text format."""
    lines: list[str] = []

    def add(name: str, value: Any, labels: dict[str, Any] | None = None) -> None:
        if labels:
            label_text = ",".join(
                f'{key}="{_metric_label(label_value)}"'
                for key, label_value in sorted(labels.items())
            )
            lines.append(f"{name}{{{label_text}}} {_metric_value(value)}")
        else:
            lines.append(f"{name} {_metric_value(value)}")

    lines.extend([
        "# HELP crowdtensord_event_index Last applied append-only event index.",
        "# TYPE crowdtensord_event_index gauge",
    ])
    add("crowdtensord_event_index", summary.get("event_index", 0))

    model = summary.get("model") or {}
    lines.extend([
        "# HELP crowdtensord_model_step Current model step counters.",
        "# TYPE crowdtensord_model_step gauge",
    ])
    add("crowdtensord_model_step", model.get("global_step", 0), {"step": "global"})
    add("crowdtensord_model_step", model.get("optimizer_step", 0), {"step": "optimizer"})
    add("crowdtensord_model_step", model.get("adapter_step", 0), {"step": "adapter"})
    micro_transformer = model.get("micro_transformer") or {}
    if isinstance(micro_transformer, dict):
        add(
            "crowdtensord_model_step",
            micro_transformer.get("optimizer_step", 0),
            {"step": "micro_transformer"},
        )
    model_bundle = model.get("model_bundle") or {}
    if isinstance(model_bundle, dict):
        add(
            "crowdtensord_model_step",
            model_bundle.get("optimizer_step", 0),
            {"step": "model_bundle"},
        )

    lines.extend([
        "# HELP crowdtensord_task_count Current task count by status.",
        "# TYPE crowdtensord_task_count gauge",
    ])
    task_counts = summary.get("task_counts") or {}
    if isinstance(task_counts, dict):
        for status in ("queued", "leased", "completed", "rejected"):
            add("crowdtensord_task_count", task_counts.get(status, 0), {"status": status})

    lines.extend([
        "# HELP crowdtensord_results_total Accepted and rejected result counters.",
        "# TYPE crowdtensord_results_total counter",
    ])
    add("crowdtensord_results_total", summary.get("accepted_results", 0), {"result": "accepted"})
    add("crowdtensord_results_total", summary.get("rejected_results", 0), {"result": "rejected"})

    lines.extend([
        "# HELP crowdtensord_model_updates_total Applied model update counters.",
        "# TYPE crowdtensord_model_updates_total counter",
    ])
    add("crowdtensord_model_updates_total", summary.get("model_updates", 0), {"target": "dense"})
    add("crowdtensord_model_updates_total", summary.get("adapter_updates", 0), {"target": "adapter"})
    add(
        "crowdtensord_model_updates_total",
        summary.get("micro_transformer_updates", 0),
        {"target": "micro_transformer"},
    )
    add(
        "crowdtensord_model_updates_total",
        summary.get("model_bundle_updates", 0),
        {"target": "model_bundle"},
    )

    lines.extend([
        "# HELP crowdtensord_audit_results_total Replay audit counters.",
        "# TYPE crowdtensord_audit_results_total counter",
    ])
    add("crowdtensord_audit_results_total", summary.get("audit_results", 0), {"result": "total"})
    add("crowdtensord_audit_results_total", summary.get("audit_rejections", 0), {"result": "rejected"})

    lines.extend([
        "# HELP crowdtensord_claims_total Claim routing counters.",
        "# TYPE crowdtensord_claims_total counter",
    ])
    add("crowdtensord_claims_total", summary.get("blocked_claims", 0), {"result": "blocked"})
    add("crowdtensord_claims_total", summary.get("incompatible_claims", 0), {"result": "incompatible"})

    lines.extend([
        "# HELP crowdtensord_miner_workload_blocks Current blocked miner/workload pairs.",
        "# TYPE crowdtensord_miner_workload_blocks gauge",
    ])
    add("crowdtensord_miner_workload_blocks", _nested_workload_count(summary.get("quarantined_miners")), {"source": "auto"})
    add(
        "crowdtensord_miner_workload_blocks",
        _nested_workload_count(summary.get("effective_quarantined_miners")),
        {"source": "effective"},
    )
    add("crowdtensord_miner_workload_blocks", _nested_workload_count(summary.get("manual_blocked_miners")), {"source": "manual"})

    lines.extend([
        "# HELP crowdtensord_trust_overrides Current manual trust override count by mode.",
        "# TYPE crowdtensord_trust_overrides gauge",
    ])
    for mode, count in _trust_override_mode_counts(summary).items():
        add("crowdtensord_trust_overrides", count, {"mode": mode})

    lines.extend([
        "# HELP crowdtensord_staleness Current accepted-result staleness summary.",
        "# TYPE crowdtensord_staleness gauge",
    ])
    add("crowdtensord_staleness", summary.get("max_staleness", 0), {"summary": "max"})
    add("crowdtensord_staleness", summary.get("avg_staleness", 0.0), {"summary": "avg"})

    return "\n".join(lines) + "\n"


def version_payload() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "default_workload_type": DEFAULT_WORKLOAD_TYPE,
        "api_status": API_STATUS,
    }


def readiness_payload(
    summary: dict[str, Any],
    *,
    miner_required: bool,
    observer_required: bool,
    admin_configured: bool,
    miner_registry_configured: bool,
    miner_policy_summary: dict[str, Any] | None = None,
    operator_registry_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "ok": True,
        **version_payload(),
        "event_index": summary.get("event_index", 0),
        "task_counts": dict(summary.get("task_counts") or {}),
        "task_lanes": [dict(lane) for lane in summary.get("task_lanes") or []],
        "auth": {
            "miner_required": bool(miner_required),
            "observer_required": bool(observer_required),
            "admin_configured": bool(admin_configured),
            "miner_registry_configured": bool(miner_registry_configured),
            "operator_registry_configured": bool(
                operator_registry_summary
                and int(operator_registry_summary.get("operator_count") or 0) > 0
            ),
        },
    }
    if miner_policy_summary:
        payload["miner_policy_summary"] = miner_policy_summary
    if operator_registry_summary:
        payload["operator_registry_summary"] = operator_registry_summary
    return payload


def create_app(
    *,
    state_dir: str | Path = "state",
    lease_seconds: float = 15.0,
    inner_steps: int = 500,
    backlog: int = 1,
    task_lanes: list[dict[str, Any]] | None = None,
    reaper_interval: float = 1.0,
    cors_origins: list[str] | None = None,
    replay_audit: bool = False,
    outer_optimizer: str = OPTIMIZER_DILOCO_MOMENTUM,
    delta_format: str = "dense_float",
    admin_token: str | None = None,
    operator_token_registry: str | Path | None = None,
    inference_session_rate_limit: int = 0,
    inference_session_rate_window_seconds: float = 0.0,
    miner_token: str | None = None,
    miner_token_registry: str | Path | None = None,
    observer_token: str | None = None,
    micro_llm_artifact: str | Path | None = None,
    real_llm_model_id: str = "",
    real_llm_backend: str = "hf_transformers_cpu",
    real_llm_partition_mode: str = "full",
    hf_cache_dir: str | Path | None = None,
):
    try:
        from fastapi import FastAPI, Header, HTTPException, Query, Response
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI is not installed. Run: pip install -r requirements.txt") from exc

    store = StateStore(
        state_dir,
        lease_seconds=lease_seconds,
        inner_steps=inner_steps,
        backlog=backlog,
        task_lanes=task_lanes,
        replay_audit=replay_audit,
        outer_optimizer=outer_optimizer,
        delta_format=delta_format,
        micro_llm_artifact=micro_llm_artifact,
        real_llm_model_id=real_llm_model_id or "sshleifer/tiny-gpt2",
        real_llm_backend=real_llm_backend,
        real_llm_partition_mode=real_llm_partition_mode,
        hf_cache_dir=hf_cache_dir,
    )
    configured_admin_token = admin_token if admin_token is not None else os.environ.get("CROWDTENSOR_ADMIN_TOKEN", "")
    configured_operator_registry_path = (
        operator_token_registry
        if operator_token_registry is not None
        else os.environ.get("CROWDTENSOR_OPERATOR_TOKEN_REGISTRY", "")
    )
    configured_operator_registry = load_operator_token_registry(configured_operator_registry_path)
    configured_miner_token = miner_token if miner_token is not None else os.environ.get("CROWDTENSOR_MINER_TOKEN", "")
    configured_miner_registry_path = (
        miner_token_registry
        if miner_token_registry is not None
        else os.environ.get("CROWDTENSOR_MINER_TOKEN_REGISTRY", "")
    )
    configured_miner_registry = load_miner_token_registry(configured_miner_registry_path)
    configured_observer_token = observer_token if observer_token is not None else os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", "")
    if configured_admin_token:
        configured_admin_token = validate_token_verifier(configured_admin_token, field_name="admin token")
    if configured_miner_token:
        configured_miner_token = validate_token_verifier(configured_miner_token, field_name="miner token")
    if configured_observer_token:
        configured_observer_token = validate_token_verifier(configured_observer_token, field_name="observer token")
    session_create_history: dict[str, list[float]] = {}
    session_create_history_lock = threading.Lock()
    configured_session_rate_limit = max(0, int(inference_session_rate_limit or 0))
    configured_session_rate_window_seconds = max(0.0, float(inference_session_rate_window_seconds or 0.0))
    if (configured_session_rate_limit > 0) != (configured_session_rate_window_seconds > 0):
        raise ValueError(
            "inference session rate limit and window seconds must be set together"
        )

    class ClaimRequest(BaseModel):
        miner_id: str = Field(default="anonymous", min_length=1)
        capabilities: dict[str, Any] = Field(default_factory=dict)

    class LeaseRequest(BaseModel):
        lease_token: str = Field(min_length=1)
        attempt: int = Field(ge=1)
        runtime_status: dict[str, Any] = Field(default_factory=dict)

    class ResultRequest(LeaseRequest):
        idempotency_key: str | None = None
        local_delta: list[float] | None = None
        pseudo_gradient: list[float] | None = None
        compressed_delta: dict[str, Any] | None = None
        probe_result: dict[str, Any] | None = None
        adapter_delta: dict[str, Any] | None = None
        bundle_delta: dict[str, Any] | None = None
        inference_result: dict[str, Any] | None = None
        inference_results: list[dict[str, Any]] | None = None
        external_llm_result: dict[str, Any] | None = None
        external_llm_results: list[dict[str, Any]] | None = None
        sharded_inference_result: dict[str, Any] | None = None
        metrics: dict[str, Any] = Field(default_factory=dict)

    class TrustOverrideRequest(BaseModel):
        miner_id: str = Field(min_length=1)
        workload_type: str = Field(min_length=1)
        mode: str = Field(min_length=1)
        reason: str = ""

    class InferenceSessionRequest(BaseModel):
        request_count: int = Field(default=4, ge=1, le=8)
        decode_steps: int = Field(default=1, ge=1, le=4)
        max_new_tokens: int = Field(default=1, ge=1, le=32)
        scenario_id: str = ""
        runtime: str = "python-cli"
        backend: str = "cpu"
        workload_type: str = WORKLOAD_MODEL_BUNDLE_INFER
        prompt: str | None = None
        prompt_texts: list[str] | None = None
        hf_model_id: str = ""
        partition_mode: str = ""

    def _token_operator_roles(token: str | None) -> set[str]:
        if token is None:
            return set()
        for entry in configured_operator_registry.values():
            if entry["enabled"] and token_matches(token, entry["token"]):
                return {str(role) for role in entry.get("roles") or []}
        return set()

    def _token_operator_identity(token: str | None) -> tuple[str, set[str]]:
        if token is None:
            return "", set()
        for operator_id, entry in configured_operator_registry.items():
            if entry["enabled"] and token_matches(token, entry["token"]):
                return str(operator_id), {str(role) for role in entry.get("roles") or []}
        return "", set()

    def _roles_allow(roles: set[str], required_roles: set[str]) -> bool:
        if OPERATOR_ROLE_OWNER in roles or OPERATOR_ROLE_ADMIN in roles:
            return True
        return bool(roles & required_roles)

    def require_admin(token: str | None, *, roles: set[str] | None = None) -> None:
        required_roles = set(roles or {OPERATOR_ROLE_ADMIN})
        if not configured_admin_token and not configured_operator_registry:
            raise HTTPException(status_code=403, detail="admin token is not configured")
        if configured_admin_token and token_matches(token, configured_admin_token):
            return
        operator_roles = _token_operator_roles(token)
        if operator_roles and _roles_allow(operator_roles, required_roles):
            return
        if operator_roles:
            raise HTTPException(status_code=403, detail="operator token lacks required role")
        raise HTTPException(status_code=403, detail="invalid admin token")

    def admin_subject(token: str | None) -> str:
        operator_id, _roles = _token_operator_identity(token)
        if operator_id:
            return f"operator:{operator_id}"
        if configured_admin_token and token_matches(token, configured_admin_token):
            return "legacy-admin"
        return "unknown"

    def enforce_inference_session_rate_limit(token: str | None, *, workload_type: str) -> None:
        if configured_session_rate_limit <= 0 or configured_session_rate_window_seconds <= 0:
            return
        subject = admin_subject(token)
        now = time.monotonic()
        cutoff = now - configured_session_rate_window_seconds
        with session_create_history_lock:
            history = [seen for seen in session_create_history.get(subject, []) if seen >= cutoff]
            if len(history) < configured_session_rate_limit:
                history.append(now)
                session_create_history[subject] = history
                return
            observed_count = len(history)
        store.record_control_plane_blocked(
            reason="inference_session_rate_limited",
            endpoint="/admin/inference-sessions",
            subject=subject,
            workload_type=workload_type,
            window_seconds=configured_session_rate_window_seconds,
            limit=configured_session_rate_limit,
            observed_count=observed_count,
        )
        raise HTTPException(
            status_code=429,
            detail={
                "reason": "inference_session_rate_limited",
                "window_seconds": configured_session_rate_window_seconds,
                "limit": configured_session_rate_limit,
            },
        )

    def token_matches_registry(token: str | None) -> bool:
        if token is None:
            return False
        for entry in configured_miner_registry.values():
            if entry["enabled"] and token_matches(token, entry["token"]):
                return True
        return False

    def token_matches_shared(token: str | None) -> bool:
        if not configured_miner_token:
            return False
        return token_matches(token, configured_miner_token)

    def require_miner(token: str | None) -> None:
        if not configured_miner_token and not configured_miner_registry:
            return
        if token_matches_registry(token) or token_matches_shared(token):
            return
        raise HTTPException(status_code=401, detail="invalid miner token")

    def require_claim_miner(miner_id: str, token: str | None) -> None:
        if not configured_miner_token and not configured_miner_registry:
            return
        miner_name = str(miner_id or "")
        if miner_name in configured_miner_registry:
            entry = configured_miner_registry[miner_name]
            if not entry["enabled"]:
                raise HTTPException(status_code=401, detail="miner token is disabled")
            if token_matches(token, entry["token"]):
                return
            raise HTTPException(status_code=401, detail="invalid miner token")
        if token_matches_shared(token):
            return
        raise HTTPException(status_code=401, detail="invalid miner token")

    def require_observer(token: str | None) -> None:
        if not configured_observer_token:
            return
        if not token_matches(token, configured_observer_token):
            raise HTTPException(status_code=401, detail="invalid observer token")

    async def reaper_loop() -> None:
        while True:
            store.reap_expired()
            await asyncio.sleep(reaper_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(reaper_loop())
        app.state.reaper_task = task
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="CrowdTensorD Coordinator", version=SERVICE_VERSION, lifespan=lifespan)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=[
                "content-type",
                "x-crowdtensor-admin-token",
                "x-crowdtensor-miner-token",
                "x-crowdtensor-observer-token",
            ],
        )
    app.state.store = store

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": SERVICE_NAME, "version": SERVICE_VERSION}

    @app.get("/version")
    def version() -> dict:
        return version_payload()

    @app.get("/ready")
    def ready() -> dict:
        try:
            store.reap_expired()
            summary = store.summary()
            return readiness_payload(
                summary,
                miner_required=bool(configured_miner_token or configured_miner_registry),
                observer_required=bool(configured_observer_token),
                admin_configured=bool(configured_admin_token or configured_operator_registry),
                miner_registry_configured=bool(configured_miner_registry),
                miner_policy_summary=miner_registry_policy_summary(configured_miner_registry),
                operator_registry_summary=operator_registry_summary(configured_operator_registry),
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"ok": False, "reason": str(exc)}) from exc

    @app.get("/state")
    def state(
        x_crowdtensor_observer_token: str | None = Header(default=None),
    ) -> dict:
        require_observer(x_crowdtensor_observer_token)
        store.reap_expired()
        return store.summary()

    @app.get("/metrics")
    def metrics(
        x_crowdtensor_observer_token: str | None = Header(default=None),
    ) -> Response:
        require_observer(x_crowdtensor_observer_token)
        store.reap_expired()
        return Response(
            build_metrics_text(store.summary()),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/admin/events")
    def admin_events(
        limit: int = Query(default=50, ge=0, le=500),
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token, roles={OPERATOR_ROLE_AUDITOR})
        return {
            "events": store.event_tail(limit=limit),
            "limit": min(500, max(0, int(limit))),
        }

    @app.get("/admin/results")
    def admin_results(
        limit: int = Query(default=50, ge=0, le=500),
        status: str = Query(default="any"),
        miner_id: str = Query(default=""),
        workload_type: str = Query(default=""),
        task_id: str = Query(default=""),
        session_id: str = Query(default=""),
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token, roles={OPERATOR_ROLE_AUDITOR})
        def query_value(value, default):
            if hasattr(value, "default"):
                value = value.default
            return default if value is None else value

        limit_value = query_value(limit, 50)
        status_value = query_value(status, "any")
        miner_value = query_value(miner_id, "")
        workload_value = query_value(workload_type, "")
        task_value = query_value(task_id, "")
        session_value = query_value(session_id, "")
        try:
            return {
                "results": store.result_ledger(
                    limit=limit_value,
                    status=status_value,
                    miner_id=miner_value,
                    workload_type=workload_value,
                    task_id=task_value,
                    session_id=session_value,
                ),
                "limit": min(500, max(0, int(limit_value))),
                "status": status_value,
                "miner_id": miner_value,
                "workload_type": workload_value,
                "task_id": task_value,
                "session_id": session_value,
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/admin/accounting")
    def admin_accounting(
        limit: int = Query(default=50, ge=0, le=500),
        status: str = Query(default="any"),
        miner_id: str = Query(default=""),
        workload_type: str = Query(default=""),
        session_id: str = Query(default=""),
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token, roles={OPERATOR_ROLE_ACCOUNTING})

        def query_value(value, default):
            if hasattr(value, "default"):
                value = value.default
            return default if value is None else value

        limit_value = query_value(limit, 50)
        status_value = query_value(status, "any")
        miner_value = query_value(miner_id, "")
        workload_value = query_value(workload_type, "")
        session_value = query_value(session_id, "")
        try:
            accounting = store.miner_accounting_summary(
                limit=limit_value,
                status=status_value,
                miner_id=miner_value,
                workload_type=workload_value,
                session_id=session_value,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return apply_accounting_policy_summary(accounting, configured_miner_registry)

    @app.get("/admin/settlement")
    def admin_settlement(
        limit: int = Query(default=50, ge=0, le=500),
        miner_id: str = Query(default=""),
        workload_type: str = Query(default=""),
        session_id: str = Query(default=""),
        unit_price_microcredits: int = Query(default=0, ge=0),
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token, roles={OPERATOR_ROLE_ACCOUNTING})

        def query_value(value, default):
            if hasattr(value, "default"):
                value = value.default
            return default if value is None else value

        limit_value = query_value(limit, 50)
        miner_value = query_value(miner_id, "")
        workload_value = query_value(workload_type, "")
        session_value = query_value(session_id, "")
        price_value = query_value(unit_price_microcredits, 0)
        try:
            settlement = store.miner_settlement_draft(
                limit=limit_value,
                miner_id=miner_value,
                workload_type=workload_value,
                session_id=session_value,
                unit_price_microcredits=price_value,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return apply_settlement_policy_summary(settlement, configured_miner_registry)

    @app.get("/admin/session-stream")
    def admin_session_stream(
        limit: int = Query(default=50, ge=0, le=500),
        session_id: str = Query(default=""),
        workload_type: str = Query(default=WORKLOAD_REAL_LLM_SHARDED_INFER),
        max_new_tokens: int = Query(default=0, ge=0, le=32),
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token, roles={OPERATOR_ROLE_AUDITOR})

        def query_value(value, default):
            if hasattr(value, "default"):
                value = value.default
            return default if value is None else value

        limit_value = query_value(limit, 50)
        session_value = str(query_value(session_id, "") or "")
        workload_value = str(query_value(workload_type, WORKLOAD_REAL_LLM_SHARDED_INFER) or "")
        max_tokens_value = int(query_value(max_new_tokens, 0) or 0)
        events = store.session_stream_events(
            session_id=session_value,
            max_new_tokens=max_tokens_value or None,
            limit=limit_value,
            workload_type=workload_value,
        )
        counts = [int(event.get("generated_token_count") or 0) for event in events]
        complete = bool(
            max_tokens_value > 0
            and all(count in counts for count in range(1, max_tokens_value + 1))
        )
        return {
            "schema": "admin_session_stream_v1",
            "ok": bool(session_value),
            "session_id": session_value,
            "workload_type": workload_value,
            "max_new_tokens": max_tokens_value or None,
            "event_count": len(events),
            "events": events,
            "progress_counts": counts,
            "stream_progress_complete": complete,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "diagnosis_codes": (
                ["session_stream_progress_complete"]
                if complete
                else (["session_stream_progress_ready"] if events else ["session_stream_pending"])
            ),
        }

    @app.post("/admin/inference-sessions")
    def admin_inference_sessions(
        request: InferenceSessionRequest,
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token)
        requested_workload = str(request.workload_type or WORKLOAD_MODEL_BUNDLE_INFER).strip()
        aliases = {
            "model-bundle": WORKLOAD_MODEL_BUNDLE_INFER,
            "external-llm": WORKLOAD_EXTERNAL_LLM_INFER,
            "sharded-model-bundle": WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
            "sharded": WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
            "micro-llm-sharded": WORKLOAD_MICRO_LLM_SHARDED_INFER,
            "micro-llm-shard": WORKLOAD_MICRO_LLM_SHARDED_INFER,
            "real-llm-sharded": WORKLOAD_REAL_LLM_SHARDED_INFER,
            "real-llm-shard": WORKLOAD_REAL_LLM_SHARDED_INFER,
        }
        workload_type = aliases.get(requested_workload, requested_workload)
        enforce_inference_session_rate_limit(
            x_crowdtensor_admin_token,
            workload_type=workload_type,
        )
        created_by_subject = admin_subject(x_crowdtensor_admin_token)
        try:
            if workload_type == WORKLOAD_MODEL_BUNDLE_INFER:
                session = store.create_readonly_inference_task(
                    request_count=request.request_count,
                    scenario_id=request.scenario_id,
                    required_runtime=request.runtime,
                    required_backend=request.backend,
                    required_protocol_version=DEFAULT_PROTOCOL_VERSION,
                    created_by_subject=created_by_subject,
                )
            elif workload_type == WORKLOAD_EXTERNAL_LLM_INFER:
                session = store.create_readonly_external_llm_task(
                    request_count=request.request_count,
                    required_runtime=request.runtime,
                    required_backend=request.backend,
                    required_protocol_version=DEFAULT_PROTOCOL_VERSION,
                    created_by_subject=created_by_subject,
                )
            elif workload_type == WORKLOAD_SHARDED_MODEL_BUNDLE_INFER:
                session = store.create_sharded_inference_session(
                    request_count=request.request_count,
                    scenario_id=request.scenario_id,
                    required_runtime=request.runtime,
                    required_backend=request.backend,
                    required_protocol_version=DEFAULT_PROTOCOL_VERSION,
                    created_by_subject=created_by_subject,
                )
            elif workload_type == WORKLOAD_MICRO_LLM_SHARDED_INFER:
                session = store.create_micro_llm_sharded_inference_session(
                    request_count=request.request_count,
                    decode_steps=request.decode_steps,
                    prompt_texts=(
                        request.prompt_texts
                        if request.prompt_texts is not None
                        else ([request.prompt] if request.prompt else None)
                    ),
                    required_runtime=request.runtime,
                    required_backend=request.backend,
                    required_protocol_version=DEFAULT_PROTOCOL_VERSION,
                    created_by_subject=created_by_subject,
                )
            elif workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
                llm_backend = "hf_transformers_cuda" if request.backend == "cuda" else "hf_transformers_cpu"
                session = store.create_real_llm_sharded_inference_session(
                    request_count=request.request_count,
                    max_new_tokens=request.max_new_tokens,
                    prompt_texts=(
                        request.prompt_texts
                        if request.prompt_texts is not None
                        else ([request.prompt] if request.prompt else None)
                    ),
                    required_runtime=request.runtime,
                    required_backend=request.backend,
                    model_id=request.hf_model_id,
                    llm_backend=llm_backend,
                    partition_mode=request.partition_mode,
                    required_protocol_version=DEFAULT_PROTOCOL_VERSION,
                    created_by_subject=created_by_subject,
                )
            else:
                raise ValueError(
                    f"unsupported inference session workload_type: {requested_workload}"
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            detail = str(exc)
            if (
                workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER
                and "requires optional Hugging Face dependencies" in detail
            ):
                raise HTTPException(status_code=503, detail=detail) from exc
            raise
        return {
            **session,
            "workload_type": workload_type,
            "created_by_subject": created_by_subject,
            "task_id": session.get("task_id") or session.get("stage_1_task_id") or session.get("stage_0_task_id"),
            "result_query": (
                f"/admin/results?task_id={session.get('task_id') or session.get('stage_1_task_id') or session.get('stage_0_task_id')}"
                f"&workload_type={workload_type}"
            ),
            "claim_requirements": session["task_requirements"],
        }

    @app.post("/admin/trust-overrides")
    def admin_trust_overrides(
        request: TrustOverrideRequest,
        x_crowdtensor_admin_token: str | None = Header(default=None),
    ) -> dict:
        require_admin(x_crowdtensor_admin_token)
        try:
            return store.set_trust_override(
                request.miner_id,
                request.workload_type,
                request.mode,
                reason=request.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/tasks/claim")
    def claim_task(
        request: ClaimRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict:
        require_claim_miner(request.miner_id, x_crowdtensor_miner_token)
        scoped_capabilities, policy_block_reason = enforce_miner_join_policy(
            miner_id=request.miner_id,
            capabilities=request.capabilities,
            registry=configured_miner_registry,
        )
        if policy_block_reason:
            store.record_claim_blocked(
                request.miner_id,
                capabilities=request.capabilities,
                reason=policy_block_reason,
                blocked_workloads=[WORKLOAD_REAL_LLM_SHARDED_INFER],
            )
            raise HTTPException(status_code=503, detail=policy_block_reason)
        policy = miner_join_policy_for(request.miner_id, configured_miner_registry)
        quota_task_limit = int(policy.get("quota_task_limit") or 0) if policy else 0
        if quota_task_limit > 0:
            usage = store.miner_claim_usage(request.miner_id)
            if int(usage.get("claim_count") or 0) >= quota_task_limit:
                reason = "join_policy_quota_exhausted"
                store.record_claim_blocked(
                    request.miner_id,
                    capabilities=scoped_capabilities,
                    reason=reason,
                    blocked_workloads=[str(policy.get("read_only_workload") or WORKLOAD_REAL_LLM_SHARDED_INFER)],
                )
                raise HTTPException(status_code=503, detail=reason)
        claim_rate_limit = int(policy.get("claim_rate_limit") or 0) if policy else 0
        claim_rate_window_seconds = float(policy.get("claim_rate_window_seconds") or 0.0) if policy else 0.0
        if claim_rate_limit > 0 and claim_rate_window_seconds > 0:
            usage = store.miner_claim_rate_usage(
                request.miner_id,
                window_seconds=claim_rate_window_seconds,
            )
            if int(usage.get("claim_count") or 0) >= claim_rate_limit:
                reason = "join_policy_rate_limited"
                store.record_claim_blocked(
                    request.miner_id,
                    capabilities=scoped_capabilities,
                    reason=reason,
                    blocked_workloads=[str(policy.get("read_only_workload") or WORKLOAD_REAL_LLM_SHARDED_INFER)],
                )
                raise HTTPException(status_code=429, detail=reason)
        store.reap_expired()
        try:
            return store.claim_task(request.miner_id, capabilities=scoped_capabilities)
        except NoTaskAvailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/tasks/{task_id}/heartbeat")
    def heartbeat(
        task_id: str,
        request: LeaseRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict:
        require_miner(x_crowdtensor_miner_token)
        try:
            return store.heartbeat(
                task_id,
                lease_token=request.lease_token,
                attempt=request.attempt,
                runtime_status=request.runtime_status,
            )
        except LeaseConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/tasks/{task_id}/result")
    def result(
        task_id: str,
        request: ResultRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict:
        require_miner(x_crowdtensor_miner_token)
        try:
            return store.complete_task(
                task_id,
                lease_token=request.lease_token,
                attempt=request.attempt,
                idempotency_key=request.idempotency_key,
                local_delta=request.local_delta,
                pseudo_gradient=request.pseudo_gradient,
                compressed_delta=request.compressed_delta,
                probe_result=request.probe_result,
                adapter_delta=request.adapter_delta,
                bundle_delta=request.bundle_delta,
                inference_result=request.inference_result,
                inference_results=request.inference_results,
                external_llm_result=request.external_llm_result,
                external_llm_results=request.external_llm_results,
                sharded_inference_result=request.sharded_inference_result,
                metrics=request.metrics,
            )
        except ResultRejected as exc:
            raise HTTPException(status_code=422, detail=exc.validation) from exc
        except LeaseConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def parse_task_lane(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("task lane must use runtime:backend:count[:workload_type]")
    runtime, backend, count_text = parts[:3]
    workload_type = parts[3] if len(parts) == 4 else DEFAULT_WORKLOAD_TYPE
    try:
        count = int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("task lane count must be an integer") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("task lane count must be non-negative")
    return {
        "runtime": runtime or "any",
        "backend": backend or "any",
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "workload_type": workload_type or DEFAULT_WORKLOAD_TYPE,
        "count": count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CrowdTensorD Phase 1 Coordinator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--inner-steps", type=int, default=500)
    parser.add_argument("--backlog", type=int, default=1)
    parser.add_argument(
        "--task-lane",
        action="append",
        type=parse_task_lane,
        dest="task_lanes",
        default=None,
        help="runtime:backend:count[:workload_type] lane to keep queued; repeat for multiple lanes",
    )
    parser.add_argument("--reaper-interval", type=float, default=1.0)
    parser.add_argument(
        "--replay-audit",
        action="store_true",
        help="enable deterministic replay audit for supported training workloads",
    )
    parser.add_argument(
        "--outer-optimizer",
        choices=sorted(SUPPORTED_OUTER_OPTIMIZERS),
        default=OPTIMIZER_DILOCO_MOMENTUM,
        help="outer optimizer for new dense DiLoCo state",
    )
    parser.add_argument(
        "--delta-format",
        choices=sorted(SUPPORTED_DELTA_FORMATS),
        default="dense_float",
        help="claim-time delta transport format for diloco_train tasks",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help="admin token for control-plane endpoints; falls back to CROWDTENSOR_ADMIN_TOKEN",
    )
    parser.add_argument(
        "--operator-token-registry",
        default=None,
        help="JSON per-operator token registry path; falls back to CROWDTENSOR_OPERATOR_TOKEN_REGISTRY",
    )
    parser.add_argument(
        "--inference-session-rate-limit",
        type=int,
        default=0,
        help="max /admin/inference-sessions creates per operator/admin subject per window; 0 disables",
    )
    parser.add_argument(
        "--inference-session-rate-window-seconds",
        type=float,
        default=0.0,
        help="window seconds for --inference-session-rate-limit; set both to enable request abuse protection",
    )
    parser.add_argument(
        "--miner-token",
        default=None,
        help="shared token for Miner task endpoints; falls back to CROWDTENSOR_MINER_TOKEN",
    )
    parser.add_argument(
        "--miner-token-registry",
        default=None,
        help="JSON per-miner token registry path; falls back to CROWDTENSOR_MINER_TOKEN_REGISTRY",
    )
    parser.add_argument(
        "--observer-token",
        default=None,
        help="shared token for /state and /metrics; falls back to CROWDTENSOR_OBSERVER_TOKEN",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        default=None,
        help="allowed browser origin for local browser Miner clients; repeat for multiple origins",
    )
    parser.add_argument(
        "--micro-llm-artifact",
        default=os.environ.get("CROWDTENSOR_MICRO_LLM_ARTIFACT", ""),
        help="path to a file-backed micro_llm_artifact_v1 manifest or directory",
    )
    parser.add_argument(
        "--real-llm-model-id",
        default=os.environ.get("CROWDTENSOR_HF_MODEL_ID", "sshleifer/tiny-gpt2"),
        help="Hugging Face causal LM id for real_llm_sharded_infer sessions",
    )
    parser.add_argument(
        "--real-llm-backend",
        choices=["hf_transformers_cpu", "hf_transformers_cuda", "cpu", "cuda", "auto"],
        default=os.environ.get("CROWDTENSOR_REAL_LLM_BACKEND", "hf_transformers_cpu"),
        help=(
            "backend for real_llm_sharded_infer sessions; cuda schedules CUDA-capable Miner tasks "
            "and defers torch CUDA runtime checks to the Miner"
        ),
    )
    parser.add_argument(
        "--real-llm-partition-mode",
        choices=["full", "stage-local", "stage_local"],
        default=os.environ.get("CROWDTENSOR_REAL_LLM_PARTITION_MODE", "full"),
        help="real_llm_sharded_infer partition mode; stage-local moves only stage-owned modules to the target device",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=os.environ.get("CROWDTENSOR_HF_CACHE_DIR", ""),
        help="optional Hugging Face cache directory for real_llm_sharded_infer",
    )
    args = parser.parse_args()
    if args.replay_audit and args.delta_format == DELTA_FORMAT_SIGN_COMPRESSED_EF:
        parser.error("--delta-format sign_compressed_ef cannot be used with --replay-audit")
    return args


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit("uvicorn is not installed. Run: pip install -r requirements.txt") from exc

    app = create_app(
        state_dir=args.state_dir,
        lease_seconds=args.lease_seconds,
        inner_steps=args.inner_steps,
        backlog=args.backlog,
        task_lanes=args.task_lanes,
        reaper_interval=args.reaper_interval,
        cors_origins=args.cors_origins or ["http://127.0.0.1:8765", "http://localhost:8765"],
        replay_audit=args.replay_audit,
        outer_optimizer=args.outer_optimizer,
        delta_format=args.delta_format,
        admin_token=args.admin_token,
        operator_token_registry=args.operator_token_registry,
        inference_session_rate_limit=args.inference_session_rate_limit,
        inference_session_rate_window_seconds=args.inference_session_rate_window_seconds,
        miner_token=args.miner_token,
        miner_token_registry=args.miner_token_registry,
        observer_token=args.observer_token,
        micro_llm_artifact=args.micro_llm_artifact,
        real_llm_model_id=args.real_llm_model_id,
        real_llm_backend=args.real_llm_backend,
        real_llm_partition_mode=args.real_llm_partition_mode,
        hf_cache_dir=args.hf_cache_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
