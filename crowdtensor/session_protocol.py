"""Public session/task protocol helpers for CrowdTensor product-facing flows.

This module deliberately contains no networking or model execution code.  It is
the narrow, safe summary layer between user-facing commands, Coordinator task
creation, P2P-lite route discovery, and public evidence artifacts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .protocol import (
    DEFAULT_PROTOCOL_VERSION,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
    WORKLOAD_REAL_LLM_SHARDED_INFER,
)


SESSION_PROTOCOL_SCHEMA = "session_protocol_v1"
SESSION_RESULT_SCHEMA = "session_result_summary_v1"
MAX_PROMPT_CHARS = 256
MAX_NEW_TOKENS = 32

WORKLOAD_ALIASES = {
    "real-llm-sharded": WORKLOAD_REAL_LLM_SHARDED_INFER,
    "real-llm-shard": WORKLOAD_REAL_LLM_SHARDED_INFER,
    "gpu-generation": WORKLOAD_REAL_LLM_SHARDED_INFER,
    "cpu-real-llm": WORKLOAD_REAL_LLM_SHARDED_INFER,
}

RAW_PUBLIC_KEYS = {
    "prompt",
    "prompt_text",
    "prompt_texts",
    "output_text",
    "generated_text",
    "generated_token_ids",
    "next_token_text",
    "next_token_id",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "hidden_state",
    "input_ids",
    "logits",
    "inference_result",
    "inference_results",
    "sharded_inference_result",
    "external_llm_result",
    "external_llm_results",
}
SECRET_VALUE_FRAGMENTS = (
    "CROWDTENSOR_ADMIN_TOKEN",
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "Bearer ",
)


def stable_hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def stable_hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_workload(workload: str | None) -> str:
    normalized = str(workload or WORKLOAD_REAL_LLM_SHARDED_INFER).strip()
    return WORKLOAD_ALIASES.get(normalized, normalized)


def normalize_backend(backend: str | None) -> str:
    normalized = str(backend or "cpu").strip().lower()
    if normalized in {"gpu", "cuda", "hf_transformers_cuda"}:
        return "cuda"
    if normalized in {"cpu", "hf_transformers_cpu", ""}:
        return "cpu"
    raise ValueError(f"unsupported session backend: {backend}")


def normalize_stage_mode(stage_mode: str | None) -> str:
    normalized = str(stage_mode or "split").strip().lower()
    if normalized not in {"stage0", "stage1", "both", "split"}:
        raise ValueError(f"unsupported stage_mode: {stage_mode}")
    return normalized


def required_stage_capabilities(*, backend: str, stage_mode: str = "split") -> list[str]:
    resolved_backend = normalize_backend(backend)
    resolved_stage = normalize_stage_mode(stage_mode)
    if resolved_backend == "cuda":
        stage0 = REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY
        stage1 = REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY
    else:
        stage0 = REAL_LLM_SHARDED_STAGE0_CAPABILITY
        stage1 = REAL_LLM_SHARDED_STAGE1_CAPABILITY
    if resolved_stage == "stage0":
        return [stage0]
    if resolved_stage == "stage1":
        return [stage1]
    return [stage0, stage1]


def build_session_request(
    *,
    prompt_text: str,
    workload_type: str = WORKLOAD_REAL_LLM_SHARDED_INFER,
    backend: str = "cpu",
    stage_mode: str = "split",
    max_new_tokens: int = 16,
    scenario_id: str = "public-swarm-product-rc",
    route_source: str = "coordinator-url",
) -> dict[str, Any]:
    prompt = str(prompt_text or "")
    if not prompt:
        raise ValueError("prompt_text is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt_text must be at most {MAX_PROMPT_CHARS} characters")
    token_limit = int(max_new_tokens)
    if token_limit < 1 or token_limit > MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    workload = normalize_workload(workload_type)
    resolved_backend = normalize_backend(backend)
    resolved_stage = normalize_stage_mode(stage_mode)
    if workload != WORKLOAD_REAL_LLM_SHARDED_INFER:
        raise ValueError(f"unsupported public session workload_type: {workload_type}")
    capabilities = required_stage_capabilities(backend=resolved_backend, stage_mode=resolved_stage)
    request = {
        "schema": SESSION_PROTOCOL_SCHEMA,
        "workload_type": workload,
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "backend": resolved_backend,
        "stage_mode": resolved_stage,
        "scenario_id": str(scenario_id or "public-swarm-product-rc"),
        "max_new_tokens": token_limit,
        "request_count": 1,
        "prompt_hash": stable_hash_text(prompt),
        "prompt_chars": len(prompt),
        "route_requirements": {
            "runtime": "python-cli",
            "backend": resolved_backend,
            "stage_count": 2,
            "required_capabilities": capabilities,
            "route_source": str(route_source or "coordinator-url"),
        },
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "read_only_workload": True,
            "not_production": True,
            "not_full_p2p": True,
        },
    }
    assert_public_safe(request)
    return request


def coordinator_payload_for_request(session_request: dict[str, Any], *, prompt_text: str) -> dict[str, Any]:
    """Build the private Coordinator payload corresponding to a public request."""
    return {
        "workload_type": session_request.get("workload_type") or WORKLOAD_REAL_LLM_SHARDED_INFER,
        "request_count": 1,
        "max_new_tokens": int(session_request.get("max_new_tokens") or 1),
        "runtime": "python-cli",
        "backend": session_request.get("backend") or "cpu",
        "prompt": str(prompt_text or ""),
        "scenario_id": session_request.get("scenario_id") or "public-swarm-product-rc",
        "partition_mode": "stage-local",
    }


def peer_has_capability(peer: dict[str, Any], capability: str) -> bool:
    caps = peer.get("capabilities") if isinstance(peer.get("capabilities"), dict) else {}
    collections = [
        caps.get("real_llm_sharded_stage_capabilities"),
        caps.get("capabilities"),
        caps.get("supported_capabilities"),
    ]
    for values in collections:
        if isinstance(values, list) and capability in values:
            return True
    return False


def build_route_decision(
    session_request: dict[str, Any],
    *,
    coordinator_url: str = "",
    peer_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    requirements = session_request.get("route_requirements") if isinstance(session_request.get("route_requirements"), dict) else {}
    required = [str(item) for item in requirements.get("required_capabilities") or []]
    peers = [peer for peer in (peer_catalog or []) if isinstance(peer, dict)]
    matched: dict[str, str] = {}
    for capability in required:
        peer_id = ""
        for peer in peers:
            if peer_has_capability(peer, capability):
                peer_id = str(peer.get("peer_id") or "")
                break
        if peer_id:
            matched[capability] = peer_id
    coordinator_candidates = [
        peer for peer in peers
        if str(peer.get("role") or "") == "coordinator"
        and ((peer.get("urls") or {}).get("coordinator") or peer.get("coordinator_url"))
    ]
    resolved_coordinator = str(coordinator_url or "")
    if not resolved_coordinator and coordinator_candidates:
        first = coordinator_candidates[0]
        urls = first.get("urls") if isinstance(first.get("urls"), dict) else {}
        resolved_coordinator = str(urls.get("coordinator") or first.get("coordinator_url") or "")
    missing = [capability for capability in required if capability not in matched]
    usable = bool(resolved_coordinator) and not missing
    return {
        "schema": "session_route_decision_v1",
        "usable_now": usable,
        "coordinator_url_present": bool(resolved_coordinator),
        "coordinator_url": resolved_coordinator,
        "route_source": requirements.get("route_source") or ("peer-bootstrap" if coordinator_candidates else "coordinator-url"),
        "backend": session_request.get("backend"),
        "workload_type": session_request.get("workload_type"),
        "required_capabilities": required,
        "matched_capabilities": matched,
        "missing_capabilities": missing,
        "diagnosis_codes": (
            ["session_route_ready"]
            if usable
            else ["coordinator_route_missing" if not resolved_coordinator else "stage_capability_missing"]
        ),
    }


def find_generation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1
    queue: list[Any] = [payload]
    seen: set[int] = set()
    while queue:
        item = queue.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        generation = item.get("generation") if isinstance(item.get("generation"), dict) else {}
        validation = item.get("validation") if isinstance(item.get("validation"), dict) else {}
        for candidate in (generation, validation, item):
            if not isinstance(candidate, dict):
                continue
            score = 0
            if candidate.get("generated_text_hash"):
                score += 8
            if candidate.get("generated_token_count"):
                score += 4
            if candidate.get("decoded_tokens_match"):
                score += 2
            if score > best_score:
                best = candidate
                best_score = score
        for value in item.values():
            if isinstance(value, dict):
                queue.append(value)
            elif isinstance(value, list):
                queue.extend(entry for entry in value if isinstance(entry, dict))
    return best


def safe_generation_summary(payload: dict[str, Any], *, max_new_tokens: int | None = None) -> dict[str, Any]:
    generation = find_generation_summary(payload)
    count = int(generation.get("generated_token_count") or 0)
    limit = int(max_new_tokens or generation.get("max_new_tokens") or 0)
    ready = bool(count and (limit <= 0 or count >= limit))
    summary = {
        "schema": SESSION_RESULT_SCHEMA,
        "generated_token_count": count,
        "max_new_tokens": limit or None,
        "generated_text_hash": generation.get("generated_text_hash"),
        "decoded_tokens_match": generation.get("decoded_tokens_match"),
        "multi_token_generation_ready": ready,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    assert_public_safe(summary)
    return summary


def assert_public_safe(value: Any) -> None:
    leaks = public_leak_paths(value)
    if leaks:
        raise ValueError("public payload contains raw or secret fields: " + ", ".join(leaks[:5]))


def public_leak_paths(value: Any, *, path: str = "$") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in RAW_PUBLIC_KEYS:
                leaks.append(child_path)
            leaks.extend(public_leak_paths(item, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaks.extend(public_leak_paths(item, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        for fragment in SECRET_VALUE_FRAGMENTS:
            if fragment in value:
                leaks.append(path)
    return leaks
